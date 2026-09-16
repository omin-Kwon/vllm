# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA recurrent (decode) kernel.

The decode path hands the kernel column slices of the merged ``q|k|v`` conv
output and of the fused ``qkvbfg_a`` projection (beta), so q/k/v/beta are
token-strided rather than contiguous. The kernel must read them in place,
match a pure-PyTorch recurrence, and reject layouts it cannot address.
"""

import pytest
import torch

from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only Triton kernel"
)

H, D = 16, 128
LOWER_BOUND = -5.0


@pytest.mark.parametrize("mode", ["replay", "sketch"])
def test_window_rejects_simultaneous_qmamba_quantization(monkeypatch, mode):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.ops.glm_window.config import create_cache

    monkeypatch.setenv("NS_GDN_QBITS", "8")
    monkeypatch.delenv("NS_GDN_QGRAN", raising=False)
    monkeypatch.delenv("NS_GDN_QSR", raising=False)
    config = SimpleNamespace(additional_config={"kda_window": {"mode": mode}})
    with pytest.raises(ValueError, match="Q-Mamba DSQ cannot be combined"):
        create_cache(config, 0, 64, 128, -5.0)


def test_window_checkpoint_load_ignores_model_cuda_default_device(tmp_path):
    """vLLM constructs layers inside a CUDA default-device context."""
    from vllm.model_executor.layers.mamba.ops.glm_window.sketch import load_checkpoint

    pack = dict(
        meta=dict(
            schema="glm_kda_fullgram_allocation_v1",
            K=128,
            V=128,
            W=16,
            group_size=1,
            exact_Z=128,
            query_anchors=False,
            embedded_state=False,
            allocation_pivot_dependent=False,
            allocation_objective="joint_dot_sq_sum_fullgram",
        ),
        frames={0: torch.eye(128).repeat(64, 1, 1)},
        ranks={0: torch.full((64,), 4, dtype=torch.int32)},
    )
    path = tmp_path / "runtime.pt"
    torch.save(pack, path)
    with torch.device("cuda"):
        result = load_checkpoint(str(path))
    assert result["frames"][0].device.type == "cpu"


@pytest.mark.parametrize("graph", [False, True])
@torch.inference_mode()
def test_window_replay_survives_reorder_eviction_release_and_prefill(graph):
    """Physical pages may outlive slots; freed pages must never be flushed."""
    from vllm.model_executor.layers.mamba.ops.glm_window.cache import ReplayCache

    torch.manual_seed(217)
    heads, capacity, batch = 3, 4, 3
    state = torch.empty_strided(
        (9, heads, 128, 128),
        (heads * 16384 + 256, 16384, 128, 1),
        device="cuda",
        dtype=torch.float32,
    )
    state.normal_(std=0.1)
    expected_state = state.clone()
    cache = ReplayCache(heads, capacity, state.device)
    ids = torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda")
    data = [
        torch.randn(batch, heads, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    data[3].sub_(4)
    data.append(torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16))
    a = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads, 128, device="cuda")

    def execute():
        return cache.step(state, ids, *data, a, bias)

    execute()  # Compile before capture; no live request state is retained.
    cache.reset()
    state.copy_(expected_state)
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = execute()
        cache.reset()
        state.copy_(expected_state)

    for t in range(257):
        # More physical pages than slots exercises eviction of absent owners.
        active = [1 + (t // 19) % 7, 1 + (t // 19 + 3) % 7, 0]
        if t % 2:
            active[:2] = active[:2][::-1]
        ids.copy_(torch.tensor(active, device="cuda", dtype=torch.int32))
        if t % 29 == 0:
            victim = ids[:1]
            cache.release_finished(victim)
            state[active[0]].zero_()
            expected_state[active[0]].zero_()
        if t % 11 == 0:
            cache.before_prefill(
                state, ids[:1], torch.ones(1, device="cuda", dtype=torch.bool)
            )
            torch.testing.assert_close(
                state[active[0]], expected_state[active[0]], rtol=2e-5, atol=2e-6
            )
        data[0].normal_()
        if graph:
            g.replay()
        else:
            out = execute()
        native, _ = fused_recurrent_kda(
            q=data[0].unsqueeze(0),
            k=data[1].unsqueeze(0),
            v=data[2].unsqueeze(0),
            g=data[3].unsqueeze(0),
            beta=data[4].unsqueeze(0),
            a_log=a,
            g_bias=bias,
            initial_state=expected_state,
            cu_seqlens=torch.arange(batch + 1, device="cuda", dtype=torch.int32),
            ssm_state_indices=ids,
            use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True,
            compute_gate=True,
            lower_bound=-5.0,
        )
        torch.testing.assert_close(out[:2], native[0, :2], rtol=0.008, atol=1e-4)
        assert torch.count_nonzero(out[2]) == 0
    all_ids = torch.arange(1, 9, device="cuda", dtype=torch.int32)
    cache.before_prefill(state, all_ids, torch.ones(8, device="cuda", dtype=torch.bool))
    torch.testing.assert_close(state, expected_state, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("mode", ["replay", "p4", "p6"])
@torch.inference_mode()
def test_mixed_window_decode_never_enters_flashkda_prefill(mode):
    """Repeated prefill arrivals must not reset an existing decode window."""
    from types import SimpleNamespace

    import vllm._flashkda_C  # noqa: F401

    from vllm.model_executor.layers.mamba.ops.glm_window.cache import ReplayCache
    from vllm.model_executor.layers.mamba.ops.glm_window.routing import window_attention

    torch.manual_seed(49)
    heads, tokens = 4, 66  # Two decode rows and one 64-token prefill.
    state = torch.randn(5, heads, 128, 128, device="cuda") * 0.1
    reference = state.clone()
    if mode == "replay":
        cache = ReplayCache(heads, 4, state.device)
        solo = ReplayCache(heads, 4, state.device)
    else:
        from vllm.model_executor.layers.mamba.ops.glm_window.sketch import SketchCache

        frame = torch.eye(128, device="cuda").repeat(heads, 1, 1)
        ranks = torch.tensor([1, 3, 7, 28])
        cache = SketchCache(frame, ranks, 4, int(mode[1:]))
        solo = SketchCache(frame, ranks, 4, int(mode[1:]))
    ids = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    initial = torch.tensor([True, True, False], device="cuda")
    metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        num_spec_decodes=0,
        num_prefills=1,
        num_prefill_tokens=64,
        non_spec_state_indices_tensor=ids,
        prefill_state_indices=ids[2:],
        prefill_has_initial_state=initial[2:],
        prefill_query_start_loc=torch.tensor([0, 64], device="cuda", dtype=torch.int32),
    )
    a = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads * 128, device="cuda")
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(64, heads, 1),
        device="cuda",
        dtype=torch.uint8,
    )
    last = torch.empty(1, heads, 128, 128, device="cuda")
    calls = []

    def prefill(q, k, v, g, beta, initial_state, cu_seqlens, out):
        assert q.shape[1] == 64
        calls.append(q.shape[1])
        torch.ops._flashkda_C.fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta,
            128**-0.5,
            out,
            workspace,
            a,
            bias.view(heads, 128),
            -5.0,
            initial_state.contiguous(),
            last,
            cu_seqlens.contiguous(),
            None,
            None,
        )
        return out, last

    for t in range(35):
        data = [
            torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        data[3].sub_(4)
        data.append(torch.randn(1, tokens, heads, device="cuda", dtype=torch.bfloat16))
        out = torch.empty_like(data[2])
        window_attention(cache, state, metadata, *data, a, bias, out, prefill)
        expected = solo.step(reference, ids[:2], *[x[0, :2] for x in data], a, bias)
        torch.testing.assert_close(out[0, :2], expected, rtol=0, atol=0)
        assert cache.pool.pos[cache.owners > 0].tolist() == [(t + 1) % 16] * 2
        # Prefill output/state must also match a standalone native invocation.
        pf_out = torch.empty_like(out[:, 2:])
        zero = torch.zeros_like(last)
        prefill(
            data[0][:, 2:],
            data[1][:, 2:],
            data[2][:, 2:],
            data[3][:, 2:],
            data[4][:, 2:],
            zero,
            metadata.prefill_query_start_loc,
            pf_out,
        )
        torch.testing.assert_close(out[:, 2:], pf_out, rtol=0, atol=0)
        torch.testing.assert_close(state[3], last[0], rtol=0, atol=0)
    assert len(calls) == 70


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference for one sequence: ``[T, H, D]`` inputs, ``[H, D, D]``
    (v-major) state; mirrors the kernel's in-kernel gate, beta sigmoid and
    q/k l2norm.
    """
    q, k, v, raw_g, raw_beta = (x.float() for x in (q, k, v, raw_g, raw_beta))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = LOWER_BOUND * torch.sigmoid(a_log.exp()[:, None] * (raw_g + g_bias))
    beta = torch.sigmoid(raw_beta)
    s = state.clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        s = s * gate[t].exp()[:, None, :]
        u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, k[t]))
        s = s + u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, q[t])
    return out, s


def make_inputs(num_seqs: int, query_len: int, device: torch.device):
    """Token-strided q/k/v/beta as the decode path produces them: column
    slices of a merged ``[T, q|k|v]`` conv output and of the fused
    ``[T, qkv|beta|f_a|g_a]`` projection.
    """
    T, proj = num_seqs * query_len, H * D
    qkv = torch.randn(T, 3 * proj, dtype=torch.bfloat16, device=device)
    projected = torch.randn(
        T, 3 * proj + H + 2 * D, dtype=torch.bfloat16, device=device
    )
    q, k, v = (x.view(1, T, H, D) for x in qkv.split(proj, dim=-1))
    beta = projected[:, 3 * proj : 3 * proj + H].unsqueeze(0)
    # (A size-1 token dim gets an arbitrary stride from `view`.)
    assert T == 1 or (q.stride(1) == 3 * proj and beta.stride(1) == projected.stride(0))
    inputs = dict(
        q=q,
        k=k,
        v=v,
        g=torch.randn(1, T, H, D, dtype=torch.bfloat16, device=device),
        beta=beta,
        a_log=0.5 * torch.randn(H, dtype=torch.float32, device=device),
        g_bias=0.1 * torch.randn(H * D, dtype=torch.float32, device=device),
        cu_seqlens=torch.arange(0, T + 1, query_len, dtype=torch.int32, device=device),
    )
    # Slot 0 is NULL_BLOCK_ID; sequences own random distinct slots (one per
    # token in the spec-decode layout).
    slots = torch.randperm(T, device=device).to(torch.int32) + 1
    if query_len == 1:
        inputs["ssm_state_indices"] = slots
    else:
        inputs["ssm_state_indices"] = slots.view(num_seqs, query_len)
        inputs["num_accepted_tokens"] = torch.randint(
            1, query_len + 1, (num_seqs,), dtype=torch.int32, device=device
        )
    state = torch.randn(T + 1, H, D, D, dtype=torch.float32, device=device)
    return inputs, state


@pytest.mark.parametrize("pivots", [4, 6])
@torch.inference_mode()
def test_window_sketch_matches_independent_fp64_pivot_oracle(pivots):
    from vllm.model_executor.layers.mamba.ops.glm_window.sketch import SketchCache

    from .window_reference import Oracle, coefficient

    torch.manual_seed(23)
    torch.set_num_threads(2)
    ranks = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 12, 28, 60, 128])
    heads = len(ranks)
    frame = torch.linalg.qr(torch.randn(heads, 128, 128, dtype=torch.float64)).Q.float()
    initial = torch.randn(4, heads, 128, 128) * 0.1
    initial[:, 1].zero_()
    dependent = initial[:, 7] @ frame[7]
    dependent[:, :, 1] = dependent[:, :, 0]
    initial[:, 7] = dependent @ frame[7].T
    state = initial.cuda()
    cache = SketchCache(frame.cuda(), ranks, capacity=3, pivots=pivots)
    ref = Oracle(initial, frame, ranks, pivots)
    a = torch.randn(heads) * 0.1
    bias = torch.randn(heads, 128) * 0.1
    ids = torch.tensor([2, 1, 0], device="cuda", dtype=torch.int32)
    for t in range(35):
        if t == 19:
            ids.copy_(torch.tensor([1, 2, 0], device="cuda", dtype=torch.int32))
        data = [
            torch.randn(3, heads, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        data[3].sub_(3)
        data.append(torch.randn(3, heads, device="cuda", dtype=torch.bfloat16))
        out = cache.step(state, ids, *data, a.cuda(), bias.cuda())
        expected = ref.step(
            ids.tolist(), data[0], data[1], data[2], data[3], data[4], a, bias
        )
        relative = (out.double().cpu() - expected).norm() / expected.norm()
        assert relative < 0.007, (t, relative)
        assert torch.count_nonzero(out[2]) == 0
        if t == 0:
            for slot, physical in enumerate(cache.owners.tolist()):
                if physical <= 0:
                    continue
                for h, rank in enumerate(ranks.tolist()):
                    if rank in (0, 128):
                        continue
                    expected_phi = coefficient(
                        initial[physical, h], frame[h], rank, pivots
                    )
                    actual = cache.pool.phi[slot, h, :, :rank].double().cpu()
                    error = (
                        actual - expected_phi
                    ).norm() / expected_phi.norm().clamp_min(1e-12)
                    assert error < 8e-4, (rank, error)
        if t in (15, 31):
            torch.testing.assert_close(
                state[1:3].double().cpu(), ref.state[1:3], rtol=1e-3, atol=3e-6
            )
    cache.before_prefill(state, ids[:2], torch.ones(2, device="cuda", dtype=torch.bool))
    torch.testing.assert_close(
        state[1:3].double().cpu(), ref.state[1:3], rtol=1e-3, atol=3e-6
    )


@pytest.mark.parametrize("pivots", [4, 6])
@torch.inference_mode()
def test_window_flush_ignores_corrupted_sketch_metadata(pivots):
    from vllm.model_executor.layers.mamba.ops.glm_window.sketch import SketchCache

    torch.manual_seed(137)
    h = 5
    state = torch.randn(3, h, 128, 128, device="cuda") * 0.1
    dense = state.clone()
    frame = torch.eye(128, device="cuda").repeat(h, 1, 1)
    cache = SketchCache(frame, torch.tensor([0, 1, 3, 7, 28]), 2, pivots)
    ids = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    a = torch.zeros(h, device="cuda")
    bias = torch.zeros(h, 128, device="cuda")
    cu = torch.arange(3, device="cuda", dtype=torch.int32)
    for t in range(32):
        data = [
            torch.randn(2, h, 128, device="cuda", dtype=torch.bfloat16)
            for _ in range(4)
        ]
        data[3].sub_(3)
        data.append(torch.randn(2, h, device="cuda", dtype=torch.bfloat16))
        if t in (7, 23):
            cache.pool.phi.fill_(float("nan"))
            cache.pool.latch.fill_(float("nan"))
        out = cache.step(state, ids, *data, a, bias)
        expected, _ = fused_recurrent_kda(
            q=data[0].unsqueeze(0),
            k=data[1].unsqueeze(0),
            v=data[2].unsqueeze(0),
            g=data[3].unsqueeze(0),
            beta=data[4].unsqueeze(0),
            a_log=a,
            g_bias=bias,
            initial_state=dense,
            ssm_state_indices=ids,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True,
            compute_gate=True,
            lower_bound=-5.0,
        )
        if t in (15, 31):
            torch.testing.assert_close(out, expected[0], rtol=0.008, atol=1e-4)
            torch.testing.assert_close(state, dense, rtol=1e-3, atol=3e-6)


def run_kernel(inputs: dict, state: torch.Tensor) -> torch.Tensor:
    out, _ = fused_recurrent_kda(
        **inputs,
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
        sigmoid_beta=True,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    return out


@pytest.mark.parametrize("mode", ["replay", "p4"])
@torch.inference_mode()
def test_window_4096_step_graph_lifecycle_keeps_native_state(mode):
    """64 slots, 96 physical pages, churn/pauses/padding and partial handoffs."""
    from vllm.model_executor.layers.mamba.ops.glm_window.cache import ReplayCache
    from vllm.model_executor.layers.mamba.ops.glm_window.sketch import SketchCache

    torch.manual_seed(300)
    h, batch = 2, 32
    state = torch.randn(97, h, 128, 128, device="cuda") * 0.1
    reference = state.clone()
    if mode == "replay":
        cache = ReplayCache(h, 64, state.device)
    else:
        cache = SketchCache(
            torch.eye(128, device="cuda").repeat(h, 1, 1), torch.tensor([7, 28]), 64, 4
        )
    ids = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    data = [
        torch.randn(batch, h, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    data[3].sub_(4)
    data.append(torch.randn(batch, h, device="cuda", dtype=torch.bfloat16))
    a, bias = torch.zeros(h, device="cuda"), torch.zeros(h, 128, device="cuda")
    cache.step(state, ids, *data, a, bias)
    cache.reset()
    state.copy_(reference)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = cache.step(state, ids, *data, a, bias)
    cache.reset()
    state.copy_(reference)
    initial = torch.ones(1, device="cuda", dtype=torch.bool)
    for step in range(4096):
        values = [(row + 7 * (step // 37)) % 96 + 1 for row in range(batch)]
        if step % 2:
            values.reverse()
        values[-1] = 0
        ids.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
        if step % 97 == 0:
            cache.release_finished(ids[:1])
            state[values[0]].zero_()
            reference[values[0]].zero_()
        if step % 53 == 0:
            cache.before_prefill(state, ids[1:2], initial)
            torch.testing.assert_close(
                state[values[1]], reference[values[1]], rtol=1e-3, atol=4e-6
            )
        data[0].normal_()
        graph.replay()
        native, _ = fused_recurrent_kda(
            q=data[0].unsqueeze(0),
            k=data[1].unsqueeze(0),
            v=data[2].unsqueeze(0),
            g=data[3].unsqueeze(0),
            beta=data[4].unsqueeze(0),
            a_log=a,
            g_bias=bias,
            initial_state=reference,
            ssm_state_indices=ids,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            sigmoid_beta=True,
            compute_gate=True,
            lower_bound=-5.0,
        )
        assert torch.isfinite(out).all()
        if mode == "replay":
            torch.testing.assert_close(out[:-1], native[0, :-1], rtol=0.008, atol=1e-4)
    all_ids = torch.arange(1, 97, device="cuda", dtype=torch.int32)
    cache.before_prefill(state, all_ids, torch.ones_like(all_ids, dtype=torch.bool))
    torch.testing.assert_close(state, reference, rtol=1e-3, atol=4e-6)


@pytest.mark.parametrize(
    ("num_seqs", "query_len"), [(1, 1), (7, 1), (3, 3)], ids=["1x1", "7x1", "3x3"]
)
@torch.inference_mode()
def test_fused_recurrent_kda_matches_reference(num_seqs: int, query_len: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, query_len, device)
    expected_state = state.clone()
    out = run_kernel(inputs, state)

    indices = inputs["ssm_state_indices"].view(num_seqs, query_len)
    accepted = inputs.get("num_accepted_tokens")
    expected = torch.empty_like(out[0], dtype=torch.float32)
    for n in range(num_seqs):
        first = indices[n, 0 if accepted is None else accepted[n] - 1]
        s = expected_state[first]
        for t in range(query_len):
            tok = slice(n * query_len + t, n * query_len + t + 1)
            expected[tok], s = naive_recurrent_kda(
                inputs["q"][0, tok],
                inputs["k"][0, tok],
                inputs["v"][0, tok],
                inputs["g"][0, tok],
                inputs["beta"][0, tok],
                inputs["a_log"],
                inputs["g_bias"].view(H, D),
                s,
            )
            expected_state[indices[n, t]] = s

    torch.testing.assert_close(out[0].float(), expected, rtol=1e-2, atol=1e-3)
    used = indices.flatten().long()
    torch.testing.assert_close(state[used], expected_state[used], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("num_seqs", "query_len"), [(7, 1), (3, 3)], ids=["7x1", "3x3"]
)
@torch.inference_mode()
def test_fused_recurrent_kda_strided_inputs_bit_identical_to_contiguous(
    num_seqs: int, query_len: int
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, query_len, device)
    for name in ("q", "k", "v", "beta"):
        assert not inputs[name].is_contiguous()
    contiguous = {
        name: x.contiguous() if name in ("q", "k", "v", "beta") else x
        for name, x in inputs.items()
    }
    state_ref = state.clone()
    out_ref = run_kernel(contiguous, state_ref)
    out = run_kernel(inputs, state)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=0)


@torch.inference_mode()
def test_fused_recurrent_kda_rejects_unaddressable_layouts():
    """Layouts the token-stride addressing cannot express must fail loudly
    rather than read the wrong tokens: a batch slice of a wider buffer
    (``stride(0) != T * stride(1)``), overlapping tokens, and a head-strided
    (transposed) block.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(1, 1, device)
    T = 4
    base = torch.randn(4, T, 2 * H * D, dtype=torch.bfloat16, device=device)
    bad_q = {
        "batch slice": base[::2, :, : H * D].view(2, T, H, D),
        "overlapping tokens": base[:1, :, : H * D].as_strided(
            (1, T, H, D), (0, D, D, 1)
        ),
        "head-strided": base[:1, :, : H * D].view(1, T, D, H).transpose(2, 3),
    }
    for q in bad_q.values():
        broken = dict(inputs, q=q, k=q, v=q)
        broken["cu_seqlens"] = None if q.shape[0] > 1 else inputs["cu_seqlens"]
        with pytest.raises(AssertionError, match=r"torch.Size"):
            run_kernel(broken, state)
