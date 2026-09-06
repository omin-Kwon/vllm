# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check the experimental step against the unchanged CUDA Phi implementation.

Guard identity elision and packed per-head addressing, graph replay, padding,
heterogeneous dense/sketch heads, and compatibility with exact WY refresh.
"""

import math

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops.gdn_sketch import (
    StepPlan,
    pack_metadata,
    step,
)
from vllm.third_party.flash_linear_attention.ops.gdn_step_full_cuda import (
    step as cuda_step,
)
from vllm.third_party.flash_linear_attention.ops.gdn_step_phi_cuda import gdn_step_cuda


@pytest.mark.parametrize("g", [4, 32, 64, 128])
def test_full_runtime_graph_with_paged_rings_and_staggered_flush(g, monkeypatch):
    """Catch runtime dispatch, page-stride, and device flush-count errors."""
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent_replayssm import (
        fused_recurrent_gated_delta_rule_replayssm as decode,
    )
    from vllm.third_party.flash_linear_attention.ops.gdn_full_runtime import (
        FullCoordinateRuntime,
    )
    from vllm.third_party.flash_linear_attention.ops.gdn_ls6_epilogue_cuda import (
        gdn_ls6_epilogue,
    )

    monkeypatch.setenv("NS_GDN_STEP_IMPL", "cuda")
    monkeypatch.setenv("NS_GDN_FLUSH_IMPL", "stream")
    torch.manual_seed(982)
    ns, batch, h, hv, k, w = 5, 4, 3, 9, 128, 16
    shapes = [(hv, k, k), (hv, w, k), (h, w, k), (hv, w)]
    sizes = [math.prod(shape) for shape in shapes]
    page = ((sum(sizes) + 127) // 128 + 1) * 128
    initial = torch.randn(ns, page, device="cuda") * 0.03

    def buffers():
        storage = initial.clone()
        views, offset = [], 0
        for shape, size in zip(shapes, sizes):
            views.append(storage[:, offset : offset + size].view(ns, *shape))
            offset += size
        for ring in views[1:]:
            ring.zero_()
        return storage, views

    ids = torch.tensor([3, 1, 2, 0], dtype=torch.int32, device="cuda")
    mapping = ids.new_tensor([0, 2, 3, 1, 4])
    widths = ids.new_tensor([0, 1, g, g - 1, g, max(1, g - 3), 0, g, g])
    rows = ids.new_tensor([3, 1, 2, 0, 3])
    mixed = torch.randn(batch, 2 * h * k + hv * k, device="cuda", dtype=torch.bfloat16)
    a = torch.full((batch, hv), -3.0, device="cuda")
    b = torch.randn_like(a)
    a_log = torch.zeros(hv, device="cuda")
    bias = torch.zeros_like(a_log)
    runtime = FullCoordinateRuntime(batch, h, hv, g, "cuda")
    worlds = []
    for use_full in (False, True):
        storage, (state, writes, keys, gates) = buffers()
        if not use_full:
            state = state.contiguous()
            # The unchanged legacy wrapper accepts only contiguous rings.
            writes, keys, gates = [x.contiguous() for x in (writes, keys, gates)]
        u = torch.full((ns, hv, g, k), float("nan"), device="cuda")
        phi = torch.full_like(u, float("nan"))
        aq = torch.zeros(ns, hv, g, device="cuda")
        ak = torch.zeros_like(aq)
        factors = torch.zeros(ns, hv, w, g, device="cuda")
        beta = torch.zeros(ns, hv, w, device="cuda")
        qbar = torch.zeros(ns, h, k, device="cuda")
        kbar = torch.zeros_like(qbar)
        output = torch.zeros(batch, 1, hv, k, dtype=mixed.dtype, device="cuda")
        pos = ids.new_tensor([15, 0, 7, 0])
        common = dict(
            ls6_ubar=u,
            ls6_phi=phi,
            ls6_mh=widths,
            ls6_fs=factors,
            ls6_r=128,
            ls6_map=mapping,
            ls6_beta=beta,
        )
        if use_full:
            runtime.refresh(state, ids, mapping, widths, u, phi)
            common["ls6_full_workspace"] = runtime
        else:
            gdn_ls6_epilogue(
                state,
                rows,
                rows[batch:],
                mapping,
                qbar,
                kbar,
                widths,
                u,
                phi,
                aq,
                ak,
                batch,
                h,
                hv,
                k,
                k,
                g,
                128,
            )
            common.update(
                ls6_aq=aq,
                ls6_ak=ak,
                fz_nf=torch.where(widths > 0, 0, k).int(),
                fz_u=torch.zeros(ns, hv, k, device="cuda"),
                fz_z=torch.zeros(ns, hv, k, device="cuda"),
                fz_qbar=qbar,
                fz_kbar=kbar,
            )
        kwargs = dict(
            mixed_qkv=mixed,
            a=a,
            b=b,
            A_log=a_log,
            dt_bias=bias,
            scale=k**-0.5,
            initial_state=state,
            d_cache=writes,
            k_cache=keys,
            g_cache=gates,
            out=output,
            ssm_state_indices=ids,
            write_pos=pos,
            use_qk_l2norm_in_kernel=True,
            **common,
        )

        # The legacy path records beta in its model hook; the new step fuses it.
        def token(kwargs=kwargs, pos=pos, beta=beta, use_full=use_full):
            if not use_full:
                beta[mapping[ids].long(), :, pos.long()] = torch.sigmoid(b)
            decode(**kwargs)
            pos.copy_((pos + 1) % w)

        token()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            token()
        # CUDA graphs retain pointers, so keep every input allocation alive.
        worlds.append((graph, storage, output, state, kwargs))
    for _ in range(48):
        for graph, *_ in worlds:
            graph.replay()
        torch.testing.assert_close(worlds[0][2], worlds[1][2], atol=4e-3, rtol=4e-3)
        torch.testing.assert_close(worlds[0][3], worlds[1][3], atol=5e-5, rtol=4e-3)
    for _, storage, _, _, _ in worlds:
        torch.testing.assert_close(storage[:, -128:], initial[:, -128:], atol=0, rtol=0)


@pytest.mark.parametrize("pattern", ["random", "collinear", "zero"])
@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_augmented_qr_preserves_ridge_projection(monkeypatch, pattern, scale):
    """Guard the ridge numerator, mixed-width padding, and compact addressing.

    The independent double-precision solve is an oracle only. The research
    kernels run FP32, with three TF32 products for the coefficient matrix.
    """
    import os
    from pathlib import Path

    if os.environ.get("NS_GDN_QR_RESEARCH_TESTS") != "1":
        pytest.skip("Opt-in B200 research kernel; build gdn_qr_mgs.cu first")
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "benchmarks/kernels"))
    from gdn_qr_research import QRWorkspace

    torch.manual_seed(718)
    ns, h, hv, g = 5, 11, 33, 32
    state = torch.randn(ns, hv, 128, 128, device="cuda") * scale
    if pattern == "collinear":
        state = state[..., :1].expand_as(state).clone()
    elif pattern == "zero":
        state.zero_()
    u = torch.full((ns, hv, g, 128), float("nan"), device="cuda")
    phi = torch.full_like(u, float("nan"))
    widths = torch.arange(hv, dtype=torch.int32, device="cuda")
    rows = widths.new_tensor([3, 1, 2, 0, 3])
    mapping = widths.new_tensor([0, 3, 1, 2, 4])
    workspace = QRWorkspace(4, h, hv, g, "cuda", ridge=0.1, mgs=True)
    # Refresh needs only the boundary state and output metadata.
    args = (state, None, None, None, rows, mapping, widths, None, u, phi)
    workspace.refresh(*args)
    for slot in [1, 2, 3]:
        compact = int(mapping[slot])
        for m in range(1, 33):
            matrix = state[slot, m].double()
            sketch = matrix[:, :m]
            eta = 0.1 * matrix.square().sum() / 128
            expected = torch.zeros_like(matrix)
            if pattern != "zero":
                gram = sketch.T @ sketch + eta * torch.eye(m, device="cuda")
                numerator = sketch.T @ matrix
                numerator[:, :m] += eta * torch.eye(m, device="cuda")
                expected = sketch @ torch.linalg.solve(gram, numerator)
            got = u[compact, m, :m].double().T @ phi[compact, m, :m].double()
            torch.testing.assert_close(got, expected, atol=5e-6 * scale, rtol=1e-4)
            assert torch.count_nonzero(u[compact, m, m:]) == 0
            assert torch.count_nonzero(phi[compact, m, m:]) == 0
    assert u[[0, 4]].isnan().all() and phi[[0, 4]].isnan().all()
    assert u[:, 0].isnan().all() and phi[:, 0].isnan().all()
    before = (u.clone(), phi.clone())
    rows[-1] = 0
    workspace.refresh(*args)
    torch.testing.assert_close(u, before[0], atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(phi, before[1], atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("g", [4, 8, 16, 20, 32, 36, 64, 128])
@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16])
def test_full_dimension_gpu_flush_across_windows(g, gate_dtype):
    """Catch refresh/fold/addressing errors through actual next-window outputs.

    The unchanged stream+solve is one oracle; a dense recurrence and FP64
    boundary coefficient solve independently check its mathematical contract.
    Full-width metadata is poisoned to catch unintended Phi reads.
    """
    from vllm.third_party.flash_linear_attention.ops.gdn_flush_full_cuda import (
        FlushWorkspace,
    )
    from vllm.third_party.flash_linear_attention.ops.gdn_flush_stream_cuda import (
        gdn_flush_stream,
    )

    torch.manual_seed(173)
    dev = "cuda"
    nx, ns, batch, h, hv, k, w = 5, 4, 4, 3, 9, 128, 16
    widths = [0, 0, 0, g, 1, max(1, g - 3), g, g - 1, g]
    mh = torch.tensor(widths, dtype=torch.int32, device=dev)
    mapping = torch.tensor([0, 3, 1, 2, 0], dtype=torch.int32, device=dev)
    idx = torch.tensor([3, 0, 1, 2], dtype=torch.int32, device=dev)
    rows = torch.tensor([3, 1, 2, 0, 3], dtype=torch.int32, device=dev)
    pos = torch.zeros(batch, dtype=torch.int32, device=dev)
    backing = torch.randn(nx, hv * k * k + 128, device=dev) * 0.03
    state = backing.as_strided((nx, hv, k, k), (backing.stride(0), k * k, k, 1))
    initial = state.clone()
    padding = backing[:, -128:].clone()
    reference_state = initial.clone()
    u = torch.full((ns, hv, g, k), float("nan"), device=dev)
    phi = torch.full_like(u, float("nan"))
    ref_u, ref_phi = u.clone(), phi.clone()
    anchors = torch.zeros(ns, hv, g, device=dev)
    aq, ak = anchors.clone(), anchors.clone()
    qbar = torch.randn(ns, h, k, device=dev) * 0.1
    kbar = torch.randn_like(qbar) * 0.1
    zero = torch.zeros(ns, hv, k, device=dev)
    beta_ring = torch.zeros(ns, hv, w, device=dev)
    nf = torch.where(mh > 0, 0, k).int()

    def rings():
        return [
            torch.zeros(*shape, device=dev)
            for shape in [(nx, hv, w, k), (nx, h, w, k), (nx, hv, w), (ns, hv, w, g)]
        ]

    actual, reference = rings(), rings()
    workspace = FlushWorkspace(batch, h, hv, g, dev, ridge=0.1, grid=2)
    args = (
        state,
        actual[0],
        actual[1],
        actual[2],
        rows,
        mapping,
        mh,
        beta_ring,
        u,
        phi,
    )
    workspace.refresh(*args)
    for slot in (1, 2, 3):
        c = int(mapping[slot])
        for head, m in enumerate(widths):
            if not m:
                continue
            hs = initial[slot, head].double()
            gram = hs.T @ hs
            gram += 0.1 * gram.trace() / k * torch.eye(k, device=dev)
            want = torch.linalg.solve(gram[:m, :m], gram[:m]).float()
            ref_phi[c, head].zero_()
            ref_phi[c, head, :m] = want
            ref_u[c, head].zero_()
            ref_u[c, head, :m] = hs[:, :m].T.float()
            torch.testing.assert_close(u[c, head], ref_u[c, head], atol=0, rtol=0)
            if m < k:
                torch.testing.assert_close(
                    phi[c, head], ref_phi[c, head], atol=3e-5, rtol=3e-3
                )

    mix = torch.randn(batch, 2 * h * k + hv * k, device=dev, dtype=torch.bfloat16)
    a = torch.full((batch, hv), -3.0, device=dev, dtype=gate_dtype)
    b = torch.zeros_like(a)
    a_log, bias = (
        torch.zeros(hv, device=dev, dtype=gate_dtype),
        torch.zeros(hv, device=dev, dtype=gate_dtype),
    )
    out, expected = (
        torch.empty(batch, hv, k, device=dev, dtype=mix.dtype),
        torch.empty(batch, hv, k, device=dev, dtype=mix.dtype),
    )

    def launch_step():
        cuda_step(
            mix,
            a,
            b,
            a_log,
            bias,
            out,
            state,
            actual[0],
            actual[1],
            actual[2],
            idx,
            pos,
            u,
            phi,
            mh,
            actual[3],
            mapping,
            k**-0.5,
            beta_ring=beta_ring,
        )

    def old_flush():
        gdn_flush_stream(
            reference_state,
            reference[0],
            reference[1],
            reference[2],
            rows,
            batch,
            qbar,
            kbar,
            mapping,
            mh,
            beta_ring,
            ref_u,
            ref_phi,
            aq,
            ak,
            h,
            hv,
            k,
            k,
            w,
            g,
            k,
            ridge=0.1,
            grid=2,
        )

    launch_step()
    # Initialize the old extension outside capture, with an empty flush list.
    rows[-1] = 0
    old_flush()
    torch.accelerator.synchronize()
    step_graph, flush_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(step_graph):
        launch_step()
    with torch.cuda.graph(flush_graph):
        workspace.flush(*args)
    rows[-1] = 3
    for tensor in actual:
        tensor.zero_()
    dense = initial.double()
    for window in range(3):
        for t in range(w):
            mix.normal_()
            a.normal_(-3, 0.2)
            b.normal_()
            pos.fill_(t)
            be = b.float().sigmoid().to(gate_dtype).float()
            step_graph.replay()
            for row, slot in ((0, 3), (2, 1), (3, 2)):
                torch.testing.assert_close(
                    beta_ring[mapping[slot], :, t], be[row], atol=1e-7, rtol=1e-6
                )
            gdn_step_cuda(
                batch,
                mix,
                a,
                b,
                a_log,
                bias,
                expected,
                reference_state,
                reference[0],
                reference[1],
                reference[2],
                idx,
                pos,
                k**-0.5,
                nf,
                zero,
                zero,
                ref_u,
                ref_phi,
                mh,
                aq,
                ak,
                reference[3],
                h,
                hv,
                k,
                k,
                w,
                g,
                k,
                True,
                False,
                ls6_map=mapping,
            )
            torch.testing.assert_close(out, expected, atol=4e-3, rtol=4e-3)
            for got, want in zip(actual, reference):
                torch.testing.assert_close(got, want, atol=5e-5, rtol=4e-3)
            for row, slot in ((0, 3), (2, 1), (3, 2)):
                keys = actual[1][slot, :, t].double().repeat_interleave(3, dim=0)
                alpha = actual[2][slot, :, t].double().exp()[:, None, None]
                val = mix[row, 2 * h * k :].reshape(hv, k).double()
                sh = dense[slot]
                delta = be[row, :, None].double() * (
                    val - alpha[:, :, 0] * (sh @ keys[:, :, None])[:, :, 0]
                )
                dense[slot] = alpha * sh + delta[:, :, None] * keys[:, None, :]
        # Empty graph replay must not touch anything, even after previous work.
        rows[-1] = 0
        snapshot = state.clone()
        flush_graph.replay()
        torch.testing.assert_close(state, snapshot, atol=0, rtol=0)
        if window == 1:
            rows[-1] = 1
            flush_graph.replay()
            old_flush()
            torch.testing.assert_close(state[1:3], snapshot[1:3], atol=0, rtol=0)
            rows[:2] = torch.tensor([1, 2], dtype=torch.int32, device=dev)
            rows[-1] = 2
            flush_graph.replay()
            old_flush()
            rows[:3] = torch.tensor([3, 1, 2], dtype=torch.int32, device=dev)
            rows[-1] = 3
        else:
            rows[-1] = 3
            flush_graph.replay()
            old_flush()
        torch.testing.assert_close(state, reference_state, atol=5e-5, rtol=4e-3)
        torch.testing.assert_close(state.double(), dense, atol=5e-5, rtol=4e-3)
        torch.testing.assert_close(backing[:, -128:], padding, atol=0, rtol=0)
        for slot in (1, 2, 3):
            c = int(mapping[slot])
            for head, m in enumerate(widths):
                if not m:
                    continue
                torch.testing.assert_close(
                    u[c, head], ref_u[c, head], atol=5e-5, rtol=4e-3
                )
                if m == k:
                    assert phi[c, head].isnan().all()
                else:
                    torch.testing.assert_close(
                        phi[c, head], ref_phi[c, head], atol=5e-5, rtol=4e-3
                    )
                    hs = state[slot, head].double()
                    gram = hs.T @ hs
                    gram += 0.1 * gram.trace() / k * torch.eye(k, device=dev)
                    want = torch.linalg.solve(gram[:m, :m], gram[:m]).float()
                    torch.testing.assert_close(
                        phi[c, head, :m], want, atol=5e-5, rtol=4e-3
                    )
        assert phi[0].isnan().all() and u[0].isnan().all()


@pytest.mark.parametrize("hpg", [1, 2, 3, 4])
def test_full_dimension_flush_worklist_rollover(hpg):
    """Exercise table refills, shared-key groups, and every actual width."""
    from vllm.third_party.flash_linear_attention.ops.gdn_flush_full_cuda import (
        FlushWorkspace,
    )
    from vllm.third_party.flash_linear_attention.ops.gdn_flush_stream_cuda import (
        gdn_flush_stream,
    )

    torch.manual_seed(91)
    h = 33 if hpg == 4 else 7
    nx, hv, g = 15, h * hpg, 128
    dev = "cuda"
    widths = (
        [*range(129), 0, 32, 128]
        if hpg == 4
        else ([0, 1, 5, 8, 64, 127, 128] * hpg)[:hv]
    )
    mh = torch.tensor(widths, dtype=torch.int32, device=dev)
    mapping = torch.tensor([0, *range(13, 0, -1), 14], dtype=torch.int32, device=dev)
    rows = torch.tensor([*range(1, 14), 0, 13], dtype=torch.int32, device=dev)
    state = torch.randn(nx, hv, 128, 128, device=dev) * 0.1
    initial = state.clone()
    ref = state.clone()
    writes = torch.randn(nx, hv, 16, 128, device=dev) * 0.03
    keys = torch.nn.functional.normalize(
        torch.randn(nx, h, 16, 128, device=dev), dim=-1
    )
    gates = -torch.rand(nx, hv, 16, device=dev) * 0.1
    beta = torch.rand(nx, hv, 16, device=dev)
    u = torch.full((nx, hv, g, 128), float("nan"), device=dev)
    phi = torch.full_like(u, float("nan"))
    ref_u, ref_phi = u.clone(), phi.clone()
    qbar = torch.randn(nx, h, 128, device=dev)
    kbar = torch.randn_like(qbar)
    aq, ak = torch.zeros(nx, hv, g, device=dev), torch.zeros(nx, hv, g, device=dev)
    workspace = FlushWorkspace(14, h, hv, g, dev, ridge=0.01, grid=1)
    workspace.flush(state, writes, keys, gates, rows, mapping, mh, beta, u, phi)
    gdn_flush_stream(
        ref,
        writes,
        keys,
        gates,
        rows,
        14,
        qbar,
        kbar,
        mapping,
        mh,
        beta,
        ref_u,
        ref_phi,
        aq,
        ak,
        h,
        hv,
        128,
        128,
        16,
        g,
        128,
        ridge=0.01,
        grid=1,
    )
    torch.testing.assert_close(state, ref, atol=5e-5, rtol=4e-3)
    torch.testing.assert_close(state[[0, 14]], initial[[0, 14]], atol=0, rtol=0)
    for head, m in enumerate(widths):
        if m:
            torch.testing.assert_close(
                u[1:14, head], ref_u[1:14, head], atol=5e-5, rtol=4e-3
            )
            if m < 128:
                torch.testing.assert_close(
                    phi[1:14, head], ref_phi[1:14, head], atol=5e-5, rtol=4e-3
                )
            else:
                assert phi[:, head].isnan().all()


@pytest.mark.parametrize(
    "mode", ["plain", "buckets", "shared", "cuda", "cuda_bf16_gates"]
)
@pytest.mark.parametrize("g", [8, 32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_dimension_step(g, dtype, mode):
    torch.manual_seed(83)
    dev = "cuda"
    ns, h, hv, k, v, w = 3, 2, 6, 128, 128, 16
    widths = [g, 0, max(1, g - 3), g, 127 if g == 128 else 1, g]
    mh = torch.tensor(widths, device=dev, dtype=torch.int32)
    idx = torch.tensor([2, 0, 1], device=dev, dtype=torch.int32)
    mapping = torch.tensor([0, 2, 1], device=dev, dtype=torch.int32)
    pos = torch.zeros(ns, device=dev, dtype=torch.int32)
    backing = torch.randn(ns, hv * v * k + 128, device=dev) * 0.03
    state = backing.as_strided((ns, hv, v, k), (backing.stride(0), v * k, k, 1))
    initial = state.clone()
    phi = torch.zeros(ns, hv, g, k, device=dev)
    u = torch.zeros(ns, hv, g, v, device=dev)
    for slot in (1, 2):
        c = int(mapping[slot])
        for head, m in enumerate(widths):
            if m:
                hs = state[slot, head].double()
                gram = hs.T @ hs
                gram += 0.1 * gram.trace() / k * torch.eye(k, device=dev)
                phi[c, head, :m] = torch.linalg.solve(gram[:m, :m], gram[:m]).float()
                u[c, head, :m] = hs[:, :m].T.float()
    packed, offsets = pack_metadata(phi, widths)
    assert packed.shape == (ns, sum(m * (k - m) for m in widths))

    def rings():
        return [
            torch.zeros(*shape, device=dev)
            for shape in [(ns, hv, w, v), (ns, h, w, k), (ns, hv, w), (ns, hv, w, g)]
        ]

    actual, reference = rings(), rings()
    mix_storage = torch.empty(ns, 2 * h * k + hv * v + 4, device=dev, dtype=dtype)
    mix = mix_storage[:, :-4]
    gate_dtype = torch.bfloat16 if mode == "cuda_bf16_gates" else torch.float32
    a = torch.empty(ns, hv, device=dev, dtype=gate_dtype)
    beta = torch.empty_like(a)
    a_log = torch.zeros(hv, device=dev, dtype=gate_dtype)
    bias = torch.zeros_like(a_log)
    out = torch.empty(ns, hv, v, device=dev, dtype=dtype)
    expected = torch.empty_like(out)
    zero = torch.zeros(ns, hv, v, device=dev)
    anchors = torch.zeros(ns, hv, g, device=dev)
    nf = torch.where(mh > 0, 0, k).int()

    plan = (
        None
        if mode.startswith("cuda") or mode == "plain"
        else StepPlan(widths, ns, h, dev, share_keys=mode == "shared")
    )

    def launch():
        if mode.startswith("cuda"):
            cuda_step(
                mix,
                a,
                beta,
                a_log,
                bias,
                out,
                state,
                actual[0],
                actual[1],
                actual[2],
                idx,
                pos,
                u,
                phi,
                mh,
                actual[3],
                mapping,
                k**-0.5,
            )
            return
        step(
            mix,
            a,
            beta,
            a_log,
            bias,
            out,
            state,
            actual[0],
            actual[1],
            actual[2],
            idx,
            pos,
            mapping,
            u,
            packed,
            offsets,
            mh,
            actual[3],
            k**-0.5,
            plan=plan,
        )

    mix.normal_()
    a.fill_(-3)
    beta.zero_()
    launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for tensor in actual:
        tensor.zero_()
    dense = initial.double()
    betas = []
    for t in range(w):
        mix.normal_()
        a.normal_(-3, 0.2)
        beta.normal_()
        pos.fill_(t)
        graph.replay()
        gdn_step_cuda(
            ns,
            mix,
            a,
            beta,
            a_log,
            bias,
            expected,
            state,
            reference[0],
            reference[1],
            reference[2],
            idx,
            pos,
            k**-0.5,
            nf,
            zero,
            zero,
            u,
            phi,
            mh,
            anchors,
            anchors,
            reference[3],
            h,
            hv,
            k,
            v,
            w,
            g,
            k,
            True,
            False,
            ls6_map=mapping,
        )
        tol = 3e-5 if dtype == torch.float32 else 4e-3
        torch.testing.assert_close(out, expected, atol=tol, rtol=3e-3)
        for got, want in zip(actual, reference):
            torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-3)
        betas.append(torch.sigmoid(beta).clone())
        for row, slot in ((0, 2), (2, 1)):
            for head in range(hv):
                key = actual[1][slot, head // 3, t].double()
                alpha = actual[2][slot, head, t].double().exp()
                be = betas[-1][row, head].double()
                val = mix[
                    row, 2 * h * k + head * v : 2 * h * k + (head + 1) * v
                ].double()
                sh = dense[slot, head]
                dense[slot, head] = alpha * sh + torch.outer(
                    be * (val - alpha * (sh @ key)), key
                )
    # Materialize the exact block-WY boundary from the new kernel's rings.
    for row, slot in ((0, 2), (2, 1)):
        for head, m in enumerate(widths):
            keys = actual[1][slot, head // 3].double()
            logs = actual[2][slot, head].double()
            gamma = logs.sum().exp()
            decay = (logs.sum() - logs.cumsum(0)).exp()
            corrected = decay[:, None] * actual[0][slot, head].double()
            if m:
                native: list[torch.Tensor] = []
                for t in range(w):
                    f = keys[t].clone()
                    if t:
                        f -= (keys[:t] @ keys[t]) @ torch.stack(native)
                    native.append(betas[t][row, head].double() * f)
                corrected -= gamma * (
                    torch.stack(native) @ initial[slot, head].double().T
                )
            boundary = gamma * initial[slot, head].double() + corrected.T @ keys
            torch.testing.assert_close(
                boundary, dense[slot, head], atol=3e-5, rtol=3e-3
            )
    torch.testing.assert_close(state, initial, atol=0, rtol=0)
