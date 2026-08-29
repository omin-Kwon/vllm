# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.fused_recurrent_replayssm import (
    fused_recurrent_gated_delta_rule_replayssm,
)

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Gated delta rule Triton kernels require a CUDA-alike or XPU device.",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype, strided_mixed_qkv: bool
):
    torch.manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = 32
    H = 4
    HV = 8  # grouped value attention: HV must be divisible by H
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device(DEVICE)

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn((B, qkv_dim + 64), device=device, dtype=dtype)
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn((B, qkv_dim), device=device, dtype=dtype)

    a = torch.randn((B, HV), device=device, dtype=dtype)
    b = torch.randn((B, HV), device=device, dtype=dtype)
    A_log = torch.randn((HV,), device=device, dtype=dtype)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases). Index 0 is
    # reserved as NULL_BLOCK_ID (CUDA graph padding), so valid slots start at 1.
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    ssm_state_indices[-3:] = -1

    state0 = torch.randn((B + 1, HV, V, K), device=device, dtype=dtype)
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).to(dtype).unsqueeze(1)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    # Output rows for PAD_SLOT_ID entries are never written (uninitialized in
    # both paths), so compare only the valid rows.
    valid = ssm_state_indices > 0
    torch.testing.assert_close(out_packed[valid], out_ref[valid], rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


def test_packed_decode_supports_large_batch_head_grid():
    B, H, HV, K, V = 1024, 8, 64, 1, 1
    device = torch.device(DEVICE)
    gates = torch.empty((B, HV), device=device)
    params = torch.empty((HV,), device=device)
    out = torch.empty((B, 1, HV, V), device=device)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.empty((B, 2 * H * K + HV * V), device=device),
        a=gates,
        b=gates,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=torch.empty((1, HV, V, K), device=device),
        out=out,
        ssm_state_indices=torch.zeros((B,), device=device, dtype=torch.int32),
    )

    assert torch.count_nonzero(out).item() == 0


def test_replayssm_decode_advances_across_flush_boundary():
    """Replay output and checkpoint must track the materialized recurrence."""
    torch.manual_seed(0)
    device = torch.device(DEVICE)
    dtype = torch.float32
    batch, num_k_heads, num_v_heads = 1, 2, 4
    key_dim = value_dim = 128
    cache_len = 4
    num_steps = 2 * cache_len + 1
    qkv_dim = 2 * num_k_heads * key_dim + num_v_heads * value_dim

    mixed_qkv = 0.1 * torch.randn(num_steps, qkv_dim, device=device, dtype=dtype)
    a = 0.1 * torch.randn(num_steps, num_v_heads, device=device, dtype=dtype)
    b = 0.1 * torch.randn_like(a)
    A_log = torch.zeros(num_v_heads, device=device, dtype=dtype)
    dt_bias = torch.zeros_like(A_log)
    state_indices = torch.tensor([1], device=device, dtype=torch.int32)
    state_baseline = 0.1 * torch.randn(
        2, num_v_heads, value_dim, key_dim, device=device, dtype=dtype
    )
    state_replay = state_baseline.clone()
    d_cache = torch.empty(
        2, num_v_heads, cache_len, value_dim, device=device, dtype=dtype
    )
    k_cache = torch.empty(
        2, num_k_heads, cache_len, key_dim, device=device, dtype=dtype
    )
    g_cache = torch.empty(2, num_v_heads, cache_len, device=device, dtype=torch.float32)

    for step in range(num_steps):
        out_baseline = torch.empty(
            batch, 1, num_v_heads, value_dim, device=device, dtype=dtype
        )
        out_replay = torch.empty_like(out_baseline)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[step : step + 1],
            a=a[step : step + 1],
            b=b[step : step + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=state_baseline,
            out=out_baseline,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mixed_qkv[step : step + 1],
            a=a[step : step + 1],
            b=b[step : step + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=state_replay,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            out=out_replay,
            ssm_state_indices=state_indices,
            write_pos=torch.tensor(
                [step % cache_len], device=device, dtype=torch.int32
            ),
            use_qk_l2norm_in_kernel=True,
        )

        torch.testing.assert_close(out_replay, out_baseline, rtol=3e-3, atol=3e-3)
        if step % cache_len == cache_len - 1:
            torch.testing.assert_close(
                state_replay[1], state_baseline[1], rtol=3e-3, atol=3e-3
            )


def test_replayssm_matches_flash_next_precision_across_two_flushes():
    """Exercise the Flash-Next head geometry and its mixed cache dtypes."""
    torch.manual_seed(17)
    device = torch.device(DEVICE)
    activation_dtype = torch.bfloat16
    state_dtype = torch.float32
    batch, num_k_heads, num_v_heads = 1, 16, 48
    key_dim = value_dim = 128
    cache_len = 16
    num_steps = 2 * cache_len + 1
    qkv_dim = 2 * num_k_heads * key_dim + num_v_heads * value_dim

    mixed_qkv = (0.1 * torch.randn(num_steps, qkv_dim, device=device)).to(
        activation_dtype
    )
    a = (0.1 * torch.randn(num_steps, num_v_heads, device=device)).to(activation_dtype)
    b = (0.1 * torch.randn_like(a)).to(activation_dtype)
    A_log = torch.zeros(num_v_heads, device=device, dtype=activation_dtype)
    dt_bias = torch.zeros_like(A_log)
    state_indices = torch.tensor([1], device=device, dtype=torch.int32)
    state_baseline = 0.1 * torch.randn(
        2,
        num_v_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=state_dtype,
    )
    state_replay = state_baseline.clone()
    d_cache = torch.empty(
        2,
        num_v_heads,
        cache_len,
        value_dim,
        device=device,
        dtype=activation_dtype,
    )
    k_cache = torch.empty(
        2,
        num_k_heads,
        cache_len,
        key_dim,
        device=device,
        dtype=activation_dtype,
    )
    g_cache = torch.empty(2, num_v_heads, cache_len, device=device, dtype=torch.float32)

    for step in range(num_steps):
        out_baseline = torch.empty(
            batch,
            1,
            num_v_heads,
            value_dim,
            device=device,
            dtype=activation_dtype,
        )
        out_replay = torch.empty_like(out_baseline)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv[step : step + 1],
            a=a[step : step + 1],
            b=b[step : step + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=state_baseline,
            out=out_baseline,
            ssm_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mixed_qkv[step : step + 1],
            a=a[step : step + 1],
            b=b[step : step + 1],
            A_log=A_log,
            dt_bias=dt_bias,
            scale=key_dim**-0.5,
            initial_state=state_replay,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            out=out_replay,
            ssm_state_indices=state_indices,
            write_pos=torch.tensor(
                [step % cache_len], device=device, dtype=torch.int32
            ),
            use_qk_l2norm_in_kernel=True,
        )

        if step == 0:
            torch.testing.assert_close(out_replay, out_baseline, rtol=0, atol=0)
        else:
            torch.testing.assert_close(out_replay, out_baseline, rtol=5e-3, atol=5e-4)
        if step % cache_len == cache_len - 1:
            torch.testing.assert_close(
                state_replay[1], state_baseline[1], rtol=5e-3, atol=5e-4
            )


def test_replayssm_rejects_short_cursor_metadata():
    device = torch.device(DEVICE)
    batch, num_k_heads, num_v_heads = 2, 1, 1
    key_dim = value_dim = 16
    mixed_qkv = torch.zeros(
        batch,
        2 * num_k_heads * key_dim + num_v_heads * value_dim,
        device=device,
    )
    gates = torch.zeros(batch, num_v_heads, device=device)
    params = torch.zeros(num_v_heads, device=device)
    state = torch.zeros(2, num_v_heads, value_dim, key_dim, device=device)
    out = torch.empty(batch, 1, num_v_heads, value_dim, device=device)

    with pytest.raises(ValueError, match="at least B=2 entries"):
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mixed_qkv,
            a=gates,
            b=gates,
            A_log=params,
            dt_bias=params,
            scale=key_dim**-0.5,
            initial_state=state,
            d_cache=torch.zeros(2, num_v_heads, 4, value_dim, device=device),
            k_cache=torch.zeros(2, num_k_heads, 4, key_dim, device=device),
            g_cache=torch.zeros(2, num_v_heads, 4, device=device, dtype=torch.float32),
            out=out,
            ssm_state_indices=torch.ones(batch, device=device, dtype=torch.int32),
            write_pos=torch.zeros(1, device=device, dtype=torch.int32),
        )
