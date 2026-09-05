# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, index_topk=2048),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192


@pytest.mark.skipif(not current_platform.is_cuda(), reason="SM120 GPU required")
@pytest.mark.parametrize("capacity", [2048, 2176])
@pytest.mark.parametrize("num_queries", [2, 65])
def test_nope_packed_cache_attention_matches_unpadded_reference(capacity, num_queries):
    """Padding must preserve attention over the actual quantized cached KV."""
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
        FlashInferMLASparseSM120Impl,
    )

    if not current_platform.is_device_capability_family(120):
        pytest.skip("SM120 GPU required")
    torch.manual_seed(17)
    indices = torch.full((num_queries, capacity), -1, dtype=torch.int32, device="cuda")
    indices[:, :6] = torch.arange(6, device="cuda")
    # The widened kpool tail must contribute; it cannot be truncated to top-k.
    indices[:, -1] = 6
    indices[0] = -1
    with set_current_vllm_config(_fake_vllm_config("glm5_next")):
        impl = FlashInferMLASparseSM120Impl(
            num_heads=8,
            head_size=512,
            scale=1 / 16,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8_ds_mla",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            kv_lora_rank=512,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            topk_indices_buffer=indices,
        )
    kv = torch.randn(7, 512, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(1, 64, 656, dtype=torch.uint8, device="cuda")
    impl.do_kv_cache_update(
        kv,
        kv.new_empty(7, 1, 0),
        cache,
        torch.arange(7, device="cuda"),
        "fp8_ds_mla",
        torch.ones(1, device="cuda"),
    )
    rows = cache[0, :7]
    assert not rows[:, 528:].count_nonzero()
    decoded = (
        rows[:, :512].contiguous().view(torch.float8_e4m3fn).float().reshape(7, 4, 128)
        * rows[:, 512:528].contiguous().view(torch.float32)[..., None]
    ).reshape(7, 512)
    query = torch.randn(num_queries, 8, 512, dtype=torch.bfloat16, device="cuda")
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(num_queries, dtype=torch.int32, device="cuda"),
        block_table=torch.zeros(1, 1, dtype=torch.int32, device="cuda"),
        block_size=64,
        topk_tokens=2048,
    )
    output, _ = impl.forward_mqa(query, cache, metadata, None)
    # SM120 also quantizes query tiles to FP8 with power-of-two scales.
    q_tiles = query.float().reshape(num_queries, 8, 4, 128)
    q_scale = torch.exp2(
        (q_tiles.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448).log2().ceil()
    )
    q_decoded = ((q_tiles / q_scale).to(torch.float8_e4m3fn).float() * q_scale).reshape(
        num_queries, 8, 512
    )
    expected = (q_decoded @ decoded.T / 16).softmax(-1) @ decoded
    expected[0] = 0
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
