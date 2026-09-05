# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

import math
from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True
    supports_dense_mha_prefill = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        if self.qk_rope_head_dim not in (0, 64):
            raise NotImplementedError(
                "SM120 packed sparse MLA supports RoPE dim 0 or 64"
            )
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    def do_kv_cache_update(
        self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
    ) -> None:
        # The packed V32 cache has 64 physical RoPE channels. Zero padding
        # both key and query preserves NoPE attention without changing scale.
        if self.qk_rope_head_dim == 0:
            k_pe = k_pe.new_zeros((*k_pe.shape[:-1], 64))
        super().do_kv_cache_update(
            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
        )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if self.qk_rope_head_dim == 0:
            q = torch.nn.functional.pad(q, (0, 64))

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        nonempty_rows = (topk_indices >= 0).any(dim=1)[:, None, None]

        topk_indices_physical = cast(
            torch.Tensor,
            triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token[:num_actual_toks],
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=topk_indices.shape[1],
            ),
        )

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        if topk_indices_physical.shape[1] > 2048:
            # SM120 decode dispatch stops at 2048 entries. GLM kpool also
            # appends a recent-token tail. Preserve it with a softmax merge
            # using the kernel's base-2 log-sum-exp, not a second normalization.
            partials, normalizers = [], []
            for chunk in topk_indices_physical.split(2048, dim=1):
                width = next(n for n in (128, 512, 1024, 2048) if n >= chunk.shape[1])
                if num_actual_toks > 64:
                    width = 2048
                chunk = torch.nn.functional.pad(
                    chunk, (0, width - chunk.shape[1]), value=-1
                ).contiguous()
                valid = (chunk >= 0).any(dim=1)[:, None]
                partial, lse = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
                    query=q.unsqueeze(1),
                    kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
                    workspace_buffer=self._workspace_buffer,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_rope_head_dim=64,
                    block_tables=chunk.unsqueeze(1),
                    seq_lens=None,
                    max_seq_len=width,
                    bmm1_scale=self.scale,
                    bmm2_scale=1.0,
                    sparse_mla_top_k=width,
                    kv_scale_format=self.kv_scale_format,
                    return_lse=True,
                )
                partials.append(torch.where(valid[..., None], partial.squeeze(1), 0))
                normalizers.append(
                    lse.reshape(num_actual_toks, self.num_heads).masked_fill(
                        ~valid, -torch.inf
                    )
                )
            weights = (torch.stack(normalizers) * math.log(2)).softmax(0).nan_to_num()
            output = (torch.stack(partials).float() * weights[..., None]).sum(0)
            return torch.where(nonempty_rows, output.to(q.dtype), 0), None

        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=64,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=topk_indices_physical.shape[1],
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=topk_indices_physical.shape[1],
            kv_scale_format=self.kv_scale_format,
        )
        return torch.where(nonempty_rows, out.squeeze(1), 0), None
