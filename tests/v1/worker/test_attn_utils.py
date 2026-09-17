# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded-page handling in create_kv_cache_views.

Guards that a page_size_padded spec strides the block dimension by the padded page
while keeping per-block content compact, so padding bytes at the end of each page are
never addressed by the logical view.
"""

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MLAAttentionSpec,
    compute_layout_strides,
    create_kv_cache_views,
)
from vllm.v1.worker.gpu.attn_utils import (
    get_attn_cg_support,
    get_query_lens_mismatch_unsupported_backend,
)
from vllm.v1.worker.utils import (
    AttentionGroup,
    allocate_kv_cache,
    copy_kv_cache_blocks_inplace,
)


@pytest.fixture(autouse=True)
def _allow_cpu_cache_tests_without_cuda(monkeypatch):
    # Host cache ownership/copy tests do not need CUDA-backed pinned memory.
    if not torch.accelerator.is_available():
        monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)


class _FakeMetadataBuilder:
    def __init__(self, support: AttentionCGSupport):
        self.support = support

    def get_cudagraph_support(self, *_args):
        return self.support


class _TargetBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return True


class _DraftBackend:
    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        return False


def test_attention_checks_preserve_global_and_target_scoped_support():
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    target_group = AttentionGroup(
        _TargetBackend,
        ["target"],
        spec,
        0,  # type: ignore[arg-type]
    )
    target_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.ALWAYS)  # type: ignore[list-item]
    ]
    draft_group = AttentionGroup(
        _DraftBackend,
        ["draft"],
        spec,
        0,  # type: ignore[arg-type]
    )
    draft_group.metadata_builders = [
        _FakeMetadataBuilder(AttentionCGSupport.UNIFORM_BATCH)  # type: ignore[list-item]
    ]
    groups = [[target_group, draft_group]]

    # The runner-wide execution mode must still honor the drafter's limit.
    unfiltered = get_attn_cg_support(groups, None)  # type: ignore[arg-type]
    assert unfiltered.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert unfiltered.min_cg_attn_backend == "_DraftBackend"

    # Adaptive verification validates only the target's varlen graphs.
    target_only = get_attn_cg_support(
        groups,
        None,  # type: ignore[arg-type]
        checked_layer_names={"target"},
    )
    assert target_only.min_cg_support == AttentionCGSupport.ALWAYS
    assert target_only.min_cg_attn_backend is None
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        is None
    )

    # Shared target/draft groups still participate in target-scoped checks.
    draft_group.layer_names.append("target")
    target_with_shared_group = get_attn_cg_support(
        groups,
        None,  # type: ignore[arg-type]
        checked_layer_names={"target"},
    )
    assert target_with_shared_group.min_cg_support == AttentionCGSupport.UNIFORM_BATCH
    assert (
        get_query_lens_mismatch_unsupported_backend(
            groups,
            checked_layer_names={"target"},
        )
        == "_DraftBackend"
    )


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = dense_kv_cache_views(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


@pytest.mark.parametrize(
    ("kernel_block_sizes", "expected_num_blocks", "expected_num_states"),
    [
        (None, 4, 64),
        ([256], 4, 64),
        ([64], 16, 16),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                layers=["layer.0"],
                layer_stride=num_pages * spec.page_size_bytes,
                block_stride=spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = allocate_kv_cache(
        config, torch.device("cpu"), KVCacheLayout.LBHNC, kernel_block_sizes
    )

    assert caches["layer.0"].shape == (expected_num_blocks, 1, expected_num_states, 128)


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    copy_kv_cache_blocks_inplace(caches, num_blocks, copies)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


def test_fixed_block_stride_propagates_outward_in_lhbnc():
    num_blocks = 3
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    natural = compute_layout_strides(spec, num_blocks, num_layers, KVCacheLayout.LHBNC)
    block_stride = natural[1] + 8

    strides = compute_layout_strides(
        spec,
        num_blocks,
        num_layers,
        KVCacheLayout.LHBNC,
        fixed_strides=(None, block_stride, None, None, None),
    )

    assert strides[1] == block_stride
    assert strides[2] == block_stride * num_blocks
    assert strides[0] == strides[2] * spec.num_heads


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC stores each head group separately, so a block's bytes are scattered
    # across L*H regions.
    layout = KVCacheLayout.LHBNC
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        num_head_slots=2,
        state_content_bytes=2 * 2 * 4,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize(
    "layout,num_layers",
    [
        (KVCacheLayout.LBHNC, 2),
        # Splitting needs a manager block to be one dense page, which a
        # block-outermost layout only gives when the block holds one layer.
        (KVCacheLayout.BLHNC, 1),
    ],
)
def test_copy_kv_cache_blocks_with_virtual_block_splitting(
    layout: KVCacheLayout, num_layers: int
):
    num_blocks = 4
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8)
    caches = dense_kv_cache_views(
        raw,
        spec,
        num_blocks,
        num_layers,
        layout,
        kernel_block_size=spec.block_size // physical_per_logical,
    )

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(cache.shape[0]):
            cache[block_idx].fill_(100 * layer_idx + block_idx)
    expected = [[cache[i].clone() for i in range(cache.shape[0])] for cache in caches]

    copy_kv_cache_blocks_inplace(
        caches,
        num_blocks,
        [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
    )

    dst_start = 2 * physical_per_logical
    for layer_idx, cache in enumerate(caches):
        for physical_idx in range(physical_per_logical):
            torch.testing.assert_close(
                cache[dst_start + physical_idx], expected[layer_idx][physical_idx]
            )


def _compact_cache_views(block_size, device="cpu", kernel=64):
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=128, dtype=torch.float8_e4m3fn
    )
    stride = 8 * spec.page_size_bytes
    raw = torch.full((4 * stride,), 23, dtype=torch.int8, device=device)
    tensor = KVCacheTensor(
        size=raw.numel(),
        layers=[str(i) for i in range(8)],
        layer_stride=spec.page_size_bytes,
        block_stride=stride,
    )
    return raw, create_kv_cache_views(raw, spec, 4, KVCacheLayout.BLHNC, tensor, kernel)


@pytest.mark.parametrize("block_size,kernel", [(2240, 16), (2176, 128)])
def test_compact_manager_block_ownership_and_layer_disjointness(
    block_size, kernel, monkeypatch
):
    monkeypatch.setenv("VLLM_NEMOTRON_COMPACT_KV_CACHE_BLOCK_SIZE", "2112")
    raw, caches = _compact_cache_views(block_size, kernel=kernel)
    ratio = block_size // kernel
    tile = 8 * block_size * 512
    for layer, cache in enumerate(caches):
        # Write all virtual pages of manager block 2; adjacent manager blocks
        # stand in for live Mamba states and must not be touched.
        cache[2 * ratio : 3 * ratio].view(torch.int8).fill_(layer + 1)
    assert torch.all(raw[: 2 * tile] == 23)
    assert torch.all(raw[3 * tile :] == 23)
    independent = raw[2 * tile : 3 * tile].reshape(ratio, 8, 2, kernel, 256)
    for layer in range(8):
        assert torch.all(independent[:, layer] == layer + 1)
        assert torch.all(
            caches[layer][2 * ratio : 3 * ratio].view(torch.int8) == layer + 1
        )


@pytest.mark.parametrize("block_size,kernel", [(2240, 16), (2176, 128)])
def test_compact_flashinfer_reads_scattered_fp8_kv_without_cross_layer_corruption(
    block_size, kernel, monkeypatch
):
    monkeypatch.setenv("VLLM_NEMOTRON_COMPACT_KV_CACHE_BLOCK_SIZE", "2112")
    import flashinfer

    import vllm._custom_ops  # noqa: F401

    torch.manual_seed(22)
    raw, caches = _compact_cache_views(block_size, "cuda", kernel)
    ratio = block_size // kernel
    # Cross both kernel-page and manager-page boundaries; physical block order
    # differs from request order to exercise real block-table indirection.
    length = block_size + 3
    physical_blocks = [2, 1]
    virtual = [b * ratio + k for b in physical_blocks for k in range(ratio)]
    virtual = virtual[: (length + kernel - 1) // kernel]
    slots = torch.tensor(
        [virtual[t // kernel] * kernel + t % kernel for t in range(length)],
        dtype=torch.int64,
        device="cuda",
    )
    scale = torch.tensor(1.0, device="cuda")
    expected = []
    for layer, cache in enumerate(caches):
        k = torch.randn(length, 2, 128, dtype=torch.bfloat16, device="cuda") * 0.3
        v = torch.randn_like(k) * 0.3
        k_cache, v_cache = cache.transpose(1, 2).split(128, dim=-1)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            k, v, k_cache, v_cache, slots, "fp8", scale, scale
        )
        expected.append(
            (
                k.to(torch.float8_e4m3fn).to(torch.bfloat16),
                v.to(torch.float8_e4m3fn).to(torch.bfloat16),
            )
        )
    torch.accelerator.synchronize()
    # Every layer must survive writes to all other layers.
    indices = torch.tensor(virtual, dtype=torch.int32, device="cuda")
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="HND", use_tensor_cores=True
    )
    wrapper.plan(
        torch.tensor([0, len(virtual)], dtype=torch.int32, device="cuda"),
        indices,
        torch.tensor([(length - 1) % kernel + 1], dtype=torch.int32, device="cuda"),
        32,
        2,
        128,
        kernel,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
    )
    q = torch.randn(1, 32, 128, dtype=torch.bfloat16, device="cuda")
    for cache, (k, v) in zip(caches, expected):
        actual_k = cache[slots // kernel, :, slots % kernel, :128].to(torch.bfloat16)
        actual_v = cache[slots // kernel, :, slots % kernel, 128:].to(torch.bfloat16)
        torch.testing.assert_close(actual_k, k, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, v, rtol=0, atol=0)
        ck, cv = cache.split(128, dim=-1)
        actual = wrapper.run(q, (ck, cv))
        # Same kernel and values, separate dense pages: isolates stride handling
        # from the kernel's normal rounding error.
        dense = wrapper.run(q, (ck.contiguous(), cv.contiguous()))
        torch.testing.assert_close(actual, dense, rtol=0, atol=0)
        trt_args = dict(
            query=q.to(torch.float8_e4m3fn),
            workspace_buffer=workspace,
            block_tables=indices.unsqueeze(0),
            seq_lens=torch.tensor([length], dtype=torch.int32, device="cuda"),
            max_seq_len=length,
            bmm1_scale=128**-0.5,
            bmm2_scale=1.0,
            out_dtype=torch.bfloat16,
            kv_layout="HND",
            backend="trtllm-gen",
        )
        trt = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            kv_cache=(ck, cv), **trt_args
        )
        trt_dense = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            kv_cache=(ck.contiguous(), cv.contiguous()), **trt_args
        )
        torch.testing.assert_close(trt, trt_dense, rtol=0, atol=0)
        ref = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0).float(),
                k.transpose(0, 1).repeat_interleave(16, dim=0).unsqueeze(0).float(),
                v.transpose(0, 1).repeat_interleave(16, dim=0).unsqueeze(0).float(),
            )
            .squeeze(0)
            .transpose(0, 1)
            .to(torch.bfloat16)
        )
        torch.testing.assert_close(actual, ref, rtol=0.03, atol=0.002)
    tile = 8 * block_size * 512
    assert torch.all(raw[:tile] == 23)
    assert torch.all(raw[3 * tile :] == 23)


@pytest.mark.parametrize("block_size,state_bytes", [(2112, 4255744), (2240, 4558848)])
def test_compact_allocator_preserves_recurrent_state_when_copying_attention_blocks(
    block_size, state_bytes, monkeypatch
):
    monkeypatch.setenv("VLLM_NEMOTRON_COMPACT_KV_CACHE_BLOCK_SIZE", "2112")
    from unittest.mock import MagicMock

    from vllm.config import CacheConfig
    from vllm.v1.core.kv_cache_utils import (
        KVCacheBlockCopy,
        _get_compact_nemotron_groups,
        get_kv_cache_config_from_groups,
    )
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.worker.utils import allocate_kv_cache, copy_kv_cache_blocks_inplace

    config = MagicMock()
    config.cache_config = CacheConfig()
    config.cache_config.num_gpu_blocks_override = None
    config.cache_config.kv_cache_layout = "BLHNC"
    state = MambaSpec(
        block_size=block_size, shapes=((state_bytes,),), dtypes=(torch.uint8,)
    )
    attention = FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=128, dtype=torch.float8_e4m3fn
    )
    specs = {f"s{i}": state for i in range(40)}
    specs.update({f"a{i}": attention for i in range(8)})
    groups = _get_compact_nemotron_groups(specs)
    assert [len(g.layer_names) for g in groups] == [2] * 20 + [8]
    tile = 8 * block_size * 512
    allocation = get_kv_cache_config_from_groups(config, groups, 4 * tile)
    assert allocation.num_blocks == 4
    caches = allocate_kv_cache(
        allocation, torch.device("cpu"), KVCacheLayout.BLHNC, [block_size] * 20 + [64]
    )
    caches["s0"][1].fill_(31)
    caches["s1"][1].fill_(37)
    ratio = block_size // 64
    attention_caches = [caches[f"a{i}"] for i in range(8)]
    for i, c in enumerate(attention_caches):
        c[2 * ratio : 3 * ratio].view(torch.int8).fill_(i + 1)
    copy_kv_cache_blocks_inplace(
        attention_caches, 4, [KVCacheBlockCopy(src_block_id=2, dst_block_id=3)]
    )
    for i, c in enumerate(attention_caches):
        assert torch.all(c[3 * ratio :].view(torch.int8) == i + 1)
    assert torch.all(caches["s0"][1] == 31)
    assert torch.all(caches["s1"][1] == 37)
