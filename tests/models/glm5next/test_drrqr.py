# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from vllm.models.glm5next.drrqr import (
    apply_glm5_drrqr_mask,
    build_glm5_drrqr_mask,
)


def _write_indices(tmp_path):
    ratio_dir = tmp_path / "ratios" / "sparsity-0.500"
    ratio_dir.mkdir(parents=True)
    prune = torch.tensor(
        [
            [4, 5, 6, 7],
            [0, 1, 2, 3],
            [1, 3, 5, 7],
            [0, 2, 4, 6],
        ]
    )
    payload = {
        "metadata": {
            "linear_attention_family": "glm5_kda",
            "pruning_ratio": 0.5,
            "num_key_heads": 4,
            "head_dim": 8,
            "kept_head_dim": 4,
        },
        "layers": {2: {"prune_local_sorted": prune}},
    }
    torch.save(payload, ratio_dir / "indices.pt")
    manifest = {
        "ratios": {
            "0.500": {
                "indices_pt": "ratios/sparsity-0.500/indices.pt",
            }
        }
    }
    (tmp_path / "indices-manifest.json").write_text(json.dumps(manifest))


def test_build_glm5_drrqr_mask_selects_local_tp_heads(tmp_path):
    _write_indices(tmp_path)

    mask, kept_head_dim, path = build_glm5_drrqr_mask(
        str(tmp_path),
        sparsity=0.5,
        layer_idx=2,
        num_heads=4,
        head_dim=8,
        tp_rank=1,
        tp_size=2,
    )

    assert path.name == "indices.pt"
    assert kept_head_dim == 4
    assert mask.tolist() == [
        [True, False, True, False, True, False, True, False],
        [False, True, False, True, False, True, False, True],
    ]


def test_build_glm5_drrqr_mask_rejects_unavailable_ratio(tmp_path):
    _write_indices(tmp_path)

    with pytest.raises(ValueError, match="available"):
        build_glm5_drrqr_mask(
            str(tmp_path),
            sparsity=0.625,
            layer_idx=2,
            num_heads=4,
            head_dim=8,
            tp_rank=0,
            tp_size=2,
        )


def test_build_glm5_drrqr_mask_supports_reserved_875_ratio(tmp_path):
    ratio_dir = tmp_path / "ratios" / "sparsity-0.875"
    ratio_dir.mkdir(parents=True)
    prune = torch.arange(16, 128).expand(64, -1).clone()
    torch.save(
        {
            "metadata": {
                "linear_attention_family": "glm5_kda",
                "pruning_ratio": 0.875,
                "num_key_heads": 64,
                "head_dim": 128,
                "kept_head_dim": 16,
            },
            "layers": {44: {"prune_local_sorted": prune}},
        },
        ratio_dir / "indices.pt",
    )
    (tmp_path / "indices-manifest.json").write_text(
        json.dumps(
            {
                "ratios": {
                    "0.875": {
                        "indices_pt": (
                            "ratios/sparsity-0.875/indices.pt"
                        ),
                    }
                }
            }
        )
    )

    mask, kept_head_dim, _ = build_glm5_drrqr_mask(
        str(tmp_path),
        sparsity=0.875,
        layer_idx=44,
        num_heads=64,
        head_dim=128,
        tp_rank=1,
        tp_size=2,
    )

    assert mask.shape == (32, 128)
    assert kept_head_dim == 16
    assert bool(mask[:, :16].all())
    assert not bool(mask[:, 16:].any())


def test_apply_glm5_drrqr_mask_is_in_place():
    mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    query = torch.arange(16, dtype=torch.float32).view(2, 8)
    key = (query + 100).clone()
    query_ptr = query.data_ptr()
    key_ptr = key.data_ptr()

    query, key = apply_glm5_drrqr_mask(
        query,
        key,
        mask,
        local_num_heads=2,
        head_dim=4,
    )

    assert query.data_ptr() == query_ptr
    assert key.data_ptr() == key_ptr
    assert not bool(query.view(2, 2, 4)[:, 0, 1::2].any())
    assert not bool(query.view(2, 2, 4)[:, 1, 0::2].any())
    assert not bool(key.view(2, 2, 4)[:, 0, 1::2].any())
    assert not bool(key.view(2, 2, 4)[:, 1, 0::2].any())
