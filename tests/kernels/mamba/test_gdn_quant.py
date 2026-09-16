# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSQ configuration and selected-state updates, including graph replay."""

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.gdn_quant import (
    bits_from_env,
    fake_quant_dsq,
    quantize_slots_,
)


@pytest.mark.parametrize("bits", [4, 6, 8, 10])
def test_dsq_preserves_zero_state(bits):
    state = torch.zeros(2, 3, 5)
    torch.testing.assert_close(fake_quant_dsq(state, bits), state, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("NS_GDN_QBITS", "3", "unsupported"),
        ("NS_GDN_QBITS", "x", "integer"),
        ("NS_GDN_QGRAN", "per_tensor", "DSQ only"),
        ("NS_GDN_QSR", "1", "round-to-nearest"),
    ],
)
def test_dsq_rejects_unsupported_settings(monkeypatch, name, value, error):
    monkeypatch.setenv("NS_GDN_QBITS", "4")
    monkeypatch.setenv("NS_GDN_QGRAN", "dsq_qm")
    monkeypatch.setenv("NS_GDN_QSR", "0")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=error):
        bits_from_env()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA DSQ kernel")
@pytest.mark.parametrize("bits", [4, 6, 8, 10])
@pytest.mark.parametrize("shape", [(2, 7, 13), (2, 128, 128)])
def test_dsq_updates_only_selected_slots(bits, shape):
    torch.manual_seed(42)
    # A view with a non-dense slot stride exercises the actual cache layout.
    backing = torch.randn(12, *shape, device="cuda", dtype=torch.float32)
    state = backing[::2]
    original = backing.clone()
    slots = torch.tensor([3, 0, 1, -1, 6], device="cuda", dtype=torch.int32)
    mask = torch.tensor([True, False, True, True, True], device="cuda")
    quantize_slots_(state, slots, bits, mask)
    expected = original.clone()
    for slot in (1, 3):
        expected[2 * slot] = fake_quant_dsq(original[2 * slot], bits)
    torch.testing.assert_close(backing, expected, rtol=1e-5, atol=1e-6)
    for slot in (0, 2, 4, 5):
        torch.testing.assert_close(state[slot], original[2 * slot], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
def test_dsq_graph_replay_reads_current_mask():
    torch.manual_seed(42)
    initial = torch.randn(4, 2, 128, 128, device="cuda")
    state = initial.clone()
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    mask = torch.zeros(2, device="cuda", dtype=torch.bool)
    quantize_slots_(state, slots, 8, mask)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantize_slots_(state, slots, 8, mask)
    mask[0] = True
    graph.replay()
    expected = initial.clone()
    expected[1] = fake_quant_dsq(initial[1], 8)
    torch.testing.assert_close(state, expected, rtol=1e-5, atol=1e-6)
