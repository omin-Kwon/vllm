# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch-config selection for the ReplaySSM Mamba2 output_only decode kernel.

Mirrors ``mamba_ssm.py``: a hard-coded heuristic per kernel, plus an
``override`` context manager for benchmarks/tests/config sweeps. Hardware is
auto-detected (Blackwell vs not) so call sites need not thread it through.
"""

import functools
from contextlib import contextmanager

from vllm.platforms import current_platform
from vllm.triton_utils import triton


@functools.cache
def _is_blackwell() -> bool:
    try:
        return current_platform.is_device_capability_family(100)
    except Exception:
        return False


# Per-kernel overrides keyed by the kernel name passed to get_replayssm_config.
_overrides: dict[str, tuple] = {}


@contextmanager
def override_replayssm_config(kernel: str, config: tuple):
    """Pin ``kernel``'s launch config for the duration of the context."""
    prev = _overrides.get(kernel)
    _overrides[kernel] = config
    try:
        yield
    finally:
        if prev is None:
            _overrides.pop(kernel, None)
        else:
            _overrides[kernel] = prev


def _dstate_tile(dstate: int, tile: int) -> int:
    return max(16, min(tile, triton.next_power_of_2(dstate)))


def _mamba2_output_only(dstate, L, is_blackwell):
    # (block_size_m, num_warps, nf_dstate_tile, fl_dstate_tile, num_stages);
    # decoupled dstate tiling, serving-batch optimum, dtype-independent per device.
    if is_blackwell:
        return 64, 1, _dstate_tile(dstate, 32), _dstate_tile(dstate, 64), 2
    return 16, 1, _dstate_tile(dstate, 64), _dstate_tile(dstate, 128), 2


def _l_bucket(cache_len: int) -> int:
    """Bucket the ring length: L<=8 -> 8, 8<L<=16 -> 16, L>16 -> 32."""
    if cache_len <= 8:
        return 8
    if cache_len <= 16:
        return 16
    return 32


# (block_v, num_warps, num_stages, nk) for the GDN ReplaySSM decode kernel.
# Measured on the deployment shape (bs=256, Qwen3.8 geometry): BV64/nk4/w1/s2.
_GDN_DECODE_BY_L = {8: (64, 1, 2, 4), 16: (64, 1, 2, 4), 32: (64, 1, 2, 4)}


def get_replayssm_config(kernel: str, **shape) -> tuple:
    """Return the launch config for ``kernel`` (override > tuned default).

    kernel: "mamba2_output_only". ``shape`` carries the keying dims (dstate;
    ``L`` for the buffer length, default 16); hardware is auto-detected.
    """
    if kernel in _overrides:
        return _overrides[kernel]
    bw = _is_blackwell()
    if kernel == "mamba2_output_only":
        return _mamba2_output_only(shape["dstate"], shape.get("L", 16), bw)
    # ⚠ GDN port: add **only this key**. Wholesale-copying the fork's version of
    # this file breaks Mamba2 — its mamba2_output_only returns a different arity
    # (callers unpack 5, the fork returns 2) and decode dies with
    # "not enough values to unpack". See scale/PORT_GDN_REPLAY.md.
    if kernel == "gdn_decode":
        return _GDN_DECODE_BY_L[_l_bucket(shape.get("L", 16))]
    raise ValueError(f"unknown ReplaySSM kernel config key: {kernel}")
