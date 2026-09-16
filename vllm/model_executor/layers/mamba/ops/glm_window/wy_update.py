# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered KDA window update in native coordinates, with per-channel decay.

P_i = product_{t<=i} alpha_t; a_i=P_i*k_i; c_i=k_i/P_i.
L_ij=beta_i*(a_i dot c_j), j<i. The triangular solve yields
Delta_i=beta_i*(v_i-S0*a_i)-sum_{j<i} L_ij*Delta_j.
S_W=(S0+Delta*C)*P_W. This is not the scalar-gate GDN shortcut.
"""

from vllm.triton_utils import tl, triton


@triton.jit
def wy_update(raw, KR, VR, GR, BR, slot, h, H: tl.constexpr):
    t = tl.arange(0, 16)
    k = tl.arange(0, 128)
    v = tl.arange(0, 128)
    base = (slot * H + h) * 16
    keys = tl.load(KR + (base + t[:, None]) * 128 + k[None, :]).to(tl.float32)
    keys *= tl.rsqrt(tl.sum(keys * keys, axis=1) + 1.0e-6)[:, None]
    values = tl.load(VR + (base + t[:, None]) * 128 + v[None, :]).to(tl.float32)
    logs = tl.load(GR + (base + t[:, None]) * 128 + k[None, :])
    beta = tl.load(BR + base + t)
    prefix_mask = (t[:, None] >= t[None, :]).to(tl.float32)
    log_prefix = tl.dot(prefix_mask, logs, input_precision="ieee")
    a = keys * tl.exp(log_prefix)
    c = keys * tl.exp(-log_prefix)
    lower = tl.dot(a, tl.trans(c), input_precision="ieee") * beta[:, None]
    lower = tl.where(t[:, None] > t[None, :], lower, 0.0)
    rhs = (
        tl.trans(values) - tl.dot(raw, tl.trans(a), input_precision="tf32x3")
    ) * beta[None, :]
    for i in tl.static_range(16):
        column = tl.sum(tl.where(t[None, :] == i, rhs, 0.0), axis=1)
        factors = tl.sum(tl.where(t[None, :] == i, lower, 0.0), axis=1)
        rhs -= column[:, None] * factors[None, :]
    result = raw + tl.dot(rhs, c, input_precision="tf32x3")
    final_prefix = tl.exp(tl.sum(tl.where(t[:, None] == 15, log_prefix, 0.0), axis=0))
    return result * final_prefix[None, :]
