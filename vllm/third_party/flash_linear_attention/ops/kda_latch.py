# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental r=K KDA LatchSSM, with raw-write exact boundary updates.

Port of the *restored* GDN projected-f_s path (575c2a875f), not its earlier
state-dependent-write approximation. State layout is (slot, head, V, K).

For H=S0.T, a fixed full-column-rank Omega and metric ridge eta:
    U = H Omega
    Phi = (H.T U + eta Omega) (U.T U + eta Omega.T Omega)^-1
The inverse is folded into Phi, as in the GDN Phi implementation. r=K
removes coefficient truncation and anchors, but not the rank-G read error.

The decode kernel implements paper eq. wyread with Z replaced by Phi:
    f_t = beta_t [Phi.T(d_t*k_t) - sum_{s<t} f_s <ell_s(t),k_t>]
    c_t = Phi.T(d_t*q_t) - sum_{s<=t} f_s <ell_s(t),q_t>
The independent replay writes are
    u_t = beta_t [v_t - sum_{s<t} u_s <ell_s(t),k_t>].
Output is U c_t + sum_{s<=t} u_s <ell_s(t),q_t>.

The flush reads ONLY raw k/v/log-decay/beta, never u/f/Phi/U. This first
correctness implementation refreshes Phi/U using torch after the flush;
that epilogue rereads the state and is not a traffic-optimized flush.
This module is an opt-in experimental API, not wired into serving yet.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _step(
    Q,
    KIn,
    VIn,
    Gate,
    Beta,
    A,
    Bias,
    Slots,
    Pos,
    State,
    U,
    Phi,
    LatchHeads,
    KR,
    VR,
    GR,
    BR,
    PrefixR,
    FR,
    UR,
    Out,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    G: tl.constexpr,
    WG: tl.constexpr,
    W: tl.constexpr,
    SAFE: tl.constexpr,
    LOWER: tl.constexpr,
    SCALE: tl.constexpr,
    NORMALIZE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(Slots + row)
    pos = tl.load(Pos + slot)
    kh = tl.arange(0, K)
    vv = tl.arange(0, V)
    gg = tl.arange(0, WG)
    tt = tl.arange(0, W)
    q = tl.load(Q + (row * H + head) * K + kh).to(tl.float32)
    k = tl.load(KIn + (row * H + head) * K + kh).to(tl.float32)
    v = tl.load(VIn + (row * H + head) * V + vv).to(tl.float32)
    if NORMALIZE:
        q = q * tl.rsqrt(tl.sum(q * q) + 1e-6)
        k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
    q = q * SCALE
    raw_g = tl.load(Gate + (row * H + head) * K + kh).to(tl.float32)
    raw_g += tl.load(Bias + head * K + kh)
    amplitude = tl.exp(tl.load(A + head))
    if SAFE:
        log_a = LOWER / (1.0 + tl.exp(-amplitude * raw_g))
    else:
        log_a = -amplitude * tl.where(raw_g > 20.0, raw_g, tl.log(1.0 + tl.exp(raw_g)))
    beta = tl.sigmoid(tl.load(Beta + row * H + head).to(tl.float32))
    base = (slot * H + head) * W
    prev = tl.load(PrefixR + (base + pos - 1) * K + kh, mask=pos > 0, other=0.0)
    prefix = prev + log_a
    tl.store(KR + (base + pos) * K + kh, k)
    tl.store(VR + (base + pos) * V + vv, v)
    tl.store(GR + (base + pos) * K + kh, log_a)
    tl.store(BR + base + pos, beta)
    tl.store(PrefixR + (base + pos) * K + kh, prefix)
    is_latch = tl.load(LatchHeads + head)
    if is_latch:
        past_k = tl.load(
            KR + (base + tt[:, None]) * K + kh[None, :],
            mask=tt[:, None] < pos,
            other=0.0,
        )
        past_prefix = tl.load(
            PrefixR + (base + tt[:, None]) * K + kh[None, :],
            mask=tt[:, None] < pos,
            other=0.0,
        )
        # exp(prefix_t - prefix_s) avoids forming 1/d_s (overflow at W=16).
        ell = past_k * tl.exp(prefix[None, :] - past_prefix)
        kk = tl.sum(ell * k[None, :], axis=1)
        kq = tl.sum(ell * q[None, :], axis=1)
        phi = tl.load(
            Phi + (slot * H + head) * K * G + kh[:, None] * G + gg[None, :],
            mask=gg[None, :] < G,
            other=0.0,
        )
        dk = tl.exp(prefix) * k
        dq = tl.exp(prefix) * q
        fs = tl.load(
            FR + (base + tt[:, None]) * G + gg[None, :],
            mask=(tt[:, None] < pos) & (gg[None, :] < G),
            other=0.0,
        )
        f = beta * (
            tl.sum(phi * dk[:, None], axis=0) - tl.sum(fs * kk[:, None], axis=0)
        )
        cur_kq = tl.sum(k * q)
        c = (
            tl.sum(phi * dq[:, None], axis=0)
            - tl.sum(fs * kq[:, None], axis=0)
            - f * cur_kq
        )
        us = tl.load(
            UR + (base + tt[:, None]) * V + vv[None, :],
            mask=tt[:, None] < pos,
            other=0.0,
        )
        u = beta * (v - tl.sum(us * kk[:, None], axis=0))
        latch = tl.load(
            U + (slot * H + head) * V * G + vv[:, None] * G + gg[None, :],
            mask=gg[None, :] < G,
            other=0.0,
        )
        out = (
            tl.sum(latch * c[None, :], axis=1)
            + tl.sum(us * kq[:, None], axis=0)
            + u * cur_kq
        )
        tl.store(FR + (base + pos) * G + gg, f, mask=gg < G)
        tl.store(UR + (base + pos) * V + vv, u)
    else:
        # Existing dense recurrence; this head updates state on every step.
        sp = State + (slot * H + head) * V * K + vv[:, None] * K + kh[None, :]
        state = tl.load(sp) * tl.exp(log_a[None, :])
        delta = beta * (v - tl.sum(state * k[None, :], axis=1))
        state += delta[:, None] * k[None, :]
        out = tl.sum(state * q[None, :], axis=1)
        tl.store(sp, state)
    tl.store(Out + (row * H + head) * V + vv, out)


@triton.jit
def _flush(
    Slots,
    Pos,
    State,
    KR,
    VR,
    GR,
    BR,
    LatchHeads,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    BV: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)
    slot = tl.load(Slots + row)
    if (tl.load(Pos + slot) == W - 1) & tl.load(LatchHeads + head):
        kk = tl.arange(0, K)
        vv = block * BV + tl.arange(0, BV)
        sp = State + (slot * H + head) * V * K + vv[:, None] * K + kk[None, :]
        state = tl.load(sp, mask=vv[:, None] < V, other=0.0)
        base = (slot * H + head) * W
        for t in range(W):
            k = tl.load(KR + (base + t) * K + kk)
            v = tl.load(VR + (base + t) * V + vv, mask=vv < V, other=0.0)
            log_a = tl.load(GR + (base + t) * K + kk)
            beta = tl.load(BR + base + t)
            state *= tl.exp(log_a[None, :])
            delta = beta * (v - tl.sum(state * k[None, :], axis=1))
            state += delta[:, None] * k[None, :]
        tl.store(sp, state, mask=vv[:, None] < V)


class KDALatchState:
    """Eager single-token API for correctness and small-model experiments.

    Inputs are post-convolution q/k/v and raw gate/beta logits. q/k are
    normalized exactly as GLM by default. Explicit slot IDs allow shuffled,
    independently progressing sequences. Duplicate slot IDs are rejected.
    All heads share a rank G; a boolean per-head mask selects dense fallback.
    """

    @torch.no_grad()
    def __init__(
        self,
        state: torch.Tensor,
        omega: torch.Tensor,
        *,
        window: int = 16,
        ridge: float = 1e-4,
        latch_heads: torch.Tensor | None = None,
    ):
        if state.ndim != 4 or state.shape[-2:] != (128, 128):
            raise ValueError("Initial KDA port requires state (slots, heads, 128, 128)")
        if not state.is_cuda or state.dtype != torch.float32:
            raise ValueError("KDA latch requires CUDA FP32 persistent state")
        n, h, v, k = state.shape
        if omega.ndim != 3 or omega.shape[:2] != (h, k):
            raise ValueError("omega must have shape (heads, K, G)")
        if not 1 <= omega.shape[-1] <= k or window != 16 or ridge <= 0:
            raise ValueError("Require 1 <= G <= K, W=16 and positive metric ridge")
        if omega.device != state.device or not torch.isfinite(omega).all():
            raise ValueError("omega must be finite and on the state device")
        if (torch.linalg.matrix_rank(omega.double()) < omega.shape[-1]).any():
            raise ValueError("omega must have full column rank")
        self.state = state.clone().contiguous()
        self.omega = omega.double().contiguous()
        self.window, self.ridge = window, ridge
        self.rank = omega.shape[-1]
        self.pos = torch.zeros(n, device=state.device, dtype=torch.int32)
        if latch_heads is None:
            latch_heads = torch.ones(h, device=state.device, dtype=torch.bool)
        if latch_heads.shape != (h,) or latch_heads.device != state.device:
            raise ValueError("latch_heads must have shape (heads,) on the state device")
        self.latch_heads = latch_heads.bool().contiguous()
        opts = dict(device=state.device, dtype=torch.float32)
        self.k = torch.zeros(n, h, window, k, **opts)
        self.v = torch.zeros(n, h, window, v, **opts)
        self.log_a = torch.zeros_like(self.k)
        self.prefix = torch.zeros_like(self.k)
        self.beta = torch.zeros(n, h, window, **opts)
        self.f = torch.zeros(n, h, window, self.rank, **opts)
        self.u = torch.zeros_like(self.v)
        self.latch = torch.empty(n, h, v, self.rank, **opts)
        self.phi = torch.empty(n, h, k, self.rank, **opts)
        self.refresh(torch.arange(n, device=state.device))

    @torch.no_grad()
    def refresh(self, slots: torch.Tensor):
        """Metric-ridge epilogue; intentionally unoptimized, FP64 solve.

        Ridge belongs to H.T H, not just U.T U. For a general Omega this
        adds eta*Omega.T Omega to the Gram AND eta*Omega to the numerator.
        """
        h = self.state[slots].double()
        u = h @ self.omega
        eta = self.ridge * h.square().sum(dim=(-2, -1)) / h.shape[-1]
        eta = eta.clamp_min(1e-12)[..., None, None]
        numerator = h.transpose(-1, -2) @ u + eta * self.omega
        gram = u.transpose(-1, -2) @ u + eta * (
            self.omega.transpose(-1, -2) @ self.omega
        )
        phi = torch.linalg.solve(gram, numerator.transpose(-1, -2)).transpose(-1, -2)
        self.latch[slots] = u.float()
        self.phi[slots] = phi.float()

    @torch.no_grad()
    def reset(self, slots: torch.Tensor, state: torch.Tensor):
        """Seed from exact prefill state, also discarding an abandoned window."""
        self.state[slots] = state
        self.pos[slots] = 0
        self.refresh(slots)

    @torch.no_grad()
    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        a_log: torch.Tensor,
        bias: torch.Tensor,
        *,
        slots: torch.Tensor | None = None,
        safe_gate: bool = True,
        lower_bound: float = -5.0,
        normalize: bool = True,
    ) -> torch.Tensor:
        n, h, vdim, kdim = self.state.shape
        if slots is None:
            slots = torch.arange(n, device=q.device)
        if slots.ndim != 1 or slots.numel() == 0:
            raise ValueError("Require a nonempty one-dimensional slot list")
        if (
            (slots < 0).any()
            or (slots >= n).any()
            or slots.unique().numel() != slots.numel()
        ):
            raise ValueError("Slot IDs must be distinct and in bounds")
        shape = (slots.numel(), h, kdim)
        if any(x.shape != shape for x in (q, k, v, raw_gate)):
            raise ValueError(f"q/k/v/gate must have shape {shape}")
        if (
            raw_beta.shape != shape[:2]
            or a_log.numel() != h
            or bias.numel() != h * kdim
        ):
            raise ValueError("Invalid beta, A_log or bias shape")
        if safe_gate and (lower_bound >= 0 or lower_bound < -5):
            raise ValueError("Initial port supports GLM bounded log-decay in [-5, 0]")
        tensors = (q, k, v, raw_gate, raw_beta, a_log, bias, slots)
        if any(x.device != self.state.device for x in tensors):
            raise ValueError("All tensors must share the state device")
        q, k, v, raw_gate, raw_beta = (
            x.contiguous() for x in (q, k, v, raw_gate, raw_beta)
        )
        out = torch.empty_like(v)
        slots = slots.to(torch.int32).contiguous()
        a_log, bias = a_log.float().contiguous(), bias.float().contiguous()
        _step[(slots.numel(), h)](
            q,
            k,
            v,
            raw_gate,
            raw_beta,
            a_log,
            bias,
            slots,
            self.pos,
            self.state,
            self.latch,
            self.phi,
            self.latch_heads,
            self.k,
            self.v,
            self.log_a,
            self.beta,
            self.prefix,
            self.f,
            self.u,
            out,
            h,
            kdim,
            vdim,
            self.rank,
            triton.next_power_of_2(self.rank),
            self.window,
            safe_gate,
            lower_bound,
            kdim**-0.5,
            normalize,
            num_warps=8,
        )
        _flush[(slots.numel(), h, triton.cdiv(vdim, 8))](
            slots,
            self.pos,
            self.state,
            self.k,
            self.v,
            self.log_a,
            self.beta,
            self.latch_heads,
            h,
            kdim,
            vdim,
            self.window,
            8,
            num_warps=1,
        )
        flushed = slots[self.pos[slots] == self.window - 1]
        self.pos[slots] = (self.pos[slots] + 1) % self.window
        if flushed.numel():
            self.refresh(flushed)
        return out
