# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent FP64 reference: explicit metric solve and ordered KDA recurrence."""

import torch


def coefficient(state, frame, rank, pivots):
    if rank in (0, 128):
        return None
    s = state.double() @ frame.double()
    mu = s.square().sum() / 128
    s = s / torch.sqrt(mu if mu > 0 else torch.ones_like(mu))
    q: list[torch.Tensor] = []
    for i in range(min(rank, pivots)):
        u = s[:, i]
        v = u.clone()
        for _ in range(2):
            for b in q:
                v = v - b * (b @ v)
        keep = v.square().sum() > 1e-12 * u.square().sum()
        q.append(v / v.norm() if keep else torch.zeros_like(v))
    z = torch.stack(q) @ s
    residual = (s.square().sum(0) - z.square().sum(0)).clamp_min(0)
    residual[: min(rank, pivots)] = 0
    metric = z.T @ z + torch.diag(residual)
    # Dense GxG solve independent of production's P-dimensional Woodbury solve.
    a = torch.linalg.solve(
        metric[:rank, :rank] + 0.1 * torch.eye(rank, dtype=torch.float64),
        metric[:rank, :],
    )
    return frame.double() @ a.T


class Oracle:
    def __init__(self, state, frame, ranks, pivots):
        self.state = state.double().cpu().clone()
        self.frame = frame.double().cpu()
        self.ranks = ranks.cpu().tolist()
        self.pivots = pivots
        self.start = self.state.clone()
        n, h, _, _ = state.shape
        self.transition = (
            torch.eye(128, dtype=torch.float64).expand(n, h, 128, 128).clone()
        )
        self.replay = torch.zeros_like(self.state)
        self.pos = [0] * n
        self.maps = {}

    def step(self, ids, q, k, v, gate, beta, a_log, bias):
        q, k, v, gate, beta, a_log, bias = [
            x.double().cpu() for x in (q, k, v, gate, beta, a_log, bias)
        ]
        q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / 128**0.5
        k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = torch.exp(
            -5 * torch.sigmoid(a_log.exp()[None, :, None] * (gate + bias[None]))
        )
        beta = beta.sigmoid()
        out = torch.zeros_like(v)
        for b, s in enumerate(ids):
            if s <= 0:
                continue
            if self.pos[s] == 0:
                for h, m in enumerate(self.ranks):
                    self.maps[s, h] = coefficient(
                        self.start[s, h], self.frame[h], m, self.pivots
                    )
            for h, m in enumerate(self.ranks):
                kh = k[b, h]
                qh = q[b, h]
                bt = beta[b, h]
                for mat, write in (
                    (self.state, True),
                    (self.replay, True),
                    (self.transition, False),
                ):
                    x = mat[s, h] * decay[b, h][None, :]
                    delta = -(x @ kh) * bt
                    if write:
                        delta = delta + bt * v[b, h]
                    mat[s, h] = x + delta[:, None] * kh[None, :]
                if m in (0, 128) or self.pos[s] == 15:
                    out[b, h] = self.state[s, h] @ qh
                else:
                    effective = self.transition[s, h] @ qh
                    coef = self.maps[s, h].T @ effective
                    out[b, h] = (
                        self.start[s, h] @ self.frame[h, :, :m]
                    ) @ coef + self.replay[s, h] @ qh
            self.pos[s] += 1
            if self.pos[s] == 16:
                self.start[s] = self.state[s]
                self.transition[s] = torch.eye(128, dtype=torch.float64)
                self.replay[s].zero_()
                self.pos[s] = 0
        return out
