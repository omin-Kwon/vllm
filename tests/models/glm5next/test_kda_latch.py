# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paper audit of the r=128 KDA port, against independent dense transitions.

Run directly (unittest) to avoid unrelated engine fixtures, or with pytest.
The oracle explicitly multiplies KxK transitions and uses augmented lstsq;
it never uses the kernel's f_s/u_s recurrences or Phi normal-equation solve.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda
from vllm.third_party.flash_linear_attention.ops.kda_latch import KDALatchState


def inputs(batch=2, heads=2, steps=49, seed=72):
    torch.manual_seed(seed)
    shape = (steps, batch, heads, 128)
    base = torch.randn(batch, heads, 128, device="cuda")
    # Persistent, correlated keys/queries make erase corrections substantial.
    k = base + 0.35 * torch.randn(*shape, device="cuda")
    q = base + 0.5 * torch.randn(*shape, device="cuda")
    v = torch.randn(*shape, device="cuda")
    gate = -3 + torch.randn(*shape, device="cuda")
    beta = torch.randn(steps, batch, heads, device="cuda") + 1
    a = torch.zeros(heads, device="cuda")
    bias = torch.linspace(-0.4, 0.4, 128, device="cuda").expand(heads, -1).clone()
    return [x.bfloat16() for x in (q, k, v, gate, beta)] + [a, bias]


def normalize(q, k, gate, beta, a, bias, safe=True):
    q, k, gate, beta = (x.double() for x in (q, k, gate, beta))
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() / 128**0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    amp = a.double().exp()[None, :, None]
    z = gate + bias.double()
    log_a = -5 * (amp * z).sigmoid() if safe else -amp * torch.nn.functional.softplus(z)
    return q, k, log_a.exp(), beta.sigmoid()


class PaperOracle:
    def __init__(self, state, omega, ridge=1e-4, latch_heads=None):
        self.state = state.double().clone()
        self.start = self.state.clone()
        self.omega = omega.double()
        self.ridge = ridge
        n, h, v, k = state.shape
        self.eye = torch.eye(k, dtype=torch.float64, device=state.device)
        self.product = self.eye.expand(n, h, k, k).clone()
        self.replay = torch.zeros_like(self.state)
        self.pos = [0] * n
        self.latch_heads = [True] * h if latch_heads is None else latch_heads.tolist()

    def step(self, q, k, v, decay, beta, slots=None):
        if slots is None:
            slots = list(range(len(self.pos)))
        outs = []
        v = v.double()
        for row, slot in enumerate(slots):
            erase = self.eye - beta[row, :, None, None] * (
                k[row, :, :, None] * k[row, :, None, :]
            )
            transition = erase * decay[row, :, None, :]
            write = beta[row, :, None, None] * v[row, :, :, None] * k[row, :, None, :]
            self.state[slot] = self.state[slot] @ transition.transpose(-1, -2) + write
            self.product[slot] = transition @ self.product[slot]
            self.replay[slot] = self.replay[slot] @ transition.transpose(-1, -2) + write
            effective = (self.product[slot].transpose(-1, -2) @ q[row, :, :, None])[
                ..., 0
            ]
            heads = []
            for head, use_latch in enumerate(self.latch_heads):
                if not use_latch:
                    heads.append(self.state[slot, head] @ q[row, head])
                    continue
                h0, om = self.start[slot, head], self.omega[head]
                eta = max(self.ridge * h0.square().sum().item() / 128, 1e-12)
                u = h0 @ om
                # Independent metric-ridge least squares, not Phi @ effective.
                mat = torch.cat([u, eta**0.5 * om])
                target = torch.cat([h0 @ effective[head], eta**0.5 * effective[head]])
                c = torch.linalg.lstsq(mat, target[:, None]).solution[:, 0]
                heads.append(u @ c + self.replay[slot, head] @ q[row, head])
            outs.append(torch.stack(heads))
            self.pos[slot] += 1
            if self.pos[slot] == 16:
                self.start[slot] = self.state[slot]
                self.product[slot] = self.eye
                self.replay[slot].zero_()
                self.pos[slot] = 0
        return torch.stack(outs)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestKDALatch(unittest.TestCase):
    measurements: dict[str, float] = {}

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False

    def case(self, rank, heads=2, slots=2):
        torch.manual_seed(19)
        state = torch.randn(slots, heads, 128, 128, device="cuda") * 0.3
        om = torch.linalg.qr(
            torch.randn(heads, 128, rank, device="cuda", dtype=torch.float64)
        )[0]
        return state, om

    def assert_error(self, got, ref, atol=8e-4, rtol=6e-3):
        torch.testing.assert_close(got.double(), ref.double(), atol=atol, rtol=rtol)
        self.assertTrue(torch.isfinite(got).all())
        key = self._testMethodName
        delta = (got.double() - ref.double()).abs().max().item()
        self.measurements[key] = max(self.measurements.get(key, 0), delta)

    def test_stored_fs_equals_projection_of_full_wy_factor(self):
        state, om = self.case(8, slots=1)
        latch = KDALatchState(state, om)
        q, k, v, g, b, a, bias = inputs(batch=1, steps=16)
        product = (
            torch.eye(128, device="cuda", dtype=torch.float64)
            .expand(1, 2, 128, 128)
            .clone()
        )
        for t in range(16):
            _, kn, decay, beta = normalize(q[t], k[t], g[t], b[t], a, bias)
            # pi_t = beta_t M_(1:t-1).T D_t k_t, directly from the transition.
            pi = (
                beta[..., None]
                * (product.transpose(-1, -2) @ (decay * kn)[..., None])[..., 0]
            )
            expected = (latch.phi.double().transpose(-1, -2) @ pi[..., None])[..., 0]
            latch.step(q[t], k[t], v[t], g[t], b[t], a, bias)
            self.assert_error(latch.f[:, :, t], expected, atol=2e-6, rtol=2e-4)
            erase = torch.eye(128, device="cuda", dtype=torch.float64) - (
                beta[..., None, None] * kn[..., :, None] * kn[..., None, :]
            )
            product = (erase * decay[..., None, :]) @ product

    def test_full_basis_matches_existing_glm_dense_over_flushes(self):
        state, om = self.case(128)
        latch = KDALatchState(state, om)
        # vLLM reserves physical cache slot 0 as NULL_BLOCK_ID.
        dense = torch.cat([torch.zeros_like(state[:1]), state])
        q, k, v, g, b, a, bias = inputs()
        for t in range(len(q)):
            got = latch.step(q[t], k[t], v[t], g[t], b[t], a, bias)
            ref, _ = fused_recurrent_kda(
                q=q[t, :, None],
                k=k[t, :, None],
                v=v[t, :, None],
                g=g[t, :, None],
                beta=b[t, :, None],
                initial_state=dense,
                ssm_state_indices=torch.arange(1, 3, device="cuda", dtype=torch.int32),
                use_qk_l2norm_in_kernel=True,
                sigmoid_beta=True,
                a_log=a,
                g_bias=bias,
                compute_gate=True,
                lower_bound=-5.0,
            )
            self.assert_error(got, ref[:, 0])
            if (t + 1) % 16 == 0:
                self.assert_error(latch.state, dense[1:], atol=3e-5, rtol=2e-4)

    def test_low_rank_matches_paper_not_dense_and_keeps_exact_boundaries(self):
        for rank in (5, 8, 32):
            with self.subTest(rank=rank):
                state, om = self.case(rank)
                # Nonorthogonal basis catches incorrect Gram-only ridge.
                om = om * torch.linspace(0.6, 1.8, rank, device="cuda")
                flags = torch.tensor([True, False], device="cuda")
                latch = KDALatchState(state, om, latch_heads=flags)
                oracle = PaperOracle(state, om, latch_heads=flags)
                q, k, v, g, b, a, bias = inputs(steps=33)
                for t in range(len(q)):
                    qn, kn, decay, beta = normalize(q[t], k[t], g[t], b[t], a, bias)
                    ref = oracle.step(qn, kn, v[t], decay, beta)
                    got = latch.step(q[t], k[t], v[t], g[t], b[t], a, bias)
                    self.assert_error(got, ref)
                    if (t + 1) % 16 == 0:
                        self.assert_error(
                            latch.state, oracle.state, atol=3e-5, rtol=2e-4
                        )
                    elif t == 0:
                        self.assertTrue(torch.equal(latch.state[:, 0], state[:, 0]))

    def test_fs_omission_negative_control(self):
        state, om = self.case(8)
        good, mutant = KDALatchState(state, om), KDALatchState(state, om)
        q, k, v, g, b, a, bias = inputs(steps=3)
        for t in range(3):
            expected = good.step(q[t], k[t], v[t], g[t], b[t], a, bias)
            mutant.f.zero_()  # Deliberately remove previous projected erase factors.
            wrong = mutant.step(q[t], k[t], v[t], g[t], b[t], a, bias)
        error = (wrong.float() - expected.float()).abs().max().item()
        self.assertGreater(error, 2e-3, f"fs omission was invisible: {error}")
        self.assertTrue(good.f.abs().max() > 1e-3)
        self.measurements["fs_omission_output_max_abs"] = error

    def test_nonflush_uses_latch_instead_of_hidden_dense_read(self):
        state, om = self.case(8)
        good = KDALatchState(state, om)
        changed_state = KDALatchState(state, om)
        empty_latch = KDALatchState(state, om)
        # Preserve compact metadata and vary only the checkpoint after refresh.
        changed_state.state.fill_(19)
        empty_latch.latch.zero_()
        q, k, v, g, b, a, bias = inputs(steps=1)
        args = (q[0], k[0], v[0], g[0], b[0], a, bias)
        expected = good.step(*args)
        self.assertTrue(torch.equal(expected, changed_state.step(*args)))
        missing = empty_latch.step(*args)
        error = (missing.float() - expected.float()).abs().max().item()
        self.assertGreater(error, 1e-3)
        self.measurements["missing_latch_output_max_abs"] = error

    def test_flush_is_independent_of_all_approximate_metadata(self):
        state, om = self.case(8)
        clean, poisoned = KDALatchState(state, om), KDALatchState(state, om)
        q, k, v, g, b, a, bias = inputs(steps=16)
        for t in range(16):
            if t == 15:
                poisoned.f.fill_(3)
                poisoned.u.fill_(-7)
                poisoned.latch.fill_(11)
                poisoned.phi.fill_(-5)
            clean.step(q[t], k[t], v[t], g[t], b[t], a, bias)
            poisoned.step(q[t], k[t], v[t], g[t], b[t], a, bias)
        self.assertTrue(torch.equal(clean.state, poisoned.state))

    def test_asynchronous_slots_reset_and_reorder(self):
        state, om = self.case(8, slots=3)
        latch = KDALatchState(state, om)
        oracle = PaperOracle(state, om)
        q, k, v, g, b, a, bias = inputs(batch=3, steps=40)
        for t in range(40):
            ids = [2, 0] if t % 3 else [1, 2, 0]
            slots = torch.tensor(ids, device="cuda")
            qn, kn, decay, beta = normalize(
                q[t, ids], k[t, ids], g[t, ids], b[t, ids], a, bias
            )
            ref = oracle.step(qn, kn, v[t, ids], decay, beta, ids)
            got = latch.step(
                q[t, ids],
                k[t, ids],
                v[t, ids],
                g[t, ids],
                b[t, ids],
                a,
                bias,
                slots=slots,
            )
            self.assert_error(got, ref)
            for slot in ids:
                if oracle.pos[slot] == 0:
                    self.assert_error(
                        latch.state[slot], oracle.state[slot], atol=3e-5, rtol=2e-4
                    )
        ids = torch.tensor([2], device="cuda")
        latch.reset(ids, state[ids])
        fresh = KDALatchState(state[ids], om)
        got = latch.step(
            q[0, ids], k[0, ids], v[0, ids], g[0, ids], b[0, ids], a, bias, slots=ids
        )
        ref = fresh.step(q[0, ids], k[0, ids], v[0, ids], g[0, ids], b[0, ids], a, bias)
        self.assertTrue(torch.equal(got, ref))

    def test_extreme_channel_decay_stays_finite(self):
        state, om = self.case(128, slots=1)
        latch = KDALatchState(state, om)
        oracle = PaperOracle(state, om)
        q, k, v, g, b, a, bias = inputs(batch=1, steps=17)
        g[..., ::2] = 100  # cumulative log decay -80 at W=16
        g[..., 1::2] = -100
        for t in range(17):
            qn, kn, decay, beta = normalize(q[t], k[t], g[t], b[t], a, bias)
            ref = oracle.step(qn, kn, v[t], decay, beta)
            got = latch.step(q[t], k[t], v[t], g[t], b[t], a, bias)
            self.assert_error(got, ref)

    def test_softplus_gate_preserves_small_model_recurrence(self):
        state, om = self.case(8, slots=1)
        latch = KDALatchState(state, om)
        oracle = PaperOracle(state, om)
        q, k, v, g, b, a, bias = inputs(batch=1, steps=17)
        for t in range(17):
            qn, kn, decay, beta = normalize(q[t], k[t], g[t], b[t], a, bias, safe=False)
            ref = oracle.step(qn, kn, v[t], decay, beta)
            got = latch.step(q[t], k[t], v[t], g[t], b[t], a, bias, safe_gate=False)
            self.assert_error(got, ref)


if __name__ == "__main__":
    suite = unittest.main(verbosity=2, exit=False)
    if path := os.environ.get("KDA_TEST_REPORT"):
        Path(path).write_text(
            json.dumps(
                dict(
                    tests=suite.result.testsRun,
                    failures=len(suite.result.failures),
                    errors=len(suite.result.errors),
                    gpu=torch.cuda.get_device_name(),
                    max_absolute_errors=TestKDALatch.measurements,
                ),
                indent=2,
            )
            + "\n"
        )
    sys.exit(not suite.result.wasSuccessful())
