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

    def test_partial_flush_handoff_uses_raw_ring_and_is_idempotent(self):
        state, om = self.case(8, slots=2)
        mask = torch.tensor([True, False], device="cuda")
        latch = KDALatchState(state, om, latch_heads=mask)
        oracle = PaperOracle(state, om, latch_heads=mask)
        q, k, v, g, b, a, bias = inputs(batch=2, steps=7)
        for t in range(7):
            latch.step(q[t], k[t], v[t], g[t], b[t], a, bias)
            qn, kn, decay, beta = normalize(q[t], k[t], g[t], b[t], a, bias)
            oracle.step(qn, kn, v[t], decay, beta)
        untouched = latch.state[0].clone()
        for tensor in (latch.f, latch.u, latch.latch, latch.phi):
            tensor.fill_(float("nan"))
        slots = torch.tensor([1], device="cuda")
        latch.flush_pending(slots)
        self.assert_error(latch.state[1], oracle.state[1], atol=3e-5, rtol=2e-4)
        self.assertTrue(torch.equal(latch.state[0], untouched))
        self.assertEqual(latch.pos.tolist(), [7, 0])
        saved = latch.state.clone()
        latch.flush_pending(slots)
        self.assertTrue(torch.equal(latch.state, saved))
        fresh = KDALatchState(saved[1:2], om, latch_heads=mask)
        got = latch.step(
            q[0, 1:],
            k[0, 1:],
            v[0, 1:],
            g[0, 1:],
            b[0, 1:],
            a,
            bias,
            slots=slots,
        )
        expected = fresh.step(
            q[0, 1:],
            k[0, 1:],
            v[0, 1:],
            g[0, 1:],
            b[0, 1:],
            a,
            bias,
        )
        self.assertTrue(torch.equal(got, expected))

    def test_engine_slot_handoff_reorder_and_reuse(self):
        from vllm.models.glm5next.nvidia.kda_latch import KDALatchCache

        state, om = self.case(8, slots=2)
        persistent = torch.cat([torch.zeros_like(state[:1]), state.clone()])
        adapter = KDALatchCache(om)
        oracle = PaperOracle(state, om)
        q, k, v, g, b, a, bias = inputs(batch=2, steps=19)
        for t in range(19):
            order = [1, 0] if t % 2 else [0, 1]
            ids = torch.tensor([i + 1 for i in order], device="cuda")
            got = adapter.step(
                persistent,
                ids,
                q[t, order],
                k[t, order],
                v[t, order],
                g[t, order],
                b[t, order],
                a,
                bias,
                -5.0,
            )
            qn, kn, decay, beta = normalize(
                q[t, order], k[t, order], g[t, order], b[t, order], a, bias
            )
            expected = oracle.step(qn, kn, v[t, order], decay, beta, order)
            self.assert_error(got, expected)
            if t == 15:
                self.assert_error(persistent[1:], oracle.state, atol=3e-5, rtol=2e-4)
        abandoned = persistent[1].clone()
        for tensor in (adapter.slots[2].f, adapter.slots[2].u):
            tensor.fill_(float("nan"))
        ids = torch.tensor([2, 1], device="cuda")
        adapter.before_prefill(
            persistent, ids, torch.tensor([True, False], device="cuda")
        )
        self.assert_error(persistent[2], oracle.state[1], atol=3e-5, rtol=2e-4)
        self.assertTrue(torch.equal(persistent[1], abandoned))
        self.assertEqual(adapter.slots, {})
        self.assertEqual(adapter.decode_rows, 38)
        persistent[1] = state[0]
        fresh = KDALatchState(persistent[ids], om)
        got = adapter.step(persistent, ids, q[0], k[0], v[0], g[0], b[0], a, bias, -5.0)
        expected = fresh.step(q[0], k[0], v[0], g[0], b[0], a, bias)
        self.assertTrue(torch.equal(got, expected))
        self.assertTrue(torch.equal(persistent[0], torch.zeros_like(persistent[0])))

    def test_compact_pool_eviction_materializes_exact_state_before_reuse(self):
        from vllm.models.glm5next.nvidia.kda_latch import BatchedKDALatchCache

        state, om = self.case(8, slots=3)
        persistent = torch.cat([torch.zeros_like(state[:1]), state.clone()])
        cache = BatchedKDALatchCache(
            om, torch.tensor([8, 8], device="cuda"), capacity=1
        )
        dense = state.double().clone()
        q, k, v, g, b, a, bias = inputs(batch=1, steps=6)
        previous = None
        for t in range(6):
            slot = t % 3
            oracle = PaperOracle(dense[slot : slot + 1], om)
            qn, kn, decay, beta = normalize(q[t], k[t], g[t], b[t], a, bias)
            expected = oracle.step(qn, kn, v[t], decay, beta)
            got = cache.step(
                persistent,
                torch.tensor([slot + 1], device="cuda"),
                q[t],
                k[t],
                v[t],
                g[t],
                b[t],
                a,
                bias,
                -5.0,
            )
            self.assert_error(got, expected)
            if previous is not None:
                self.assert_error(persistent[previous + 1], dense[previous])
            dense[slot] = oracle.state[0]
            previous = slot
            self.assertEqual(cache.live, {slot + 1})
            self.assertEqual(len(cache.pool.state), 1)
        assert previous is not None
        cache.before_prefill(
            persistent,
            torch.tensor([previous + 1], device="cuda"),
            torch.tensor([True], device="cuda"),
        )
        self.assert_error(persistent[1:], dense, atol=3e-5, rtol=2e-4)
        self.assertTrue((persistent[0] == 0).all())

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


class TestCalibrationStatistics(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_heterogeneous_batched_allocation_matches_separate_heads(self):
        from vllm.models.glm5next.nvidia.kda_latch import BatchedKDALatchCache

        torch.manual_seed(812)
        state = torch.randn(3, 3, 128, 128, device="cuda") * 0.01
        state[0].zero_()
        omega = torch.linalg.qr(torch.randn(3, 128, 8, device="cuda")).Q
        ranks = torch.tensor([2, 8, 0], device="cuda")
        cache = BatchedKDALatchCache(omega, ranks)
        controls = [
            KDALatchState(
                state[1:, h : h + 1],
                omega[h : h + 1, :, : max(1, int(rank))],
                latch_heads=torch.tensor([bool(rank)], device="cuda"),
            )
            for h, rank in enumerate(ranks)
        ]
        q, k, v, g, b, a, bias = inputs(batch=2, heads=3, steps=19)
        for t in range(19):
            order = [1, 0] if t % 2 else [0, 1]
            slots = torch.tensor(order, device="cuda")
            got = cache.step(
                state,
                slots + 1,
                q[t, order],
                k[t, order],
                v[t, order],
                g[t, order],
                b[t, order],
                a,
                bias,
                -5.0,
            )
            expected = torch.cat(
                [
                    control.step(
                        q[t, order, h : h + 1],
                        k[t, order, h : h + 1],
                        v[t, order, h : h + 1],
                        g[t, order, h : h + 1],
                        b[t, order, h : h + 1],
                        a[h : h + 1],
                        bias[h : h + 1],
                        slots=slots,
                    )
                    for h, control in enumerate(controls)
                ],
                1,
            )
            torch.testing.assert_close(got, expected, atol=2e-5, rtol=2e-4)
        cache.pool.phi.fill_(float("nan"))
        cache.pool.f.fill_(float("nan"))
        cache.before_prefill(
            state,
            torch.tensor([2, 1], device="cuda"),
            torch.tensor([True, False], device="cuda"),
        )
        for h, control in enumerate(controls):
            control.flush_pending(torch.tensor([1], device="cuda"))
            torch.testing.assert_close(
                state[2, h], control.state[1, 0], atol=2e-5, rtol=2e-4
            )
        self.assertEqual(cache.live, set())
        self.assertTrue((state[0] == 0).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_replay_preserves_reordering_eviction_padding_and_handoff(self):
        from vllm.models.glm5next.nvidia.kda_latch import BatchedKDALatchCache
        from vllm.third_party.flash_linear_attention.ops import kda_latch_graph

        torch.manual_seed(394)
        storage = torch.full((5, 3, 128, 132), float("nan"), device="cuda")
        state = storage[..., :128]
        state.copy_(torch.randn_like(state) * 0.03)
        state[0].zero_()
        omega = torch.linalg.qr(torch.randn(3, 128, 32, device="cuda")).Q
        ranks = torch.tensor([2, 32, 0], device="cuda")
        ref = BatchedKDALatchCache(omega, ranks, capacity=2)
        cache = kda_latch_graph.KDALatchGraphCache(omega, ranks, capacity=2)
        reference_state = state.clone()
        data = inputs(batch=2, heads=3, steps=40)
        a, bias = data[5:]
        ids = torch.tensor([1, 2], device="cuda")
        packed = torch.empty(2, 4, 3, 128, device="cuda", dtype=torch.bfloat16)
        packed_beta = torch.empty(2, 6, device="cuda", dtype=torch.bfloat16)
        args = [packed[:, index] for index in range(4)] + [packed_beta[:, :3]]
        for destination, source in zip(args, data[:5]):
            destination.copy_(source[0])
        for _ in range(3):
            cache.step(state, ids, *args, a, bias)
            ref.step(reference_state, ids, *args, a, bias, -5.0)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = cache.step(state, ids, *args, a, bias)
        for t in range(40):
            physical = [1, 2] if t < 20 else [3, 1]
            if t % 2:
                physical.reverse()
            if t == 30:
                physical[1] = 0
            ids.copy_(torch.tensor(physical, device="cuda"))
            for destination, source in zip(args, data[:5]):
                destination.copy_(source[t])
            graph.replay()
            valid = torch.tensor(
                [i for i, slot in enumerate(physical) if slot > 0], device="cuda"
            )
            expected = ref.step(
                reference_state,
                ids[valid],
                *(x[valid] for x in args),
                a,
                bias,
                -5.0,
            )
            torch.testing.assert_close(out[valid], expected, atol=1e-4, rtol=0.005)
            if t == 25:
                flags = torch.tensor([True, False], device="cuda")
                cache.before_prefill(state, ids, flags)
                ref.before_prefill(reference_state, ids, flags)
                torch.testing.assert_close(state, reference_state, atol=1e-5, rtol=2e-4)
        flags = torch.ones(2, device="cuda", dtype=torch.bool)
        cache.before_prefill(state, ids, flags)
        ref.before_prefill(reference_state, ids, flags)
        torch.testing.assert_close(state, reference_state, atol=1e-5, rtol=2e-4)
        self.assertTrue((state[0] == 0).all())
        self.assertTrue(torch.isnan(storage[..., 128:]).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_refresh_preserves_metric_ridge_for_ill_conditioned_state(self):
        from vllm.third_party.flash_linear_attention.ops import kda_latch_graph

        torch.manual_seed(492)
        for rank in (1, 8, 32, 52):
            with self.subTest(rank=rank):
                state = torch.randn(2, 2, 128, 128, device="cuda")
                state *= torch.logspace(-5, 0, 128, device="cuda")
                omega = torch.linalg.qr(
                    torch.randn(2, 128, rank, device="cuda")
                ).Q * torch.linspace(0.6, 1.8, rank, device="cuda")
                ranks = torch.tensor([rank, max(1, rank // 2)], device="cuda")
                ref = KDALatchState(state, omega, head_ranks=ranks)
                cache = kda_latch_graph.KDALatchGraphCache(omega, ranks, capacity=2)
                cache.pool.state.copy_(state)
                cache.refresh(torch.arange(2, device="cuda", dtype=torch.int32))
                query = torch.randn(2, 2, 128, 8, device="cuda")
                expected = ref.latch @ (ref.phi.transpose(-1, -2) @ query)
                actual = cache.pool.latch @ (cache.pool.phi.transpose(-1, -2) @ query)
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=0.003)

    def test_paired_prefix_curves_match_explicit_residuals(self):
        from benchmarks.kernels.glm_kda_calibration_stats import (
            paired_curves,
            prefix_qr,
        )

        torch.manual_seed(918)
        u = torch.randn(2, 3, 7, 4, dtype=torch.float64)
        u[..., 2] = u[..., 0]
        q, rejected = prefix_qr(u)
        self.assertTrue(rejected[..., 2].all())
        output = torch.randn(2, 5, 3, 7, dtype=torch.float64)
        grad = torch.randn_like(output)
        got = paired_curves(grad, output, q)
        for rank in range(5):
            residual = output - torch.einsum(
                "uhvm,uwhm->uwhv",
                q[..., :rank],
                torch.einsum("uhvm,uwhv->uwhm", q[..., :rank], output),
            )
            torch.testing.assert_close(
                got["output_error_sum"][:, rank], residual.square().sum((0, 1, 3))
            )
            torch.testing.assert_close(
                got["joint_dot_sq_sum"][:, rank],
                (grad * residual).sum(-1).square().sum((0, 1)),
            )

    @unittest.skipUnless(torch.accelerator.device_count() >= 2, "Two GPUs required")
    def test_calibration_features_launch_on_input_device(self):
        from benchmarks.kernels.glm_kda_calibration_stats import features

        q, k, v, g, b, a, bias = inputs(batch=1, heads=2, steps=16)
        _, _, decay, beta = normalize(q[:, 0], k[:, 0], g[:, 0], b[:, 0], a, bias)
        args = (
            q[:, 0][None],
            k[:, 0][None],
            v[:, 0][None],
            decay.log().float()[None],
            beta.float()[None],
        )
        reference = features(*args)
        other = (
            torch.accelerator.current_device_index() + 1
        ) % torch.accelerator.device_count()
        actual = features(*(x.to(f"cuda:{other}") for x in args))
        for got, expected in zip(actual, reference):
            if isinstance(got, torch.Tensor):
                torch.testing.assert_close(got.to(expected.device), expected)
            else:
                self.assertAlmostEqual(got, expected, places=6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_calibration_features_match_dense_matrix_transition(self):
        from benchmarks.kernels.glm_kda_calibration_stats import features

        q, k, v, g, b, a, bias = inputs(batch=1, heads=2, steps=32)
        qn, kn, decay, beta = normalize(q[:, 0], k[:, 0], g[:, 0], b[:, 0], a, bias)
        starts, x, boundary, output, _ = features(
            q[:, 0][None],
            k[:, 0][None],
            v[:, 0][None],
            decay.log().float()[None],
            beta.float()[None],
        )
        state = torch.zeros(2, 128, 128, device="cuda", dtype=torch.float64)
        eye = torch.eye(128, device="cuda", dtype=torch.float64)
        for t in range(32):
            if t % 16 == 0:
                origin, product = state.clone(), eye.expand(2, -1, -1).clone()
                torch.testing.assert_close(
                    starts[t // 16].double(), state, atol=2e-5, rtol=2e-4
                )
            transition = (
                eye - beta[t, :, None, None] * kn[t, :, :, None] * kn[t, :, None, :]
            ) * decay[t, :, None, :]
            state = (
                state @ transition.transpose(-1, -2)
                + beta[t, :, None, None]
                * v[t, 0].double()[..., None]
                * kn[t, :, None, :]
            )
            product = transition @ product
            effective = (product.transpose(-1, -2) @ qn[t, :, :, None])[..., 0]
            torch.testing.assert_close(
                x[t // 16, t % 16].double(), effective, atol=2e-6, rtol=2e-4
            )
            torch.testing.assert_close(
                output[t].double(),
                (state @ qn[t, :, :, None])[..., 0],
                atol=2e-5,
                rtol=2e-4,
            )
            torch.testing.assert_close(
                boundary[t // 16, t % 16].double(),
                (origin @ effective[..., None])[..., 0],
                atol=2e-5,
                rtol=2e-4,
            )


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
