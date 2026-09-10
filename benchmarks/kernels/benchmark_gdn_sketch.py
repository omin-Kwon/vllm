# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare experimental packed GDN step with the full-K CUDA Phi reference.

Synthetic fixed-position step timings only; excludes refresh and model hooks.
Run with TORCH_CUDA_ARCH_LIST=10.0 on B200 and a writable reference build cache.
"""

import argparse
import json
import os
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops.gdn_sketch import (
    StepPlan,
    pack_metadata,
    step,
)
from vllm.third_party.flash_linear_attention.ops.gdn_step_full_cuda import (
    step as cuda_step,
)
from vllm.third_party.flash_linear_attention.ops.gdn_step_phi_cuda import gdn_step_cuda


def measure(call):
    for _ in range(3):
        call()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(32):
            call()
    samples = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 32)
    return sorted(samples)[2]


def run(batch, g, dense_mix, heterogeneous=False, cuda_only=False, position=8):
    ns, h, hv, k, v, w = batch + 1, 16, 48, 128, 128, 16
    widths = (
        ([0, 1, 5, 8, 20, 64, 65, 127, 128] * 6)[:hv]
        if heterogeneous
        else [0 if dense_mix and i % 4 == 0 else g for i in range(hv)]
    )

    def zeros(*shape):
        return torch.zeros(*shape, device="cuda")

    state = zeros(ns, hv, v, k)
    writes, keys, gates = zeros(ns, hv, w, v), zeros(ns, h, w, k), zeros(ns, hv, w)
    factors = zeros(ns, hv, w, g)
    u, phi = zeros(ns, hv, g, v), zeros(ns, hv, g, k)
    u.normal_(0, 0.1)
    phi.normal_(0, 0.01)
    for head, m in enumerate(widths):
        phi[:, head, :m, :m] = torch.eye(m, device="cuda")
    packed, offsets = pack_metadata(phi, widths)
    mh = torch.tensor(widths, device="cuda", dtype=torch.int32)
    indices = torch.arange(1, ns, device="cuda", dtype=torch.int32)
    positions = torch.full_like(
        indices, position if isinstance(position, int) else position[0]
    )
    mapping = torch.arange(ns, device="cuda", dtype=torch.int32)
    mix = zeros(batch, 2 * h * k + hv * v).normal_().bfloat16()
    a = zeros(batch, hv).fill_(-3)
    beta = zeros(batch, hv)
    a_log, bias = zeros(hv), zeros(hv)
    out = zeros(batch, hv, v).bfloat16()
    frozen, anchors = zeros(ns, hv, v), zeros(ns, hv, g)
    nf = torch.where(mh > 0, 0, k).int()

    def old():
        gdn_step_cuda(
            batch,
            mix,
            a,
            beta,
            a_log,
            bias,
            out,
            state,
            writes,
            keys,
            gates,
            indices,
            positions,
            k**-0.5,
            nf,
            frozen,
            frozen,
            u,
            phi,
            mh,
            anchors,
            anchors,
            factors,
            h,
            hv,
            k,
            v,
            w,
            g,
            k,
            True,
            False,
            ls6_map=mapping,
        )

    plan = None

    def new():
        step(
            mix,
            a,
            beta,
            a_log,
            bias,
            out,
            state,
            writes,
            keys,
            gates,
            indices,
            positions,
            mapping,
            u,
            packed,
            offsets,
            mh,
            factors,
            k**-0.5,
            plan=plan,
        )

    def cuda():
        cuda_step(
            mix,
            a,
            beta,
            a_log,
            bias,
            out,
            state,
            writes,
            keys,
            gates,
            indices,
            positions,
            u,
            phi,
            mh,
            factors,
            mapping,
            k**-0.5,
        )

    if cuda_only:
        samples = []
        for p in [position] if isinstance(position, int) else position:
            positions.fill_(p)
            old_us, cuda_us = measure(old), measure(cuda)
            samples.append(
                dict(
                    position=p,
                    reference_us=old_us,
                    cuda_us=cuda_us,
                    speedup=old_us / cuda_us,
                )
            )
        old_us = sum(r["reference_us"] for r in samples) / len(samples)
        cuda_us = sum(r["cuda_us"] for r in samples) / len(samples)
        return dict(
            batch=batch,
            g=g,
            dense_mix=dense_mix,
            heterogeneous=heterogeneous,
            position=position,
            reference_us=old_us,
            cuda_us=cuda_us,
            speedup=old_us / cuda_us,
            samples=samples,
        )
    old_us, new_us = measure(old), measure(new)
    plan = StepPlan(widths, batch, h, "cuda", share_keys=False)
    bucket_us = measure(new)
    plan = StepPlan(widths, batch, h, "cuda", share_keys=True)
    shared_us = measure(new)
    return dict(
        batch=batch,
        g=g,
        dense_mix=dense_mix,
        heterogeneous=heterogeneous,
        reference_us=old_us,
        packed_us=new_us,
        bucket_us=bucket_us,
        shared_us=shared_us,
        speedup=old_us / new_us,
        optimization_speedup=new_us / shared_us,
        shared_vs_reference=old_us / shared_us,
    )


def run_flush(
    batch, g, dense_mix, heterogeneous=False, replayssm=False, fixed_solve_plan=False
):
    """Time real flushes and 16-step cycles, excluding model hooks and setup.

    Fixed input sequences let both paths use precomputed beta history. The new
    path also records beta in its step. The old path's external beta capture and
    mean bookkeeping are excluded, so no speedup is attributed to their removal.
    """
    from vllm.third_party.flash_linear_attention.ops.gdn_flush_full_cuda import (
        FlushWorkspace,
    )
    from vllm.third_party.flash_linear_attention.ops.gdn_flush_stream_cuda import (
        gdn_flush_stream,
    )

    ns, h, hv, k, w = batch + 1, 16, 48, 128, 16
    widths = (
        ([0, 1, 5, 8, 20, 64, 65, 127, 128] * 6)[:hv]
        if heterogeneous
        else [0 if dense_mix and i % 4 == 0 else g for i in range(hv)]
    )

    def zeros(*shape):
        return torch.zeros(*shape, device="cuda")

    state = zeros(ns, hv, k, k).normal_(0, 0.03)
    writes, keys, gates = zeros(ns, hv, w, k), zeros(ns, h, w, k), zeros(ns, hv, w)
    factors, u, phi = zeros(ns, hv, w, g), zeros(ns, hv, g, k), zeros(ns, hv, g, k)
    mh = torch.tensor(widths, device="cuda", dtype=torch.int32)
    indices = torch.arange(1, ns, device="cuda", dtype=torch.int32)
    mapping = torch.arange(ns, device="cuda", dtype=torch.int32)
    rows = torch.cat([indices, torch.tensor([batch], dtype=torch.int32, device="cuda")])
    positions = [torch.full_like(indices, t) for t in range(w)]
    mix = zeros(w, batch, 2 * h * k + hv * k).normal_().bfloat16()
    a, b = zeros(w, batch, hv).normal_(-3, 0.2), zeros(w, batch, hv).normal_()
    a_log, bias, out = zeros(hv), zeros(hv), zeros(batch, hv, k).bfloat16()
    beta_ring = zeros(ns, hv, w)
    beta_ring[1:] = b.sigmoid().permute(1, 2, 0)
    qbar, kbar = zeros(ns, h, k).normal_(0, 0.1), zeros(ns, h, k).normal_(0, 0.1)
    aq, ak, frozen = zeros(ns, hv, g), zeros(ns, hv, g), zeros(ns, hv, k)
    nf = torch.where(mh > 0, 0, k).int()
    workspace = FlushWorkspace(
        batch,
        h,
        hv,
        g,
        "cuda",
        ridge=0.1,
        widths=mh if fixed_solve_plan else None,
    )
    args = (state, writes, keys, gates, rows, mapping, mh, beta_ring, u, phi)
    workspace.refresh(*args)
    # The reference reads full-width Phi; the new refresh intentionally skips it.
    for head, m in enumerate(widths):
        if m == k:
            phi[:, head] = torch.eye(k, device="cuda")

    def old_step(t):
        gdn_step_cuda(
            batch,
            mix[t],
            a[t],
            b[t],
            a_log,
            bias,
            out,
            state,
            writes,
            keys,
            gates,
            indices,
            positions[t],
            k**-0.5,
            nf,
            frozen,
            frozen,
            u,
            phi,
            mh,
            aq,
            ak,
            factors,
            h,
            hv,
            k,
            k,
            w,
            g,
            k,
            True,
            False,
            ls6_map=mapping,
        )

    def new_step(t, step_widths=mh):
        cuda_step(
            mix[t],
            a[t],
            b[t],
            a_log,
            bias,
            out,
            state,
            writes,
            keys,
            gates,
            indices,
            positions[t],
            u,
            phi,
            step_widths,
            factors,
            mapping,
            k**-0.5,
            beta_ring=beta_ring,
        )

    def old_flush():
        gdn_flush_stream(
            state,
            writes,
            keys,
            gates,
            rows,
            batch,
            qbar,
            kbar,
            mapping,
            mh,
            beta_ring,
            u,
            phi,
            aq,
            ak,
            h,
            hv,
            k,
            k,
            w,
            g,
            k,
            ridge=0.1,
        )

    def new_flush():
        workspace.flush(*args)

    def old_cycle():
        for t in range(w):
            old_step(t)
        old_flush()

    def new_cycle():
        for t in range(w):
            new_step(t)
        new_flush()

    # Warm the exact ring convention with a complete input window.
    for t in range(w):
        new_step(t)
    original = [x.clone() for x in (state, writes, keys, gates, factors, u, phi)]

    def reset():
        for dst, src in zip((state, writes, keys, gates, factors, u, phi), original):
            dst.copy_(src)
        aq.zero_()
        ak.zero_()

    replay_timings = {}
    if replayssm:
        from vllm.third_party.flash_linear_attention.ops.gdn_flush_cuda import (
            _ext as replay_extension,
        )
        from vllm.third_party.flash_linear_attention.ops.gdn_flush_cuda import (
            gdn_flush_cuda,
        )

        # Keep this translation unit's build files separate from stream/step.
        cache = os.environ.get("NS_GDN_CUDA_BUILD_DIR")
        os.environ["NS_GDN_CUDA_BUILD_DIR"] = os.environ.get(
            "NS_GDN_REPLAY_FLUSH_BUILD_DIR", "/disk2/omin/.cache/gdn_replay_flush"
        )
        try:
            replay_extension()
        finally:
            if cache is None:
                del os.environ["NS_GDN_CUDA_BUILD_DIR"]
            else:
                os.environ["NS_GDN_CUDA_BUILD_DIR"] = cache

        # ReplaySSM stores exact deltas; sketch heads store raw-WY writes.
        # Generate each representation from the same state and input window.
        dense_widths = torch.zeros_like(mh)
        reset()
        for t in range(w):
            new_step(t, dense_widths)
        dense_writes = writes.clone()
        reset()
        dense_u, dense_phi = zeros(ns, hv, 8, k), zeros(ns, hv, 8, k)
        dense_aq, dense_ak = zeros(ns, hv, 8), zeros(ns, hv, 8)
        sm = torch.cuda.get_device_properties(state.device).multi_processor_count

        def replay_flush():
            gdn_flush_cuda(
                4 * sm,
                state,
                dense_writes,
                keys,
                gates,
                indices,
                rows,
                batch,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                h,
                hv,
                k,
                k,
                w,
                1,
            )

        def replay_stream():
            gdn_flush_stream(
                state,
                dense_writes,
                keys,
                gates,
                rows,
                batch,
                qbar,
                kbar,
                mapping,
                dense_widths,
                None,
                dense_u,
                dense_phi,
                dense_aq,
                dense_ak,
                h,
                hv,
                k,
                k,
                w,
                8,
                k,
                ridge=0.1,
            )

        # Prove the compared flushes produce the same boundary, despite the
        # different write-ring representations. Initialization is untimed.
        replay_flush()
        replay_boundary = state.clone()
        reset()
        new_flush()
        torch.testing.assert_close(state, replay_boundary, atol=5e-5, rtol=4e-3)
        boundary_error = (state - replay_boundary).abs().max().item()
        reset()
        replay_stream()
        torch.testing.assert_close(state, replay_boundary, atol=5e-5, rtol=4e-3)
        stream_error = (state - replay_boundary).abs().max().item()
        for name, call in [
            ("replayssm_flush_us", replay_flush),
            ("replayssm_stream_total_us", replay_stream),
        ]:
            reset()
            replay_timings[name] = measure(call)
        replay_timings.update(
            boundary_max_abs=boundary_error,
            replay_stream_boundary_max_abs=stream_error,
            replay_default_grid=4 * sm,
            stream_default_grid=min(2 * sm, batch * hv),
        )

    timings = {}
    for name, call in [
        ("reference_flush_us", old_flush),
        ("full_flush_us", new_flush),
        ("reference_cycle_us", old_cycle),
        ("full_cycle_us", new_cycle),
    ]:
        reset()
        timings[name] = measure(call)
        if not torch.isfinite(state).all() or not torch.isfinite(out).all():
            raise RuntimeError("nonfinite benchmark state/output")
    reset()
    new_flush()
    for phase, name in [(1, "prep_us"), (2, "stream_us"), (3, "solve_us")]:
        timings[name] = measure(lambda phase=phase: workspace._run(phase, *args))
    return dict(
        batch=batch,
        g=g,
        dense_mix=dense_mix,
        heterogeneous=heterogeneous,
        fixed_solve_plan=fixed_solve_plan,
        **timings,
        **replay_timings,
        flush_speedup=timings["reference_flush_us"] / timings["full_flush_us"],
        cycle_speedup=timings["reference_cycle_us"] / timings["full_cycle_us"],
        workspace_bytes=sum(
            x.numel() * x.element_size()
            for x in (
                workspace.scratch,
                workspace.prep,
                workspace.prep_i,
                workspace._solve_heads,
            )
        ),
    )


def analyze_flush_width(m, g=None, profile=False, fixed_solve_plan=False):
    """Isolate actual m from allocation G, with every head using the same m."""
    from vllm.third_party.flash_linear_attention.ops.gdn_flush_full_cuda import (
        FlushWorkspace,
    )

    torch.manual_seed(194)
    batch, h, hv, k, w = 128, 16, 48, 128, 16
    ns = batch + 1
    g = max(4, (m + 3) // 4 * 4) if g is None else g
    if not 1 <= m <= g <= 128 or g % 4:
        raise ValueError("Require 1 <= m <= G <= 128, with G a multiple of four")
    state = torch.randn(ns, hv, k, k, device="cuda") * 0.03
    writes = torch.randn(ns, hv, w, k, device="cuda") * 0.03
    keys = torch.nn.functional.normalize(
        torch.randn(ns, h, w, k, device="cuda"), dim=-1
    )
    gates = torch.full((ns, hv, w), -0.05, device="cuda")
    beta = torch.full_like(gates, 0.5)
    u = torch.zeros(ns, hv, g, k, device="cuda")
    phi = torch.zeros_like(u)
    mapping = torch.arange(ns, device="cuda", dtype=torch.int32)
    rows = torch.cat([mapping[1:], mapping.new_tensor([batch])])
    widths = torch.full((hv,), m, dtype=torch.int32, device="cuda")
    workspace = FlushWorkspace(
        batch,
        h,
        hv,
        g,
        "cuda",
        ridge=0.1,
        widths=widths if fixed_solve_plan else None,
    )
    args = (state, writes, keys, gates, rows, mapping, widths, beta, u, phi)
    workspace.flush(*args)
    torch.accelerator.synchronize()
    if profile:
        torch.cuda.profiler.start()
        workspace.flush(*args)
        torch.accelerator.synchronize()
        torch.cuda.profiler.stop()
        return dict(batch=batch, m=m, g=g, profile=True)
    timings = {}
    for phase, name in [
        (0, "flush_us"),
        (1, "prep_us"),
        (2, "stream_us"),
        (3, "solve_us"),
    ]:
        timings[name] = measure(lambda phase=phase: workspace._run(phase, *args))
    if not torch.isfinite(state).all() or not torch.isfinite(phi).all():
        raise RuntimeError("nonfinite width-sweep state/metadata")
    mc = (m + 7) // 8 * 8
    capacity = next(cap for cap in (8, 16, 32, 48, 64, 128) if m <= cap)
    solve_stride = g if capacity == 128 else capacity
    return dict(
        batch=batch,
        m=m,
        g=g,
        fixed_solve_plan=fixed_solve_plan,
        **timings,
        coefficient_refresh=0 < m < k,
        gram_padded_width=mc,
        gram_row_tiles_per_chunk=2 if m <= 8 else 1,
        gram_columns_per_chunk=8 if m <= 8 else 16,
        gram_slices=8 if m < k else 0,
        gram_shared_atomics=False,
        solve_rhs=k - m,
        solve_columns_per_warp=32 if m <= 64 else 1,
        solve_row_capacity=capacity,
        solve_launches=(
            int(m < k)
            if fixed_solve_plan
            else 1 + sum(g > lower for lower in (8, 16, 32, 48, 64))
        ),
        solve_shared_bytes=(
            0
            if fixed_solve_plan and m == k
            else 4
            * (
                solve_stride * (solve_stride + 1)
                + (capacity * (129 - capacity) if capacity <= 64 else 64 * 65)
            )
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-only", action="store_true")
    parser.add_argument("--all-nonflush", action="store_true")
    parser.add_argument("--flush", action="store_true")
    parser.add_argument("--replayssm", action="store_true")
    parser.add_argument("--width-sweep", action="store_true")
    parser.add_argument("--profile-width", type=int)
    parser.add_argument("--allocation-g", type=int)
    parser.add_argument("--fixed-solve-plan", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(83)
    results = []
    if args.width_sweep or args.profile_width is not None:
        widths = (
            [args.profile_width]
            if args.profile_width is not None
            else [4, 8, 16, 17, 24, 32, 33, 48, 64, 80, 96, 120, 127, 128]
        )
        for m in widths:
            result = analyze_flush_width(
                m,
                args.allocation_g,
                args.profile_width is not None,
                args.fixed_solve_plan,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
            args.output.write_text(json.dumps(results, indent=2) + "\n")
        return
    cases = [
        (1, 8, False, False),
        (32, 8, False, False),
        (128, 8, False, False),
        (128, 8, True, False),
        (128, 32, True, False),
        (128, 128, True, False),
        (128, 128, True, True),
    ]
    if args.replayssm:
        cases = [
            (batch, g, dense_mix, False)
            for batch in (128, 256)
            for g, dense_mix in ((8, False), (32, True))
        ]
    for batch, g, dense_mix, heterogeneous in cases:
        result = (
            run_flush(
                batch,
                g,
                dense_mix,
                heterogeneous,
                replayssm=args.replayssm,
                fixed_solve_plan=args.fixed_solve_plan,
            )
            if args.flush or args.replayssm
            else run(
                batch,
                g,
                dense_mix,
                heterogeneous,
                cuda_only=args.cuda_only,
                position=list(range(15)) if args.all_nonflush else 8,
            )
        )
        results.append(result)
        print(json.dumps(result), flush=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
