"""Experimental SM100 exact GDN checkpoint fold using native tcgen05.

The Qwen3.8-Flash-Next specialization keeps the checkpoint tile in shared
memory.  A native TF32 tensor-core product forms ``S0 @ K.T`` and the same
tile is then consumed by the exact triangular correction and Replay16 fold.
This file is intentionally kept behind an opt-in switch until the numerical
and performance gates in ``scale/verify_gdn_fs.sh`` pass.
"""

from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import _tcgen05, simple_tma_copy


class Sm100ExactGdnFold:
    H = 16
    HV = 48
    K = 128
    V = 128
    W = 16

    def __init__(self, profile_mode: int = 0):
        self.profile_mode = profile_mode
        # Mode 4 tiles the value dimension into two independent 64-row CTAs.
        # It duplicates only the tiny key/Gram work, while halving shared
        # memory and permitting two checkpoint CTAs to reside on an SM.
        self.M = 64 if profile_mode in (4, 8, 9) else self.V
        self.WARPS = 4 if (profile_mode == 6 or self.M == 64) else 10
        self.STAGE_THREADS = min(self.WARPS * 32, 256)

    @cute.jit
    def _make_tile_tma(self, tensor: cute.Tensor, rows: cutlass.Constexpr[int]):
        # Logical row-major [rows, K], represented in the canonical 128-byte
        # swizzle expected by tcgen05's K-major shared descriptor.
        num_elems = 128 // (tensor.element_type.width // 8)
        swizzle = cute.make_swizzle(3, 4, 3)
        layout = cute.make_layout(
            (1, 1, rows, (num_elems, self.K // num_elems)),
            stride=(0, 0, num_elems, (1, rows * num_elems)),
        )
        layout = cute.make_composed_layout(swizzle, 0, layout)
        atom = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(tensor, (None, None, None, num_elems)),
            layout,
            cta_tiler=(1, 1, rows, self.K),
        )
        return atom

    @cute.jit
    def __call__(
        self,
        h0: cute.Tensor,
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        flush_list: cute.Tensor,
        n_ptr: cute.Tensor,
        ls6_mh: cute.Tensor,
        ls6_beta: cute.Tensor,
        ls6_map: cute.Tensor,
        stream: CUstream,
    ):
        h0_tma = self._make_tile_tma(h0, self.M)
        k_tma = self._make_tile_tma(k_cache, self.W)
        self.kernel(
            h0,
            h0_tma,
            k_cache,
            k_tma,
            d_cache,
            g_cache,
            flush_list,
            n_ptr,
            ls6_mh,
            ls6_beta,
            ls6_map,
        ).launch(
            grid=(
                (flush_list.shape[0] - 1)
                * self.HV
                * (self.V // self.M),
                1,
                1,
            ),
            block=(self.WARPS * 32, 1, 1),
            min_blocks_per_mp=2 if self.profile_mode == 5 else 1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        h0: cute.Tensor,
        h0_tma: cpasync.TmaInfo,
        k_cache: cute.Tensor,
        k_tma: cpasync.TmaInfo,
        d_cache: cute.Tensor,
        g_cache: cute.Tensor,
        flush_list: cute.Tensor,
        n_ptr: cute.Tensor,
        ls6_mh: cute.Tensor,
        ls6_beta: cute.Tensor,
        ls6_map: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        work, _, _ = cute.arch.block_idx()
        value_tiles = self.V // self.M
        rr = work // (self.HV * value_tiles)
        rem = work - rr * self.HV * value_tiles
        ihv = rem // value_tiles
        value_tile = rem - ihv * value_tiles
        value_base = value_tile * self.M
        ih = ihv // (self.HV // self.H)

        # The first prototype launches the compact list's full capacity.  The
        # production wrapper supplies a latch-only work list; avoiding a
        # staged early-return also keeps every CTA's barrier path identical.
        sidx = flush_list[rr]
        cidx = ls6_map[sidx]

        smem = cutlass.utils.SmemAllocator()

        def alloc_tma(info):
            return smem.allocate_tensor(
                Float32,
                info.smem_layout.outer,
                byte_alignment=128,
                swizzle=info.smem_layout.inner,
            )[0, 0, None, None]

        s_h0 = alloc_tma(h0_tma)
        s_k = alloc_tma(k_tma)
        # Once gamma*S0 has been seeded into TMEM, the state tile is dead.
        # Reuse its first 16 KiB as the padded K-major B operand for D @ K.T;
        # this keeps the whole M=64 CTA below the two-resident shared limit.
        swizzle = cute.make_swizzle(3, 4, 3)
        kt_mma_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.K, (32, 1)), stride=(32, (1, self.K * 32))
            ),
        )
        s_kt_mma = cute.make_tensor(
            s_h0.iterator, kt_mma_layout.outer
        )
        # The second TF32 MMA has reduction extent W=16.  Pad it to one
        # 128-byte swizzle atom (32 fp32 values) so D[V,W] and K.T[K,W]
        # can be consumed directly by tcgen05.
        d_mma_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.M, (32, 1)), stride=(32, (1, self.M * 32))
            ),
        )
        s_d_mma = smem.allocate_tensor(
            Float32,
            d_mma_layout.outer,
            byte_alignment=128,
            swizzle=d_mma_layout.inner,
        )
        s_gram = smem.allocate_tensor(
            Float32, cute.make_layout((self.W, self.W), stride=(self.W, 1)),
            byte_alignment=16,
        )
        s_prefix = smem.allocate_array(Float32, self.W)
        s_beta = smem.allocate_array(Float32, self.W)
        tma_bar = smem.allocate_array(Int64, 1)
        mma_bar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tma_bar, 1)
                cute.arch.mbarrier_init(mma_bar, 1)
                cute.arch.mbarrier_init_fence()
        elif warp == self.WARPS - 1:
            cpasync.prefetch_descriptor(h0_tma.atom)
            cpasync.prefetch_descriptor(k_tma.atom)
        cute.arch.sync_threads()

        if warp == self.WARPS - 2:
            # This kernel uses columns [base, base+159].  Reserving 256 rather
            # than the old blanket 512 permits two independent CTAs to own
            # tensor memory on the same SM.  The allocator may return a
            # nonzero base, so every access below is relative to tbase.
            tcols = 32 if self.profile_mode in (7, 8, 9) else 256
            _tcgen05.alloc(taddr, ncols=tcols)

        if warp == self.WARPS - 1:
            with cute.arch.elect_one():
                nbytes = (self.M + self.W) * self.K * 4
                cute.arch.mbarrier_arrive_and_expect_tx(tma_bar, nbytes)
            h0_tiles = cute.local_tile(
                h0_tma.tma_tensor[sidx, ihv, None, None],
                (self.M, self.K),
                (value_tile, 0),
            )
            simple_tma_copy(
                h0_tma.atom,
                h0_tiles,
                s_h0,
                tma_bar,
            )
            simple_tma_copy(
                k_tma.atom,
                k_tma.tma_tensor[sidx, ih, None, None],
                s_k,
                tma_bar,
            )

        if warp == 0:
            cute.arch.mbarrier_wait(tma_bar, 0)
        cute.arch.sync_threads()
        tbase = taddr[0]

        if tid < self.STAGE_THREADS:
            # Stage raw writes and the state-independent key Gram matrix.
            # The corrected/scaled operand overwrites these slots after the
            # projected full-WY product arrives, so no second D buffer exists.
            for q in cutlass.range_constexpr(
                self.W * self.M // self.STAGE_THREADS
            ):
                x = tid + q * self.STAGE_THREADS
                s = x // self.M
                v_stage = x - s * self.M
                s_d_mma[v_stage, (s, 0)] = d_cache[
                    sidx, ihv, s, value_base + v_stage
                ]

            if tid < self.W:
                prefix = Float32(0.0)
                for j in cutlass.range(self.W):
                    if j <= tid:
                        prefix += g_cache[sidx, ihv, j]
                s_prefix[tid] = prefix
                s_beta[tid] = ls6_beta[cidx, ihv, tid]

            for q in cutlass.range_constexpr(
                self.W * self.W // self.STAGE_THREADS
            ):
                x = tid + q * self.STAGE_THREADS
                j = x // self.W
                s = x - j * self.W
                dot = Float32(0.0)
                if j < s and cutlass.const_expr(
                    self.profile_mode == 0
                    or self.profile_mode == 3
                    or self.profile_mode == 4
                    or self.profile_mode == 5
                    or self.profile_mode == 6
                    or self.profile_mode == 7
                    or self.profile_mode == 9
                ):
                    for kk in cutlass.range_constexpr(self.K):
                        dot += s_k[j, (kk % 32, kk // 32)] * s_k[
                            s, (kk % 32, kk // 32)
                        ]
                s_gram[j, s] = dot

        cute.arch.sync_threads()

        # Full normalized WY factor, computed once in key space:
        #   F_s = beta_s (k_s - sum_{j<s} F_j <k_j,k_s>).
        # This replaces the old 128 independent value-row triangular solves.
        # The raw K-major ring is overwritten in place; final raw keys are
        # reloaded later into the dead state allocation.
        if tid < self.K and cutlass.const_expr(self.profile_mode != 2):
            f_hist = cute.make_rmem_tensor(self.W, Float32)
            for s in cutlass.range_constexpr(self.W):
                prior = Float32(0.0)
                for j in cutlass.range_constexpr(self.W):
                    if j < s:
                        prior += f_hist[j] * s_gram[j, s]
                fval = s_beta[s] * (
                    s_k[s, (tid % 32, tid // 32)] - prior
                )
                f_hist[s] = fval
                s_k[s, (tid % 32, tid // 32)] = fval

        cute.arch.sync_threads()

        # P[v,s] = S0[v,:] F_s now contains the complete exact correction;
        # no value-direction causal solve remains below.
        if warp == self.WARPS - 2 and cutlass.const_expr(
            self.profile_mode != 2
        ):
            idesc = _tcgen05.make_tf32_idesc(self.M, self.W)
            desc_template = _tcgen05.make_sdesc_128B_swizzle(self.M * 128)
            h_base = desc_template | (s_h0[None, None].iterator.toint() >> 4)
            k_template = _tcgen05.make_sdesc_128B_swizzle(self.V * 128)
            k_base = k_template | (s_k[None, None].iterator.toint() >> 4)
            for kb in cutlass.range_constexpr(self.K // 32):
                for ki in cutlass.range_constexpr(32 // 8):
                    h_desc = h_base | ((kb * self.M * 128 + ki * 32) >> 4)
                    k_desc = k_base | ((kb * self.W * 128 + ki * 32) >> 4)
                    _tcgen05.mma_tf32(
                        tbase, h_desc, k_desc, idesc, kb > 0 or ki > 0
                    )
            _tcgen05.commit(mma_bar)

        if warp < 4 and cutlass.const_expr(
            self.profile_mode != 2
        ):
            if warp == 0:
                cute.arch.mbarrier_wait(mma_bar, 0)
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            _tcgen05.fence_after_thread_sync()
            # M=64 distributes four logical 16-row groups at TMEM rows
            # 0/32/64/96; only the low 16 lanes in each load are live.
            # M=128 uses the same four addresses with all 32 lanes live.
            trow = warp * 32
            p = _tcgen05.ld(trow, tbase, "32x32b", self.W)
            _tcgen05.wait_ld()
            rows_per_group = self.M // 4
            if lane < rows_per_group:
                v_corr = warp * rows_per_group + lane
                total = s_prefix[self.W - 1]
                for s in cutlass.range_constexpr(self.W):
                    h = cute.math.exp(
                        s_prefix[s], fastmath=True
                    ) * p[s]
                    corr = s_d_mma[v_corr, (s, 0)] - h
                    d_cache[sidx, ihv, s, value_base + v_corr] = corr
                    s_d_mma[v_corr, (s, 0)] = corr * cute.math.exp(
                        total - s_prefix[s], fastmath=True
                    )
                if cutlass.const_expr(self.profile_mode == 8):
                    for s in cutlass.range_constexpr(self.W):
                        d_cache[sidx, ihv, s, value_base + v_corr] = p[s]
                for s in cutlass.range_constexpr(self.W, 32):
                    s_d_mma[v_corr, (s, 0)] = 0.0

        if tid < self.M and cutlass.const_expr(self.profile_mode == 2):
            v_fold = tid
            total = s_prefix[self.W - 1]
            for s in cutlass.range_constexpr(self.W):
                s_d_mma[v_fold, (s, 0)] = d_cache[
                    sidx, ihv, s, value_base + v_fold
                ] * cute.math.exp(total - s_prefix[s], fastmath=True)
            for s in cutlass.range_constexpr(self.W, 32):
                s_d_mma[v_fold, (s, 0)] = 0.0

        cute.arch.sync_threads()

        if cutlass.const_expr(self.profile_mode == 8):
            if warp == self.WARPS - 2:
                _tcgen05.dealloc(tbase, ncols=32)
            return

        if cutlass.const_expr(self.profile_mode == 9):
            # Two threads share each value row and own four columns out of
            # every eight-column strip.  This retains the mature scalar fold
            # while the 49-KiB tile and 128-thread CTA permit two residents.
            v_scalar = tid // 2
            ko = (tid % 2) * 4
            total_scale_scalar = cute.math.exp(
                s_prefix[self.W - 1], fastmath=True
            )
            for kb in cutlass.range(self.K // 8):
                acc = cute.make_rmem_tensor(4, Float32)
                for ki in cutlass.range_constexpr(4):
                    k = kb * 8 + ko + ki
                    acc[ki] = (
                        s_h0[v_scalar, (k % 32, k // 32)]
                        * total_scale_scalar
                    )
                for s in cutlass.range(self.W):
                    dv = s_d_mma[v_scalar, (s, 0)]
                    for ki in cutlass.range_constexpr(4):
                        k = kb * 8 + ko + ki
                        acc[ki] += dv * s_k[s, (k % 32, k // 32)]
                for ki in cutlass.range_constexpr(4):
                    k = kb * 8 + ko + ki
                    h0[sidx, ihv, value_base + v_scalar, k] = acc[ki]
            cute.arch.sync_threads()
            if warp == self.WARPS - 2:
                _tcgen05.dealloc(tbase, ncols=32)
            return

        if cutlass.const_expr(self.profile_mode == 7):
            if warp == self.WARPS - 2:
                _tcgen05.dealloc(tbase, ncols=32)
            return

        # Seed the second accumulator with gamma*S0 before reusing the state
        # allocation.  For M=64 the four warps own 16 live rows apiece at
        # TMEM row bases 0/32/64/96; the upper 16 lanes store only unused rows.
        if warp < 4:
            trow_seed = warp * 32
            rows_seed = self.M // 4
            v_seed = warp * rows_seed + lane
            gamma_seed = cute.math.exp(
                s_prefix[self.W - 1], fastmath=True
            )
            for kb in cutlass.range_constexpr(self.K // 32):
                seed = cute.make_rmem_tensor(32, Float32)
                for j in cutlass.range_constexpr(32):
                    if lane < rows_seed:
                        seed[j] = gamma_seed * s_h0[
                            v_seed, (j, kb)
                        ]
                    else:
                        seed[j] = 0.0
                _tcgen05.st(
                    trow_seed,
                    tbase + 32 + kb * 32,
                    "32x32b",
                    32,
                    seed,
                )
            _tcgen05.wait_st()
            _tcgen05.fence_before_thread_sync()

        cute.arch.sync_threads()

        # Repack raw keys into the first 16 KiB of the now-dead state tile as
        # a conventional K-major [N=128,Kred=32] operand.  Padding columns
        # 16..31 are zero so only the two real K=8 instructions contribute.
        if tid < self.STAGE_THREADS:
            for q in cutlass.range_constexpr(
                self.K * self.W // self.STAGE_THREADS
            ):
                x = tid + q * self.STAGE_THREADS
                n = x // self.W
                s = x - n * self.W
                s_kt_mma[n, (s, 0)] = k_cache[sidx, ih, s, n]
            for q in cutlass.range_constexpr(
                self.K * (32 - self.W) // self.STAGE_THREADS
            ):
                x = tid + q * self.STAGE_THREADS
                n = x // (32 - self.W)
                s = x - n * (32 - self.W) + self.W
                s_kt_mma[n, (s, 0)] = 0.0

        cute.arch.sync_threads()

        if warp == self.WARPS - 2:
            idesc = _tcgen05.make_tf32_idesc(self.M, self.K)
            desc_template = _tcgen05.make_sdesc_128B_swizzle(self.M * 128)
            d_base = desc_template | (s_d_mma[None, None].iterator.toint() >> 4)
            kt_template = _tcgen05.make_sdesc_128B_swizzle(self.K * 128)
            kt_base = kt_template | (
                s_kt_mma[None, None].iterator.toint() >> 4
            )
            for ki in cutlass.range_constexpr(self.W // 8):
                d_desc = d_base | ((ki * 32) >> 4)
                kt_desc = kt_base | ((ki * 32) >> 4)
                _tcgen05.mma_tf32(
                    tbase + 32,
                    d_desc,
                    kt_desc,
                    idesc,
                    True,
                )
            _tcgen05.commit(mma_bar)

        cute.arch.sync_threads()

        if warp < 4:
            if warp == 0:
                cute.arch.mbarrier_wait(
                    mma_bar,
                    0
                    if cutlass.const_expr(self.profile_mode == 2)
                    else 1,
                )
            cute.arch.barrier(barrier_id=1, number_of_threads=128)
            _tcgen05.fence_after_thread_sync()
            trow = warp * 32
            rows_per_group = self.M // 4
            v_out = warp * rows_per_group + lane
            for kb in cutlass.range_constexpr(self.K // 32):
                out = _tcgen05.ld(
                    trow, tbase + 32 + kb * 32, "32x32b", 32
                )
                _tcgen05.wait_ld()
                if lane < rows_per_group:
                    for j in cutlass.range_constexpr(32):
                        k = kb * 32 + j
                        h0[sidx, ihv, value_base + v_out, k] = out[j]

        cute.arch.sync_threads()
        if warp == self.WARPS - 2:
            _tcgen05.dealloc(tbase, ncols=256)

    @cache
    @staticmethod
    def compile(profile_mode: int = 0):
        nx = cute.sym_int()
        ns = cute.sym_int()
        bf = cute.sym_int()
        h0 = make_fake_tensor(Float32, (nx, 48, 128, 128), divisibility=16)
        d = make_fake_tensor(Float32, (nx, 48, 16, 128), divisibility=16)
        k = make_fake_tensor(Float32, (nx, 16, 16, 128), divisibility=16)
        g = make_fake_tensor(Float32, (nx, 48, 16), divisibility=4)
        fl = make_fake_tensor(Int32, (bf,), divisibility=1)
        nptr = make_fake_tensor(Int32, (1,), divisibility=1)
        mh = make_fake_tensor(Int32, (48,), divisibility=1)
        beta = make_fake_tensor(Float32, (ns, 48, 16), divisibility=4)
        mapping = make_fake_tensor(Int32, (nx,), divisibility=1)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm100ExactGdnFold(profile_mode),
            h0,
            d,
            k,
            g,
            fl,
            nptr,
            mh,
            beta,
            mapping,
            stream,
            options="--enable-tvm-ffi",
        )


class Sm100ExactGdnFold4:
    """Four-warp shared-resident state fold with one native tcgen05 read.

    The first version kept all 128 state columns plus the 16-element solve
    history live in every thread.  That reaches the architectural 255-register
    ceiling and spills the state to local memory.  Stage S0 once in shared
    memory instead, finish the triangular solve, and then consume the state in
    short K-column strips after the solve temporaries are dead.
    """

    H = 16
    HV = 48
    K = 128
    V = 128
    W = 16
    WARPS = 4

    @cute.jit
    def __call__(
        self,
        h0: cute.Tensor,
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        flush_list: cute.Tensor,
        n_ptr: cute.Tensor,
        ls6_mh: cute.Tensor,
        ls6_beta: cute.Tensor,
        ls6_map: cute.Tensor,
        stream: CUstream,
    ):
        self.kernel(
            h0,
            d_cache,
            k_cache,
            g_cache,
            flush_list,
            n_ptr,
            ls6_mh,
            ls6_beta,
            ls6_map,
        ).launch(
            grid=((flush_list.shape[0] - 1) * self.HV, 1, 1),
            block=(self.WARPS * 32, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        h0: cute.Tensor,
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        flush_list: cute.Tensor,
        n_ptr: cute.Tensor,
        ls6_mh: cute.Tensor,
        ls6_beta: cute.Tensor,
        ls6_map: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        work, _, _ = cute.arch.block_idx()
        rr = work // self.HV
        ihv = work - rr * self.HV
        ih = ihv // (self.HV // self.H)
        sidx = flush_list[rr]
        cidx = ls6_map[sidx]

        smem = cutlass.utils.SmemAllocator()
        swizzle = cute.make_swizzle(3, 4, 3)
        h_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.V, (32, self.K // 32)),
                stride=(32, (1, self.V * 32)),
            ),
        )
        k_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.W, (32, self.K // 32)),
                stride=(32, (1, self.W * 32)),
            ),
        )
        s_h0 = smem.allocate_tensor(
            Float32,
            h_layout.outer,
            byte_alignment=128,
            swizzle=h_layout.inner,
        )
        s_k = smem.allocate_tensor(
            Float32,
            k_layout.outer,
            byte_alignment=128,
            swizzle=k_layout.inner,
        )
        s_d = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.V, self.W), stride=(self.W, 1)),
            byte_alignment=128,
        )
        s_gram = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.W, self.W), stride=(self.W, 1)),
            byte_alignment=16,
        )
        s_prefix = smem.allocate_array(Float32, self.W)
        s_beta = smem.allocate_array(Float32, self.W)
        mma_bar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mma_bar, 1)
                cute.arch.mbarrier_init_fence()
            _tcgen05.alloc(taddr)

        # One thread owns one V row.  This is the sole global S0 read; shared
        # memory is both the tensor-core operand and the later fold source.
        # Runtime loops are intentional here.  Fully unrolling either copy or
        # all sixteen output strips lets the scheduler keep 128 independent
        # values live and recreates the same spill problem we are avoiding.
        for k in cutlass.range(self.K):
            value = h0[sidx, ihv, tid, k]
            s_h0[tid, (k % 32, k // 32)] = value

        for q in cutlass.range_constexpr(self.W * self.K // 128):
            x = tid + q * 128
            s = x // self.K
            k = x - s * self.K
            s_k[s, (k % 32, k // 32)] = k_cache[sidx, ih, s, k]
        for q in cutlass.range_constexpr(self.W * self.V // 128):
            x = tid + q * 128
            s = x // self.V
            v = x - s * self.V
            s_d[v, s] = d_cache[sidx, ihv, s, v]

        if tid < self.W:
            prefix = Float32(0.0)
            for j in cutlass.range(self.W):
                if j <= tid:
                    prefix += g_cache[sidx, ihv, j]
            s_prefix[tid] = prefix
            s_beta[tid] = ls6_beta[cidx, ihv, tid]

        cute.arch.sync_threads()

        if warp == 0:
            idesc = _tcgen05.make_tf32_idesc(self.V, self.W)
            desc_template = _tcgen05.make_sdesc_128B_swizzle(self.V * 128)
            h_base = desc_template | (s_h0[None, None].iterator.toint() >> 4)
            k_base = desc_template | (s_k[None, None].iterator.toint() >> 4)
            for kb in cutlass.range_constexpr(self.K // 32):
                for ki in cutlass.range_constexpr(32 // 8):
                    h_desc = h_base | ((kb * self.V * 128 + ki * 32) >> 4)
                    k_desc = k_base | ((kb * self.W * 128 + ki * 32) >> 4)
                    _tcgen05.mma_tf32(0, h_desc, k_desc, idesc, kb > 0 or ki > 0)
            _tcgen05.commit(mma_bar)

        # Gram work overlaps the asynchronous tensor-core product.
        for q in cutlass.range_constexpr(self.W * self.W // 128):
            x = tid + q * 128
            j = x // self.W
            s = x - j * self.W
            dot = Float32(0.0)
            if j < s:
                for k in cutlass.range_constexpr(self.K):
                    dot += s_k[j, (k % 32, k // 32)] * s_k[
                        s, (k % 32, k // 32)
                    ]
                dot *= cute.math.exp(s_prefix[s] - s_prefix[j], fastmath=True)
            s_gram[j, s] = dot

        cute.arch.sync_threads()
        if warp == 0:
            cute.arch.mbarrier_wait(mma_bar, 0)
        cute.arch.sync_threads()
        _tcgen05.fence_after_thread_sync()

        p = _tcgen05.ld(warp * 32, 0, "32x32b", self.W)
        _tcgen05.wait_ld()
        hist = cute.make_rmem_tensor(self.W, Float32)
        total = s_prefix[self.W - 1]
        for s in cutlass.range_constexpr(self.W):
            prev = Float32(0.0)
            for j in cutlass.range_constexpr(self.W):
                if j < s:
                    prev += hist[j] * s_beta[s] * s_gram[j, s]
            h = (
                s_beta[s]
                * cute.math.exp(s_prefix[s], fastmath=True)
                * p[s]
                - prev
            )
            hist[s] = h
            corr = s_d[tid, s] - h
            d_cache[sidx, ihv, s, tid] = corr
            s_d[tid, s] = corr * cute.math.exp(total - s_prefix[s], fastmath=True)

        cute.arch.sync_threads()

        # Process only eight columns at a time.  The P/history registers are
        # dead here, so this stays well below the spill threshold while still
        # exposing enough independent FMAs to hide shared-memory latency.
        total_scale = cute.math.exp(total, fastmath=True)
        for kb in cutlass.range(self.K // 8):
            acc = cute.make_rmem_tensor(8, Float32)
            for ki in cutlass.range_constexpr(8):
                k = kb * 8 + ki
                acc[ki] = (
                    s_h0[tid, (k % 32, k // 32)] * total_scale
                )
            for s in cutlass.range(self.W):
                dv = s_d[tid, s]
                for ki in cutlass.range_constexpr(8):
                    k = kb * 8 + ki
                    acc[ki] += dv * s_k[s, (k % 32, k // 32)]
            for ki in cutlass.range_constexpr(8):
                k = kb * 8 + ki
                h0[sidx, ihv, tid, k] = acc[ki]

        cute.arch.sync_threads()
        if warp == 0:
            _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile():
        nx = cute.sym_int()
        ns = cute.sym_int()
        bf = cute.sym_int()
        h0 = make_fake_tensor(Float32, (nx, 48, 128, 128), divisibility=16)
        d = make_fake_tensor(Float32, (nx, 48, 16, 128), divisibility=16)
        k = make_fake_tensor(Float32, (nx, 16, 16, 128), divisibility=16)
        g = make_fake_tensor(Float32, (nx, 48, 16), divisibility=4)
        fl = make_fake_tensor(Int32, (bf,), divisibility=1)
        nptr = make_fake_tensor(Int32, (1,), divisibility=1)
        mh = make_fake_tensor(Int32, (48,), divisibility=1)
        beta = make_fake_tensor(Float32, (ns, 48, 16), divisibility=4)
        mapping = make_fake_tensor(Int32, (nx,), divisibility=1)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm100ExactGdnFold4(),
            h0,
            d,
            k,
            g,
            fl,
            nptr,
            mh,
            beta,
            mapping,
            stream,
            options="--enable-tvm-ffi",
        )


class Sm100ExactGdnFused:
    """Role-specialized 2KV exact refresh for Qwen on SM100.

    Warpgroup 0 owns the Replay16 state tile in registers.  Warpgroup 1 owns
    the native tensor-core checkpoint projection and causal correction.  The
    two groups meet only through corrected, replay-scaled deltas in shared
    memory, so the extra exactness work runs between the baseline checkpoint
    load and store without a second HBM state read.
    """

    H = 16
    HV = 48
    K = 128
    V = 128
    W = 16
    WARPS = 8
    TV = 8
    TK = 16

    @cute.jit
    def __call__(
        self,
        h0: cute.Tensor,
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        flush_list: cute.Tensor,
        n_ptr: cute.Tensor,
        ls6_mh: cute.Tensor,
        ls6_beta: cute.Tensor,
        ls6_map: cute.Tensor,
        stream: CUstream,
    ):
        self.kernel(
            h0,
            d_cache,
            k_cache,
            g_cache,
            flush_list,
            n_ptr,
            ls6_mh,
            ls6_beta,
            ls6_map,
        ).launch(
            grid=((flush_list.shape[0] - 1) * self.HV, 1, 1),
            block=(self.WARPS * 32, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        h0: cute.Tensor,
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        flush_list: cute.Tensor,
        n_ptr: cute.Tensor,
        ls6_mh: cute.Tensor,
        ls6_beta: cute.Tensor,
        ls6_map: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        work, _, _ = cute.arch.block_idx()
        rr = work // self.HV
        ihv = work - rr * self.HV
        ih = ihv // (self.HV // self.H)
        sidx = flush_list[rr]
        cidx = ls6_map[sidx]

        smem = cutlass.utils.SmemAllocator()
        swizzle = cute.make_swizzle(3, 4, 3)
        h_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.V, (32, self.K // 32)),
                stride=(32, (1, self.V * 32)),
            ),
        )
        k_layout = cute.make_composed_layout(
            swizzle,
            0,
            cute.make_layout(
                (self.W, (32, self.K // 32)),
                stride=(32, (1, self.W * 32)),
            ),
        )
        s_h0 = smem.allocate_tensor(
            Float32,
            h_layout.outer,
            byte_alignment=128,
            swizzle=h_layout.inner,
        )
        s_k = smem.allocate_tensor(
            Float32,
            k_layout.outer,
            byte_alignment=128,
            swizzle=k_layout.inner,
        )
        s_d = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.V, self.W), stride=(self.W, 1)),
            byte_alignment=128,
        )
        s_gram = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.W, self.W), stride=(self.W, 1)),
            byte_alignment=16,
        )
        s_prefix = smem.allocate_array(Float32, self.W)
        s_beta = smem.allocate_array(Float32, self.W)
        mma_bar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        if warp == 4:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mma_bar, 1)
                cute.arch.mbarrier_init_fence()
            _tcgen05.alloc(taddr)
        cute.arch.sync_threads()
        # Keep the state tensor scoped to this staged branch.  Named barriers
        # coordinate the two warpgroups without forcing the DSL to allocate
        # its 128 accumulators in the correction warps too.
        if warp < 4:
            cute.arch.setmaxregister_increase(248)
            local = tid
            tk = local % self.TK
            tv = local // self.TK
            k0 = tk * (self.K // self.TK)
            v0 = tv * (self.V // self.TV)
            state = cute.make_rmem_tensor(
                (self.V // self.TV, self.K // self.TK), Float32
            )
            # Warpgroup 1 owns the sole global checkpoint copy.  Wait for it,
            # then populate the long-lived fold accumulators from shared while
            # that group runs tcgen05 and the triangular solve.
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
            for vi in cutlass.range_constexpr(self.V // self.TV):
                for ki in cutlass.range_constexpr(self.K // self.TK):
                    v = v0 + vi
                    k = k0 + ki
                    state[vi, ki] = s_h0[v, (k % 32, k // 32)]
            cute.arch.barrier(barrier_id=2, number_of_threads=256)

            total_scale = cute.math.exp(
                s_prefix[self.W - 1], fastmath=True
            )
            for vi in cutlass.range_constexpr(self.V // self.TV):
                for ki in cutlass.range_constexpr(self.K // self.TK):
                    state[vi, ki] *= total_scale
            for s in cutlass.range(self.W):
                dv = cute.make_rmem_tensor(self.V // self.TV, Float32)
                kk = cute.make_rmem_tensor(self.K // self.TK, Float32)
                for vi in cutlass.range_constexpr(self.V // self.TV):
                    dv[vi] = s_d[v0 + vi, s]
                for ki in cutlass.range_constexpr(self.K // self.TK):
                    k = k0 + ki
                    kk[ki] = s_k[s, (k % 32, k // 32)]
                for vi in cutlass.range_constexpr(self.V // self.TV):
                    for ki in cutlass.range_constexpr(self.K // self.TK):
                        state[vi, ki] += dv[vi] * kk[ki]
            for vi in cutlass.range_constexpr(self.V // self.TV):
                for ki in cutlass.range_constexpr(self.K // self.TK):
                    h0[sidx, ihv, v0 + vi, k0 + ki] = state[vi, ki]
            cute.arch.barrier(barrier_id=3, number_of_threads=256)
        else:
            # P (16), solve history (16), and address temporaries fit under
            # 160; the released registers are borrowed by the state group.
            cute.arch.setmaxregister_decrease(160)
            local = tid - 128
            # Sole global S0 read.  The same shared tile feeds both tcgen05
            # and the register-resident Replay16 fold.
            for q in cutlass.range_constexpr(self.V * self.K // 128):
                x = local + q * 128
                v = x // self.K
                k = x - v * self.K
                s_h0[v, (k % 32, k // 32)] = h0[sidx, ihv, v, k]
            for q in cutlass.range_constexpr(self.W * self.K // 128):
                x = local + q * 128
                s = x // self.K
                k = x - s * self.K
                s_k[s, (k % 32, k // 32)] = k_cache[sidx, ih, s, k]
            for q in cutlass.range_constexpr(self.W * self.V // 128):
                x = local + q * 128
                s = x // self.V
                v = x - s * self.V
                s_d[v, s] = d_cache[sidx, ihv, s, v]
            if local < self.W:
                prefix = Float32(0.0)
                for j in cutlass.range(self.W):
                    if j <= local:
                        prefix += g_cache[sidx, ihv, j]
                s_prefix[local] = prefix
                s_beta[local] = ls6_beta[cidx, ihv, local]

            cute.arch.barrier(barrier_id=1, number_of_threads=256)

            if warp == 4:
                idesc = _tcgen05.make_tf32_idesc(self.V, self.W)
                desc_template = _tcgen05.make_sdesc_128B_swizzle(self.V * 128)
                h_base = desc_template | (
                    s_h0[None, None].iterator.toint() >> 4
                )
                k_base = desc_template | (
                    s_k[None, None].iterator.toint() >> 4
                )
                for kb in cutlass.range_constexpr(self.K // 32):
                    for ki in cutlass.range_constexpr(32 // 8):
                        h_desc = h_base | (
                            (kb * self.V * 128 + ki * 32) >> 4
                        )
                        k_desc = k_base | (
                            (kb * self.W * 128 + ki * 32) >> 4
                        )
                        _tcgen05.mma_tf32(
                            0, h_desc, k_desc, idesc, kb > 0 or ki > 0
                        )
                _tcgen05.commit(mma_bar)

            for q in cutlass.range_constexpr(2):
                x = local + q * 128
                if x < self.W * self.W:
                    j = x // self.W
                    s = x - j * self.W
                    dot = Float32(0.0)
                    if j < s:
                        for k in cutlass.range_constexpr(self.K):
                            dot += s_k[j, (k % 32, k // 32)] * s_k[
                                s, (k % 32, k // 32)
                            ]
                        dot *= cute.math.exp(
                            s_prefix[s] - s_prefix[j], fastmath=True
                        )
                    s_gram[j, s] = dot

            cute.arch.barrier(barrier_id=4, number_of_threads=128)
            if warp == 4:
                cute.arch.mbarrier_wait(mma_bar, 0)
            cute.arch.barrier(barrier_id=5, number_of_threads=128)
            _tcgen05.fence_after_thread_sync()

            p = _tcgen05.ld((warp - 4) * 32, 0, "32x32b", self.W)
            _tcgen05.wait_ld()
            v = (warp - 4) * 32 + (tid % 32)
            hist = cute.make_rmem_tensor(self.W, Float32)
            total = s_prefix[self.W - 1]
            for s in cutlass.range_constexpr(self.W):
                prev = Float32(0.0)
                for j in cutlass.range_constexpr(self.W):
                    if j < s:
                        prev += hist[j] * s_beta[s] * s_gram[j, s]
                h = (
                    s_beta[s]
                    * cute.math.exp(s_prefix[s], fastmath=True)
                    * p[s]
                    - prev
                )
                hist[s] = h
                corr = s_d[v, s] - h
                d_cache[sidx, ihv, s, v] = corr
                s_d[v, s] = corr * cute.math.exp(
                    total - s_prefix[s], fastmath=True
                )

            cute.arch.barrier(barrier_id=2, number_of_threads=256)
            cute.arch.barrier(barrier_id=3, number_of_threads=256)
            if warp == 4:
                _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile():
        nx = cute.sym_int()
        ns = cute.sym_int()
        bf = cute.sym_int()
        h0 = make_fake_tensor(Float32, (nx, 48, 128, 128), divisibility=16)
        d = make_fake_tensor(Float32, (nx, 48, 16, 128), divisibility=16)
        k = make_fake_tensor(Float32, (nx, 16, 16, 128), divisibility=16)
        g = make_fake_tensor(Float32, (nx, 48, 16), divisibility=4)
        fl = make_fake_tensor(Int32, (bf,), divisibility=1)
        nptr = make_fake_tensor(Int32, (1,), divisibility=1)
        mh = make_fake_tensor(Int32, (48,), divisibility=1)
        beta = make_fake_tensor(Float32, (ns, 48, 16), divisibility=4)
        mapping = make_fake_tensor(Int32, (nx,), divisibility=1)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm100ExactGdnFused(),
            h0,
            d,
            k,
            g,
            fl,
            nptr,
            mh,
            beta,
            mapping,
            stream,
            options="--enable-tvm-ffi",
        )


class Sm100ExactGdnCorrection:
    """Native tcgen05 P=S0 K^T followed by the FP32 16-step solve.

    This kernel deliberately stops before the Replay16 state fold.  It is the
    compact, spill-free half of the two-pass design: the mature CUDA fold then
    consumes the corrected deltas.  Keeping the second skinny GEMM out also
    cuts shared memory enough for two CTAs per SM.
    """

    H = 16
    HV = 48
    K = 128
    V = 128
    W = 16
    WARPS = 10

    @cute.jit
    def _make_tile_tma(self, tensor: cute.Tensor, rows: cutlass.Constexpr[int]):
        num_elems = 128 // (tensor.element_type.width // 8)
        swizzle = cute.make_swizzle(3, 4, 3)
        layout = cute.make_layout(
            (1, 1, rows, (num_elems, self.K // num_elems)),
            stride=(0, 0, num_elems, (1, rows * num_elems)),
        )
        layout = cute.make_composed_layout(swizzle, 0, layout)
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(tensor, (None, None, None, num_elems)),
            layout,
            cta_tiler=(1, 1, rows, self.K),
        )

    @cute.jit
    def __call__(
        self,
        h0: cute.Tensor,
        d_cache: cute.Tensor,
        k_cache: cute.Tensor,
        g_cache: cute.Tensor,
        flush_list: cute.Tensor,
        n_ptr: cute.Tensor,
        ls6_mh: cute.Tensor,
        ls6_beta: cute.Tensor,
        ls6_map: cute.Tensor,
        stream: CUstream,
    ):
        h0_tma = self._make_tile_tma(h0, self.V)
        k_tma = self._make_tile_tma(k_cache, self.W)
        self.kernel(
            h0,
            h0_tma,
            k_tma,
            d_cache,
            g_cache,
            flush_list,
            n_ptr,
            ls6_mh,
            ls6_beta,
            ls6_map,
        ).launch(
            grid=((flush_list.shape[0] - 1) * self.HV, 1, 1),
            block=(self.WARPS * 32, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        h0: cute.Tensor,
        h0_tma: cpasync.TmaInfo,
        k_tma: cpasync.TmaInfo,
        d_cache: cute.Tensor,
        g_cache: cute.Tensor,
        flush_list: cute.Tensor,
        n_ptr: cute.Tensor,
        ls6_mh: cute.Tensor,
        ls6_beta: cute.Tensor,
        ls6_map: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        work, _, _ = cute.arch.block_idx()
        rr = work // self.HV
        ihv = work - rr * self.HV
        ih = ihv // (self.HV // self.H)
        sidx = flush_list[rr]
        cidx = ls6_map[sidx]

        smem = cutlass.utils.SmemAllocator()

        def alloc_tma(info):
            return smem.allocate_tensor(
                Float32,
                info.smem_layout.outer,
                byte_alignment=128,
                swizzle=info.smem_layout.inner,
            )[0, 0, None, None]

        s_h0 = alloc_tma(h0_tma)
        s_k = alloc_tma(k_tma)
        s_gram = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.W, self.W), stride=(self.W, 1)),
            byte_alignment=16,
        )
        s_prefix = smem.allocate_array(Float32, self.W)
        s_beta = smem.allocate_array(Float32, self.W)
        tma_bar = smem.allocate_array(Int64, 1)
        mma_bar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tma_bar, 1)
                cute.arch.mbarrier_init(mma_bar, 1)
                cute.arch.mbarrier_init_fence()
        elif warp == 5:
            cpasync.prefetch_descriptor(h0_tma.atom)
            cpasync.prefetch_descriptor(k_tma.atom)
        cute.arch.sync_threads()

        if warp == 4:
            _tcgen05.alloc(taddr)
        if warp == 5:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    tma_bar, (self.V + self.W) * self.K * 4
                )
            simple_tma_copy(
                h0_tma.atom,
                h0_tma.tma_tensor[sidx, ihv, None, None],
                s_h0,
                tma_bar,
            )
            simple_tma_copy(
                k_tma.atom,
                k_tma.tma_tensor[sidx, ih, None, None],
                s_k,
                tma_bar,
            )
        if warp == 0:
            cute.arch.mbarrier_wait(tma_bar, 0)
        cute.arch.sync_threads()

        if tid < self.W:
            prefix = Float32(0.0)
            for j in cutlass.range(self.W):
                if j <= tid:
                    prefix += g_cache[sidx, ihv, j]
            s_prefix[tid] = prefix
            s_beta[tid] = ls6_beta[cidx, ihv, tid]

        if warp == 4:
            idesc = _tcgen05.make_tf32_idesc(self.V, self.W)
            desc_template = _tcgen05.make_sdesc_128B_swizzle(self.V * 128)
            h_base = desc_template | (s_h0[None, None].iterator.toint() >> 4)
            k_base = desc_template | (s_k[None, None].iterator.toint() >> 4)
            for kb in cutlass.range_constexpr(self.K // 32):
                for ki in cutlass.range_constexpr(32 // 8):
                    h_desc = h_base | ((kb * self.V * 128 + ki * 32) >> 4)
                    k_desc = k_base | ((kb * self.W * 128 + ki * 32) >> 4)
                    _tcgen05.mma_tf32(
                        0, h_desc, k_desc, idesc, kb > 0 or ki > 0
                    )
            _tcgen05.commit(mma_bar)

        # The Gram matrix is independent of V and overlaps the asynchronous
        # checkpoint MMA.  Invalid/lower entries are explicitly zero.
        for q in cutlass.range_constexpr(2):
            x = tid + q * self.WARPS * 32
            if x < self.W * self.W:
                j = x // self.W
                s = x - j * self.W
                dot = Float32(0.0)
                if j < s:
                    for k in cutlass.range_constexpr(self.K):
                        dot += s_k[j, (k % 32, k // 32)] * s_k[
                            s, (k % 32, k // 32)
                        ]
                    dot *= cute.math.exp(
                        s_prefix[s] - s_prefix[j], fastmath=True
                    )
                s_gram[j, s] = dot

        cute.arch.sync_threads()
        if warp == 0:
            cute.arch.mbarrier_wait(mma_bar, 0)
        cute.arch.sync_threads()

        if warp < 4:
            _tcgen05.fence_after_thread_sync()
            p = _tcgen05.ld(warp * 32, 0, "32x32b", self.W)
            _tcgen05.wait_ld()
            v = warp * 32 + lane
            hist = cute.make_rmem_tensor(self.W, Float32)
            for s in cutlass.range_constexpr(self.W):
                prev = Float32(0.0)
                for j in cutlass.range_constexpr(self.W):
                    if j < s:
                        prev += hist[j] * s_beta[s] * s_gram[j, s]
                h = (
                    s_beta[s]
                    * cute.math.exp(s_prefix[s], fastmath=True)
                    * p[s]
                    - prev
                )
                hist[s] = h
                d_cache[sidx, ihv, s, v] -= h

        cute.arch.sync_threads()
        if warp == 4:
            _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile():
        nx = cute.sym_int()
        ns = cute.sym_int()
        bf = cute.sym_int()
        h0 = make_fake_tensor(Float32, (nx, 48, 128, 128), divisibility=16)
        d = make_fake_tensor(Float32, (nx, 48, 16, 128), divisibility=16)
        k = make_fake_tensor(Float32, (nx, 16, 16, 128), divisibility=16)
        g = make_fake_tensor(Float32, (nx, 48, 16), divisibility=4)
        fl = make_fake_tensor(Int32, (bf,), divisibility=1)
        nptr = make_fake_tensor(Int32, (1,), divisibility=1)
        mh = make_fake_tensor(Int32, (48,), divisibility=1)
        beta = make_fake_tensor(Float32, (ns, 48, 16), divisibility=4)
        mapping = make_fake_tensor(Int32, (nx,), divisibility=1)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            Sm100ExactGdnCorrection(),
            h0,
            d,
            k,
            g,
            fl,
            nptr,
            mh,
            beta,
            mapping,
            stream,
            options="--enable-tvm-ffi",
        )


def gdn_exact_fold_tcgen(
    h0: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    flush_list: torch.Tensor,
    n_ptr: torch.Tensor,
    ls6_mh: torch.Tensor,
    ls6_beta: torch.Tensor,
    ls6_map: torch.Tensor,
) -> None:
    Sm100ExactGdnFused.compile()(
        h0,
        d_cache,
        k_cache,
        g_cache,
        flush_list,
        n_ptr,
        ls6_mh,
        ls6_beta,
        ls6_map,
    )


def gdn_exact_fold_tcgen_old(
    h0: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    flush_list: torch.Tensor,
    n_ptr: torch.Tensor,
    ls6_mh: torch.Tensor,
    ls6_beta: torch.Tensor,
    ls6_map: torch.Tensor,
    profile_mode: int = 0,
) -> None:
    Sm100ExactGdnFold.compile(profile_mode)(
        h0,
        d_cache,
        k_cache,
        g_cache,
        flush_list,
        n_ptr,
        ls6_mh,
        ls6_beta,
        ls6_map,
    )


def gdn_exact_correction_tcgen(
    h0: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    flush_list: torch.Tensor,
    n_ptr: torch.Tensor,
    ls6_mh: torch.Tensor,
    ls6_beta: torch.Tensor,
    ls6_map: torch.Tensor,
) -> None:
    Sm100ExactGdnCorrection.compile()(
        h0,
        d_cache,
        k_cache,
        g_cache,
        flush_list,
        n_ptr,
        ls6_mh,
        ls6_beta,
        ls6_map,
    )
