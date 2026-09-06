# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research batched solvers on identical GDN Gram scratch and Phi layouts.

Compile gdn_solver_candidates.cu with MathDx first. Timings include packing,
all library calls, and Phi writes; setup and correctness checks are excluded.
"""

import argparse
import ctypes
import json
from pathlib import Path

import torch

from vllm.third_party.flash_linear_attention.ops.gdn_flush_full_cuda import (
    FlushWorkspace,
)


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
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 32)
    return sorted(samples)[2]


class Candidate:
    def __init__(self, library, m, g, batch, hv=48):
        self.m, self.g, self.n, self.hv = m, g, batch * hv, hv
        p, i = ctypes.c_void_p, ctypes.c_int
        self.lib = ctypes.CDLL(str(library))
        self.lib.create_context.argtypes = []
        self.lib.create_context.restype = p
        self.lib.destroy_context.argtypes = [p]
        self.lib.destroy_context.restype = None
        self.lib.library_solve.argtypes = [p] + [i] * 5 + [p] * 13
        self.lib.library_solve.restype = i
        self.lib.dx_solve.argtypes = [i, i, p, p, i, i, i] + [p] * 5
        self.lib.dx_solve.restype = i
        self.gj = ctypes.CDLL(str(library.parent / "libgdn_solver_gauss_jordan.so"))
        self.gj.gj_solve.argtypes = [i, p, p, i, i, i] + [p] * 5
        self.gj.gj_solve.restype = i
        self.context = self.lib.create_context()
        if not self.context:
            raise RuntimeError("CUDA library handle creation failed")
        self.a = torch.empty(self.n, m, m, device="cuda")
        self.b = torch.empty(self.n, 128 - m, m, device="cuda")
        self.inv, self.x = torch.empty_like(self.a), torch.empty_like(self.b)
        self.pointers = torch.empty(4 * self.n, device="cuda", dtype=torch.int64)
        self.info = torch.zeros(self.n, device="cuda", dtype=torch.int32)
        self.pivots = torch.empty(self.n, m, device="cuda", dtype=torch.int32)

    def close(self):
        if self.context:
            self.lib.destroy_context(self.context)
            self.context = None

    def run(self, method, scratch, phi, rows=None, mapping=None, widths=None):
        ptr = lambda x: None if x is None else x.data_ptr()
        stream = torch.cuda.current_stream().cuda_stream
        if method.startswith("cusolverdx"):
            threads = {
                "cusolverdx": 128,
                "cusolverdx_auto": 0,
                "cusolverdx_padded": -1,
            }[method]
            status = self.lib.dx_solve(
                self.m,
                threads,
                ptr(scratch),
                ptr(phi),
                self.g,
                self.n,
                self.hv,
                ptr(rows),
                ptr(mapping),
                ptr(widths),
                ptr(self.info),
                stream,
            )
        elif method == "gauss_jordan":
            status = self.gj.gj_solve(
                self.m,
                ptr(scratch),
                ptr(phi),
                self.g,
                self.n,
                self.hv,
                ptr(rows),
                ptr(mapping),
                ptr(widths),
                ptr(self.info),
                stream,
            )
        else:
            variant = {
                "potrf_trsm": 0,
                "lu_trsm": 1,
                "matinv_gemm": 2,
                "lu_inverse_gemm": 3,
            }[method]
            status = self.lib.library_solve(
                self.context,
                variant,
                self.m,
                self.g,
                self.n,
                self.hv,
                *map(
                    ptr,
                    (
                        scratch,
                        phi,
                        self.a,
                        self.b,
                        self.inv,
                        self.x,
                        self.pointers,
                        self.info,
                        self.pivots,
                        rows,
                        mapping,
                        widths,
                    ),
                ),
                stream,
            )
        if status:
            raise RuntimeError(f"{method} returned status {status}")


def make_case(m, batch=128, g=None):
    torch.manual_seed(194)
    h, hv, k, w = 16, 48, 128, 16
    g = (m + 3) // 4 * 4 if g is None else g
    ns = batch + 1
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
    workspace = FlushWorkspace(batch, h, hv, g, "cuda", ridge=0.1)
    args = (state, writes, keys, gates, rows, mapping, widths, beta, u, phi)
    workspace.flush(*args)
    return workspace, args


def run_case(library, m, batch=128, g=None, methods=None):
    workspace, args = make_case(m, batch, g)
    g = workspace.g
    candidate = Candidate(library, m, g, batch)
    baseline = args[-1].clone()
    sc = workspace.scratch.reshape(batch * 48, 128, g)
    a, b = sc[:, :m, :m], sc[:, m:, :m].transpose(-1, -2)
    oracle = torch.linalg.solve(a[:16].double(), b[:16].double())
    results = []
    if methods is None:
        methods = ["current", "potrf_trsm", "lu_trsm", "lu_inverse_gemm", "cusolverdx"]
        if m <= 32:
            methods.insert(3, "matinv_gemm")
    for method in methods:
        out = torch.full_like(args[-1], float("nan"))
        if method == "current":
            call = lambda out=out: workspace._run(3, *args[:-1], out)
        else:
            call = lambda method=method, out=out: candidate.run(
                method, workspace.scratch, out, args[4], args[5], args[6]
            )
        record = dict(method=method, m=m, g=g, batch=batch, systems=batch * 48)
        try:
            call()
            torch.accelerator.synchronize()
            got = out[1:].reshape(batch * 48, g, 128)[:, :m, m:]
            torch.testing.assert_close(out[1:], baseline[1:], atol=5e-5, rtol=4e-3)
            torch.testing.assert_close(got[:16].double(), oracle, atol=5e-5, rtol=4e-3)
            denominator = torch.linalg.vector_norm(
                a, dim=(-2, -1)
            ) * torch.linalg.vector_norm(got, dim=(-2, -1)) + torch.linalg.vector_norm(
                b, dim=(-2, -1)
            )
            residual = torch.linalg.vector_norm(a @ got - b, dim=(-2, -1)) / denominator
            record.update(
                us=measure(call),
                max_abs_vs_current=(out[1:] - baseline[1:]).abs().max().item(),
                max_abs_vs_fp64=(got[:16].double() - oracle).abs().max().item(),
                max_backward_residual=residual.max().item(),
                info_max=candidate.info.abs().max().item()
                if method != "current"
                else 0,
            )
        except (RuntimeError, AssertionError) as error:
            record["error"] = str(error)
        print(json.dumps(record), flush=True)
        results.append(record)
    candidate.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--widths", type=int, nargs="+", default=[8, 16, 32, 33, 64, 80, 96, 127]
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--allocation-g", type=int)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    for m in args.widths:
        results.extend(
            run_case(args.library, m, args.batch, args.allocation_g, args.methods)
        )
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
