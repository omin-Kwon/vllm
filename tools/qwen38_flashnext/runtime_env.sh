#!/usr/bin/env bash
# Portable runtime envelope for Qwen3.8-Flash-Next NVFP4 with PLE CPU offload.
# Source this file after optionally setting Q38NEXT_ENV or Q38NEXT_PYTHON.

set -euo pipefail

Q38NEXT_VLLM_REPO=${Q38NEXT_VLLM_REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)}

if [[ -z "${Q38NEXT_PYTHON:-}" ]]; then
    for candidate in \
        "${Q38NEXT_ENV:+${Q38NEXT_ENV}/bin/python}" \
        "${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}" \
        "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}" \
        "$(command -v python 2>/dev/null || true)"; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            Q38NEXT_PYTHON=${candidate}
            break
        fi
    done
fi
[[ -n "${Q38NEXT_PYTHON:-}" && -x "${Q38NEXT_PYTHON}" ]] || {
    echo "Set Q38NEXT_PYTHON or Q38NEXT_ENV to the vLLM runtime environment." >&2
    return 2 2>/dev/null || exit 2
}

Q38NEXT_ENV=${Q38NEXT_ENV:-$(cd -- "$(dirname -- "${Q38NEXT_PYTHON}")/.." && pwd -P)}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH:-${CUDA_HOME}/bin/ptxas}
[[ -x "${TRITON_PTXAS_PATH}" ]] || {
    echo "Missing CUDA ptxas: ${TRITON_PTXAS_PATH}" >&2
    return 2 2>/dev/null || exit 2
}

export Q38NEXT_VLLM_REPO Q38NEXT_PYTHON Q38NEXT_ENV CUDA_HOME TRITON_PTXAS_PATH
export PATH="${Q38NEXT_ENV}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${Q38NEXT_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${Q38NEXT_VLLM_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_PLE_CPU_OFFLOAD=1
# The branch documents a remaining MRV2 PLE warmup hang. MRV1 is the validated
# path and supplies ReplaySSM's decode anchor, so do not silently use defaults.
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=${VLLM_PLE_OFFLOAD_READY_TIMEOUT:-1800}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}

# Prefer a native build produced by `pip install -e .`.  When this fresh clone
# is paired with an ABI-identical editable build in the selected environment,
# reuse its ignored build artifacts after a tracked-source hash check.
if [[ ! -f "${Q38NEXT_VLLM_REPO}/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so" \
      || ! -f "${Q38NEXT_VLLM_REPO}/vllm/third_party/flashmla/flash_mla_interface.py" ]]; then
    "${Q38NEXT_PYTHON}" \
        "${Q38NEXT_VLLM_REPO}/tools/qwen38_flashnext/bootstrap_native_artifacts.py" \
        --repo "${Q38NEXT_VLLM_REPO}"
fi
