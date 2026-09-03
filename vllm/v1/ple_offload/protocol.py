# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC message definitions for PLE CPU offload."""

from dataclasses import dataclass

import msgspec
import torch

# ---------------------------------------------------------------------------
# IPC message dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PleOffloadRegistration:
    """Sent once from each GPU worker during offload setup."""

    worker_id: int
    tp_rank: int
    dp_rank: int
    # CUDA tensors are serialized through PyTorch CUDA IPC.
    gpu_output_buffers: dict[str, torch.Tensor]
    sem_flag_tensors: dict[str, torch.Tensor]
    # CPU tensors are allocated in shared memory and registered once.
    input_ids_buf: torch.Tensor
    query_start_loc_buf: torch.Tensor
    ngram_context_buf: torch.Tensor | None
    # CPU shared-memory sequence acknowledged after the offload worker has
    # snapshotted a request's inputs.  The connector must not overwrite the
    # buffers above until this reaches the preceding request id.
    input_ack_buf: torch.Tensor


@dataclass
class PleOffloadRequest:
    """Sent by each DP rank's TP rank zero at every inference step."""

    dp_rank: int
    num_tokens: int
    num_reqs: int
    # Monotonic per-DP sequence used to protect the single shared input slot.
    request_id: int = 0
    # Index into the connector's input-readiness event ring. Local to the
    # requesting worker; the CPU offload process ignores it.
    event_idx: int = 0


_PLE_OFFLOAD_REQUEST_DECODER = msgspec.msgpack.Decoder(PleOffloadRequest)
