# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed batches retain windowed decode and native FlashKDA prefill."""

from vllm.models.glm5next.common.kda import gather_initial_states, scatter_states


def window_attention(
    cache, state, metadata, q, k, v, gate, beta, a_log, bias, out, prefill
):
    """Route decode-first GDN metadata without handing decode rows to prefill.

    Inputs are post-convolution tensors [1, token, head, channel]. Only actual
    prefill rows can materialize/release window ownership. Continuing decode
    rows keep their ring, coefficients and positions even on mixed steps.
    """
    nd = metadata.num_decodes
    if metadata.num_decode_tokens != nd:
        raise ValueError("Window decode requires one token per request")
    if metadata.num_spec_decodes:
        raise ValueError("Speculative batches are unsupported")
    if metadata.num_prefills:
        cache.mixed_decode_tokens += nd
        ids = metadata.prefill_state_indices
        initial = metadata.prefill_has_initial_state
        cu = metadata.prefill_query_start_loc
        assert ids is not None and initial is not None and cu is not None
        cache.before_prefill(state, ids, initial)
        initial_state = gather_initial_states(state, ids, initial)
        end = nd + metadata.num_prefill_tokens
        _, final_state = prefill(
            q=q[:, nd:end],
            k=k[:, nd:end],
            v=v[:, nd:end],
            g=gate[:, nd:end],
            beta=beta[:, nd:end],
            initial_state=initial_state,
            cu_seqlens=cu,
            out=out[:, nd:end],
        )
        scatter_states(state, final_state, ids)
    if nd:
        ids = metadata.non_spec_state_indices_tensor[:nd]
        out[:, :nd].copy_(
            cache.step(
                state,
                ids,
                q[0, :nd],
                k[0, :nd],
                v[0, :nd],
                gate[0, :nd],
                beta[0, :nd],
                a_log,
                bias,
            ).unsqueeze(0)
        )
