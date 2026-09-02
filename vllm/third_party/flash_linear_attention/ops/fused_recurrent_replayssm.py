# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

from __future__ import annotations

import os

import torch

from vllm.model_executor.layers.mamba.ops.replayssm_config import (
    get_replayssm_config,
)
from vllm.triton_utils import tl, triton


@triton.jit
def fused_recurrent_gated_delta_rule_replayssm_kernel(
    mixed_qkv, a, b, A_log, dt_bias, o, h0, ht,
    d_cache, k_cache, g_cache, ssm_state_indices, write_pos, scale,
    # [nested_ssm 2026-08-25] freeze: head 별 hot 폭과 두 접힘(u=읽기, z=갱신).
    # state 는 회전 좌표계에 있으므로 hot = 앞 nf_h 열이고, cold 타일은
    # h0 를 **안 읽는다** — 그게 read 절감의 실체다. cold 기여는 u/z 로 대체.
    fz_nf, fz_u, fz_z, stride_fz_slot: tl.constexpr,
    # FZ_V2 입력: 다음 윈도우의 q̄/k̄ (회전 좌표계, cold 마스크는 커널이 건다).
    # 슬롯별 (H,K) — 호스트가 EMA 로 만든다(고정 shape, state 를 안 읽는다).
    fz_qbar, fz_kbar, stride_fzb_slot: tl.constexpr,
    # [nested_ssm 2026-09-02] LS6 래치 — 논문 §4 eq:m-steploop.
    #   ls6_ubar (NX,HV,G,V)  Ū = Ũ M0⁻¹.  **flush 가 M0⁻¹ 을 접어 둔다** — 스텝에
    #                         (G,G) 역행렬이 없다(Super 커널과 같은 수법).
    #   ls6_z    (H,K,G)      기저 Z. 스텝은 앞 r 행만 쓰고(Ẑ) f_s 는 전 K 를 쓴다.
    #   ls6_zbar (H,K,G)      Z̄ = Z M0⁻¹. **M0⁻¹ 은 층 상수**라 호스트가 미리 접는다 —
    #                         flush 는 Ū = S_new·Z̄ 만 하면 되고 역행렬이 어디에도 없다.
    #   ls6_aq/ak(NX,HV,G)    앵커 Zᵀq̄ / Zᵀk̄
    #   ls6_fs   (NX,HV,W,G)  슬롯별 소거 기여 f_s — 트래픽 (15) 의 W·G 항
    #   ls6_mh   (HV,)        head 별 래치 수. **0 이면 dense fallback**(기본 모드)
    ls6_ubar, ls6_z, ls6_zbar, ls6_aq, ls6_ak, ls6_fs, ls6_mh,
    stride_ls6_u_slot: tl.constexpr, stride_ls6_fs_slot: tl.constexpr,
    stride_ls6_a_slot: tl.constexpr,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    # [nested_ssm 2026-08-25] state 의 head/v/k stride 를 **인자로** 받는다.
    # 하드코딩(i_hv*V*K + o_v*K + o_kt)이면 (HV,K,V) 전치 뷰를 넘길 수 없다.
    # Mamba2 커널이 stride 구동이라 '할당만 전치하고 view 를 그대로 넘기는'
    # 무수정 정합이 가능했다(KERNEL.md §1). 여기도 같은 자리를 만들어 둔다.
    # 지금은 (HV,V,K) 라 (V*K, K, 1) 이 들어와 **비트 동일**하다.
    stride_state_h: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_d_slot: tl.constexpr,
    stride_k_slot: tl.constexpr,
    stride_g_slot: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NK: tl.constexpr, BKT: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    # [nested_ssm 2026-08-25] tl.dot 정밀도 손잡이 — Mamba2 커널의 DOT_INPUT_PRECISION
    # 과 같은 규약. 기본 'tf32' 는 기존 동작이고, 'ieee' 로 올리면 torch 참조와
    # 타이트하게 대조된다. (참조 대조에서 나온 상대 1e-3 이 정확히 TF32 정밀도였다.)
    DOT_INPUT_PRECISION: tl.constexpr,
    FZ_ON: tl.constexpr,
    # [nested_ssm 2026-08-25] 분기 특화. Mamba2 KERNEL.md §9 와 같은 처방 —
    # flush 쪽 tl.dot 과 큰 타일이 레지스터를 잡아 **비-flush 호출의 점유율까지** 깎는다.
    # 0=둘 다(기존) / 1=비-flush 전용 / 2=flush 전용. 런처가 1,2 로 두 번 런치한다.
    BR: tl.constexpr,
    # PREF: h0 선발행(위 주석) / EVICT: state 로드에 evict_first
    PREF: tl.constexpr,
    EVICT: tl.constexpr,
    # [nested_ssm 2026-08-25] FZ_V2 — **flush 분기가 다음 윈도우의 u/z 를 직접 쓴다.**
    # Mamba2 KERNEL.md §7 과 같은 처방이다. v1(호스트가 1/W 로 상각 갱신)은 데이터
    # 의존 행 gather 때문에 eager 를 강제하고, Mamba2 에서 그게 e2e -33% 였다.
    # flush 는 어차피 state 전체를 재구성하므로 `b_h_new_c` 가 레지스터에 있는 동안
    # 접으면 **추가 state 읽기 0** 이다. 호스트 지분이 0 이 되어 CUDA graph 로 돌아온다.
    # GDN 은 접힘이 둘이라(u=읽기, z=갱신) q̄·k̄ 를 둘 다 받아 둘 다 누적한다.
    FZ_V2: tl.constexpr,
    # [nested_ssm 2026-09-02] LS6_ON=False 면 아래 블록은 **컴파일에서 사라진다** —
    # 기존 경로와 비트 단위로 같다. LS6_G 는 tl.arange 용 2의 거듭제곱 패딩이고
    # 실제 래치 수는 ls6_mh 가 head 마다 준다(0 = dense fallback).
    LS6_ON: tl.constexpr,
    LS6_G: tl.constexpr,
    LS6_R: tl.constexpr,
    # ⚠ **기본 False.** ReplaySSM 의 링은 delta-rule 갱신 d_s = β(v − αSk) 를 담으므로
    #   소거가 **이미 그 안에 있다**. 재구성이 S = total_decay·S₀ + Σ replay_decay_s·d_s k_sᵀ
    #   로 정확하고(실측 2.9e-16), 체크포인트 몫은 순수 스칼라 감쇠다. 그래서 래치가
    #   근사할 것은 S₀ᵀq̂ 뿐이고 소거 보정이 필요 없다 — 넣으면 **이중계산**이다
    #   (실측: 상대 2.5e-2, 빼면 5.4e-4 = tf32 정밀도).
    #   논문의 f_s(W·G 항)는 링이 원시 (k̂,v,β)를 담고 키를 M 으로 밀고 가는 정식화의
    #   것이다. 우리 링은 그 자리에 d_s 를 담는다. 다른 링 형태를 쓸 때를 위해 코드는
    #   남기되 기본은 끈다.
    LS6_ERASE: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V

    # Resolve the physical state slot; zero the output and bail for padded rows.
    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    # ⚠ [nested_ssm 2026-08-26] 여기에 **상한 가드를 넣었다가 뺐다.** 기록을 남긴다.
    #   동기: 하한(state_idx<=0)만 있고 상한이 없으며, 커널 인자에 슬롯 수조차 없어
    #     검사할 수단이 없다. 2026-08-25 에 `IndexError: index 258 / size 256` 이 실제로
    #     났고, 그때 수정은 파이썬 쪽 _gdnr 버퍼만 4096 으로 키운 것이라 커널의
    #     out-of-bounds 는 그대로 남았다(경보기만 뗀 셈).
    #   뺀 이유 둘:
    #     ① **전제가 실측과 다르다.** 실제 표 조건(aime25 240요청, max_new=65536)에서
    #        최고 슬롯 938 / 한계 939 로 **초과가 일어나지 않는다**.
    #     ② **구현이 위험하다.** NUM_SLOTS 를 `tl.constexpr` 로 넣었더니 워밍업의
    #        ssm_state.shape[0]=256 이 그래프에 박히고, 실제 디코드의 슬롯(최대 938)이
    #        전부 초과로 걸려 **정상 요청의 출력이 0** 이 되었다(a64 AIME25 0/240).
    #        런타임에 변하는 값을 constexpr 로 박으면 안 된다.
    #   다시 넣으려면: 슬롯 수를 **런타임 스칼라 인자**로 받고, 그래프 캡처 아래에서
    #     정상 슬롯이 안 걸리는지 먼저 증명할 것. 그리고 조용히 0 을 내놓는 대신
    #     호스트에서 시끄럽게 죽일 것 — 0 출력도 결국 조용한 오염이다.
    if state_idx <= 0:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty), mask=mask_v)
        return

    # Per-row buffer cursor and flush flag; valid (committed) cache positions.
    # vLLM: write_pos is per decode row (i_n), not per physical slot.
    b_write_pos = tl.load(write_pos + i_n).to(tl.int64)
    b_is_flush = b_write_pos == MAX_CACHE_LEN - 1
    if BR == 1 and b_is_flush:
        return
    if BR == 2 and not b_is_flush:
        return
    cache_valid = o_c < b_write_pos

    # Gate for the current token: decay g, its exp alpha, and the beta mixing weight.
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    alpha_val = tl.exp(g_val)
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    # Replay decay over the committed cache, from the cached per-step gates g.
    p_g_main = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + o_c
    b_g_all = tl.load(p_g_main, mask=cache_valid, other=0.0).to(tl.float32)
    b_g_prefix = tl.cumsum(b_g_all, axis=0)
    b_g_total = tl.sum(b_g_all, axis=0)
    b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
    b_total_decay = tl.exp(b_g_total)
    # ⚠ FZ_V2 도 nf_h 가 필요하다. flush 런치는 flush-exact 때문에 **FZ_ON=False** 로
    #   도는데, 그때 nf_h=K 로 떨어지면 flush 의 cold 마스크 `o_kt >= nf_h` 가 공집합이
    #   되어 u/z 에 **0 이 기록된다** — 다음 윈도우가 cold 기여 없이 돌고, 값은 그럴듯해
    #   보이므로 조용히 품질만 깎인다. (실측: 상대 0.48 로 갈렸다.)
    if FZ_ON or FZ_V2:
        nf_h = tl.load(fz_nf + i_hv).to(tl.int32)
    else:
        nf_h = K

    # Cached delta-rule update vectors d (K-independent), scaled by the replay decay.
    p_d_main = d_cache + state_idx * stride_d_slot + ((i_hv * MAX_CACHE_LEN + o_c[None, :]) * V + o_v[:, None])
    b_d_all = tl.load(p_d_main, mask=mask_v[:, None] & cache_valid[None, :], other=0).to(tl.float32)
    b_d_scaled_tc = (b_d_all * b_replay_decay[None, :]).to(p_o.dtype.element_ty)  # [BV, BC]

    # Current token value (for the delta-rule update).
    v_off = (2 * H * K) + i_hv * V + o_v
    b_v = tl.load(mixed_qkv + i_n * stride_mixed_qkv_tok + v_off, mask=mask_v, other=0).to(tl.float32)

    # Optional q/k L2 norm: full-vector reciprocal norms (computed, not kept).
    if USE_QK_L2NORM_IN_KERNEL:
        o_kf = tl.arange(0, BK)
        mask_kf = o_kf < K
        p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
        qf = tl.load(p_mix + i_h * K + o_kf, mask=mask_kf, other=0).to(tl.float32)
        kf = tl.load(p_mix + H * K + i_h * K + o_kf, mask=mask_kf, other=0).to(tl.float32)
        q_rnorm = 1.0 / tl.sqrt(tl.sum(qf * qf) + 1e-6)
        k_rnorm = 1.0 / tl.sqrt(tl.sum(kf * kf) + 1e-6)
    else:
        q_rnorm = 1.0
        k_rnorm = 1.0

    # Reconstruct the state from the checkpoint + cached (d, k) in K tiles and read
    # it with the current q and k. K-tiling avoids holding a full [BV, BK] tile.
    # Also append the current key chunk to the ring cache (non-flush only).
    b_state_q = tl.zeros([BV], dtype=tl.float32)
    b_state_k = tl.zeros([BV], dtype=tl.float32)
    # LS6 누산기. ⚠ Triton SSA 라 루프 밖에서 정의해야 루프 안 갱신이 보인다.
    #   b_kq/b_kk : ⟨k̂_s, q̂⟩ — **링 키를 재생용으로 이미 읽으므로 추가 트래픽 0**
    #   b_zdq/b_zdk: Ẑᵀ(q̂−q̄) 의 타일 누적
    b_kq = tl.zeros([BC], dtype=tl.float32)
    b_kk = tl.zeros([BC], dtype=tl.float32)
    b_zdq = tl.zeros([LS6_G], dtype=tl.float32)
    b_zdk = tl.zeros([LS6_G], dtype=tl.float32)
    b_zk6 = tl.zeros([LS6_G], dtype=tl.float32)   # Zᵀk̂_t (f_s 의 앞항)
    cur_kq = tl.zeros([1], dtype=tl.float32)
    write_k = (not b_is_flush) and (i_v == 0) and (i_hv == i_h * (HV // H))
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        mask_kt = o_kt < K
        p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
        q_c = tl.load(p_mix + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32) * q_rnorm
        k_c = tl.load(p_mix + H * K + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32) * k_rnorm
        q_cs = q_c * scale
        cur_kq += tl.sum(k_c * q_cs)

        # [nested_ssm 2026-08-25] **PREF — h0 타일을 먼저 발행한다.**
        # 원래 순서는 k_cache 로드 -> tl.dot -> h0 로드였다. h0 와 k_cache 는 서로
        # 독립인데 h0 의 DRAM 지연이 dot **뒤에** 통째로 노출된다. Mamba2 가 ncu 로
        # 같은 자리를 짚었다: "대역폭이 아니라 지연 바운드 — 스케줄러 사이클 46%가
        # eligible warp 없음, stall 은 long_scoreboard 압도". 먼저 발행하면 그 지연이
        # dot 계산 뒤로 숨는다. 축소 순서는 그대로라 **수치는 동일하다**.
        # ⚠ Triton SSA 라 if 안에서 만든 값은 밖에서 안 보인다 — 양쪽 분기에서 정의한다.
        #   cold 분기의 zeros 는 **로드가 아니라 레지스터**다(트래픽 0). 그래도 레지스터
        #   압박이 될 수 있어 PREF=0 으로 옛 순서와 A/B 할 수 있게 뒀다.
        # EVICT: 비-flush 에서는 state 가 스트리밍이라 재사용이 없다 — evict_first 로
        # L2 를 안 더럽히면 대조(off) 가 580->549us 로 좋아진다.
        # ⚠ **BR=1 에서만.** flush 커널(BR=2/0)은 이 루프를 돈 뒤 flush 블록에서
        #   **같은 h0 를 다시 읽는다**. evict_first 를 걸면 그 재사용이 깨져 flush 가
        #   608 -> 750us 로 나빠진다(실측). 실제평균으로는 손해다.
        _hot = (not FZ_ON) or (kk * BKT < nf_h)
        hot_mask = mask_kt if not FZ_ON else (mask_kt & (o_kt < nf_h))
        p_h0_c = h0 + state_idx * stride_init_state_token + i_hv * stride_state_h + o_v[:, None] * stride_state_v + o_kt[None, :] * stride_state_k
        _ev = EVICT and BR == 1
        if PREF:
            if _hot:
                if _ev:
                    b_h0_c = tl.load(p_h0_c, mask=mask_v[:, None] & hot_mask[None, :],
                                     other=0, eviction_policy="evict_first").to(tl.float32)
                else:
                    b_h0_c = tl.load(p_h0_c, mask=mask_v[:, None] & hot_mask[None, :],
                                     other=0).to(tl.float32)
            else:
                b_h0_c = tl.zeros([BV, BKT], dtype=tl.float32)

        # Reconstruct this K tile of the state: S = total_decay * S_0 + d_scaled . k_cache.
        # ⚠ 링 재생(tl.dot)은 **전 K 타일**에서 그대로다 — 그게 exact replay 다.
        #   프리즈가 없애는 것은 체크포인트 h0 읽기뿐이다.
        p_k_c = k_cache + state_idx * stride_k_slot + ((i_h * MAX_CACHE_LEN + o_c[:, None]) * K + o_kt[None, :])
        b_k_all_c = tl.load(p_k_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0).to(p_o.dtype.element_ty)
        b_ring_c = tl.dot(
            b_d_scaled_tc, b_k_all_c, input_precision=DOT_INPUT_PRECISION
        ).to(tl.float32)  # [BV, BKT]
        if PREF:
            if _hot:
                b_h_c = b_h0_c * b_total_decay + b_ring_c
            else:
                b_h_c = b_ring_c
        else:
            if _hot:
                if _ev:
                    _h0 = tl.load(p_h0_c, mask=mask_v[:, None] & hot_mask[None, :],
                                  other=0, eviction_policy="evict_first").to(tl.float32)
                else:
                    _h0 = tl.load(p_h0_c, mask=mask_v[:, None] & hot_mask[None, :],
                                  other=0).to(tl.float32)
                b_h_c = _h0 * b_total_decay + b_ring_c
            else:
                b_h_c = b_ring_c

        if LS6_ON:
            if LS6_ERASE:
                # 링 키는 재생 때문에 어차피 읽었다 — 얹는 것은 산술뿐이다.
                _kc32 = b_k_all_c.to(tl.float32)
                b_kq += tl.sum(_kc32 * q_cs[None, :], axis=1)
                b_kk += tl.sum(_kc32 * k_c[None, :], axis=1)
            _og6 = tl.arange(0, LS6_G)
            _m6 = mask_kt & (o_kt < LS6_R)
            _zh6 = tl.load(ls6_z + i_h * K * LS6_G
                           + o_kt[:, None] * LS6_G + _og6[None, :],
                           mask=_m6[:, None], other=0.0).to(tl.float32)
            _qb6 = tl.load(fz_qbar + state_idx * stride_fzb_slot + i_h * K + o_kt,
                           mask=_m6, other=0.0).to(tl.float32)
            _kb6 = tl.load(fz_kbar + state_idx * stride_fzb_slot + i_h * K + o_kt,
                           mask=_m6, other=0.0).to(tl.float32)
            b_zdq += tl.sum(_zh6 * tl.where(_m6, q_cs - _qb6, 0.0)[:, None], axis=0)
            b_zdk += tl.sum(_zh6 * tl.where(_m6, k_c - _kb6, 0.0)[:, None], axis=0)

        # Read the state with q and k (accumulated across K tiles).
        b_state_q += tl.sum(b_h_c * q_cs[None, :], axis=1)
        b_state_k += tl.sum(b_h_c * k_c[None, :], axis=1)

        if write_k:
            p_cur_k = k_cache + state_idx * stride_k_slot + ((i_h * MAX_CACHE_LEN + b_write_pos) * K + o_kt)
            tl.store(p_cur_k, k_c.to(p_o.dtype.element_ty), mask=mask_kt & (b_write_pos < MAX_CACHE_LEN))
        if LS6_ON and LS6_ERASE:
            # f_s 의 앞항 Zᵀ(d_{[s]}⊙k̂_s) — 전 K 를 쓴다(절단하면 소거가 부정확해진다).
            # ⚠ **소거를 끄면 이 읽기를 하면 안 된다.** K×G/head/step 이라 dense state
            #   읽기(K·V)와 같은 크기다 — 켜 둔 채로 재면 배수가 1.44x 에서 포화한다
            #   (실측 2026-09-02). 트래픽을 재는 커널에 안 쓰는 읽기를 남기면 안 된다.
            _zf6 = tl.load(ls6_z + i_h * K * LS6_G
                           + o_kt[:, None] * LS6_G + tl.arange(0, LS6_G)[None, :],
                           mask=mask_kt[:, None], other=0.0).to(tl.float32)
            b_zk6 += tl.sum(_zf6 * k_c[:, None], axis=0)

    # Current-token output: alpha*(S q) + d_cur * (k . q), with the new update
    # vector d_cur = beta * (v - alpha*(S k)).
    if LS6_ON and FZ_ON:
        # ⚠ **FZ_ON 과 함께 켜진다.** 래치는 '건너뛴 state 읽기'를 대신하는 것이므로,
        #   FZ_ON=False(전량 읽기, flush-exact) 일 때 더하면 **이중계산**이다
        #   (실측: 상대 1.15). 접기(flush)는 LS6_ON 만 보고 항상 돈다 — 쓰기니까.
        # ── 래치가 cold 체크포인트 몫을 낸다 (논문 §4 eq:m-steploop) ──────────
        #   x = d_{[t]}·(A + Ẑᵀδ) − Σ_s f_s·⟨k̂_s⊘d_{[s]}, d_{[t]}⊙q̂⟩
        #   그런데 그 안쪽 괄호가 **커널에 이미 있다**:
        #       b_replay_decay[s] = exp(g_total − g_prefix[s]) = d_{[t]}/d_{[s]}
        #   이라 소거합은 b_replay_decay·b_kq 를 f_s 로 축소하면 끝이다.
        # ⚠ **dense fallback**: ls6_mh[hv]=0 이면 _gm 이 전부 거짓이라 래치 항이 0 이
        #   되고, fz_nf[hv]=K 라 위 타일 루프가 state 를 전량 읽는다(정확). 새 기구가
        #   필요 없는 이유다 — Super 와 같은 표현이고 이것이 **기본 모드**다.
        _og6 = tl.arange(0, LS6_G)
        _mh6 = tl.load(ls6_mh + i_hv).to(tl.int32)
        _gm = _og6 < _mh6
        _fs6 = tl.load(ls6_fs + state_idx * stride_ls6_fs_slot
                       + i_hv * (MAX_CACHE_LEN * LS6_G)
                       + o_c[:, None] * LS6_G + _og6[None, :],
                       mask=cache_valid[:, None] & _gm[None, :], other=0.0).to(tl.float32)
        _aq6 = tl.load(ls6_aq + state_idx * stride_ls6_a_slot + i_hv * LS6_G + _og6,
                       mask=_gm, other=0.0).to(tl.float32)
        _ak6 = tl.load(ls6_ak + state_idx * stride_ls6_a_slot + i_hv * LS6_G + _og6,
                       mask=_gm, other=0.0).to(tl.float32)
        if LS6_ERASE:
            _erq = tl.sum(_fs6 * (b_replay_decay * b_kq)[:, None], axis=0)
            _erk = tl.sum(_fs6 * (b_replay_decay * b_kk)[:, None], axis=0)
        else:
            _erq = tl.zeros([LS6_G], dtype=tl.float32)
            _erk = tl.zeros([LS6_G], dtype=tl.float32)
        _xq = b_total_decay * (_aq6 + b_zdq) - _erq
        _xk = b_total_decay * (_ak6 + b_zdk) - _erk
        # Ū 에는 M0⁻¹ 이 접혀 있다 — 여기 역행렬이 없다. q/k 가 **같은 Ū 를 공유**하고,
        # 그래서 GDN 의 두 접힘이 트래픽을 두 배로 만들지 않는다(앵커만 두 벌).
        _ub6 = tl.load(ls6_ubar + state_idx * stride_ls6_u_slot + i_hv * (LS6_G * V)
                       + _og6[:, None] * V + o_v[None, :],
                       mask=_gm[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        b_state_q += tl.sum(_ub6 * _xq[:, None], axis=0)
        b_state_k += tl.sum(_ub6 * _xk[:, None], axis=0)
    elif FZ_ON:
        # cold 체크포인트 기여. 참조의 ck_q = tdec*(Σ_hot h0·q + u) 와 같은 자리다.
        p_fu = fz_u + state_idx * stride_fz_slot + i_hv * V + o_v
        b_state_q += tl.load(p_fu, mask=mask_v, other=0).to(tl.float32) * b_total_decay
        p_fz = fz_z + state_idx * stride_fz_slot + i_hv * V + o_v
        b_state_k += tl.load(p_fz, mask=mask_v, other=0).to(tl.float32) * b_total_decay
    b_state_q *= alpha_val
    b_state_k *= alpha_val
    b_d_cur = beta_val * (b_v - b_state_k)
    b_o = b_state_q + b_d_cur * tl.sum(cur_kq)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    if BR != 1 and b_is_flush:
        # Flush: fold the current token into the checkpoint, S_t = alpha*S + d_t k_t^T,
        # and persist it. Re-walk K chunks to rebuild S before applying the update.
        if FZ_V2:
            # 다음 윈도우의 u = Σ_cold S_new·q̄,  z = Σ_cold S_new·k̄.
            u_acc = tl.zeros([BV], dtype=tl.float32)
            z_acc = tl.zeros([BV], dtype=tl.float32)
        if LS6_ON:
            u6_acc = tl.zeros([BV, LS6_G], dtype=tl.float32)   # Ū (V,G) 조각
        for kkf in range(NK):
            o_kt = kkf * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
            k_c = tl.load(p_mix + H * K + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32) * k_rnorm
            p_h0_c = h0 + state_idx * stride_init_state_token + i_hv * stride_state_h + o_v[:, None] * stride_state_v + o_kt[None, :] * stride_state_k
            b_h0_c = tl.load(p_h0_c, mask=mask_v[:, None] & mask_kt[None, :], other=0).to(tl.float32)
            p_k_c = k_cache + state_idx * stride_k_slot + ((i_h * MAX_CACHE_LEN + o_c[:, None]) * K + o_kt[None, :])
            b_k_all_c = tl.load(p_k_c, mask=cache_valid[:, None] & mask_kt[None, :], other=0).to(p_o.dtype.element_ty)
            b_h_c = b_h0_c * b_total_decay + tl.dot(
                b_d_scaled_tc, b_k_all_c, input_precision=DOT_INPUT_PRECISION
            ).to(tl.float32)
            b_h_new_c = alpha_val * b_h_c + b_d_cur[:, None] * k_c[None, :]
            p_ht_c = ht + state_idx * stride_final_state_token + i_hv * stride_state_h + o_v[:, None] * stride_state_v + o_kt[None, :] * stride_state_k
            tl.store(p_ht_c, b_h_new_c.to(p_ht_c.dtype.element_ty), mask=mask_v[:, None] & mask_kt[None, :])
            if LS6_ON:
                # Ū = S_new·Z̄ 를 **레지스터에 있는 동안** 접는다 — 추가 state 읽기 0.
                # M0⁻¹ 은 Z̄ 에 이미 들어 있다(층 상수라 호스트가 미리 접었다).
                _hn6 = b_h_new_c.to(p_ht_c.dtype.element_ty).to(tl.float32)
                _zb6 = tl.load(ls6_zbar + i_h * K * LS6_G
                               + o_kt[:, None] * LS6_G + tl.arange(0, LS6_G)[None, :],
                               mask=mask_kt[:, None], other=0.0).to(tl.float32)
                u6_acc += tl.sum(_hn6[:, :, None] * _zb6[None, :, :], axis=1)
            if FZ_V2:
                # ⚠ **저장된 값**을 쓴다. 다음 윈도우의 비-flush 스텝이 hot 쪽은 이
                #   텐서에서 읽으므로, u/z 도 같은 반올림을 거친 값에서 나와야 hot+cold
                #   합이 한 텐서에서 나온 것이 된다(Mamba2 가 같은 이유로 state_store 사용).
                _hn = b_h_new_c.to(p_ht_c.dtype.element_ty).to(tl.float32)
                _cold = mask_kt & (o_kt >= nf_h)
                _qb = tl.load(fz_qbar + state_idx * stride_fzb_slot + i_h * K + o_kt,
                              mask=_cold, other=0.0).to(tl.float32)
                _kb = tl.load(fz_kbar + state_idx * stride_fzb_slot + i_h * K + o_kt,
                              mask=_cold, other=0.0).to(tl.float32)
                u_acc += tl.sum(_hn * _qb[None, :], axis=1)
                z_acc += tl.sum(_hn * _kb[None, :], axis=1)
        if FZ_V2:
            # 새 체크포인트에 대응하는 u/z 를 슬롯에 남긴다 — 다음 윈도우의 비-flush
            # 스텝들이 cold state 를 한 번도 안 읽고 이 값을 쓴다(호스트 개입 0).
            # i_v 블록마다 자기 o_v 구간만 쓰므로 경합 없음.
            tl.store(fz_u + state_idx * stride_fz_slot + i_hv * V + o_v, u_acc, mask=mask_v)
            tl.store(fz_z + state_idx * stride_fz_slot + i_hv * V + o_v, z_acc, mask=mask_v)
        if LS6_ON:
            tl.store(ls6_ubar + state_idx * stride_ls6_u_slot + i_hv * (LS6_G * V)
                     + tl.arange(0, LS6_G)[None, :] * V + o_v[:, None],
                     u6_acc, mask=mask_v[:, None])
    else:
        # Non-flush: append the current token's update vector d and gate g to the
        # cache (the k chunks were already written inside the loop above).
        p_cur_d = d_cache + state_idx * stride_d_slot + ((i_hv * MAX_CACHE_LEN + b_write_pos) * V + o_v)
        tl.store(p_cur_d, b_d_cur.to(p_cur_d.dtype.element_ty), mask=mask_v & (b_write_pos < MAX_CACHE_LEN))
        if LS6_ON and LS6_ERASE and i_v == 0:
            # ── f_s 를 슬롯에 남긴다 (부록 eq:wyread) ──────────────────────────
            #   f_t = β_t [ Zᵀ(d_{[t]}⊙k̂_t) − Σ_{s<t} f_s ⟨ℓ_s(t), k̂_t⟩ ]
            # 필요한 것이 전부 이미 있다:
            #   d_{[t]}          = alpha_val · b_total_decay   (현재 토큰까지)
            #   ⟨ℓ_s(t), k̂_t⟩   = (d_{[t]}/d_{[s]})·⟨k̂_s,k̂_t⟩
            #                    = alpha_val · b_replay_decay[s] · b_kk[s]
            # b_kk 는 링 재생 루프가 이미 누적했다 — 추가 읽기 0.
            # ⚠ **쓴 뒤 다시 안 고친다.** 슬롯이 생길 때 확정되는 값이라, 나중 스텝이
            #   손대면 WY 표현이 깨진다.
            _og6f = tl.arange(0, LS6_G)
            _mh6f = tl.load(ls6_mh + i_hv).to(tl.int32)
            _gmf = _og6f < _mh6f
            _dt6 = alpha_val * b_total_decay
            _fprev = tl.load(ls6_fs + state_idx * stride_ls6_fs_slot
                             + i_hv * (MAX_CACHE_LEN * LS6_G)
                             + o_c[:, None] * LS6_G + _og6f[None, :],
                             mask=cache_valid[:, None] & _gmf[None, :], other=0.0).to(tl.float32)
            _w6 = alpha_val * b_replay_decay * b_kk
            _fnew = beta_val * (_dt6 * b_zk6 - tl.sum(_fprev * _w6[:, None], axis=0))
            tl.store(ls6_fs + state_idx * stride_ls6_fs_slot
                     + i_hv * (MAX_CACHE_LEN * LS6_G)
                     + b_write_pos * LS6_G + _og6f,
                     _fnew, mask=_gmf & (b_write_pos < MAX_CACHE_LEN))
        if i_v == 0:
            p_cur_g = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + b_write_pos
            tl.store(p_cur_g, g_val, mask=b_write_pos < MAX_CACHE_LEN)


# ═══════════════════════════════════════════════════════════════════════════════
# [nested_ssm 2026-09-02] v2 — **물질화 없는 스텝 + 압축 리스트 위의 지속형 flush.**
#
#   v1 은 링을 tl.dot 으로 [V,K] 타일에 물질화한 뒤 q·k 로 읽었다. 그 타일이 레지스터
#   255 + smem 을 잡아 점유율 12.5% 에 묶이고 명령 발행 바운드였다(nf=0 천장에서도
#   SM 47%, DRAM 7%). state 읽기를 다 없애도 110us 가 남은 이유.
#
#   스텝(모든 행):  (Σ_s d_s k_sᵀ) q̂ = Σ_s d_s (k_s·q̂) — κ_s 를 먼저 만들고 d 링을
#     [W]→[V] 로 접는다. tl.dot·smem 없음. flush 행도 여기서 **정확히** 읽고(hot 경계
#     무시, 래치·cold 항 미사용) d_cur·k̂·g 를 링의 마지막 슬롯 W-1 에 그냥 쓴다.
#   flush(압축된 행만): S_new = d'·S₀ + Σ_{s<W} r'_s d_s k_sᵀ — 슬롯 W-1 을 포함한
#     **균일한** 재생이라 현재 토큰 의존이 없다(r'_{W-1}=1, d'=α·d_[t]).
#     [FBV,K] 타일에 rank-1 갱신 W 번(FMA, f32x2). Ū 는 재귀형, cold 접기는 직접형.
#     한 스텝에 flush 하는 행은 ~B/W 라, B×HV 격자로 띄우면 빈 CTA 가 대부분이고
#     (12288 개 ≈ 9us, 49152 개 ≈ 28us 실측) 행 하나가 1 워프에 직렬화돼 꼬리가
#     50us 를 넘는다. → 작은 압축 커널이 flush 행 목록을 만들고, 지속형 격자(NP)가
#     (행, head, v-분할) 작업을 나눠 가진다. 전 행 flush(프리필 직후)도 같은 커널.
#   분기 특화(BR)는 v2 에 없다 — 갈라 둘 이유(flush 의 큰 타일)가 스텝 커널에서 사라졌다.
#
#   **불변식(호스트 책임):** Ū 는 시딩 때 S₀Z̄ 로 접어 둔다(flush 가 재귀로만 갱신).
#   호출 규약·버퍼·의미는 v1 과 같다(NS_GDN_KERNEL=v1 로 되돌릴 수 있다).
#   LS6_ERASE 는 v2 에 없다(기본 꺼짐이고 우리 링에서는 이중계산이다 — 위 주석).
# ═══════════════════════════════════════════════════════════════════════════════
@triton.jit
def _gdn_v2_step_kernel(
    mixed_qkv, a, b, A_log, dt_bias, o, h0,
    d_cache, k_cache, g_cache, ssm_state_indices, write_pos, scale,
    fz_nf, fz_u, fz_z, stride_fz_slot: tl.constexpr,
    fz_qbar, fz_kbar, stride_fzb_slot: tl.constexpr,
    ls6_ubar, ls6_z, ls6_zbar, ls6_zk, ls6_aq, ls6_ak, ls6_mh, ls6_map,
    stride_ls6_u_slot: tl.constexpr, stride_ls6_a_slot: tl.constexpr,
    stride_ls6_zk_slot: tl.constexpr,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr, stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_state_h: tl.constexpr, stride_state_v: tl.constexpr, stride_state_k: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_d_slot: tl.constexpr, stride_k_slot: tl.constexpr, stride_g_slot: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, BC: tl.constexpr,
    NK: tl.constexpr, BKT: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    FZ_ON: tl.constexpr, FZ_V2: tl.constexpr,
    LS6_ON: tl.constexpr, LS6_G: tl.constexpr, LS6_R: tl.constexpr, LS6_RP: tl.constexpr,
    EVICT: tl.constexpr, LS6_MAP: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_n = tl.program_id(1)
    i_hv = tl.program_id(2)
    i_h = i_hv // (HV // H)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    o_c = tl.arange(0, BC)

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    if state_idx <= 0:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty), mask=mask_v)
        return
    b_write_pos = tl.load(write_pos + i_n).to(tl.int64)
    # 슬롯 간접: fz_*/ls6_* 는 compact 슬롯(NS) 으로 잡는다 — ls6_map[NX] (호스트 LRU). 없으면 항등.
    if LS6_MAP:
        cidx = tl.load(ls6_map + state_idx).to(tl.int64)
    else:
        cidx = state_idx
    b_is_flush = b_write_pos == MAX_CACHE_LEN - 1
    cache_valid = o_c < b_write_pos

    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    alpha_val = tl.exp(g_val)
    beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)

    p_g = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + o_c
    b_g_all = tl.load(p_g, mask=cache_valid, other=0.0).to(tl.float32)
    b_g_prefix = tl.cumsum(b_g_all, axis=0)
    b_g_total = tl.sum(b_g_all, axis=0)
    b_replay_decay = tl.where(cache_valid, tl.exp(b_g_total - b_g_prefix), 0.0)
    b_total_decay = tl.exp(b_g_total)
    if FZ_ON or FZ_V2:
        nf_h = tl.load(fz_nf + i_hv).to(tl.int32)
    else:
        nf_h = K
    # flush 행은 exact — state 를 전부 읽는다(hot 경계 무시). 래치/cold 항도 안 쓴다.
    if b_is_flush:
        nf_eff = K
    else:
        nf_eff = nf_h

    o_k = tl.arange(0, BK)
    mask_k = o_k < K
    p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_f = tl.load(p_mix + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    k_f = tl.load(p_mix + H * K + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        q_rnorm = 1.0 / tl.sqrt(tl.sum(q_f * q_f) + 1e-6)
        k_rnorm = 1.0 / tl.sqrt(tl.sum(k_f * k_f) + 1e-6)
    else:
        q_rnorm = 1.0
        k_rnorm = 1.0
    q_sc = q_rnorm * scale
    write_k = (i_v == 0) and (i_hv == i_h * (HV // H))

    # ── 읽기 패스: κ_s (링 키), hot 체크포인트 열 — 물질화 없음 ──────────────
    kap_q = tl.zeros([BC], dtype=tl.float32)
    kap_k = tl.zeros([BC], dtype=tl.float32)
    cur_kq = tl.zeros([1], dtype=tl.float32)
    hq = tl.zeros([BV], dtype=tl.float32)
    hk = tl.zeros([BV], dtype=tl.float32)
    if LS6_ON:
        # zk_s = Z̄ᵀk̂_s 를 링(ls6_zk)에 같이 둔다(write_k CTA 만). flush 의 Ū 재귀가
        # k 링×Z̄(K·G 곱 + 교차 레인 합, v-split 마다 반복)를 안 하게 하기 위함.
        # ⚠ 루프 안에서 타일마다 읽으면 L2 왕복이 NK 번 직렬화돼 스텝이 72→82us 였다.
        #   루프 **앞**에서 한 번에 띄워(마스크로 write_k 아닌 CTA 는 0비용) 뒤에서 접는다.
        o_g6 = tl.arange(0, LS6_G)
        zb_f = tl.load(ls6_zbar + i_h * K * LS6_G + o_k[:, None] * LS6_G + o_g6[None, :],
                       mask=(mask_k & write_k)[:, None], other=0.0).to(tl.float32)
    for kk in range(NK):
        o_kt = kk * BKT + tl.arange(0, BKT)
        m_kt = o_kt < K
        q_c = tl.load(p_mix + i_h * K + o_kt, mask=m_kt, other=0).to(tl.float32) * q_sc
        k_c = tl.load(p_mix + H * K + i_h * K + o_kt, mask=m_kt, other=0).to(tl.float32) * k_rnorm
        cur_kq += tl.sum(k_c * q_c)
        p_kr = k_cache + state_idx * stride_k_slot + (i_h * MAX_CACHE_LEN + o_c[:, None]) * K + o_kt[None, :]
        k_t = tl.load(p_kr, mask=cache_valid[:, None] & m_kt[None, :], other=0).to(tl.float32)
        kap_q += tl.sum(k_t * q_c[None, :], axis=1)
        kap_k += tl.sum(k_t * k_c[None, :], axis=1)
        if write_k:
            tl.store(k_cache + state_idx * stride_k_slot + (i_h * MAX_CACHE_LEN + b_write_pos) * K + o_kt,
                     k_c.to(k_cache.dtype.element_ty), mask=m_kt)
        if kk * BKT < nf_eff:
            m_h = m_kt & (o_kt < nf_eff)
            p_h = h0 + state_idx * stride_init_state_token + i_hv * stride_state_h \
                + o_v[:, None] * stride_state_v + o_kt[None, :] * stride_state_k
            if EVICT:
                h_t = tl.load(p_h, mask=mask_v[:, None] & m_h[None, :], other=0,
                              eviction_policy="evict_first").to(tl.float32)
            else:
                h_t = tl.load(p_h, mask=mask_v[:, None] & m_h[None, :], other=0).to(tl.float32)
            hq += tl.sum(h_t * q_c[None, :], axis=1)
            hk += tl.sum(h_t * k_c[None, :], axis=1)
    if LS6_ON:
        if write_k:
            # ⚠ 링에 저장된 k̂ (k_cache dtype, bf16 이면 반올림됨) 과 **같은 값**으로 zk 를 내야
            #   flush 의 S 접기(k_r0 = 링) 와 Ū 재귀(zk) 가 일치한다. fp32 k̂ 로 내면 Ū 가
            #   Z̄ᵀS 에서 bf16 몫(≈1e-3) 만큼 영구히 어긋난다 (ieee 항등 검사에서 잡힘, 2026-09-02).
            k_st = (k_f * k_rnorm).to(k_cache.dtype.element_ty).to(tl.float32)
            zk_acc = tl.sum(zb_f * k_st[:, None], axis=0)
            tl.store(ls6_zk + cidx * stride_ls6_zk_slot + (i_h * MAX_CACHE_LEN + b_write_pos) * LS6_G + o_g6,
                     zk_acc)
    kap_q = kap_q * b_replay_decay
    kap_k = kap_k * b_replay_decay
    p_dr = d_cache + state_idx * stride_d_slot + (i_hv * MAX_CACHE_LEN + o_c[:, None]) * V + o_v[None, :]
    d_ring = tl.load(p_dr, mask=cache_valid[:, None] & mask_v[None, :], other=0).to(tl.float32)
    s_q = tl.sum(d_ring * kap_q[:, None], axis=0)
    s_k = tl.sum(d_ring * kap_k[:, None], axis=0)

    if LS6_ON and FZ_ON:
        # 래치: x = d_[t]·(A + Ẑᵀδ),  읽기 = Ū x.  mh=0 이면 dense fallback (건너뜀)
        mh = tl.load(ls6_mh + i_hv).to(tl.int32)
        if (mh > 0) and (not b_is_flush):
            o_g = tl.arange(0, LS6_G)
            m_g = o_g < mh
            o_r = tl.arange(0, LS6_RP)
            m_r = (o_r < LS6_R) & (o_r < K)
            z_t = tl.load(ls6_z + i_h * K * LS6_G + o_r[:, None] * LS6_G + o_g[None, :],
                          mask=m_r[:, None] & m_g[None, :], other=0.0).to(tl.float32)
            q_r = tl.load(p_mix + i_h * K + o_r, mask=m_r, other=0).to(tl.float32) * q_sc
            k_r = tl.load(p_mix + H * K + i_h * K + o_r, mask=m_r, other=0).to(tl.float32) * k_rnorm
            qb = tl.load(fz_qbar + cidx * stride_fzb_slot + i_h * K + o_r, mask=m_r, other=0.0).to(tl.float32)
            kb = tl.load(fz_kbar + cidx * stride_fzb_slot + i_h * K + o_r, mask=m_r, other=0.0).to(tl.float32)
            zdq = tl.sum(z_t * (q_r - qb)[:, None], axis=0)
            zdk = tl.sum(z_t * (k_r - kb)[:, None], axis=0)
            aq = tl.load(ls6_aq + cidx * stride_ls6_a_slot + i_hv * LS6_G + o_g, mask=m_g, other=0.0).to(tl.float32)
            ak = tl.load(ls6_ak + cidx * stride_ls6_a_slot + i_hv * LS6_G + o_g, mask=m_g, other=0.0).to(tl.float32)
            xq = aq + zdq
            xk = ak + zdk
            ub = tl.load(ls6_ubar + cidx * stride_ls6_u_slot + i_hv * (LS6_G * V)
                         + o_g[:, None] * V + o_v[None, :],
                         mask=m_g[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
            hq += tl.sum(ub * xq[:, None], axis=0)
            hk += tl.sum(ub * xk[:, None], axis=0)
    elif FZ_ON:
        if not b_is_flush:
            hq += tl.load(fz_u + cidx * stride_fz_slot + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
            hk += tl.load(fz_z + cidx * stride_fz_slot + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)

    b_state_q = alpha_val * (hq * b_total_decay + s_q)
    b_state_k = alpha_val * (hk * b_total_decay + s_k)
    b_v = tl.load(p_mix + (2 * H * K) + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
    b_d_cur = beta_val * (b_v - b_state_k)
    b_o = b_state_q + b_d_cur * tl.sum(cur_kq)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    # flush 행이면 슬롯 W-1 — flush 커널이 균일 재생으로 접는다.
    p_cur_d = d_cache + state_idx * stride_d_slot + ((i_hv * MAX_CACHE_LEN + b_write_pos) * V + o_v)
    tl.store(p_cur_d, b_d_cur.to(p_cur_d.dtype.element_ty), mask=mask_v)
    if i_v == 0:
        tl.store(g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN + b_write_pos, g_val)


@triton.jit
def _gdn_v2_compact_kernel(write_pos, ssm_state_indices, flush_list, B,
                           stride_indices_seq: tl.constexpr, MAX_CACHE_LEN: tl.constexpr, BB: tl.constexpr):
    """flush 행(write_pos == W-1, 유효 슬롯) 목록을 flush_list[0:n] 에, n 을 flush_list[B] 에."""
    o_b = tl.arange(0, BB)
    m_b = o_b < B
    wp = tl.load(write_pos + o_b, mask=m_b, other=0)
    si = tl.load(ssm_state_indices + o_b * stride_indices_seq, mask=m_b, other=0)
    is_f = m_b & (wp == MAX_CACHE_LEN - 1) & (si > 0)
    f1 = is_f.to(tl.int32)
    pos = tl.cumsum(f1, axis=0) - 1
    # ⚠ 목록에는 **state_idx** 를 넣는다(행 번호가 아니라). flush 커널이 행→슬롯 종속 로드를 안 하게.
    tl.store(flush_list + pos, si.to(tl.int32), mask=is_f)
    tl.store(flush_list + B, tl.sum(f1, axis=0))


@triton.jit
def _gdn_v2_flush_kernel(
    h0, d_cache, k_cache, g_cache, ssm_state_indices, flush_list, n_ptr,
    fz_nf, fz_u, fz_z, stride_fz_slot: tl.constexpr,
    fz_qbar, fz_kbar, stride_fzb_slot: tl.constexpr,
    ls6_ubar, ls6_z, ls6_zk, ls6_mh, ls6_beta, ls6_corr_ak,
    ls6_corr_kbar, ls6_map,
    stride_ls6_u_slot: tl.constexpr, stride_ls6_zk_slot: tl.constexpr,
    stride_ls6_beta_slot: tl.constexpr,
    stride_ls6_corr_ak_slot: tl.constexpr,
    stride_ls6_corr_kbar_slot: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_state_h: tl.constexpr, stride_state_v: tl.constexpr, stride_state_k: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_d_slot: tl.constexpr, stride_k_slot: tl.constexpr, stride_g_slot: tl.constexpr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, FBV: tl.constexpr,
    NSPLIT: tl.constexpr, VJOBS: tl.constexpr, BC: tl.constexpr,
    MAX_CACHE_LEN: tl.constexpr, FZ_V2: tl.constexpr,
    LS6_ON: tl.constexpr, LS6_EXACT: tl.constexpr,
    LS6_G: tl.constexpr, LS6_GP: tl.constexpr,
    LS6_R: tl.constexpr, LS6_RP: tl.constexpr, NP: tl.constexpr,
    FDOT: tl.constexpr, LS6_MAP: tl.constexpr,
):
    """persistent flush. 일감 = (flush 행, hv) 하나 = state 한 장(V×K).
    S_new = tot'·S₀ + (r'∘D)ᵀ·K 를 텐서코어(tf32x3 ≈ fp32, FDOT=2 면 ieee)로 접는다.
    LS6_EXACT 면 접기 전에 온라인 키-읽기 오차를 S₀에서 정확히 복원하고, 아래의
    삼각 점화로 창 안의 모든 delta write 를 보정한다. 그러면 저장하는 창 경계
    state 는 dense delta rule 결과와 같고, 래치 오차는 다음 창으로 넘어가지 않는다.
    링 K [W,BK] 는 일감당 한 번 올려 두고 V 를 FBV 씩 돌린다 — v-split 을 그리드에 펼치면
    K 링을 split 마다 다시 읽어(FBV=16 이면 S₀ 읽기와 같은 양) 스칼라 외적 루프가
    FMA 당 2.7 명령으로 발행 바운드(1024us)였다.
    FZ 접기 u/z = S_new·[q̄_cold k̄_cold 0…] 도 [BK,16] 한 번의 dot 로(축 1 리덕션 대신).
    Ū 재귀는 스텝이 넣어 둔 zk 링으로 g 마다 [FBV,W] 곱-합."""
    pid = tl.program_id(0)
    n_flush = tl.load(n_ptr)
    n_work = n_flush * HV * VJOBS
    o_c = tl.arange(0, BC)
    m_c = o_c < MAX_CACHE_LEN          # BC 는 16 이상으로 올림된다 — W<16 이면 마스크 필수
    o_kf = tl.arange(0, BK)
    m_kf = o_kf < K
    o_16 = tl.arange(0, 16)
    for w in range(pid, n_work, NP):
        work = w // VJOBS
        i_vjob = w % VJOBS
        r = work // HV
        i_hv = work % HV
        i_h = i_hv // (HV // H)
        state_idx = tl.load(flush_list + r).to(tl.int64)
        if LS6_MAP:
            cidx = tl.load(ls6_map + state_idx).to(tl.int64)   # compact 슬롯 (스텝과 같은 규약)
        else:
            cidx = state_idx
        # 균일 재생: 슬롯 W-1(현재 토큰) 포함. tot' = exp(Σ_{s<W} g), r'_s = exp(Σ_{s<W} g − prefix_s)
        p_gs = g_cache + state_idx * stride_g_slot + i_hv * MAX_CACHE_LEN
        g_all = tl.load(p_gs + o_c, mask=m_c, other=0.0).to(tl.float32)
        g_prefix = tl.cumsum(g_all, axis=0)
        g_total = tl.sum(g_all, axis=0)
        tot = tl.exp(g_total)
        replay = tl.where(m_c, tl.exp(g_total - g_prefix), 0.0)
        p_kr0 = k_cache + state_idx * stride_k_slot + (i_h * MAX_CACHE_LEN + o_c[:, None]) * K + o_kf[None, :]
        k_r0 = tl.load(p_kr0, mask=m_c[:, None] & m_kf[None, :], other=0).to(tl.float32)
        if FZ_V2:
            nf_h = tl.load(fz_nf + i_hv).to(tl.int32)
            _cold = m_kf & (o_kf >= nf_h)
            _qb = tl.load(fz_qbar + cidx * stride_fzb_slot + i_h * K + o_kf, mask=_cold, other=0.0).to(tl.float32)
            _kb = tl.load(fz_kbar + cidx * stride_fzb_slot + i_h * K + o_kf, mask=_cold, other=0.0).to(tl.float32)
            qk_op = tl.where(o_16[None, :] == 0, _qb[:, None],
                             tl.where(o_16[None, :] == 1, _kb[:, None], 0.0))
        if LS6_ON:
            p_zk = ls6_zk + cidx * stride_ls6_zk_slot + (i_h * MAX_CACHE_LEN + o_c) * LS6_G
            o_gp = tl.arange(0, LS6_GP)
            m_gall = o_gp < LS6_G
        if LS6_EXACT:
            # 스텝이 쓴 hat(d)_s 를 dense delta-rule d_s 로 되돌리는 삼각 보정.
            # 현재 창의 앵커는 host 부기가 다음 창 값으로 바꾸기 전에 corr_* 에 복사했다.
            mh = tl.load(ls6_mh + i_hv).to(tl.int32)
            nf_h = tl.load(fz_nf + i_hv).to(tl.int32)
            m_gp = o_gp < mh
            o_rp = tl.arange(0, LS6_RP)
            m_rp = (o_rp < LS6_R) & (o_rp < K)
            z_r = tl.load(
                ls6_z + i_h * K * LS6_G
                + o_rp[:, None] * LS6_G + o_gp[None, :],
                mask=m_rp[:, None] & m_gp[None, :], other=0.0,
            ).to(tl.float32)
            p_krr = (
                k_cache + state_idx * stride_k_slot
                + (i_h * MAX_CACHE_LEN + o_c[:, None]) * K
                + o_rp[None, :]
            )
            k_rr = tl.load(
                p_krr, mask=m_c[:, None] & m_rp[None, :], other=0.0
            ).to(tl.float32)
            kbar_r = tl.load(
                ls6_corr_kbar + cidx * stride_ls6_corr_kbar_slot
                + i_h * K + o_rp,
                mask=m_rp, other=0.0,
            ).to(tl.float32)
            ak = tl.load(
                ls6_corr_ak + cidx * stride_ls6_corr_ak_slot
                + i_hv * LS6_G + o_gp,
                mask=m_gp, other=0.0,
            ).to(tl.float32)
            if FDOT == 2:
                xk = ak[None, :] + tl.dot(
                    k_rr - kbar_r[None, :], z_r,
                    input_precision="ieee",
                )
                kappa = tl.dot(
                    k_r0, tl.trans(k_r0), input_precision="ieee"
                )
            else:
                xk = ak[None, :] + tl.dot(
                    k_rr - kbar_r[None, :], z_r,
                    input_precision="tf32x3",
                )
                kappa = tl.dot(
                    k_r0, tl.trans(k_r0), input_precision="tf32x3"
                )
            beta_all = tl.load(
                ls6_beta + cidx * stride_ls6_beta_slot
                + i_hv * MAX_CACHE_LEN + o_c,
                mask=m_c, other=0.0,
            ).to(tl.float32)
        for i_vlocal in range(NSPLIT):
            i_vs = i_vjob * NSPLIT + i_vlocal
            o_vf = i_vs * FBV + tl.arange(0, FBV)
            m_vf = o_vf < V
            p_dT = d_cache + state_idx * stride_d_slot + (i_hv * MAX_CACHE_LEN + o_c[None, :]) * V + o_vf[:, None]
            d_fold = tl.load(
                p_dT, mask=m_vf[:, None] & m_c[None, :], other=0
            ).to(tl.float32)
            p_hf = h0 + state_idx * stride_init_state_token + i_hv * stride_state_h \
                + o_vf[:, None] * stride_state_v + o_kf[None, :] * stride_state_k
            S_0 = tl.load(
                p_hf, mask=m_vf[:, None] & m_kf[None, :], other=0
            ).to(tl.float32)
            if LS6_ON:
                p_u_all = (
                    ls6_ubar + cidx * stride_ls6_u_slot
                    + i_hv * (LS6_G * V)
                    + o_gp[None, :] * V + o_vf[:, None]
                )
                ub_old = tl.load(
                    p_u_all,
                    mask=m_vf[:, None] & m_gall[None, :],
                    other=0.0,
                ).to(tl.float32)
            if LS6_EXACT:
                # hot prefix 읽기는 온라인과 dense 양쪽에 동일하므로 소거한다.
                # epsilon_s = exp(prefix_s) * (Ubar*x_s - S0*k_s,cold).
                k_cold = tl.where(
                    (o_kf[None, :] >= nf_h) & m_c[:, None],
                    k_r0,
                    0.0,
                )
                ub_active = tl.where(m_gp[None, :], ub_old, 0.0)
                if FDOT == 2:
                    h_latch = tl.dot(
                        ub_active, tl.trans(xk), input_precision="ieee"
                    )
                    h_cold = tl.dot(
                        S_0, tl.trans(k_cold), input_precision="ieee"
                    )
                else:
                    h_latch = tl.dot(
                        ub_active, tl.trans(xk), input_precision="tf32x3"
                    )
                    h_cold = tl.dot(
                        S_0, tl.trans(k_cold), input_precision="tf32x3"
                    )
                epsilon = tl.where(
                    o_c[None, :] < MAX_CACHE_LEN - 1,
                    (h_latch - h_cold) * tl.exp(g_prefix)[None, :],
                    0.0,
                )
                delta = tl.zeros([FBV, BC], dtype=tl.float32)
                # delta_s = -beta_s (epsilon_s + sum_{j<s} rho_{j,s} delta_j)
                # rho_{j,s} = <k_j,k_s> exp(prefix_s-prefix_j).
                for ss in range(MAX_CACHE_LEN):
                    epsilon_s = tl.sum(
                        tl.where(o_c[None, :] == ss, epsilon, 0.0),
                        axis=1,
                    )
                    prefix_s = tl.sum(
                        tl.where(o_c == ss, g_prefix, 0.0), axis=0
                    )
                    kappa_s = tl.sum(
                        tl.where(o_c[None, :] == ss, kappa, 0.0),
                        axis=1,
                    )
                    rho_s = tl.where(
                        (o_c < ss) & m_c,
                        kappa_s * tl.exp(prefix_s - g_prefix),
                        0.0,
                    )
                    prev_s = tl.sum(delta * rho_s[None, :], axis=1)
                    beta_s = tl.sum(
                        tl.where(o_c == ss, beta_all, 0.0), axis=0
                    )
                    delta_s = -beta_s * (epsilon_s + prev_s)
                    delta = tl.where(
                        o_c[None, :] == ss, delta_s[:, None], delta
                    )
                d_fold -= delta
            d_T = d_fold * replay[None, :]
            S_t = S_0 * tot
            if FDOT == 2:
                S_t = tl.dot(d_T, k_r0, acc=S_t, input_precision="ieee")
            else:
                S_t = tl.dot(d_T, k_r0, acc=S_t, input_precision="tf32x3")
            S_st = S_t.to(p_hf.dtype.element_ty)
            tl.store(p_hf, S_st, mask=m_vf[:, None] & m_kf[None, :])
            if FZ_V2:
                _hn = S_st.to(tl.float32)
                if FDOT == 2:
                    uz = tl.dot(_hn, qk_op, input_precision="ieee")
                else:
                    uz = tl.dot(_hn, qk_op, input_precision="tf32x3")
                tl.store(fz_u + cidx * stride_fz_slot + i_hv * V + o_vf,
                         tl.sum(tl.where(o_16[None, :] == 0, uz, 0.0), axis=1), mask=m_vf)
                tl.store(fz_z + cidx * stride_fz_slot + i_hv * V + o_vf,
                         tl.sum(tl.where(o_16[None, :] == 1, uz, 0.0), axis=1), mask=m_vf)
            if LS6_ON:
                # Ū 재귀: Ū_new = tot'·Ū_old + Σ_{s<W} r'_s d_s (Z̄ᵀk_s).  Ū_old = S₀Z̄ 가 전제(호스트 시딩).
                #   exact 보정에서 읽은 Ū_old 타일을 재사용하고,
                #   G 열 스칼라 루프를 dot 하나로 바꿘다. r' 는 d_T 에 접혀 있다.
                zk_all = tl.load(
                    p_zk[:, None] + o_gp[None, :],
                    mask=m_c[:, None] & m_gall[None, :],
                    other=0.0,
                ).to(tl.float32)
                if FDOT == 2:
                    u_new = tot * ub_old + tl.dot(
                        d_T, zk_all, input_precision="ieee"
                    )
                else:
                    u_new = tot * ub_old + tl.dot(
                        d_T, zk_all, input_precision="tf32x3"
                    )
                tl.store(
                    p_u_all,
                    u_new,
                    mask=m_vf[:, None] & m_gall[None, :],
                )


_SM_COUNT = {}


_STEP_FALLBACK_WARNED: dict = {}


def _sm_count():
    d = torch.cuda.current_device()
    if d not in _SM_COUNT:
        _SM_COUNT[d] = torch.cuda.get_device_properties(d).multi_processor_count
    return _SM_COUNT[d]


def fused_recurrent_gated_delta_rule_replayssm(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    d_cache: torch.Tensor,
    k_cache: torch.Tensor,
    g_cache: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    write_pos: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
    # [nested_ssm 2026-08-25] freeze. 셋 다 주면 켜진다(부분만 주면 죽는다 —
    # 조용히 프리즈 없이 도는 것을 막는다).
    # [nested_ssm 2026-09-02] LS6 래치. 일곱을 **다 주면** 켜진다(일부만 주면 죽는다 —
    #   반쯤 켜진 상태가 조용히 틀린 값을 내는 것보다 낫다).
    ls6_ubar: torch.Tensor | None = None,
    ls6_z: torch.Tensor | None = None,
    ls6_zbar: torch.Tensor | None = None,
    ls6_aq: torch.Tensor | None = None,
    ls6_ak: torch.Tensor | None = None,
    ls6_fs: torch.Tensor | None = None,
    ls6_mh: torch.Tensor | None = None,
    # v2 전용: zk 링 (NX,H,W,G) = Z̄ᵀk̂_s. 스텝이 쓰고 flush 의 Ū 재귀가 읽는다.
    ls6_zk: torch.Tensor | None = None,
    # v2 exact-flush: beta 링과 flush 직전의 이전-창 key 앵커.
    ls6_beta: torch.Tensor | None = None,
    ls6_corr_ak: torch.Tensor | None = None,
    ls6_corr_kbar: torch.Tensor | None = None,
    ls6_r: int = 0,
    # 슬롯 간접 (NX,) int32: fz_*/ls6_* 버퍼의 dim0 이 NX 가 아니라 compact NS 일 때. None = 항등.
    ls6_map: torch.Tensor | None = None,
    fz_nf: torch.Tensor | None = None,
    fz_u: torch.Tensor | None = None,
    fz_z: torch.Tensor | None = None,
    # FZ_V2: 주면 flush 분기가 다음 윈도우의 u/z 를 직접 쓴다(호스트 부기 0).
    fz_qbar: torch.Tensor | None = None,
    fz_kbar: torch.Tensor | None = None,
    block_v: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    nk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached GDN autoregressive decode (one new token per sequence).

    Same call surface as ``fused_recurrent_gated_delta_rule_packed_decode``
    plus the three ring caches (``d_cache``/``k_cache``/``g_cache``) and the
    per-decode-row ``write_pos`` cursor. ``initial_state`` is both the
    checkpoint read (h0) and the (flush-only) checkpoint write (ht), in place.
    """
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")
    if write_pos.ndim != 1 or write_pos.dtype != torch.int32:
        raise ValueError("`write_pos` must be a 1D int32 tensor.")
    B = mixed_qkv.shape[0]
    # ── Flash-Next 전용 런타임에만 있는 입력 검증 (2026-09-01 포팅) ──────────
    # 두 런타임의 **기반 코드 차이는 이 12줄이 전부다**(실측: pristine 대 pristine
    # diff 13줄). Triton 커널 본체는 동일하다 — 그래서 우리 freeze 훅 319줄이
    # 그대로 얹힌다. 검증을 지우지 않고 그대로 들고 오는 이유는, 지우면 두 트리가
    # 갈리기 시작하고 다음 병합에서 어느 쪽이 원본인지 알 수 없게 되기 때문이다.
    if ssm_state_indices.ndim != 1 or ssm_state_indices.dtype != torch.int32:
        raise ValueError("`ssm_state_indices` must be a 1D int32 tensor.")
    if write_pos.shape[0] < B or ssm_state_indices.shape[0] < B:
        raise ValueError(
            f"`write_pos` and `ssm_state_indices` must have at least B={B} entries."
        )
    if write_pos.device != mixed_qkv.device:
        raise ValueError("`write_pos` must be on the same device as `mixed_qkv`.")
    if ssm_state_indices.device != mixed_qkv.device:
        raise ValueError(
            "`ssm_state_indices` must be on the same device as `mixed_qkv`."
        )
    num_state_slots, HV, V, K = initial_state.shape
    qkv_dim = mixed_qkv.shape[1]
    q_dim = (qkv_dim - HV * V) // 2
    if q_dim <= 0 or q_dim % K != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}, K={K}."
        )
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}.")
    max_cache_len = d_cache.shape[2]

    # Launch config (block_v, num_warps, num_stages, nk) from the L-keyed config
    # module; explicit kwargs override. Lets benchmarks/the config sweep pin it via
    # override_replayssm_config("gdn_decode", ...).
    cfg_bv, cfg_nw, cfg_ns, cfg_nk = get_replayssm_config(
        "gdn_decode", L=max_cache_len
    )
    if block_v is None:
        block_v = cfg_bv
    if num_warps is None:
        num_warps = cfg_nw
    if num_stages is None:
        num_stages = cfg_ns
    if nk is None:
        nk = cfg_nk

    # Cache shape sanity (per state slot): d=(HV, L, V), k=(H, L, K), g=(HV, L).
    if tuple(d_cache.shape[1:]) != (HV, max_cache_len, V):
        raise ValueError(
            f"`d_cache` per-slot shape must be {(HV, max_cache_len, V)} "
            f"(got {tuple(d_cache.shape[1:])})."
        )
    if tuple(k_cache.shape[1:]) != (H, max_cache_len, K):
        raise ValueError(
            f"`k_cache` per-slot shape must be {(H, max_cache_len, K)} "
            f"(got {tuple(k_cache.shape[1:])})."
        )
    if tuple(g_cache.shape[1:]) != (HV, max_cache_len):
        raise ValueError(
            f"`g_cache` per-slot shape must be {(HV, max_cache_len)} "
            f"(got {tuple(g_cache.shape[1:])})."
        )
    if g_cache.dtype != torch.float32:
        raise ValueError(f"`g_cache` must be float32 (got {g_cache.dtype}).")

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"Cached decode kernel only supports NK_global=1 (got K={K}, BK={BK}).")
    # [nested_ssm 2026-08-25] 설정 표는 L 로만 키잉돼 있어 **K 가 작으면 nk 가 과하다**
    # (단위검사의 K=32 에서 nk=4 -> BKT=8 < 16 로 죽었다). tl.dot 최소폭 16 을 지키도록
    # 클램프한다. 실모델(K=128)에서는 클램프가 걸리지 않아 스윕 결과 그대로다.
    nk = max(1, min(nk, BK // 16))
    while nk > 1 and BK % nk != 0:
        nk -= 1
    BKT = BK // nk
    if BKT < 16:
        raise ValueError(f"BKT={BKT} must be >=16 for tl.dot (nk={nk}, BK={BK}).")
    # K-tiling keeps the per-program tile small enough that BV=64 (NV=1, half the
    # grid -> fewer redundant cache/metadata loads) fits without register
    # spilling.
    BV = block_v if block_v is not None else min(triton.next_power_of_2(V), 64)
    BC = max(16, triton.next_power_of_2(max_cache_len))

    # ⚠ 그리드는 **분기마다** 다시 잡는다. BV 가 분기별로 다른데 그리드를 바깥에서
    #   한 번만 계산하면, BV 가 더 작은 분기는 V 의 일부만 갱신하고 나머지는 **아예
    #   손대지 않은 채** 남는다. 조용히 틀리고, 심지어 일을 덜 해서 **빨라 보인다**
    #   (실측: flush BV=16 이 608->325us 로 보였는데 V 의 1/4 만 처리한 것이었다).
    #   2026-08-25 에 분기별 설정을 넣으면서 만든 결함이다.
    _n_fz = sum(t is not None for t in (fz_nf, fz_u, fz_z))
    if _n_fz not in (0, 3):
        raise ValueError(
            f"freeze 인자는 셋 다 주거나 하나도 주지 말 것 (받은 개수 {_n_fz}). "
            "일부만 주면 프리즈가 조용히 빠진 채 돈다.")
    _fz_on = _n_fz == 3
    _n_v2 = sum(t is not None for t in (fz_qbar, fz_kbar))
    _ls6_t = (ls6_ubar, ls6_z, ls6_zbar, ls6_aq, ls6_ak, ls6_fs, ls6_mh)
    _n_ls6 = sum(t is not None for t in _ls6_t)
    if _n_ls6 not in (0, len(_ls6_t)):
        raise ValueError(
            "LS6 인자는 일곱을 다 주거나 하나도 주지 말아야 한다 — 반쯤 켜지면 "
            f"조용히 틀린 값이 나온다 (지금 {_n_ls6}/{len(_ls6_t)})")
    _ls6_on = _n_ls6 == len(_ls6_t)
    _ls6_g = int(ls6_ubar.shape[-2]) if _ls6_on else 1
    _ls6_r = int(ls6_r) if _ls6_on else 0
    _corr_t = (ls6_beta, ls6_corr_ak, ls6_corr_kbar)
    _n_corr = sum(t is not None for t in _corr_t)
    if _n_corr not in (0, len(_corr_t)):
        raise ValueError(
            "LS6 exact-flush 인자는 beta/corr_ak/corr_kbar 세 개를 다 "
            f"주거나 하나도 주지 말아야 한다 (지금 {_n_corr}/3)"
        )
    if _n_corr and not _ls6_on:
        raise ValueError("LS6 래치 없이 exact-flush 버퍼만 줄 수 없다")
    _exact_env = os.environ.get("NS_GDN_LS6_EXACT_FLUSH", "")
    if _exact_env not in ("", "0", "1"):
        raise ValueError("NS_GDN_LS6_EXACT_FLUSH는 0 또는 1이어야 한다")
    # 새 host hook은 보정 버퍼를 주므로 기본 exact. 기존 low-level
    # LS6 호출은 버퍼가 없으면 근사 flush를 유지하되, 1을 명시하면 없이 못 돈다.
    _ls6_exact = _ls6_on and (
        _exact_env == "1" or (_exact_env == "" and _n_corr == len(_corr_t))
    )
    if _ls6_exact and _n_corr != len(_corr_t):
        raise ValueError(
            "LS6 exact-flush를 켰으므로 "
            "ls6_beta/ls6_corr_ak/ls6_corr_kbar가 필요하다. "
            "예전 근사 flush 재현은 NS_GDN_LS6_EXACT_FLUSH=0."
        )
    # 링이 delta-rule 갱신을 담으면 소거는 이미 그 안에 있다 — 기본 끔.
    _ls6_erase = bool(int(os.environ.get('NS_GDN_LS6_ERASE', '0')))
    if _ls6_on and (_ls6_g & (_ls6_g - 1)):
        raise ValueError(f"LS6_G 는 tl.arange 용이라 2의 거듭제곱이어야 한다: {_ls6_g}")
    if _n_v2 not in (0, 2):
        raise ValueError("fz_qbar/fz_kbar 는 둘 다 주거나 하나도 주지 말 것 "
                         f"(받은 개수 {_n_v2}). 하나만 주면 조용히 v1 로 돈다.")
    if _n_v2 == 2 and not _fz_on:
        raise ValueError("FZ_V2 는 freeze(fz_nf/u/z) 위에만 얹힌다")
    _fz_v2 = _n_v2 == 2
    # [nested_ssm 2026-08-25] 분기 특화: NS_GDN_SPLIT=1 이면 BR=1(비-flush)/BR=2(flush)
    # 로 **두 번 런치**한다. 각 런치는 자기 분기가 아닌 행을 조기 종료하므로 총 일은 같고,
    # flush 쪽 큰 타일이 비-flush 커널의 레지스터를 안 잡는다(Mamba2 KERNEL.md §9).
    # [nested_ssm 2026-08-25] 배포 형태는 **스텝마다 전 배치가 같은 분기**다
    # (엔진 링이 W=16 로 정렬돼 15/16 스텝은 전부 비-flush, 1/16 은 전부 flush).
    # Mamba2 의 NS_TWO_GRAPH 처럼 그래프를 두 벌 떠 두고 그 스텝에 필요한 쪽만
    # 런치하는 것이 배포 형태다. NS_GDN_BR 로 한쪽만 런치시켜 그 형태를 그대로 잰다.
    # 지정 안 하면 둘 다 런치한다 — 혼합 스텝용 안전판.
    _br_only = os.environ.get("NS_GDN_BR", "")
    _split = os.environ.get("NS_GDN_SPLIT", "1") == "1"
    _fx = os.environ.get("NS_GDN_FLUSH_EXACT", "1") == "1"
    # [nested_ssm 2026-08-25] 분기마다 **최적 런치 설정이 다르다**. 비-flush 는
    # 좁은 K타일(BKT=16)이 결정적이고(nf<BKT 여야 절감이 안 샌다), flush 는 K 를
    # 전부 읽으므로 넓은 타일 + 많은 워프가 낫다. 실측(bs=256, Qwen3.8 기하):
    #   비-flush nf=8 : BV64/nk8/w1/s3 = 187us   (BV64/nk2/w1/s3 기본값은 364us)
    #   flush        : BV32/nk2/w4/s2 = 1187us  (같은 기본값이면 1375us)
    # 분기 특화로 런치가 이미 둘이라 설정을 나누는 데 추가 비용이 없다.
    # flush 는 K 를 전부 읽으므로 nf 와 무관하고, 실측도 설정에 거의 둔감하다
    # (BV 16~128 에서 1189~1218us). 최적 64,2,4,2 = 1189.5us.
    # ⚠ 예전에 'BV=16 이 325us' 로 보였던 것은 **그리드를 분기별로 안 잡아** V 의
    #   1/4 만 처리한 결과였다. 일을 덜 해서 빨라 보였고 출력이 조용히 깨졌다.
    _flcfg = os.environ.get("NS_GDN_V1_FLUSH_CFG", "64,2,4,2")     # v1 전용(NS_GDN_KERNEL=v1)
    _fbv, _fnk, _fnw, _fns = (int(x) for x in _flcfg.split(","))
    _brs = (int(_br_only),) if _br_only else ((1, 2) if _split else (0,))
    # ⚠ flush-exact 는 **BR=2 런치에 FZ_ON=False 를 주는 것**으로 구현된다. 분기가
    #   안 갈리면(BR=0) 걸 자리가 없어서, 참조(_FLUSH_EXACT)만 exact 가 되고 커널은
    #   안 되는 **조용한 규약 불일치**가 생긴다. 실제로 verify_gdn_all 이 여기서
    #   상대 0.83 으로 걸렸다. 조용히 넘기지 않고 죽인다.
    if _fz_on and _fx and 0 in _brs:
        raise ValueError(
            "NS_GDN_SPLIT=0(BR=0)에서는 flush-exact 를 걸 수 없다 — "
            "NS_GDN_FLUSH_EXACT=0 으로 맞추거나 SPLIT=1 을 쓸 것. "
            "참조만 exact 가 되어 조용히 다른 결과가 나온다.")
    # [nested_ssm 2026-09-02] v2 커널(기본). NS_GDN_KERNEL=v1 로 이전 융합 커널.
    _kv = os.environ.get("NS_GDN_KERNEL", "v2")
    if _ls6_exact and _kv != "v2":
        raise ValueError("LS6 exact-flush는 NS_GDN_KERNEL=v2에만 구현되어 있다")
    if ls6_map is not None:
        if _kv != "v2":
            raise ValueError("ls6_map(슬롯 간접)은 v2 커널에만 있다")
        if ls6_map.dtype != torch.int32 or ls6_map.dim() != 1 or not ls6_map.is_contiguous():
            raise TypeError(f"ls6_map 은 연속 int32 (NX,) 여야 한다: {ls6_map.dtype} {tuple(ls6_map.shape)}")
        if ls6_map.shape[0] < initial_state.shape[0]:
            raise ValueError(f"ls6_map 길이 {ls6_map.shape[0]} < NX={initial_state.shape[0]}")
        for _nm, _t in (("ls6_ubar", ls6_ubar), ("ls6_aq", ls6_aq), ("ls6_ak", ls6_ak), ("ls6_fs", ls6_fs),
                        ("ls6_zk", ls6_zk), ("ls6_beta", ls6_beta), ("ls6_corr_ak", ls6_corr_ak),
                        ("ls6_corr_kbar", ls6_corr_kbar), ("fz_u", fz_u), ("fz_z", fz_z),
                        ("fz_qbar", fz_qbar), ("fz_kbar", fz_kbar)):
            if _t is not None and _t.shape[0] != ls6_ubar.shape[0]:
                raise ValueError(f"ls6_map 사용 시 슬롯 버퍼 dim0 은 모두 NS 로 같아야 한다: {_nm} {_t.shape[0]} != {ls6_ubar.shape[0]}")
    if _kv == "v2":
        if _ls6_erase:
            raise ValueError("LS6_ERASE 는 v2 에 없다 — NS_GDN_KERNEL=v1 로 돌릴 것")
        if _ls6_on and ls6_zk is None:
            raise ValueError("v2 LS6 는 ls6_zk (NX,H,W,G) 링이 필요하다 — 없으면 Ū 가 조용히 틀린다")
        if _ls6_on and tuple(ls6_zk.shape) != (ls6_ubar.shape[0], H, max_cache_len, _ls6_g):
            raise ValueError(f"ls6_zk 형상 {tuple(ls6_zk.shape)} != (NX,H,W,G)="
                             f"{(ls6_ubar.shape[0], H, max_cache_len, _ls6_g)}")
        if _ls6_exact:
            if not _fz_on:
                raise ValueError("LS6 exact-flush는 head별 hot prefix(fz_nf)가 필요하다")
            _exact_shapes = {
                "ls6_beta": (ls6_ubar.shape[0], HV, max_cache_len),
                "ls6_corr_ak": (ls6_ubar.shape[0], HV, _ls6_g),
                "ls6_corr_kbar": (ls6_ubar.shape[0], H, K),
            }
            for _nm, _t in zip(_exact_shapes, _corr_t):
                if tuple(_t.shape) != _exact_shapes[_nm]:
                    raise ValueError(
                        f"{_nm} 형상 {tuple(_t.shape)} != {_exact_shapes[_nm]}"
                    )
                if _t.dtype != torch.float32 or not _t.is_contiguous():
                    raise TypeError(
                        f"{_nm}은 연속 float32여야 한다: "
                        f"{_t.dtype}, contiguous={_t.is_contiguous()}"
                    )
            if any(
                t.dtype != torch.float32
                for t in (initial_state, d_cache, k_cache)
            ):
                raise TypeError(
                    "LS6 exact-flush는 state/d/k 링을 float32로 유지해야 한다"
                )
        _stcfg = os.environ.get("NS_GDN_V2_CFG", "128,4,1,2")      # 스텝: bv,nk,nw,ns
        _sbv, _snk, _snw, _sns = (int(x) for x in _stcfg.split(","))
        if block_v is not None:
            _sbv = block_v
        if nk is not None:
            _snk = nk
        if num_warps is not None:
            _snw = num_warps
        if num_stages is not None:
            _sns = num_stages
        _nk2 = max(1, min(_snk, BK // 16))
        while _nk2 > 1 and BK % _nk2 != 0:
            _nk2 -= 1
        _ls6_rp = max(16, triton.next_power_of_2(max(1, _ls6_r))) if _ls6_on else 16
        _ls6_gp = max(16, triton.next_power_of_2(_ls6_g)) if _ls6_on else 16
        # exact 보정은 dot 누산기가 많아 2 warp가 B300에서 낫다.
        _fl_default = "32,12,2" if _ls6_exact else "32,8,1"
        _flcfg = os.environ.get("NS_GDN_FLUSH_CFG", _fl_default)
        _fbv, _fps, _fnw = (int(x) for x in _flcfg.split(","))
        _fbv = min(_fbv, V)
        _nsplit = triton.cdiv(V, _fbv)
        _np = _sm_count() * _fps
        _common = dict(
            stride_fz_slot=(fz_u.stride(0) if _fz_on else 0),
            stride_fzb_slot=(fz_qbar.stride(0) if _fz_v2 else 0),
            stride_ls6_zk_slot=(ls6_zk.stride(0) if _ls6_on else 0),
            ls6_zk=ls6_zk if _ls6_on else mixed_qkv,
            stride_ls6_u_slot=(ls6_ubar.stride(0) if _ls6_on else 0),
            stride_init_state_token=initial_state.stride(0),
            stride_state_h=initial_state.stride(1),
            stride_state_v=initial_state.stride(2),
            stride_state_k=initial_state.stride(3),
            stride_indices_seq=ssm_state_indices.stride(0),
            stride_d_slot=d_cache.stride(0),
            stride_k_slot=k_cache.stride(0),
            stride_g_slot=g_cache.stride(0),
            H=H, HV=HV, K=K, V=V, BK=BK, BC=BC, MAX_CACHE_LEN=max_cache_len,
            FZ_V2=_fz_v2, LS6_ON=_ls6_on, LS6_G=_ls6_g)
        _brs2 = os.environ.get("NS_GDN_BR", "")
        _step_cuda = False
        if _brs2 != "2" and os.environ.get("NS_GDN_STEP_IMPL", "cuda") == "cuda":
            # CUDA 스텝(기본). Triton 판은 226~253 reg 로 점유 12.5% 지연 바운드였다 — gdn_step_cuda.py 머리말.
            from .gdn_step_cuda import gdn_step_cuda, gdn_step_supported
            _why = gdn_step_supported(mixed_qkv, out, initial_state, d_cache, k_cache, g_cache,
                                      ssm_state_indices, write_pos, H, HV, K, V, max_cache_len, _ls6_g, _ls6_on, _ls6_r)
            if _why is None:
                _step_cuda = True
                gdn_step_cuda(
                    B, mixed_qkv, a, b, A_log, dt_bias, out, initial_state, d_cache, k_cache, g_cache,
                    ssm_state_indices, write_pos, scale,
                    fz_nf if _fz_on else None, fz_u if _fz_on else None, fz_z if _fz_on else None,
                    fz_qbar if _fz_v2 else None, fz_kbar if _fz_v2 else None,
                    ls6_ubar if _ls6_on else None, ls6_z if _ls6_on else None, ls6_zbar if _ls6_on else None,
                    ls6_mh if _ls6_on else None, ls6_aq if _ls6_on else None, ls6_ak if _ls6_on else None,
                    ls6_zk if _ls6_on else None,
                    H, HV, K, V, max_cache_len, _ls6_g, _ls6_r, use_qk_l2norm_in_kernel,
                    os.environ.get("NS_GDN_EVICT", "1") == "1", ls6_map=ls6_map)
            elif not _STEP_FALLBACK_WARNED.get(_why):
                _STEP_FALLBACK_WARNED[_why] = True
                print(f"[gdn v2] CUDA 스텝 미지원({_why}) — Triton 스텝으로 되돌린다", flush=True)
        if _brs2 != "2" and not _step_cuda:
            _gdn_v2_step_kernel[(triton.cdiv(V, _sbv), B, HV)](
                mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, o=out,
                h0=initial_state,
                d_cache=d_cache, k_cache=k_cache, g_cache=g_cache,
                ssm_state_indices=ssm_state_indices, write_pos=write_pos, scale=scale,
                fz_nf=fz_nf if _fz_on else mixed_qkv,
                fz_u=fz_u if _fz_on else mixed_qkv,
                fz_z=fz_z if _fz_on else mixed_qkv,
                fz_qbar=fz_qbar if _fz_v2 else mixed_qkv,
                fz_kbar=fz_kbar if _fz_v2 else mixed_qkv,
                ls6_ubar=ls6_ubar if _ls6_on else mixed_qkv,
                ls6_z=ls6_z if _ls6_on else mixed_qkv,
                ls6_zbar=ls6_zbar if _ls6_on else mixed_qkv,
                ls6_aq=ls6_aq if _ls6_on else mixed_qkv,
                ls6_ak=ls6_ak if _ls6_on else mixed_qkv,
                ls6_mh=ls6_mh if _ls6_on else write_pos,
                ls6_map=ls6_map if ls6_map is not None else write_pos, LS6_MAP=ls6_map is not None,
                stride_ls6_a_slot=(ls6_aq.stride(0) if _ls6_on else 0),
                stride_mixed_qkv_tok=mixed_qkv.stride(0),
                stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
                BV=_sbv, NK=_nk2, BKT=BK // _nk2, SOFTPLUS_THRESHOLD=20.0,
                USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
                FZ_ON=_fz_on, LS6_R=_ls6_r, LS6_RP=_ls6_rp,
                EVICT=os.environ.get("NS_GDN_EVICT", "1") == "1",
                num_warps=_snw, num_stages=_sns, **_common)
        if _brs2 != "1":
            _flist = torch.empty(B + 1, dtype=torch.int32, device=write_pos.device)
            _gdn_v2_compact_kernel[(1,)](write_pos, ssm_state_indices, _flist, B,
                                         stride_indices_seq=ssm_state_indices.stride(0),
                                         MAX_CACHE_LEN=max_cache_len,
                                         BB=triton.next_power_of_2(B), num_warps=4)
            if not _ls6_exact and os.environ.get("NS_GDN_FLUSH_IMPL", "cuda") == "cuda" and _ls6_g <= 64 and max_cache_len <= 16 \
                    and initial_state.dtype == torch.float32 and d_cache.dtype == torch.float32:
                # CUDA flush(기본). Triton 판은 발행 바운드였다 — gdn_flush_cuda.py 머리말 참조.
                from .gdn_flush_cuda import gdn_flush_cuda
                _grid = int(os.environ.get("NS_GDN_FLUSH_GRID", "0")) or _sm_count() * 4
                gdn_flush_cuda(
                    _grid, initial_state, d_cache, k_cache, g_cache, ssm_state_indices, _flist, B,
                    fz_nf if _fz_v2 else None, fz_u if _fz_v2 else None, fz_z if _fz_v2 else None,
                    fz_qbar if _fz_v2 else None, fz_kbar if _fz_v2 else None,
                    ls6_ubar if _ls6_on else None, ls6_zk if _ls6_on else None,
                    H, HV, K, V, max_cache_len, _ls6_g, ls6_map=ls6_map)
                return out, initial_state
            _gdn_v2_flush_kernel[(_np,)](
                h0=initial_state, d_cache=d_cache, k_cache=k_cache, g_cache=g_cache,
                ssm_state_indices=ssm_state_indices, flush_list=_flist, n_ptr=_flist[B:],
                fz_nf=fz_nf if _fz_on else write_pos,
                fz_u=fz_u if _fz_on else mixed_qkv,
                fz_z=fz_z if _fz_on else mixed_qkv,
                fz_qbar=fz_qbar if _fz_v2 else mixed_qkv,
                fz_kbar=fz_kbar if _fz_v2 else mixed_qkv,
                ls6_ubar=ls6_ubar if _ls6_on else mixed_qkv,
                ls6_z=ls6_z if _ls6_on else mixed_qkv,
                ls6_mh=ls6_mh if _ls6_on else write_pos,
                ls6_beta=ls6_beta if _ls6_exact else mixed_qkv,
                ls6_corr_ak=ls6_corr_ak if _ls6_exact else mixed_qkv,
                ls6_corr_kbar=ls6_corr_kbar if _ls6_exact else mixed_qkv,
                ls6_map=ls6_map if ls6_map is not None else write_pos, LS6_MAP=ls6_map is not None,
                stride_ls6_beta_slot=(ls6_beta.stride(0) if _ls6_exact else 0),
                stride_ls6_corr_ak_slot=(ls6_corr_ak.stride(0) if _ls6_exact else 0),
                stride_ls6_corr_kbar_slot=(ls6_corr_kbar.stride(0) if _ls6_exact else 0),
                FBV=_fbv,
                NSPLIT=1 if _ls6_exact else _nsplit,
                VJOBS=_nsplit if _ls6_exact else 1,
                NP=_np,
                LS6_EXACT=_ls6_exact, LS6_GP=_ls6_gp,
                LS6_R=_ls6_r, LS6_RP=_ls6_rp,
                FDOT=int(os.environ.get("NS_GDN_FLUSH_DOT", "1")),   # 1 tf32x3(≈fp32), 2 ieee
                num_warps=_fnw, num_stages=1, **_common)
        return out, initial_state

    for _br in _brs:
        _isfl = _br == 2
        _bv = _fbv if _isfl else BV
        _nk = _fnk if _isfl else nk
        _nw = _fnw if _isfl else num_warps
        _ns = _fns if _isfl else num_stages
        _nk = max(1, min(_nk, BK // 16))
        while _nk > 1 and BK % _nk != 0:
            _nk -= 1
        grid = (triton.cdiv(V, _bv), B, HV)
        fused_recurrent_gated_delta_rule_replayssm_kernel[grid](
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, o=out,
            h0=initial_state, ht=initial_state,
            d_cache=d_cache, k_cache=k_cache, g_cache=g_cache,
            ssm_state_indices=ssm_state_indices, write_pos=write_pos, scale=scale,
            stride_mixed_qkv_tok=mixed_qkv.stride(0),
            stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
            stride_init_state_token=initial_state.stride(0),
            stride_final_state_token=initial_state.stride(0),
            stride_state_h=initial_state.stride(1),
            stride_state_v=initial_state.stride(2),
            stride_state_k=initial_state.stride(3),
            stride_indices_seq=ssm_state_indices.stride(0),
            stride_d_slot=d_cache.stride(0),
            stride_k_slot=k_cache.stride(0),
            stride_g_slot=g_cache.stride(0),
            H=H, HV=HV, K=K, V=V, BK=BK, BV=_bv, BC=BC, NK=_nk, BKT=BK // _nk,
            MAX_CACHE_LEN=max_cache_len, SOFTPLUS_THRESHOLD=20.0,
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            fz_nf=fz_nf if _fz_on else mixed_qkv,
            fz_u=fz_u if _fz_on else mixed_qkv,
            fz_z=fz_z if _fz_on else mixed_qkv,
            stride_fz_slot=(fz_u.stride(0) if _fz_on else 0),
            fz_qbar=fz_qbar if _fz_v2 else mixed_qkv,
            fz_kbar=fz_kbar if _fz_v2 else mixed_qkv,
            stride_fzb_slot=(fz_qbar.stride(0) if _fz_v2 else 0),
            # LS6 래치. 안 주면 LS6_ON=False 라 커널에서 통째로 사라진다(기존과 비트 동일).
            ls6_ubar=ls6_ubar if _ls6_on else mixed_qkv,
            ls6_z=ls6_z if _ls6_on else mixed_qkv,
            ls6_zbar=ls6_zbar if _ls6_on else mixed_qkv,
            ls6_aq=ls6_aq if _ls6_on else mixed_qkv,
            ls6_ak=ls6_ak if _ls6_on else mixed_qkv,
            ls6_fs=ls6_fs if _ls6_on else mixed_qkv,
            ls6_mh=ls6_mh if _ls6_on else write_pos,
            stride_ls6_u_slot=(ls6_ubar.stride(0) if _ls6_on else 0),
            stride_ls6_fs_slot=(ls6_fs.stride(0) if _ls6_on else 0),
            stride_ls6_a_slot=(ls6_aq.stride(0) if _ls6_on else 0),
            # ⚠ **끄지 않는다.** flush 는 Ū 를 접어야 하고 그건 읽기가 아니라 쓰기다.
            #   여기서 끄면 Ū 가 영영 0 이고, 그러면 state 를 전량 읽는 설정에서는
            #   답이 맞아 검사가 조용히 통과한다(실측: 그렇게 두 번 속았다).
            #   대신 **읽기 쪽**을 FZ_ON 으로 묶는다 — 아래 커널 주석 참조.
            LS6_ON=_ls6_on,
            LS6_G=_ls6_g,
            LS6_R=_ls6_r,
            LS6_ERASE=_ls6_erase,
            FZ_V2=_fz_v2,
            # flush 런치(BR=2)는 어차피 state 전체를 읽으므로 얼려서 아낄 트래픽이 0 이다.
            # FZ_ON=False 로 주면 정확도는 exact 로 올라가고 freeze 비용은 사라진다.
            FZ_ON=(_fz_on and not (_fx and _br == 2)),
            BR=_br,
            PREF=os.environ.get("NS_GDN_PREF", "1") == "1",
            EVICT=os.environ.get("NS_GDN_EVICT", "1") == "1",
            DOT_INPUT_PRECISION=os.environ.get("NS_GDN_DOT_PREC", "tf32"),
            num_warps=_nw, num_stages=_ns,
        )
    return out, initial_state
