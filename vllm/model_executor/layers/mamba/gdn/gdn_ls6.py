"""[nested_ssm 2026-09-02] LS6 래치(LatchSSM) — Flash-Next GDN 모델측 훅.

커널(third_party/flash_linear_attention/ops/{fused_recurrent_replayssm, gdn_step_cuda,
gdn_flush_cuda}.py, v8e+map)은 이미 래치 산술을 다 갖고 있다. 여기는 커널이 **안 하는**
호스트 몫이다 (scale/HANDOFF·bake_q38fn_ls6.py 머리말의 규약):

  ① 회전  q' = R q̂, k' = R k̂ (head 별, conv 뒤·l2norm 앞) — prefill 과 decode 양쪽.
          state 는 그러면 저절로 S' = S Rᵀ 로 쌓인다(GDN 갱신이 직교 R 에 공변).
  ② 슬롯 버퍼  Ū (NS,HV,G,V) 등을 **compact 슬롯** NS(≈max_num_seqs) 로 잡고
          ls6_map (NX,) 으로 mamba 블록 → compact 행을 잇는다. NX 는 Flash-Next 에서
          ~2.9k(페이지 등화로 attention 블록 수와 같다)라 직접 인덱싱은 Ū 만 80 GB 다.
          ⚠ 배정표는 **층마다** 따로다: vLLM 하이브리드 KV 캐시는 GDN 층을 여러 그룹으로
          나누고 그룹마다 블록표가 달라 같은 요청이 층마다 다른 블록 번호를 받는다. 전역
          배정표 하나를 leader 블록 번호로 채우면 다른 그룹 층은 미배정(행 0)을 읽는다 —
          2026-09-02 e2e 에서 완전 기저가 plain 과 안 맞던 원인.
  ③ prefill 시딩  Ū = S Z̄, q̄/k̄ = 마지막 ≤W 토큰 평균(창끝 규약), aq/ak = Zᵀq̄/Zᵀk̄,
          fz_u/z = S q̄_cold. prefill 청크마다 정확한 state 에서 다시 시딩한다.
  ④ decode 부기  창 합 누적 → flush 행(write_pos==W-1)에서 q̄ ← (1-λ)q̄ + λ·mean,
          aq/ak 재계산. **커널 호출 앞**에서 한다(flush 가 다음 창의 u/z 를 새 앵커로 쓴다).
          전부 고정 형상 device 연산 — CUDA 그래프에 그대로 잡힌다.
  ⑤ 혼합 웨이브(prefill+decode)의 decode 행도 링 커널로 보낸다 — dense 갱신으로 보내면
          write_pos 는 전진하는데 링은 비어 있어 다음 flush 가 틀린다.

기저: 배분기의 Ω 열은 E-직교(유클리드 직교가 아니다). 커널은 Z̄ 를 k-head 당 하나만
받고 같은 그룹의 v-head 들이 각자 m_h 만큼 앞 열을 쓰므로, Z = QR(Ω[:, :m_g]) 의 Q 로
바꿔 싣는다 — 접두 span 이 보존되고 직교라 Z̄ = Z 가 모든 m_h 에서 맞는다(Super 의
bake_super_embed 와 같은 처리).

켜기: NS_GDN_LS6=<q38fn_ls6_*_dn.pt>  (--use-replayssm, state fp32 필수)
  NS_GDN_LS6_NSLOT   compact 슬롯 수 (기본 max_num_seqs+8)
  NS_GDN_LS6_LAM     앵커 EMA λ (기본 0.45 — 캘리브와 같아야 한다)
  NS_GDN_LS6_EXACT_FLUSH=0  예전 근사 flush 재현용(기본 exact)
"""
import os
import re

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PATH = os.environ.get("NS_GDN_LS6", "")
_LAM = float(os.environ.get("NS_GDN_LS6_LAM", "0.45"))
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_CKPT = None


def armed() -> bool:
    return bool(_PATH)


def _ckpt():
    global _CKPT
    if _CKPT is None:
        d = torch.load(_PATH, map_location="cpu", weights_only=False)
        if d.get("ls6_family") != "gdn" or d.get("ls6_embedded", True):
            raise ValueError(f"NS_GDN_LS6={_PATH}: gdn 계열·명시 기저(ls6_embedded=False) ckpt 가 아니다")
        mt, nf, K = d["m_table"], d["nf_table"], int(d["K"])
        if not bool(((mt > 0) ^ (nf == K)).all()):
            raise ValueError("ckpt 불변식 (m>0) xor (nf==K) 위반 — bake_ls6_dense 를 안 거쳤다")
        idx = {}
        for li, nm in enumerate(d["layer_names"]):
            m = _LAYER_RE.search(nm)
            if m is None:
                raise ValueError(f"layer_names[{li}]={nm!r} 에서 층 번호를 못 읽는다")
            idx[int(m.group(1))] = li
        d["_layer_idx"] = idx
        _CKPT = d
    return _CKPT


class _Shared:
    """층 하나의 슬롯 배정표 (layer._ls6_sh). 블록 번호 공간이 층(그룹)마다 달라 층마다 하나씩이다."""

    def __init__(self):
        self.map = None       # (NX,) int32  블록 → compact 행 (0 = 미배정/패딩)
        self.owner = None     # (NS,) int64  compact 행 → 블록 (-1 = 빈 행)
        self.seen = None      # (NS,) int64  마지막으로 본 스텝
        self.step = None      # (1,) int64   device 스텝 카운터 (그래프 안에서 +1)
        self.never = None     # (1,) int64   1<<62 device 상수
        self.NS = 0

    def init(self, NX, NS, dev):
        self.map = torch.zeros(NX, dtype=torch.int32, device=dev)
        self.owner = torch.full((NS,), -1, dtype=torch.int64, device=dev)
        self.seen = torch.full((NS,), -1, dtype=torch.int64, device=dev)
        self.seen[0] = 1 << 62                                 # 행 0 = 패딩/쓰레기 행, 절대 안 준다
        self.owner[0] = 0
        self.step = torch.zeros(1, dtype=torch.int64, device=dev)
        self.never = torch.full((1,), 1 << 62, dtype=torch.int64, device=dev)   # 그래프 안에서 host 스칼라 복사 금지
        self.NS = NS


_LEADER = [None]          # 로그·진단 카운터용 첫 활성 층


def configure_layer(layer, prefix: str, vllm_config=None):
    layer._ls6_active = False
    if not armed():
        return
    m = _LAYER_RE.search(prefix)
    if m is None:
        raise ValueError(f"NS_GDN_LS6: prefix {prefix!r} 에서 층 번호를 못 읽는다")
    d = _ckpt()
    li = d["_layer_idx"].get(int(m.group(1)))
    if li is None:
        raise ValueError(f"NS_GDN_LS6: 층 {m.group(1)} 이 ckpt 의 layer_names 에 없다")
    if vllm_config is not None:
        if not vllm_config.cache_config.use_replayssm:
            raise RuntimeError("NS_GDN_LS6 는 --use-replayssm 위에만 얹힌다")
        if vllm_config.speculative_config is not None:
            raise RuntimeError("NS_GDN_LS6 는 speculative decoding 을 지원하지 않는다")
        if vllm_config.cache_config.mamba_ssm_cache_dtype != "float32":
            raise RuntimeError("NS_GDN_LS6: state 는 fp32 규약 (--mamba-ssm-cache-dtype float32)")
    if getattr(layer, "tp_size", 1) != 1:
        raise RuntimeError("NS_GDN_LS6 는 TP=1 만 (head 분할 미구현)")
    if getattr(layer, "_gdn_prune_active", False):
        raise RuntimeError("NS_GDN_LS6 와 NS_GDN_PRUNE 는 같이 못 켠다")
    K, V, W, r, rep = int(d["K"]), int(d["V"]), int(d["W"]), int(d["r"]), int(d["rep"])
    HK, HV = layer.num_k_heads, layer.num_v_heads
    if (layer.head_k_dim, layer.head_v_dim, HV // HK) != (K, V, rep):
        raise ValueError(f"ckpt 기하 (K,V,rep)=({K},{V},{rep}) != 층 ({layer.head_k_dim},{layer.head_v_dim},{HV // HK})")
    if vllm_config is not None and W != int(vllm_config.cache_config.replayssm_buffer_len):
        raise ValueError(f"ckpt W={W} != replayssm_buffer_len={vllm_config.cache_config.replayssm_buffer_len}")
    mt = d["m_table"][li].long()                       # (HV,)
    nf = d["nf_table"][li].long()
    mg = d["m_group"][li].long()                       # (HK,)
    if int(mt.max()) == 0:
        # 전부 dense 인 층: 래치도 회전도 필요 없다 — 순정 replay 경로 그대로.
        layer._ls6_active = False
        logger.info("[ls6] %s: 층 전체 dense — 훅 없음", prefix)
        return
    G = max(4, 1 << (int(mt.max()) - 1).bit_length())
    if G > 64 and os.environ.get("NS_GDN_STEP_IMPL", "cuda") == "cuda":
        # Triton step/flush 는 G=128 도 받는다(검증용 완전 기저) — 배포는 CUDA 커널이라 64 가 천장.
        raise ValueError(f"층 {li}: m 최대 {int(mt.max())} → G={G} > 64 (CUDA step/flush 한계). --mmax 64 로 배분할 것")
    OM = d["omega"][li].double()                       # (HK,MM,K)
    # 모델 구성은 default device=cuda 문맥에서 돈다 — ckpt 텐서(CPU)와 섞이지 않게 CPU 를 박는다.
    Z = torch.zeros(HK, K, G, dtype=torch.float64, device="cpu")
    for h in range(HK):
        m_g = int(mg[h])
        if m_g == 0:
            continue
        Q, Rq = torch.linalg.qr(OM[h, :m_g, :].T)      # (K,m_g): 접두 span 보존 + 직교
        Q = Q * torch.sign(torch.diagonal(Rq)).clamp(min=0).mul(2).sub(1)[None, :]
        Z[h, :, :m_g] = Q
    dev = torch.device("cuda")
    R = d[li].float()                                   # (HK,K,K) 행=방향
    layer._ls6 = dict(
        li=li, K=K, V=V, W=W, r=r, rep=rep, HK=HK, HV=HV, G=G,
        R=R.to(dev),
        Z=Z.float().to(dev).contiguous(), Zbar=Z.float().to(dev).contiguous(),
        Zrep=Z.float().repeat_interleave(rep, 0).to(dev).contiguous(),        # (HV,K,G)
        mh=mt.to(torch.int32).to(dev).contiguous(), nf=nf.to(torch.int32).to(dev).contiguous(),
        cold=(torch.arange(K, device="cpu")[None, :] >= nf[:, None]).float().to(dev),         # (HV,K)
        ns_env=int(os.environ.get("NS_GDN_LS6_NSLOT", "0")),
        max_num_seqs=int(vllm_config.scheduler_config.max_num_seqs) if vllm_config is not None else 0,
    )
    layer._ls6_bufs = None
    layer._ls6_sh = _Shared()
    layer._ls6_active = True
    layer._ls6_name = f"L{li}"
    if _LEADER[0] is None:
        _LEADER[0] = prefix
    layer._ls6_leader = _LEADER[0] == prefix
    logger.info("[ls6] %s: ckpt 층 %d  래치 head %d/%d  m max %d  G %d  r %d  W %d%s",
                prefix, li, int((mt > 0).sum()), HV, int(mt.max()), G, r, W, "  (leader)" if layer._ls6_leader else "")


# ─────────────────────────── ① 회전 ───────────────────────────
def rotate_mixed(layer, mixed_qkv):
    """conv 뒤 q,k 를 head 별로 제자리 회전 q' = R q. (T, 2·HK·K + HV·V) 의 앞 2·HK·K 만."""
    if mixed_qkv is None or not getattr(layer, "_ls6_active", False):
        return
    L6 = layer._ls6
    HK, K = L6["HK"], L6["K"]
    T = mixed_qkv.shape[0]
    qk = mixed_qkv[:, : 2 * HK * K].view(T, 2, HK, K)
    # fp32 로 곱하고 버퍼 dtype 으로 내린다 — bf16 곱은 회전 자체가 오차 3e-3 을 얹는다.
    qk.copy_(torch.einsum("hij,tzhj->tzhi", L6["R"], qk.float()).to(qk.dtype))


# ─────────────────────────── ② 슬롯 버퍼 ───────────────────────────
def _bufs(layer, ssm_state):
    b = layer._ls6_bufs
    sh = layer._ls6_sh
    NX = ssm_state.shape[0]
    # 프로파일링 더미 실행은 NX=4 짜리 임시 state 로 돈다 — 실제 KV 캐시(NX≈수천)가 오면
    # 배정표·버퍼를 버리고 다시 잡는다(그래프 캡처 전이라 안전).
    if sh.map is not None and sh.map.shape[0] != NX:
        if layer._ls6_leader:
            logger.info("[ls6] NX %d → %d: 슬롯 배정표·버퍼 재할당", sh.map.shape[0], NX)
        sh.map = None
        b = None
    if b is not None and b["ubar"].shape[0] == sh.NS:
        return b
    L6 = layer._ls6
    dev = ssm_state.device
    if sh.map is None:
        NS = L6["ns_env"] or (L6["max_num_seqs"] + 8)
        NS = min(NS, NX)
        sh.init(NX, NS, dev)
        if layer._ls6_leader:
            logger.info("[ls6] 슬롯 배정표: NX=%d → compact NS=%d (층마다)", NX, NS)
    NS = sh.NS
    HK, HV, K, V, W, G = (L6[k] for k in ("HK", "HV", "K", "V", "W", "G"))
    f32 = dict(dtype=torch.float32, device=dev)
    b = dict(
        ubar=torch.zeros(NS, HV, G, V, **f32),
        aq=torch.zeros(NS, HV, G, **f32), ak=torch.zeros(NS, HV, G, **f32),
        fs=torch.zeros(NS, HV, W, G, **f32), zk=torch.zeros(NS, HK, W, G, **f32),
        beta=torch.zeros(NS, HV, W, **f32),
        corr_ak=torch.zeros(NS, HV, G, **f32),
        corr_kbar=torch.zeros(NS, HK, K, **f32),
        fz_u=torch.zeros(NS, HV, V, **f32), fz_z=torch.zeros(NS, HV, V, **f32),
        qbar=torch.zeros(NS, HK, K, **f32), kbar=torch.zeros(NS, HK, K, **f32),
        qacc=torch.zeros(NS, HK, K, **f32), kacc=torch.zeros(NS, HK, K, **f32),
        cnt=torch.zeros(NS, **f32),
    )
    layer._ls6_bufs = b
    if layer._ls6_leader:
        tot = sum(t.numel() * 4 for t in b.values())
        logger.info("[ls6] 슬롯 버퍼 층당 %.0f MB (Ū %.0f MB, G=%d)", tot / 2**20, b["ubar"].numel() * 4 / 2**20, G)
    return b


def _assign_slots(sh, pf_idx, has_init):
    """층마다. 새 시퀀스(has_init=False)에 compact 행을 준다: 이미 그 블록이 행을 갖고 있으면
    재사용, 아니면 가장 오래 안 본 행(LRU)을 뺏는다. 살아 있는 요청은 스텝마다 보이므로 희생자는
    죽은 행이다 — 그래도 직전 스텝에 보인 행을 뺏게 되면 NS 가 모자란 것이니 죽인다."""
    xs = pf_idx.tolist()
    hi = has_init.tolist()
    step = int(sh.step.item())
    for x, h in zip(xs, hi):
        if x <= 0:
            continue
        c = int(sh.map[x])
        if c > 0 and int(sh.owner[c]) == x:
            sh.seen[c] = step
            continue
        if h:
            raise RuntimeError(f"[ls6] 블록 {x} 의 prefill 이어붙임인데 compact 행이 없다 — 슬롯이 뺏겼다(NS 부족)")
        v = int(torch.argmin(sh.seen))
        if int(sh.seen[v]) >= step - 1 and int(sh.owner[v]) >= 0:
            raise RuntimeError(f"[ls6] compact 슬롯 부족: 희생자 행 {v} 가 스텝 {int(sh.seen[v])} (지금 {step}) 에 보였다. "
                               f"NS_GDN_LS6_NSLOT 을 올릴 것 (지금 {sh.NS})")
        old = int(sh.owner[v])
        if old > 0:
            sh.map[old] = 0
        sh.owner[v] = x
        sh.map[x] = v
        sh.seen[v] = step


# ─────────────────────────── ③ prefill 시딩 ───────────────────────────
def _qk_hat(qk, scale):
    """(…,2,HK,K) → 커널 단위의 q̂ (l2norm × scale), k̂ (l2norm)."""
    q, k = qk[..., 0, :, :].float(), qk[..., 1, :, :].float()
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    return q, k


@torch.no_grad()
def seed_prefill(layer, ssm_state, pf_idx, has_init, mixed_pf, cu_seqlens, scale):
    """prefill 청크 뒤(ssm_state[pf_idx] 가 정확한 state 로 갱신된 뒤). mixed_pf 는 **회전된**
    conv 출력의 prefill 토큰 부분, cu_seqlens 는 그 안의 경계."""
    if not getattr(layer, "_ls6_active", False):
        return
    L6 = layer._ls6
    b = _bufs(layer, ssm_state)
    sh = layer._ls6_sh
    _assign_slots(sh, pf_idx, has_init)
    HK, K, W, rep = L6["HK"], L6["K"], L6["W"], L6["rep"]
    x = pf_idx.long()
    c = sh.map[x].long()                                                     # (n,)
    cu = cu_seqlens.long()
    st, en = cu[:-1], cu[1:]
    pos = en[:, None] - W + torch.arange(W, device=x.device)[None, :]         # (n,W) 창끝 W 토큰
    valid = pos >= st[:, None]
    pos = pos.clamp(min=0, max=max(int(mixed_pf.shape[0]) - 1, 0))
    qk = mixed_pf[:, : 2 * HK * K].view(-1, 2, HK, K)
    qh, kh = _qk_hat(qk[pos], scale)                                         # (n,W,HK,K)
    m = valid.to(qh.dtype)[:, :, None, None]
    cnt = valid.sum(1).clamp(min=1).to(qh.dtype)[:, None, None]
    qbar = (qh * m).sum(1) / cnt
    kbar = (kh * m).sum(1) / cnt                                             # (n,HK,K)
    b["qbar"][c] = qbar
    b["kbar"][c] = kbar
    b["qacc"][c] = 0
    b["kacc"][c] = 0
    b["cnt"][c] = 0
    Z = L6["Z"]
    b["aq"][c] = torch.einsum("hkg,nhk->nhg", Z, qbar).repeat_interleave(rep, 1)
    b["ak"][c] = torch.einsum("hkg,nhk->nhg", Z, kbar).repeat_interleave(rep, 1)
    b["corr_ak"][c] = b["ak"][c]
    b["corr_kbar"][c] = kbar
    S = ssm_state[x].float()                                                 # (n,HV,V,K)
    b["ubar"][c] = torch.einsum("nhvk,hkg->nhgv", S, L6["Zrep"])
    cold = L6["cold"]
    b["fz_u"][c] = torch.einsum("nhvk,nhk->nhv", S, qbar.repeat_interleave(rep, 1) * cold)
    b["fz_z"][c] = torch.einsum("nhvk,nhk->nhv", S, kbar.repeat_interleave(rep, 1) * cold)
    b["fs"][c] = 0
    b["zk"][c] = 0
    b["beta"][c] = 0
    del S


# ─────────────────────────── ④ decode 부기 ───────────────────────────
@torch.no_grad()
def decode_bookkeep(
    layer, ssm_state, mixed_dec, beta_logits, idx, write_pos, scale
):
    """커널 호출 **앞**. 고정 형상·device 연산만(그래프 캡처 가능). 패딩 행(idx≤0)은 행 0 으로."""
    if not getattr(layer, "_ls6_active", False):
        return None
    L6 = layer._ls6
    b = _bufs(layer, ssm_state)
    HK, K, W, rep = L6["HK"], L6["K"], L6["W"], L6["rep"]
    sh = layer._ls6_sh
    x = idx.long().clamp(min=0)
    c = sh.map[x].long()                                                     # (T,)  미배정 → 0
    T = x.shape[0]
    sh.step += 1
    sh.seen[c] = sh.step
    sh.seen[0:1].copy_(sh.never)                      # 행 0 (패딩) 은 LRU 후보에서 항상 뺀다 (device 상수)
    qk = mixed_dec[:, : 2 * HK * K].view(T, 2, HK, K)
    qh, kh = _qk_hat(qk, scale)                                              # (T,HK,K)
    b["qacc"].index_add_(0, c, qh)
    b["kacc"].index_add_(0, c, kh)
    b["cnt"].index_add_(0, c, torch.ones_like(c, dtype=torch.float32))
    beta = torch.sigmoid(beta_logits.float()).to(beta_logits.dtype).float()
    b["beta"][c, :, write_pos.long()] = beta
    fl = (write_pos == (W - 1))                                              # (T,) flush 행
    cnt = b["cnt"][c].clamp(min=1)[:, None, None]
    qb_old, kb_old = b["qbar"][c], b["kbar"][c]
    qb_new = (1 - _LAM) * qb_old + _LAM * (b["qacc"][c] / cnt)
    kb_new = (1 - _LAM) * kb_old + _LAM * (b["kacc"][c] / cnt)
    f3 = fl[:, None, None]
    b["corr_kbar"][c] = torch.where(f3, kb_old, b["corr_kbar"][c])
    ak_old = b["ak"][c]
    b["corr_ak"][c] = torch.where(f3, ak_old, b["corr_ak"][c])
    qb = torch.where(f3, qb_new, qb_old)
    kb = torch.where(f3, kb_new, kb_old)
    b["qbar"][c] = qb
    b["kbar"][c] = kb
    b["qacc"][c] = torch.where(f3, torch.zeros_like(qb), b["qacc"][c])
    b["kacc"][c] = torch.where(f3, torch.zeros_like(kb), b["kacc"][c])
    b["cnt"][c] = torch.where(fl, torch.zeros_like(cnt[:, 0, 0]), b["cnt"][c])
    Z = L6["Z"]
    b["aq"][c] = torch.einsum("hkg,thk->thg", Z, qb).repeat_interleave(rep, 1)
    b["ak"][c] = torch.einsum("hkg,thk->thg", Z, kb).repeat_interleave(rep, 1)
    return kernel_kwargs(layer, ssm_state)


def kernel_kwargs(layer, ssm_state):
    """fused_recurrent_gated_delta_rule_replayssm 에 넘길 LS6/freeze 인자."""
    if not getattr(layer, "_ls6_active", False):
        return {}
    L6 = layer._ls6
    b = _bufs(layer, ssm_state)
    return dict(
        ls6_ubar=b["ubar"], ls6_z=L6["Z"], ls6_zbar=L6["Zbar"], ls6_aq=b["aq"], ls6_ak=b["ak"],
        ls6_fs=b["fs"], ls6_mh=L6["mh"], ls6_zk=b["zk"], ls6_r=L6["r"], ls6_map=layer._ls6_sh.map,
        ls6_beta=b["beta"], ls6_corr_ak=b["corr_ak"], ls6_corr_kbar=b["corr_kbar"],
        fz_nf=L6["nf"], fz_u=b["fz_u"], fz_z=b["fz_z"], fz_qbar=b["qbar"], fz_kbar=b["kbar"],
    )


# ─────────────────────────── 진단: 층별 dense 대조 ───────────────────────────
# NS_GDN_LS6_CHECK=1 이면 decode 스텝마다 같은 입력을 **LS6 없이**(dense, 행 복사본) 한 번 더
# 돌려 층별 |Δ| 를 로그한다. 완전 기저(m=K, r=K, Z=I) 면 0 에 가까워야 한다. eager 전용.
CHECK = bool(os.environ.get("NS_GDN_LS6_CHECK", ""))
_CHK_STEPS = int(os.environ.get("NS_GDN_LS6_CHECK_STEPS", "40"))
_chk_n = [0]


@torch.no_grad()
def check_begin(layer, ssm_state, d_cache, k_cache, g_cache, mixed, a, b, idx, write_pos, scale):
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent_replayssm import (
        fused_recurrent_gated_delta_rule_replayssm as _gdn)

    if not getattr(layer, "_ls6_active", False):
        return None
    if layer._ls6_leader:
        _chk_n[0] += 1
    if _chk_n[0] > _CHK_STEPS:
        return None
    x = idx.long()
    valid = x > 0
    xc = x.clamp(min=0)
    B = x.shape[0]

    def gather(t):
        z = torch.zeros((1,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
        return torch.cat([z, t[xc].clone()], 0)

    S, D, Kc, G = gather(ssm_state), gather(d_cache), gather(k_cache), gather(g_cache)
    ridx = torch.where(valid, torch.arange(1, B + 1, device=x.device), torch.zeros_like(x)).to(idx.dtype)
    out = torch.empty(B, 1, layer.num_v_heads // layer.tp_size, layer.head_v_dim, dtype=mixed.dtype, device=mixed.device)
    _gdn(mixed_qkv=mixed, a=a, b=b, A_log=layer.A_log, dt_bias=layer.dt_bias, scale=scale, initial_state=S,
         d_cache=D, k_cache=Kc, g_cache=G, out=out, ssm_state_indices=ridx, write_pos=write_pos,
         use_qk_l2norm_in_kernel=True)
    # 버퍼 일관성: Ū 는 flush 사이에 S·Z̄ 여야 한다 (state 는 flush 에서만 바뀐다)
    L6 = layer._ls6
    b_ = _bufs(layer, ssm_state)
    c = layer._ls6_sh.map[xc].long()
    ue = torch.einsum("nhvk,hkg->nhgv", S[1:], L6["Zrep"])
    du = ((b_["ubar"][c] - ue).flatten(1).abs().max(1).values / ue.flatten(1).abs().max(1).values.clamp(min=1e-6))
    return dict(S=S, out=out, x=x, valid=valid, wp=write_pos, step=_chk_n[0], du=du[valid].tolist(), c=c.tolist())


@torch.no_grad()
def check_end(layer, chk, ssm_state, out_ls6):
    if chk is None:
        return
    W = layer._ls6["W"]
    _MANT = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24}[out_ls6.dtype]
    ref = chk["out"].float().flatten(1)[chk["valid"]]
    got = out_ls6.float().flatten(1)[chk["valid"]]
    e = ((got - ref).abs().max(1).values / ref.abs().max(1).values.clamp(min=1e-6)).tolist()
    # 출력은 모델 dtype(bf16) 으로 저장되므로 |Δy| 는 반올림 경계에서 1 ulp 가 튄다 — 원소별 ulp 로 나눈
    # 최대값과 다른 원소 수를 같이 찍는다. 항등 구성이면 "≤1 ulp, 몇 개" 여야 한다.
    ulp = torch.ldexp(torch.ones_like(ref), torch.frexp(ref.clamp(min=1e-30).abs())[1] - _MANT)
    eu = ((got - ref).abs() / ulp).max(1).values.tolist()
    nd = (got != ref).sum(1).tolist()
    fl = (chk["wp"] == W - 1) & chk["valid"]
    es = ""
    if bool(fl.any()):
        xs = chk["x"][fl]
        Sg = ssm_state[xs].float()
        Sr = chk["S"][1:][fl].float()                                        # 참조 행 i+1 ↔ 배치 행 i
        es = "  flush ΔS/|S| " + " ".join(f"{v:.1e}" for v in ((Sg - Sr).flatten(1).abs().max(1).values / Sr.flatten(1).abs().max(1).values.clamp(min=1e-6)).tolist())
    logger.info("[ls6-check] step %d %s wp %s slot %s |Δy|/|y| %s  ulp %s  n≠ %s  ΔŪ %s%s", chk["step"],
                layer._ls6_name, chk["wp"].tolist(), chk["c"], " ".join(f"{v:.1e}" for v in e),
                " ".join(f"{v:.1f}" for v in eu), " ".join(str(int(v)) for v in nd),
                " ".join(f"{v:.1e}" for v in chk["du"]), es)
