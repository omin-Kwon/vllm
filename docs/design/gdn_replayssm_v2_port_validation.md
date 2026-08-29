# Qwen3.8-Flash-Next GDN ReplaySSM 포팅 및 검증 기록

## 목적과 범위

이 문서는 Qwen3.8-Flash-Next(`qwen4_exp`)의 GDN(Gated DeltaNet) 디코드
경로에 ReplaySSM을 포팅하면서 발견하고 수정한 문제, 최종 구현 방식, 실제 GPU
검증 결과를 정리한다.

대상 저장소와 기준은 다음과 같다.

- 저장소: `/disk2/omin/vllm-qwen38next`
- 브랜치: `qwen38next`
- 베이스: `89d0bb71a` (Qwen3.8-Flash-Next 지원 PR 기준)
- 최종 수정 커밋: `9df6640e8` (`Fix GDN ReplaySSM cursor wiring in v2 runner`)
- 검증 모델: `/disk2/models/Qwen3.8-Flash-Next-NVFP4`
- 검증 GPU: NVIDIA B300 1장, TP1

이 작업은 GDN ReplaySSM의 수학이나 Triton 계산식을 새로 설계한 작업이 아니다.
기존 구현을 새 vLLM 트리에 연결하고, v2 모델 러너에서 요청별 ring cursor가 정확히
전달되도록 배선을 수정한 작업이다.

## 관련 커밋

| 커밋 | 내용 |
| --- | --- |
| `829661230` | GDN ReplaySSM 최초 포팅 |
| `af6d7c3e2` | CUDA graph capture 시 메타데이터 생성 허용 |
| `b2839ae6c` | 1차 정적 검토에서 발견한 cache/state 배선 문제 수정 |
| `18e61a1bb` | ReplaySSM ring이 없는 Mamba 계층의 공유 builder 보호 |
| `c8f62fa7f` | 실제 디코드에서 cursor를 추측하지 않고 fail-closed 처리 |
| `9df6640e8` | v2 러너의 GDN ReplaySSM cursor 배선과 회귀 테스트 완성 |

## 최초 증상과 원인

ReplaySSM OFF/ON을 `temperature=0`으로 비교했을 때 ON 출력이 다음과 같이
같은 구절을 반복했다.

```text
OFF: Paris. The capital of Germany is Berlin. ...
ON : Paris. The capital of France is Paris.
     The capital of France is Paris. ...
```

원인은 `af6d7c3e2`에서 넣은 `write_pos=0` 폴백이었다.

- `replayssm_decode_base_cpu`는 기존 v1 러너에서만 채워지고 있었다.
- Qwen3.8-Flash-Next는 v2 러너를 사용하므로 실제 디코드에서도 해당 값이
  항상 `None`이었다.
- CUDA graph 캡처용 더미 메타데이터에만 사용될 것으로 가정한 폴백이 실제
  디코드마다 발동했다.
- 결과적으로 모든 토큰이 ring의 0번 위치를 덮어써 recurrent state가 정상적으로
  전진하지 않았다.

즉, 캡처 중 NULL state slot에 대한 `write_pos=0` 자체는 무해하지만,
"그 폴백은 캡처 중에만 실행된다"는 전제가 틀렸다. 조용한 오답을 막기 위해 현재
구현은 필요한 CPU 메타데이터가 없으면 명시적으로 실패한다.

## 최종 cursor 설계

### 요청별 anchor와 cursor

v2 러너의 CPU 입력 상태에서 다음 두 값을 매 스텝 attention metadata로 전달한다.

- `prefill_len_np`: 해당 요청의 현재 prefill 길이이자 ReplaySSM ring anchor
- `num_computed_tokens_np`: 현재까지 계산된 토큰 수

GDN ReplaySSM의 쓰기 위치는 다음과 같이 계산한다.

```text
write_pos = (num_computed_tokens - prefill_len) % buffer_len
```

이 anchor는 일반적인 ring flush마다 갱신하지 않는다. ReplaySSM은 ring이 찼을 때
checkpoint state를 전진시키고 ring을 다시 0부터 재사용하므로 modulo cursor만으로
충분하다. 요청이 재등록되는 경우에는 새 `prefill_len`을 anchor로 사용한다.

### CUDA graph에서 CPU cursor를 사용한다는 의미

CUDA graph 내부가 CPU 메모리를 직접 읽는 것은 아니다. 흐름은 다음과 같다.

```text
v2 CPU input state
  -> 요청별 write_pos 계산
  -> 고정 주소 GPU buffer(decode_write_pos_d)로 복사
  -> 캡처된 GDN ReplaySSM kernel이 그 buffer의 최신 값을 읽음
```

CUDA graph는 GPU buffer 주소를 고정해서 캡처하지만, 그 주소에 담긴 값은 replay
직전에 갱신할 수 있다. 따라서 요청별 cursor가 바뀌어도 graph를 다시 캡처할 필요가
없다.

full CUDA graph에서 batch padding으로 추가된 더미 row에는 CPU 배열을 0으로
padding한다. 이 row는 유효한 state slot을 갖지 않으며 kernel의 state index 검사에서
계산 대상에서 제외된다.

### 단일 토큰 prefill과 지원하지 않는 모드

GDN ReplaySSM kernel에는 Mamba2 ReplaySSM의 `is_flush`와 같은 독립적인 강제 flush
입력이 없다. 따라서 아직 prefill 중인 한 토큰짜리 prompt tail은 토큰 수만 보고
decode로 오인하지 않고 `is_prefilling`을 기준으로 prefill 경로에 남긴다.

같은 이유로 별도 flush 의미가 필요한 align mode는 임의로 지원하지 않고
`NotImplementedError`로 fail-closed 처리한다.

## GDN ReplaySSM kernel 변경 범위

오염되지 않은 원본의 기준 파일은 다음이다.

```text
/disk2/omin/miniconda3/envs/vllm_gdn/lib/python3.12/site-packages/vllm/
model_executor/layers/fla/ops/fused_recurrent_replayssm.py.ns027orig
```

현재 Triton kernel과 이 파일을 대조했을 때 계산 경로의 변경은 없다. 추가된 것은
호출 전 Python 입력 검증 12줄뿐이다.

- SSM state index의 차원과 dtype 검사
- batch 길이 일치 검사
- tensor device 일치 검사

다음 항목은 원본 그대로 유지했다.

- replay 수식과 recurrent checkpoint 갱신
- BF16 `d`/`k` ring과 FP32 `g` ring
- flush 시점과 ring 재사용 방식
- block tiling, `tl.dot`, launch configuration

구 vLLM 환경의 접미사 없는 현재 파일에는 `nested_ssm`, freeze, pruning/quantization
연구 코드가 섞여 있다. 따라서 포팅 정합성을 검토할 때는 그 파일이 아니라
`.ns027orig`를 기준으로 해야 한다.

## 최종 수정 파일

`9df6640e8`에서 변경한 핵심 파일은 다음과 같다.

- `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`
- `vllm/v1/attention/backends/gdn_attn.py`
- `vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py`
- `tests/v1/attention/test_replayssm_metadata_builder.py`
- `tests/v1/attention/test_gdn_metadata_builder.py`
- `tests/v1/worker/test_mamba_hybrid_model_state.py`

## 자동화 테스트 결과

### CPU metadata 및 공유 경로 회귀 검사

```bash
PYTHONPATH=/disk2/omin/miniconda3/envs/vllm029_q38next/lib/python3.12/site-packages \
  .venv/bin/python -m pytest -q \
  tests/v1/attention/test_replayssm_metadata_builder.py \
  tests/v1/attention/test_gdn_metadata_builder.py \
  tests/v1/worker/test_mamba_hybrid_model_state.py
```

결과: `31 passed, 14 warnings`

여기에는 v2 cursor 배선, CUDA graph padding, 단일 토큰 prefill, 요청 재등록,
Qwen3.5 호환성, Mamba2 공유 경로 회귀 검사가 포함된다.

### GPU Triton kernel 검사

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/disk2/omin/miniconda3/envs/vllm029_q38next/lib/python3.12/site-packages \
  .venv/bin/python -m pytest -q \
  tests/kernels/test_fused_recurrent_packed_decode.py
```

결과: `10 passed, 14 warnings`

Ruff 검사와 `git diff --check`도 통과했다.

## 실제 Qwen3.8-Flash-Next GPU 검증

### 공통 실행 조건

- v2 model runner
- TP1
- `max_model_len=2048`
- `max_num_batched_tokens=2048`
- `max_num_seqs=8`
- prefix caching 비활성화
- FlashInfer autotune 비활성화
- `mamba_cache_mode=none`
- `mamba_ssm_cache_dtype=float32`
- ReplaySSM ON은 `buffer_len=16`, Triton Mamba backend 사용

OFF와 ON 모두 실제 모델 부팅과 4개 prompt의 40-token 생성에 성공했다.
기존의 동일 구절 반복 증상은 사라졌다.

### 메모리 결과

두 실행 모두 vLLM이 KV cache에 사용할 수 있다고 계산한 메모리는 `85.17 GiB`였다.
다만 ReplaySSM ON은 `d`/`k`/`g` ring page가 추가되므로 같은 byte budget에서 수용
가능한 token 수가 줄었다.

| 설정 | GPU KV cache token 수 | 최대 동시성 표시 |
| --- | ---: | ---: |
| OFF | 673,206 | 328.71x |
| ON | 622,299 | 303.86x |

token capacity 감소율은 약 7.56%다. 이는 처리속도 감소율이 아니라 cache page당
메모리 증가를 반영한 값이다. 정식 throughput/TTFT/TPOT benchmark는 아직 수행하지
않았다.

새 compile hash로 처음 부팅한 한 차례는 `torch.compile`의 일회성 peak 때문에 KV
cache 계산이 약 `-9.99 GiB`가 되어 실패했다. compile cache가 준비된 뒤 같은 설정을
다시 실행하면 정상 부팅했다. 이는 cursor 로직과 별개의 운영상 주의점이다.

## OFF/ON 결과가 완전히 같지 않은 이유

수정 후 greedy exact match는 CUDA graph와 eager 모두 4개 중 1개였다. eager에서도
같은 수준이므로 CUDA graph cursor 전달 문제로 볼 수 없다.

첫 prefill token의 log probability는 OFF/ON이 정확히 일치했다. 차이는 ReplaySSM
decode가 시작된 뒤 누적된다.

- OFF는 매 token마다 전체 FP32 recurrent state를 직접 갱신한다.
- ON은 FP32 checkpoint를 유지하면서 BF16 `d`/`k`와 FP32 `g`를 ring에 저장하고,
  필요할 때 state를 재구성한다.
- 수학적으로 같은 업데이트여도 BF16 저장과 연산 순서 차이 때문에 작은 수치 오차가
  생긴다.
- 이 차이가 36개 GDN layer와 큰 MoE 경로를 통과하면서 증폭되고, 확률이 비슷한
  token의 greedy argmax를 바꿀 수 있다.

실제 `Once upon a time`의 첫 decode에서는 다음과 같이 상위 두 token 순서가 바뀌었다.

```text
OFF: " in"    -0.6476, " there" -0.7726
ON : " there" -0.5875, " in"    -0.8375
```

모델과 유사한 실제 기하(`H=16`, `HV=48`, `K=V=128`, `L=16`)에서 33 step과
두 번의 flush를 수행한 kernel 단독 검사는 다음 오차를 보였다.

- 최대 output absolute error: `3.0517578125e-05`
- 평균 relative error: 약 `0.00114`
- flush state 최대 오차: 약 `1.16e-4`
- step 0: exact match

`d`/`k` ring을 FP32로 바꾼 실험도 모델 출력 차이를 실질적으로 제거하지 못했다.
따라서 현재 남은 문자열 차이는 ring cursor 정체가 아니라 Replay 방식의 비트 단위
수치 비동등성과 모델의 민감도로 설명된다.

정확도 평가는 greedy 문자열 exact match만으로 결론내리지 말고, teacher-forced logits,
perplexity, downstream task 평가를 함께 사용해야 한다.

## 재현용 serve 설정

아래는 검증에 사용한 핵심 설정을 정리한 예시다. ON/OFF 성능 비교에서는 ReplaySSM
옵션을 제외한 backend와 모든 실행 조건을 동일하게 맞춰야 한다.

```bash
PATH=/disk2/omin/miniconda3/envs/vllm029_q38next/bin:$PATH \
FLASHINFER_WORKSPACE_BASE=/disk2/omin \
VLLM_CACHE_ROOT=/disk2/omin/.cache/vllm \
VLLM_USE_V2_MODEL_RUNNER=1 \
CUDA_VISIBLE_DEVICES=0 \
/disk2/omin/miniconda3/envs/vllm029_q38next/bin/vllm serve \
  /disk2/models/Qwen3.8-Flash-Next-NVFP4 \
  --host 127.0.0.1 \
  --port 18988 \
  --trust-remote-code \
  --language-model-only \
  --kv-cache-dtype auto \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95 \
  --no-enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --mamba-cache-mode none \
  --mamba-ssm-cache-dtype float32 \
  --mamba-backend triton
```

ReplaySSM ON에는 다음 옵션을 추가한다.

```text
--use-replayssm --replayssm-buffer-len 16
```

## 검증 산출물

실행 원자료는 다음 host-local 경로에 있으며 Git에는 포함하지 않았다.

- `/disk2/omin/tmp/gdn_equiv_fixed_check.sh`
- `/disk2/omin/tmp/gdn_equiv_fixed/off.server.log`
- `/disk2/omin/tmp/gdn_equiv_fixed/on.server.log`
- `/disk2/omin/tmp/gdn_equiv_fixed/off.json`
- `/disk2/omin/tmp/gdn_equiv_fixed/on.json`
- `/disk2/omin/tmp/gdn_equiv_fixed/off.details.json`
- `/disk2/omin/tmp/gdn_equiv_fixed/on.details.json`

## 현재 결론과 남은 검증

현재 상태에서 다음은 확인됐다.

- v2 러너에서 요청별 GDN ReplaySSM cursor가 실제 decode와 CUDA graph replay에 전달된다.
- ring이 0번 위치에 고정되던 결함과 반복 출력 증상이 제거됐다.
- 원본 GDN ReplaySSM 계산식을 임의로 변경하지 않았다.
- OFF/ON 모두 실제 Qwen3.8-Flash-Next 모델의 부팅과 생성에 성공했다.
- 공유 Mamba2/Qwen3.5 경로의 관련 회귀 테스트가 통과했다.

다음은 아직 별도 검증이 필요하다.

- prefix caching을 켠 실제 end-to-end resume 동작
- speculative decoding과 ReplaySSM 조합
- align mode (현재 의도적으로 미지원)
- warmed cache에서 동일 조건으로 수행한 정식 tokens/s, TTFT, TPOT benchmark
- perplexity 또는 downstream task 기반의 품질 영향 측정

