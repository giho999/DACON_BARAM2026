# TASK-W2CHAIN — 동결 체인 스왑 결정실험 리포트

> 작성: 2026-08-09 · 작업 루트: `/home/gpu_04/DACON_baram2026/`
> 지시서: `opencode_TASK_W2CHAIN_20260809.md` · 선등록: `work_a_package/PREREG_w2chain_20260809.md`
> **핵심 질문**: METRICW W2의 체인 전 이득(+0.01768)이 기존 체인(iso/shift/M1/coordBA/CNN)에
> 이미 흡수됐는가? → **답: 그렇다. 중복률 149.7%, Δ_체인후 음수.**

---

## 0. 요약 (완료 보고 형식)

```
[TASK-W2CHAIN 완료]
체인 추출 검증: chain_frozen(W0) == run.py 원본  PASS (캘리만 0.62906 / M1 0.63093,
                submission_A.md5 6d5e4214… 재현)
2024 홀드아웃 TOTAL (3seed 평균):
  chain(W0) = 0.63402   chain(W1) = 0.62610   chain(W2) = 0.62524
Δ_체인후(W2) = -0.00878   (seed별: 42=-0.00916 1337=-0.00898 2024=-0.00821,  양수 0/3)
Δ_체인후(W1) = -0.00793   (seed별: -0.00676 -0.00788 -0.00914, 양수 0/3)
Δ_체인전 = +0.01768  →  중복률 r = 149.7%
중복 발생 단계: blend +0.01725 → iso +0.01095 → shift -0.00671(역전) → M1 -0.00931 → coordBA -0.00916
그룹별 TOTAL 기여 (1/6 희석): G1 +0.00261 / G2 -0.00056 / G3 +0.00109 (합 +0.00315, seed42)
캘리 창 × 2024 겹침: 있음 (cal 09-10 + M1 eval 11-12 self-ref 모두 2024 내부 — in-sample)
판정 밴드: 제출 금지 (<0)
제출 후보 md5: eef91a618062f91a09ce6336b0771cf0 (submission_metricw_W2chain_frozen.csv)
결정성: PASS (2회 실행 md5 일치)
이상 징후: 없음
```

---

## 1. 체인 추출 검증

`chain_frozen.py`가 `run.py` 체인을 함수로 분해. 동결 상수는 원본 v7 GBDT
(`m1_holdout_preds_2024.parquet`, clean-downweight 0.3)로 적합:

| 항목 | run.py 원본 | chain_frozen | 일치 |
|---|---|---|---|
| 캘리만 TOTAL | 0.62906 | 0.62906 | ✅ |
| M1 TOTAL | 0.63093 | 0.63093 | ✅ |
| submission_A.md5 | `6d5e4214dfeadb898d5af81245308d58` | 재현 | ✅ (README의 재현 md5와 동일) |

**동결 상수**: shifts=[0.03, 0.07, 0.04], M1 prod_shift=[0.045, 0.04, 0.055].
저장: `work_a_package/w2chain/frozen_constants.pkl` (+.isos).

---

## 2. 연도전이 홀드아웃 결과 (2022-23 → 2024, 대회 평가식 원본)

> 동결 체인 통과 후, 평가는 2024 전체(8778행). cal 창(09-10)·M1 eval(11-12)이 2024 내부 —
> **in-sample 오염 있음** (METRICW 체인 전과 동일 조건, W0 대비 상대 비교라 공통 오염).

| 가중 | 42 | 1337 | 2024 | 3-seed 평균 |
|---|---|---|---|---|
| **chain(W0)** | 0.63308 | 0.63391 | 0.63509 | **0.63402** |
| chain(W1) | 0.62632 | 0.62602 | 0.62594 | 0.62610 |
| chain(W2) | 0.62392 | 0.62493 | 0.62687 | **0.62524** |

### 체인 전 vs 체인 후

| 가중 | Δ_체인전 (vs W0) | Δ_체인후 (vs W0) | seed 양수(체인후) | 중복률 r |
|---|---|---|---|---|
| W1 | +0.01348 | **−0.00793** | 0/3 | 158.8% |
| **W2** | **+0.01768** | **−0.00878** | **0/3** | **149.7%** |

- **중복률 > 100%**: 체인이 W2 이득을 전부 흡수했을 뿐 아니라 W2가 오히려 방해.
- 지시서 §4 판정표 **"< 0 → 제출 금지, 2차 평가 소명 자산으로만 보존"** 해당.

---

## 3. 중복 발생 단계 추적 (seed42, W0 vs W2)

| 단계 | chain(W0) | chain(W2) | Δ(W2−W0) |
|---|---|---|---|
| blend 직후 | 0.60953 | 0.62678 | **+0.01725** |
| iso 직후 | 0.61257 | 0.62353 | +0.01095 |
| **shift 직후** | 0.62981 | 0.62310 | **−0.00671 (역전)** |
| M1 직후 | 0.63236 | 0.62305 | −0.00931 |
| coordBA | 0.63308 | 0.62392 | −0.00916 |

**해석**: iso는 W2의 마스크 교정 이득 중 일부만 흡수(+0.017→+0.011). **shift 캘리가 나머지를
흡수하고도 과잉 보정**하여 부호를 뒤집음. M1/coordBA는 역전을 심화. 즉 "iso/shift 캘리가 이미
W2가 교정하려는 마스크 절단편향을 부분 흡수"했다는 지시서 §0-1 우려가 **실증적으로 확인**됨.

---

## 4. 그룹별 분해 (seed42, W2 vs W0, 체인 후) — ΔTOTAL=(1/6)Σ_g(ΔFICR_g−ΔNMAE_g)

| 그룹 | ΔTOTAL_g | ΔFICR_g | ΔNMAE1_g | TOTAL 기여 |
|---|---|---|---|---|
| G1 | −0.00010 | +0.00774 | −0.00795 | +0.00261 |
| **G2** | **−0.01868** | **−0.02036** | −0.01701 | −0.00056 |
| G3 | −0.00869 | −0.00542 | −0.01197 | +0.00109 |
| 합 | | | | +0.00315 |

- **G2가 역전의 핵심**: 체인 후 W2의 G2 FICR이 −0.02036 급락. W2의 마스크 가중이 G2에서
  iso/shift 캘리와 정면 충돌 (shift 상수 +0.07로 과잉 보정된 구간에서 W2 예측이 더 손상).
- G1/G3는 체인 후에도 FICR 개선(+0.0077/+0.0054 음수 아님)이나 NMAE 손실(−0.008/−0.012)로
  TOTAL 기여가 상쇄.

---

## 5. 캘리 창 × 2024 겹침 (in-sample 고지)

- 평가 구간: **2024 전체 (01-01~12-31, 8778행)** — METRICW와 동일 분할.
- 체인 상수 적합 창: iso/shift = cal **2024-09-01~11-01**, M1 = eval **2024-11-01~12-31** (self-ref).
- **둘 다 2024 내부 → in-sample 오염 있음.** 단, W0/W2 모두 동일 체인 상수를 통과하므로
  Δ_체인후의 상대 비교는 유효 (공통 오염). 절대값(0.62~0.63)은 낙관적일 수 있음.

---

## 6. 제출 후보

`submission_metricw_W2chain_frozen.csv` — W2 2025 test 3-seed 평균 GBDT + 동결 CNN
(seed42, `task56_dm/cnn_test_pred_2025.npz`) → 동결 체인(iso/shift/M1) + coordBA.

- **md5 `eef91a618062f91a09ce6336b0771cf0`** · SHA-256 `0b86b54a07c359ea53b0a0eb33dd09d1450d36f84267c1ed0291023d106c98af`
- 가드: 8760행 · NaN 0 · 음수 0 · cap초과 0 · G3 max_cf 0.9236
- **즉시 제출 금지** — 판정 "제출 금지"이므로 2차 평가 소명 자산으로만 보존.

---

## 7. 결정성

- `chain_frozen.py`·`build_submission.py` 모두 결정적. 제출 후보 동일 커맨드 2회 실행
  **md5 일치** (`eef91a61…`) 확인.
- 체인 상수(iso/shift/M1)는 원본 GBDT로 1회 적합 후 pickle 고정 — 재현 시 동일.

---

## 8. 재현 커맨드

```bash
cd /home/gpu_04/DACON_baram2026 && source activate lgaimers
# 1) 체인 추출 검증 + 동결 상수 생성
python3 work_a_package/w2chain/chain_frozen.py repro
# 2) W0/W1/W2 × 3seed 체인 통과 (2024 홀드아웃 평가)
python3 work_a_package/w2chain/chain_frozen.py run_all
# 3) 단계별 추적
python3 work_a_package/w2chain/stage_trace.py
# 4) 제출 후보
python3 work_a_package/w2chain/build_submission.py
```

- METRICW 예측 입력: `work_a_package/metricw/pred_{W0,W1,W2}_seed{42,1337,2024}.parquet`
  (TASK-METRICW 산출물, 결정적)
- 환경: lgaimers conda env (numpy 1.26.4 · pandas 2.0.3 · sklearn 1.8.0)

---

## 9. 2차 평가 소명 관점

- **METRICW(W2)는 "체인 전 맨몸 GBDT"에서 +0.01768** — 이것은 원리적으로 유효한 발견으로
  소명 가능 (학습 목적함수를 평가식과 정합 → 마스크 절단편향 제거).
- **그러나 기존 제출 체인(iso/shift 캘리)이 이미 이 편향을 흡수하고 있어, W2를 체인에
  얹으면 오히려 악화(−0.00878)**. 즉 "W2 = 캘리 대체/보완 레버가 아니라, 캘리가 있는 한
  중복"이라는 결론. W2를 쓰려면 체인에서 shift 캘리를 제거해야 하나, 이는 **연도 적합 상수
  재설계 = 8연속 실패 구조 회귀** 위험이 있어 이번 실험 범위 밖.
- **판정: 제출 금지. 2차 평가에서는 METRICW(체인 전 원리)를 소명 자산으로, W2CHAIN(중복
  정량화 149.7%)을 "왜 제출 안 했는지"의 근거로 활용 권장.**

---

## 10. 산출물

```
work_a_package/w2chain/chain_frozen.py         # 체인 추출 (동결 상수 포함)
work_a_package/w2chain/stage_trace.py          # 단계별 추적
work_a_package/w2chain/build_submission.py     # 제출 후보 생성
work_a_package/w2chain/W2CHAIN_report.md       # 본 문서
work_a_package/w2chain/frozen_constants.pkl(+.isos)  # 동결 상수
work_a_package/w2chain/w2chain_raw.json        # 3seed 결과
submission_metricw_W2chain_frozen.csv          # 제출 후보 (제출 금지, 소명 보존)
work_a_package/PREREG_w2chain_20260809.md      # 선등록
work_a_package/metricw/pred_test_W2_seed{42,1337,2024}.parquet  # W2 2025 test 예측
```
