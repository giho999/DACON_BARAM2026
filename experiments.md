
---

## Experiment: pipeline_v1_lightgbm_baseline

**Date:** 2026-07-31

### Overview
First clean baseline pipeline based on EDA insights. Single LightGBM models per KPX group with physics-based features derived from raw LDAPS/GFS weather data.

### Model
- Type: LightGBM Regressor (separate model per group)
- Params: num_leaves=127, lr=0.05, n_estimators=2000, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, min_child_samples=50
- Target transform: log1p
- Early stopping: 100 rounds on 10% holdout
- Seed: 42

### Features
- 1052 features (weather grid pivots + physics-derived + time + SCADA proxy)
- LDAPS 16 grids + GFS 9 grids, key_grids only for derived features
- Physics features: ws³, air density, hub-height wind, shear, turbulence, stability
- SCADA power curve proxy (3 features, physics-informed bridge)
- LDAPS test gap filling (3 timestamps, linear interpolation)
- Missing: median fill for features, interpolation for G1/G2 labels

### CV Scores (corrected FICR metric)
| Group | 2022→2023 | 2022-2023→2024 | Avg |
|-------|-----------|----------------|-----|
| G1 | 0.5419 | 0.5679 | 0.5549 |
| G2 | 0.6076 | 0.5828 | 0.5952 |
| G3 | N/A (skip) | 0.5495 | 0.5495 |
| **Overall** | | | **0.5667** |

### Test Predictions (2025)
| Group | Mean kWh | CF | Zeros | Cap Hits |
|-------|----------|-----|-------|----------|
| G1 | 5,979 | 27.7% | 1 | 6 |
| G2 | 6,270 | 29.0% | 0 | 0 |
| G3 | 4,900 | 23.3% | 0 | 0 |

### Bugs Fixed
1. FICR metric: replaced wrong implementation with correct DACON formula
2. g3_available: removed as data leakage risk
3. SCADA proxy: separated G1/G2 turbine groups
4. Test NaN: fixed inf handling
5. Final model: added holdout for early stopping
6. G3 CV: explicit skip for invalid fold

### Files
- Submission: `.opencode/pipeline_v1/submission.csv`
- Pipeline: `.opencode/pipeline_v1/preprocessing.py`, `train_model.py`, `generate_submission.py`

---

## Experiment: TASK54 S8A — G3 synthetic 2022 target 통합 (pipeline_v2)

**Date:** 2026-08-01

### Model
- Type: pipeline_v2 4-GBDT(LGB+XGB+Cat+LGB-L1) + MLP → (5g+1m)/6 blend, B_m1_adaptive
- Params: PIPE_G3_SYNTH=1, PIPE_G3_SYNTH_FULL_WEIGHT=1 (synthetic 행 full weight 1.0), SEED=42, gate 규약 cal=2024-09~10 → eval=2024-11~12

### CV
- GATE CV B_m1_adaptive: 0.63288 → **0.64381 (+0.01094)**
- G3 gate +0.03281, RAW full-2024 G3 +0.03844, RAW 2024-11~12 +0.04298, G1/G2 delta 0.000
- 3-seed 검증 {42,123,2024}: raw +0.0148 (3/3 양성), gate +0.0510 (3/3 양성)

### LB
- 미제출 (코드리뷰 후 LB 프로브 1회 예정 — 잔여 LB 2회 중 1회 소모, 승인 필요)

### Notes
- S6 satfrac 전이실패 교훈과 달리 raw 구간에서도 양성 → 캘리브레이션 아티팩트 아님
- C_l1_synergy 재판정 유보 (C 0.64477 > B 0.64381, +0.00136, 노이즈 경계)
- 상세 기록: `task54_g3synthetic/EXPERIMENT_LOG.md`

---

## Experiment: TASK37 M1_v2 (작업물 A) — 팀원 원본 base에 adaptive 밴드 M1 적용

**Date:** 2026-08-01

### Model
- Type: v7 GBDT + frozen CNN(seed42) blend (w=0.20)
- Base: 팀원 원본 submission_ficr_w1_v7_cnn(0.65183).csv (LB 0.65184 확보됨)

### Params
- P1: adaptive 밴드 [0.05,0.20] / [0.08,0.25] / [0.03,0.15]
- P2: OOS 게이트 (cal 09-10 → eval 11-12)
- P3: shift grid @0.005

### CV
- OOS FICR cali: 0.62811 → 최고 밴드 [0.03,0.15] 0.62982 (+0.00171)
- 원본 밴드 [0.05,0.20] 0.62936 대비 +0.00046; grid 세분화 효과 +0.00007 (noise 내)
- 프로덕션 shift: G1=+0.045, G2=+0.040, G3=+0.055 (11-12월 자기자신 적합, 팀 관례)

### LB
- **0.65403** (2026-08-01 16:56 제출); 1-NMAE 0.87408, FICR 0.43398
- 기존 팀 최고(0.65293) 대비 **+0.00111** 개선

### Notes
- P3(그리드 세분화)는 noise 내로 기각. P1 adaptive 밴드 [0.03,0.15]가 원본 [0.05,0.20]보다 OOS에서 우세
- 출력: `submission_A_m1v2.csv`

---

## Experiment: pipeline_v2 C_m1_adaptive (작업물 B) — G3 synth + L1 synergy + adaptive 밴드 M1

**Date:** 2026-08-01

### Model
- Type: pipeline_v2 4-GBDT + MLP, G3 synth 통합, blend (5g+1l1+1m)/7

### Params
- PIPE_G3_SYNTH=1, adaptive 밴드 3종 중 CV 최고 선택

### CV (gate cal 09-10 → eval 11-12)
- C_m1_adaptive **0.64536** (+0.00155 vs B_m1_adaptive 0.64381, +0.00195 vs A 0.64341)
- 선택 밴드: [0.03,0.15]

### LB
- 미제출

### Notes
- auto-pick이 C_m1_adaptive 선택. 두 작업물 모두 [0.03,0.15] 밴드 선택 — 일관된 신호
- 단 C−B = +0.00155는 noise SE≈0.0028 내로 통계적 동률
- 출력: `submission_g3synth_v2.csv`, `cv_results_g3synth_v2.json`

---

## Experiment: A+B 크로스 앙상블 (Step 2) — 팀원 원본(v7+CNN) × pipeline_v2 블렌드

**Date:** 2026-08-01

### Model
- A = v7 GBDT + frozen CNN(seed42) blend(w=0.20) + iso+shift 캘리 + [0.03,0.15] 밴드
- B = pipeline_v2 (5*gbdt+1*mlp)/6 + FORCE_FORMS 캘리 + adaptive 밴드

### Params
- 제출물 레벨 블렌드 w*A + (1-w)*B, w ∈ {0~1 @ 0.05}
- gate cal 09-10 → eval 11-12 (1464행)

### CV
- A 단독 0.63093, B 단독(B_adaptive) 0.64381
- **최적 w=0.30 → 0.64542** (+0.00161 vs B 단독)
- C_m1_adaptive(0.64536)와는 +0.00006으로 noise 내 동률
- w 곡선 비매끄러움(noise 큼), w>0.4 급락

### LB
- 미제출

### Notes
- 앙상블 이득은 C_adaptive와 비교 시 noise 내 동률 → 단일 레버로 30등 목표(+0.005) 부족
- FICR이 리더보드 격차의 핵심(우리 0.43234 vs 30위 0.44084)
- 출력: `submission_ens_w30.csv` (test 제출물 블렌드, w=0.30)

---

## Experiment: E1 — G3 iso 캘리 제거 (FORCE_FORMS G3: iso+shift → shift_only, PIPE_G3_FORM=shift_only)

**Date:** 2026-08-01

### Model
- Type: pipeline_v2 (4-GBDT+MLP, G3 synth), B_m1_adaptive

### Params
- G3 캘리를 shift_only로 변경 (iso 제거)
- cal=09-10 → eval 11-12 (게이트, 1464행)

### CV
- 게이트 B_m1_adaptive: 0.64381 → **0.66311 (+0.01930)**; C_m1_adaptive 0.64536 → 0.66254
- Full-2024 검증 (cal 09-10 → full 2024): total 0.63824(iso) vs 0.63842(E1) 동률, FI는 iso +0.0017 우세
- 계절별 (cal 01-06 → eval 07-12 OOS): 여름 E1 -0.010 FI / 가을 E1 -0.018 FI / 겨울(11-12) E1 **+0.038 FI** / 하반기 전체 E1 **+0.012 FI**

### LB
- 미제출

### Notes
- E1은 겨울 레짐 특화 개선. iso는 여름~가을에 이득, E1은 겨울에 큰 이득(하반기 전체로는 E1 우세)
- 2025 test(1년) 기준 채택 여부는 사용자/팀 판단 대기. 팀 LB-proven 설정 변경이므로 code-reviewer 검토 필요
- 출력: `submission_g3synth_E1.csv`, `cv_results_g3synth_E1.json`

---

## Experiment: E1 코드리뷰 후속 수정 (CR-2 / IM-3 / IM-4 반영) — E1 재검증

**Date:** 2026-08-01

### Model
- Type: pipeline_v2 (4-GBDT+MLP, G3 synth, PIPE_G3_FORM=shift_only)

### Params
- G3 캘리 shift_only + 코드리뷰 지적 3건 수정
- CR-2: train_gbdt.py/train_mlp.py에 synth_guard 추가 — G3_SYNTH=False면 G3 마스크에서 2022 행 명시 제외 (무음 arm-C 차단). 검증: flag OFF 시 G3 2022 행 0, flag ON 시 8664 유지
- IM-3: config.py에 PIPE_G3_FORM 화이트리스트 검증(shift_only/iso+shift/iso_only) + generate_submission.py에 활성 env 배너 출력
- IM-4: noise_caveat을 실제 실행 델타로 동적 생성 (이전 stale 문구 수정)

### CV
- E1 재실행 결과 동일 — B_m1_adaptive **0.66311** (게이트)
- variant deltas: B−A = +0.00026, C−A = −0.00075

### LB
- **0.64590** (2026-08-01 22:41 제출); 1-NMAE 0.87219, FICR 0.41962
- 최종(A 0.65403) 대비 TOTAL -0.00813, FICR -0.01436 → **기각 (REJECTED)**

### Notes
- 코드리뷰 판정 "조건부 채택". E1의 게이트 +0.019는 겨울 창 특화이며 프로덕션 기대는 ±0.003 동률 (CR-1)
- 30등 목표(+0.0085)의 일부만 기여하는 소폭 레버. probe 결과 ≥0 → E1 채택, 음수 → iso+shift 유지 → **probe 음수로 기각 확정**
- 출력: `submission_g3synth_E1_v2.csv`, `cv_results_g3synth_E1_v2.json`

---

## Experiment: E1 LB probe (제출) — G3 shift_only 연중 검증 결과 기각 확정

**Date:** 2026-08-01

### Model
- Type: pipeline_v2 (4-GBDT+MLP, G3 synth, PIPE_G3_FORM=shift_only), B_m1_adaptive

### Params
- G3 캘리 shift_only (iso 제거) + 코드리뷰 반영 (CR-2/IM-3/IM-4)
- 파일: `submission_g3synth_E1_v2.csv`, 제출 2026-08-01 22:41:47

### CV
- 로컬 게이트: 0.66311 (+0.01930 vs iso+shift)

### LB
- **TOTAL 0.64590** / 1-NMAE 0.87219 / FICR 0.41962
- 최종(submission_A_m1v2.csv, LB 0.65403) 대비 TOTAL **-0.00813**, FICR **-0.01436**

### 판정
- **기각 (REJECTED)** — LB 대폭 악화
- 겨울 창(11-12월) 특화 shift_only가 연중 2025년에선 여름/가을 FI 손실(-0.010/-0.018)로 압도당함
- 게이트(겨울 eval)가 연중 구조를 반영 못 한 '겨울 검증 한계' 사례
- 최종파일 불변: `submission_A_m1v2.csv` (LB 0.65403) 유지

### Notes
- 로컬 게이트 +0.01930이 LB에서 -0.00813으로 반전 — 게이트 창 선택이 채택/기각 판단에 결정적
- E1 채택 조건(probe ≥ 0) 불충족 → iso+shift 유지

---

## Experiment: M1v3 — M1 밴드/이중블렌드 최적화 (작업물A 기반)

**Date:** 2026-08-02

### Model
- Type: v7+CNN base에 M1 이중블렌드 적용
- Base: submission_ficr_w1_v7_cnn(0.65183).csv

### Params
- M1 밴드: [0.005,0.12]×[0.01,0.10] 이중블렌드 (w=0.6)
- Shifts: G1 +0.08, G2 +0.09, G3 +0.08 (vs 작업물A: +0.045/+0.040/+0.055)
- 밴드 하한 인하(0.03→0.005)로 더 낮은 예측까지 보정, 상한 축소(0.15→0.12)로 희석 방지

### CV
- Holdout 생산 프로토콜: +0.00493 FICR vs 작업물A M1
- Holdout 전체 최적화: +0.00143 FICR (shift만)

### LB
- **TOTAL 0.65377** / 1-NMAE 0.87400 / FICR 0.43355
- 작업물A(0.65403) 대비 TOTAL **-0.00026**, FICR -0.00043

### 판정
- **기각 (REJECTED)** — Holdout +0.005 FICR이 LB에서 역전
- E1과 동일 패턴: 2024 holdout 최적화 ≠ 2025 LB 전이
- M1 파라미터 튜닝만으로는 LB 개선 불가 — post-processing 한계 도달

### Notes
- 출력: `submission_A_m1v3.csv`

---

## Experiment: pc_pred G3 — 파워커브 기반 G3 2022 합성 라벨

**Date:** 2026-08-02

### Model
- Type: pipeline_v2 (4-GBDT+MLP), G3 2022 합성 라벨을 pc_pred(파워커브)로 교체
- Base: cache_train.parquet에서 G3 2022만 pc_pred × scale로 대체

### Params
- G3 2022 합성: RF(G2-corr 0.9881, G2 거의 복사본) → pc_pred × 0.9835 (G2-corr 0.8617)
- pc_pred: 물리 파워커브 모델이 G3 위치 풍속으로 예측한 발전량 (Wind-only, G2 무관)
- PIPE_G3_SYNTH=1, PIPE_CACHE_TRAIN=cache_train_pcg3.parquet
- PIPE_G3_FORM=iso+shift (E1 실패 교훈 반영)

### 동기
- TASK33 감사: RF 보간 G3가 G2에 과의존(corr 0.988 vs 자연 0.914) → G3 고유 패턴 학습 불가
- TASK54 증명: 불완전한 합성 라벨조차 전가중치로 추가 시 G3 CV +0.03~0.036
- → 더 나은(더 다양한) 합성 라벨이면 추가 이득 확실

### CV (gate cal 09-10 → eval 11-12)
- G3 cv_tr: 17,519행 (2022 synthetic + 2023, 2배 증가)
- B_m1_adaptive: **0.64714** (+0.00333 vs RF 합성)
- **C_m1_adaptive: 0.64957** (+0.00421 vs RF 합성)
- C_l1_synergy: 0.64877

### LB
- **미제출** — `submission_pcg3_C.csv` 대기

### Notes
- C_m1_adaptive = (5*gbdt + 1*l1 + 1*mlp)/7, L1 멤버로 outlier 강건성 증가
- pc_pred G3가 G2 의존도 0.862(자연 0.914 수준) → 모델이 UNISON 고유 파워커브 패턴 학습
- E1과 달리 iso+shift 캘리 사용 (shift_only 아님)
- config.py: CACHE_TRAIN이 PIPE_CACHE_TRAIN 환경변수 지원하도록 변경됨
- 입력: `cache_train_pcg3.parquet` (pc_pred G3 2022 적용본)
- 출력: `submission_pcg3.csv` (B), `submission_pcg3_C.csv` (C)

---

## Experiment: S2-V2 하네스 반증 테스트 (H-TEST) — 2026-08-05 재실행 재현 확인 + 코드리뷰 완료

**Date:** 2026-08-05

### Overview
- S2 (구조 유도형 조건부 shift, G1/G2 전용) 검증 하네스의 변별력을 4개 과거 실험으로 반증 검증
- 파일: `endgame_0804/s2_condshift/h_test.py` → `h_test_results.npz`
- 하네스: 2023↔2024 교차 검증, 채점은 `pipeline_v2/metrics.py`의 DACON 정의 3그룹 TOTAL (competition_metric 3그룹 평균, score_fn/score_g12 미사용)

### 검증 결과 (하네스Δ 평균 / 2023→24 / 2024→23 / LB Δ)
| 변환 | 하네스Δ | 2023→24 | 2024→23 | LB Δ | 판정 |
|---|---|---|---|---|---|
| g12_opt (G1+0.035c, G2+0.010c) | **+0.01307** | +0.01375 | +0.01239 | **-0.00500** | 방향 반대 (하네스가 실패를 통과시킴) |
| g3shift03 (G3+0.03c) | +0.00405 | +0.00495 | +0.00315 | **-0.00289** | 방향 반대 |
| poly2 (G3 Ridge) | -0.00150 | +0.00111 | -0.00411 | **-0.01080** | 과소 (평가연도 라벨 in-sample fit으로 전이회귀 오염 — MAJOR 지적) |
| M1 [0.03,0.15] 밴드 shift (양성대조) | +0.00553 | +0.00568 | +0.00539 | **+0.00111** | 유일하게 방향 일치 |

### 판정 규칙
- 하네스(g12_opt) ≥ +0.004 → 하네스 반증 → **S2 REJECTED로 기록, 종료**

### 최종 판정
- **S2 REJECTED** — 하네스(g12_opt)=+0.01307 ≥ +0.004 문턱 초과, 양방향 모두 반증
- S2-V3(절단 편향 분해), S2-V4(재유도·게이트)는 반증으로 인해 **미진행** (지시 규칙상 종료)

### 코드리뷰
- BLOCKER 없음, MAJOR 1건 (poly2 in-sample fit leakage로 전이회귀 오염), 판정 자체는 타당
- 전이 회귀: 4쌍 slope=0.3534, intercept=-0.00626, corr=0.4277 — 단, poly2 점 제외 시 3쌍 slope=-0.42, corr=-0.66으로 부호 반전 (해석 유보 사항으로 함께 기록)

### 재현성
- 8/4 기록과 8/5 재실행 값이 소수점 5자리까지 일치 (결정적 실행)

### S1 관련 확인사항
- PID 1208865 학습은 이미 완주 완료 (`endgame_0804/s1_wsavg/S1_report.md`: "PID 1208865 완주 약 4h40m, 시스템 로드 186")
- OMP_NUM_THREADS=16, n_jobs=16 제한 규칙이 이후 학습에 적용됨 (S1 2차 학습 361→161 스레드, 4h40m→7분)
- 이중 연도 게이트 12회 학습은 S2 REJECTED 종료로 인해 불필요하게 되었음 (S2-V4가 V2/V3 통과 시에만 진행되는 조건부 작업이므로)

### Notes
- 참조: `endgame_0804/reports/S1S2_FINAL.md` (S2 REJECTED 확정 반영)

---

## Experiment: S1(ws avg) seed42 2024 holdout 속보 — 조기 중단 판정 (REJECTED)

**Date:** 2026-08-05

### Overview
- S1(ws avg) seed 42의 2024 연도전이 holdout 중간 속보 — 조기 중단 판정 확정
- 파일: `endgame_0804/s1_wsavg/S1_seed42_holdout_speed.md`
- 프로토콜: 작업물A run.py 방식 (A/S1 공통, 캘리/M1 파라미터는 각각 자체 fit)

### Model
- Type: S1 ws avg — 기존 454피처 + ws avg 22개 = **476피처**, 4-GBDT(LGB+XGB+Cat+LGB-L1) 앙상블 + clean-downweight(0.3), seed 42
- 학습: 2022-01-01~2023-12-31 (17,419행) → 2024 전체 8,778행 예측 (연도전이 holdout)

### Params
- v7 GBDT + frozen CNN(seed42, w=0.2) 블렌드 → 캘리(cal 09-10 → eval 11-12, iso+shift, SHIFT_GRID=arange(-0.08,0.09,0.01)) → M1 밴드 [0.03,0.15]×cap
- G3 캘리: iso+shift 강제 (작업물A와 동일, v2 수정 반영)
- 실행: OMP=16/n_jobs=16, 343초 (load 75~85)

### CV (eval 2024-11~12, metrics.py competition_metric 3그룹 평균)
| 지표 | 작업물A | S1 | Δ |
|---|---|---|---|
| TOTAL | 0.63093 | 0.62558 | -0.00535 |
| 1-NMAE | 0.87526 | 0.86993 | -0.00533 |
| FICR | 0.38659 | 0.38123 | -0.00536 |

- 그룹별: G1 +0.0085 / G2 -0.0115 / G3 -0.0130
- 결합비 ρ = +1.0055 (FICR/1-NMAE)

### 판정
- **조기 중단 (S1 REJECTED 계열)** — Δ(1-NMAE)=-0.00533 ≤ 0, |Δ|≥0.0005 → 실질 열화, 노이즈 아님
- 11회 이중 연도 게이트 학습 큐잉 안 함 (자원 절약)
- 지시서 판정 트리: S1 조기중단 → **S4 착수 (조건 b)** 로 전환
- ws avg 피처는 2024 홀드아웃에서 G2/G3를 악화시킴 — v1에서 REJECTED된 이유(전이 실패) 재확인

### LB
- 미제출 (속보 판정용 holdout 전용)

### Notes
- 캘리/M1 적합 shift (A vs S1): cal G1 +0.030→+0.040 / G2 +0.070→+0.030 / G3 +0.040→+0.080 · M1 G1 +0.045→+0.045 / G2 +0.040→+0.090 / G3 +0.055→+0.025
- 산출물: `endgame_0804/s1_wsavg/s1_holdout_preds_2024.parquet` (8,778행), `s1_gen_holdout_preds.py`, `s1_speed_decomp.py`, `S1_seed42_holdout_speed.md`

## Experiment: S4(미사용 LDAPS 격자 30피처) seed42 2024 holdout 속보 — 조기 중단 판정 (REJECTED)

**Date:** 2026-08-05

### Overview
- S4(미사용 LDAPS 격자 30피처) seed 42의 2024 연도전이 holdout 중간 속보 — 조기 중단 판정 확정
- 파일: `endgame_0804/s4_grids/S4_seed42_holdout_speed.md`
- 프로토콜: 작업물A run.py 방식 (A/S4 공통, 캘리/M1 파라미터는 각각 자체 fit) — S1-G와 동일 프로토콜 재사용

### Model
- Type: S4 — 원본 v9 454피처 + 미사용 LDAPS 격자 [1,2,3,8,9,10,13,14,15,16] 집계 30피처 = **484피처**, 4-GBDT(LGB+XGB+Cat+LGB-L1) 앙상블 + clean-downweight(0.3), seed 42
  - Tier-1 (20): ldaps_unused_ws/pmsl/sp/t2m_mean/max/std/min, 10u/10v_std, 50MU/MV_range_mean
  - Tier-2 (10): spatial_grad_ws_unused_5/6/11/12, spatial_grad_pmsl_unused_5/6, spatial_grad_t2m_unused_5/6, spatial_grad_ws_ns, spatial_grad_t2m_ns
- 병합: merged_train_v9(`kst_dtm`) inner join ldaps_unused_feats_train(`forecast_kst_dtm`) → 26,303행 (ldaps 26,304행 중 초과분 1행 제외), test 8,760행
- 학습: 2022-01-01~2023-12-31 (17,419행) → 2024 전체 8,778행 예측 (연도전이 holdout)

### Params
- v7 GBDT + frozen CNN(seed42, w=0.2) 블렌드 → 캘리(cal 09-10 → eval 11-12, iso+shift, SHIFT_GRID=arange(-0.08,0.09,0.01)) → M1 밴드 [0.03,0.15]×cap
- 실행: OMP=16/n_jobs=16, 97초 (load 55~75)

### CV (eval 2024-11~12, metrics.py competition_metric 3그룹 평균)
| 지표 | 작업물A | S4 | Δ |
|---|---|---|---|
| TOTAL | 0.63093 | 0.61864 | -0.01229 |
| 1-NMAE | 0.87526 | 0.86783 | -0.00743 |
| FICR | 0.38659 | 0.36944 | -0.01715 |

- 그룹별 FICR Δ: G1 -0.00290 / G2 -0.00654 / G3 -0.02743 (G3 열화 지배)
- 결합비 ρ = +2.3093 (FICR/1-NMAE) — FICR 열화가 NMAE 열화보다 2.3배

### 판정
- **조기 중단 (S4 REJECTED 계열)** — Δ(1-NMAE)=-0.00743 ≤ 0, |Δ|≥0.0005 → 실질 열화, 노이즈 아님
- 11회 이중 연도 게이트 학습 큐잉 안 함 (자원 절약)
- 지시서 판정 트리: S1 조기중단 → S4 착수(조건 b) → S4 역시 조기 중단
- S1(-0.00533)보다도 열위 — 정보형 레버 2종(S1, S4) 모두 2024 홀드아웃에서 실패
- 미사용 LDAPS 격자 피처는 2024 홀드아웃에서 G3를 특히 악화(G3 -0.0274) — 물리적 공간정보 추가가 전이에 도움이 되지 않음

### LB
- 미제출 (속보 판정용 holdout 전용)

### Notes
- 작업물A 재현 검증: `s4_speed_decomp.py`가 A 파이프라인 재계산 → **TOTAL 0.63093 재현 확인**
- 캘리/M1 적합 shift (A vs S4): cal G1 +0.030→+0.030 / G2 +0.070→+0.040 / G3 +0.040→+0.080 · M1 G1 +0.045→+0.070 / G2 +0.040→+0.075 / G3 +0.055→+0.020
- G3 열화 지배 (nmae 0.1469→0.1663); 그룹별 FICR Δ: G1 -0.00290 / G2 -0.00654 / G3 -0.02743
- 산출물: `endgame_0804/s4_grids/s4_holdout_preds_2024.parquet` (8,778행), `s4_gen_holdout_preds.py`, `s4_speed_decomp.py`, `S4_seed42_holdout_speed.md`, `S4-1_report.md`, `build_ldaps_unused_feats.py`, `ldaps_unused_feats_train/test.csv`

---

## Experiment: SV-1 그룹별 3분해 — SV-2 착수 금지 (SKIP)

**Date:** 2026-08-05

### Overview
- S1/S4 2024 holdout 속보의 "G1 +0.0085" 신호가 그룹별 3분해(Δ(1-NMAE) vs ΔFICR) 후에도 실체가 있는지 재검증
- 파일: `endgame_0804/sv_salvage/sv1_group_decomp.py`, `endgame_0804/sv_salvage/SV1_report.md`
- 채점: `metrics.py` competition_metric_parts, eval 2024-11~12, 파이프라인 CNN(w=0.2)+캘리+M1 [0.03,0.15]

### CV — S1 그룹별 3분해
| 그룹 | Δ(1-NMAE) | ΔFICR |
|---|---|---|
| G1 | **-0.00018** (노이즈 수준) | **+0.00853** |
| G2 | +0.00099 | -0.01154 |
| G3 | -0.01682 | -0.01303 |

### CV — S4 그룹별 3분해
| 그룹 | Δ(1-NMAE) | ΔFICR |
|---|---|---|
| G1 | -0.00032 | -0.00290 |
| G2 | -0.00253 | -0.00654 |
| G3 | -0.01944 | -0.02743 |

### 판정
- **SV-2 착수 금지 (SKIP)** — 지시서 규율("G1 신호가 FICR 단독(NMAE ≤ 0)이면 SV-2 착수 금지") 적용
- **핵심 발견**: 기존 속보의 "G1 +0.0085"는 FICR 단독 신호였음 — Δ(1-NMAE)=-0.00018 (노이즈 수준)으로 NMAE 측면 이득 없음
- S4는 그룹별 전부 음수, G3 제외 시 양수 그룹 없음
- 3-seed×2년 게이트 학습 큐잉 안 함 (자원 절약)

### LB
- 미제출 (판정용 분해 전용)

### Notes
- S1의 G2/G3은 NMAE/FICR 모두 음수, G1만 FICR 단독 양성 → SV-2(재유도·게이트) 레버의 근거 부족으로 종료

---

## Experiment: FIN-2 2차 평가 전환 — 재현 dry-run + 패키지 매니페스트 + 재현성 문서 갱신

**Date:** 2026-08-05

### Overview
- 2차 평가 대비 작업물A 재현성 검증 및 제출 문서 패키지 최종화 (FIN-2a~d)

### FIN-2a 재현 dry-run
- `work_a_package/run.py` 재실행 → **10.1초**
- `submission_A.csv` = `submission_A_m1v2.csv`와 **md5 `6d5e4214dfeadb898d5af81245308d58` byte 일치** (max diff 0.0)
- 스코어: 캘리만 0.62906 / M1 적용 **0.63093** / M1 shift [0.045,0.040,0.055] — 기존 기록 일치

### FIN-2b~d 문서 갱신
- **FIN-2b 패키지 매니페스트**: ENDGAME2_package_manifest.md — 최종 제출본 `submission_A_m1v2.csv`(LB 0.65403, SHA-256 `8ab47f88…`)로 교체, `submission_m1_bandcorrected.csv`는 이전 최종(하위 계보) 승계, work_a_package/ 재현 패키지 추가
- **FIN-2c REPRODUCIBILITY_STATEMENT**: §0 3층 표·§9 계보를 작업물A 기준으로 갱신, §9-2(작업물A 상세 + v10/Open-Meteo 미사용 명시), **§11 H-TEST 방법론 절 신규 추가** (발표 자산), S0/P0 크로스레퍼런스
- **FIN-2d 2차평가 체크리스트**: 최종파일 선택 확인 대상 `submission_A_m1v2.csv`로 교체, 8/13·8/14 09:30 2회 재확인 일정 명기, 팀원 3인 재학/휴학 증명서 8/10까지 발급 의무화 리마인드
- 검증: 3개 문서 SHA-256 최종 재계산 반영 (REPRO `7f51be91…`, 체크리스트 `91c1b87a…`)

### LB
- 최종 제출본 불변: `submission_A_m1v2.csv` (LB 0.65403)

### Notes
- 코드리뷰 S2: `work_a_package/run.py`의 M1 shift 적합이 eval 구간 라벨 self-ref(in-sample, 0.63093은 낙관적 스코어) — 발표 시 in-sample 명시 권고. BLOCKER 아님.

---

## Experiment: W5 그룹 중심 NWP 보간 — W5-1b 중복성 검사 기각 확정 (FIN-2 전환)

**Date:** 2026-08-06

### Overview
W5(그룹별 중심 NWP 보간) — v3 규율 유지, GPU 정책 변경(학번→gpu_0N 계정) 대응 중 신규 최우선 후보.
W5-0(지오메트리) → W5-1(피처) → W5-1b(중복성) → **REJECTED_BY_REDUNDANCY** → FIN-2 전환.

### W5-0 지오메트리 (W5_GEOMETRY.md)
- 세 그룹이 서로 다른 LDAPS 셀: G1→셀5, G2→셀6, G3→셀12 (거리 0.57~0.82km)
- 그룹 간 최대 2.17km(G1-G3) > LDAPS 1.5km 해상도 / GFS는 세 그룹 모두 셀5 (25km, 무효 대조로 적합)
- ⚠ 지시서 G1 중심(37.28038N) 오기 — 재계산 37.28713N 채택

### W5-1 피처 (w5_build_features.py)
- IDW(p=2) 그룹당 8열(ws50/ws10/wd_sin/wd_cos/temp/pres/turb/blh), LDAPS+GFS × real/noise 12 CSV
- 잡음대조: 중심 ±0.05°·최소 2km 이격 (초기 0.3~1.5km는 같은 셀이라 대조력 부재 → 강화)

### W5-1b 중복성 — REJECTED_BY_REDUNDANCY (핵심)
- LDAPS 18/24, GFS 18/24 컬럼 max|corr| ≥ 0.9 (C1 선례 기준)
- **구조적 원인**: v9가 이미 원천 셀(g4~g12)·공간구배(`spatial_grad_ws_5_6/5_12`) 보유
  → IDW 보간은 기존 셀의 가중평균(재조합)에 불과. 그룹차(g1−g3) vs v9 셀차(g5−g12) corr 0.994
- W5 기전("그룹 간 차이를 살린다")이 v9 피처에 이미 포함됨 — 학습 불필요 판정
- **인간 판정(2026-08-06)**: 기각 확정 → M2~M4(GPU 이전/환경검증/GPU 빌드) 불필요 → FIN-2 전환

### W5-4 문서화 (수치와 무관, 발표 자산)
- REPRODUCIBILITY_STATEMENT §12: V126 288.7/U136 289.1 W/m² ("G3 병목은 기종이 아님"),
  G3 2사업장(태백가덕산+태백원동) 2.05km 배열, LDAPS 1.5km vs 그룹간 2.17km, W1 실패 원인(굽은 능선 단일축 오류)
- §13: gpu_0N 실행 환경 분리 기록 (경로 정본 /home/student 유지)
- 발표_골격 1장·6장 갱신

### GPU 정책 대응 (M1~M4)
- M1 준비 완료: 이전 최소집합 6.42GB 산출, pre_migration_sha256.txt 기록
- gpu_04 계정 확보 확인 (인간) — 그러나 W5 기각으로 M2~M4 실행 불요
- W5-2c(CPU 스크린)·W5-3(기전) 생략 — 중복성 기각이 학습 전 게이트

### LB
- 최종 제출본 불변: `submission_A_m1v2.csv` (LB 0.65403, SHA-256 `8ab47f88…` 재확인)

### 산출물
- `endgame_0804/w5_ctr_interp/` — W5_GEOMETRY.md, W5_FEATURE_VERIFY.md, W5_1b_redundancy.md,
  w5_build_features.py, w5_redundancy.py, merged_*_w5*.csv (12벌), launch_w5.sh(미사용, 보존)
- `endgame_0804/reports/pre_migration_sha256.txt`

### Notes
- W5-2c 미실행: 중복성 검사(W5-1b)가 스크린의 학습 전 단계로 기각이므로 6회 CPU 학습(3~6시간) 절약
- FIN-2e: 매니페스트에 W5 기각·문서 SHA-256 갱신 반영
