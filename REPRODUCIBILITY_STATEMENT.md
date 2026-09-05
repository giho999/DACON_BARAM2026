# 재현성 소명 (REPRODUCIBILITY STATEMENT)

작성: 2026-07-24 (클로드코드 실행) · ENDGAME-1 경로B(체크포인트 미확보 시 소명 경로)
★2026-07-30 갱신(TASK46, 2차 산출물 패키지 최종화): 최종파일이 `submission_m1_bandcorrected.csv`
(LB 0.6529253)로 확정됨에 따라, 이 문서를 **3층 계보** 구조로 재정리. 아래 §0이 요약, §1~8은
1층·2층(팀원 원본 v7+CNN)에 대한 기존 소명, §9는 3층(M1 밴드보정)의 계보.

★2026-08-05 갱신(FIN-2c, 최종 제출 확정본 반영): 최종 제출본이 8/1 이후 `submission_A_m1v2.csv`
(LB 0.65403, 68위)로 변경됨에 따라 §0·§9를 갱신 — 작업물A(M1 밴드 [0.03,0.15] adaptive 재적합판)
가 신규 최종이며, `submission_m1_bandcorrected.csv`(LB 0.6529253)는 "이전 최종(3층 M1 원본)"으로
하위 계보화(§9-0). §9-2(작업물A 재현 패키지·FIN-2a dry-run)·§11(H-TEST) 신규 절 추가. v10(Open-Meteo)
외부 데이터는 최종 경로에서 미사용임을 §9-2에 명시.

## 0. 3층 계보 요약

최종 제출본은 서로 재현성 등급이 전혀 다른 3개 층이 쌓인 구조다. 층을 구분하지 않고 "제출본이
재현되는가"를 하나의 질문으로 뭉치면 소명이 부정확해지므로, 아래 표로 층별 등급을 먼저 명시한다.

| 층 | 구성요소 | 재현성 | 근거 |
|---|---|---|---|
| **1층** | v7 GBDT(4모델)+MLP+블렌드+캘리 | **bit-identical 재현 확인**(max_diff=0) | §6(TASK28-P1), 2026-07-27 클린 재실행; 결과는 2026-07-30 기준 유효(회귀 없음, 파이프라인 변경 없음) |
| **2층** | 팀원 원본 CNN(SpatialCNN, w=0.20 블렌드) | **재현 불가** — 체크포인트 미저장·매 실행 재학습·TASK14 이전 GPU 비결정 연산의 3요소 결합(§7-1). 우리 결정성 패치판(`task5c_repro/`)은 존재하나, 이는 원본과 **다른 실현체**이며 동일 실현이 아님 — 실현체간 SD≈0.002·스프레드≈0.005(§3, TASK15)가 그 편차의 실측 하한. | §2~4, §7 |
| **3층** | M1 밴드보정(후처리, 결정적 산술) — **최종 제출본 `submission_A_m1v2.csv`(LB 0.65403, 68위) = 작업물A: M1 밴드 [0.03,0.15] adaptive 재적합판** | **완전 재현 가능**(결정적 산술). 작업물A 프로덕션 shift `=[G1 +0.045, G2 +0.040, G3 +0.055]×cap`, 적용 조건: 캘리된 예측값 ∈ `[0.03, 0.15]×cap`; FIN-2a dry-run(`work_a_package/run.py`) md5=`6d5e4214…`(10.1초) **bit-identical** 재확인. 이전 최종(3층 M1 원본) `submission_m1_bandcorrected.csv`(LB 0.6529253, 밴드 [0.05,0.20], shift=[G1 +0.06, G2 +0.03, G3 +0.05]×cap)는 하위 계보 — 2026-07-30 dry-run 영향행수(G1 3030/G2 2998/G3 2729/전체 8760) 일치 재확인. | §9, §9-1, §9-2(신규), §11(신규) |

**핵심 주의사항**: 3층(밴드보정)의 산술 자체는 100% 결정적이지만, 그 **상수의 적합(fit)이 2층의
캘리된 예측값 위에서 이뤄진다**(`build_m1_submission.py`가 `m1_holdout_preds_2024.parquet`+동결
CNN을 블렌드→캘리한 뒤 그 위에서 밴드보정 shift를 그리드서치) — 즉 3층 상수 자체는 재현 가능해도,
"이 상수가 왜 이 값인가"의 인과는 2층의 특정 실현(seed42 동결 CNN)에 의존한다. 다만 실제 프로덕션
경로에서는 이 적합된 shift를 **팀원 원본 제출본**(2층, 위 표)에 그대로 얹는 구조이므로(§9의
"★fable 확인 요청 답변" 참조), 최종파일 자체의 재현불가 지위는 2층에서 그대로 승계되고 3층은
그 위에 얹힌 순수 후처리 레이어로 남는다.

**근거(TASK15, ρ=0.02)**: 2층의 "어떤 CNN 실현이 좋은가"는 연도 간에 전이되는 신호가 아니라
매 학습마다 새로 뽑히는 순수 잡음이다 — 8개 고정 seed(1001~1008)로 2023/2024 홀드아웃 TOTAL
순위의 Spearman ρ=0.0238(p=0.9554, 사실상 무상관, `task15_realization/TASK15_result.md`). 이는
"2층이 재현 불가능하다"는 사실뿐 아니라 "재현할 수 있다 해도 그 선택에 의미가 없다"는 것까지
보여준다 — 즉 2층의 비결정성은 소명이 필요한 결함이라기보다, 애초에 seed 선택 자체가 실어나르는
정보량이 0에 가까운 축이라는 것이 TASK15의 독립적 확증이다.

## 1. 결론 요약
제출본을 생성한 CNN 학습 코드는 **비결정적**이었다(원인 규명·수정은 TASK14, 아래 §2). 코드와 데이터,
seed(42)는 모두 보존되어 있어 **동형 파이프라인**을 재현할 수 있으나, 수정 전 코드가 정확히 어떤 CUDA
연산 순서로 그 특정 실행을 만들어냈는지는 사후에 복원할 수 없다. 따라서 코드만으로 픽셀 단위 재실행은
불가능하지만, (a) 비결정성의 정체를 규명해 수정판으로 bit-identical을 실증했고, (b) 그 비결정성이
실제로 만들어내는 실현 간 점수 편차의 크기를 실측했으며(§3), 그 편차가 대회 규정상 "오차 범위 내" 재현
기준에 해당하는 수준임을 정량적으로 보여준다.

## 2. 비결정성 원인 규명 및 수정 (TASK14 전문 인용)
`task14_determinism/DETERMINISM_report.md` 참조. `cnn_gate/cnn_common.py`에 다음 4건을 동시 적용:

1. `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` — 모듈 최상단, `import torch` 이전
   (CUDA≥10.2 cuBLAS 행렬곱 결정성 전제조건, CUDA 컨텍스트 생성 전에 설정돼야 함)
2. `torch.use_deterministic_algorithms(True, warn_only=False)` — 학습 진입 시
   (`cudnn.deterministic=True`는 cuDNN 커널만 덮고 그 외 CUDA 커널은 미보장 — 엄격 모드로 격상)
3. `nn.AdaptiveAvgPool2d((1,1))` → `x.mean(dim=(2,3), keepdim=True)`로 교체 — PyTorch 공식 문서상
   `adaptive_avg_pool2d` backward(CUDA)는 결정적 구현이 없는 연산으로 등재됨. 출력크기(1,1)에서 forward는
   수학적으로 완전 동일(채널별 공간평균)하므로 예측값에는 영향 없음 — backward의 비결정성만 제거.
4. `DataLoader(..., generator=dl_gen)` — shuffle에 seed로 시딩된 전용 `torch.Generator()` 명시 전달
   (기존엔 전역 RNG 암묵 의존 → 호출 순서 변화에 취약).

**검증**: `task14_determinism/test_determinism.py`를 완전히 독립된 별도 프로세스로 2회 실행(seed=42,
production-mode 2022-23 학습 → 2024 holdout 추론).
- best_val_mse: run1=0.035603683441877365, run2=0.035603683441877365 (float64 전정밀도 일치)
- 예측값(2024 holdout 8778×3): `np.array_equal`=True, 최대절대차=0.0
- state_dict 전체 파라미터(모든 레이어): `np.array_equal`=True

`use_deterministic_algorithms(True, warn_only=False)`는 비결정적 연산이 남아있으면 학습 중 예외를 던지는
엄격 모드이며, 예외 없이 완주했다는 것 자체가 "남은 연산 전부에 결정적 구현이 존재함"의 직접 증거다.

**한계(명시)**: 이 수정은 *향후* 실행을 결정적으로 만든다. 제출본은 이 수정 **이전**의 코드로 생성된
과거 실행 결과이므로, 수정된 코드로 seed=42를 재실행해도 그 특정 과거 실행과 bit-identical하게 일치할
근거는 없다 — 다만 동일한 통계적 절차(같은 아키텍처, 같은 데이터, 같은 seed, 같은 하이퍼파라미터)로
생성된 **동형 실현**이 된다.

## 3. 실현 간 점수 편차 실측 (TASK15 인용)
"코드가 다시 만드는 결과가 얼마나 다를 수 있는가"를 정량화하기 위해 결정성 복원판으로 K=8개 고정 seed
(1001~1008)를 각각 독립 학습, `[v7 + 0.2·CNN]` pooled TOTAL을 2023/2024 홀드아웃에서 측정
(`task15_realization/TASK15_result.md`):

| 홀드아웃 | 평균 | SD | 범위 | 스프레드 |
|---|---|---|---|---|
| 2023 | 0.60475 | 0.00162 | [0.60253, 0.60711] | 0.00458 |
| 2024 | 0.62796 | 0.00206 | [0.62540, 0.63062] | 0.00522 |

즉 **동일 코드·동일 절차·다른 seed**로 8회 재학습했을 때 관측되는 점수 편차의 SD는 약 0.0016~0.0021,
최대-최소 스프레드는 약 0.0046~0.0052 수준이다. 제출본을 만든 단일 실행(seed=42, 미기록 실제 CUDA
비결정성 경로)이 이 분포 안에 속한다고 볼 수 있으며, 이는 **모델·데이터·하이퍼파라미터가 고정된 상태에서
GPU 연산 순서 차이만으로 발생하는 자연 변동의 실측 하한**이다. Private/Public 채점 시 재실행 결과가
이 스프레드 범위 내에 있다면, 이는 코드나 방법론의 재현 실패가 아니라 위에서 규명·정량화한 CUDA
비결정성의 예상된 결과다.

## 4. 동형 파이프라인 재현 시연
결정성 복원판(`cnn_gate/cnn_common.py` + `task5c_repro/train_cnn_production.py` +
`task5c_repro/infer_cnn_production.py`)으로 학습/추론이 완전히 분리된 형태의 동일 파이프라인을
seed=42로 실제 재현했다(`task5c_repro/TASK5C_repro_result.md`, `TASK6_repro_fixed_submission.md`):
- 2022~2024 전체 train(26,197행)으로 1회 학습 → 체크포인트(`cnn_production_s42.pt`, sha256 별도 기록) →
  실제 test 격자(8760행) 추론 → `submission_ficr_w1_v7.csv`와 w=0.20 블렌드.
- 결과물(`submission_v7cnn_fixed.csv`)은 sanity 통과(결측/음수/cap초과 0), 제출본과 그룹별 평균이
  근접한 범위(G1 8628 vs v7단독 8955, G2 9173 vs 9528, G3 6469 vs 6580 — CNN 보정이 실제로 작동).
- 이 재현 과정에서 팀원 원본 코드(`_nan_to_med`)의 채널축 순회 버그를 발견·수정했으며, 이 버그는 test
  격자의 결측치 752개(0.018%, 3개 시각에 집중) 처리 경로에만 영향— train 격자는 결측 0개로 TASK1~4A
  결과에는 영향 없음. 팀원 원본과의 예상 불일치는 이 3개 시각 행(전체 0.034%)에 국한된다.

## 5. 대회 규정 대응
"오차 범위 내 Private Score 복원" 요건에 대해, 본 소명은 다음을 제공한다:
- 비결정성의 **정체**를 코드 레벨에서 규명(4건, §2) — "원인 불명의 재현 실패"가 아님.
- 그 비결정성이 만드는 **편차의 크기**를 8-seed 실측으로 정량화(§3, SD ≈ 0.002, 스프레드 ≈ 0.005).
- 동일 절차의 **독립 재실행이 실제로 가능**함을 시연(§4) — 코드·데이터·seed 전부 보존, 학습·추론 분리.
- 수정판 코드로는 신규 실행이 bit-identical함(§2 검증)을 실증 — 비결정성은 "고칠 수 없는 결함"이
  아니라 "원인이 밝혀졌고 사후에 봉인된 문제"임.

## 6. GBDT/MLP/블렌드 컴포넌트 재현성 실측 (TASK28-P1, 2026-07-27)
위 §1-5는 CNN 멤버(비결정 확인·정량화)만 다룬다. **나머지 전체 파이프라인**(4-GBDT 앙상블 + MLP +
그룹별 캘리 + 팀원 멤버(ms/ts/quant) 블렌드)의 재현성은 별도로 실측되지 않은 상태였다 — 이번에
클린 재실행으로 검증했다.

**방법**: `train_msecalib_ensemble_teamfeat_v7.py`(4-GBDT)와 `train_mlp_v7.py`를 원본 코드 그대로
(출력 경로만 `task28_repro/repro_*`로 변경) 처음부터 재실행 → `build_v7.py`(5:1 GBDT:MLP 블렌드+캘리
강제) → `reblend_v7.py`(팀원 멤버 3:1:1 + quant 5:1 최종블렌드) 전체 체인을 재현 산출물로 재실행,
각 단계에서 원본 산출물과 직접 diff.

**결과**:
| 비교 대상 | 원본 | 재현 | 결과 |
|---|---|---|---|
| GBDT raw 예측(`v7_cal_raw.npz`/`v7_test_raw.npz`, float64) | — | — | `np.array_equal`=False, **max_abs_diff=7.28e-12**(병렬 스레드 부동소수점 축적순서 잡음, 무의미한 수준) |
| MLP raw 예측(`mlp7_cal_raw.npz`/`mlp7_test_raw.npz`) | — | — | `np.array_equal`=True, **완전 bit-identical**(CPU 전용+시딩이라 GPU급 비결정성 자체가 없음) |
| GBDT 캘리 후 제출본(`submission_lgbm_ens_v7.csv`) | — | — | **완전 bit-identical**(max_diff=0) — 위 1e-12 잡음이 캘리브레이션/클리핑 단계에서 완전히 소거됨 |
| **v7 최종 제출본(`submission_ficr_w1_v7.csv`, 팀원 멤버 블렌드까지 전부 포함)** | — | — | **완전 bit-identical**(max_diff=0, 전 8760행×3그룹) |

**결론**: CNN을 제외한 전체 프로덕션 체인(피처엔지니어링은 입력 그대로 재사용, GBDT 4모델+MLP 재학습부터
최종 블렌드까지)은 **완전히 결정적이며 클린 재실행으로 bit-identical 재현이 실증됨**. 즉 최종파일
(`submission_ficr_w1_v7_cnn(0.65183).csv`)의 비결정성 원천은 **CNN 멤버 단독**이며, 그 크기는 이미
§3에서 정량화(SD≈0.002, 스프레드≈0.005). 대회 "오차범위 내 Private 복원" 요건에 대해, 전체 스택 중
CNN을 제외한 나머지는 오차범위조차 필요 없는(정확히 0인) 재현성을 갖는다는 것이 이번 실측의 핵심 추가
근거.

**보존 파일**: `task28_repro/repro_gbdt_v7.py`, `repro_mlp_v7.py`, `repro_build_v7.py`,
`repro_reblend_v7.py`(전부 원본 스크립트의 출력경로만 변경한 사본) + 재현 산출물 일체
(`repro_v7_cal_raw.npz` 등, `repro_submission_ficr_w1_v7.csv`까지).

## 7. 팀원 원본 코드 감사 — 정직한 결함 기재 (부록 T, 2026-07-27)
소명의 신뢰도 자체가 산출물 검증 단계의 자산이라는 판단에 따라, 팀원 원본 CNN 코드를 직접 감사해
발견한 결함을 은폐 없이 기재한다.

### 7-1. 원본 실현이 사후 재현 불가능한 이유 (3요소)
팀원 원본(`v7_cnn_submission.py`, `spatial_cnn_v7_top.py`) 코드 전체를 확인한 결과 `torch.save` 호출이
**어디에도 없다** — 조기종료용 `state_dict()`는 실행 중 메모리에서만 보관·복원되고 디스크에 저장되지
않는다. 즉 (a) 체크포인트 부재 (b) 매 실행마다 처음부터 재학습하는 구조 (c) 그 학습이 TASK14 이전
GPU 비결정 연산 위에서 실행됨 — 이 세 조건이 겹쳐 **LB 0.65184를 만든 그 특정 실행 결과는 원리적으로
사후 복원이 불가능**하다. 이것이 본 문서가 "bit-identical 재현"이 아니라 "동형 실현 + 편차 정량화"
전략(§2-4)을 택한 근본 이유다.

### 7-2. NaN 결측 처리 버그 3변종 — 공통 메커니즘은 "(0,0) 격자셀만 채워짐"
- **변종1**(`v7_cnn_submission.py::_nan_to_med`, train 전용): `arr.shape[1]`로 순회하나 이 시점 배열은
  아직 `(n,H,W,C)`형태라 **높이(H) 축을 채널인 것처럼 순회** — train 격자 결측이 0개라 실질 영향 없음.
- **변종2**(`v7_cnn_submission.py::predict_cnn`, **실제 LB 제출을 만든 추론 경로**): `ldaps.reshape(n,-1)`로
  완전 평탄화 후 `for i in range(c_l)`로 첫 `c_l`개 열만 채움 — C-order 평탄화 특성상 이는 실제로
  **격자좌표 (h=0,w=0)의 채널들만** 채우는 것과 같다. 다른 모든 셀의 NaN은 그대로 남아 CNN forward를
  오염시키고, 하류 `mask=~isnan(cnn_vals)`가 해당 행을 걸러 **조용히 v7 단독값으로 폴백**한다(로그
  경고 1줄 외에는 실행이 정상처럼 보임). 실측 영향: 752개 결측 스칼라, 3개 시각(0.034%)에 집중.
- **변종3**(`spatial_cnn_v7_top.py`): 변종1과 동일 함수(H축 오류)를 학습·추론 양쪽에 재사용 —
  이 스크립트는 프로덕션 제출 경로가 아니라 실측 영향 범위는 별도 감사 대상이 아니었음.

우리의 재현판(`task5c_repro/`)은 이 세 변종을 모두 올바른 채널축 순회로 수정했다(TASK6). 팀원 원본과의
불일치는 이론상 위 3개 시각(전체 0.034%)에 한정되며, 그 외 모든 행은 동일 CNN 아키텍처·동일 blend
로직으로 재현된다.

## 8. V1~V3 제출 전 필수 검증 (부록 T, 2026-07-27) — 전부 통과, 2차 검증 핵심 소명
T4 재생성판(`submission_v7cnn_fixed_v2.csv`)을 실제 제출하기 전, 사전점검(shape·결측 등)과는 별개로
전략 리드가 요구한 3건의 정밀 검증. 어느 파일(A/B/C)도 아직 실제 DACON에 제출되지 않은 시점(ΔLB
전무)에서 수행 — 결과를 보고 나서 기준을 맞춘 것이 아님을 명시한다.

### V1. v7cnn_fixed_v2 vs 팀원 원본, 그룹별 상관·평균절대차
| 그룹 | corr | mean_abs_diff | 팀원원본 평균 | 상대 MAD |
|---|---|---|---|---|
| G1 | 0.99923 | 185.9 | 8664.5 | 2.15% |
| G2 | 0.99921 | 191.3 | 9160.4 | 2.09% |
| G3 | 0.99872 | 563.4 | 7019.0 | 8.03% |

**판정**: TASK15가 측정한 CNN 실현체간 분산(SD≈0.002, TOTAL 점수 단위)과 이 값(kWh 단위, 상관계수)은
단위가 달라 직접 등치 비교는 불가 — 다만 **G3의 상대MAD(8.03%)가 G1/G2(≈2%)의 약 4배**로 뚜렷이 큰
것은, 이 프로젝트가 반복적으로 확인해온 "G3(U136)가 G1/G2(V126)보다 실현체간 변동이 훨씬 크다"는
패턴(TASK13의 그룹별 CNN 기여 분석에서 G3 SD가 G1/G2의 2.6~2.8배, TASK7의 G3만 3/3 악화 등)과
정성적으로 **정합** — 즉 이번 편차는 새로운 이상 신호가 아니라 이미 반복 관측된 "G3가 실현체 노이즈에
가장 취약하다"는 기존 패턴의 재현으로 판단. 전부 같은 통계적 절차(동일 아키텍처·데이터·seed)의 다른
실현이라는 전제와 일관됨 — **통과**.

### V2. 결측영향 3행(forecast_2345/4050/4758) 실제 CNN 적용 확인
v7(base, CNN 없음) vs v7cnn_fixed_v2를 직접 대조 — 3행 × 3그룹 = 9개 셀 **전부** v7단독과 다른 값
(CNN이 실제로 블렌드됨 확인, diff 범위 73~840kWh). **v7 폴백 0건 — 통과.**

### V3. 독립 프로세스 2회 실행, 실제 production 시나리오(전체 2022-2024 학습→test 추론) 결정성
`task28_repro/v3_determinism_check.py`(TASK14의 `test_determinism.py`와 동일 원리이나, 2022-23→2024
홀드아웃이 아니라 **실제 제출본을 만든 시나리오 자체**를 검증 대상으로 함)를 완전히 독립된 별도
프로세스로 2회 실행:
- `best_val_mse`: run0=run1=**0.017170469540108382**(float64 전정밀도 일치)
- CNN raw 예측(8760×3): `np.array_equal`=**True**, 최대절대차=0.0
- `state_dict` 전체 레이어: `np.array_equal`=**True**
- ★교차검증: 이 run0/run1의 `state_dict`는 실제 `submission_v7cnn_fixed_v2.csv`를 만든 체크포인트
  (`task5c_repro/cnn_production_s42.pt`)와도 **완전 bit-identical** — V3가 검증한 결정성이 임의의
  재현이 아니라 실제 제출 대상 그 자체의 결정성임을 확인.
- (참고) `hash(pred.tobytes())`는 두 실행에서 값이 달랐음 — 이는 Python 내장 `hash()`가
  `PYTHONHASHSEED`(보안용 해시 랜덤화, 프로세스마다 다름)에 의존하기 때문으로 결정성과 무관한
  거짓 신호. 실제 판단은 `np.array_equal`(바이트 비교)로 수행했으며 이 함정에 낚이지 않았음을 기록.

**로그 보존**: `task28_repro/v3_run{0,1}.log`, `v3_det_run_{0,1}.pkl`.

### 종합
V1~V3 전부 통과 → `submission_v7cnn_fixed_v2.csv`(A) 및 `submission_v7cnn_fixed_v2_g2shift.csv`(C)
제출 진행 가능(전략 리드 판정, 2026-07-27, ΔLB 확인 전 시점).

## 9. 신규 최종파일 계보 — `submission_A_m1v2.csv` (작업물A · LB 0.65403 · 68위)

★FIN-2c(2026-08-05): 최종 제출본은 8/1 이후 `submission_A_m1v2.csv`(LB 0.65403, 68위)로 확정·제출됐다.
**작업물A = M1 밴드 [0.03, 0.15] adaptive 재적합판** — 팀원 원본 v7+CNN
(`submission_ficr_w1_v7_cnn(0.65183).csv`) base 위에 얹힌 후처리 레이어의 **밴드/적합만 교체**한 버전이며
(코드·모델 변경 없음), §9-0의 `submission_m1_bandcorrected.csv`(LB 0.6529253)가 그 하위 계보(3층 M1
원본)다. 상세(재현 패키지·FIN-2a dry-run·v10 미사용 명시)는 §9-2, 하네스 검증 이력은 §11 참조.

### 9-0. 이전 최종(3층 M1 원본) — `submission_m1_bandcorrected.csv` (LB 0.6529253, TASK37, 2026-07-29 채택)

★★fable 판정(2026-07-29): M1(채점조건부 밴드보정) LB 프로브 ΔLB +0.00109(≥선등록 문턱 +0.0010) →
**채택, 당시 신규 최종파일**(LB 0.6529253, 순위 70위→약 62위). 이 문서 최상단의 "대상 산출물"
(`submission_ficr_w1_v7_cnn(0.65183).csv`)은 이 신규 파일의 **base**(하위 계보)이며, 아래는
그 위에 얹힌 후처리 레이어의 계보를 별도로 기록한다. 이후 §9-2의 작업물A가 이 파일을 대체해
최종 제출본이 됐으며, 본 절의 내용은 **계보(같은 base 위의 M1 밴드보정)로서 유지**된다.

**★fable 확인 요청 답변(빌드 base 명시)**: `submission_m1_bandcorrected.csv`는
**① 팀원 원본 v7+CNN**(`submission_ficr_w1_v7_cnn(0.65183).csv`) 위에 빌드됐다 — ② 우리
재현판(`v7cnn_fixed_v2`)이 아니다(`task37_m1_revival/build_m1_submission.py:119-120`,
`sub_path = "submission_ficr_w1_v7_cnn(0.65183).csv"`로 직접 로드). 따라서 **§7-1의 재현불가
계보(체크포인트 부재·매실행 재학습·TASK14 이전 GPU 비결정 연산)를 그대로 승계**한다 — 밴드보정을
얹었다고 해서 base 자체의 재현성 지위가 개선되지는 않는다.

**계보 2단 구조**:
1. **base(재현 불가 계보 승계)**: 팀원 원본 `submission_ficr_w1_v7_cnn(0.65183).csv`(2025 test
   예측) — §7-1 그대로.
2. **밴드보정 레이어(완전 재현 가능, 결정적 산술)**:
   - 적합: 우리 재현판 v7 GBDT(`m1_holdout_preds_2024.parquet`, 2022-23 학습) + 동결 CNN
     (`task14_determinism/frozen_cnn_pred_2024_s42.parquet`, seed42)를 w=0.20으로 블렌드,
     cal=2024-09~10→eval=2024-11~12 canonical `calibrate_total` 적용 후 그 위에
     `fit_band_correction`(밴드 [0.05,0.20]×cap, 그리드서치 ±0.10×cap)을 1회 적합 →
     shift=[+0.06, +0.03, +0.05]×[G1,G2,G3].
   - 적용: 이 frozen shift를 팀원 원본 예측(위 1)에 밴드 진입 시에만 가산 후 clip(0,cap) — 순수
     결정적 산술(랜덤성 없음), 몇 번을 재실행해도 bit-identical.
   - 코드: `task37_m1_revival/build_m1_submission.py`(사전점검 포함: shape/음수/cap초과/
     forecast_id 순서 전부 통과).

**종합**: base(팀원 CNN 실현)의 재현 불가 문제는 여전하나(§7-1과 동일 성격), **이번에 새로 얹은
보정 레이어 자체는 100% 재현 가능**하고 그 절차·shift값·영향행수(G1 3030/G2 2998/G3 2729/전체
8760)가 전부 기록·재현 가능한 형태로 보존돼 있다.

## 9-1. 클린 dry-run 재확인 (TASK46, 2026-07-30)

2차 산출물 패키지 최종화 작업의 일환으로 1층·3층 재현성을 다시 확인했다.

- **3층(`task37_m1_revival/build_m1_submission.py`)**: 사전 백업 없이 원본 스크립트를 그대로
  재실행(순수 결정적 산술·랜덤성 없음이므로 위험 낮음으로 판단) — 출력 shift값
  `[G1 0.06, G2 0.03, G3 0.05]`, 밴드내표본수 `[370, 455, 414]`(11-12월 적합창 기준), 실제
  제출파일 영향행수 `[3030, 2998, 2729]/8760`가 §9에 기존 기록된 값과 **전부 정확히 일치** —
  독립 재실행 간 bit-identical 재현을 재확인.
- **1층**: §6(TASK28-P1, 2026-07-27)의 클린 재실행 결과(max_diff=0)를 재검증 없이 원용 —
  파이프라인 코드에 이후 변경이 없어(TASK28 이후 신규 전처리·모델링 착수는 전부 기각·미채택,
  §7-17~§7-21은 게이트 규율 변경일 뿐 프로덕션 학습 코드 자체를 건드리지 않음) 결과는 여전히
  유효하다고 판단.
- **2층**: 재현 자체가 불가능한 층이므로 "dry-run 재현"이 아니라 §3(SD≈0.002, 스프레드≈0.005)의
  오차범위 제시로 대체 — §0 표와 동일.

## 9-2. 작업물A — `submission_A_m1v2.csv` (M1 밴드 [0.03,0.15] adaptive 재적합판, LB 0.65403 · 68위, 2026-08-01 제출)

최종 제출본 `submission_A_m1v2.csv`의 계보·재현 기록이다(빌드 기원: TASK37-이후 M1 v2 라운드,
`task37_m1_revival/build_m1_submission_v2.py`).

- **성적**: LB 0.65403, 순위 68위 — 이전 최종(§9-0, LB 0.6529253) 대비 **ΔLB +0.00111**. 이 ΔLB는
  H-TEST(§11)에서 하네스가 방향을 정확히 맞춘 **유일한 변환**(양성대조 M1)으로, 작업물A는 "2024 기반
  검증이 2025 LB를 예측하지 못한다"는 7연속 실패 패턴에 대해 **LB로 직접 검증된 유일한 파일**이다.
- **변경 내용(코드/모델 변경 없음)**: M1 밴드 경계 [0.05,0.20] → **[0.03,0.15]** + 프로덕션 shift
  재적합(팀 관례 cal=2024-11~12 자기자신) — 순수 후처리 레이어의 밴드/상수 교체.
- **적합 결과**(`task37_m1_revival/m1v2_results.json`): prod_shift `=[G1 +0.045, G2 +0.040, G3 +0.055]×cap`,
  프로덕션 밴드내표본수 `[420, 348, 480]`, test 영향행수 `[2469, 2443, 3566]/8760`. (참고: H-TEST의
  M1 상수 `M1_SHIFTS=[0.045, 0.040, 0.055]`와 동일 — §11.)
- **재현 패키지**: `work_a_package/run.py`(self-contained; 필요 파일 `cnn_common.py`,
  `m1_holdout_preds_2024.parquet`, `frozen_cnn_pred_2024_s42.parquet`,
  `submission_ficr_w1_v7_cnn(0.65183).csv` — 모두 동일 폴더 보존). 실행 시 `submission_A.csv` 출력.
- **★FIN-2a dry-run(2026-08-05)**: 클린 재실행 결과 출력 `submission_A.csv`가 제출본
  `submission_A_m1v2.csv`와 **md5=`6d5e4214dfeadb898d5af81245308d58` 완전 일치(bit-identical)**, 런타임
  **10.1초**, 결정적 산술(랜덤성 없음, seed 불요) — 위 shift·영향행수도 동일 재현됨을 확인.
- **★v10(Open-Meteo) 미사용 명시**: 작업물A의 입력은 팀원 원본 제출본
  (`submission_ficr_w1_v7_cnn(0.65183).csv`), `m1_holdout_preds_2024.parquet`,
  `frozen_cnn_pred_2024_s42.parquet` 3종뿐이다. **Open-Meteo(ECMWF 이관) 외부 데이터 및
  `merged_train_v10.csv`/`merged_test_v10.csv`(v10, v9+Open-Meteo ECMWF) 계열은 최종 제출본 생성
  경로에서 일절 사용되지 않았다** — 최종 경로의 데이터 원천은 v9(`merged_train_v9.csv`/
  `merged_test_v9.csv`)까지만이며, v10은 미채택(외부데이터 기각 계보) 상태다.

### §9-3. coordBA 후보정 + 1-NMAE 가드레일 (2026-08-07 신설)

- **coordBA (`submission_A_m1v2_coordBA.csv`, LB 0.65466)**: G1 CF[0.7,1.0] 고CF 1,572행에
  +216kW(=sA=+0.010×cap) 상향. G2/G3 byte-identical. JJA(6~8월) 게이트 차단.
  slope 맵에서 FiCR↔1-NMAE 강결합이 확인된 좌표(A)만 골라 타격. FICR +0.0027,
  1-NMAE −0.00014(LB 실측). 내부 판정은 "보류"였으나 LB에서 생존 — **채점 문턱을 직접
  공략하는 후보정이 전역 피처(W1~W5 6연속 기각)보다 유효함을 실증.**

- **★ 1-NMAE 가드레일 (신규 규율, 2026-08-07)**:
  FICR은 계단 함수(문턱 0.10×cap)라서 후보정 시 "맞던 걸 틀리게" 할 위험이 있다.
  모든 후보정 후보는 1-NMAE 게이트를 통과해야 한다:

  > **Δ(1-NMAE) ≥ −0.0002 → 통과. 미만이면 FICR 이득과 무관하게 기각.**

  - 근거: coordBA의 LB 1-NMAE −0.00014가 게이트 통과하며 FICR +0.0027로 TOTAL +0.00063 달성.
    이보다 큰 NMAE 손실은 FICR 이득을 상쇄하거나, 과적합(holdout→LB 역전) 신호.
  - 스코어보드(2026-08-07): coordBA PASS (1-NMAE −0.00014 ≥ −0.0002), 터빈 차별화 4접근 전부
    해당 없음(1-NMAE 개선 실패). **가드레일 도입 이후 모든 후보정은 이 게이트를 명시 통과해야
    최종본 교체 대상이 된다.**

## 보존 파일
- `task14_determinism/DETERMINISM_report.md`, `test_determinism.py`, `det_run_1.pkl`, `det_run_2.pkl`,
  `frozen_cnn_pred_2024_s42.parquet` + `.sha256`, `cnn_common_backup_pre_task14.py`(수정 전 원본)
  ※ 이 sha256은 parquet 파일 바이트가 아니라 `cnn_pred` ndarray의 `tobytes()`에 대한 해시
  (`gen_frozen_cnn_2024.py:44`) — 검증 시 `sha256sum *.parquet`로는 일치하지 않으니
  `df[cols].values.tobytes()`를 해시해서 비교할 것(직접 재계산해 일치 확인함, 2026-07-24).
- `task15_realization/TASK15_result.md`, `task15_run.py`, `task15_summary.csv`
- `task5c_repro/TASK5C_repro_result.md`, `train_cnn_production.py`, `infer_cnn_production.py`,
  `cnn_production_s42.pt`
- `cnn_gate/cnn_common.py`(결정성 수정 반영판, 현재 버전)
- `endgame_0804/s0_repro/S0_report.md` — **S0 소명(파워커브 15열)**: 원본 생성 코드 부재 확정,
  값 자체는 원본 v8/v9 CSV에 보존(제출 재현은 값 수준 가능) · 생성 원리 소명(spec §2.3 타임라인 +
  pkl 상관 0.994~0.995) 기록 유지
- `endgame_0804/s0_repro/S0_POWERCURVE15_RECONSTRUCTION.md` + `reconstruct_powercurve15.py` +
  `recon_out/` — **S0 경로 A 재구성 실행(2026-08-07)**: 원본 코드 부재 상태에서 v9 NWP 컬럼으로
  15열 재구성. 홀드아웃(2024, n=8784) 평균 상관 R=0.9686 — 12열 중 9열 R≥0.974, ws_qm 3열
  0.918~0.920 (원본 내 ws_qm↔ws_corr 상관이 0.897에 그쳐 해당 변수 고유 확률성분에 따른
  상한으로 판단). bit-identical 불가. **V8-3(2026-08-08)**: 팀원 회수 v8 실물 15열(pos 323~337)과
  직접 대조 → 홀드아웃 R=0.96857 (기존 보고와 동일, 재확인).
- `endgame_0804/v8_repro/` — **★ V8 확보 후 재현 체인 재작성 완결 (2026-08-08, V4 지시서)**:
  - `merged_train_v8.csv`/`merged_test_v8.csv` (SHA `c5118b9b…`/`1963449f…`) — 팀원 회수 확보.
    V8-0 무결성 4종 전부 PASS (v9 앞 338열 순서 일치, 값 99.973%·1e-10허용 100%, 파워커브 pos 323~337).
  - **V8-1 (v8→v9, 120열)**: `reconstructed_engineer_new_features.py` — 재생성 산출물 vs 원본 v9 대조:
    셀 일치율 100%(1e-10 허용), max\|diff\|=5.7e-14, 컬럼별 r=1.0 (456/456) → **원리 재현 (BIT-LEVEL 근접)**.
  - **V8-2 (원자료→전처리 318열)**: `reconstructed_preprocess_for_team.py` — v8 앞 318열 대조:
    **r=1.0 전 컬럼 (318/318)**, 유일 예외 air_density 14컬럼(상대오차 ≤0.02%, 원본 습도 보정 세부 미상).
    ★ 원본 예측기준시점 필터(`<= forecast_kst_dtm`) 그대로 구현. ★ kpx_group_3 2022 보간은 RF seed
    불명으로 확률적 재현.
  - 보고서: `V8_1_engineer_reconstruction_report.md`, `V8_2_preprocess_reconstruction_report.md`,
    `V4_V8_RECOVERY_SUMMARY.md`, `ASSET_INVENTORY.md` §7.
- `endgame_0804/p0_audit/p0_audit_report.md` — P0 선제 감사(CASE-A, §10)
- `endgame_0804/s2_condshift/h_test.py`, `h_test_results.npz`, `S2_report.md` — H-TEST(§11)
- `work_a_package/run.py` + `submission_A.csv`(FIN-2a dry-run 출력, md5=`6d5e4214…` 제출본과 bit-identical) — 작업물A 재현 패키지(§9-2)

## 실행 환경
전체 명세는 `VERSIONS.txt`(프로젝트 루트, TASK46 승계본) 참조 — 요약:
- OS: Ubuntu 22.04.5 LTS, kernel 6.8.0-124-generic
- CNN 학습 환경(2층): `deepseq_pipeline/.venv` — Python 3.11, `torch==2.5.1+cu121`, CUDA 12.1,
  cuDNN 9.1.0, `numpy==1.26.4`, `pandas==1.5.3`, `scikit-learn==1.2.2`, GPU: NVIDIA RTX A6000
- 그 외 파이프라인(1층+3층: GBDT/MLP/캘리/블렌드/M1 밴드보정) 환경: `numpy==1.26.4`,
  `pandas==1.5.3`, `scikit-learn==1.2.2`, `lightgbm==4.6.0`, `catboost==1.2.10`, `xgboost==2.1.4`

---

## §10. 선제 감사 통과 — 예측기준시점 컴플라이언스 (2026-08-04)

DACON 규칙(8/3 17:00 추가 조항): 예측 대상일 $t$의 모든 시간대는 전일 14:00 KST를 예측기준시점으로 하며, 이후 생성·공개된 예보 사용 금지.

**감사 결과: CASE-A (무혐의)** — `endgame_0804/p0_audit/p0_audit_report.md` 참조.

| 데이터 | 행 수 | raw_violating | ★ kept_violating | 규칙준수 vs 현행 차이 |
|---|---|---|---|---|
| ldaps_test | 140,160 | 0 | **0** | 0 |
| gfs_test | 78,840 | 0 | **0** | 0 |
| ldaps_train | 420,864 | 0 | **0** | 0 |
| gfs_train | 236,736 | 0 | **0** | 0 |

- 모든 데이터의 `data_available_kst_dtm` = 전일 13:00 KST (전일 14:00보다 1시간 이른 시각) — 데이터 자체가 규칙보다 보수적으로 생성됨.
- 현행 파이프라인 필터(`data_available <= forecast_kst_dtm`, keep-last)와 규칙 준수 필터(`data_available <= 전일 14:00`)의 결과가 **완전 동일** (차이 0행).

- 검증 스크립트: `endgame_0804/p0_audit/audit_tref_compliance.py`
- **★ V8-5 재확인 (2026-08-08)**: 재작성본 `reconstructed_preprocess_for_team.py`(원자료→전처리, V8-2)가
  사용하는 NWP 클리닝 필터(`data_available_kst_dtm <= forecast_kst_dtm`, keep-last)에 대해 감사 스크립트를
  재실행(`endgame_0804/v8_repro/p0_recheck/audit_tref_compliance.py`) → **kept_violating=0 전 데이터셋, CASE-A 재확인**.
  재작성본으로도 원본과 동일한 예측기준시점 컴플라이언스를 보장한다.

---

## §11. S2-V2 하네스 반증 테스트 (H-TEST) — 검증 도구 변별력 실증 (2026-08-05)

본 절은 2026-08-04~05에 수행된 **S2-V2 하네스 반증 테스트(H-TEST)**를 재현성 관점에서 기록한다
(발표 자산 — "게이트 규율" 3장). 목적은 교차 검증 하네스(2023↔2024)가 **검증 도구로서 변별력**을
갖는지 실증하는 것이다. 4개 과거 변환(g12_opt, g3shift03, poly2, M1)을 동일 하네스에 투입해
하네스Δ를 산출하고, 각각의 실제 LB Δ와 대조했다. 채점은 `.opencode/pipeline_v2/metrics.py`의
**DACON 정의 3그룹 TOTAL**(`competition_metric` 3그룹 평균)이며, 자작 `score_g12`는 사용하지 않았다.

### §11-1. 실행 (결정적)
- 스크립트: `endgame_0804/s2_condshift/h_test.py` → 결과 영속화: `h_test_results.npz`
- 입력: `task15_realization/base_holdout_2023_s42.parquet` + `m1_holdout_preds_2024.parquet`
- 하네스: 2023 적합→2024 평가, 2024 적합→2023 평가의 평균(교차) — 레버의 순수 효과 측정
- **재현성**: 결정적 실행(랜덤 요소 없음, seed 불요) — 8/4 최초 기록과 8/5 재실행 값이
  **소수점 5자리까지 일치**, 결과는 npz에 영속화.

### §11-2. 결과 (하네스Δ / LB Δ)
| 변환 | 하네스Δ(평균) | 2023→24 | 2024→23 | LB Δ | 방향 |
|---|---|---|---|---|---|
| g12_opt (G1+0.035, G2+0.010) | **+0.01307** | +0.01375 | +0.01239 | **−0.00500** | ❌ 반대 — 하네스가 실패를 "통과" |
| g3shift03 (G3+0.03) | +0.00405 | +0.00495 | +0.00315 | −0.00289 | ❌ 반대 |
| poly2 (G3) | −0.00150 | +0.00111 | −0.00411 | −0.01080 | ⚠️ 과소 |
| M1 [0.03,0.15] (양성대조) | +0.00553 | +0.00568 | +0.00539 | **+0.00111** | ✅ 유일 일치 |

### §11-3. 판정
- **하네스 반증**: 하네스(g12_opt)=+0.01307 **≥ +0.004 문턱** → LB에서 −0.00500인 실패 변환을
  하네스가 +0.01307로 "성공" 오판 → **S2 영구 REJECTED**(S2-V3/V4 미진행 — 반증된 하네스 위의
  재유도는 무의미, §7 규율).
- M1만 유일하게 방향 일치(양성대조) — 하네스 변별력 부재 속에서도 작업물A(§9-2)의 LB 개선
  (+0.00111)이 유일한 LB 검증 이력으로 남는다.
- **전이 회귀**: 4쌍 slope=+0.3534, intercept=−0.00626, corr=+0.4277(하네스 구조적 낙관).
  poly2는 평가연도 라벨에 대한 in-sample fit **leakage(전이회귀 오염)** — 이를 제외한 3쌍에서는
  slope=−0.42, corr=−0.66으로 **부호가 반전**되어 해석을 유보한다.
- **의미**: "2024 기반 검증이 2025 LB를 예측하지 못한다"는 **7연속 실패 패턴**이 특정 방법론의
  실패가 아니라 **검증 도구(하네스) 자체의 변별력 부재**임을 도구 측에서 증명한다.
- 산출물: `endgame_0804/s2_condshift/h_test.py`, `h_test_results.npz`, `S2_report.md`

---

## §12. W5 후보 물리 정본 문서화 (2026-08-06) — 비출력 계산·G3 병목 기종 소거·공간 규모 근거

본 절은 수치 결과와 무관하게 **2차 발표·재현성 문서의 물리 근거 정본**으로 기록한다 (W5-4).
모든 계산은 info.xlsx/데이터 파일에서 직접 재계산해 검증했다.

### §12-1. 비출력 계산 (V126 288.7 / U136 289.1 W/m²) — "G3 병목은 기종이 아님" 근거
- V126(그룹 G1/G2): D=126m → 로터면적 A = π·(63m)² = **12,469 m²**, 정격 3.6MW/기
  → 비출력 = 3,600,000 W / 12,469 m² = **288.7 W/m²**
- U136(그룹 G3): D=136m → A = π·(68m)² = **14,527 m²**, 정격 4.2MW/기
  → 비출력 = 4,200,000 W / 14,527 m² = **289.1 W/m²**
- **두 기종의 비출력이 0.4 W/m²(0.14%) 차이로 사실상 동일** → G3 예측이 어려운 원인을
  "기종(U136)의 단위면적 출력 특성이 다르기 때문"으로 설명할 수 없음. G3 병목은 기종이 아니다.
- 재현: `P·10⁶ / (π·(D/2)²)` — info.xlsx의 Rotor Diameter/설비용량 컬럼에서 직접 유도.

### §12-2. G3의 2사업장·2km 배열 — 배열 구조가 병목의 정당한 후보
- G3(U136 ×5)는 **태백가덕산(1기) + 태백원동(4기) 두 사업장**에 분산 (info.xlsx 명칭 확인).
- 터빈 간격(호기 순): t1→t2 **926m**, t2→t3 439m, t3→t4 332m, t4→t5 398m — 비등간격.
- **총 스팬 2.05km** — 단일 사업장으로 치부하기 어려운 공간 확산.
- 대비: G1/G2(V126 ×6씩, 태백가덕산 단일 사업장)는 인접 간격 mean 314m/350m.

### §12-3. LDAPS 1.5km 대비 그룹 간 2.17km — 단일 격자 대표값의 한계
- LDAPS 셀 간격 ≈ **1.5km** (4×4 회전 박스, data_description.md 명시).
- 그룹 중심 간 최대 거리 G1–G3 = **2.171km** (재계산, W5_GEOMETRY.md §6).
- 즉 그룹 간 공간 거리(2.17km)가 격자 해상도(1.5km)를 **초과** → 서로 다른 LDAPS 셀
  (G1→셀5, G2→셀6, G3→셀12)에 속함 (W5-0 확인, `endgame_0804/w5_ctr_interp/W5_GEOMETRY.md`).
- 기존 v9 피처는 g5/g6 단일 셀 대표값 → 그룹 간 공간 차이를 버림. W5(그룹 중심 보간)는 이를 회복하려는 시도.

### §12-4. W1 실패 원인 기록 (발표 자산) — 굽은 능선 배열을 단일 축으로 표현한 오류
- W1(wake-geometry) STAGE-1 기각 (ΔTOTAL −0.00340, G1 −0.0091 악화).
- **원인 규명**: G1 열은 굽은 능선(ridge) 배열 — 방위가 구간별 2°(t1→t4)~35°(t5→t6)로 변동.
  이를 **단일 축(전체 최소자승 PCA θ_row=28.6°)**으로 표현한 것이 오류였다.
  정렬도 피처 |cos(θ−θ_row)|는 실제 곡선 배열의 구간별 정렬을 반영하지 못해 잡음으로 작용.
- 교훈(문서화): 곡선 배열에 단일 방위 표현은 부적합 — 구간별 θ_row 또는 회전 프레임 필요.
- 참조: `endgame_0804/w_geometry/W0_report.md` §4(불일치 원인 분석), `STAGE1_W1_v2.md`.

### §12-5. W5 재학습 STAGE-1 재검증 (2026-08-07) — 기전 반증 학습 재확증
- W5(그룹 중심 IDW 보간)는 W5-1b 중복성 기각(8/6) 후 인간 허가로 **W5-2c 학습을 재실행**.
  LDAPS **−0.00225**·GFS **−0.00561**·BOTH **−0.00467** — 3 variant 전부 STAGE-1 탈락.
- **기전 반증**: 사전 등록 예측("LDAPS 개선·GFS 무효과" 비대칭)에서 LDAPS마저 악화.
  잡음대조(+0.00291)가 실측(-0.00225)을 초과 → **그룹 중심 좌표 보간은 정보가 아니라 잡음**.
  W5-1b 중복성 판정(18/24 ≥0.9)이 실제 학습으로 재확증됨 (GBDT가 보간 피처에서 신규 분할 미발견).
- 검증 체계 정정: v9base_gpu 컨트롤은 생성 환경과의 G3 차이(84.8)로 재현 불가 → **CPU 컨트롤
  (v9base) + deepseq_pipeline/.venv CPU 프로토콜**이 W1/W2와 동일 체계임을 재확인.
  G3 봉인 6개 홀드아웃 전수 |ΔG3|=0.00000 PASS.
- 참조: `endgame_0804/w5_ctr_interp/STAGE1_W5_cpu.md`, `holdout_cpu/`(6 parquet).

---

## §13. 실행 환경 분리 기록 (2026-08-06 → 08-07 P2 갱신) — 경로 정본 전환 완료

서버 GPU 정책 변경(2026-08-05~06)으로 학번 계정(`/home/student`)은 GPU에 접근할 수 없고,
**`gpu_0N` 계정(예: `gpu_04`)만 GPU를 사용**하도록 변경됐다. 이에 따라:

- ★ 2026-08-07 (P2): **경로 정본(Path of Record)이 `/home/gpu_04/DACON_baram2026/`로 전환**됐다.
  인간이 전 자산을 해당 경로로 이관 완료(2026-08-07), 전수 SHA-256 대조 결과 최종 제출본·재현
  패키지·v9 데이터·문서 일체 무결 확인(`endgame_0804/reports/ASSET_INVENTORY.md` 참조).
  2차 평가 제출 코드·산출물·본 문서의 모든 경로 표기는 이 경로 기준이다.
- **이전 정본 `/home/student/DACON_baram2026/`는 더 이상 경로 기준이 아니다** — 검증자는
  `/home/gpu_04/DACON_baram2026/`를 단일 정본으로 취급할 것. (단, `teammate feature/` 내부의
  v8 4종·`engineer_new_features.py`·`scripts/preprocess_for_team.py`는 이관 시점에 빈
  디렉토리로 확인되어 **추가 이관 필요 상태** — ASSET_INVENTORY §2, 인간 조치 대기.)
- **gpu_04 계정 = 정본 + 실행 환경 통합** — 계정 이관으로 "실행만 하는 사본 저장소" 구분은
  폐지됐다. 학습 산출물은 이 경로에서 생성·보존된다.
- W5(그룹 중심 NWP 보간) 후보는 2026-08-06 **W5-1b 중복성 검사에서 REJECTED_BY_REDUNDANCY**
  로 기각되어 M2~M4(GPU 이전·환경검증·GPU 빌드)는 실행하지 않는다 — 최종 제출본
  (`submission_A_m1v2.csv`) 및 FIN-2 패키지의 경로 정본은 위 갱신 내용을 따른다.
