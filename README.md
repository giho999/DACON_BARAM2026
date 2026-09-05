# 제3회 풍력발전량 예측 AI 경진대회

DACON 제3회 풍력발전량 예측 AI 경진대회(2026.07.06 ~ 08.14) 참가 기록.
태백가덕산풍력(64.2MW, 17기)의 시간단위 발전량을 KPX 3개 그룹별로 예측한다.
입력은 NWP 예보(LDAPS 1.5km 16격자 / GFS 0.25° 9격자), 평가 대상은 2025년 전체 8,760시간.

| 항목 | 값 |
|---|---|
| 재현 정본 | `submission_A_m1v2.csv` — Public LB **0.65403** (68위) |
| 기록상 최고 Public | `submission_A_m1v2_coordBA.csv` — Public LB **0.65466** (67위) |
| 평가 산식 | `0.5 × (1 − NMAE) + 0.5 × FICR`, 실제 발전량이 설비용량 10% 이상인 시간대만 채점 |
| 계보 | v7 GBDT+MLP 블렌드 → 팀원 CNN w=0.20 → iso/shift 캘리브레이션 → M1 밴드 보정 → coordBA 후보정 |
| 재현성 | `work_a_package/run.py` 로 bit-identical 재현 (md5 `6d5e4214dfeadb898d5af81245308d58`, 약 10초, 결정적) |

설비용량은 G1 21,600 / G2 21,600 / G3 21,000 kWh.
G1·G2는 VESTAS V126 각 6기, G3는 UNISON U136 5기이고 허브고도는 117m로 동일하다.
FICR은 시간별 오차율 `e = |ŷ−y|/cap` 에 대해 `e ≤ 0.06 → 4.0원`, `e ≤ 0.08 → 3.0원`, 그 외 0원인
**계단형 단가**를 발전량 가중 합산한 뒤 이론상 최대 정산금으로 나눈 값이다. 이 계단 구조가
아래 모든 설계 결정의 출발점이다.

---

# 방법론

## 1. EDA — 무엇이 병목인지부터 확정

**점수 축 분해.** 인접 순위 팀들은 NMAE가 아니라 FICR로 우리를 이겼다. 상위권 대비 격차의
약 60%가 FICR이었고, FICR은 고출력 시간대 오차를 6%cap 밑으로 눌러야 계단이 뛰는 구조라
가중치로 짜낼 수 있는 레버가 아니라 base 정밀도의 비선형 하류 보상이었다.

**분포 이동.** train(2022–2024) vs test(2025) adversarial validation **AUC 0.965**.
이후 재측정한 D2 진단도 AUC 0.9321. "train에서 좋아도 test에서 무너진다"가 상수라는 뜻이고,
이 사실 하나가 뒤의 연도전이 홀드아웃 게이트 전체의 근거가 된다.

**G3 병목.** 세 그룹 중 G3만 구조적으로 나쁘다.

| 지표 | 값 |
|---|---|
| FICR | G3 0.3268 vs G1+G2 0.4331 (gap **−0.106**) |
| 중고풍속대 편향 | 평균 **−1,216 kWh** 과소예측 |
| 실현체 변동성(SD) | G1/G2의 **2.6~2.8배** |
| 학습 데이터 | 2022 라벨이 공식 NaN → 실질 **2023 1년치만** |

원인 후보를 EDA로 좁혔다. 기종 차이 가설은 비출력 계산으로 반증됐고(V126 288.7 W/m² vs
U136 289.1 W/m², 차이 0.14%), 실제 구조적 차이는 **배열**이었다 — G3만 태백가덕산 1기 +
태백원동 4기의 2사업장 2.05km 확산 배열이고 G1/G2는 단일 사업장(터빈 간격 314m/350m)이다.
그룹 간 거리(G1–G3 2.17km)가 LDAPS 해상도 1.5km를 넘어서, 그룹별로 서로 다른 격자 셀이
필요하다는 것도 여기서 나왔다.

**관측 대 예보 편향 (SCADA 10분 원자료 기반).**

1. 나셀 풍속계는 전단 외삽 자유류 풍속 대비 약 **16% 낮게** 측정 — 문헌상 NTF 보정계수 범위(5~15%)와 정합
2. LDAPS는 고풍속을 과대예보 — SCADA/LDAPS-50m 비율이 풍속 구간 전체에서 단조 감소(1.041 → 0.772), 16~30m/s 구간에서 0.772
3. GFS 100m는 지형 평탄화로 약한 프록시 — 평균 3.46 m/s vs 실측 허브고도 6.99 m/s (0.25° 격자가 능선을 평탄화)

SCADA ↔ KPX 그룹 매핑(G1=vestas01-06, G2=vestas07-12, G3=unison01-05)은 상관 0.95~0.97로 사전 검증했다.

**스택 가치 정량화.** 원시 격자를 그대로 GBDT에 넣은 것(TOTAL 0.62198) 대비 전체 프로덕션
스택(피처엔지니어링 + CNN + 캘리 + 블렌드)은 0.62933 — 차이 **0.00735**. 우리가 손댈 수 있는
층 전체의 가치가 0.7%p라는 뜻이고, 이 숫자가 이후 "어디에 시간을 쓸지" 판단의 기준이 됐다.

## 2. 전처리 — 원자료 → 458열

3단계로 쌓았고, 각 단계 산출물을 별도 파일로 고정해 재현 가능하게 유지했다.

```
원자료 (LDAPS 16격자 + GFS 9격자 + SCADA 10분 + KPX 라벨)
  │
  ├─① 전처리 파이프라인 (9단계)          → preprocessed_train/test.csv   26,303 × 323
  ├─② 파워커브 계열 15열                 → merged_train/test_v8.csv      26,303 × 338
  └─③ 신규 NWP 피처 120열                → merged_train/test_v9.csv      26,303 × 458
```

**① 9단계 파이프라인**

1. 원자료 로드 (train_labels / ldaps / gfs / scada_vestas / scada_unison)
2. 날짜 변환 & SCADA QC — Unison 풍향 −180~180° → 0~360° 정규화
3. **NWP 클리닝 — `data_available_kst_dtm ≤ forecast_kst_dtm` 필터** 후 `(forecast_kst_dtm, grid_id)` 중복 제거(최신 유지). 리키지 방지의 1차 방어선
4. 히스토리컬 풍속 바이어스 맵 — SCADA 실측 vs LDAPS/GFS grid5 풍속의 month×hour 평균 편차 288구간
5. NWP 피처 엔지니어링 — 공기밀도(온도·습도·기압), ws / ws² / ws³, 풍향 sin·cos, 바이어스 보정계수(LDAPS 0.95 · GFS 1.02), 멀티그리드 집계(LDAPS grids [4,5,6,7,12] × mean/max/std, GFS grids [1~9] × mean/max/std)
6. 시계열 피처 — hour, month, day_of_year, hour_sin/cos
7. 공간 그라디언트 — grid 간 풍속·기압 차(`spatial_grad_ws_5_6`), 전단(`wind_shear_5_50m`), 대기 안정도
8. 시차 피처 — `lag1` / `lead1` / `rolling_3h`
9. G3 결측 라벨 RandomForest 보간 (200 trees, max_depth 8, G1/G2 + NWP 피처 기반)

**② 파워커브 계열 15열** — 그룹별 5계열 × 3그룹:
`ws_corr`(보정 풍속) · `pc_pred`(파워커브 발전량 예측) · `pc_clean`(정제 파워커브) ·
`ws_qm`(분위수 매핑 풍속) · `bestlayer_vcubed`(최적 고도 풍속³).
17기 개별 터빈 RF 파워커브 모델에서 유도했고 `ws_corr → pc_pred` 상관은 G1 0.9940 / G2 0.9954 / G3 0.9954.

**③ 신규 120열** — LDAPS 76열(grids 5,6,11,12 × 19: 난류, 경계층 전단, 운량, 지표상태, 10m 풍속) +
GFS 44열(grids 5,6 × 22: 850/700/500hPa 풍속, 전단, 돌풍, 허브고도 U/V, PBL, lapse rate).

최종 학습 피처 453개의 구성은 GFS 250 (55%) · LDAPS 175 (39%) · 파워커브 6 · 공간 그라디언트 4 ·
최적층 v³ 3 · 시간 파생 3 · 풍속 보정 3 · quantile map 3 · 윈드시어 2 · 기타 4.

**피처 금지 규율** — `clean_1/2/3`은 sample_weight 전용, `y_corr_1/2/3`은 보정 참조 전용.
피처로 흘러들면 assert로 차단한다.

## 3. 모델링

**GBDT 4종** — LGB-MSE + XGBoost + CatBoost + LGB-L1.
`n_estimators=317, max_depth=6, lr=0.03, num_leaves=64, subsample=0.8, colsample=0.7, subsample_freq=1`.
이상치 마스크는 피처가 아니라 **sample_weight**로 들어간다 — clean 1.0 / dirty 0.3.

**MLP** — 456→256→128→64→1, BatchNorm + ReLU + Dropout 0.3, Adam 1e-3, 가중 L1 손실.

**SpatialCNN (dual-branch)** — 이 프로젝트에서 4개월간 유일하게 FICR 축을 움직인 레버.

```
LDAPS 4×4 격자 ─ Conv3×3(64) → BN → Conv3×3(128) → AdaptiveAvgPool ┐
                                                                    ├─ concat → MLP head → 3그룹 출력
GFS   3×3 격자 ─ Conv3×3(64) → BN → Conv3×3(128) → AdaptiveAvgPool ┘
```

핵심은 구조가 아니라 **입력**이다. 손으로 만든 물리 피처(허브 외삽·밀도·경험적 파워커브)는
전부 실패했는데, base가 이미 v³·공기밀도·전단을 다 갖고 있어 노이즈였기 때문이다.
반면 원시 격자를 그대로 소비한 CNN은 LB **+0.00258**(전이율 91%)을 냈다.
"새 알고리즘"이 아니라 "새 정보원"이 오차 구조를 가른다는 것 — GBM→ExtraTrees 교체는 예측 상관
r=0.99로 사실상 같은 신호였다.

**결정성 복원** — 원본 CNN은 체크포인트 미저장 + 매 실행 재학습 + GPU 비결정 연산으로 재현이
불가능했다. `CUBLAS_WORKSPACE_CONFIG` + `deterministic_algorithms` + `AdaptiveAvgPool2d` → 수동 mean
+ DataLoader generator 명시로 bit-identical 실행을 복원하고, seed42 실현체를 동결해
`frozen_cnn_pred_2024_s42.parquet`로 보존했다. 실현체 간 변동은 8-seed 실측 SD ≈ 0.002.
seed로 좋은 실현체를 고르는 것은 금지했다 — 2023 vs 2024 홀드아웃 순위 상관이 **ρ = 0.02**(무상관)로,
"좋은 seed"에 전이 근거가 없음을 실증했기 때문이다.

## 4. 앙상블 & 후처리

```
base   = (5·GBDT + 1·MLP) / 6
ens    = (3·base + 1·ms + 1·ts) / 5          # ms=multistage, ts=timeseries (팀원 멤버)
final  = (5·ens + 1·quant) / 6                # quant = FICR 전용 고분위 quantile 멤버
sub    = 0.8·final + 0.2·CNN
```

**캘리브레이션 (그룹별 형태 고정)**

| 그룹 | 형태 | 그리드 | 선택 기준 |
|---|---|---|---|
| G1 | shift_only | ±0.08 | FICR-max |
| G2 | shift_only | ±0.08 | FICR-max |
| G3 | **iso + shift** | ±0.08 | FICR-max |

G3만 iso가 필수다. G3 bias는 비선형(저풍속 과대·중고풍속 과소)이라 slope=1로 고정하는
shift_only는 평균 −762 kWh 하향을 만든다. 이걸 shift_only로 바꾼 실험(E1)은 로컬 게이트
**+0.01930**을 통과하고도 LB **−0.00813**으로 무너졌다.

**M1 밴드 보정** — 채택된 유일한 신규 레버.
기존 캘리는 그룹당 상수 1개를 전 예측범위에 균일 적용한다. M1은 **이미 캘리된 예측값 중
`[0.03, 0.15] × cap` 구간에 들어온 행에만** 2차 shift를 그룹당 1개씩(총 3개) 더한다.
채점 카운트 문턱(정답의 10%cap) 근방에 몰린 행을 6%/8% 계단의 유리한 쪽으로 밀어넣는 개입이다.
효과 분해가 **FICR 84% / NMAE 16%** 로 비대칭인 것 자체가 이 기전과 정합한다.
로컬 +0.00121 → LB +0.00109, **전이율 90%** — 프로젝트 최고.

전이된 이유가 중요하다. 같은 FICR 계단 구조를 **학습 손실함수**에 넣은 v8b는 LB −0.00089,
표본가중으로 넣은 TASK43도 −0.00140/−0.00074로 전부 음전이했다. 사후 조정은 행 단위로 독립이라
"채점 안 되는 행의 오차는 공짜"라는 전제가 정확히 성립하지만, 학습 단계에서는 모델이 전체 행이
공유하는 함수라 그 전제가 깨진다. **무엇을 이용하는가가 아니라 어디에 적용하는가가 전이를 가른다.**

**coordBA 후보정** — G1 CF[0.7, 1.0] 고CF 1,572행에 +216kW(=0.010×cap) 상향, G2/G3는 byte-identical,
JJA(6~8월) 차단. LB FICR +0.0027 / 1-NMAE −0.00014 → TOTAL +0.00063.

**1-NMAE 가드레일** — coordBA 이후 신설한 규율. FICR이 계단 함수라 후보정은 "맞던 걸 틀리게" 만들
위험이 있다. 모든 후보정 후보는 `Δ(1-NMAE) ≥ −0.0002`를 통과해야 하고, 미달이면 FICR 이득과
무관하게 기각한다.

## 5. 검증 규율 — 이 프로젝트의 실제 산출물

로컬에서 좋아 보인 것이 LB에서 뒤집히는 일이 반복됐고, 그래서 판정 체계 자체를 설계했다.

- **rolling forward CV** — 12창 pooled, 3-seed. 월평균 채점의 겨울 편향을 확인하고 pooled로 전환
- **연도전이 홀드아웃 게이트** — 2022–23 학습 → 2024 평가, 채택 문턱 +0.0010. v8b(로컬 +0.00191 → LB −0.00089) 실패 직후 필수화
- **사전등록(PREREG)** — 실행 전에 판정 규칙을 문서로 고정하고 리포트로 닫는다. `work_a_package/PREREG_*.md` 참조
- **잡음 대조** — 같은 개수의 순수 가우시안 잡음 컬럼을 병행 투입. 일 블록 부트스트랩 SE가 0.00119라 채택 문턱 +0.0010은 **0.84 SE**에 불과하다. 즉 문턱 단독은 판별력이 없다. 실제로 잡음이 real과 구분되지 않거나 더 나은 점수를 낸 사례가 4건 나왔다(ECMWF, clean-downweight, LDAPS 16셀 확장, u\* 마찰속도). M1이 진짜 신호였던 근거는 문턱 통과가 아니라 real(+0.00121)과 잡음(−0.00259) 사이의 **간격 3.2 SE** 였다

이 규율이 실제로 한 일:

| 사례 | 로컬 | LB | 결과 |
|---|---|---|---|
| v8b (value-loss) | +0.00191, 전 게이트 통과 | −0.00089 | 가양성 차단 실패 → 게이트 강화 계기 |
| v9 (딥시계열) | 게이트1 +0.00153 통과 | (미제출) | **게이트2에서 차단 — 헛제출 방지** |
| E1 (G3 shift_only) | +0.01930 | −0.00813 | 캘리 형태 축 폐쇄 |
| dm 재구현 | +0.02506 | −0.01982 | 결정이론 레이어 폐쇄 |
| M1 | +0.00121, 잡음과 3.2 SE 분리 | **+0.00109** | **채택** |

최종적으로 로컬 +0.004 ~ +0.034 개선이 LB에서 **8연속 반전**했다. 배제한 축을 표로 남겼다 —
전처리 48+ 후보 0건 통과, 모델 클래스 16종 중 채택 0건(GBDT+MLP+CNN 조합만 생존), 외부 NWP
데이터 경로 4중 봉쇄, 손실함수 개입 전량 음전이. 기각 건 재점검도 0/3 역전.

---

## 재현

`run.py`는 학습 없이 추론·후처리만 한다. 필요한 예측 parquet과 base 제출본이
`work_a_package/` 안에 함께 들어 있어 클론 직후 바로 실행된다.

```bash
cd work_a_package
python run.py            # 약 10초, 결정적 (seed 불요) → submission_A.csv 생성
cmp submission_A.csv ../submission_A_m1v2.csv   # 차이 없음
```

학습 단계 전체 재현은 원본 데이터가 필요하다 (아래 "저장소에 없는 것" 참고).

- CNN 학습: `task5c_repro/train_cnn_production.py`
- CNN 추론: `task5c_repro/infer_cnn_production.py`
- 동결 가중치: `task5c_repro/cnn_production_s42.pt`

## 저장소 구조

```
├─ VERIFIER_PACKAGE.md          2차 평가 검증자용 안내 (최종파일 식별·재현·소명 인덱스)
├─ REPRODUCIBILITY_STATEMENT.md 재현성 소명 전문
├─ DACON_풍력_전체기록.md         단일 진실원 — 전 실험 로그
├─ 파이프라인_통합.md             데이터 계보·모델 파이프라인·실행 명령
├─ raw_to_v9_dataset_spec.md    원자료 → v9 데이터셋 생성 명세
├─ 실험결과_통합.md               전 실험 결과·실패 패턴·교훈
├─ 모델_스코어_대장.md            모델별 로컬/예상/실제 LB 대장
├─ 발표_골격_v1.md                2차 평가 발표 골격
├─ 외 문서 40여 건
│
├─ work_a_package/              최종본 생성 파이프라인 (self-contained)
│   ├─ run.py                   최종 제출본 재현 스크립트
│   ├─ PREREG_*.md              실험 사전등록 문서
│   ├─ metricw/ metricw_p2/     metric-aware weighting 실험 (W0/W1/W2 × seed 3종)
│   ├─ w2chain/ p2chain/        동결 체인 스왑 실험
│   └─ retroharness/            과거 구간 역검증 하네스
│
└─ task5c_repro/                CNN 학습·추론 재현 패키지
```

각 실험은 실행 전 사전등록 문서(`PREREG_*.md`)를 작성하고 결과 리포트(`*_report.md`)로 닫는
구조로 진행했다.

## 저장소에 없는 것

대회 제공 데이터와 대용량 캐시는 저작권·용량 사유로 제외했다 (`.gitignore` 참고).

- `data/` — GFS / LDAPS 예보, SCADA 10분 원자료, train_labels 등 (약 650MB)
- `cache_train.parquet`, `cache_test.parquet` — 통합 피처 캐시 (459 cols, 약 130MB)
- `grid_tensors_cache_test.pkl` — CNN 격자 텐서 캐시

데이터는 [DACON 대회 페이지](https://dacon.io/)에서 직접 받아 `data/` 에 두면 된다.

## 규정 준수

- 예측기준시점 이후 정보 미사용 — NWP 클리닝 단계에서 `data_available_kst_dtm ≤ forecast_kst_dtm` 필터로 강제. 상세는 `REPRODUCIBILITY_STATEMENT.md`
- 외부 데이터 사용 없음 — 4개 경로(공개 NWP 실사, JMA API, 원시 GRIB, 기상청 대용량신청) 전부 규칙 또는 커버리지에서 막혀 미채택. `외부데이터_정리.md`
- API 기반 원격 모델 추론 없음 — 학습·추론 전부 자체 서버(RTX A6000)에서 수행

## 환경

Ubuntu 22.04.5 / Python 3.11 · numpy 1.26.4 · pandas 1.5.3 · scikit-learn 1.2.2 ·
lightgbm 4.6.0 · catboost 1.2.10 · xgboost 2.1.4 · torch 2.5.1+cu121 (CUDA 12.1) · GPU RTX A6000

## 라이선스

코드·문서는 MIT (`LICENSE`). 대회 제공 데이터는 포함되어 있지 않으며 DACON 이용약관을 따른다.
