# BARAM 2026 — 원자료 → merged_v9 데이터셋 전체 생성 명세서

> 작성일: 2026-08-01 · **갱신: 2026-08-08 (V8 확보 + 전 체인 재작성 완결, V4 지시서)**
> 상태: **체인 재현 완결** — 원자료→전처리·v8→v9 각 단계를 재작성 코드로 검증 완료 (V8-1/V8-2)

---

## 0. 한눈에 보는 데이터 생성 체인

```
원자료 (데이터/open/)
  │
  ├─① 전처리 파이프라인        ── preprocess_for_team.py (0.6244 레시피) ✅ 재작성본(V8-2, r=1.0)
  │     LDAPS/GFS NWP 피처 엔지니어링 + SCADA 기반 보정 + 시계열 피처
  │     → preprocessed_train/test_data.csv  (323열) [원본 산출물 부재 → 재작성본 대체]
  │
  ├─② 파워커브 계열 피처 15열 추가  ── v8 CSV 실물 확보(8/8) ✅ / 생성 원본 코드는 미확보 ⚠️
  │     ws_corr / pc_pred / pc_clean / ws_qm / bestlayer_vcubed × 3그룹   S0 재구성 R=0.96857 (v8 직접 대조)
  │     → merged_train_v8.csv (338열) / merged_test_v8.csv (336열)
  │
  └─③ 신규 피처 120열 추가     ── engineer_new_features.py ✅ 재작성본(V8-1, 100%)
        LDAPS 76열 (grids 5,6,11,12) + GFS 44열 (grids 5,6)
        → merged_train_v9.csv (458열) / merged_test_v9.csv (456열)
```

---

## 1. 단계 ① — 전처리 파이프라인 (원자료 → 323열)

### 1.1 산출물 및 검증 상태

| 산출물 | shape | 검증 상태 |
|---|---|---|
| `preprocessed_train_data.csv` | 26,303 × 323 | ✅ v8과 321개 수치 컬럼 100% 일치 확인 (아래 §3) |
| `preprocessed_test_data.csv` | 8,760 × 321 | 참조 |

### 1.2 생성 스크립트

- **정본**: `scripts/preprocess_for_team.py` (515줄, docstring: "0.6244 baseline 재현용")
- **원본 소스**: `00_quick_validation.ipynb` Cell 1~13의 전처리 로직을 추출한 것
- **산출 부산물**: `scripts/preprocessing_summary.txt` (실행 로그)

### 1.3 처리 단계 (9단계)

1. **원자료 로드**: `train_labels.csv`, `ldaps_train/test.csv`, `gfs_train/test.csv`, `scada_vestas/unison_train.csv`, `sample_submission.csv` (경로: `config.DATA_DIR = 데이터/open/`)
2. **날짜 변환 & SCADA QC**: `kst_dtm`/`forecast_kst_dtm`/`data_available_kst_dtm`을 datetime 변환, Unison 풍향 −180~180° → 0~360° 정규화
3. **NWP 클리닝**: `data_available_kst_dtm <= forecast_kst_dtm` 필터 → 정렬 → `(forecast_kst_dtm, grid_id)` 중복 제거(최신 유지)
4. **히스토리컬 풍속 바이어스 맵**: SCADA 실측 풍속 vs LDAPS/GFS grid 5 풍속의 month×hour 평균 편차 (288개 구간)
5. **NWP 피처 엔지니어링** (`process_nwp_features`):
   - 공기 밀도(온도/습도/기압 기반), 풍속(ws/ws²/ws³), 풍향(sin/cos), 바이어스 보정 계수(LDAPS 0.95, GFS 1.02)
   - LDAPS: 50m/5m 풍속, GFS: 80m/100m/850hPa 풍속·안정도
   - **멀티그리드 집계**: LDAPS grids [4,5,6,7,12] × (mean/max/std), GFS grids [1~9] × (mean/max/std)
6. **시계열 피처**: hour, month, day_of_year, hour_sin/cos
7. **공간 기울기/참조 피처**: grid간 풍속·기압 차(예: `spatial_grad_ws_5_6`), 전단(wind_shear_5_50m), 대기 안정도
8. **시차 피처**: `ldaps/gfs_ws_mean_lag1/lead1/rolling_3h` (bfill/ffill)
9. **Group 3 결측 라벨 보간**: kpx_group_3 결측치를 RandomForest(200 trees, max_depth=8)로 kpx_group_1/2 + NWP 피처 기반 예측 대체

### 1.4 특징

- **타깃 포함**: `kpx_group_1/2/3` (Group 3는 RF 보간 포함)
- **메타 컬럼**: `split`(train/val), `kst_dtm`, `forecast_id`
- **기본 분할**: train=2022-2023, val=2024 (단, CSV에는 `split` 라벨로 모두 저장)

---

## 2. 단계 ② — 파워커브 계열 피처 15열 추가 (323열 → v8 338열)

### 2.1 산출물

| 산출물 | shape | 위치 |
|---|---|---|
| `merged_train_v8.csv` | 26,303 × 338 | `teammate feature/` |
| `merged_test_v8.csv` | 8,760 × 336 | `teammate feature/` |

### 2.2 추가된 컬럼 (정확히 15개)

| prefix | 의미 | 개수 |
|---|---|---|
| `ws_corr_kpx_group_{1,2,3}` | 보정 풍속 (NWP 풍속 계열, ldaps_ws_mean_g7/g12와 corr 0.94) | 3 |
| `pc_pred_kpx_group_{1,2,3}` | 파워커브 기반 그룹 발전량 예측 (kW) | 3 |
| `pc_clean_kpx_group_{1,2,3}` | 파워커브 클린 버전 예측 (kW) | 3 |
| `ws_qm_kpx_group_{1,2,3}` | 분위수 매핑 풍속 | 3 |
| `bestlayer_vcubed_kpx_group_{1,2,3}` | 최적층(최적 고도) 풍속³ | 3 |

- 배치: v8 컬럼 순서 pos 323~337 (전처리 컬럼 뒤, 그룹별 5개씩 1/2/3 순서)
- **구조 검증**: v8의 나머지 323열은 단계① 산출물과 kst_dtm 정렬 기준 값 100% 일치 (§4)

### 2.3 생성 주체 및 근거 (2026-08-01 검증)

**판정: 사용자(본 팀원)가 같은 날 오전의 파워커브 작업 연장선에서 생성한 것으로 강력 추정** — 단, 정확한 생성 코드는 로컬 저장소에 없음.

| 근거 | 내용 |
|---|---|
| ✅ 타임라인 | 07-09 06:27 전처리 실행 → **07:43 `01_scada_power_curve.ipynb` 생성** → **07:52 `train_power_curve_prediction.csv` 생성** → **09:45 v8 생성** (모두 같은 날) |
| ✅ 값 관계 | `models/turbine_power_curve_models.pkl`(17개 터빈 RF 파워커브 모델)의 `RF(ws_corr) → pc_pred` 상관이 **G1 0.9940 / G2 0.9954 / G3 0.9954** — 같은 생성 원리(풍속→파워커브→발전량) |
| ✅ 주제 일치 | 15열 이름(파워커브 예측, 보정풍속, 분위수 매핑, 최적층) = 사용자가 그날 오전 수행한 SCADA 파워커브 작업과 동일 주제 |
| ⚠️ 정확 코드 미확보 | 15열을 *대입/생성*하는 코드는 로컬 저장소 전수 검색 결과 0건 (모든 스크립트는 `TEAM` 세트로 묶어 제외 참조만 함) |
| ⚠️ 정확 일치 아님 | pkl RF 모델 출력과 15열의 배율이 상수가 아님 (std 2.7~3.0) → 15열은 pkl과 *같은 아이디어의 다른 버전* 코드로 생성됨 |

**재현에 필요한 것**: 07-09 오전 Colab/학교 서버의 파워커브 피처 생성 노트북 (v8 생성 직전 작업분) — 2차 평가 점수복원 규정상 요구됨

---

## 3. 단계 ③ — 신규 피처 120열 추가 (v8 → v9) ✅ 검증 완료

### 3.1 산출물

| 산출물 | shape | 위치 |
|---|---|---|
| `merged_train_v9.csv` | 26,303 × 458 | `teammate feature/` |
| `merged_test_v9.csv` | 8,760 × 456 | `teammate feature/` |

### 3.2 생성 스크립트

- **정본**: `engineer_new_features.py` (프로젝트 루트, 343줄) — 유일한 v9 생성 코드
- 생성 시점: v9 파일 최종 수정 2026-07-10 07:41 (v8 07-09 09:45, v10 07-26)

### 3.3 처리 단계 (4단계)

| Phase | 내용 |
|---|---|
| A. 정렬 검증 | v8 vs 원시 LDAPS/GFS 타임스탬프 교차 확인 (train 26303/26304, test 8760/8760 일치) |
| B. LDAPS 신규 피처 (grids [5,6,11,12]) | grid별 19개 × 4 = **76열**: 난류(`turbulence_u/v/50m`), 경계층 전단(`blshear_x/y/mag`), 운량(`vlcdc/hcc/lcc/mcc/tcc`), 지표상태(`temp_2m/dewpoint_2m/rh_2m/pressure_msl/pressure_surf`), 10m 풍속(`10u/10v/ws_10m_raw`) |
| C. GFS 신규 피처 (grids [5,6]) | grid별 22개 × 2 = **44열**: 고도별 풍속(`ws_850/700/500hPa`), 전단(`shear_850_700/700_500/total`), 풍향(`wind_dir_850`, `dir_shear_850_700`), 돌풍(`gust`), 허브고도 U/V(`u/v_80m`, `u/v_100m`), 경계층(`pbl_u/v`), 온도·안정도(`temp_850/700`, `lapse_rate`), 기타(`tcc/dlwrf/dswrf/prate`) |
| D. 병합·저장 | train: v8 `kst_dtm` ↔ 신규 `forecast_kst_dtm` left join / test: `forecast_kst_dtm` 기준 left join → v9 저장 |

### 3.4 재현 검증 결과 (2026-08-01, 원본 무결성 유지하며 수행)

**절차**: 원본 스크립트를 복사해 출력 경로만 임시 폴더로 변경한 복사본으로 실행 → 재생성 결과와 원본 v9를 정밀 비교. 원본 6개 CSV는 실행 전/후 SHA-256 해시로 무결성 확인 (해시 100% 동일, git 변경 0건).

| 비교 항목 | merged_train_v9 | merged_test_v9 |
|---|---|---|
| shape | (26303, 458) | (8760, 456) |
| 컬럼 집합 | 동일 | 동일 |
| key(kst_dtm) | 동일 | 동일 |
| NaN 패턴 차이 | 0 셀 | 0 셀 |
| 문자열 컬럼 | 동일 (1개) | 동일 (2개) |
| 수치 값 | 11,994,168/11,994,168 셀 (100.000000%), max\|diff\|=0 | 3,968,280/3,968,280 셀 (100.000000%), max\|diff\|=0 |
| **판정** | **PASS** | **PASS** |

---

## 4. 단계 간 일치 검증 (2026-08-01)

| 검증 | 방법 | 결과 |
|---|---|---|
| v8 ⊇ 전처리 323열 | kst_dtm 정렬 후 321개 공통 수치 컬럼 비교 | **100.000000% 일치** (max diff 4.5e-13 = 부동소수점 노이즈) |
| v8 = 전처리 + 15열 | v8 컬럼 집합 − 전처리 컬럼 집합 = 정확히 15개, pos 323~337에 배치 | 확인 |
| 15열 = 파워커브 계열 | `models/turbine_power_curve_models.pkl`(17 RF 모델)의 `RF(ws_corr) → pc_pred` 상관 | **G1 0.9940 / G2 0.9954 / G3 0.9954** (같은 생성 원리, 정확 코드는 아님) |
| v9 = v8 + 120열 | `engineer_new_features.py` 재실행 → 원본 v9와 100% 일치 | **PASS** |

---

## 5. 전체 데이터 파일 규격 요약

| 파일 | shape | 단계 | 검증 |
|---|---|---|---|
| `preprocessed_train_data.csv` | 26,303 × 323 | ① | ✅ v8과 일치 |
| `preprocessed_test_data.csv` | 8,760 × 321 | ① | 참조 |
| `merged_train_v8.csv` | 26,303 × 338 | ② | ✅ 전처리 323 + 팀 15 확인 |
| `merged_test_v8.csv` | 8,760 × 336 | ② | 참조 |
| `merged_train_v9.csv` | 26,303 × 458 | ③ | ✅ 재현 검증 PASS |
| `merged_test_v9.csv` | 8,760 × 456 | ③ | ✅ 재현 검증 PASS |
| `merged_train_v10.csv` | (v9 + Open-Meteo ECMWF) | v10 | — (별도 작업) |
| `merged_test_v10.csv` | (v9 + Open-Meteo ECMWF) | v10 | — (별도 작업) |

---

## 6. 재현 절차 (전체 체인) — ★ 2026-08-08 V8 확보로 전 구간 재작성 완료

> V8 확보 후 전 체인 재작성·검증 완료 (V4 지시서, 상세: `endgame_0804/v8_repro/V8_1_engineer_reconstruction_report.md`,
> `V8_2_preprocess_reconstruction_report.md`, `endgame_0804/reports/ASSET_INVENTORY.md` §7).

```bash
# ① 전처리 (원자료 → 318열) — 재작성본 (원본 미확보)
python endgame_0804/v8_repro/reconstructed_preprocess_for_team.py
#   → recon_out/v8_rebuilt/preprocessed_{train,test}_reconstructed.csv
#   검증: v8 앞 318열과 컬럼별 r=1.0 (318/318), air_density만 상대오차 ≤0.02% (원리 재현)

# ② 파워커브 피처 15열 (318 → v8 338열) — v8 CSV 실물 확보 (정확 생성 코드는 여전히 미확보)
#    S0 재구성(R=0.9686)이 v8 직접 대조로 재확인됨 (V8-3, 홀드아웃 R=0.96857)

# ③ 신규 피처 120열 (v8 → v9 458열) — 재작성본 (원본 미확보)
python endgame_0804/v8_repro/reconstructed_engineer_new_features.py
#   → recon_out/v9_rebuilt/merged_{train,test}_v9_rebuilt.csv
#   검증: 원본 v9와 셀 일치율 100%(1e-10 허용), max|diff|=5.7e-14, 컬럼별 r=1.0 (456/456)
```

---

## 7. 미해결 항목 (2차 평가 점수복원 대비) — ★ 2026-08-08 갱신

| # | 항목 | 상태 | 비고 |
|---|---|---|---|
| 1 | 파워커브 피처 15열 **생성 원본 코드** | ⚠️ **원본 미확보 (재현은 가능)** | v8 CSV 실물 확보 + S0 재구성 R=0.96857 (v8 직접 대조, V8-3). bit-identical은 불가하나 상관 재현으로 소명 |
| 2 | `preprocessed_train/test_data.csv` **원본 산출물** | ⚠️ 원본 부재 (재작성 대체) | `recon_out/v8_rebuilt/preprocessed_*_reconstructed.csv` (V8-2, r=1.0) |
| 3 | **v8→v9 생성 원본 코드** (`engineer_new_features.py`) | ✅ 재작성 완료 | V8-1, 셀 일치율 100%(1e-10 허용) |
| 4 | **원자료→전처리 원본 코드** (`scripts/preprocess_for_team.py`) | ✅ 재작성 완료 | V8-2, r=1.0 전 컬럼 |
| 5 | kpx_group_3 2022년 RF 보간 **seed** | ⚠️ 불명 | bit-identical 재현 불가 — "확률적 재현". TASK33 감사로 성격 규명됨 |
| 6 | v10 생성 과정 문서화 | 미해결 | `scripts/fetch_openmeteo_weather.py` 참조 (v10 미사용) |

---

## 8. 관련 문서

| 문서 | 위치 | 내용 |
|---|---|---|
| 기존 v9 명세서 | `data/intermediate/merged_v9_creation_spec.md` | 단계③만 상세 (본 문서의 하위 집합) |
| 피처 카탈로그 | `data/intermediate/existing_feature_catalog.md` | v9 458열 구성, TEAM 피처 제외 규칙, 모델 피처 474개 |
| 전처리→제출 플랜 | `.omo/plans/dacon-preprocessing-to-submission.md` | v9를 실험 입력으로 사용 |
| 프로젝트 히스토리 | `project_history.md` | 주요 데이터 위치 기록 |
| 팀원 핸드오프 | `the_highest_score_code_and_handoff_0723/0729/` | v7+CNN 파이프라인, M1 후처리 |
