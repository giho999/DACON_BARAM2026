# merged_train_v9.csv / merged_test_v9.csv 생성 명세서

> 작성일: 2026-08-01
> 상태: **재현 검증 완료 (PASS)** — 생성 스크립트 재실행 결과가 원본 파일과 100% 일치 확인

---

## 1. 개요

| 항목 | 내용 |
|---|---|
| 산출물 | `teammate feature/merged_train_v9.csv`, `teammate feature/merged_test_v9.csv` |
| 생성 스크립트 | `engineer_new_features.py` (프로젝트 루트, 343줄) |
| 생성 시점 | v9 파일 최종 수정: 2026-07-10 07:41 (v8은 07-09 09:45) |
| 목적 | v8(기존 팀 merge 데이터)에 원시 LDAPS/GFS의 신규 피처 120개를 엔지니어링하여 병합 |

**v9 = v8 + 원시 LDAPS/GFS에서 새로 만든 120개 피처 (LDAPS 76 + GFS 44)**

---

## 2. 입력 데이터

| 입력 | 경로 | 역할 |
|---|---|---|
| `merged_train_v8.csv` | `teammate feature/` | 베이스 학습 데이터 (26303×338) |
| `merged_test_v8.csv` | `teammate feature/` | 베이스 테스트 데이터 (8760×336) |
| `ldaps_train.csv` / `ldaps_test.csv` | `데이터/open/train|test/` | 신규 LDAPS 피처 원천 |
| `gfs_train.csv` / `gfs_test.csv` | `데이터/open/train|test/` | 신규 GFS 피처 원천 |
| `config.py` | 프로젝트 루트 | `DATA_DIR` 경로 참조 |

---

## 3. 생성 과정 (4단계)

### Phase A — 데이터 정렬 검증
- v8 파일과 원시 LDAPS/GFS를 `forecast_kst_dtm` 기준으로 교차 검증.
- 확인 결과: train은 merged(kst_dtm) ∩ raw(forecast_kst_dtm) = 26303/26304 타임스탬프 일치, test는 8760/8760 완전 일치.

### Phase B — LDAPS 신규 피처 (grids [5, 6, 11, 12])
원시 LDAPS에서 grid별 19개 피처 × 4 grid = **76개** 생성:
- 난류: `ldaps_turbulence_u/v/50m_{g}`
- 경계층 전단: `ldaps_blshear_x/y/mag_{g}`
- 운량: `ldaps_vlcdc/hcc/lcc/mcc/tcc_{g}`
- 지표 상태: `ldaps_temp_2m/dewpoint_2m/rh_2m/pressure_msl/pressure_surf_{g}`
- 연직 풍속 프로파일: `ldaps_10u/10v/ws_10m_raw_{g}`

### Phase C — GFS 신규 피처 (grids [5, 6])
원시 GFS에서 grid별 22개 피처 × 2 grid = **44개** 생성:
- 고도별 풍속·전단: `gfs_ws_850hPa/700hPa/500hPa_{g}`, `gfs_shear_850_700/700_500/total_{g}`
- 풍향: `gfs_wind_dir_850_{g}`, `gfs_dir_shear_850_700_{g}`
- 돌풍: `gfs_gust_{g}`
- 허브고도 U/V: `gfs_u/v_80m_{g}`, `gfs_u/v_100m_{g}`
- 경계층: `gfs_pbl_u/v_{g}`
- 온도·안정도: `gfs_temp_850/700_{g}`, `gfs_lapse_rate_{g}`
- 기타: `gfs_tcc/dlwrf/dswrf/prate_{g}`

### Phase D — v8에 병합하여 v9 저장
- **train**: v8의 `kst_dtm` ↔ 신규 LDAPS/GFS의 `forecast_kst_dtm` left join 후 중복 컬럼 제거 → (26303, 458)
- **test**: `forecast_kst_dtm` 기준 left join → (8760, 456)
- `teammate feature/merged_train_v9.csv`, `merged_test_v9.csv`로 저장 (index=False, utf-8)

---

## 4. 산출물 규격

| 파일 | shape | 비고 |
|---|---|---|
| `merged_train_v9.csv` | 26303 × 458 | 338(v8) + 120(신규) |
| `merged_test_v9.csv` | 8760 × 456 | 336(v8) + 120(신규) |

신규 피처 NaN 발생 시 스크립트가 WARNING으로 출력 (실행 시 NaN WARNING 0건).

---

## 5. 재현 방법

```bash
# 요구사항: python3, pandas, numpy
python engineer_new_features.py
# 실행 위치: 프로젝트 루트 (입력 v8/원시데이터 경로가 코드에 상대 경로로 하드코딩됨)
```

주의: 원본 스크립트는 `teammate feature/`에 직접 덮어쓰므로, 재현 시에는 **출력 경로를 수정한 복사본**으로 실행할 것을 권장 (아래 검증 절차 참조).

---

## 6. 재현 검증 결과 (2026-08-01 실행)

**절차**: 원본 스크립트를 복사해 **출력 경로만 임시 폴더로 변경**한 복사본으로 실행 → 재생성 결과와 원본 v9를 정밀 비교. 원본 파일은 실행 전/후 SHA-256 해시로 무결성 확인.

| 비교 항목 | merged_train_v9 | merged_test_v9 |
|---|---|---|
| shape | (26303, 458) = (26303, 458) | (8760, 456) = (8760, 456) |
| 컬럼 집합 | 동일 | 동일 |
| key(kst_dtm) 값 | 동일 | 동일 |
| NaN 패턴 차이 | 0 셀 | 0 셀 |
| 문자열 컬럼 | 동일 (1개) | 동일 (2개) |
| 수치 값 | 11,994,168/11,994,168 셀 일치 (100.000000%), max\|diff\|=0 | 3,968,280/3,968,280 셀 일치 (100.000000%), max\|diff\|=0 |
| **판정** | **PASS** | **PASS** |

**원본 무결성**: 검증 실행 전후 `teammate feature/` 6개 CSV의 SHA-256 해시 100% 동일, `git status` 변경 0건 → 검증 과정에서 원본이 1바이트도 변경되지 않음.

---

## 7. 관련 문서 및 데이터

| 문서 | 위치 | 내용 |
|---|---|---|
| 피처 카탈로그 | `data/intermediate/existing_feature_catalog.md` | v9의 458열 구성, TEAM 피처 제외 규칙, 모델 피처 474개 규명 |
| 전처리→제출 플랜 | `.omo/plans/dacon-preprocessing-to-submission.md` | v9를 실험 입력으로 사용 |
| 프로젝트 히스토리 | `project_history.md` | 주요 데이터 위치 기록 |
| 팀원 핸드오프 | `the_highest_score_code_and_handoff_0723/Handoff_document_for_the_agent.txt`, `GIHO1/2/3.txt` | v9를 입력으로 사용하는 v7 파이프라인 |

> ⚠️ 참고: `scripts/preprocess_for_team.py`는 `preprocessed_train/test_data.csv`(v9 이전 0.6244 레시피)를 생성하는 별개 파이프라인이며, v9 생성에는 사용되지 않음. v9 생성의 유일한 스크립트는 `engineer_new_features.py`.
