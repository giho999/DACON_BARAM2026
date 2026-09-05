# ENDGAME-2. 산출물 패키지 매니페스트

작성: 2026-07-24 (클로드코드 실행) · 8/17 2차 산출물 준비용
★★2026-07-30 갱신(TASK46, 2차 산출물 패키지 최종화): 최종파일이 TASK37/38 M1 밴드보정 채택으로
`submission_m1_bandcorrected.csv`(LB 0.6529253)로 교체됨. 탐색은 이 시점부로 완전 종결(전처리
48후보 0/48·모델클래스 16종·데이터경로 4중봉쇄, 기각건 재점검 0/3, LB 프로브 5회 전량 소진).
★★★2026-08-01 이후 갱신(FIN-2b, 작업물A 채택): 최종파일이 작업물A 제출본
`submission_A_m1v2.csv`(LB 0.65403, 68위)로 교체됨. `submission_m1_bandcorrected.csv`
(LB 0.65293)는 "이전 최종(3층 M1 원본, 하위 계보)"으로 승계 기록 유지. 작업물A 재현 패키지
`work_a_package/`에서 bit-identical 재현 확인(FIN-2a dry-run, md5 = 6d5e4214dfeadb898d5af81245308d58) —
아래 매니페스트가 제출 대상 최종판. §0(신규)에 제출 파일 목록·체크섬·디렉터리 구조를 확정한다.
★★★★2026-08-06 갱신(FIN-2e, W5 후보 기각·문서 갱신): W5(그룹 중심 NWP 보간) 후보가 W5-1b
중복성 검사에서 REJECTED_BY_REDUNDANCY로 기각 확정(인간 판정) → M2~M4(GPU 이전/빌드) 불필요,
FIN-2 전환 확정. W5-4 문서화로 `REPRODUCIBILITY_STATEMENT.md`(§12·§13 추가)와
`발표_골격_v1.md`(1장 G3 병목 소거 근거·6장 W1 실패 원인 추가)의 SHA-256이 갱신됨(아래 표
반영). **최종 제출본 `submission_A_m1v2.csv` SHA-256 불변 확인** (`8ab47f88…` 일치).
★★★★★2026-08-07 갱신(FIN-2 완결, F1~F3): 검증자용 `README.md` 신규 작성(패키지 루트),
규정 대조표(§0-A) 신규, 4대 소명 인덱스(§0-B) 신규, `pre_migration_sha256.txt` 재검증
(10항목 전부 원본과 일치) + v9 데이터 SHA 기록. **경고: `teammate feature/` 원본 경로가
gpu_04에서 접근 불가(Permission denied) — v8 원본·`engineer_new_features.py` gpu_04에
부재로 확인, ESCALATE (§0-C).**
★★★★★★2026-08-07 갱신(S0-C 후속 G2 문구 정정): `S0_POWERCURVE15_RECONSTRUCTION.md`·
`REPRODUCIBILITY_STATEMENT.md`(§보존 파일 인용부)·`endgame_0804/reports/ASSET_INVENTORY.md`(§6)의
**"목표(≥0.95) 달성" 표현을 실제 수치 기반으로 교체** — "12열 중 9열 R≥0.974, ws_qm 3열 0.918~0.920
(원본 내 ws_qm↔ws_corr 상관 0.897에 따른 상한으로 판단)". 재구성본이 소명 전용이며 제출 예측값
생성에 사용되지 않음(제출물은 원본 v9 15열 기반)을 S0_POWERCURVE15_RECONSTRUCTION.md 첫 문단에 명시.
갱신 SHA-256: `S0_POWERCURVE15_RECONSTRUCTION.md` = `3e7a119f…`,
`REPRODUCIBILITY_STATEMENT.md` = `7b2143fc…`(아래 §0 표 반영), `ASSET_INVENTORY.md` = `cf5091df…`.
**최종 제출본 `submission_A_m1v2.csv` SHA-256 불변** (`8ab47f88…`).
★★★★★★★2026-08-08 갱신(V4 지시서, V8 확보·재현 체인 완결): 팀원 회수로
`merged_train_v8.csv`(`c5118b9b…`)·`merged_test_v8.csv`(`1963449f…`) 확보. V8-0 무결성 4종 PASS →
① v8→v9(120열)를 `reconstructed_engineer_new_features.py`로 재작성·대조(셀 100%·1e-10허용, r=1.0 456/456),
② 원자료→전처리를 `reconstructed_preprocess_for_team.py`로 재작성·대조(r=1.0 318/318),
③ S0 재구성을 v8 직접 대조로 재확인(R=0.96857). **원자료→v8→v9 전 체인 재현 가능 상태로 완결.**
최종 제출본 `submission_A_m1v2.csv` SHA-256 **불변** (`8ab47f88…`). 신규 자산 SHA는 §0-E 표·ASSET_INVENTORY §2b/§7 참조.
갱신 문서 SHA-256 (본 매니페스트 §0-E 갱신분): `V4_V8_RECOVERY_SUMMARY.md` = `4af5db8b…`,
`S0_POWERCURVE15_RECONSTRUCTION.md` = `d080bdc1…`(V8-3 추가), `ASSET_INVENTORY.md` = `e53e9e4f…`(§2b/§7),
`raw_to_v9_dataset_spec.md` = `0cc4785a…`(§6·§7), `REPRODUCIBILITY_STATEMENT.md` = `70ecd059…`(§보존 파일 + §10 V8-5 재확인).

---

## 0-A. 대회 규정 대조표 (2026-08-07 FIN-2 완결)

| # | 규정 항목 | 판정 | 근거 파일 (경로) |
|---|---|---|---|
| 1 | train/inference 코드 분리 | **충족** | 학습: `task5c_repro/train_cnn_production.py`, `.opencode/pipeline_v2/train_gbdt.py`, `train_mlp_v7.py` · 추론: `task5c_repro/infer_cnn_production.py`, `work_a_package/run.py`(후처리·추론 전용, 학습 없음 — 10.1초 결정적 실행, REPRO §9-2) |
| 2 | 모델 파일 | **충족** | `task5c_repro/cnn_production_s42.pt`(CNN 가중치) · `work_a_package/m1_holdout_preds_2024.parquet` + `frozen_cnn_pred_2024_s42.parquet`(홀드아웃 예측, shift 적합용) |
| 3 | UTF-8 인코딩 | **충족** | `file` 검사: run.py/cnn_common.py/train_cnn_production.py/infer_cnn_production.py/train_gbdt.py 전부 "UTF-8 text executable" · v9 CSV는 utf-8 저장(`merged_v9_creation_spec.md` §60) |
| 4 | 라이브러리 버전·OS 기재 | **충족** | `VERSIONS.txt`(단일 진실원): OS Ubuntu 22.04.5 LTS/kernel 6.8.0-124, numpy 1.26.4/pandas 1.5.3/sklearn 1.2.2/lgbm 4.6.0/catboost 1.2.10/xgboost 2.1.4, CNN env torch 2.5.1+cu121/CUDA 12.1/cuDNN 9.1.0/RTX A6000 |
| 5 | 외부데이터 없음 명시 (v10 미사용) | **충족** | `외부데이터_정리.md` §1~7 · `REPRODUCIBILITY_STATEMENT.md` §4(grep 전수검사 0건) · v10(`merged_train/test_v10.csv`)은 프로덕션 참조 0건 — 프로덕션은 v9 원본 CSV만 사용 |

## 0-B. ★ 4대 소명 인덱스 (검증자용 — 한 번에 찾기)

| # | 소명 항목 | 요지 | 문서 위치 |
|---|---|---|---|
| ① | 파워커브 15열 | 생성 코드 원본 부재 확정, 값은 원본 CSV에 보존 → 제출 재현 가능(값 수준) | `endgame_0804/s0_repro/S0_report.md` · `raw_to_v9_dataset_spec.md` §2 |
| ② | CNN 동결 | seed42 동결 CNN(비결정성 원인 규명 TASK14, 실현체 비전이 ρ=0.02 TASK15) — 동결 예측 parquet로 후처리만 | `task14_determinism/DETERMINISM_report.md` · `task15_realization/TASK15_result.md` · `REPRODUCIBILITY_STATEMENT.md` §2~3 · `work_a_package/frozen_cnn_pred_2024_s42.parquet` |
| ③ | M1 in-sample | M1 밴드 shift는 cal 11-12 자기자신(self-ref) 적합 — in-sample 특성 사전 고지 | `task37_m1_revival/M1_mechanism_plain.md` · `TASK37_result.md` · `REPRODUCIBILITY_STATEMENT.md` §9 · `work_a_package/run.py` §3d |
| ④ | 프로토콜 드리프트 | A(GPU) vs 후보(CPU) 프로토콜 차이 → GPU 컨트롤 생성으로 완전 해소(ΔTOTAL +0.00000) | `endgame_0804/common/C0_control_vs_A.md` · `V3_SESSION_SUMMARY.md` §2 · `REPRODUCIBILITY_STATEMENT.md` §13(경로·실행환경 분리) |

## 0-C. ★ F1 teammate feature/ 확인 결과 (2026-08-07) — ✅ 해소 (계정 이관 완료)

> ★★ 2026-08-07 P2 정정: "소실 아님 — 인간이 전 자산을 gpu_04로 이동 완료". 경로 정본은
> `/home/gpu_04/DACON_baram2026/`로 전환(아래 §0-D). 아래 ①은 이관 후 재확인 결과이며,
> ②③은 여전히 **빈 디렉토리/부재로 남아 있어 추가 이관 필요**(ASSET_INVENTORY §2).

- ① 접근 문제: 이전 F1에서 "Permission denied(학번 계정 0700)"였으나, **2026-08-07 인간이 전 자산을
  gpu_04로 이관 완료** → 경로 정본 `/home/gpu_04/DACON_baram2026/`에서 전 자산 확인·SHA-256 대조
  **전부 통과** (최종본 `8ab47f88…`, v9 `269f1f6b…`/`11dfc2fa…`, work_a_package, 모델, 문서 일체).
- ② `teammate feature/` 디렉토리는 생성됐으나 **내부가 비어 있음** — v8 4종(`merged_train_v8.csv`
  /`merged_test_v8.csv` + 팀 15열 계열) 및 `engineer_new_features.py`(v9 생성 스크립트),
  `scripts/preprocess_for_team.py`(0.6244 전처리 정본)는 **gpu_04에 미이관 상태**.
- ③ **판정**: 제출·재현 핵심은 무결(제출 영향 없음). 원자료→v8→v9 **전체 체인 재현 코드**는
  추가 이관 필요 → **인간 조치 대기**(ASSET_INVENTORY §2 목록).
  - **★ V8 갱신 (2026-08-08)**: 팀원 회수로 `merged_train_v8.csv`/`merged_test_v8.csv` 확보 +
    v8→v9·원자료→전처리 재작성 완료 → ③의 미이관 상태 **해소** (V8-1/V8-2 보고서, §0-E, ASSET_INVENTORY §2b).
- 관련 문서: `endgame_0804/reports/ASSET_INVENTORY.md`(P1 전수 대조) · `pre_migration_sha256.txt` ·
  `merged_v9_creation_spec.md` · `raw_to_v9_dataset_spec.md`.

## 0-D. ★ 경로 정본 전환·상대경로화 기록 (2026-08-07 P2·P3)

| 항목 | 내용 |
|---|---|
| 경로 정본 | `/home/student/DACON_baram2026/` → **`/home/gpu_04/DACON_baram2026/`** (P2, 2026-08-07) |
| 문서 갱신 | 파이프라인_통합 §9 경로표 · REPRO §13 · README §6·6-1 · 인수인계서_통합 §3·§9 · 본 매니페스트 §0-C |
| 상대경로화 (P3) | `.opencode/pipeline_v2/g3_seed_robust.py` 절대경로 2건 → `BARAM_ROOT`/`PIPE_*` env 기반 상대경로 치환 · config.py에 `BARAM_ROOT` override 추가 |
| 검증 | `work_a_package/run.py` 재실행 → `submission_A.csv` md5 = `6d5e4214dfeadb898d5af81245308d58` (제출본과 byte 일치) 확인 |
| 남은 누락 | ~~v8 4종 · engineer_new_features.py · scripts/preprocess_for_team.py~~ → **★ 2026-08-08 해소** (v8 확보 + 재작성 완료, ASSET_INVENTORY §2b) |

## 0. 제출 대상 파일 목록·체크섬 (SHA-256, 2026-08-05 갱신)

| 파일 | 역할 | SHA-256 |
|---|---|---|
| `submission_A_m1v2.csv` | **최종 제출본**(작업물A, LB 0.65403, 68위) — `work_a_package/run.py`로 bit-identical 재현 확인(md5 일치) | `8ab47f88d2e7a14b6d3eb294f6a21693c016c758323045e977f7fd5e39be06d8` |
| `submission_m1_bandcorrected.csv` | **이전 최종**(3층 M1 원본, LB 0.65293, 하위 계보) — 작업물A 채택으로 승계 | `9c330724adf194cc991eb24a6b6d1a2292c95e51b658ac74c861c017b3e79d44` |
| `submission_ficr_w1_v7_cnn(0.65183).csv` | 최종본의 base(팀원 원본 v7+CNN, LB 0.65184) — 재현불가 계보 보존용 | `bb2f2e60a3b7df05ca9ce2c2349e6222e6bab4d6146ee6db5fc62a8d2badb0d0` |
| `submission_ficr_w1_v7.csv` | 1층(CNN 편입 전) 참고본 — bit-identical 재현 확인됨(§6/TASK28-P1) | `b0d2bf0a689abbcbf83b519da7e88c20b41f78f5ede41837dddc72370953f73f` |
| `REPRODUCIBILITY_STATEMENT.md` | 재현성 소명(3층 계보 + §11 H-TEST, G2 문구 정정 + V8-5 예측기준시점 재확인 반영) | `70ecd059a9551c3c0bfb20255bac7c48a90d76995d95e9d84caa84af74d4554f` |
| `VERSIONS.txt` | 환경·라이브러리 버전 명세(단일 진실원) | `7b3d05d8fdf43ede25e62573014af9abdb3cc34a548dd7f51e548854e5a0d24c` |
| `발표_골격_v1.md` | 발표 자료 골격(TASK46 갱신판) | `b75b2ab51c00f66291dcc4b27289c1c5d6aeaaa521de7aa21d041dd36836af3d` |
| `2차평가_체크리스트_v1.md` | 2차 평가 체크리스트(2026-08-05 FIN-2d 갱신판 — 최종파일 `submission_A_m1v2.csv`) | `91c1b87ae2471d2f7e7347a628fa1808f2dd1b4dfeef584941bc38f1b8a86755` |
| `task37_m1_revival/build_m1_submission.py` | 3층(M1 밴드보정) 빌드 스크립트, 완전 재현 가능·결정적 | `bd7ac16498074fa69729c9dbb6360ca67dd241b596c449db28693f18d05ea4f5` |

체크섬 재계산: `sha256sum <파일명>`(위 표는 CSV/MD/PY 파일 자체의 바이트 해시 — §5의 CNN 예측
ndarray 해시(`tobytes()`, sha256 별도)와는 대상이 다르므로 혼동하지 않을 것).

### §0-E. ★ V8 확보·재현 체인 완결 자산 (2026-08-08, V4 지시서)

| 파일 | 역할 | SHA-256 |
|---|---|---|
| `teammate feature/merged_train_v8.csv` | **v8 정본 (팀원 회수)** | `c5118b9b2eacc8c3e268f2a62ea1026470596217b133269b41aafb111e82489c` |
| `teammate feature/merged_test_v8.csv` | **v8 정본 (팀원 회수)** | `1963449f913a4eec4466f49ddcf5b04b223d62cfc1038327ad46c7a2ac7c9bb1` |
| `endgame_0804/v8_repro/reconstructed_engineer_new_features.py` | v8→v9(120열) 재작성 코드 | `d7262a989efd4e0aa20483365325f0945dff70679b37d37c7e4bf0f4a99af4d8` |
| `endgame_0804/v8_repro/reconstructed_preprocess_for_team.py` | 원자료→전처리 재작성 코드 | `460d1ef11790ad8cdb4b4527df96a3a0b6b84e7f4bdde1a3c2b2dddaa1b02be0` |
| `endgame_0804/v8_repro/recon_out/v9_rebuilt/merged_train_v9_rebuilt.csv` | 재생성 v9 train (r=1.0) | `8da21b09cca587e4bc86a388cd60fa41ac4368da8eb4659c77a2411be967184a` |
| `endgame_0804/v8_repro/recon_out/v9_rebuilt/merged_test_v9_rebuilt.csv` | 재생성 v9 test (r=1.0) | `c05f14ddf8173c745b44a3449f33c33208bba67494c05d857c2173928e28443f` |
| `endgame_0804/v8_repro/recon_out/v8_rebuilt/preprocessed_train_reconstructed.csv` | 재생성 전처리 train (r=1.0) | `ab11676ab5432d35ade0615c3c5d8bead1e2e8c8794919799eec10d8b08fed91` |
| `endgame_0804/v8_repro/recon_out/v8_rebuilt/preprocessed_test_reconstructed.csv` | 재생성 전처리 test (r=1.0) | `5f84e42dc4f61099392c5998df0de0c0ee11c66e4fb7c93a7769719e2751de8e` |
| `endgame_0804/v8_repro/V8_1_engineer_reconstruction_report.md` | V8-1 보고 | `a0e3c4bb6195767297d20b3ed1b6992da462fd592bcdb677206b0110c246985a` |
| `endgame_0804/v8_repro/V8_2_preprocess_reconstruction_report.md` | V8-2 보고 | `00800d0430da5d313415ec0b1420225bd0368e932aff223add8d383e66c737c3` |
| `endgame_0804/reports/V4_V8_RECOVERY_SUMMARY.md` | V8 최종 요약 | `4af5db8be32f8d3f9303f6afdcebb932ee103898c502908945d0726410915f21` |

## 1. 디렉터리 구조 (제출 패키지 기준, 최상위만)
```
DACON_baram2026/
├── README.md                             # ★ 검증자용 안내(최종파일 식별·재현 명령·환경·4대 소명 인덱스) — FIN-2 완결 신규
├── submission_A_m1v2.csv                 # 최종 제출본(작업물A, LB 0.65403, 68위)
├── submission_m1_bandcorrected.csv        # 이전 최종(3층 M1 원본, LB 0.65293, 하위 계보)
├── submission_ficr_w1_v7_cnn(0.65183).csv # base(팀원 원본, 재현불가 계보)
├── submission_ficr_w1_v7.csv              # 1층 참고본(bit-identical 재현)
├── REPRODUCIBILITY_STATEMENT.md           # 재현성 소명(3층 계보)
├── VERSIONS.txt                           # 환경 버전 명세
├── ENDGAME2_package_manifest.md           # 본 문서
├── 발표_골격_v1.md                         # 발표 자료 골격
├── 2차평가_체크리스트_v1.md                 # 제출 체크리스트
├── DACON_풍력_전체기록.md                   # 단일 진실원(전 실험 로그)
├── 외부데이터_정리.md                       # 외부데이터 실사 소명
├── D1_ecmwf_autopsy.md / D2_shift_diagnosis.md / D3_ficr_decomposition.md  # fable 진단트랙
├── M1_scored_conditional_calib.md         # M1 원 스펙 문서
├── cnn_gate/                              # CNN 결정성 수정판 공용 유틸(cnn_common.py)
├── task5c_repro/                          # CNN 학습/추론 분리 재현본
├── task14_determinism/                    # CNN 결정성 규명(TASK14) + 동결 예측
├── task15_realization/                    # CNN 실현체 비전이 검정(ρ=0.02)
├── task28_repro/                          # 1층 전체 클린 재현(bit-identical)
├── task36_stress_gate_audit/              # D2 스트레스 게이트 전수감사(폐기 판정)
├── task37_m1_revival/                     # 3층 M1 밴드보정 빌드·게이트 재현
├── work_a_package/                        # 작업물A 재현 패키지(최종 제출본 생성, FIN-2b)
└── (그 외 task*_*/ 는 전부 기각된 프로브 — 배제 공간 지도 근거 보존용, 발표_골격_v1.md 6장 참조)
```

## 2. 학습/추론 코드 분리본 점검
- `cnn_gate/cnn_common.py` — CNN 공용 유틸, TASK14 결정성 수정 반영된 현재 버전(4건 패치 적용).
  수정 전 원본은 `task14_determinism/cnn_common_backup_pre_task14.py`로 보존.
- `task5c_repro/train_cnn_production.py` + `infer_cnn_production.py` — 학습·추론 완전 분리본.
  `cnn_common.py`를 런타임 import하므로 TASK14 패치가 자동 반영됨(단, 기존 저장된
  `cnn_production_s42.pt`는 패치 이전 학습 결과 — 통계적으로 동형이나 bit-identical 재현 대상은 아님).
- 결론: 코드만으로 CNN 멤버를 처음부터 재현 가능함을 TASK5-C/TASK6에서 실증(재현본
  `submission_v7cnn_fixed.csv` 생성 및 sanity 통과).

## 3. 개발환경·라이브러리 버전
전체 명세는 `VERSIONS.txt`(TASK46 승계본, 단일 진실원) 참조 — 요약:
- OS: Ubuntu 22.04.5 LTS, kernel 6.8.0-124-generic
- CNN 학습 환경(`deepseq_pipeline/.venv`): Python 3.11, `torch==2.5.1+cu121`, CUDA 12.1, cuDNN 9.1.0,
  `numpy==1.26.4`, `pandas==1.5.3`, `scikit-learn==1.2.2`. GPU: NVIDIA RTX A6000.
- 그 외 파이프라인(GBDT/캘리/블렌드/M1 밴드보정) 환경: `numpy==1.26.4`, `pandas==1.5.3`,
  `scikit-learn==1.2.2`, `lightgbm==4.6.0`, `catboost==1.2.10`, `xgboost==2.1.4`

## 4. 외부데이터 사용 여부
없음 — 확인 완료. 프로덕션 경로(`build_v7.py` 등)와 CNN 파이프라인(`cnn_gate/`, `task5c_repro/`)
전체를 `grep`으로 점검한 결과 `asos_pipeline` / `gefs_pipeline` / `a_prime_probe` / ICON 등 외부데이터
계열에 대한 참조 0건. 해당 계열들은 전부 별도 프로브로 시도됐다가 게이트 실패로 기각되어 프로덕션에
편입되지 않았음(각 프로브의 기각 근거는 해당 TASK 결과 md에 기록). **이 사실 자체가 소명 부담이 없음을
뒷받침** — 대회 제공 데이터(LDAPS/GFS/KPX)만으로 전 스택이 구성됨.

## 5. 보존 파일 목록 (경로 확정)
- `submission_v7cnn_fixed.csv` (프로젝트 루트) — TASK6 재현 정본 후보
- `task7_ficr_recal/submission_task8_g1g2ficr.csv` — ※경로 정정: 루트 아님, `task7_ficr_recal/` 하위
- `task14_determinism/frozen_cnn_pred_2024_s42.parquet` + `.sha256`
  (sha256은 parquet 파일 바이트가 아니라 예측 ndarray `tobytes()` 해시 — `gen_frozen_cnn_2024.py:44`,
  검증 시 `df[cols].values.tobytes()`를 해시해 비교. `REPRODUCIBILITY_STATEMENT.md` §보존 파일에도 명기함)
- TASK 결과 문서 전체: `task14_determinism/DETERMINISM_report.md`,
  `task15_realization/TASK15_result.md`, `task5c_repro/TASK5C_repro_result.md`,
  `TASK6_repro_fixed_submission.md`, `CNN_TASK1_gate_reverse_validation.md`,
  `cnn_gate/TASK2_production_transfer_result.md`, `cnn_gate/TASK3_candidate_results.md`,
  `D1_ecmwf_autopsy.md`, `D2_shift_diagnosis.md`, `D3_ficr_decomposition.md`,
  `M1_scored_conditional_calib.md`
- `REPRODUCIBILITY_STATEMENT.md` (프로젝트 루트, ENDGAME-1 경로B 소명 문서 + §9 M1 계보)
- `task36_stress_gate_audit/`(존재 시) 또는 record §4-44(TASK36) — D2 스트레스 게이트셋 전수감사
  (진양성0·위양성1·위음성1 → 영구 폐기 판정 근거)
- `task37_m1_revival/`: `build_m1_submission.py`(3층 빌드, 결정적), `TASK37_result.md`,
  `M1_mechanism_plain.md`(팀원 공유용 기전 설명), `m1_gate_y1_3seed.py` + 게이트 결과 csv/json
- `task38_band_maximization/TASK38_decomposition_result.md` — FICR 84%/NMAE 16% 분해 정합성 확인
- `work_a_package/` (프로젝트 루트) — **작업물A 재현 패키지(최종 제출본 계보)**: `run.py`(self-contained, bit-identical 재현), `cnn_common.py`,
  `m1_holdout_preds_2024.parquet` + `frozen_cnn_pred_2024_s42.parquet` (shift 적합용 2024 예측),
  `submission_ficr_w1_v7_cnn(0.65183).csv`(입력 base), `submission_A_m1v2.csv`(출력 = 최종 제출본), `README.md`.
  FIN-2a dry-run으로 `submission_A.csv` 재현 → md5 `6d5e4214dfeadb898d5af81245308d58` 로 `submission_A_m1v2.csv`와 byte 일치(max diff 0.0, 전 8760행)

## 6. TASK28-P1 전체 파이프라인 재현성 실측 (2026-07-27)
`REPRODUCIBILITY_STATEMENT.md` §6 참조. GBDT(4모델)+MLP+블렌드+캘리 전체 체인을 `task28_repro/`에서
클린 재실행 → `submission_ficr_w1_v7.csv`(CNN 편입 전 v7 최종본)까지 **완전 bit-identical** 재현
확인(max_diff=0, 전 8760행). 비결정성은 CNN 멤버 단독(§3 기존 정량화, SD≈0.002)으로 이미 국한돼
있었음이 이번 실측으로 재확인됨. 보존 파일: `task28_repro/repro_gbdt_v7.py`, `repro_mlp_v7.py`,
`repro_build_v7.py`, `repro_reblend_v7.py` + 산출물 npz/csv 전체.

## 7. 최종 제출본 (2026-08-01 이후 확정, §0 표와 동일 — 여기서는 서사만 요약)
`submission_A_m1v2.csv`(LB 0.65403, 68위) — **작업물A 채택**. 구성: 팀원 원본
`submission_ficr_w1_v7_cnn(0.65183).csv`(LB 0.65184) + 캘리(iso+shift, cal 09-10 → eval 11-12) + M1 밴드
보정을 밴드 [0.03, 0.15] adaptive로 재적합(2024 홀드아웃 self-ref 적합, shifts [0.045, 0.040, 0.055]×cap,
코드·모델 변경 없음 — `work_a_package/README.md` 참조). 이전 최종 `submission_m1_bandcorrected.csv`
(LB 0.65293, 3층 M1 원본, 밴드 [0.05, 0.20])보다 ΔLB +0.00111로 채택, 하위 계보로 승계 기록 유지.
`work_a_package/run.py`로 bit-identical 재현 확인(md5 = `6d5e4214dfeadb898d5af81245308d58`, max diff 0.0, 전 8760행).
8/13·8/14 DACON "채점 대상 선택" 상태 재확인 대상은 이제 `submission_A_m1v2.csv`
(`2차평가_체크리스트_v1.md` §1 참조), 추가 제출 금지.
