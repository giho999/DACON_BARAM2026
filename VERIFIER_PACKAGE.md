# BARAM 2026 — 최종 산출물 패키지 안내 (검증자용)

> 팀: 냥볶밥 · 제3회 풍력발전량 예측 AI 경진대회 (DACON)
> 본 문서: 검증자(2차 평가 심사)가 **최종파일 식별 → 재현 → 소명 확인**을 한 번에 할 수 있도록 정리.
> 작성: 2026-08-07 (FIN-2 완결). 한국어 기준.

---

## 1. 최종 제출 파일 (정본)

| 항목 | 값 |
|---|---|
| **최종 제출본** | `submission_A_m1v2.csv` |
| SHA-256 | `8ab47f88d2e7a14b6d3eb294f6a21693c016c758323045e977f7fd5e39be06d8` |
| Public LB | **0.65403 (68위)** |
| 계보 | 팀원 원본 `submission_ficr_w1_v7_cnn(0.65183).csv`(LB 0.65184) + 캘리(iso+shift) + M1 밴드 보정 [0.03, 0.15] adaptive 재적합 |
| 생성 코드 | `work_a_package/run.py` (self-contained, 결정적 산술) |
| 재현 결과 | bit-identical — md5 `6d5e4214dfeadb898d5af81245308d58`, max diff 0.0, 전 8760행 |

이전 최종(하위 계보): `submission_m1_bandcorrected.csv` (LB 0.65293, 밴드 [0.05, 0.20]).

## 2. 재현 명령 (3줄)

```bash
export BARAM_ROOT=/home/gpu_04/DACON_baram2026          # 검증자는 여기만 지정 (P3)
cd "$BARAM_ROOT/work_a_package" && python run.py         # (1) 최종본 재현 — 약 10.1초, 출력 submission_A.csv
md5sum "$BARAM_ROOT/work_a_package/submission_A.csv" "$BARAM_ROOT/submission_A_m1v2.csv"  # (2) 둘 다 6d5e4214… 이면 byte 일치
cmp "$BARAM_ROOT/work_a_package/submission_A.csv" "$BARAM_ROOT/submission_A_m1v2.csv"      # (3) max diff 0.0 확인
```

- run.py는 학습 없이 추론·후처리만 수행 (CNN/GBDT 예측 parquet 로드 → 블렌드 → 캘리 → M1 밴드).
- CNN/GBDT **학습** 전체 재현: `task5c_repro/train_cnn_production.py`(CNN, `deepseq_pipeline/.venv`),
  `.opencode/pipeline_v2/train_gbdt.py`(GBDT 4종, ~20분) / `train_mlp_v7.py`(MLP, ~5분).

## 3. 소요 시간

| 단계 | 소요 |
|---|---|
| `run.py` (후처리 재현) | **10.1초** (결정적, seed 불요) |
| GBDT 4종 학습 | ~20분 |
| MLP 학습 | ~5분 |
| CNN 학습 | 수십 분 (GPU, RTX A6000) |

## 4. 환경 스펙 (단일 진실원: `VERSIONS.txt`)

- OS: Ubuntu 22.04.5 LTS, kernel 6.8.0-124-generic
- GBDT/MLP/캘리/블렌드/M1: numpy 1.26.4 · pandas 1.5.3 · scikit-learn 1.2.2 · lightgbm 4.6.0 · catboost 1.2.10 · xgboost 2.1.4
- CNN: Python 3.11 · torch 2.5.1+cu121 · CUDA 12.1 · cuDNN 9.1.0 · GPU NVIDIA RTX A6000

## 5. ★ 4대 소명 인덱스 (검증 시 참조)

| # | 소명 | 요지 | 문서 위치 |
|---|---|---|---|
| ① | 파워커브 15열 | 생성 코드 원본 부재 확정, 값은 원본 CSV에 보존 → 제출 재현 가능(값 수준) | `endgame_0804/s0_repro/S0_report.md` · `raw_to_v9_dataset_spec.md` §2 |
| ② | CNN 동결 | seed42 동결 CNN — 비결정성 규명(TASK14)·실현체 비전이(ρ=0.02, TASK15) | `task14_determinism/DETERMINISM_report.md` · `task15_realization/TASK15_result.md` · `REPRODUCIBILITY_STATEMENT.md` §2~3 |
| ③ | M1 in-sample | M1 밴드 shift는 11-12월 자기자신(self-ref) 적합 — in-sample 특성 사전 고지 | `task37_m1_revival/M1_mechanism_plain.md` · `REPRODUCIBILITY_STATEMENT.md` §9 · `work_a_package/run.py` §3d |
| ④ | 프로토콜 드리프트 | A(GPU) vs 후보(CPU) 차이 → GPU 컨트롤로 완전 해소(ΔTOTAL +0.00000) | `endgame_0804/common/C0_control_vs_A.md` · `V3_SESSION_SUMMARY.md` §2 · `REPRODUCIBILITY_STATEMENT.md` §13 |

## 6. 경로 정본 (2026-08-07 P2 전환)

| 구분 | 경로 | 역할 |
|---|---|---|
| **경로 정본 (Path of Record)** | `/home/gpu_04/DACON_baram2026/` | 산출물·코드·문서의 **원본 저장소** (2026-08-07 계정 이관 완료, SHA-256 전수 대조 통과). 모든 경로 표기는 이 기준. |
| 이전 정본 (폐기) | `/home/student/DACON_baram2026/` | 8/7 이전 경로 기준. 더 이상 사용하지 않음 (GPU 정책 변경으로 GPU 접근 불가). |

- 검증자는 **`/home/gpu_04/DACON_baram2026/`를 단일 정본**으로 취급하면 된다.
- 단, `teammate feature/` 내부의 v8 4종·`engineer_new_features.py`·`scripts/preprocess_for_team.py`는
  이관 시점에 빈 디렉토리로 확인 — **추가 이관 필요** (`endgame_0804/reports/ASSET_INVENTORY.md` §2).

## 6-1. 검증자 실행 안내 — BARAM_ROOT만 지정하면 실행 가능

패키지 내 모든 스크립트는 절대경로 하드코딩 없이 **프로젝트 루트 기준 상대경로**를 사용한다
(P3, 2026-08-07 치환 완료). 실행 환경을 바꿔도 루트만 지정하면 동작한다:

```bash
export BARAM_ROOT=/home/gpu_04/DACON_baram2026    # 또는 패키지를 내려받은 임의 위치
export PIPE_CACHE_TRAIN="$BARAM_ROOT/cache_train.parquet"   # (필요 시) 데이터 override
cd "$BARAM_ROOT/work_a_package" && python run.py   # 최종본 재현
```

- `BARAM_ROOT` 미지정 시 프로젝트 루트(패키지 위치)를 자동 사용 — 별도 설정 불요.
- 상세: `ENDGAME2_package_manifest.md` §0-D(경로 전환·상대경로화 기록), ASSET_INVENTORY §3.

## 7. 참고 문서

| 문서 | 내용 |
|---|---|
| `ENDGAME2_package_manifest.md` | 패키지 매니페스트(규정 대조표 §0-A, 4대 소명 인덱스 §0-B, 제출 목록 §0) |
| `REPRODUCIBILITY_STATEMENT.md` | 재현성 소명 전체 (3층 계보 + §10~13) |
| `VERSIONS.txt` | 환경·라이브러리 버전 단일 진실원 |
| `2차평가_체크리스트_v1.md` | 제출 절차·일정 체크리스트 |
| `외부데이터_정리.md` | 외부데이터 실사 소명 (사용 없음) |
| `DACON_풍력_전체기록.md` | 단일 진실원 (전 실험 로그) |
