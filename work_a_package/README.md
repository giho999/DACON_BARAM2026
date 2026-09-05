# 작업물 A 재현 패키지

## 개요
팀 최종 제출본(submission_m1_bandcorrected.csv, LB 0.65293)의
M1 밴드 보정 밴드를 [0.05, 0.20] → [0.03, 0.15] 로 변경한 버전입니다.

- **LB 결과**: **0.65403** (기존 +0.00111)
- **변경 내용**: M1 밴드 경계 + shift 재적합 (코드/모델 변경 없음)

## 필요 환경
- Python 3.8+
- numpy, pandas, scikit-learn

```bash
pip install numpy pandas scikit-learn
```

## 필요 파일 (모두 같은 폴더에)
| 파일 | 설명 | 크기 |
|---|---|---|
| run.py | 실행 스크립트 (이 파일) | 6KB |
| cnn_common.py | score_fn, blend_with_fallback (CNN 게이트 공통) | 18KB |
| m1_holdout_preds_2024.parquet | v7 GBDT 2024년 홀드아웃 예측 (shift 적합용) | 396KB |
| frozen_cnn_pred_2024_s42.parquet | 동결 CNN seed42 2024년 예측 (shift 적합용) | 298KB |
| submission_ficr_w1_v7_cnn(0.65183).csv | **입력** — 팀원 원본 제출파일 (base, LB 0.65184) | 532KB |
| submission_A_m1v2.csv | **출력** — 제출 완료본 (LB 0.65403, run.py로 재현 가능) | 599KB |

## 실행

```bash
python run.py
```

## 출력
- `submission_A.csv` — DACON 제출용 파일 (8760행)
- 콘솔에 로컬 FICR 점수 출력

## 동작 원리
1. v7 GBDT + frozen CNN(seed42) 블렌드 (w=0.20) → 2024년 홀드아웃 예측
2. calibrate_total (iso+shift, cal=2024-09~10 → eval=2024-11~12)
3. M1 밴드 보정: [0.03, 0.15] 밴드로 11-12월 자기자신에 shift 적합
4. 동일 shift를 팀원 원본 제출파일(2025 test)에 적용

## 비고
- run.py는 재현성 보장을 위해 모든 함수를 자체 포함 (외부 의존성: numpy, pandas, sklearn, cnn_common.py)
- 원본 build_m1_submission_v2.py (task37_m1_revival/) 에서 포팅됨
- 생성일: 2026-08-01
