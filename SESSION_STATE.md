# SESSION_STATE — opencode 세션 (2026-08-02)

> **진짜 진입점은 `.opencode/pipeline_v2/SESSION_STATE.md`입니다 — 이 파일은 요약본.**

## 현재 상태 (3줄)
- **최종파일**: `submission_A_m1v2.csv` (Public LB 0.65403, 68위)
- **30위 컷**: 0.65788 (격차 -0.00385), **마감 8/14 09:59**
- **🔴 제출 대기**: `submission_pcg3_C.csv` — pc_pred G3 합성 + L1-synergy, CV +0.00421

## 제출 이력
| # | 파일 | LB | 판정 |
|---|---|---|---|
| 1 | submission_m1_bandcorrected.csv | 0.65293 | 팀 이전 |
| 2 | **submission_A_m1v2.csv** | **0.65403** | ✅ 최종 |
| 3 | submission_g3synth_E1_v2.csv | 0.64590 | ❌ shift_only |
| 4 | submission_A_m1v3.csv | 0.65377 | ❌ M1 튜닝 |
| 5 | **submission_pcg3_C.csv** | 미제출 | 🔴 대기 |

## 실패 패턴
- **E1, M1v3 모두**: holdout 게이트 +0.005~+0.019 → LB에서 역전 또는 소멸
- 근본 원인: 2024 최적화가 2025 LB에 전이되지 않음
- 교훈: 파라미터 튜닝(캘리, M1)으로는 추가 개선 불가

## pc_pred G3 접근 (현재)
- **변경**: G3 2022 라벨을 RF(G2-corr 0.988) → pc_pred 파워커브(G2-corr 0.862)로 교체
- **레벨**: 전처리 데이터 — 파라미터 튜닝이 아님
- **기대**: 데이터 다양성 증가 → 모델 일반화 개선 → 2025 전이 기대

## 닫힌 축 (요약)
- 전처리: 48+8 후보 중 0/56 채택
- 모델 클래스: 16종 전부 기각
- 데이터 경로: 4중 봉쇄
- 후처리: M1 밴드/캘리 튜닝 한계 도달 (E1, M1v3)

## 단일 진실원
- `.opencode/pipeline_v2/SESSION_STATE.md` — **주 진입점**
- `.opencode/pipeline_v2/REVIEW_E1.md` — E1 실패 분석
- `experiments.md` — 전체 실험 로그
