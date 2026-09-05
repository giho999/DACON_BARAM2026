# 제3회 풍력발전량 예측 AI 경진대회

DACON 제3회 풍력발전량 예측 AI 경진대회(2026.07.06 ~ 08.14) 참가 기록.
기상예보(GFS/LDAPS) 기반으로 제주 3개 KPX 그룹(설비용량 21.6MW / 21.6MW / 21.0MW)의
시간별 발전량을 예측한다.

| 항목 | 값 |
|---|---|
| 최종 제출본 | `submission_A_m1v2.csv` |
| Public LB | **0.65403** |
| 평가 산식 | `0.5 × (1 − NMAE) + 0.5 × FICR` (실제 발전량이 설비용량 10% 이상인 시간대만 평가) |
| 계보 | v7 블렌드 + 팀원 CNN(w=0.20) → isotonic + shift 캘리브레이션 → M1 밴드 보정 [0.03, 0.15] |
| 재현성 | `work_a_package/run.py` 로 bit-identical 재현 (md5 `6d5e4214dfeadb898d5af81245308d58`) |

## 재현

`run.py` 는 학습 없이 추론·후처리만 수행한다. 필요한 예측 parquet과 base 제출본이
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
├─ 모델_스코어_대장.md            모델별 LB/로컬 스코어 대장
├─ 실험결과_통합.md 외 문서 40여 건
│
├─ work_a_package/              최종본 생성 파이프라인 (self-contained)
│   ├─ run.py                   최종 제출본 재현 스크립트
│   ├─ metricw/ metricw_p2/     metric-aware weighting 실험 (W0/W1/W2, seed 3종)
│   ├─ w2chain/ p2chain/        체인 고정(frozen) 실험
│   └─ retroharness/            과거 구간 역검증 하네스
│
└─ task5c_repro/                CNN 학습·추론 재현 패키지
```

각 실험은 실행 전 사전등록 문서(`PREREG_*.md`)를 작성하고 결과 리포트(`*_report.md`)로
닫는 구조로 진행했다.

## 저장소에 없는 것

대회 제공 데이터와 대용량 캐시는 저작권·용량 사유로 제외했다 (`.gitignore` 참고).

- `data/` — GFS / LDAPS 예보, SCADA 10분 원자료, train_labels 등 (약 650MB)
- `cache_train.parquet`, `cache_test.parquet` — 통합 피처 캐시 (459 cols, 약 130MB)
- `grid_tensors_cache_test.pkl` — CNN 격자 텐서 캐시

데이터는 [DACON 대회 페이지](https://dacon.io/)에서 직접 받아 `data/` 에 두면 된다.

## 규정 준수

- 예측기준시점 이후 정보 미사용 (data leakage 방지 규칙 준수) — 상세는 `REPRODUCIBILITY_STATEMENT.md`
- 외부 데이터 사용 없음 — `외부데이터_정리.md`
- API 기반 원격 모델 추론 없음, 학습·추론 전부 자체 서버(RTX A6000)에서 수행

## 환경

Ubuntu 22.04.5 / Python 3.11 · numpy 1.26.4 · pandas 1.5.3 · scikit-learn 1.2.2 ·
lightgbm 4.6.0 · catboost 1.2.10 · xgboost 2.1.4 · torch 2.5.1+cu121 (CUDA 12.1)

## 라이선스

코드·문서는 MIT (`LICENSE`). 대회 제공 데이터는 포함되어 있지 않으며 DACON 이용약관을 따른다.
