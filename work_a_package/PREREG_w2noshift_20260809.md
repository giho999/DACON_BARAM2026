[선등록] TASK-W2NOSHIFT — W2 × shift 캘리 계수 스케일링 결정실험

가설: W2CHAIN에서 역전 원인이 shift 캘리(G2 +0.07 과잉 보정, W2 마스크 가중과 정면 충돌)로
      국소화됨. W2 예측에서 shift 계수를 축소(0.0/0.5)하면 W2의 마스크 교정 이득이
      캘리와의 중복 없이 드러날 수 있다. iso/M1/coordBA/블렌드/CNN은 동결 유지.

변형 3종 (이 외 탐색·추가 생성 금지, 재학습·시드추가 금지):
  V1: chain(W2) with shift 계수 × 0.0  (전 그룹 완전 제거)
  V2: chain(W2) with shift 계수 × 0.5  (전 그룹 절반)
  V3: chain(W2) with shift 계수 × 0.0, 단 kpx_group_2에만 적용 (G1/G3는 원래 shift 유지)

측정: 2024 연도전이 홀드아웃(학습 2022-23) · 대회 평가식 원본 · 3seed(42/1337/2024).
대조: chain(W0) 원본 (= 0.63402), chain(W2) 원본 (= 0.62524) — W2CHAIN 재사용.
필수: 단계별 trace(blend→iso→shift→M1→coordBA) 전 변형 기록.
      shift 캘리 적합 창(cal 09-10)과 2024 겹침 구간 명시.
      ★ 비겹침 부분구간(2024-01-01~08-31) 값 별도 산출 — in-sample 오염이 덜한 유일한 숫자.

판정 (사후 변경 금지):
  [전구간 Δ(vs chain(W0)) ≥ +0.005] AND [비겹침 부분구간 Δ ≥ +0.002] AND [3seed 중 2/3 양수]
      → 제출 후보 채택
  하나라도 미달 → 기각. 변형 추가 생성 금지.

산출: work_a_package/w2noshift/ (report · results.json · submission 후보)
제출: 사용자 수행 (자동 제출 금지).
