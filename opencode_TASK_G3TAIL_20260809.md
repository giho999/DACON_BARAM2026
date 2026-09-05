# [opencode 작업지시] TASK-G3TAIL — G3 상단 tail 확장 프로브

> 이 문서를 opencode 세션에 그대로 붙여넣고 실행시킬 것.
> 작업 루트: `/home/gpu_04/DACON_baram2026/`
> 작성 시각: 2026-08-09 (1차 마감 D-5)

---

## 0. 역할과 전제

너는 DACON BARAM 2026 팀 `냥볶밥`의 실행 에이전트다.
현재 LB 0.65465(67위), 30위 컷 0.65983 → **격차 −0.00518**.

**이 작업은 신규 모델 학습이 아니다.** 기존 최고점 제출본에 대한
**단조(monotone)·순서보존 후처리 변환 1종**을 적용해 프로브 파일을 생성하고,
가드 검증 리포트를 만드는 것이 전부다.

### 0-1. 왜 이 작업을 하는가 (진단 근거 — 재확인 대상)

1. 인수인계서의 "병목 100% FICR" 진단은 **낡았다**. 컷이 0.87496→0.87850으로 올라
   NMAE도 −0.00457 적자다. 갭 기여도 = NMAE 44.2% / FICR 55.8%.
2. "후처리 헤드룸 ≈ 0"(D3)은 **리더보드가 반증**한다. 우리와 NMAE가 같거나 더
   나쁜 팀(12·14·19위)이 FICR을 +0.015~0.021 더 낸다. → 헤드룸은 base가 아니라
   **오차 분포 모양**에 있다.
3. **핵심 발견**: `submission_A_m1v2_coordBA.csv`의 G3 예측은 CF **0.9019**에서
   물리적으로 막혀 있다. 실측 G3는 CF 1.006까지 간다.
   - `actual > 0.982` (= pmax+0.08) → **e>0.08 확정 → price 0 확정**: FICR 가중 **3.49%**
   - `actual > 0.962` (= pmax+0.06) → **만점 불가**: FICR 가중 **6.94%**
   - 예측 CF 질량비(예측/실측, masked≥0.10 기준): `[0.85,0.95)=0.06`, `[0.95,1.10)=0.00`
   - G3 FICR 가중의 **44.1%가 CF≥0.7 구간**에 있는데 우리는 그 상단을 비워두고 있다.
4. G1/G2도 `[0.95,1.10)` 질량비 0.26 / 0.24로 결손이나, 상한(1.000/0.984)은 정상.
   → **이번 프로브는 G3 단독. G1/G2는 손대지 않는다.**

### 0-2. 왜 8연속 실패와 다른가

| 레버 | 스텝 | 적용 | ΔLB |
|---|---|---|---|
| 고출력 +5%cap 보정 (#8) | +0.05 cap | 균일·광범위 | −0.01548 |
| 좌표BA (G1 고CF) | +0.01 cap | 조건부 1,572행 | **+0.00063** |

반응곡선은 **강하게 오목**하다. 실패는 "방향이 틀림"이 아니라 **"스텝 과대 + 균일 적용"**이다.
이번 변환은 (a) 순서보존 단조라 하위 87%가 불변, (b) 스텝이 좌표BA 스케일과 동급,
(c) **shift가 아니라 ceiling release**로 §3 닫힌 레버 목록에 없다.

---

## 1. 절대 금지 (위반 시 산출물 폐기)

- ❌ 신규 모델 학습·재학습·시드 탐색·CNN 관련 일체 (TASK15 로또 증명)
- ❌ `pipeline_v2` 계보 파일 사용
- ❌ G1 / G2 컬럼 값 변경 — **바이트 단위 동일 유지가 하드 요구사항**
- ❌ 2025 평가구간 정답·실측 운영데이터·외부데이터 사용
- ❌ D2 스트레스셋 사용 (참고지표로도 금지)
- ❌ 임의 파라미터 추가 탐색 (아래 λ 사다리 외 값 생성 금지)
- ❌ 같은 날 복수 LB 프로브

---

## 2. 입력

| 항목 | 경로 |
|---|---|
| base 제출본 (LB 0.65465) | `work_a_package/submission_A_m1v2_coordBA.csv` |
| 학습 라벨 (분포 대조용) | `train_labels.csv` (2022–2024) |
| 형식 기준 | `sample_submission.csv` |

용량 상수: `kpx_group_1=21600`, `kpx_group_2=21600`, `kpx_group_3=21000` (kWh)

---

## 3. 변환 스펙 (이것만 구현. 변형 금지)

그룹 $g$, 용량 $C_g$, 앵커 $a$, 확장계수 $\lambda \ge 1$에 대해

$$
\hat y' \;=\; C_g \cdot \mathrm{clip}\!\left(
\begin{cases}
\hat y / C_g, & \hat y/C_g \le a \\[4pt]
a + \lambda\left(\hat y/C_g - a\right), & \hat y/C_g > a
\end{cases}
,\; 0,\; 1\right)
$$

- **본 프로브 고정값: `g = kpx_group_3` 단독, `a = 0.70`**
- λ 사다리: `{1.00(널), 1.10, 1.20, 1.35, 1.50}`
- 단조 증가 보존, 앵커 이하 구간 완전 불변, 상한 clip = 1.0·cap

### 3-1. 기대 산출값 (검증 assertion — 반드시 일치해야 함)

base G3: `pred_max_cf = 0.9019`, `cf>0.70 인 행 = 1,129행 (12.89%)`

| λ | new_max_cf | 변경행 | Δ평균(변경행) | Δmax | Δ평균(전체 8760행) |
|---|---|---|---|---|---|
| 1.00 | 0.9019 | **0** | — | 0 | 0 |
| 1.10 | 0.9221 | 1129 | 0.00663 cap | 0.02019 cap | 0.000854 cap |
| **1.20** | **0.9423** | **1129** | **0.01326 cap** | **0.04038 cap** | **0.001708 cap** |
| **1.35** | **0.9725** | **1129** | **0.02320 cap** | **0.07066 cap** | **0.002990 cap** |
| 1.50 | 1.0000 | 1129 | 0.03313 cap | 0.09911 cap | 0.004270 cap |

값이 위 표와 다르면 **base 파일이 내가 분석한 것과 다르다는 뜻** → 즉시 중단하고 보고.

---

## 4. 구현 요구사항 (바이트 동일성 확보)

G1/G2의 **바이트 동일성**을 float 재포맷으로 깨뜨리면 안 된다.
→ **CSV를 문자열로 읽고, 변경 대상 컬럼만 수치 변환 후 재포맷, 나머지 필드는 원문 문자열 그대로 기록**할 것.

```python
# work_a_package/make_g3tail_probe.py  (신규 생성)
import sys, csv, hashlib, numpy as np

CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
TARGET, ANCHOR = "kpx_group_3", 0.70

def fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"

def transform(src, dst, lam, anchor=ANCHOR, target=TARGET):
    with open(src, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    j = hdr.index(target); C = CAP[target]
    n_chg = 0; deltas = []
    for r in body:
        orig = r[j]; cf = float(orig) / C
        new_cf = anchor + lam * (cf - anchor) if cf > anchor else cf
        new_cf = min(max(new_cf, 0.0), 1.0)
        new_s = fmt(new_cf * C)
        if new_s != orig:
            n_chg += 1; deltas.append(new_cf - cf)
        r[j] = new_s
    with open(dst, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([hdr] + body)
    d = np.array(deltas) if deltas else np.zeros(1)
    return dict(n_chg=n_chg, d_mean=float(d.mean()), d_max=float(d.max()),
                d_all=float(d.sum() / len(body)))

if __name__ == "__main__":
    src = "work_a_package/submission_A_m1v2_coordBA.csv"
    for lam in [1.00, 1.10, 1.20, 1.35, 1.50]:
        dst = f"work_a_package/submission_coordBA_g3tail_l{int(lam*100):03d}.csv"
        st = transform(src, dst, lam)
        md5 = hashlib.md5(open(dst, "rb").read()).hexdigest()
        print(lam, st, md5)
```

> ⚠️ `fmt()`가 base의 원문 표기를 재현하지 못하면 λ=1.00 널테스트가 실패한다.
> **널테스트 실패 시 `fmt()`를 base 표기에 맞게 고칠 것** (base는 소수 3자리, 후행 0 제거 형식).
> 그래도 안 맞으면 **"변경 대상 행만 재포맷하고 나머지 행의 해당 필드는 원문 유지"** 로 전환.

---

## 5. 가드 (3층) — 전부 PASS해야 제출 후보

### L1 — 미접촉 보장
- `kpx_group_1`, `kpx_group_2` 컬럼의 **원문 문자열이 base와 100% 일치** (문자열 비교, 부동소수 비교 아님)
- `forecast_id`, `forecast_kst_dtm` 순서·값 100% 일치
- 행 수 = 8760, 헤더 = `forecast_id,forecast_kst_dtm,kpx_group_1,kpx_group_2,kpx_group_3`

### L2 — 변경량 정합
- 변경 행 수 == **1129** (λ>1.00), == **0** (λ=1.00)
- `Δ평균 / Δmax / 전체평균Δ`가 §3-1 표와 소수 5자리까지 일치
- 앵커 이하(`cf ≤ 0.70`) 행 중 변경된 행 == **0**

### L3 — 형식·물리 정합
- NaN / 빈값 / 음수 **0건**
- `0 ≤ 값 ≤ cap` 전부 만족
- **단조성 보존**: base G3 값 순위와 변환 후 순위의 Spearman ρ == 1.0 (완전 일치)
- UTF-8, `sample_submission.csv`와 컬럼·행수·ID 완전 대응

### NULL — 잡음대조 (필수)
- **λ=1.00 산출물의 md5 == base 파일의 md5**
- 불일치 시 **전체 산출물 폐기**, 파이프라인 버그로 간주하고 §4 경고문 따라 수정 후 재실행

---

## 6. 검증 리포트 필수 항목

`work_a_package/g3tail_report.md`로 생성. 아래 표를 base / λ=1.20 / λ=1.35 각각에 대해 산출:

1. **G3 몰수 가중 표** — `train_labels.csv`의 G3 masked(≥0.10cap) 실측 분포 기준
   - `actual > pmax+0.06` 의 FICR 가중 비율 (만점 불가)
   - `actual > pmax+0.08` 의 FICR 가중 비율 (0점 확정)
   - **base 기준 기대값: 6.94% / 3.49%. λ=1.20에서 둘 다 ≈0%로 떨어져야 한다.**
2. **CF 구간별 질량비 표** (예측/실측, masked 기준)
   구간: `[0.10,0.30) [0.30,0.50) [0.50,0.70) [0.70,0.85) [0.85,0.95) [0.95,1.10)`
   - base G3 기대값: `1.35 / 0.78 / 1.16 / 1.06 / 0.06 / 0.00`
   - 변환 후 `[0.85,0.95)` 비율이 0.06 → 유의하게 상승해야 성공
3. **가드 L1/L2/L3 + NULL 판정 결과** (PASS/FAIL, 근거 수치)
4. **md5 / SHA-256** 전 산출물
5. **재현 커맨드** 1줄

리포트에 **로컬 스코어 게이트 수치는 넣지 말 것.** 로컬 게이트는 8연속으로 반증됐고,
이번 판정 기준은 §7의 LB 프로브다. 로컬 수치를 적으면 판단이 오염된다.

---

## 7. 선등록 (제출 전 반드시 작성)

`work_a_package/PREREG_g3tail_20260809.md`:

```
[선등록] TASK-G3TAIL
가설: G3 예측 상한 0.9019는 모델 산출물의 구조적 결손이며,
      FICR 가중 3.49%(0점 확정) + 6.94%(만점 불가)를 몰수하고 있다.
      순서보존 단조 tail 확장으로 이를 해제하면 FICR이 상승한다.
변환: g=kpx_group_3 단독, a=0.70, λ ∈ {1.20, 1.35}. G1/G2 불변.
판정: ΔLB(vs 0.65465) 기준
      ≥ +0.0010  → 채택, 다음 λ로 진행
      |ΔLB| < 0.0010 → 무판정, 더 큰 λ로 1회만 재프로브
      ≤ −0.0010  → 이 레버 영구기각, G1/G2 트랙(a=0.85)으로 전환
프로브 예산: 2회 (8/10 λ=1.20 → 8/11 λ=1.35). 하루 1회 초과 금지.
방어선: submission_A_m1v2_coordBA.csv (0.65465)를 항상 선택 상태로 복귀 가능하게 유지.
```

**선등록 파일을 먼저 커밋한 뒤에 제출할 것.** 사후 문턱 조정 금지.

---

## 8. 실행 순서

1. base 파일 md5 기록 → 리포트에 명기
2. `make_g3tail_probe.py` 생성 및 λ 5종 산출
3. **NULL 테스트 (λ=1.00 md5 일치)** — 실패 시 여기서 중단·수정·재실행
4. 가드 L1/L2/L3 전수 검증
5. §3-1 기대값 표와 대조 (불일치 시 중단·보고)
6. `g3tail_report.md` 생성
7. `PREREG_g3tail_20260809.md` 작성
8. `DACON_풍력_전체기록.md`(단일 진실원)에 TASK-G3TAIL 항목 추가
9. `모델_스코어_대장.md`에 후보 등록 (LB 칸은 비워둠)

## 9. 산출물 목록

```
work_a_package/make_g3tail_probe.py
work_a_package/submission_coordBA_g3tail_l100.csv   (널테스트용, 제출 금지)
work_a_package/submission_coordBA_g3tail_l110.csv
work_a_package/submission_coordBA_g3tail_l120.csv   ← 1차 제출 후보
work_a_package/submission_coordBA_g3tail_l135.csv   ← 2차 제출 후보
work_a_package/submission_coordBA_g3tail_l150.csv   (예비)
work_a_package/g3tail_report.md
work_a_package/PREREG_g3tail_20260809.md
```

## 10. 완료 보고 형식

```
[TASK-G3TAIL 완료]
base md5: ...
NULL(λ=1.00) 일치: PASS/FAIL
가드: L1 ../ L2 ../ L3 ..
몰수 가중: base 6.94%/3.49% → λ1.20 ..%/..%  → λ1.35 ..%/..%
질량비 [0.85,0.95): base 0.06 → λ1.20 .. → λ1.35 ..
1차 제출 권고: submission_coordBA_g3tail_l120.csv (md5 ...)
이상 징후: (없음 / 기술)
```

---

## 11. 규정 안전성 메모 (2차 평가 대비)

- 본 변환은 **예측값에 대한 단조 후처리**이며 예보 데이터 외 정보를 쓰지 않는다.
- 앵커·λ의 근거가 되는 CF 분포는 `train_labels.csv`(2022–2024, 공개 학습데이터)에서만
  산출했다. 평가구간 정답·실측 운영데이터·외부데이터 일절 미사용.
- **단, λ의 최종값이 LB로 결정되면 2차 발표에서 소명이 약해진다.**
  → 반드시 별도 트랙(TASK-METRICW, 지표 정합 가중 재학습)을 병행해
  λ가 임의 상수가 아니라 **평가 마스크 절단편향의 교정량**임을 원리로 유도할 것.
  (별도 지시서 참조)
