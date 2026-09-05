#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-G3TAIL — G3 상단 tail 확장 프로브 (단조·순서보존 후처리 변환)
==================================================================
지시서: opencode_TASK_G3TAIL_20260809.md (§4 구현 요구사항 기반)

변환: g=kpx_group_3 단독, a=0.70, λ ∈ {1.00, 1.10, 1.20, 1.35, 1.50}
  y' = C_g * clip( y/C_g                    if y/C_g <= a
                   a + λ*(y/C_g - a)        if y/C_g > a  , 0, 1 )

바이트 동일성 요구사항 (§4):
- G1/G2 컬럼은 원문 문자열 그대로 유지 (재포맷 금지)
- 널테스트(λ=1.00)는 base 파일과 md5 일치해야 함.
  base 원문에 3자리를 넘는 소수(예: '15849.8545')가 있어 §4 fmt() 단독으로는
  재현 불가 → 지시서 §4 경고문의 "변경 대상 행만 재포맷하고 나머지 행의
  해당 필드는 원문 유지" 방식으로 전환.
  - 변경 판정: 수치 변화(|Δcf| > 1e-9) 기준. λ=1.00은 수치 변화 0 → 전 행 원문 유지
"""
import sys, csv, hashlib, numpy as np

CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
TARGET, ANCHOR = "kpx_group_3", 0.70
EPS = 1e-9  # 수치 변화 판정 임계 (float 재조합 잡음 제거)


def fmt(v: float) -> str:
    """base 원문 표기 재현: 소수 3자리, 후행 0 제거."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def transform(src, dst, lam, anchor=ANCHOR, target=TARGET):
    with open(src, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    j = hdr.index(target)
    C = CAP[target]
    n_chg = 0
    deltas = []
    for r in body:
        orig = r[j]
        cf = float(orig) / C
        new_cf = anchor + lam * (cf - anchor) if cf > anchor else cf
        new_cf = min(max(new_cf, 0.0), 1.0)
        if abs(new_cf - cf) > EPS:  # 수치 변화 행만 재포맷, 나머지는 원문 유지
            n_chg += 1
            deltas.append(new_cf - cf)
            r[j] = fmt(new_cf * C)
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
        print(f"λ={lam:4.2f}  {st}  md5={md5}")
