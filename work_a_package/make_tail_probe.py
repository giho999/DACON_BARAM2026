#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-G12TAIL — 그룹별 상단 tail 확장 프로브 (누적 매트릭스, 단조·순서보존 후처리 변환)
======================================================================================
지시서: opencode_TASK_G12TAIL_20260810.md

변환 (TASK-G3TAIL과 동일 수식, 파라미터 확장):
  y' = C_g * clip( y/C_g                    if y/C_g <= a
                   a + λ*(y/C_g - a)        if y/C_g > a  , 0, 1 )

파라미터 (탐색 금지):
  kpx_group_3: a=0.70, λ∈{1.20,1.35}
  kpx_group_2: a=0.80, λ∈{1.20,1.35}
  kpx_group_1: a=0.80, λ∈{1.20,1.35}

바이트 동일성 (G3TAIL에서 검증된 방식 그대로):
- --spec 미지정 그룹은 원문 문자열 그대로 통과 (바이트 불변)
- 변경 판정: 수치 변화(|Δcf| > 1e-9) 기준. 변경 행만 재포맷, 나머지 원문 유지.
- λ=1.00 (널)은 전 그룹 수치 변화 0 → 전 행 원문 유지 → base와 md5 일치해야 함.

CLI:
  python3 make_tail_probe.py --src BASE --dst OUT --spec kpx_group_3:0.70:1.20 --spec kpx_group_2:0.80:1.20
"""
import argparse, csv, hashlib, sys, numpy as np

CAP = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
EPS = 1e-9  # 수치 변화 판정 임계 (float 재조합 잡음 제거)


def fmt(v: float) -> str:
    """base 원문 표기 재현: 소수 3자리, 후행 0 제거."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def transform(src, dst, specs):
    """specs: list[(column, anchor, lam)] — 순서 무관, 그룹별 독립 적용."""
    with open(src, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    hdr, body = rows[0], rows[1:]
    col_idx = {name: hdr.index(name) for name, _, _ in specs}
    stats = {}
    for name, anchor, lam in specs:
        j = col_idx[name]
        C = CAP[name]
        n_chg = 0
        deltas = []
        for r in body:
            orig = r[j]
            cf = float(orig) / C
            new_cf = anchor + lam * (cf - anchor) if cf > anchor else cf
            new_cf = min(max(new_cf, 0.0), 1.0)
            if abs(new_cf - cf) > EPS:
                n_chg += 1
                deltas.append(new_cf - cf)
                r[j] = fmt(new_cf * C)
        d = np.array(deltas) if deltas else np.zeros(1)
        stats[name] = dict(anchor=anchor, lam=lam, n_chg=n_chg,
                           d_mean=float(d.mean()), d_max=float(d.max()),
                           d_all=float(d.sum() / len(body)))
    with open(dst, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows([hdr] + body)
    return stats


def parse_spec(s):
    """'kpx_group_3:0.70:1.20' → ('kpx_group_3', 0.70, 1.20)"""
    name, a, lam = s.split(":")
    if name not in CAP:
        raise SystemExit(f"ERROR: unknown group '{name}' (allowed: {list(CAP)})")
    a, lam = float(a), float(lam)
    if lam <= 0.0:
        raise SystemExit(f"ERROR: λ must be > 0 (got {lam}) — 단조성 보존 조건")
    return (name, a, lam)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--spec", action="append", required=True,
                    help="col:anchor:lam (예: kpx_group_2:0.80:1.20). 반복 사용 가능.")
    args = ap.parse_args()
    specs = [parse_spec(s) for s in args.spec]
    st = transform(args.src, args.dst, specs)
    md5 = hashlib.md5(open(args.dst, "rb").read()).hexdigest()
    for name, s in st.items():
        print(f"{name}: a={s['anchor']} λ={s['lam']}  n_chg={s['n_chg']}  "
              f"d_mean={s['d_mean']:.5f}  d_max={s['d_max']:.5f}  d_all={s['d_all']:.6f}")
    print(f"md5={md5}")
