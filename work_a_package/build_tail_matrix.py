#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-G12TAIL 전량 매트릭스 빌더 — λ∈{1.00,1.20,1.35} × 3그룹 = 27종
=========================================================================
- 앵커: G3=0.70, G2=0.80, G1=0.80 (고정)
- 파일명: submission_tail_g3l{λ}_g2l{λ}_g1l{λ}.csv (λ=1.00 그룹은 이름에서 생략)
  예: g3=1.20,g2=1.20,g1=1.00 → submission_tail_g3l120_g2l120.csv
  예: 전부 1.00 → submission_tail.csv (널테스트, base와 md5 일치해야 함)
- 변환: make_tail_probe.transform (검증된 "변경 행만 재포맷" 방식)
- 출력: JSON(결과 전체) + stdout 요약
"""
import sys, os, csv, json, hashlib, itertools
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tail_probe import transform

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "submission_A_m1v2_coordBA.csv")
SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sample_submission.csv")
OUTDIR = os.path.dirname(os.path.abspath(__file__))

CAPS = {"kpx_group_1": 21600.0, "kpx_group_2": 21600.0, "kpx_group_3": 21000.0}
ANCHORS = {"kpx_group_3": 0.70, "kpx_group_2": 0.80, "kpx_group_1": 0.80}
GROUPS = ["kpx_group_3", "kpx_group_2", "kpx_group_1"]  # 이름 규칙 순서: g3→g2→g1
LAMS = [1.00, 1.20, 1.35]


def fname_for(specs):
    """specs: [(col, anchor, lam), ...] → 파일명 (λ=1.00 그룹 생략)"""
    parts = []
    for col in GROUPS:
        lam = next((s[2] for s in specs if s[0] == col), 1.00)
        if lam != 1.00:
            parts.append(f"g{col[-1]}l{int(lam*100):03d}")
    return "submission_tail.csv" if not parts else "submission_tail_" + "_".join(parts) + ".csv"


def parse_groups(fname):
    """파일명 → {col: lam} (λ=1.00은 부재로 간주)"""
    out = {c: 1.00 for c in GROUPS}
    base = fname.replace("submission_tail_", "").replace("submission_tail.csv", "").replace(".csv", "")
    for tok in base.split("_"):
        if not tok:
            continue
        g = "kpx_group_" + tok[1]
        lam = int(tok[3:]) / 100.0  # 토큰 형식 g{l}{λ}, 예: 'g2l120' → tok[3:]='120'
        out[g] = lam
    return out


def md5_of(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    # base 읽기
    rows = list(csv.reader(open(BASE, encoding="utf-8-sig")))
    hdr, base_body = rows[0], rows[1:]
    assert len(base_body) == 8760, f"base nrows {len(base_body)} != 8760"
    col_idx = {c: hdr.index(c) for c in GROUPS}
    base_cf = {c: np.array([float(r[col_idx[c]]) / CAPS[c] for r in base_body]) for c in GROUPS}
    sample_rows = list(csv.reader(open(SAMPLE, encoding="utf-8-sig")))
    s_hdr, s_body = sample_rows[0], sample_rows[1:]
    base_md5 = md5_of(BASE)

    results = []
    all_pass = True
    combos = list(itertools.product(LAMS, repeat=3))  # (g3, g2, g1)
    for (l3, l2, l1) in combos:
        specs = [
            ("kpx_group_3", ANCHORS["kpx_group_3"], l3),
            ("kpx_group_2", ANCHORS["kpx_group_2"], l2),
            ("kpx_group_1", ANCHORS["kpx_group_1"], l1),
        ]
        dst = os.path.join(OUTDIR, fname_for(specs))
        st = transform(BASE, dst, specs)
        md5 = md5_of(dst)
        sha = sha256_of(dst)

        # 가드 재검증 (독립 재계산)
        rows = list(csv.reader(open(dst, encoding="utf-8-sig")))
        hdr2, body = rows[0], rows[1:]
        checks = {}
        # L1: 미접촉 = λ=1.00 그룹 원문 문자열 100% / 전체 id·dtm·헤더·행수
        checks["nrows"] = len(body) == 8760
        checks["hdr"] = hdr2 == hdr
        checks["id"] = all(x[0] == y[0] for x, y in zip(body, base_body))
        checks["dtm"] = all(x[1] == y[1] for x, y in zip(body, base_body))
        checks["sample"] = (hdr2 == s_hdr) and all(x[0] == y[0] for x, y in zip(body, s_body))
        l1 = {}
        for c in GROUPS:
            lam = parse_groups(os.path.basename(dst))[c]
            same = all(x[col_idx[c]] == y[col_idx[c]] for x, y in zip(body, base_body))
            if lam == 1.00:
                l1[c] = same  # 미접촉이어야 참
            else:
                l1[c] = (not same) and all(x[col_idx[c]] != "" for x in body)  # 접촉 정상
        # L2: 변경행 / Δ통계 (파일에서 직접 재계산)
        l2 = {}
        for c in GROUPS:
            lam = parse_groups(os.path.basename(dst))[c]
            v = np.array([float(x[col_idx[c]]) for x in body])
            delta = v / CAPS[c] - base_cf[c]
            chg = np.abs(delta) > 1e-9
            n_chg = int(chg.sum())
            below = int((chg & (base_cf[c] <= ANCHORS[c])).sum())
            d = delta[chg]
            l2[c] = dict(lam=lam, n_chg=n_chg, d_mean=float(d.mean()) if d.size else 0.0,
                         d_max=float(d.max()) if d.size else 0.0,
                         d_all=float(delta.mean()), below_anchor=below)
        # L3: 물리/단조
        l3 = {}
        for c in GROUPS:
            v = np.array([float(x[col_idx[c]]) for x in body])
            rho = spearmanr(base_cf[c], v / CAPS[c]).statistic
            l3[c] = dict(nan=int(np.isnan(v).sum()), neg=int((v < 0).sum()),
                         overcap=int((v > CAPS[c]).sum()), rho=round(float(rho), 6))

        null_ok = (md5 == base_md5)
        if os.path.basename(dst) == "submission_tail.csv":
            checks["null"] = null_ok
        res = dict(combo=(l3_, l2_, l1_) if False else (l3, l2, l1),
                   fname=os.path.basename(dst), md5=md5, sha256=sha,
                   l1=l1, l2=l2, l3=l3, checks=checks)
        results.append(res)
        ok = (all(l1.values()) and
              all(l2[c]["below_anchor"] == 0 and
                  l2[c]["n_chg"] in (0, 852, 1129, 1193) for c in GROUPS) and
              all(l3[c]["nan"] == 0 and l3[c]["neg"] == 0 and l3[c]["overcap"] == 0 and
                  l3[c]["rho"] == 1.0 for c in GROUPS) and
              all(checks.values()))
        if not ok:
            all_pass = False
            print(f"[FAIL] {os.path.basename(dst)}  l1={l1} l2={ {k: v['n_chg'] for k, v in l2.items()} } "
                  f"l3rho={ {k: v['rho'] for k, v in l3.items()} } checks={checks}")
        else:
            print(f"[ OK ] {os.path.basename(dst):52s} "
                  f"chg=G3:{l2['kpx_group_3']['n_chg']:4d} G2:{l2['kpx_group_2']['n_chg']:4d} G1:{l2['kpx_group_1']['n_chg']:4d} "
                  f"md5={md5[:12]}")

    out = dict(base=os.path.basename(BASE), base_md5=base_md5,
               n_files=len(results), all_guard_pass=all_pass, results=results)
    jpath = os.path.join(OUTDIR, "tail_matrix_results.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print()
    print(f"총 {len(results)}개 · ALL GUARDS: {'PASS' if all_pass else 'FAIL'} · JSON: {jpath}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
