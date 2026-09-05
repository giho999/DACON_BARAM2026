#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-RETROHARNESS — build_truthset: 정답지 구성 + md5 무결성 + transform 재현성 판정
=====================================================================================
정답지 = LB 기록 확인 + 파일 존재분만. 라벨 = LB − 0.65466 (coordBA 기준).
transform 역분석: base(coordBA) 대비 각 후보의 (그룹별) 값 차이 패턴으로
후처리 transform을 복원 — 2024 홀드아웃 base 예측에 재적용 가능 여부를 판정.
"""
import csv, hashlib, os, json
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(PROJ, "submission_A_m1v2_coordBA.csv")
CAPS = [21600.0, 21600.0, 21000.0]
BASE_LB = 0.65466

# 파일 → LB (기록 확인분만)
CANDIDATES = [
    ("submission_A_m1v2_coordBA.csv", 0.65466),
    ("work_a_package/submission_coordBA_g3tail_l120.csv", 0.65416),
    ("submission_A_m1v2.csv", 0.65403),
    ("submission_A_m1v3.csv", 0.65377),
    ("submission_m1_bandcorrected.csv", 0.65293),
    ("submission_ficr_w1_v7_cnn(0.65183).csv", 0.65184),
    ("submission_A_g3shift03.csv", 0.65114),
    ("submission_A_avgblend.csv", 0.64853),
    ("submission_dm.csv", 0.63421),
]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def read_vals(path):
    rows = list(csv.reader(open(path, encoding="utf-8-sig")))
    hdr, body = rows[0], rows[1:]
    assert hdr == ["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"], hdr
    return np.array([[float(r[2 + g]) for g in range(3)] for r in body])


def analyze_transform(base_v, cand_v):
    """그룹별 차이 패턴 → transform 타입 판정."""
    out = {}
    for g in range(3):
        delta = (cand_v[:, g] - base_v[:, g]).round(6)
        nz = delta[delta != 0]
        if len(nz) == 0:
            out[f"G{g+1}"] = {"type": "unchanged"}
            continue
        uniq = np.unique(nz)
        if len(uniq) <= 5 and np.allclose(uniq, uniq[0]):
            out[f"G{g+1}"] = {"type": "const_add", "val": float(uniq[0]), "n": int(len(nz))}
        else:
            cf = base_v[:, g] / CAPS[g]
            hit_hi = nz.max() > 0
            hit_lo = nz.min() < 0
            out[f"G{g+1}"] = {"type": "conditional_band" if (hit_hi and hit_lo) else "mixed",
                              "n": int(len(nz)), "min": float(nz.min()), "max": float(nz.max()),
                              "n_uniq": int(len(uniq))}
    return out


def main():
    base_v = read_vals(BASE)
    truth = []
    for rel, lb in CANDIDATES:
        path = os.path.join(PROJ, rel)
        if not os.path.exists(path):
            print(f"SKIP(부재) {rel} — 저장소에 파일 없음, 정답지에서 제외")
            continue
        v = read_vals(path)
        t = analyze_transform(base_v, v)
        # 재현성: 전 그룹 'unchanged'는 base 자체(라벨 0)로 무의미 — 제외
        if all(x["type"] == "unchanged" for x in t.values()):
            print(f"SKIP(무변화) {rel} — coordBA와 동일, 제외")
            continue
        truth.append(dict(file=rel, lb=lb, label=lb - BASE_LB, md5=md5(path),
                          transform=t))
        print(f"OK {rel:50s} LB={lb:.5f} Δ={lb-BASE_LB:+.5f} md5={md5(path)[:12]}")
        for g, info in t.items():
            print(f"    {g}: {info}")
    out = dict(base=os.path.basename(BASE), base_lb=BASE_LB, base_md5=md5(BASE),
               n=len(truth), candidates=truth)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "truthset.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n정답지 n={len(truth)} · 저장: truthset.json")
    return 0


if __name__ == "__main__":
    main()
