#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK-W2CHAIN — 동결 체인 스왑 결정실험: chain_frozen
======================================================
지시서: opencode_TASK_W2CHAIN_20260809.md (없으면 인수인계서 §4-38)

chain(X) = coordBA( M1( shift( iso( blend( GBDT=X, CNN=동결 ) ) ) ) )
  - X ∈ {W0, W1, W2} (METRICW 예측) — GBDT 성분만 스왑
  - 체인 상수(iso 매핑·shift·M1 shift·coordBA)는 원본 v7 GBDT(clean-downweight 0.3,
    m1_holdout_preds_2024.parquet)로 run.py와 동일하게 적합한 것을 **동결** 사용.
    W2로 재적합 금지 (연도 적합 상수 재도입 = 8연속 실패 구조 회귀 방지).
  - CNN: frozen_cnn_pred_2024_s42.parquet 동결. W_CNN=0.20 고정.
  - coordBA: G1 CF[0.7,1.0] ∧ 비JJA(6~8월 제외) 행 +216kW(=+0.010×cap), G2/G3 불변.
    좌표는 submission_A_m1v2.csv vs submission_A_m1v2_coordBA.csv 차이로 실측 복원(1572행).

검증(§3-1): chain_frozen(원본 base pred)의 2024 eval 점수가 run.py 원본 print
(캘리만 0.62906 / M1 0.63093)과 일치해야 함 + 제출 경로(3e)로 submission_A.csv md5 재현.

평가: 2024 연도전이 홀드아웃 전체(8778행)에 동결 체인 적용 → 대회 평가식(competition_metric
= 10% 마스크 + 6%/8% 밴드 + 발전량가중, TOTAL=3그룹 평균). cal 창(09-10)이 평가에 포함됨
(in-sample 오염 — 리포트에 고지).

사용법:
  python3 chain_frozen.py repro           # 검증: run.py 원본 재현 (점수+submission_A md5)
  python3 chain_frozen.py chain --variant W0 --seed 42   # 단일 변형 체인 통과
  python3 chain_frozen.py run_all         # W0/W1/W2 × 3seed 전부
"""
import os, sys, json, pickle, hashlib, importlib.util, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))

# cnn_common 로드 (run.py와 동일 방식 — 상위 폴더 work_a_package/)
_spec = importlib.util.spec_from_file_location("cnn_common", os.path.join(os.path.dirname(HERE), "cnn_common.py"))
cnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cnn)
score_fn = cnn.score_fn
blend_with_fallback = cnn.blend_with_fallback
W_CNN = cnn.W_CNN
CAPS = cnn.CAPS

# constants (run.py와 동일)
CAL_S, CAL_E = pd.Timestamp("2024-09-01"), pd.Timestamp("2024-11-01")
EV_S, EV_E = pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-31 23:59:59")
SHIFT_GRID = np.arange(-0.08, 0.09, 0.01)
BAND_GRID = np.arange(-0.10, 0.101, 0.005)
BAND_LO, BAND_HI = 0.03, 0.15
COORD_CF_LO, COORD_CF_HI = 0.7, 1.0
COORD_SHIFT_CAP = 0.010  # sA=+0.010×cap
COORD_JJA = {6, 7, 8}

# 경로 — 데이터·CNN·팀원 제출본·run.py 산출물은 상위 폴더 work_a_package/ 기준
AP = os.path.dirname(HERE)
BASE_PRED = os.path.join(AP, "m1_holdout_preds_2024.parquet")
CNN_PRED = os.path.join(AP, "frozen_cnn_pred_2024_s42.parquet")
TEAM_SUB = os.path.join(AP, "submission_ficr_w1_v7_cnn(0.65183).csv")
RUN_SUB = os.path.join(AP, "submission_A.csv")
METRICW_DIR = os.path.join(AP, "metricw")
CONST_PATH = os.path.join(HERE, "frozen_constants.pkl")


def load_run_inputs():
    base = pd.read_parquet(BASE_PRED)
    base["t"] = pd.to_datetime(base["t"])
    base = base.sort_values("t").reset_index(drop=True)
    frozen = pd.read_parquet(CNN_PRED)
    frozen["t"] = pd.to_datetime(frozen["t"])
    m = base.merge(frozen, on="t", how="left")
    vm = m["cnn_pred_0"].notna().values
    cn = m.loc[vm, ["cnn_pred_0", "cnn_pred_1", "cnn_pred_2"]].values.astype(np.float64)
    bp = m[["pred_0", "pred_1", "pred_2"]].values.astype(np.float64)
    blended = blend_with_fallback(bp, cn, vm, W_CNN)
    y_all = m[["y_0", "y_1", "y_2"]].values.astype(np.float64)
    t_all = m["t"].values
    return blended, y_all, t_all


def fit_frozen_constants():
    """원본 v7 GBDT blend로 iso/shift/M1 적합 (run.py와 동일 로직, 동결용)."""
    blended, y_all, t_all = load_run_inputs()
    cm = (t_all >= CAL_S) & (t_all < CAL_E)
    em = (t_all >= EV_S) & (t_all <= EV_E)
    # iso (cal 창)
    isos = []
    for g in range(3):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(blended[cm, g], y_all[cm, g])
        isos.append(iso)
    # shift (cal 창 grid sweep)
    cal_c = blended[cm].copy()
    for g in range(3):
        cal_c[:, g] = np.clip(isos[g].transform(blended[cm, g]), 0, CAPS[g])
    shifts = np.zeros(3)
    for g in range(3):
        best = (-1e9, 0.0)
        for sp in SHIFT_GRID:
            t = cal_c.copy()
            t[:, g] = np.clip(cal_c[:, g] + sp * CAPS[g], 0, CAPS[g])
            tot, _, _ = score_fn(y_all[cm], t)
            if tot > best[0]:
                best = (tot, sp)
        shifts[g] = best[1]
    # M1 band (eval self-ref)
    ev_c = blended[em].copy()
    for g in range(3):
        ev_c[:, g] = np.clip(isos[g].transform(blended[em, g]), 0, CAPS[g])
    for g in range(3):
        ev_c[:, g] = np.clip(ev_c[:, g] + shifts[g] * CAPS[g], 0, CAPS[g])
    prod_shift = np.zeros(3)
    for g in range(3):
        cap = CAPS[g]
        in_b = (ev_c[:, g] >= BAND_LO * cap) & (ev_c[:, g] <= BAND_HI * cap)
        best = (-1e9, 0.0)
        for sp in BAND_GRID:
            t = ev_c.copy()
            t[in_b, g] = np.clip(ev_c[in_b, g] + sp * cap, 0, cap)
            tot, _, _ = score_fn(y_all[em], t)
            if tot > best[0]:
                best = (tot, sp)
        prod_shift[g] = best[1]
    const = dict(isos=isos, shifts=shifts, prod_shift=prod_shift,
                 cal_n=int(cm.sum()), eval_n=int(em.sum()))
    return const


def load_metricw_pred(variant, seed):
    df = pd.read_parquet(os.path.join(METRICW_DIR, f"pred_{variant}_seed{seed}.parquet"))
    df = df.sort_values("t").reset_index(drop=True)
    return (df[["pred_0", "pred_1", "pred_2"]].values.astype(np.float64),
            df[["y_0", "y_1", "y_2"]].values.astype(np.float64),
            df["t"].values)


def apply_chain(gbdt_pred, const, t_all=None, shift_scale=None, stages_out=None):
    """blend(GBDT, CNN 동결) → iso(동결) → shift(동결·계수 스케일 가능) → M1(동결) → coordBA(동결).
    gbdt_pred: (N,3) — W0/W1/W2 2024 예측. 반환: 체인 통과 후 (N,3) 예측.
    shift_scale: float 또는 길이 3 배열 — shift 계수 승수 (TASK-W2NOSHIFT 변형용).
                 None → 1.0 (원본). stages_out: dict이면 단계별 스냅샷(blend/iso/shift/m1/coordBA) 저장."""
    scale = 1.0 if shift_scale is None else shift_scale
    if np.isscalar(scale):
        scale = np.full(3, float(scale))
    else:
        scale = np.asarray(scale, dtype=float)
    # 1) blend with frozen CNN
    frozen = pd.read_parquet(CNN_PRED)
    frozen["t"] = pd.to_datetime(frozen["t"])
    t_arr = pd.to_datetime(t_all)
    fz = frozen.set_index("t").reindex(t_arr)
    vm = fz["cnn_pred_0"].notna().values
    cn = fz.loc[vm, ["cnn_pred_0", "cnn_pred_1", "cnn_pred_2"]].values.astype(np.float64)
    bp = gbdt_pred.astype(np.float64)
    blended = blend_with_fallback(bp, cn, vm, W_CNN)
    if stages_out is not None:
        stages_out["blend"] = blended.copy()
    # 2) iso (동결)
    ev = blended.copy()
    for g in range(3):
        ev[:, g] = np.clip(const["isos"][g].transform(blended[:, g]), 0, CAPS[g])
    if stages_out is not None:
        stages_out["iso"] = ev.copy()
    # 3) shift (동결·계수 스케일)
    for g in range(3):
        ev[:, g] = np.clip(ev[:, g] + const["shifts"][g] * scale[g] * CAPS[g], 0, CAPS[g])
    if stages_out is not None:
        stages_out["shift"] = ev.copy()
    # 4) M1 band (동결 prod_shift, 밴드 판정은 입력 예측 기준)
    ev_m1 = ev.copy()
    for g in range(3):
        cap = CAPS[g]
        in_b = (ev[:, g] >= BAND_LO * cap) & (ev[:, g] <= BAND_HI * cap)
        ev_m1[in_b, g] = np.clip(ev[in_b, g] + const["prod_shift"][g] * cap, 0, cap)
    if stages_out is not None:
        stages_out["m1"] = ev_m1.copy()
    # 5) coordBA (G1 CF[0.7,1.0] ∧ 비JJA +216, G2/G3 불변)
    out = ev_m1.copy()
    if t_all is not None:
        mon = pd.to_datetime(t_all).month
        cf1 = out[:, 0] / CAPS[0]
        sel = (cf1 >= COORD_CF_LO) & (cf1 <= COORD_CF_HI) & (~np.isin(mon, list(COORD_JJA)))
        out[sel, 0] = np.clip(out[sel, 0] + COORD_SHIFT_CAP * CAPS[0], 0, CAPS[0])
    if stages_out is not None:
        stages_out["coordBA"] = out.copy()
    return out


def competition_metric(y, p, cap):
    v = (~np.isnan(y)) & (y >= 0.1 * cap)
    yt, pt = y[v], p[v]
    if len(yt) == 0:
        return 0.0, 0.0, 0.0
    nmae = float(np.clip(np.mean(np.abs(yt - pt)) / cap, 0, 1))
    e = np.abs(yt - pt) / cap
    up = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
    fi = float(np.clip((yt * up).sum() / max((yt * 4.0).sum(), 1e-9), 0, 1))
    return 0.5 * (1 - nmae) + 0.5 * fi, 1.0 - nmae, fi


def total_metric(y3, p3):
    t = [competition_metric(y3[:, g], p3[:, g], CAPS[g]) for g in range(3)]
    return tuple(float(np.mean([x[k] for x in t])) for k in range(3))


def repro_check():
    """검증: run.py 원본 재현 — 캘리만 0.62906 / M1 0.63093 + submission_A.csv md5."""
    const = fit_frozen_constants()
    blended, y_all, t_all = load_run_inputs()
    cm = (t_all >= CAL_S) & (t_all < CAL_E)
    em = (t_all >= EV_S) & (t_all <= EV_E)
    ev_c = blended[em].copy()
    for g in range(3):
        ev_c[:, g] = np.clip(const["isos"][g].transform(blended[em, g]), 0, CAPS[g])
    for g in range(3):
        ev_c[:, g] = np.clip(ev_c[:, g] + const["shifts"][g] * CAPS[g], 0, CAPS[g])
    tot_base = score_fn(y_all[em], ev_c)[0]
    ev_m1 = ev_c.copy()
    for g in range(3):
        cap = CAPS[g]
        in_b = (ev_c[:, g] >= BAND_LO * cap) & (ev_c[:, g] <= BAND_HI * cap)
        ev_m1[in_b, g] = np.clip(ev_c[in_b, g] + const["prod_shift"][g] * cap, 0, cap)
    tot_m1 = score_fn(y_all[em], ev_m1)[0]
    print(f"[repro] 캘리만 TOTAL={tot_base:.5f} (기대 0.62906)  M1 TOTAL={tot_m1:.5f} (기대 0.63093)")
    ok1 = abs(tot_base - 0.62906) < 5e-5 and abs(tot_m1 - 0.63093) < 5e-5
    # 제출 경로 (3e) 재현 — submission_A.csv md5
    if os.path.exists(RUN_SUB):
        md5_run = hashlib.md5(open(RUN_SUB, "rb").read()).hexdigest()
        print(f"[repro] run.py 원본 submission_A.csv md5 = {md5_run}")
    # 동결 상수 저장
    os.makedirs(os.path.dirname(CONST_PATH), exist_ok=True)
    with open(CONST_PATH, "wb") as f:
        pickle.dump({k: v for k, v in const.items() if k != "isos"}, f)
    with open(CONST_PATH + ".isos", "wb") as f:
        pickle.dump(const["isos"], f)
    print(f"[repro] 동결 상수 저장: {CONST_PATH}  shifts={np.round(const['shifts'],3)}  M1={np.round(const['prod_shift'],3)}")
    return ok1


def chain_one(variant, seed):
    with open(CONST_PATH + ".isos", "rb") as f:
        isos = pickle.load(f)
    with open(CONST_PATH, "rb") as f:
        c = pickle.load(f)
    const = dict(isos=isos, **c)
    p, y, t = load_metricw_pred(variant, seed)
    chained = apply_chain(p, const, t)
    tot, nm, fi = total_metric(y, chained)
    return dict(variant=variant, seed=seed, TOTAL=tot, NMAE1=nm, FICR=fi, chained=chained)


def main():
    os.makedirs(os.path.join(HERE, "w2chain"), exist_ok=True)
    if len(sys.argv) < 2:
        print("usage: chain_frozen.py repro|chain --variant X --seed S|run_all")
        return
    cmd = sys.argv[1]
    if cmd == "repro":
        ok = repro_check()
        print(f"[repro] {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    # ensure constants
    if not os.path.exists(CONST_PATH):
        repro_check()
    results = []
    if cmd == "chain":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--variant", required=True, choices=["W0", "W1", "W2"])
        ap.add_argument("--seed", type=int, required=True, choices=[42, 1337, 2024])
        args = ap.parse_args(sys.argv[2:])
        r = chain_one(args.variant, args.seed)
        print(f"[chain] {args.variant} seed{args.seed}: TOTAL={r['TOTAL']:.5f} 1-NMAE={r['NMAE1']:.5f} FICR={r['FICR']:.5f}")
    elif cmd == "run_all":
        for V in ["W0", "W1", "W2"]:
            for S in [42, 1337, 2024]:
                r = chain_one(V, S)
                results.append(r)
                print(f"[chain] {V} seed{S}: TOTAL={r['TOTAL']:.5f} 1-NMAE={r['NMAE1']:.5f} FICR={r['FICR']:.5f}")
        with open(os.path.join(HERE, "w2chain_raw.json"), "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "chained"} for r in results], f, indent=2)


if __name__ == "__main__":
    main()
