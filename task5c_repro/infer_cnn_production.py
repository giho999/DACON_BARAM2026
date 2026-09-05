"""
infer_cnn_production.py — TASK5-C 추론 단계: train_cnn_production.py 체크포인트 로드 -> 실제 test
격자(data/ldaps_test.csv, data/gfs_test.csv)에 예측 -> submission_ficr_w1_v7.csv와 w=0.20 블렌드 ->
최종 CSV 저장. 학습과 완전히 분리된 별도 실행(재현성 검증용).
"""
import os, sys, time, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cnn_gate"))
import numpy as np
import pandas as pd
import torch

from cnn_common import SpatialCNN, predict_cnn, CAPS, W_CNN, blend_with_fallback

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
LDAPS_TEST_CSV = os.path.join(PROJ, "data", "ldaps_test.csv")
GFS_TEST_CSV = os.path.join(PROJ, "data", "gfs_test.csv")
TEST_CACHE = os.path.join(HERE, "grid_tensors_cache_test.pkl")
CKPT = os.path.join(HERE, "cnn_production_s42.pt")
V7_SUB = os.path.join(PROJ, "submission_ficr_w1_v7.csv")
OUT = os.path.join(HERE, "submission_ficr_w1_v7_cnn_repro.csv")


def load_test_grids():
    if os.path.exists(TEST_CACHE):
        with open(TEST_CACHE, "rb") as f:
            return pickle.load(f)
    ldaps = pd.read_csv(LDAPS_TEST_CSV)
    ldaps["forecast_kst_dtm"] = pd.to_datetime(ldaps["forecast_kst_dtm"])
    ldaps_vars = sorted([c for c in ldaps.columns
                         if c not in ("forecast_kst_dtm", "data_available_kst_dtm",
                                      "grid_id", "latitude", "longitude")])
    T_list = sorted(ldaps["forecast_kst_dtm"].unique())
    n_time = len(T_list)
    time_idx = np.searchsorted(T_list, ldaps["forecast_kst_dtm"].values)
    grid_idx = ldaps["grid_id"].values.astype(int) - 1
    ldaps_grid = np.full((n_time, 16, len(ldaps_vars)), np.nan, dtype=np.float32)
    ldaps_grid[time_idx, grid_idx] = ldaps[ldaps_vars].values.astype(np.float32)
    ldaps_grid = ldaps_grid.reshape(n_time, 4, 4, len(ldaps_vars))

    gfs = pd.read_csv(GFS_TEST_CSV)
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    gfs_vars = sorted([c for c in gfs.columns
                       if c not in ("forecast_kst_dtm", "data_available_kst_dtm",
                                    "grid_id", "latitude", "longitude")])
    T_list_g = sorted(gfs["forecast_kst_dtm"].unique())
    n_time_g = len(T_list_g)
    time_idx_g = np.searchsorted(T_list_g, gfs["forecast_kst_dtm"].values)
    grid_idx_g = gfs["grid_id"].values.astype(int) - 1
    gfs_grid = np.full((n_time_g, 9, len(gfs_vars)), np.nan, dtype=np.float32)
    gfs_grid[time_idx_g, grid_idx_g] = gfs[gfs_vars].values.astype(np.float32)
    gfs_grid = gfs_grid.reshape(n_time_g, 3, 3, len(gfs_vars))

    cache = {"ldaps_grid": ldaps_grid, "gfs_grid": gfs_grid,
             "ldaps_times": np.array(T_list), "gfs_times": np.array(T_list_g)}
    with open(TEST_CACHE, "wb") as f:
        pickle.dump(cache, f)
    return cache


def match_times(times, grid_times, grid_data):
    idx = np.searchsorted(grid_times, times)
    n = len(grid_times)
    idx_c = np.clip(idx, 0, n - 1)
    valid = (idx < n) & (grid_times[idx_c] == times)
    return grid_data[idx_c[valid]], valid


def main():
    t0 = time.time()
    assert os.path.exists(CKPT), f"체크포인트 없음: {CKPT} — train_cnn_production.py 먼저 실행 필요"
    from cnn_common import DEVICE
    ck = torch.load(CKPT, map_location=DEVICE)
    model = SpatialCNN(ldaps_ch=ck["ldaps_ch"], gfs_ch=ck["gfs_ch"]).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    ns = {"lmu": ck["lmu"], "lsd": ck["lsd"], "gmu": ck["gmu"], "gsd": ck["gsd"]}
    print(f"체크포인트 로드 완료(seed={ck['seed']})", flush=True)

    te = pd.read_csv(os.path.join(PROJ, "merged_test_v9.csv"), usecols=["forecast_kst_dtm"])
    te["forecast_kst_dtm"] = pd.to_datetime(te["forecast_kst_dtm"])
    te = te.sort_values("forecast_kst_dtm").reset_index(drop=True)

    gc_ = load_test_grids()
    ldaps_m, lvalid = match_times(te["forecast_kst_dtm"].values, gc_["ldaps_times"], gc_["ldaps_grid"])
    gfs_m, gvalid = match_times(te["forecast_kst_dtm"].values, gc_["gfs_times"], gc_["gfs_grid"])
    valid = lvalid & gvalid
    idx_l = np.searchsorted(gc_["ldaps_times"], te["forecast_kst_dtm"].values)
    idx_g = np.searchsorted(gc_["gfs_times"], te["forecast_kst_dtm"].values)
    n_l, n_g = len(gc_["ldaps_times"]), len(gc_["gfs_times"])
    ldaps_valid = gc_["ldaps_grid"][np.clip(idx_l, 0, n_l - 1)[valid]]
    gfs_valid = gc_["gfs_grid"][np.clip(idx_g, 0, n_g - 1)[valid]]
    print(f"test 격자매칭: {valid.sum()}/{len(te)} ({valid.mean():.1%})", flush=True)

    cnn_raw = np.clip(predict_cnn(model, ldaps_valid, gfs_valid, ck["ldaps_med"], ck["gfs_med"], ns) * CAPS, 0, CAPS)

    v7 = pd.read_csv(V7_SUB)
    v7["forecast_kst_dtm"] = pd.to_datetime(v7["forecast_kst_dtm"])
    v7 = v7.sort_values("forecast_kst_dtm").reset_index(drop=True)
    assert (v7["forecast_kst_dtm"].values == te["forecast_kst_dtm"].values).all(), "시간 정렬 불일치"

    v7_p = v7[["kpx_group_1", "kpx_group_2", "kpx_group_3"]].values.astype(np.float64)
    blended = blend_with_fallback(v7_p, cnn_raw, valid, W_CNN)
    out = v7.copy()
    out[["kpx_group_1", "kpx_group_2", "kpx_group_3"]] = blended
    out.to_csv(OUT, index=False)

    arr = blended
    print(f"\n저장: {OUT}  shape={out.shape}")
    print(f"sanity: 결측={int(np.isnan(arr).sum())} neg={int((arr<0).sum())} over_cap={int((arr>CAPS).sum())}")
    print(f"그룹평균: {np.round(arr.mean(0),1).tolist()}  (총 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
