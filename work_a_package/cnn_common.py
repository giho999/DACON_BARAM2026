"""
cnn_gate/cnn_common.py — TASK1(게이트 역검증) 공용 유틸.
deepseq_pipeline/.venv(torch cuda) 안에서 실행.

SpatialCNN 아키텍처는 spatial_cnn_v7_top.py / v7_cnn_submission.py(팀원 원본 텍스트 복원본)와
100% 동일(dual-branch, LDAPS 4x4 + GFS 3x3). 변경한 것은 딱 두 가지 프로토콜 편향 제거뿐:
  (a) CNN 조기종료 검증셋을 cal/eval 창이 아니라 train 구간(tm) 내부 시간분할로 교체
      (train_cnn_holdout의 val_frac 분할 = tm의 시간순 마지막 val_frac).
  (b) 캘리 적합(cal)과 채점(eval)을 분리 — canonical calibrate_total(cal_p, ycal, ev_p) 사용,
      팀원 스크립트처럼 같은 cal창에서 fit+score(in-sample)하지 않음.
"""
import os, time, pickle
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # TASK14: CUDA matmul 결정성 전제조건
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.isotonic import IsotonicRegression

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGETS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CLEAN_COLS = ["clean_1", "clean_2", "clean_3"]
CAPS = np.array([21600.0, 21600.0, 21000.0])
SHIFT_GRID = np.arange(-0.08, 0.09, 0.01)

CNN_EPOCHS = 50
CNN_BATCH = 1024
CNN_LR = 8e-4
CNN_PATIENCE = 8
W_CNN = 0.20  # 고정 — 재탐색 금지(Public 과적합)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/DACON_baram2026
LDAPS_TRAIN_CSV = os.path.join(PROJ, "data", "ldaps_train.csv")
GFS_TRAIN_CSV = os.path.join(PROJ, "data", "gfs_train.csv")
GRID_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_tensors_cache.pkl")


# ──────────────────────────────────────────────
# Scoring / calibration (canonical — deepseq_pipeline/common.py, check_*.py와 동일)
# ──────────────────────────────────────────────
def score_fn(y, p):
    nm, fi = [], []
    for g in range(3):
        v = y[:, g] >= CAPS[g] * 0.1
        if v.sum() == 0:
            nm.append(np.nan); fi.append(np.nan); continue
        e = np.abs(p[v, g] - y[v, g]) / CAPS[g]; nm.append(float(e.mean()))
        up = np.select([e <= 0.06, e <= 0.08], [4.0, 3.0], 0.0)
        fi.append(float((y[v, g] * up).sum() / max((y[v, g] * 4.0).sum(), 1)))
    return 0.5 * (1 - float(np.nanmean(nm))) + 0.5 * float(np.nanmean(fi)), float(np.nanmean(nm)), float(np.nanmean(fi))


def calibrate_total(cal_p, ycal, ev_p):
    """rolling 검증용 캘리(이 세션의 모든 canonical check_*.py와 동일: 3그룹 전부 iso+shift).
    cal에서 적합, eval에서만 채점 — in-sample 금지(팀원 스크립트의 편향(b) 제거)."""
    cal_c, ev_c = cal_p.copy(), ev_p.copy()
    for g in range(3):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(cal_p[:, g], ycal[:, g])
        cal_c[:, g] = np.clip(iso.transform(cal_p[:, g]), 0, CAPS[g])
        ev_c[:, g] = np.clip(iso.transform(ev_p[:, g]), 0, CAPS[g])
    shifts = np.zeros(3)
    for g in range(3):
        best = (-1e9, 0.0)
        for sp in SHIFT_GRID:
            t = cal_c.copy(); t[:, g] = np.clip(cal_c[:, g] + sp * CAPS[g], 0, CAPS[g])
            tot, _, _ = score_fn(ycal, t)
            if tot > best[0]: best = (tot, sp)
        shifts[g] = best[1]
    for g in range(3):
        ev_c[:, g] = np.clip(ev_c[:, g] + shifts[g] * CAPS[g], 0, CAPS[g])
    return ev_c, shifts


# ──────────────────────────────────────────────
# Grid cache (train only — data/ldaps_train.csv, data/gfs_train.csv)
# ──────────────────────────────────────────────
def load_cached_grids():
    if os.path.exists(GRID_CACHE):
        print("  격자 캐시 로드...")
        with open(GRID_CACHE, "rb") as f:
            return pickle.load(f)

    print("  격자 데이터 로드 (초기, 느림)...")
    t0 = time.time()

    ldaps = pd.read_csv(LDAPS_TRAIN_CSV)
    ldaps["forecast_kst_dtm"] = pd.to_datetime(ldaps["forecast_kst_dtm"])
    ldaps_vars = sorted([c for c in ldaps.columns
                         if c not in ("forecast_kst_dtm", "data_available_kst_dtm",
                                      "grid_id", "latitude", "longitude")])
    n_vars = len(ldaps_vars)
    T_list = sorted(ldaps["forecast_kst_dtm"].unique())
    n_time = len(T_list)
    time_idx = np.searchsorted(T_list, ldaps["forecast_kst_dtm"].values)
    grid_idx = ldaps["grid_id"].values.astype(int) - 1
    ldaps_grid = np.full((n_time, 16, n_vars), np.nan, dtype=np.float32)
    ldaps_grid[time_idx, grid_idx] = ldaps[ldaps_vars].values.astype(np.float32)
    ldaps_grid = ldaps_grid.reshape(n_time, 4, 4, n_vars)
    print(f"    LDAPS: {ldaps_grid.shape} ({time.time()-t0:.1f}s)")

    t1 = time.time()
    gfs = pd.read_csv(GFS_TRAIN_CSV)
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    gfs_vars = sorted([c for c in gfs.columns
                       if c not in ("forecast_kst_dtm", "data_available_kst_dtm",
                                    "grid_id", "latitude", "longitude")])
    n_vars_g = len(gfs_vars)
    T_list_g = sorted(gfs["forecast_kst_dtm"].unique())
    n_time_g = len(T_list_g)
    time_idx_g = np.searchsorted(T_list_g, gfs["forecast_kst_dtm"].values)
    grid_idx_g = gfs["grid_id"].values.astype(int) - 1
    gfs_grid = np.full((n_time_g, 9, n_vars_g), np.nan, dtype=np.float32)
    gfs_grid[time_idx_g, grid_idx_g] = gfs[gfs_vars].values.astype(np.float32)
    gfs_grid = gfs_grid.reshape(n_time_g, 3, 3, n_vars_g)
    print(f"    GFS:   {gfs_grid.shape} ({time.time()-t1:.1f}s)")

    cache = {
        "ldaps_grid": ldaps_grid, "gfs_grid": gfs_grid,
        "ldaps_times": np.array(T_list), "gfs_times": np.array(T_list_g),
    }
    with open(GRID_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"    캐시 저장 완료")
    return cache


def match_times(times, grid_times, grid_data):
    idx = np.searchsorted(grid_times, times)
    n = len(grid_times)
    idx_c = np.clip(idx, 0, n - 1)
    valid = (idx < n) & (grid_times[idx_c] == times)
    matched = grid_data[idx[valid]]
    return matched, valid


def get_grid_sequences(times, half_window=3):
    """TASK3 후보(3) 시공간 CNN용: 각 시각 t에 대해 [t-half_window..t+half_window](2*half_window+1 프레임,
    시간간격 1h 정규 격자 확인됨)를 모아 (n_valid, T, H, W, C) 시퀀스 반환. 경계는 가장 가까운 유효
    프레임으로 클리핑(패딩 대신). valid_mask는 t 자체가 양쪽 그리드에 매칭되는지만 기준(중간 프레임 결측은
    클리핑으로 항상 채워짐)."""
    gc_ = load_cached_grids()
    n_l, n_g = len(gc_["ldaps_times"]), len(gc_["gfs_times"])
    idx_l = np.searchsorted(gc_["ldaps_times"], times)
    idx_g = np.searchsorted(gc_["gfs_times"], times)
    idx_l_c, idx_g_c = np.clip(idx_l, 0, n_l - 1), np.clip(idx_g, 0, n_g - 1)
    valid = (idx_l < n_l) & (gc_["ldaps_times"][idx_l_c] == times) & \
            (idx_g < n_g) & (gc_["gfs_times"][idx_g_c] == times)

    base_l = idx_l_c[valid]
    base_g = idx_g_c[valid]
    offsets = np.arange(-half_window, half_window + 1)
    ldaps_seq = np.stack([gc_["ldaps_grid"][np.clip(base_l + o, 0, n_l - 1)] for o in offsets], axis=1)
    gfs_seq = np.stack([gc_["gfs_grid"][np.clip(base_g + o, 0, n_g - 1)] for o in offsets], axis=1)
    return ldaps_seq, gfs_seq, valid


def get_grid_tensors(times):
    """주어진 시간배열(times, np.datetime64)에 대해 (ldaps_tensor, gfs_tensor, valid_mask) 반환.
    LDAPS/GFS 둘 다 매칭된 행만 valid=True."""
    gc_ = load_cached_grids()
    ldaps_m, lvalid = match_times(times, gc_["ldaps_times"], gc_["ldaps_grid"])
    gfs_m, gvalid = match_times(times, gc_["gfs_times"], gc_["gfs_grid"])
    # ldaps_m/gfs_m 이미 valid 행만 뽑혀 있으므로, 공통으로 다시 정렬해야 함
    valid = lvalid & gvalid
    ldaps_full, gfs_full = None, None
    idx_l = np.searchsorted(gc_["ldaps_times"], times)
    idx_g = np.searchsorted(gc_["gfs_times"], times)
    n_l, n_g = len(gc_["ldaps_times"]), len(gc_["gfs_times"])
    idx_l_c, idx_g_c = np.clip(idx_l, 0, n_l - 1), np.clip(idx_g, 0, n_g - 1)
    ldaps_full = gc_["ldaps_grid"][idx_l_c[valid]]
    gfs_full = gc_["gfs_grid"][idx_g_c[valid]]
    return ldaps_full, gfs_full, valid


# ──────────────────────────────────────────────
# Spatial CNN (dual branch: LDAPS 4×4 + GFS 3×3) — 팀원 원본과 100% 동일 아키텍처
# ──────────────────────────────────────────────
class GlobalAvgPool2d(nn.Module):
    """TASK14: AdaptiveAvgPool2d((1,1))의 결정적 대체. 출력크기(1,1)에서 두 연산은 forward가
    수학적으로 완전히 동일(채널별 공간평균) — 예측값 변화 없음, backward의 CUDA 비결정성만 제거."""
    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True)


class SpatialCNNBranch(nn.Module):
    def __init__(self, in_channels, out_dim=32, ch1=32, ch2=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, ch1, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch1), nn.ReLU(inplace=True),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch2), nn.ReLU(inplace=True),
            GlobalAvgPool2d(),
        )
        self.fc = nn.Linear(ch2, out_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return torch.relu(self.fc(h))


class SpatialCNN(nn.Module):
    """팀원 원본 기본값(ch1=32,ch2=64,hidden=32,out_dim=3)과 100% 동일. TASK3 후보용으로
    용량(ch1/ch2/hidden)과 out_dim(그룹별 단일출력)을 파라미터화."""
    def __init__(self, ldaps_ch=30, gfs_ch=35, hidden=32, ch1=32, ch2=64, out_dim=3):
        super().__init__()
        self.ldaps_branch = SpatialCNNBranch(ldaps_ch, hidden, ch1=ch1, ch2=ch2)
        self.gfs_branch = SpatialCNNBranch(gfs_ch, hidden, ch1=ch1, ch2=ch2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, max(32, hidden)), nn.BatchNorm1d(max(32, hidden)),
            nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(max(32, hidden), out_dim),
        )

    def forward(self, ldaps, gfs):
        x = torch.cat([self.ldaps_branch(ldaps), self.gfs_branch(gfs)], dim=1)
        return self.head(x)


class GridDataset(Dataset):
    def __init__(self, ldaps, gfs, y, w=None):
        self.ldaps = torch.tensor(ldaps, dtype=torch.float32)
        self.gfs = torch.tensor(gfs, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32) if w is not None else None
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        if self.w is None:
            return self.ldaps[i], self.gfs[i], self.y[i]
        return self.ldaps[i], self.gfs[i], self.y[i], self.w[i]


def _nan_to_med(arr, med):
    """채널(마지막 축) 기준 결측 대체. 주의: 팀원 원본(spatial_cnn_v7_top.py/v7_cnn_submission.py)의
    두 버전 모두 채널축이 아닌 축(H축 또는 flatten 후 잘못된 열)을 순회하는 버그가 있었음 — 이 세션에서
    발견·수정(TASK5-C, test 격자에서 실제 NaN 752개 만나 노출됨). train 격자엔 NaN이 전혀 없어
    TASK1~4A 결과에는 영향 없음(버그가 있는 채로도 무해했음 — 실행조차 안 됐던 코드경로)."""
    C = arr.shape[-1]
    flat = arr.reshape(-1, C)
    for c in range(C):
        mask = np.isnan(flat[:, c])
        if mask.any():
            flat[mask, c] = med[c]
    return flat.reshape(arr.shape)


def train_cnn_holdout(ldaps_tm, gfs_tm, y_tm_norm, times_tm, seed, val_frac=0.1,
                       sample_weight_tm=None, model_ctor=None, epochs=None, patience=None):
    """편향(a) 제거판: 조기종료 검증셋을 train 구간(tm) 내부 시간순 마지막 val_frac에서만 분리.
    cal/eval 창은 절대 건드리지 않음(모델이 그 시점을 보지 못함).

    sample_weight_tm: (n,) 또는 (n, out_dim) — TASK3 후보(1) 채점시간 가중용. train loss에만 적용,
      validation loss(조기종료 기준)는 항상 unweighted plain MSE로 고정(다른 후보와 비교 가능하게).
    model_ctor: callable(ldaps_ch, gfs_ch) -> nn.Module. 기본은 팀원 원본 SpatialCNN.
      TASK3 후보(2) 용량 증대, 후보(4) 그룹별 단일출력에 사용.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)  # TASK14: cudnn.deterministic은 커널만 덮음
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)

    epochs = epochs or CNN_EPOCHS
    patience = patience or CNN_PATIENCE

    order = np.argsort(times_tm)
    n = len(order)
    n_val = max(int(n * val_frac), 1)
    fit_idx, val_idx = order[:-n_val], order[-n_val:]

    ldaps_fit, gfs_fit, y_fit = ldaps_tm[fit_idx], gfs_tm[fit_idx], y_tm_norm[fit_idx]
    ldaps_val, gfs_val, y_val = ldaps_tm[val_idx], gfs_tm[val_idx], y_tm_norm[val_idx]
    w_fit = None
    if sample_weight_tm is not None:
        w_fit = sample_weight_tm[fit_idx]
        if w_fit.ndim == 1:
            w_fit = w_fit[:, None]

    ldaps_med = np.nanmedian(ldaps_fit.reshape(-1, ldaps_fit.shape[-1]), axis=0)
    gfs_med = np.nanmedian(gfs_fit.reshape(-1, gfs_fit.shape[-1]), axis=0)
    lt = _nan_to_med(ldaps_fit.copy(), ldaps_med)
    gt = _nan_to_med(gfs_fit.copy(), gfs_med)
    lv = _nan_to_med(ldaps_val.copy(), ldaps_med)
    gv = _nan_to_med(gfs_val.copy(), gfs_med)

    lt_t = np.transpose(lt, (0, 3, 1, 2))
    gt_t = np.transpose(gt, (0, 3, 1, 2))
    lv_t = np.transpose(lv, (0, 3, 1, 2))
    gv_t = np.transpose(gv, (0, 3, 1, 2))

    def norm_grid(a_tr, a_va):
        n_, c, h, w = a_tr.shape
        flat = a_tr.reshape(n_, c, -1)
        mu = flat.mean(axis=(0, 2), keepdims=True).reshape(1, c, 1, 1)
        sd = flat.std(axis=(0, 2), keepdims=True).reshape(1, c, 1, 1) + 1e-6
        return (a_tr - mu) / sd, (a_va - mu) / sd, mu, sd

    ltn, lvn, lmu, lsd = norm_grid(lt_t, lv_t)
    gtn, gvn, gmu, gsd = norm_grid(gt_t, gv_t)

    train_dl = DataLoader(GridDataset(ltn, gtn, y_fit, w_fit), batch_size=CNN_BATCH, shuffle=True,
                           drop_last=True, generator=dl_gen)
    val_dl = DataLoader(GridDataset(lvn, gvn, y_val), batch_size=CNN_BATCH * 2, shuffle=False)

    ctor = model_ctor or (lambda lch, gch: SpatialCNN(ldaps_ch=lch, gfs_ch=gch))
    model = ctor(ltn.shape[1], gtn.shape[1]).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    crit_none = nn.MSELoss(reduction="none")

    best_v, best_s, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for batch in train_dl:
            if w_fit is None:
                lb, gb, yb = batch
                lb, gb, yb = lb.to(DEVICE), gb.to(DEVICE), yb.to(DEVICE)
                loss = crit(model(lb, gb), yb)
            else:
                lb, gb, yb, wb = batch
                lb, gb, yb, wb = lb.to(DEVICE), gb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
                loss = (crit_none(model(lb, gb), yb) * wb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for lb, gb, yb in val_dl:
                lb, gb, yb = lb.to(DEVICE), gb.to(DEVICE), yb.to(DEVICE)
                vl += crit(model(lb, gb), yb).item() * len(yb)
                vn += len(yb)
        vl /= max(vn, 1)
        if vl < best_v - 1e-6:
            best_v, bad = vl, 0
            best_s = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_s:
        model.load_state_dict(best_s)
    model.eval()
    ns = {"lmu": lmu, "lsd": lsd, "gmu": gmu, "gsd": gsd}
    return model, ldaps_med, gfs_med, ns, best_v


@torch.no_grad()
def predict_cnn(model, ldaps, gfs, ldaps_med, gfs_med, ns):
    if len(ldaps) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    ldaps = _nan_to_med(ldaps.copy(), ldaps_med)
    gfs = _nan_to_med(gfs.copy(), gfs_med)
    lt = (np.transpose(ldaps, (0, 3, 1, 2)) - ns["lmu"]) / ns["lsd"]
    gt = (np.transpose(gfs, (0, 3, 1, 2)) - ns["gmu"]) / ns["gsd"]
    lt_t = torch.tensor(lt, dtype=torch.float32).to(DEVICE)
    gt_t = torch.tensor(gt, dtype=torch.float32).to(DEVICE)
    ps = []
    bs = CNN_BATCH * 2
    for i in range(0, len(lt_t), bs):
        ps.append(model(lt_t[i:i+bs], gt_t[i:i+bs]).cpu().numpy())
    return np.concatenate(ps, axis=0)


def blend_with_fallback(base, cnn_partial, valid_mask, w_cnn=W_CNN):
    """valid_mask=True인 행만 (1-w)*base+w*cnn, 나머지는 base 그대로(폴백)."""
    out = base.copy()
    out[valid_mask] = np.clip((1 - w_cnn) * base[valid_mask] + w_cnn * cnn_partial, 0, CAPS)
    return out
