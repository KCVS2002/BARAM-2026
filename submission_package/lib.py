# -*- coding: utf-8 -*-
"""BARAM 2026 최종 제출물(sub_055) 재현 — 공용 모듈.

본 파일은 대회 기간 검증된 코드의 축자 이식본이다. 구성 요소:
  - 라벨 정제 가중 (SCADA 파워커브 잔차 기반 이상운영 다운웨이트)
  - 외부데이터 로더 (ECMWF IFS 기압면 2종, Open-Meteo 3모델 — 모두 D-1 13:00 KST
    이전 발표분만 사용, external_data/README 참조)
  - AnEn(Analog Ensemble) 유사도 피처·가중
  - CNN sister (KMA LDAPS 1.5km 875hPa u/v 28×28 공간장 → 19분위)
  - 결정 배깅 (부트스트랩 원자 60% × 15회 FICR 기대점수 최적화 → 평균)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DATA_DIR", ROOT / "Data"))
EXT = ROOT / "external_data"
MODELS = ROOT / "models"

TARGET_COLS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
CAPACITY_KWH = {"kpx_group_1": 21600, "kpx_group_2": 21600, "kpx_group_3": 21000}

QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
# LB 검증으로 고정된 유일한 앵커 파라미터 (대회 로그 #9·#52 참조)
BASE_PARAMS = dict(
    n_estimators=700, learning_rate=0.05, num_leaves=63, min_child_samples=40,
    colsample_bytree=0.8, subsample=0.8, subsample_freq=1, random_state=42, verbose=-1,
)
GROUP_TURBINES = {
    "kpx_group_1": ("scada_vestas_train.csv", [f"vestas_wtg{i:02d}" for i in range(1, 7)]),
    "kpx_group_2": ("scada_vestas_train.csv", [f"vestas_wtg{i:02d}" for i in range(7, 13)]),
    "kpx_group_3": ("scada_unison_train.csv", [f"unison_wtg{i:02d}" for i in range(1, 6)]),
}
GROUP_META = {  # (정격 kW, 로터 m)
    "kpx_group_1": (3600, 126),
    "kpx_group_2": (3600, 126),
    "kpx_group_3": (4200, 136),
}

# AnEn 유사도 공간 (표준화 후 가중)
ANEN_FEATS = ["ldaps_ws50max", "gfs_ws100", "gfs_ws80", "gfs_ws850",
              "ldaps_ws10", "gfs_surface_0_gust",
              "gfs_ws100_dsin", "gfs_ws100_dcos",
              "hour_sin", "hour_cos", "month_sin", "month_cos"]
FEAT_W = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])

IFS10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
         "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]

K = 150
NA_TERC = {0: 125, 1: 150, 2: 175}
P_MIX = 0.15
HOUR_NS = 3600 * 10 ** 9
KP = [0.00, 0.45, 0.60, 1.00]  # 계절 축소 램프 노트 (p = pred/cap)


# ── 라벨 정제 가중 ──────────────────────────────────────────────

def label_weights(lab: pd.DataFrame) -> pd.DataFrame:
    """SCADA 파워커브 잔차로 이상운영 시간 다운웨이트 (기본 1.0, 이상 0.2)."""
    w = pd.DataFrame(1.0, index=lab.index, columns=TARGET_COLS)
    scada_cache = {}
    for tgt, (fname, turbines) in GROUP_TURBINES.items():
        if fname not in scada_cache:
            sc = pd.read_csv(DATA / "train" / fname, encoding="utf-8-sig", parse_dates=["kst_dtm"])
            sc["hour_end"] = sc["kst_dtm"].dt.ceil("h")
            scada_cache[fname] = sc
        sc = scada_cache[fname]
        ws_cols = [f"{t}_ws" for t in turbines]
        ws = sc[ws_cols].where((sc[ws_cols] >= 0) & (sc[ws_cols] < 60))
        farm_ws = ws.mean(axis=1)
        hourly_ws = farm_ws.groupby(sc["hour_end"]).mean()

        m = lab[["kst_dtm", tgt]].merge(hourly_ws.rename("ws"), left_on="kst_dtm", right_index=True)
        m = m.dropna()
        bins = (m.ws / 0.5).round().astype(int)
        curve = m.groupby(bins)[tgt].transform(lambda s: s.quantile(0.9))
        anomal = (m.ws > 5) & (m[tgt] < 0.3 * curve)
        idx = m.index[anomal]
        w.loc[w.index.isin(lab.index[lab.index.isin(idx)]), tgt] = 1.0
        w.loc[idx, tgt] = 0.2
        print(f"{tgt}: anomalous hours {anomal.sum()} / {len(m)} ({anomal.mean()*100:.1f}%)")
    return w


def q3_weights(lab: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """정제 가중 × (1 + 3·cf²) — 고출력(FICR 에너지 가중) 강조."""
    out = weights.copy()
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        out[tgt] = out[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
    return out


# ── 외부데이터 로더 (수집·누수 근거는 external_data/README 참조) ──

def load_ifs_features() -> pd.DataFrame:
    d = pd.read_csv(EXT / "ecmwf_ifs" / "ifs_point_2022_2025.csv", encoding="utf-8-sig")
    d = d.drop_duplicates(subset=["run", "fxx", "var", "lat", "lon"])
    d["forecast_kst_dtm"] = (pd.to_datetime(d.run) + pd.Timedelta(hours=9)
                             + pd.to_timedelta(d.fxx, unit="h"))
    piv = d.pivot_table(index=["forecast_kst_dtm", "lat", "lon"], columns="var", values="value")
    out = pd.DataFrame(index=piv.index.get_level_values(0).unique().sort_values())
    for lev in ("10", "925", "850"):
        u, v = piv.get(f"u{lev}"), piv.get(f"v{lev}")
        ws = np.sqrt(u ** 2 + v ** 2)
        out[f"ifs_ws{lev}"] = ws.groupby(level="forecast_kst_dtm").mean()
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(full).interpolate(limit=2)
    out.index.name = "forecast_kst_dtm"
    return out.reset_index()


def load_ifs2_features() -> pd.DataFrame:
    d = pd.read_csv(EXT / "ecmwf_ifs" / "ifs_point2_2022_2025.csv", encoding="utf-8-sig")
    d = d.drop_duplicates(subset=["run", "fxx", "var", "lat", "lon"])
    d["forecast_kst_dtm"] = (pd.to_datetime(d.run) + pd.Timedelta(hours=9)
                             + pd.to_timedelta(d.fxx, unit="h"))
    piv = d.pivot_table(index=["forecast_kst_dtm", "lat", "lon"], columns="var", values="value")
    out = pd.DataFrame(index=piv.index.get_level_values(0).unique().sort_values())
    grp = piv.groupby(level="forecast_kst_dtm")
    out["ifs_t925"] = grp["t925"].mean()
    out["ifs_dt"] = grp["t925"].mean() - grp["t850"].mean()
    ws700 = np.sqrt(piv["u700"] ** 2 + piv["v700"] ** 2)
    out["ifs_ws700"] = ws700.groupby(level="forecast_kst_dtm").mean()
    out["ifs_q850"] = grp["q850"].mean() * 1000.0
    full = pd.date_range(out.index.min(), out.index.max(), freq="h")
    out = out.reindex(full).interpolate(limit=2)
    out.index.name = "forecast_kst_dtm"
    return out.reset_index()


def load_om() -> pd.DataFrame:
    """Open-Meteo previous_day2/3 (항상 D-1 13:00 이전 발표) 3모델."""
    out = None
    for tag, fn in (("om_ec", "ecmwf_ifs025_prev_runs.csv"),
                    ("om_ic", "icon_global_prev_runs.csv"),
                    ("om_gf", "gfs_global_prev_runs.csv")):
        d = pd.read_csv(EXT / "openmeteo" / fn, encoding="utf-8-sig", parse_dates=["kst_dtm"])
        cols = {}
        for base in ("wind_speed_100m", "wind_speed_10m", "wind_gusts_10m"):
            cols[f"{base}_previous_day2"] = f"{tag}_{base.replace('wind_', '').replace('_10m', '10').replace('_100m', '100')}_d2"
        cols["wind_speed_100m_previous_day3"] = f"{tag}_speed100_d3"
        wd = "wind_direction_100m_previous_day2"
        d2 = d[["kst_dtm"] + [c for c in cols if c in d.columns] + ([wd] if wd in d.columns else [])].rename(columns=cols)
        if wd in d2.columns:
            rad = np.deg2rad(d2[wd])
            d2[f"{tag}_dir_sin"] = np.sin(rad)
            d2[f"{tag}_dir_cos"] = np.cos(rad)
            d2 = d2.drop(columns=[wd])
        d2 = d2.rename(columns={"kst_dtm": "forecast_kst_dtm"})
        out = d2 if out is None else out.merge(d2, on="forecast_kst_dtm", how="outer")
    return out


def add_consensus(df_tr: pd.DataFrame, te: pd.DataFrame) -> None:
    """3소스(LDAPS/GFS/IFS) 컨센서스 z-피처 — 통계량은 train에서만 산출."""
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    mu_t, sd_t = df_tr[tri].mean(), df_tr[tri].std()
    for d_ in (df_tr, te):
        z = (d_[tri] - mu_t) / sd_t
        d_["cons3_mean"] = z.mean(axis=1)
        d_["cons3_std"] = z.std(axis=1)
    for d_ in (df_tr, te):
        d_["ifs_shear"] = d_["ifs_ws700"] - d_["ifs_ws925"]


# ── CNN sister ─────────────────────────────────────────────────

class SisterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gemb = nn.Embedding(3, 8)
        self.head = nn.Sequential(nn.Linear(64 + 8, 64), nn.ReLU(), nn.Linear(64, 19))

    def forward(self, x, g):
        z = self.conv(x).flatten(1)
        return torch.sigmoid(self.head(torch.cat([z, self.gemb(g)], 1)))


def pinball_loss(pred, y, w):
    q = torch.tensor(QUANTILES, dtype=torch.float32)[None, :]
    diff = y[:, None] - pred
    return (torch.maximum(q * diff, (q - 1) * diff).mean(1) * w).mean()


def load_grid_hours():
    """external_data/kma_ldaps_grid/{YYYY-MM-DD}.npz → (dtm int64 ns, (N,2,28,28))."""
    dtms, fields = [], []
    for f in sorted((EXT / "kma_ldaps_grid").glob("20*.npz")):
        z = np.load(f)
        d = pd.Timestamp(f.stem)
        hrs = pd.date_range(d + pd.Timedelta(hours=1), periods=24, freq="h")
        dtms.append(hrs.values.astype("datetime64[ns]").astype(np.int64))
        fields.append(np.stack([z["u"], z["v"]], axis=1))
    return np.concatenate(dtms), np.concatenate(fields)


# ── 결정 배깅 ──────────────────────────────────────────────────

def optimize_bagged(atoms, cap, a_bar, B=15, frac=0.6, seed=42):
    from src.decision import optimize_submission
    rng = np.random.default_rng(seed)
    n = atoms.shape[1]
    k = int(n * frac)
    gs = np.empty((atoms.shape[0], B))
    for b in range(B):
        idx = np.sort(rng.choice(n, size=k, replace=False))
        gs[:, b] = optimize_submission(atoms[:, idx], cap, a_bar)
    return np.clip(gs.mean(axis=1), 0, cap)
