"""LDAPS/GFS 예보 데이터 → 물리 기반 피처 생성.

핵심 원칙 (CLAUDE.md):
- 격자별로 먼저 스칼라 풍속을 계산한 뒤 집계 (벡터 평균 금지 — v³ 편향).
- 시간축 문맥(rolling)은 같은 예보 사이클(data_available_kst_dtm) 내에서만 계산 → 누수 차단.
- 요일/주말 등 풍력과 무관한 캘린더 피처는 넣지 않는다.
"""

import numpy as np
import pandas as pd

# 터빈 중심 좌표 (info.xlsx)
SITE_LAT, SITE_LON = 37.284, 128.958

# (u컬럼, v컬럼, 접두어)
LDAPS_WIND = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "ws50max"),
    ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "ws50min"),
    ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "ws5bl"),
]
GFS_WIND = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("heightAboveGround_80_u", "heightAboveGround_80_v", "ws80"),
    ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100"),
    ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "wspbl"),
    ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "ws850"),
    ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "ws700"),
]
LDAPS_SCALAR = [
    "heightAboveGround_2_t", "heightAboveGround_2_r", "surface_0_sp",
    "meanSea_0_prmsl", "etc_0_blh", "surface_0_NDNSW",
    "etc_0_lcc", "surface_0_avg_lsprate",
]
GFS_SCALAR = [
    "heightAboveGround_2_2t", "heightAboveGround_2_2r", "surface_0_sp",
    "surface_0_gust", "surface_0_prate", "atmosphere_0_tcc",
    "isobaricInhPa_850_t", "isobaricInhPa_500_gh",
]


def _wind_scalar(df: pd.DataFrame, wind_pairs, prefix: str) -> pd.DataFrame:
    """격자별 행에서 스칼라 풍속/풍향 컬럼 추가."""
    out = df.copy()
    for u, v, name in wind_pairs:
        spd = np.sqrt(out[u] ** 2 + out[v] ** 2)
        out[f"{prefix}_{name}"] = spd
        # 풍향 (기상학 관례: 바람이 불어오는 방향)
        wd = (np.degrees(np.arctan2(-out[u], -out[v]))) % 360
        out[f"{prefix}_{name}_dsin"] = np.sin(np.radians(wd))
        out[f"{prefix}_{name}_dcos"] = np.cos(np.radians(wd))
    return out


def _aggregate(df: pd.DataFrame, prefix: str, wind_pairs, scalar_cols, n_nearest: int = 4) -> pd.DataFrame:
    """격자 차원 축소: 전 격자 mean/std + 최근접 n개 격자 개별 풍속."""
    df = _wind_scalar(df, wind_pairs, prefix)

    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    grids["dist"] = np.hypot(grids.latitude - SITE_LAT, grids.longitude - SITE_LON)
    nearest = grids.nsmallest(n_nearest, "dist").grid_id.tolist()

    speed_cols = [f"{prefix}_{n}" for _, _, n in wind_pairs]
    dir_cols = [c for c in df.columns if c.endswith(("_dsin", "_dcos"))]
    scalar_named = {c: f"{prefix}_{c}" for c in scalar_cols if c in df.columns}
    df = df.rename(columns=scalar_named)
    scalars = list(scalar_named.values())

    agg_mean = df.groupby("forecast_kst_dtm")[speed_cols + dir_cols + scalars].mean()
    agg_std = df.groupby("forecast_kst_dtm")[speed_cols].std()
    agg_std.columns = [f"{c}_gstd" for c in agg_std.columns]

    near_frames = []
    for rank, gid in enumerate(nearest):
        sub = df[df.grid_id == gid].set_index("forecast_kst_dtm")[speed_cols]
        sub.columns = [f"{c}_g{rank}" for c in sub.columns]
        near_frames.append(sub)

    return pd.concat([agg_mean, agg_std, *near_frames], axis=1).reset_index()


def _cycle_context(df: pd.DataFrame, key_cols) -> pd.DataFrame:
    """같은 예보 사이클(available_dtm 동일 = 대상일 단위 24행) 내 rolling 문맥.

    사이클 경계를 넘는 rolling은 다른(더 새로운) 사이클 정보를 섞을 수 있어 금지.
    제공 데이터는 대상일(01시~익일00시)당 한 사이클이므로 날짜 블록으로 그룹핑한다.
    """
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    block = (df.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.date
    for c in key_cols:
        g = df.groupby(block)[c]
        df[f"{c}_roll3"] = g.transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
        df[f"{c}_diff1"] = g.transform(lambda s: s.diff().fillna(0))
        df[f"{c}_dmax"] = g.transform("max")
        df[f"{c}_dmean"] = g.transform("mean")
    return df


def build_features(ldaps: pd.DataFrame, gfs: pd.DataFrame) -> pd.DataFrame:
    """forecast_kst_dtm 단위 피처 테이블 생성."""
    for d in (ldaps, gfs):
        d["forecast_kst_dtm"] = pd.to_datetime(d["forecast_kst_dtm"])

    ld = _aggregate(ldaps, "ldaps", LDAPS_WIND, LDAPS_SCALAR)
    gf = _aggregate(gfs, "gfs", GFS_WIND, GFS_SCALAR)
    feat = ld.merge(gf, on="forecast_kst_dtm", how="outer").sort_values("forecast_kst_dtm")

    # 공기밀도 및 풍력밀도 프록시 (GFS 2m 기온 + 지표기압)
    t = feat["gfs_heightAboveGround_2_2t"]
    t_kelvin = np.where(t < 200, t + 273.15, t)  # 단위 자동 판별
    rho = feat["gfs_surface_0_sp"] / (287.05 * t_kelvin)
    feat["air_density"] = rho
    feat["wpd_100m"] = rho * feat["gfs_ws100"] ** 3
    feat["wpd_ldaps50"] = rho * feat["ldaps_ws50max"] ** 3

    # 연직 시어 (허브 117m 주변 추정에 핵심)
    feat["shear_100_10"] = feat["gfs_ws100"] / feat["gfs_ws10"].clip(lower=0.1)
    feat["shear_850_100"] = feat["gfs_ws850"] / feat["gfs_ws100"].clip(lower=0.1)
    feat["ldaps_gust_ratio"] = feat["ldaps_ws50max"] / feat["ldaps_ws50min"].clip(lower=0.1)

    # 모델 간 불일치 = 예보 불확실성 프록시
    feat["ws_ldaps_gfs_diff"] = feat["ldaps_ws50max"] - feat["gfs_ws100"]

    # 사이클 내 시간 문맥
    key = ["gfs_ws100", "ldaps_ws50max", "wpd_100m", "gfs_surface_0_gust"]
    feat = _cycle_context(feat, [c for c in key if c in feat.columns])

    # 캘린더 (주기성만)
    dt = feat["forecast_kst_dtm"]
    hour, month = dt.dt.hour, dt.dt.month
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12)
    # 예보 리드타임: D-1 13:00 발표 기준, 대상 01시→12h ... 익일 00시→35h
    feat["lead_h"] = dt.dt.hour.where(dt.dt.hour > 0, 24) + 11

    return feat.reset_index(drop=True)
