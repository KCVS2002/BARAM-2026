"""캐시 우선 파이프라인 규칙의 공용 헬퍼 (#109 도입, 사용자 고정 지시).

기본 프레임(라벨+124피처+IFS10+파생, q3 가중, 컬럼 목록)을 parquet으로 1회
저장하고 이후 실험은 수 초 로드. 현행 공식(sub_029) 상류 동결이 전제 —
피처·가중이 공식적으로 바뀌면 regenerate=True로 1회 재생성한다.

사용:
    from scripts.exp_cache import load_base_frame
    df, w_q3, cols, shared = load_base_frame()

기존 캐시 자산 (fold별 CV 재료): experiments/oof_cache_q3/
    {fold}_{tgt}.npz  : qp(LGBM 19분위, 정렬·평활 완료)/anen(200)/dist/actual/a_bar/cap
    {fold}_cbqp.npz   : CatBoost 19분위 (v71b~)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

CACHE_DIR = PROJECT / "experiments" / "cache"
FRAME_PQ = CACHE_DIR / "base_frame_q3.parquet"
W_PQ = CACHE_DIR / "w_q3.parquet"
COLS_TXT = CACHE_DIR / "cols_q3.txt"

IFS10 = ["ifs_ws10", "ifs_ws925", "ifs_ws850", "cons3_mean", "cons3_std",
         "ifs_t925", "ifs_dt", "ifs_ws700", "ifs_shear", "ifs_q850"]


def load_base_frame(regenerate: bool = False):
    """(df, w_q3, cols, shared) — 현행 공식(sub_029) 학습 프레임. 캐시 우선."""
    if not regenerate and FRAME_PQ.exists() and W_PQ.exists() and COLS_TXT.exists():
        df = pd.read_parquet(FRAME_PQ)
        w_q3 = pd.read_parquet(W_PQ)
        cols = COLS_TXT.read_text(encoding="utf-8").split("\n")
        return df, w_q3, cols, cols + ["g_rated", "g_rotor", "g_id"]

    t0 = time.time()
    from scripts.train_v2_clean_shared import label_weights
    from scripts.train_v25_ifs import load_ifs_features
    from scripts.train_v31b_phys import load_ifs2_features
    from src.features import build_features
    from src.metric import CAPACITY_KWH, TARGET_COLS

    data = PROJECT / "Data"
    lab = pd.read_csv(data / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(data / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(data / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    w_q3 = label_weights(lab)
    for tgt in TARGET_COLS:
        cf = (lab[tgt] / CAPACITY_KWH[tgt]).fillna(0).to_numpy()
        w_q3[tgt] = w_q3[tgt].to_numpy() * (1 + 3 * np.clip(cf, 0, 1) ** 2)
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs_features(), on="forecast_kst_dtm", how="left")
    df = df.merge(load_ifs2_features(), on="forecast_kst_dtm", how="left")
    df["ifs_shear"] = df["ifs_ws700"] - df["ifs_ws925"]
    tri = ["ldaps_ws50max", "gfs_ws100", "ifs_ws925"]
    z = (df[tri] - df[tri].mean()) / df[tri].std()
    df["cons3_mean"] = z.mean(axis=1)
    df["cons3_std"] = z.std(axis=1)
    cols = [c for c in feat.columns if c != "forecast_kst_dtm"] + IFS10

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FRAME_PQ)
    w_q3.to_parquet(W_PQ)
    COLS_TXT.write_text("\n".join(cols), encoding="utf-8")
    print(f"[exp_cache] 기본 프레임 재생성·저장 ({time.time()-t0:.0f}s, {len(df)}행 {len(cols)}피처)", flush=True)
    return df, w_q3, cols, cols + ["g_rated", "g_rotor", "g_id"]


if __name__ == "__main__":
    df, w, cols, shared = load_base_frame(regenerate="--regen" in sys.argv)
    print(f"df {df.shape}, w {w.shape}, cols {len(cols)}, shared {len(shared)}")
