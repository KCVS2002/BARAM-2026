"""v13: GPU 시퀀스 모델 (하루 24h 동시 퀀타일 예측).

- 아키텍처: 양방향 GRU 2층 → 시간별 19퀀타일 헤드 (pinball loss)
- 입력: 하루 24h × 핵심 피처 (표준화) + 그룹 임베딩
- 출력 활용: (a) 단독 원자, (b) sub_009 원자에 제3소스로 결합
- 가설: 시퀀스 문맥이 GBM의 시간독립 예측이 놓치는 하루 레벨/타이밍 정보를 보완

기준: sub_009 구성 CV 0.6470.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from sklearn.neighbors import NearestNeighbors

from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from src.decision import QLEVELS, interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_FEATS = ["ldaps_ws50max", "ldaps_ws50min", "ldaps_ws10", "gfs_ws100", "gfs_ws80", "gfs_ws10",
             "gfs_ws850", "gfs_ws700", "gfs_surface_0_gust", "wpd_100m", "air_density",
             "gfs_ws100_dsin", "gfs_ws100_dcos", "ldaps_ws50max_dsin", "ldaps_ws50max_dcos",
             "shear_100_10", "shear_850_100", "ldaps_gust_ratio", "ws_ldaps_gfs_diff",
             "ldaps_etc_0_blh", "gfs_heightAboveGround_2_2t", "hour_sin", "hour_cos",
             "month_sin", "month_cos"]
QL = torch.tensor(QLEVELS, dtype=torch.float32)


class SeqQuantile(nn.Module):
    def __init__(self, n_feat, n_group=3, hidden=96, nq=19):
        super().__init__()
        self.emb = nn.Embedding(n_group, 8)
        self.gru = nn.GRU(n_feat + 8, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU(), nn.Linear(64, nq))

    def forward(self, x, g):
        e = self.emb(g).unsqueeze(1).expand(-1, x.shape[1], -1)
        h, _ = self.gru(torch.cat([x, e], dim=-1))
        return self.head(h)  # (B, 24, nq)


def pinball(pred, y, mask):
    ql = QL.to(pred.device).view(1, 1, -1)
    diff = y.unsqueeze(-1) - pred
    loss = torch.maximum(ql * diff, (ql - 1) * diff)
    return (loss * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1) / len(QLEVELS)


def build_days(df, feats):
    """(days, 24, F) 텐서와 (days, 24) 타깃 cf, 결측 마스크."""
    df = df.sort_values("forecast_kst_dtm").copy()
    df["tday"] = (df.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.date
    df["thour"] = (df.forecast_kst_dtm - pd.Timedelta(hours=1)).dt.hour  # 0~23 블록 내 인덱스
    days = sorted(df.tday.unique())
    X = np.full((len(days), 24, len(feats)), np.nan, dtype=np.float32)
    Y = {t: np.full((len(days), 24), np.nan, dtype=np.float32) for t in TARGET_COLS}
    dmap = {d: i for i, d in enumerate(days)}
    di = df.tday.map(dmap).to_numpy()
    hi = df.thour.to_numpy()
    X[di, hi] = df[feats].to_numpy(dtype=np.float32)
    for t in TARGET_COLS:
        Y[t][di, hi] = (df[t] / CAPACITY_KWH[t]).to_numpy(dtype=np.float32)
    return np.array(days), X, Y


def main() -> None:
    t0 = time.time()
    torch.manual_seed(42)
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    feats = [c for c in SEQ_FEATS if c in feat.columns]
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    days, X, Y = build_days(df, feats)
    oof = pd.read_parquet(PROJECT / "experiments" / "oof_quantiles.parquet")
    oof["forecast_kst_dtm"] = pd.to_datetime(oof["forecast_kst_dtm"])
    QC = [f"q{j}" for j in range(19)]
    print(f"days={len(days)}, feats={len(feats)}, device={DEV} ({time.time()-t0:.0f}s)")

    res_solo, res_blend = {}, {}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr_i = days < va_start.date()
        va_i = (days >= va_start.date()) & (days < va_end.date())

        mu = np.nanmean(X[tr_i], axis=(0, 1))
        sd = np.nanstd(X[tr_i], axis=(0, 1)) + 1e-6
        Xn = (X - mu) / sd
        Xn = np.nan_to_num(Xn)

        # 그룹 스택 학습 데이터
        xs, ys, gs = [], [], []
        for gi, tgt in enumerate(TARGET_COLS):
            xs.append(Xn[tr_i])
            ys.append(Y[tgt][tr_i])
            gs.append(np.full(tr_i.sum(), gi))
        Xtr = torch.tensor(np.concatenate(xs), device=DEV)
        Ytr = torch.tensor(np.concatenate(ys), device=DEV)
        Gtr = torch.tensor(np.concatenate(gs), dtype=torch.long, device=DEV)
        Mtr = ~torch.isnan(Ytr)
        Ytr = torch.nan_to_num(Ytr)

        model = SeqQuantile(len(feats)).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(Xtr)
        # 마지막 10%를 얼리스탑 검증으로 (시간순 뒤쪽)
        n_val = max(int(n * 0.1), 30)
        perm_tr = torch.arange(n - n_val)
        best, best_state, patience = 1e9, None, 0
        for epoch in range(200):
            model.train()
            idx = perm_tr[torch.randperm(len(perm_tr))]
            for b in range(0, len(idx), 256):
                j = idx[b:b + 256]
                opt.zero_grad()
                loss = pinball(model(Xtr[j], Gtr[j]), Ytr[j], Mtr[j])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = pinball(model(Xtr[-n_val:], Gtr[-n_val:]), Ytr[-n_val:], Mtr[-n_val:]).item()
            if vl < best - 1e-5:
                best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 10:
                    break
        model.load_state_dict(best_state)
        model.eval()

        fs_solo, fs_blend = [], []
        for gi, tgt in enumerate(TARGET_COLS):
            cap = CAPACITY_KWH[tgt]
            with torch.no_grad():
                qp_nn = model(torch.tensor(Xn[va_i], device=DEV),
                              torch.full((va_i.sum(),), gi, dtype=torch.long, device=DEV)).cpu().numpy()
            qp_nn = np.clip(qp_nn, 0, 1) * cap  # (n_days, 24, 19)
            # 시간행으로 전개
            va_days = days[va_i]
            recs = []
            for di, d in enumerate(va_days):
                for h in range(24):
                    dtm = pd.Timestamp(d) + pd.Timedelta(hours=h + 1)
                    recs.append((dtm, *np.sort(qp_nn[di, h])))
            nn_df = pd.DataFrame(recs, columns=["forecast_kst_dtm", *QC]).set_index("forecast_kst_dtm")

            o = oof[(oof.fold == fold) & (oof.target == tgt)].set_index("forecast_kst_dtm").sort_index()
            actual = o.actual.to_numpy()
            common = o.index
            qp_gbm = np.sort(smooth_quantiles_by_day(o[QC].to_numpy(), pd.Series(common)), axis=1)
            gbm = interp_atoms(qp_gbm, n=150)
            med = np.median(gbm, axis=1)

            nn_q = nn_df.reindex(common)[QC].to_numpy()
            nn_atoms = interp_atoms(np.sort(nn_q, axis=1), n=150)
            tr_lab = df[(df.forecast_kst_dtm < va_start)][tgt].dropna()
            a_bar = tr_lab[tr_lab >= cap * 0.10].mean()

            pred_solo = optimize_submission(nn_atoms, cap, a_bar)
            s1, _, _, _ = metric_single(actual, pred_solo, cap)
            fs_solo.append(s1)

            nn_rc = np.clip(nn_atoms + (med - np.median(nn_atoms, axis=1))[:, None], 0, cap)
            # 3원: GBM + 시간아날로그 + NN (모두 GBM 중앙값 재정렬)
            tr_df = df[df.forecast_kst_dtm < va_start]
            trm = tr_df[tgt].notna()
            tr_ok = tr_df.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            nn_knn = NearestNeighbors(n_neighbors=150).fit(
                ((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            va_feat = df.set_index("forecast_kst_dtm").reindex(common)[ANEN_FEATS]
            _, aidx = nn_knn.kneighbors(((va_feat.fillna(mu_a) - mu_a) / sd_a * FEAT_W).to_numpy())
            ytr_lab = (tr_ok[tgt] / cap).to_numpy()
            anen = np.sort(ytr_lab[aidx] * cap, axis=1)
            anen = np.clip(anen + (med - np.median(anen, axis=1))[:, None], 0, cap)
            atoms = np.sort(np.concatenate([gbm, anen, nn_rc], axis=1), axis=1)
            pred_bl = optimize_submission(atoms, cap, a_bar)
            s2, _, _, _ = metric_single(actual, pred_bl, cap)
            fs_blend.append(s2)
        res_solo[fold] = np.nanmean(fs_solo)
        res_blend[fold] = np.nanmean(fs_blend)
        print(f"fold {fold}: solo={res_solo[fold]:.4f} gbm+anen+nn={res_blend[fold]:.4f} "
              f"(epochs~{epoch}) ({time.time()-t0:.0f}s)")

    print(f"\n=== v13 seq: solo={np.mean(list(res_solo.values())):.4f} "
          f"3원={np.mean(list(res_blend.values())):.4f} (기준 sub_009 구성 0.6470) ===")


if __name__ == "__main__":
    main()
