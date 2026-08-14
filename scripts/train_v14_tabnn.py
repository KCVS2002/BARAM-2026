"""v14: 정형(tabular) 퀀타일 신경망 — NN 마지널 품질 가설의 정식 검증.

v13 재분석 결론: 시퀀스(결합분포) 이점은 시간별 독립 의사결정 구조상 활용 불가.
남은 질문 = "NN이 GBM보다 나은/보완적인 시간별 마지널 분포를 주는가".

설계 (소규모 정형데이터 모범 관행):
- MLP [512,512,256] + GELU + LayerNorm + dropout 0.15, 그룹 임베딩
- 출력: 기저 분위 + softplus 누적 → 비교차(non-crossing) 19분위 보장
- 3시드 앙상블 (분위 평균), 얼리스탑 (학습 내 무작위 10%)
- 평가: 단독 / GBM+AnEn 위 3원 결합 (모두 재정렬)
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.train_v11_anen import ANEN_FEATS, FEAT_W
from src.decision import QLEVELS, interp_atoms, optimize_submission, smooth_quantiles_by_day
from src.features import build_features
from src.metric import CAPACITY_KWH, TARGET_COLS, metric_single

DATA = PROJECT / "Data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
QL = torch.tensor(QLEVELS, dtype=torch.float32)
SEEDS = [42, 202, 777]


class TabQuantile(nn.Module):
    def __init__(self, n_feat, n_group=3, nq=19):
        super().__init__()
        self.emb = nn.Embedding(n_group, 12)
        self.net = nn.Sequential(
            nn.Linear(n_feat + 12, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Linear(256, nq),
        )

    def forward(self, x, g):
        raw = self.net(torch.cat([x, self.emb(g)], dim=-1))
        base = raw[:, :1]
        steps = nn.functional.softplus(raw[:, 1:])
        return torch.cat([base, base + torch.cumsum(steps, dim=1)], dim=1)  # 비교차 보장


def pinball(pred, y):
    ql = QL.to(pred.device).view(1, -1)
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(ql * diff, (ql - 1) * diff).mean()


def main() -> None:
    t0 = time.time()
    lab = pd.read_csv(DATA / "train/train_labels.csv", encoding="utf-8-sig", parse_dates=["kst_dtm"])
    feat = build_features(
        pd.read_csv(DATA / "train/ldaps_train.csv", encoding="utf-8-sig"),
        pd.read_csv(DATA / "train/gfs_train.csv", encoding="utf-8-sig"),
    )
    df = lab.rename(columns={"kst_dtm": "forecast_kst_dtm"}).merge(feat, on="forecast_kst_dtm", how="left")
    base_cols = [c for c in feat.columns if c != "forecast_kst_dtm"]
    oof = pd.read_parquet(PROJECT / "experiments" / "oof_quantiles.parquet")
    oof["forecast_kst_dtm"] = pd.to_datetime(oof["forecast_kst_dtm"])
    QC = [f"q{j}" for j in range(19)]
    print(f"device={DEV}, feats={len(base_cols)} ({time.time()-t0:.0f}s)")

    res_solo, res_tri = {}, {}
    for m0 in range(1, 12, 2):
        va_start = pd.Timestamp(2024, m0, 1, 1)
        va_end = va_start + pd.DateOffset(months=2)
        fold = f"{va_start:%Y-%m}"
        tr = df[df.forecast_kst_dtm < va_start]
        va = df[(df.forecast_kst_dtm >= va_start) & (df.forecast_kst_dtm < va_end)]

        med_fill = tr[base_cols].median()
        mu = tr[base_cols].mean()
        sd = tr[base_cols].std().replace(0, 1)

        def prep(frame):
            return ((frame[base_cols].fillna(med_fill) - mu) / sd).to_numpy(dtype=np.float32)

        xs, ys, gs = [], [], []
        for gi, tgt in enumerate(TARGET_COLS):
            m = tr[tgt].notna()
            xs.append(prep(tr.loc[m]))
            ys.append((tr.loc[m, tgt] / CAPACITY_KWH[tgt]).to_numpy(dtype=np.float32))
            gs.append(np.full(int(m.sum()), gi))
        Xtr = torch.tensor(np.concatenate(xs), device=DEV)
        Ytr = torch.tensor(np.concatenate(ys), device=DEV)
        Gtr = torch.tensor(np.concatenate(gs), dtype=torch.long, device=DEV)

        preds_by_seed = {t: [] for t in TARGET_COLS}
        for seed in SEEDS:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            n = len(Xtr)
            val_idx = torch.tensor(rng.choice(n, max(int(n * 0.1), 100), replace=False), device=DEV)
            val_mask = torch.zeros(n, dtype=torch.bool, device=DEV)
            val_mask[val_idx] = True
            tr_idx = torch.arange(n, device=DEV)[~val_mask]

            model = TabQuantile(len(base_cols)).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            best, best_state, patience = 1e9, None, 0
            for epoch in range(300):
                model.train()
                idx = tr_idx[torch.randperm(len(tr_idx), device=DEV)]
                for b in range(0, len(idx), 1024):
                    j = idx[b:b + 1024]
                    opt.zero_grad()
                    loss = pinball(model(Xtr[j], Gtr[j]), Ytr[j])
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = pinball(model(Xtr[val_idx], Gtr[val_idx]), Ytr[val_idx]).item()
                if vl < best - 1e-5:
                    best, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience >= 12:
                        break
            model.load_state_dict(best_state)
            model.eval()
            for gi, tgt in enumerate(TARGET_COLS):
                sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm")
                with torch.no_grad():
                    qp = model(torch.tensor(prep(sub), device=DEV),
                               torch.full((len(sub),), gi, dtype=torch.long, device=DEV)).cpu().numpy()
                preds_by_seed[tgt].append(np.clip(qp, 0, 1) * CAPACITY_KWH[tgt])

        fs_solo, fs_tri = [], []
        for tgt in TARGET_COLS:
            cap = CAPACITY_KWH[tgt]
            sub = va.loc[va[tgt].notna()].sort_values("forecast_kst_dtm")
            qp_nn = np.sort(np.mean(preds_by_seed[tgt], axis=0), axis=1)
            qp_nn = np.sort(smooth_quantiles_by_day(qp_nn, sub["forecast_kst_dtm"]), axis=1)
            nn_atoms = interp_atoms(qp_nn, n=150)

            o = oof[(oof.fold == fold) & (oof.target == tgt)].set_index("forecast_kst_dtm")
            o = o.reindex(sub.forecast_kst_dtm)
            qp_g = np.sort(smooth_quantiles_by_day(o[QC].to_numpy(), sub.forecast_kst_dtm), axis=1)
            gbm = interp_atoms(qp_g, n=150)
            med = np.median(gbm, axis=1)

            trm = tr[tgt].notna()
            a_tr = tr.loc[trm, tgt]
            a_bar = a_tr[a_tr >= cap * 0.10].mean()

            s1, _, _, _ = metric_single(sub[tgt].to_numpy(), optimize_submission(nn_atoms, cap, a_bar), cap)
            fs_solo.append(s1)

            tr_ok = tr.loc[trm].dropna(subset=ANEN_FEATS)
            mu_a = tr_ok[ANEN_FEATS].mean()
            sd_a = tr_ok[ANEN_FEATS].std().replace(0, 1)
            knn = NearestNeighbors(n_neighbors=150).fit(((tr_ok[ANEN_FEATS] - mu_a) / sd_a * FEAT_W).to_numpy())
            _, aidx = knn.kneighbors(((sub[ANEN_FEATS].fillna(mu_a) - mu_a) / sd_a * FEAT_W).to_numpy())
            anen = np.sort((tr_ok[tgt] / cap).to_numpy()[aidx] * cap, axis=1)
            anen = np.clip(anen + (med - np.median(anen, axis=1))[:, None], 0, cap)
            nn_rc = np.clip(nn_atoms + (med - np.median(nn_atoms, axis=1))[:, None], 0, cap)
            atoms = np.sort(np.concatenate([gbm, anen, nn_rc], axis=1), axis=1)
            s2, _, _, _ = metric_single(sub[tgt].to_numpy(), optimize_submission(atoms, cap, a_bar), cap)
            fs_tri.append(s2)
        res_solo[fold] = np.nanmean(fs_solo)
        res_tri[fold] = np.nanmean(fs_tri)
        print(f"fold {fold}: NN단독={res_solo[fold]:.4f} 3원={res_tri[fold]:.4f} ({time.time()-t0:.0f}s)")

    print(f"\n=== v14 tabNN: 단독={np.mean(list(res_solo.values())):.4f} "
          f"3원={np.mean(list(res_tri.values())):.4f} (기준: GBM단독 0.6346 / sub_009 구성 0.6470) ===")


if __name__ == "__main__":
    main()
