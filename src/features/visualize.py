"""
visualize.py
------------
Figures for each step of the TDA pipeline, in the spirit of Souto (2023). They
double as illustrations for the paper's Methodology (§9) and as sanity checks
that the topology behaves as expected in calm vs. turbulent windows.

Generated figures (saved to results/figures/):
  1. embedding.png        — Takens point cloud, calm vs turbulent window (3-D)
  2. filtration.png       — Vietoris–Rips filtration of a window at growing ε (paper Fig 3)
  3. diagrams.png         — persistence diagrams, calm vs turbulent
  4. barcode.png          — persistence barcodes, calm vs turbulent
  5. landscape.png        — persistence landscapes (§8.1), calm vs turbulent
  6. persistence_image.png— persistence images (§8.2), calm vs turbulent
  7. change_vs_rv.png      — Wasserstein topological-change signal vs realized vol
                            over time (Souto's key figure) — needs the TDA file.

Usage
-----
  python -m src.features.visualize
  python -m src.features.visualize --symbol BTC/USDT --turbulent 2022-11-09T14:00 \
      --calm 2023-09-15T12:00
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

from src.features.tda import TDAConfig, takens_embedding, _finite

FIGDIR = Path("results/figures")
CALM, TURB = "#2c7fb8", "#d7301f"   # blue / red throughout


# ── window helpers ────────────────────────────────────────────────────────────

def _load_close(symbol: str, cfg: dict) -> pd.Series:
    safe = symbol.replace("/", "-")
    close = pd.read_parquet(Path(cfg["data"]["raw_dir"]) / f"{safe}_1m.parquet")["close"]
    if close.index.tz is None:
        close.index = close.index.tz_localize("UTC")
    return close.sort_index()


def _window_series(close: pd.Series, t: str, tcfg: TDAConfig) -> np.ndarray:
    """1-min log-returns in (t-window, t], globally scaled — exactly what the
    extractor embeds (so figures show the real point cloud, not a stand-in)."""
    end = pd.Timestamp(t, tz="UTC")
    start = end - pd.Timedelta(minutes=tcfg.window_size)
    r = np.log(close).diff()
    r = r[(r.index > start) & (r.index <= end)].clip(-tcfg.ret_clip, tcfg.ret_clip).dropna()
    return r.values * tcfg.ret_scale


def _diagrams(series: np.ndarray, tcfg: TDAConfig):
    cloud = takens_embedding(series, tcfg.embedding_dim, tcfg.embedding_delay)
    dgms = ripser(cloud, maxdim=tcfg.max_homology_dim)["dgms"]
    while len(dgms) < 2:
        dgms.append(np.empty((0, 2)))
    return cloud, dgms


# ── 1. Takens embedding ───────────────────────────────────────────────────────

def fig_embedding(calm_cloud, turb_cloud, path):
    fig = plt.figure(figsize=(11, 4.6))
    for i, (cloud, title, col) in enumerate(
        [(calm_cloud, "Calm window", CALM), (turb_cloud, "Turbulent window", TURB)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=col, s=18, alpha=0.8,
                   depthshade=True)
        ax.plot(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=col, lw=0.4, alpha=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(r"$r_t$"); ax.set_ylabel(r"$r_{t-\tau}$"); ax.set_zlabel(r"$r_{t-2\tau}$")
    fig.suptitle("Takens delay embedding of 1-min returns  →  point cloud", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ── 2. Vietoris–Rips filtration ───────────────────────────────────────────────

def fig_filtration(cloud, dgms, path):
    """2-D projection of one window's cloud with edges added as ε grows."""
    pts = cloud[:, :2]
    D = squareform(pdist(cloud))                  # full-dim distances for edges
    h1 = _finite(dgms[1])
    if len(h1):
        born = h1[np.argmax(h1[:, 1] - h1[:, 0])]
        eps_vals = [0.3 * born[0], born[0], 0.5 * (born[0] + born[1]), born[1]]
    else:
        qs = np.quantile(D[D > 0], [0.05, 0.15, 0.30, 0.50])
        eps_vals = list(qs)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    for ax, eps in zip(axes, eps_vals):
        ax.scatter(pts[:, 0], pts[:, 1], c=TURB, s=16, zorder=3)
        iu = np.triu_indices(len(cloud), k=1)
        for a, b in zip(*iu):
            if D[a, b] <= eps:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                        c="0.55", lw=0.5, zorder=1)
        ax.set_title(fr"$\varepsilon$ = {eps:.2f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Vietoris–Rips filtration: edges appear as the scale ε grows "
                 "(a loop is born, then fills in)", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ── 3. Persistence diagrams ───────────────────────────────────────────────────

def _plot_diagram(ax, dgms, title):
    allpts = _finite(np.vstack([d for d in dgms if len(d)] or [np.zeros((1, 2))]))
    hi = float(allpts[:, 1].max()) if len(allpts) else 1.0
    ax.plot([0, hi], [0, hi], "k--", lw=0.8, alpha=0.6)
    for d, lab, col in [(dgms[0], r"$H_0$", "#444"), (dgms[1], r"$H_1$", TURB)]:
        d = _finite(d)
        if len(d):
            ax.scatter(d[:, 0], d[:, 1], s=22, c=col, label=lab, alpha=0.8, edgecolor="w", lw=0.3)
    ax.set_xlabel("birth"); ax.set_ylabel("death"); ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)


def fig_diagrams(calm_dgms, turb_dgms, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    _plot_diagram(axes[0], calm_dgms, "Calm window")
    _plot_diagram(axes[1], turb_dgms, "Turbulent window")
    fig.suptitle("Persistence diagrams — points far from the diagonal are real structure",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ── 4. Barcode ────────────────────────────────────────────────────────────────

def _plot_barcode(ax, dgms, title):
    y = 0; yticks_lab = []
    for d, col in [(dgms[0], "#444"), (dgms[1], TURB)]:
        d = _finite(d)
        if not len(d):
            continue
        d = d[np.argsort(d[:, 0])]
        for b, dd in d:
            ax.plot([b, dd], [y, y], c=col, lw=1.6)
            y += 1
    ax.set_title(title, fontsize=11); ax.set_xlabel("ε"); ax.set_yticks([])
    ax.set_ylim(-1, max(y, 1))


def fig_barcode(calm_dgms, turb_dgms, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _plot_barcode(axes[0], calm_dgms, "Calm window")
    _plot_barcode(axes[1], turb_dgms, "Turbulent window")
    fig.suptitle(r"Persistence barcodes ($H_0$ grey, $H_1$ red): long bars = persistent features",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ── 5. Persistence landscape ──────────────────────────────────────────────────

def _landscape_layers(dgm, n_grid=200, levels=4):
    pts = _finite(dgm)
    if len(pts) == 0:
        return np.linspace(0, 1, n_grid), np.zeros((levels, n_grid))
    b, d = pts[:, 0], pts[:, 1]
    ts = np.linspace(b.min(), d.max(), n_grid)
    tents = np.maximum(0.0, np.minimum(ts[None, :] - b[:, None], d[:, None] - ts[None, :]))
    k = min(levels, tents.shape[0])
    lam = np.sort(tents, axis=0)[::-1][:k]
    return ts, lam


def fig_landscape(calm_dgms, turb_dgms, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, dgms, title in [(axes[0], calm_dgms, "Calm window"),
                            (axes[1], turb_dgms, "Turbulent window")]:
        ts, lam = _landscape_layers(dgms[1])      # H1 landscape
        for j, layer in enumerate(lam):
            ax.plot(ts, layer, lw=1.4, label=fr"$\lambda_{{{j+1}}}$")
        ax.fill_between(ts, lam[0], alpha=0.15, color=TURB)
        ax.set_title(title, fontsize=11); ax.set_xlabel("ε")
        ax.legend(fontsize=8, loc="upper right")
    axes[0].set_ylabel("landscape value")
    fig.suptitle(r"Persistence landscapes of $H_1$ (§8.1) — taller = more topological structure",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ── 6. Persistence image ──────────────────────────────────────────────────────

def fig_persistence_image(calm_dgms, turb_dgms, path):
    from persim import PersistenceImager
    pimgr = PersistenceImager(pixel_size=0.05)
    h1c, h1t = _finite(calm_dgms[1]), _finite(turb_dgms[1])
    if len(h1c) and len(h1t):
        pimgr.fit([h1c, h1t], skew=True)
        imc, imt = pimgr.transform([h1c, h1t], skew=True)
    else:
        imc = imt = np.zeros((10, 10))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    vmax = max(np.max(imc) if np.size(imc) else 1, np.max(imt) if np.size(imt) else 1, 1e-9)
    for ax, im, title in [(axes[0], imc, "Calm window"), (axes[1], imt, "Turbulent window")]:
        ax.imshow(np.rot90(im), cmap="magma", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=11); ax.set_xlabel("birth"); ax.set_ylabel("persistence")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(r"Persistence images of $H_1$ (§8.2) — vectorized input for XGBoost",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ── 7. Topological-change signal vs realized vol (the Souto figure) ───────────

def fig_change_vs_rv(symbol, cfg, path, max_points=6000):
    safe = symbol.replace("/", "-")
    tda_path = Path(cfg["data"]["proc_dir"]) / f"{safe}_tda_features.csv.gz"
    if not tda_path.exists():
        print(f"  (skip change_vs_rv: {tda_path} not built yet)")
        return False
    df = pd.read_csv(tda_path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if len(df) > max_points:                      # thin for plotting only
        df = df.iloc[:: len(df) // max_points]

    rv = np.sqrt(df["rv_target"].clip(lower=1e-12)) * 100   # ~% vol, readable

    fig, ax1 = plt.subplots(figsize=(13, 4.6))
    ax1.plot(df.index, rv, color="0.35", lw=0.7, label="realized vol (next 30m, %)")
    ax1.set_ylabel("realized vol (%)", color="0.35")
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["tda_wass_h1"], color=TURB, lw=0.6, alpha=0.8,
             label=r"$W_2(\mathrm{Dgm}_t,\mathrm{Dgm}_{t-1})$ — $H_1$")
    ax2.set_ylabel("topological change (Wasserstein)", color=TURB)

    for p in cfg.get("turbulent_periods", []):
        s, e = pd.Timestamp(p["start"], tz="UTC"), pd.Timestamp(p["end"], tz="UTC")
        if s >= df.index.min() and e <= df.index.max():
            ax1.axvspan(s, e, color="orange", alpha=0.15)
            ax1.text(s, ax1.get_ylim()[1] * 0.92, p["name"], fontsize=7, color="darkorange")

    corr = df["tda_wass_h1"].corr(np.log(df["rv_target"].clip(lower=1e-12)))
    fig.suptitle(f"{symbol}: topological change vs realized volatility "
                 f"(corr with log-RV = {corr:+.2f})", fontsize=12)
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return True


# ── 8. Results: the model ladder (baseline → +freq control → +topology) ───────

def fig_model_ladder(path, results_dir="results"):
    """Grouped QLIKE bars showing that topology helps even after a 1-min RV
    control — the paper's key results figure."""
    summ = pd.read_csv(Path(results_dir) / "model_summary.csv")
    ladder = ["HAR-RV", "HAR+RV1", "HAR+RV1+TDA",
              "XGBoost", "XGBoost+RV1", "XGBoost+RV1+TDA"]
    colors = {"HAR-RV": "0.6", "HAR+RV1": CALM, "HAR+RV1+TDA": TURB,
              "XGBoost": "0.6", "XGBoost+RV1": CALM, "XGBoost+RV1+TDA": TURB}
    symbols = list(summ["symbol"].unique())
    fig, axes = plt.subplots(1, len(symbols), figsize=(12, 4.6), sharey=False)
    if len(symbols) == 1:
        axes = [axes]
    for ax, sym in zip(axes, symbols):
        s = summ[summ["symbol"] == sym].set_index("model").reindex(ladder)
        x = np.arange(len(ladder))
        ax.bar(x, s["qlike"].values, color=[colors[m] for m in ladder], edgecolor="k", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(ladder, rotation=35, ha="right", fontsize=8)
        ax.set_title(sym, fontsize=11); ax.set_ylabel("QLIKE (lower = better)")
        lo = s["qlike"].min() * 0.97
        ax.set_ylim(lo, s["qlike"].max() * 1.01)
        # annotate the controlled topology gain
        for grp in [(1, 2), (4, 5)]:
            b, t = s["qlike"].iloc[grp[0]], s["qlike"].iloc[grp[1]]
            ax.annotate(f"−{(b-t)/b*100:.1f}%", xy=(grp[1], t), xytext=(grp[1], t),
                        ha="center", va="bottom", fontsize=8, color=TURB, fontweight="bold")
    fig.suptitle("QLIKE across the model ladder: blue = +1-min RV control, "
                 "red = +TDA on top (topology helps beyond the control)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate TDA pipeline figures")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--turbulent", default="2022-11-09T14:00", help="UTC timestamp")
    ap.add_argument("--calm", default="2023-09-15T12:00", help="UTC timestamp")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tcfg = TDAConfig.from_yaml(cfg)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    close = _load_close(args.symbol, cfg)
    calm_s = _window_series(close, args.calm, tcfg)
    turb_s = _window_series(close, args.turbulent, tcfg)
    calm_cloud, calm_dgms = _diagrams(calm_s, tcfg)
    turb_cloud, turb_dgms = _diagrams(turb_s, tcfg)
    print(f"calm window: {len(calm_cloud)} pts, {len(_finite(calm_dgms[1]))} H1 loops | "
          f"turbulent: {len(turb_cloud)} pts, {len(_finite(turb_dgms[1]))} H1 loops")

    fig_embedding(calm_cloud, turb_cloud, FIGDIR / "1_embedding.png")
    fig_filtration(turb_cloud, turb_dgms, FIGDIR / "2_filtration.png")
    fig_diagrams(calm_dgms, turb_dgms, FIGDIR / "3_diagrams.png")
    fig_barcode(calm_dgms, turb_dgms, FIGDIR / "4_barcode.png")
    fig_landscape(calm_dgms, turb_dgms, FIGDIR / "5_landscape.png")
    fig_persistence_image(calm_dgms, turb_dgms, FIGDIR / "6_persistence_image.png")
    fig_change_vs_rv(args.symbol, cfg, FIGDIR / "7_change_vs_rv.png")
    if (Path("results") / "model_summary.csv").exists():
        fig_model_ladder(FIGDIR / "8_model_ladder.png")

    print(f"Figures written to {FIGDIR}/")


if __name__ == "__main__":
    main()
