"""
tda.py
------
Causal TDA / persistent-homology feature pipeline for high-frequency volatility
forecasting. This is the empirical realisation of the constructions in the
expository paper (REU_Paper.pdf): Takens point cloud → Vietoris–Rips filtration
(§5) → persistent homology (§6) → diagram distances (§7) → vectorization (§8).

PER-WINDOW PIPELINE (one window ends at, and includes, each 5-min RV bar t):

  1. Take the last `window` 1-MINUTE log-returns ending at bar t. Strictly causal:
     uses only returns in (t-window, t], aligned with the 5-min RV grid, so a TDA
     feature at t never peeks past t (same convention as the HAR log-RV lags).
  2. Scale GLOBALLY by a fixed constant (returns → ~percent) for numerical
     conditioning, but do NOT normalize per window. This is deliberate and follows
     Souto (2023): for volatility forecasting the return AMPLITUDE is the signal —
     persistence landscape norms track volatility precisely because windows are not
     rescaled to unit variance. Per-window z-scoring (optional, `standardize=
     "zscore"`) would discard that signal; whether the topological features add
     value *beyond* the HAR log-RV lags is then settled empirically by the
     Diebold–Mariano test and a feature-importance ablation, not by construction.
  3. Time-delay (Takens) embed into R^m with delay tau  →  point cloud (§5.1 bridge
     the paper still needs in its Methodology: time series → point cloud).
  4. Vietoris–Rips filtration over Z_2 up to H_1 (ripser).
  5. Vectorize the diagram into scalar features (§8), and record the diagram so the
     next window can take the Wasserstein/bottleneck distance to the PREVIOUS
     diagram — Souto (2023)'s topological-change signal, justified as a stable
     change detector by the Wasserstein-stability theorem (paper Thm 7.5).

HIGH FREQUENCY (locked requirement): returns are 1-minute and never downsampled.
Windows step every `step` minutes only to emit one TDA row per 5-min RV bar; the
information content of each window is full 1-min resolution.

NO LEAKAGE: every feature at timestamp t is a function of returns in (t-window, t].
The inter-window distance compares diagram(t) to the previous VALID diagram
(<= t). Degenerate (flat/illiquid) windows emit zero-structure features and do not
pollute the change signal.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ripser import ripser

# persim distance fns are optional at import time; we fall back to manual matching.
try:
    from persim import bottleneck as _persim_bottleneck
    from persim import wasserstein as _persim_wasserstein
    _HAS_PERSIM = True
except Exception:  # pragma: no cover
    _HAS_PERSIM = False


# ── Config container ──────────────────────────────────────────────────────────

@dataclass
class TDAConfig:
    embedding_dim:   int   = 3       # m   — delay-embedding dimension
    embedding_delay: int   = 1       # tau — delay in bars (1-min bars)
    window_size:     int   = 60      # minutes of 1-min returns per window
    max_homology_dim: int  = 1       # H0 + H1
    sig_threshold:   float = 0.10    # min lifetime to count as a "significant" loop
    min_distinct:    int   = 8       # < this many distinct returns ⇒ degenerate window
    wasserstein_order: int = 2       # p in the p-Wasserstein change signal
    landscape_grid:  int   = 100     # resolution for landscape-norm integration
    landscape_levels: int  = 5       # number of landscape layers kept for the norm
    ret_clip:        float = 0.20    # clip |1-min log-return| (kills residual bad prints)
    ret_scale:       float = 100.0   # GLOBAL scale (log-ret → ~percent); preserves
                                     #   cross-window volatility scale, conditions numerics
    standardize:     str   = "none"  # "none" (Souto-faithful, scale kept) | "zscore"

    @classmethod
    def from_yaml(cls, cfg: dict) -> "TDAConfig":
        t = cfg.get("tda", {})
        return cls(
            embedding_dim    = t.get("embedding_dim", 3),
            embedding_delay  = t.get("embedding_delay", 1),
            window_size      = t.get("window_size", 60),
            max_homology_dim = t.get("max_homology_dim", 1),
            sig_threshold    = t.get("sig_threshold", 0.10),
            min_distinct     = t.get("min_distinct", 8),
            wasserstein_order= t.get("wasserstein_order", 2),
            ret_scale        = t.get("ret_scale", 100.0),
            standardize      = t.get("standardize", "none"),
        )


# Canonical output columns (order matters for downstream feature lists).
TDA_FEATURES = [
    "tda_wass_h1",            # p-Wasserstein dist to previous H1 diagram  ← headline
    "tda_wass_h0",            # p-Wasserstein dist to previous H0 diagram
    "tda_bottleneck_h1",      # bottleneck dist to previous H1 diagram
    "tda_pers_entropy_h0",    # persistence entropy of H0
    "tda_pers_entropy_h1",    # persistence entropy of H1
    "tda_max_pers_h1",        # most persistent loop (max lifetime in H1)
    "tda_total_pers_h1",      # sum of H1 lifetimes
    "tda_total_pers_h0",      # sum of finite H0 lifetimes
    "tda_n_loops_h1",         # # of H1 features with lifetime > sig_threshold
    "tda_landscape_l2_h0",    # L2 norm of the H0 persistence landscape
    "tda_landscape_l2_h1",    # L2 norm of the H1 persistence landscape
    "tda_degenerate",         # 1.0 if the window was flat/illiquid, else 0.0
]


# ── Low-level geometry / topology helpers ─────────────────────────────────────

def takens_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Time-delay (Takens) embedding of a 1-D series into R^m.

        x_t  ->  (x_t, x_{t-tau}, ..., x_{t-(m-1)tau})

    Returns an (N, m) array of delay vectors, N = len(x) - (m-1)*tau.
    """
    x = np.asarray(x, dtype=float)
    span = (m - 1) * tau
    n = len(x) - span
    if n <= 0:
        return np.empty((0, m))
    # columns are progressively older lags; stack newest→oldest for readability
    cols = [x[span - j * tau : span - j * tau + n] for j in range(m)]
    return np.column_stack(cols)


def _finite(dgm: np.ndarray) -> np.ndarray:
    """Drop the infinite (eternal-component) bar and any NaNs from a diagram."""
    if dgm is None or len(dgm) == 0:
        return np.empty((0, 2))
    m = np.isfinite(dgm).all(axis=1)
    return dgm[m]


def persistence_entropy(dgm: np.ndarray) -> float:
    """
    Persistence entropy: -Σ p_i log p_i with p_i = lifetime_i / Σ lifetimes.
    Higher = topological "mass" spread across many features (disordered);
    lower = dominated by a few persistent features.
    """
    pts = _finite(dgm)
    if len(pts) == 0:
        return 0.0
    life = pts[:, 1] - pts[:, 0]
    life = life[life > 0]
    if life.size == 0:
        return 0.0
    p = life / life.sum()
    return float(-(p * np.log(p)).sum())


def landscape_l2_norm(dgm: np.ndarray, n_grid: int, levels: int) -> float:
    """
    L2 norm of the persistence landscape (paper Def 8.1). Each point (b,d) becomes
    a tent Λ(t)=max(0, min(t-b, d-t)); the landscape's ℓ-th layer is the ℓ-th
    largest tent value at each t. We integrate the first `levels` layers.

    A coordinate-free, scale-aware summary of total topological structure that is
    cheaper and more robust than persim's exact landscape for streaming use.
    """
    pts = _finite(dgm)
    if len(pts) == 0:
        return 0.0
    b, d = pts[:, 0], pts[:, 1]
    lo, hi = float(b.min()), float(d.max())
    if hi <= lo:
        return 0.0
    ts = np.linspace(lo, hi, n_grid)
    # tents: (n_pts, n_grid)
    tents = np.maximum(0.0, np.minimum(ts[None, :] - b[:, None], d[:, None] - ts[None, :]))
    k = min(levels, tents.shape[0])
    # ℓ-th largest across points at each t = top-k after descending sort
    lam = np.sort(tents, axis=0)[::-1][:k]
    dt = ts[1] - ts[0]
    return float(np.sqrt((lam ** 2).sum() * dt))


def _diagram_distance(prev: np.ndarray, cur: np.ndarray, kind: str, order: int) -> float:
    """
    Distance between two persistence diagrams (paper §7), with the eternal bar
    removed and empty-diagram cases handled (a lone diagram is matched entirely
    to the diagonal).
    """
    a, b = _finite(prev), _finite(cur)
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if _HAS_PERSIM:
        try:
            if kind == "bottleneck":
                return float(_persim_bottleneck(a, b))
            return float(_persim_wasserstein(a, b))
        except Exception:
            pass
    # Fallback: total-persistence-to-diagonal proxy (exact when one side is empty).
    def to_diag(x):
        return (x[:, 1] - x[:, 0]) / 2.0 if len(x) else np.array([0.0])
    if kind == "bottleneck":
        return float(max(to_diag(a).max(), to_diag(b).max()))
    return float((to_diag(a).sum() + to_diag(b).sum()))


# ── Per-window feature extraction ─────────────────────────────────────────────

def window_diagrams(returns: np.ndarray, cfg: TDAConfig):
    """
    Standardize → Takens embed → Vietoris–Rips → (dgm_H0, dgm_H1).
    Returns (dgms, degenerate_flag). On a degenerate (flat/illiquid) window the
    diagrams are empty and the flag is True.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    # Degeneracy guard: too few distinct prints (e.g. thin-liquidity flat bars).
    if r.size < (cfg.embedding_dim - 1) * cfg.embedding_delay + 3:
        return [np.empty((0, 2)), np.empty((0, 2))], True
    if np.unique(np.round(r, 12)).size < cfg.min_distinct:
        return [np.empty((0, 2)), np.empty((0, 2))], True
    sd = r.std()
    if sd <= 0 or not np.isfinite(sd):
        return [np.empty((0, 2)), np.empty((0, 2))], True

    if cfg.standardize == "zscore":
        series = (r - r.mean()) / sd             # scale-free (loses volatility signal)
    else:
        series = r * cfg.ret_scale               # global scale kept (Souto-faithful)
    cloud = takens_embedding(series, cfg.embedding_dim, cfg.embedding_delay)
    if len(cloud) < 3:
        return [np.empty((0, 2)), np.empty((0, 2))], True

    dgms = ripser(cloud, maxdim=cfg.max_homology_dim)["dgms"]
    # ripser returns [H0, H1, ...]; ensure two entries
    while len(dgms) < 2:
        dgms.append(np.empty((0, 2)))
    return dgms, False


def diagram_features(dgms, prev_dgms, cfg: TDAConfig) -> dict:
    """Vectorize one window's diagrams + the change vs the previous diagrams."""
    h0, h1 = dgms[0], dgms[1]
    h0f, h1f = _finite(h0), _finite(h1)

    life_h1 = (h1f[:, 1] - h1f[:, 0]) if len(h1f) else np.array([])
    life_h0 = (h0f[:, 1] - h0f[:, 0]) if len(h0f) else np.array([])

    if prev_dgms is None:
        wass_h1 = wass_h0 = bott_h1 = 0.0
    else:
        p = cfg.wasserstein_order
        wass_h1 = _diagram_distance(prev_dgms[1], h1, "wasserstein", p)
        wass_h0 = _diagram_distance(prev_dgms[0], h0, "wasserstein", p)
        bott_h1 = _diagram_distance(prev_dgms[1], h1, "bottleneck", p)

    return {
        "tda_wass_h1":         wass_h1,
        "tda_wass_h0":         wass_h0,
        "tda_bottleneck_h1":   bott_h1,
        "tda_pers_entropy_h0": persistence_entropy(h0),
        "tda_pers_entropy_h1": persistence_entropy(h1),
        "tda_max_pers_h1":     float(life_h1.max()) if life_h1.size else 0.0,
        "tda_total_pers_h1":   float(life_h1.sum()) if life_h1.size else 0.0,
        "tda_total_pers_h0":   float(life_h0.sum()) if life_h0.size else 0.0,
        "tda_n_loops_h1":      float((life_h1 > cfg.sig_threshold).sum()),
        "tda_landscape_l2_h0": landscape_l2_norm(h0, cfg.landscape_grid, cfg.landscape_levels),
        "tda_landscape_l2_h1": landscape_l2_norm(h1, cfg.landscape_grid, cfg.landscape_levels),
        "tda_degenerate":      0.0,
    }


_ZERO_FEATURES = {k: 0.0 for k in TDA_FEATURES}


def _features_for_window(window, prev_dgms, cfg: TDAConfig):
    """Return (features_dict, new_prev_dgms). On a degenerate window the previous
    diagram is carried forward unchanged so illiquidity gaps don't spike the
    change signal."""
    dgms, degenerate = window_diagrams(window, cfg)
    if degenerate:
        feats = dict(_ZERO_FEATURES)
        feats["tda_degenerate"] = 1.0
        return feats, prev_dgms
    return diagram_features(dgms, prev_dgms, cfg), dgms


# ── Parallel workers (chunked over the grid) ──────────────────────────────────

_W: dict = {}


def _init_worker(ts_ns, vals, win_ns, cfg):
    _W.update(ts_ns=ts_ns, vals=vals, win_ns=win_ns, cfg=cfg)


def _window_at(ts_ns, vals, win_ns, t):
    lo = np.searchsorted(ts_ns, t - win_ns, side="right")  # exclude (t-window]
    hi = np.searchsorted(ts_ns, t,           side="right")  # include up to t
    return vals[lo:hi]


def _process_chunk(task):
    """Compute features for a contiguous grid chunk. `prev_ns` is the grid bar
    just before the chunk (or None) so the chunk's first Wasserstein distance is
    still computed against the real previous window."""
    grid_chunk, prev_ns = task
    ts_ns, vals, win_ns, cfg = _W["ts_ns"], _W["vals"], _W["win_ns"], _W["cfg"]
    prev_dgms = None
    if prev_ns is not None:
        dgms, degenerate = window_diagrams(_window_at(ts_ns, vals, win_ns, prev_ns), cfg)
        prev_dgms = None if degenerate else dgms
    rows = []
    for t in grid_chunk:
        feats, prev_dgms = _features_for_window(
            _window_at(ts_ns, vals, win_ns, int(t)), prev_dgms, cfg)
        rows.append(feats)
    return rows


# ── Driver: one TDA row per timestamp on the RV grid ──────────────────────────

def compute_tda_features(
    close_1m: pd.Series,
    grid_index: pd.DatetimeIndex,
    cfg: TDAConfig,
    progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Compute the causal TDA feature matrix, one row per timestamp in `grid_index`
    (the 5-min RV grid). For each t we use 1-min log-returns in (t-window, t].

    `close_1m`  : 1-minute close prices (UTC DatetimeIndex).
    `grid_index`: timestamps to emit features for (must be ⊆ span of close_1m).
    `n_jobs`    : >1 parallelises across contiguous grid chunks. Each chunk carries
                  one boundary window so its first change-signal is still computed
                  against the true previous window; the only approximation is that a
                  degenerate run straddling a chunk boundary may reset the carry-
                  forward (a handful of rows out of hundreds of thousands).
    """
    # 1-min log returns, robustly clipped to kill residual bad single prints.
    logret = np.log(close_1m).diff()
    logret = logret.clip(-cfg.ret_clip, cfg.ret_clip).dropna()

    ts_ns = logret.index.values.astype("datetime64[ns]").astype(np.int64)
    vals  = logret.values.astype(float)
    win_ns = np.int64(cfg.window_size) * 60 * 1_000_000_000

    grid = pd.DatetimeIndex(grid_index)
    grid_ns = grid.values.astype("datetime64[ns]").astype(np.int64)
    n = len(grid_ns)

    def _maybe_tqdm(it, total):
        if not progress:
            return it
        try:
            from tqdm import tqdm
            return tqdm(it, total=total, desc="TDA windows", unit="win")
        except Exception:
            return it

    if n_jobs and n_jobs > 1 and n > 0:
        # Many small chunks → good load balance + progress granularity. Big arrays
        # are shipped to each worker once via the initializer, not per chunk.
        n_chunks = min(max(n_jobs * 8, 1), n)
        bounds = np.linspace(0, n, n_chunks + 1).astype(int)
        tasks = []
        for i in range(n_chunks):
            s, e = int(bounds[i]), int(bounds[i + 1])
            if s == e:
                continue
            prev_ns = int(grid_ns[s - 1]) if s > 0 else None
            tasks.append((grid_ns[s:e], prev_ns))

        rows = []
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker,
                                 initargs=(ts_ns, vals, win_ns, cfg)) as ex:
            for chunk_rows in _maybe_tqdm(ex.map(_process_chunk, tasks), total=len(tasks)):
                rows.extend(chunk_rows)
    else:
        prev_dgms = None
        rows = []
        for t in _maybe_tqdm(grid_ns, total=n):
            feats, prev_dgms = _features_for_window(
                _window_at(ts_ns, vals, win_ns, int(t)), prev_dgms, cfg)
            rows.append(feats)

    out = pd.DataFrame(rows, index=grid, columns=TDA_FEATURES)
    out.index.name = close_1m.index.name or "timestamp"
    return out
