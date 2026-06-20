#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
visualize_poincare.py
=====================

Universal visualizer for the Poincaré ball(s) learned by the HypCD family of
models (HypGCD / HypSimGCD / HypSelEx).

Given a trained checkpoint, this tool maps a dataset split into the model's
hyperbolic space(s) and produces a set of figures + scalar diagnostics that
help answer two questions:

    1. Is a *hierarchy* preserved in the ball?
       In a Poincaré ball, "depth" (geodesic distance from the origin) encodes
       granularity: general / mixed concepts sit near the centre, specific
       concepts sit near the boundary.  We therefore look at the radial
       distribution of points and class-means, build a dendrogram of the
       class-means under the *hyperbolic* metric (with its cophenetic
       correlation), and report the Gromov delta-hyperbolicity of the
       embedding.

    2. Are the points *uniformly* spread in the ball?
       We look at the radial histogram, the pairwise hyperbolic-distance
       histogram, the distribution of pairwise direction-cosines, the mean
       resultant length of the directions, and a hyperbolic uniformity loss.

Design notes
------------
* Method-agnostic w.r.t. the *training* recipe (frozen / unfrozen backbone,
  deterministic head, learnable vs. fixed curvature, etc.).  All that matters
  for visualisation is the saved weights + the geometry, which we reconstruct
  faithfully using the repo's own ``hyptorch`` math.
* Works for the three "original" variants out of the box:
      - gcd    : backbone -> DINOHead2 -> ToPoincare                (rep ball)
      - simgcd : backbone -> ToPoincare; ToPoincare -> HypLinear    (rep + cls balls)
      - selex  : backbone -> DINOHead   -> ToPoincare               (rep ball)
* The representation ball is visualised for every method.  The classification
  ball is visualised whenever a classifier checkpoint is present (SimGCD).
* Dimensions and (optionally) a learnable curvature are *inferred from the
  checkpoint* when possible, so you do not have to remember the exact dims.
* Single-branch only for now (no multi-factor / multi-curvature heads); the
  builders are kept in a small registry that is trivial to extend later.

Typical usage
-------------
    # SimGCD (gives both representation and classification balls)
    python -m visualize_poincare \
        --method simgcd --dataset_name cub --dino v1 \
        --model_path /path/to/checkpoints/model_best.pt \
        --c 0.1 --cr 2.0 --out_dir ./poincare_viz/cub_simgcd

    # GCD
    python -m visualize_poincare \
        --method gcd --dataset_name scars --dino v1 \
        --model_path /path/to/checkpoints/model_best.pt \
        --c 0.1 --cr 1.2 --hyp_dim 256 --out_dir ./poincare_viz/scars_gcd

    # SelEx
    python -m visualize_poincare \
        --method selex --dataset_name aircraft --dino v1 \
        --model_path /path/to/checkpoints/model_best.pt \
        --c 0.1 --cr 1.5 --mlp_out_dim 8192 --out_dir ./poincare_viz/air_selex

The sibling checkpoints (``*_proj_head*.pt`` and ``*_hyp_cls*.pt``) are derived
automatically from ``--model_path`` following the trainer's naming convention,
but can be overridden with ``--proj_head_path`` / ``--classifier_path``.

Outputs (per ball, prefixed with the ball name "rep" / "cls"):
    <name>_overview.png    radial + directional + 2D-disk views
    <name>_hierarchy.png   class-mean depth / dendrogram / distance heatmap
    <name>_metrics.json    all scalar diagnostics
    <name>_report.txt      short human-readable summary
    <name>_embedding.npz   raw arrays (points, labels, preds, is_old, c, ...)
"""

import os
import sys
import json
import math
import argparse
import warnings

import numpy as np

# matplotlib without a display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Make the HypCD package importable no matter where this file is placed
# (repo root, a `visualize/` subfolder, etc.) and no matter the CWD.  We search
# upward from the script's directory for the repo root -- the folder that holds
# both `hyptorch/` and `models/` -- and put it on sys.path.
# ---------------------------------------------------------------------------
def _bootstrap_repo_path():
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    cur = here
    for _ in range(8):  # walk up a few levels looking for the repo root
        if (os.path.isdir(os.path.join(cur, "hyptorch"))
                and os.path.isdir(os.path.join(cur, "models"))):
            if cur not in sys.path:
                sys.path.insert(0, cur)
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


_REPO_ROOT = _bootstrap_repo_path()

# Repo geometry + heads (pure-torch, importable without the data stack).
try:
    import hyptorch.pmath as pmath
    import hyptorch.nn as hypnn
    from models import vision_transformer as vits1
except ModuleNotFoundError as _e:
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.stderr.write(
        "\n[visualize_poincare] ERROR: could not import the HypCD package "
        f"({_e}).\n"
        "  This script needs to see the repo's `hyptorch/` and `models/` "
        "folders.\n"
        f"  - script location   : {_here}\n"
        f"  - auto-detected repo : {_REPO_ROOT}\n"
        "  Fix one of:\n"
        "    * keep this file inside the HypCD repo (root or any subfolder), or\n"
        "    * run it with the repo on PYTHONPATH, e.g.\n"
        "        PYTHONPATH=/data/projects/HypCD python draw_poincare.py ...\n\n")
    raise


def delta_hyp(dismat):
    """Gromov delta-hyperbolicity from a distance matrix.

    This is a verbatim, dependency-free copy of ``hyptorch.delta.delta_hyp``
    (that module imports torchvision at the top for an unrelated VGG helper, so
    we replicate just this pure-NumPy routine to keep the visualizer importable
    without the full data/vision stack).
    """
    p = 0
    row = dismat[p, :][np.newaxis, :]
    col = dismat[:, p][:, np.newaxis]
    XY_p = 0.5 * (row + col - dismat)
    maxmin = np.max(np.minimum(XY_p[:, :, None], XY_p[None, :, :]), axis=1)
    return np.max(maxmin - XY_p)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_tensor(x, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(x, dtype=dtype)


def ball_radius(c):
    """Euclidean radius of the Poincaré ball for curvature c (= 1/sqrt(c))."""
    return float("inf") if c <= 0 else 1.0 / math.sqrt(c)


# ----------------------------------------------------------------------------
# Hyperbolic geometry helpers (thin wrappers over the repo's pmath, so the
# curvature handling matches training exactly)
# ----------------------------------------------------------------------------
def depth_from_origin(P, c):
    """Geodesic distance of each row of P from the origin. P: (n,d) tensor."""
    return to_np(pmath.dist0(as_tensor(P), c=c))


def tangent_at_origin(P, c):
    """logmap0(P): tangent vectors at the origin. ||row|| == depth_from_origin."""
    return to_np(pmath.logmap0(as_tensor(P), c=c))


def hyp_dist_pairwise(X, Y, c, chunk=512):
    """Pairwise Poincaré distance between rows of X (a,d) and Y (b,d) -> (a,b).

    Uses the closed form

        d_c(x,y) = (1/sqrt(c)) * arccosh( 1 + 2c||x-y||^2 /
                                          ((1 - c||x||^2)(1 - c||y||^2)) )

    which is mathematically identical to the repo's
    ``(2/sqrt(c)) * artanh(sqrt(c) ||(-x) (+)_c y||)`` but only ever forms
    ``(chunk, b)`` intermediates.  The repo's ``dist_matrix`` instead builds a
    full ``(a, b, d)`` Mobius-addition tensor, which explodes for high-dim balls
    (e.g. SelEx's 8192-d ball) -> hundreds of GB.  We therefore compute it here
    in a memory-safe, chunked way.  All math is done in float64.
    """
    Xt = to_np(X).astype(np.float64)
    Yt = to_np(Y).astype(np.float64)
    a = Xt.shape[0]
    if c <= 0:   # degenerate / Euclidean fallback (not used by the HypCD models)
        out = np.zeros((a, Yt.shape[0]), dtype=np.float64)
        for i in range(0, a, chunk):
            j = min(i + chunk, a)
            diff = Xt[i:j, None, :] - Yt[None, :, :]
            out[i:j] = np.linalg.norm(diff, axis=-1)
        return out
    xn = np.sum(Xt * Xt, axis=1)                  # (a,)
    yn = np.sum(Yt * Yt, axis=1)                  # (b,)
    inv = 1.0 / math.sqrt(c)
    dx = np.clip(1.0 - c * xn, 1e-12, None)       # (a,)  (1 - c||x||^2) > 0
    dy = np.clip(1.0 - c * yn, 1e-12, None)       # (b,)
    out = np.empty((a, Yt.shape[0]), dtype=np.float64)
    for i in range(0, a, chunk):
        j = min(i + chunk, a)
        # ||x-y||^2 via the gram trick: (chunk,b)
        g = Xt[i:j] @ Yt.T
        sq = xn[i:j, None] - 2.0 * g + yn[None, :]
        sq = np.clip(sq, 0.0, None)
        denom = dx[i:j, None] * dy[None, :]
        arg = 1.0 + 2.0 * c * sq / denom
        arg = np.clip(arg, 1.0, None)             # arccosh domain
        out[i:j] = inv * np.arccosh(arg)
    return out


def hyp_distance_matrix(P, c, chunk=512):
    """Full symmetric pairwise hyperbolic distance matrix for P (n,d) -> (n,n)."""
    Pp = to_np(pmath.project(as_tensor(P), c=c))
    out = hyp_dist_pairwise(Pp, Pp, c, chunk=chunk)
    out = 0.5 * (out + out.T)                     # kill tiny numerical asymmetry
    np.fill_diagonal(out, 0.0)
    return out


def poincare_class_means(P, labels, c):
    """Hyperbolic (Einstein/Klein) mean per class.

    Returns (means [K,d] tensor, class_ids [K] int array). Falls back to a
    tangent-space mean if the Klein-model mean is numerically unstable.
    """
    Pt = pmath.project(as_tensor(P), c=c)
    classes = np.unique(labels)
    means = []
    for k in classes:
        idx = np.where(labels == k)[0]
        pts = Pt[idx]
        if pts.shape[0] == 1:
            m = pts[0]
        else:
            try:
                m = pmath.poincare_mean(pts, dim=0, c=c)
                if not torch.isfinite(m).all():
                    raise ValueError("non-finite poincare_mean")
            except Exception:
                # tangent-space mean: expmap0(mean(logmap0(x)))
                t = pmath.logmap0(pts, c=c).mean(dim=0, keepdim=True)
                m = pmath.expmap0(t, c=c).squeeze(0)
            m = pmath.project(m.unsqueeze(0), c=c).squeeze(0)
        means.append(m)
    return torch.stack(means, dim=0), classes


def global_poincare_mean(P, c):
    Pt = pmath.project(as_tensor(P), c=c)
    try:
        m = pmath.poincare_mean(Pt, dim=0, c=c)
        if not torch.isfinite(m).all():
            raise ValueError
    except Exception:
        t = pmath.logmap0(Pt, c=c).mean(dim=0, keepdim=True)
        m = pmath.expmap0(t, c=c).squeeze(0)
    return pmath.project(m.unsqueeze(0), c=c).squeeze(0)


# ----------------------------------------------------------------------------
# 2D embedding for plotting (origin-preserving)
# ----------------------------------------------------------------------------
def embed_to_disk(P, c, method="pca", extra=None, seed=0):
    """Project high-dim Poincaré points to a *unit* 2D Poincaré disk.

    The pipeline is:  P --logmap0--> tangent vectors --(linear 2D)--> u
                        --expmap0(c)--> 2D ball --(/R)--> unit disk.

    For ``method='pca'`` the linear step is an origin-preserving truncated SVD
    (no mean-centering), so the disk centre coincides with the true origin and
    the radial coordinate stays meaningful as "depth".  ``method='tsne'`` uses
    sklearn t-SNE on the tangent vectors and then linearly fills the disk.

    Parameters
    ----------
    extra : optional (m,d) array of *additional* points (e.g. class means) to
        embed with the SAME transform, so they live in the same frame.

    Returns
    -------
    xy        : (n,2) unit-disk coordinates of P
    xy_extra  : (m,2) unit-disk coordinates of `extra` (or None)
    info      : dict with the method actually used
    """
    T = tangent_at_origin(P, c)                       # (n,d)
    T_extra = tangent_at_origin(extra, c) if extra is not None else None
    R = ball_radius(c)

    method = method.lower()
    used = method
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            n = T.shape[0]
            perp = max(5, min(30, (n - 1) // 3))
            u = TSNE(n_components=2, perplexity=perp, init="pca",
                     random_state=seed).fit_transform(T)
            # t-SNE has no notion of an origin: linearly scale into the disk.
            scale = np.percentile(np.linalg.norm(u, axis=1), 99) + 1e-9
            xy = (u / scale) * 0.95
            xy = np.clip(xy, -0.999, 0.999)
            # We cannot transform `extra` consistently under t-SNE -> embed
            # them jointly if provided.
            xy_extra = None
            if T_extra is not None:
                joint = np.concatenate([T, T_extra], axis=0)
                uj = TSNE(n_components=2, perplexity=perp, init="pca",
                          random_state=seed).fit_transform(joint)
                sj = np.percentile(np.linalg.norm(uj, axis=1), 99) + 1e-9
                uj = np.clip((uj / sj) * 0.95, -0.999, 0.999)
                xy, xy_extra = uj[:len(T)], uj[len(T):]
            return xy, xy_extra, {"embed": "tsne"}
        except Exception as e:
            log(f"[warn] t-SNE unavailable/failed ({e}); falling back to PCA.")
            used = "pca"

    # ---- origin-preserving truncated SVD (no centering) ----
    from sklearn.decomposition import TruncatedSVD
    k = min(2, T.shape[1])
    svd = TruncatedSVD(n_components=max(k, 1), random_state=seed)
    u = svd.fit_transform(T)                          # (n,k)  == T @ V^T
    if u.shape[1] < 2:                                # pad if d == 1
        u = np.concatenate([u, np.zeros((u.shape[0], 1))], axis=1)
    u_extra = None
    if T_extra is not None:
        u_extra = svd.transform(T_extra)
        if u_extra.shape[1] < 2:
            u_extra = np.concatenate([u_extra, np.zeros((u_extra.shape[0], 1))], axis=1)

    def _expmap_unit(arr):
        p2 = to_np(pmath.expmap0(as_tensor(arr[:, :2]), c=c))   # disk radius R
        if math.isfinite(R):
            p2 = p2 / R                                          # -> unit disk
        nrm = np.linalg.norm(p2, axis=1, keepdims=True)
        p2 = np.where(nrm > 0.999, p2 / (nrm + 1e-9) * 0.999, p2)
        return p2

    xy = _expmap_unit(u)
    xy_extra = _expmap_unit(u_extra) if u_extra is not None else None
    return xy, xy_extra, {"embed": used,
                          "explained_variance_ratio": [float(x) for x in
                                                       getattr(svd, "explained_variance_ratio_", [])[:2]]}


# ----------------------------------------------------------------------------
# Scalar diagnostics
# ----------------------------------------------------------------------------
def radial_stats(P, c):
    d0 = depth_from_origin(P, c)
    enorm = to_np(torch.linalg.norm(as_tensor(P), dim=-1))
    R = ball_radius(c)
    rel = enorm / R if math.isfinite(R) else enorm
    return {
        "depth_mean": float(np.mean(d0)), "depth_std": float(np.std(d0)),
        "depth_min": float(np.min(d0)), "depth_max": float(np.max(d0)),
        "eucl_norm_mean": float(np.mean(enorm)), "eucl_norm_max": float(np.max(enorm)),
        "ball_radius": (None if not math.isfinite(R) else R),
        "rel_norm_mean": float(np.mean(rel)), "rel_norm_max": float(np.max(rel)),
        "frac_near_boundary_0.9R": float(np.mean(rel > 0.9)),
    }, d0


def directional_uniformity(P, sample_idx, rng):
    """Direction-based uniformity: mean resultant length + pairwise cosines."""
    X = to_np(P)[sample_idx]
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    keep = (nrm[:, 0] > 1e-8)
    dirs = X[keep] / nrm[keep]
    if dirs.shape[0] < 3:
        return {"mean_resultant_length": None, "cos_abs_mean": None}, np.array([])
    # mean resultant length of the directions (0 = perfectly spread, 1 = collapsed)
    mrl = float(np.linalg.norm(dirs.mean(axis=0)))
    # sampled pairwise cosines
    m = dirs.shape[0]
    pairs = min(20000, m * (m - 1) // 2)
    ii = rng.integers(0, m, size=pairs)
    jj = rng.integers(0, m, size=pairs)
    ok = ii != jj
    cos = np.sum(dirs[ii[ok]] * dirs[jj[ok]], axis=1)
    return {"mean_resultant_length": mrl,
            "cos_abs_mean": float(np.mean(np.abs(cos))),
            "cos_mean": float(np.mean(cos)), "cos_std": float(np.std(cos))}, cos


def hyperbolic_uniformity(D, t=1.0):
    """Wang&Isola-style uniformity on the hyperbolic distance matrix D (n,n).

    U = log E_{i != j} exp(-t * d^2).  More negative => more uniform.
    Also returns simple pairwise-distance summaries (often more readable).
    """
    iu = np.triu_indices(D.shape[0], k=1)
    d = D[iu]
    if d.size == 0:
        return {"uniformity_loss": None, "pair_dist_mean": None}, d
    val = float(np.log(np.mean(np.exp(-t * (d ** 2))) + 1e-12))
    return {"uniformity_loss_t1": val,
            "pair_dist_mean": float(np.mean(d)),
            "pair_dist_std": float(np.std(d)),
            "pair_dist_median": float(np.median(d))}, d


def alignment_separation(P, labels, means, mean_ids, c, sample_per_class=200, rng=None):
    """Intra-class spread (alignment) vs. inter-class-mean spread (separation)."""
    rng = rng or np.random.default_rng(0)
    Pp = to_np(pmath.project(as_tensor(P), c=c))
    Mp = to_np(pmath.project(means, c=c))
    intra = []
    for ki, k in enumerate(mean_ids):
        idx = np.where(labels == k)[0]
        if idx.size < 2:
            continue
        if idx.size > sample_per_class:
            idx = rng.choice(idx, size=sample_per_class, replace=False)
        d = hyp_dist_pairwise(Pp[idx], Mp[ki:ki + 1], c)  # (n,1) dist to own mean
        intra.append(d.ravel())
    intra = np.concatenate(intra) if intra else np.array([0.0])
    if means.shape[0] >= 2:
        Dm = hyp_distance_matrix(means, c)
        iu = np.triu_indices(Dm.shape[0], k=1)
        sep = Dm[iu]
    else:
        sep = np.array([0.0])
    a = float(np.mean(intra)); s = float(np.mean(sep))
    return {"alignment_intra_mean": a,
            "separation_intermean_mean": s,
            "separation_over_alignment": float(s / (a + 1e-9))}


def delta_hyperbolicity_rel(D, n_tries=8, batch_size=512, rng=None):
    """Relative Gromov delta on the (hyperbolic) distance matrix D.

    delta_rel = 2 * delta / diam, averaged over random sub-samples.
    Lower => more tree-like (more hierarchical metric structure).

    NOTE: ``delta_hyp`` forms an (m, m, m) array internally, so the per-try
    sub-sample size ``m`` is capped (``batch_size``) and the sub-matrix is cast
    to float32 -- m=512 -> ~0.5 GB, whereas using the full 2000-point matrix
    would need ~64 GB.  We compensate with several random tries.
    """
    rng = rng or np.random.default_rng(0)
    n = D.shape[0]
    if n < 4:
        return {"delta_rel_mean": None, "delta_rel_std": None, "delta_batch": None}
    bs = int(min(batch_size, n))
    vals = []
    for _ in range(n_tries):
        idx = rng.choice(n, size=bs, replace=False)
        sub = D[np.ix_(idx, idx)].astype(np.float32)
        diam = float(np.max(sub))
        if diam <= 0:
            continue
        vals.append(2.0 * float(delta_hyp(sub)) / diam)
    if not vals:
        return {"delta_rel_mean": None, "delta_rel_std": None, "delta_batch": bs}
    return {"delta_rel_mean": float(np.mean(vals)),
            "delta_rel_std": float(np.std(vals)),
            "delta_batch": bs, "delta_n_tries": len(vals)}


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def _draw_disk_axes(ax):
    circ = plt.Circle((0, 0), 1.0, fill=False, color="0.4", lw=1.2, zorder=1)
    ax.add_patch(circ)
    ax.plot(0, 0, "+", color="0.3", ms=8, mew=1.5, zorder=2)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _class_palette(n):
    if n <= 10:
        return plt.get_cmap("tab10")(np.linspace(0, 1, 10))[:n]
    if n <= 20:
        return plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:n]
    return plt.get_cmap("gist_rainbow")(np.linspace(0, 1, n))


def plot_overview(name, title, xy, xy_means, d0, labels, is_old, c,
                  cos_samples, pair_d, out_path, embed_info):
    K = int(np.unique(labels).size)
    R = ball_radius(c)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 11))

    # (0,0) disk colored by class
    ax = axes[0, 0]
    _draw_disk_axes(ax)
    pal = _class_palette(K)
    cl_to_i = {cl: i for i, cl in enumerate(np.unique(labels))}
    colors = pal[[cl_to_i[l] for l in labels]]
    ax.scatter(xy[:, 0], xy[:, 1], s=7, c=colors, alpha=0.6, linewidths=0, zorder=3)
    if xy_means is not None:
        ax.scatter(xy_means[:, 0], xy_means[:, 1], s=42, facecolors="none",
                   edgecolors="k", linewidths=1.0, zorder=5)
    ax.set_title(f"{title}\n2D disk · colored by class (K={K})", fontsize=11)
    if K <= 12:
        handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=pal[i],
                          markersize=7, label=str(cl)) for cl, i in cl_to_i.items()]
        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  framealpha=0.6, ncol=2)

    # (0,1) disk colored by depth
    ax = axes[0, 1]
    _draw_disk_axes(ax)
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=7, c=d0, cmap="viridis",
                    alpha=0.75, linewidths=0, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("geodesic depth  d(0,x)", fontsize=9)
    ax.set_title("2D disk · colored by depth (radius = hierarchy)", fontsize=11)

    # (0,2) disk old vs new
    ax = axes[0, 2]
    _draw_disk_axes(ax)
    ax.scatter(xy[~is_old, 0], xy[~is_old, 1], s=7, c="#ff7f0e", alpha=0.55,
               linewidths=0, label="novel (unseen)", zorder=3)
    ax.scatter(xy[is_old, 0], xy[is_old, 1], s=7, c="#1f77b4", alpha=0.55,
               linewidths=0, label="known (seen)", zorder=3)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.7)
    ax.set_title("2D disk · known vs. novel classes", fontsize=11)

    # (1,0) radial depth histogram
    ax = axes[1, 0]
    bins = np.linspace(float(np.min(d0)), float(np.max(d0) + 1e-9), 40)
    ax.hist(d0[is_old], bins=bins, alpha=0.55, color="#1f77b4",
            density=True, label="known")
    ax.hist(d0[~is_old], bins=bins, alpha=0.55, color="#ff7f0e",
            density=True, label="novel")
    ax.set_xlabel("geodesic depth  d(0,x)"); ax.set_ylabel("density")
    ax.set_title("Radial distribution (hierarchy / radial uniformity)", fontsize=11)
    ax.legend(fontsize=9)

    # (1,1) pairwise hyperbolic distance histogram
    ax = axes[1, 1]
    if pair_d.size:
        ax.hist(pair_d, bins=40, color="#4c72b0", alpha=0.85, density=True)
        ax.axvline(np.mean(pair_d), color="k", ls="--", lw=1,
                   label=f"mean={np.mean(pair_d):.2f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("pairwise hyperbolic distance"); ax.set_ylabel("density")
    ax.set_title("Pairwise distances (spread / uniformity)", fontsize=11)

    # (1,2) pairwise direction-cosine histogram
    ax = axes[1, 2]
    if cos_samples.size:
        ax.hist(cos_samples, bins=40, color="#55a868", alpha=0.85, density=True)
        ax.axvline(0.0, color="k", ls=":", lw=1)
        ax.axvline(float(np.mean(cos_samples)), color="k", ls="--", lw=1,
                   label=f"mean={np.mean(cos_samples):.2f}")
        ax.legend(fontsize=9)
    ax.set_xlabel("cosine between directions (x/‖x‖)")
    ax.set_ylabel("density")
    ax.set_title("Angular spread (0 ⇒ uniform directions)", fontsize=11)

    rr = "" if R == float("inf") else f" · ball radius={R:.3f}"
    fig.suptitle(f"[{name}] Poincaré ball overview · c={c:g}{rr} · "
                 f"embed={embed_info.get('embed','pca')}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_hierarchy(name, title, means, mean_ids, num_labeled, c, out_path):
    """Per-class depth, dendrogram and ordered distance heatmap of class means."""
    from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
    from scipy.spatial.distance import squareform

    K = means.shape[0]
    d0_means = depth_from_origin(means, c)
    is_old_mean = mean_ids < num_labeled
    Dm = hyp_distance_matrix(means, c)

    coph = None
    Z = None
    if K >= 3:
        condensed = squareform(Dm, checks=False)
        Z = linkage(condensed, method="average")
        try:
            coph = float(cophenet(Z, condensed)[0])
        except Exception:
            coph = None

    fig = plt.figure(figsize=(17, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 1.1], wspace=0.28)

    # (a) sorted per-class mean depth
    ax = fig.add_subplot(gs[0, 0])
    order = np.argsort(d0_means)
    cols = np.where(is_old_mean[order], "#1f77b4", "#ff7f0e")
    ax.bar(np.arange(K), d0_means[order], color=cols, width=1.0)
    ax.set_xlabel("class (sorted by depth)")
    ax.set_ylabel("class-mean depth  d(0, μ_k)")
    ax.set_title("Per-class depth gradient", fontsize=11)
    ax.legend(handles=[Line2D([0], [0], color="#1f77b4", lw=6, label="known"),
                       Line2D([0], [0], color="#ff7f0e", lw=6, label="novel")],
              fontsize=8, loc="upper left")

    # (b) dendrogram
    ax = fig.add_subplot(gs[0, 1])
    if Z is not None:
        no_labels = K > 40
        dendrogram(Z, ax=ax, color_threshold=None,
                   no_labels=no_labels,
                   labels=None if no_labels else [str(int(m)) for m in mean_ids])
        ttl = "Dendrogram of class-means (hyperbolic)"
        if coph is not None:
            ttl += f"\ncophenetic corr = {coph:.3f}"
        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel("merge distance")
        if not no_labels:
            ax.tick_params(axis="x", labelsize=6)
    else:
        ax.text(0.5, 0.5, "need >=3 classes", ha="center", va="center")
        ax.set_axis_off()

    # (c) ordered distance heatmap
    ax = fig.add_subplot(gs[0, 2])
    if Z is not None:
        leaves = dendrogram(Z, no_plot=True)["leaves"]
        Dord = Dm[np.ix_(leaves, leaves)]
    else:
        Dord = Dm
    im = ax.imshow(Dord, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "hyperbolic distance", fontsize=9)
    ax.set_title("Class-mean distance matrix\n(ordered by dendrogram)", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"[{name}] Hierarchy view · {title}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {"cophenetic_corr": coph,
            "class_mean_depth_mean": float(np.mean(d0_means)),
            "class_mean_depth_std": float(np.std(d0_means)),
            "class_mean_depth_known_mean": float(np.mean(d0_means[is_old_mean])) if is_old_mean.any() else None,
            "class_mean_depth_novel_mean": float(np.mean(d0_means[~is_old_mean])) if (~is_old_mean).any() else None}


# ----------------------------------------------------------------------------
# The per-ball analysis driver
# ----------------------------------------------------------------------------
def analyze_ball(name, P, labels, is_old, c, num_labeled, out_dir, args,
                 preds=None, title_extra=""):
    """Run all figures + metrics for one Poincaré point cloud P (n,d)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n, d = P.shape
    log(f"\n=== Analyzing '{name}' ball: n={n}, dim={d}, c={c:g} ===")

    P = pmath.project(as_tensor(P), c=c)   # safety: keep strictly inside the ball
    labels = np.asarray(labels)
    is_old = np.asarray(is_old, dtype=bool)

    # ---- class means + global mean ----
    means, mean_ids = poincare_class_means(P, labels, c)
    gmean = global_poincare_mean(P, c)
    gdepth = float(depth_from_origin(gmean.unsqueeze(0), c)[0])

    # ---- 2D embedding (points + class means in the same frame) ----
    emb_idx = np.arange(n)
    if n > args.embed_sample:
        emb_idx = rng.choice(n, size=args.embed_sample, replace=False)
    xy, xy_means, emb_info = embed_to_disk(
        to_np(P)[emb_idx], c, method=args.embed,
        extra=to_np(means), seed=args.seed)

    # ---- radial / directional / pairwise diagnostics ----
    rstats, d0 = radial_stats(P, c)
    metric_idx = np.arange(n)
    if n > args.metric_sample:
        metric_idx = rng.choice(n, size=args.metric_sample, replace=False)
    Psamp = to_np(P)[metric_idx]
    dstats, cos_samples = directional_uniformity(P, metric_idx, rng)
    Dsamp = hyp_distance_matrix(Psamp, c)
    ustats, pair_d = hyperbolic_uniformity(Dsamp, t=args.uniformity_t)
    alsep = alignment_separation(P, labels, means, mean_ids, c, rng=rng)
    deltas = delta_hyperbolicity_rel(Dsamp, n_tries=args.delta_tries,
                                     batch_size=args.delta_batch, rng=rng)

    title = (f"{args.method}/{args.dataset_name}/dino-{args.dino}"
             + (f" · {title_extra}" if title_extra else ""))

    # ---- figures ----
    overview_png = os.path.join(out_dir, f"{name}_overview.png")
    plot_overview(name, title,
                  xy, xy_means, d0[emb_idx], labels[emb_idx], is_old[emb_idx],
                  c, cos_samples, pair_d, overview_png, emb_info)
    log(f"  wrote {overview_png}")

    hier_png = os.path.join(out_dir, f"{name}_hierarchy.png")
    hstats = plot_hierarchy(name, title, means, mean_ids, num_labeled, c, hier_png)
    log(f"  wrote {hier_png}")

    # ---- metrics json ----
    metrics = {
        "name": name, "method": args.method, "dataset": args.dataset_name,
        "dino": args.dino, "split": args.split,
        "n_points": int(n), "dim": int(d), "n_classes": int(np.unique(labels).size),
        "n_known_classes": int(num_labeled),
        "curvature_c": float(c), "ball_radius": rstats["ball_radius"],
        "global_mean_depth": gdepth,
        "embedding": emb_info,
        "radial": rstats,
        "directional_uniformity": dstats,
        "distance_uniformity": ustats,
        "alignment_separation": alsep,
        "delta_hyperbolicity": deltas,
        "hierarchy": hstats,
        "predicted_acc_on_split": _maybe_acc(preds, labels),
    }
    mpath = os.path.join(out_dir, f"{name}_metrics.json")
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"  wrote {mpath}")

    # ---- raw arrays ----
    if args.save_embeddings:
        npz = os.path.join(out_dir, f"{name}_embedding.npz")
        np.savez_compressed(
            npz, points=to_np(P).astype(np.float32), labels=labels,
            is_old=is_old, preds=(np.asarray(preds) if preds is not None else np.array([])),
            class_means=to_np(means).astype(np.float32), class_mean_ids=mean_ids,
            curvature_c=np.array([c], dtype=np.float32))
        log(f"  wrote {npz}")

    # ---- short human-readable report ----
    rep = _build_report(name, metrics)
    rpath = os.path.join(out_dir, f"{name}_report.txt")
    with open(rpath, "w") as f:
        f.write(rep)
    log("  --- summary ---\n" + "\n".join("  " + l for l in rep.splitlines()))
    return metrics


def _maybe_acc(preds, labels):
    if preds is None or len(preds) == 0:
        return None
    preds = np.asarray(preds)
    if preds.shape[0] != labels.shape[0]:
        return None
    return float(np.mean(preds == labels))


def _build_report(name, m):
    R = m["radial"]
    du = m["directional_uniformity"]
    su = m["distance_uniformity"]
    al = m["alignment_separation"]
    dl = m["delta_hyperbolicity"]
    h = m["hierarchy"]
    lines = []
    lines.append(f"Poincaré-ball report : {name}")
    lines.append(f"  model            : {m['method']} | {m['dataset']} | dino-{m['dino']} | split={m['split']}")
    lines.append(f"  points / dim      : {m['n_points']} / {m['dim']}   classes={m['n_classes']} (known={m['n_known_classes']})")
    lines.append(f"  curvature c       : {m['curvature_c']:g}   ball radius={m['ball_radius']}")
    lines.append("")
    lines.append("  [Hierarchy]")
    lines.append(f"    depth  global-mean={m['global_mean_depth']:.3f}  <  class-means={h['class_mean_depth_mean']:.3f}  <  points={R['depth_mean']:.3f}")
    lines.append(f"      (increasing depth from mixed centre -> classes -> instances suggests a hierarchy)")
    if h.get("class_mean_depth_known_mean") is not None and h.get("class_mean_depth_novel_mean") is not None:
        lines.append(f"    class-mean depth  known={h['class_mean_depth_known_mean']:.3f}  novel={h['class_mean_depth_novel_mean']:.3f}")
    if h.get("cophenetic_corr") is not None:
        lines.append(f"    dendrogram cophenetic corr = {h['cophenetic_corr']:.3f}   (closer to 1 = cleaner tree on class-means)")
    if dl.get("delta_rel_mean") is not None:
        lines.append(f"    Gromov delta_rel = {dl['delta_rel_mean']:.3f} ± {dl['delta_rel_std']:.3f}   (lower = more tree-like / hierarchical)")
    lines.append("")
    lines.append("  [Uniformity]")
    lines.append(f"    radial: mean rel-norm={R['rel_norm_mean']:.3f}, frac>0.9R={R['frac_near_boundary_0.9R']:.3f}  (≈0 ⇒ hugging centre, ≈1 ⇒ hugging boundary)")
    if du.get("mean_resultant_length") is not None:
        lines.append(f"    angular: mean resultant length={du['mean_resultant_length']:.3f}  (0 ⇒ directions uniform, 1 ⇒ collapsed); |cos| mean={du['cos_abs_mean']:.3f}")
    if su.get("pair_dist_mean") is not None:
        lines.append(f"    pairwise hyp-dist mean={su['pair_dist_mean']:.3f} ± {su['pair_dist_std']:.3f}; uniformity loss(t=1)={su.get('uniformity_loss_t1')}")
    lines.append(f"    separation/alignment = {al['separation_over_alignment']:.3f}  (intra={al['alignment_intra_mean']:.3f}, inter-mean={al['separation_intermean_mean']:.3f}; higher ⇒ tighter, better separated clusters)")
    if m.get("predicted_acc_on_split") is not None:
        lines.append("")
        lines.append(f"  [Classifier] argmax accuracy on this split = {m['predicted_acc_on_split']:.4f}")
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Model building (per-method registry) + checkpoint loading
# ----------------------------------------------------------------------------
def _build_backbone(dino, feat_dim=768):
    if dino in ("v1", "vit_dino", "dino", "vits1"):
        return vits1.__dict__["vit_base"]()
    elif dino in ("v2", "dinov2", "vits2"):
        from models import vision_transformer2 as vits2
        return vits2.__dict__["vit_base"]()
    raise ValueError(f"Unknown dino version: {dino!r} (use 'v1' or 'v2')")


def _make_topoincare(c, ball_dim, cr):
    clip_r = cr if (cr is not None and cr > 0) else None
    return hypnn.ToPoincare(c=c, ball_dim=ball_dim, riemannian=False, clip_r=clip_r)


def build_modules(args, proj_sd=None, cls_sd=None):
    """Construct (backbone, projector, classifier) for the chosen method.

    Dimensions / curvature are inferred from the supplied state_dicts where
    possible; CLI values are used as the fallback / override.

    Returns a dict with backbone, projector, classifier (or None), and the
    resolved rep_dim / cls_dim / curvature.
    """
    feat_dim = args.feat_dim
    c = args.c

    # ---- recover a *learnable* curvature from the proj-head checkpoint ----
    if proj_sd is not None:
        raw_keys = [k for k in proj_sd if k.endswith("raw_c")]
        if raw_keys:
            raw = proj_sd[raw_keys[0]].float().view(-1)[0]
            c_rec = float(torch.nn.functional.softplus(raw) + 1e-5)
            log(f"[info] detected learnable curvature in checkpoint -> c={c_rec:g} "
                f"(overriding --c={args.c:g})")
            c = c_rec
            # drop so a fixed-c module can load the rest strictly
            for k in raw_keys:
                proj_sd.pop(k, None)

    method = args.method.lower()
    backbone = _build_backbone(args.dino, feat_dim)
    classifier = None

    if method in ("simgcd", "hypsimgcd", "simgcd_org", "hypsimgcd_org"):
        # backbone(768) -> ToPoincare(768) ;  ToPoincare(768) -> HypLinear(768->K)
        if cls_sd is not None and "weight" in cls_sd:
            cls_dim = int(cls_sd["weight"].shape[0])
            feat_dim = int(cls_sd["weight"].shape[1])
        else:
            cls_dim = args.num_classes
        rep_dim = feat_dim
        projector = _make_topoincare(c, rep_dim, args.cr)
        classifier = hypnn.HypLinear(in_features=feat_dim, out_features=cls_dim, c=c)

    elif method in ("gcd", "hypgcd"):
        # backbone(768) -> DINOHead2(768->hyp_dim) -> ToPoincare(hyp_dim)
        rep_dim = _infer_last_linear_out(proj_sd, prefix="0.mlp") or args.hyp_dim
        mlp = vits1.DINOHead2(in_dim=feat_dim, bottleneck_dim=rep_dim,
                              nlayers=args.num_mlp_layers, hidden_dim=args.hidden_dim)
        projector = nn.Sequential(mlp, _make_topoincare(c, rep_dim, args.cr))

    elif method in ("selex", "hypselex"):
        # backbone(768) -> DINOHead(768->out_dim) -> ToPoincare(out_dim)
        rep_dim = _infer_weightnorm_out(proj_sd, prefix="0.last_layer") or args.mlp_out_dim
        head = vits1.DINOHead(in_dim=feat_dim, out_dim=rep_dim,
                              nlayers=args.num_mlp_layers, hidden_dim=args.hidden_dim)
        projector = nn.Sequential(head, _make_topoincare(c, rep_dim, args.cr))

    else:
        raise ValueError(f"Unknown --method {args.method!r}. "
                         f"Use one of: simgcd, gcd, selex.")

    return {"backbone": backbone, "projector": projector, "classifier": classifier,
            "rep_dim": rep_dim, "cls_dim": (classifier.out_features if classifier else None),
            "c": c, "feat_dim": feat_dim}


def _infer_last_linear_out(sd, prefix):
    """Out-dim of the last 2D weight under `prefix` (DINOHead2 bottleneck)."""
    if not sd:
        return None
    cand = [(k, v) for k, v in sd.items()
            if k.startswith(prefix) and k.endswith("weight") and v.ndim == 2]
    if not cand:
        return None
    # last linear layer by numeric index in the key
    def idx(k):
        for p in k.split("."):
            if p.isdigit():
                return int(p)
        return -1
    cand.sort(key=lambda kv: idx(kv[0]))
    return int(cand[-1][1].shape[0])


def _infer_weightnorm_out(sd, prefix):
    """Out-dim of a weight_norm Linear (DINOHead last_layer): weight_g shape[0]."""
    if not sd:
        return None
    for key in (f"{prefix}.weight_g", f"{prefix}.weight_v"):
        if key in sd:
            return int(sd[key].shape[0])
    return None


def _robust_load(module, state_dict, what):
    """Load state_dict, trying strict first then non-strict with a report."""
    try:
        module.load_state_dict(state_dict, strict=True)
        log(f"[info] loaded {what} (strict).")
        return
    except Exception as e:
        log(f"[warn] strict load of {what} failed: {e}\n"
            f"       retrying with strict=False ...")
    res = module.load_state_dict(state_dict, strict=False)
    missing = list(getattr(res, "missing_keys", []))
    unexpected = list(getattr(res, "unexpected_keys", []))
    if missing:
        log(f"       missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        log(f"       unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    log(f"[info] loaded {what} (non-strict).")


def _load_ckpt(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and all(
            not torch.is_tensor(v) for k, v in obj.items() if k != "state_dict"):
        obj = obj["state_dict"]
    # strip a possible "module." prefix from DataParallel checkpoints
    if isinstance(obj, dict) and any(k.startswith("module.") for k in obj):
        obj = {k[len("module."):] if k.startswith("module.") else k: v
               for k, v in obj.items()}
    return obj


def _derive_sibling(model_path, tag):
    d = os.path.dirname(model_path)
    fn = os.path.basename(model_path)
    stem, ext = os.path.splitext(fn)
    if stem == "model":
        out = f"model_{tag}"
    elif stem.startswith("model_"):
        out = f"model_{tag}_" + stem[len("model_"):]
    else:
        out = f"{stem}_{tag}"
    return os.path.join(d, out + ext)


# ----------------------------------------------------------------------------
# Data + feature extraction
# ----------------------------------------------------------------------------
def build_dataloader(args):
    """Build a DataLoader for the requested split.

    The data stack (datasets, augmentations, torchvision) is imported lazily so
    that the rest of this module can be imported/tested without it.
    """
    from torch.utils.data import DataLoader
    from data.augmentations import get_transform
    from data.get_datasets import get_datasets, get_class_splits
    from models.model import ContrastiveLearningViewGenerator

    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    args.interpolation = 3
    args.crop_pct = 0.875
    train_transform, test_transform = get_transform(
        args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=train_transform, n_views=2)

    train_dataset, test_dataset, unlabelled_train_examples_test, _ = get_datasets(
        args.dataset_name, train_transform, test_transform, args)

    if args.split == "test":
        ds = test_dataset
    elif args.split == "train_unlabelled":
        ds = unlabelled_train_examples_test
    else:
        raise ValueError("--split must be 'train_unlabelled' or 'test'")

    loader = DataLoader(ds, num_workers=args.num_workers, batch_size=args.batch_size,
                        shuffle=False, pin_memory=False)
    return loader, args.num_labeled_classes, args.num_classes


@torch.no_grad()
def extract_features(modules, loader, device, max_samples, want_cls):
    """Run inference and collect Poincaré points + labels.

    Returns dict with: rep (n,d) np, cls (n,k) np or None, labels (n,), preds
    (n,) or None.
    """
    backbone = modules["backbone"].to(device).eval()
    projector = modules["projector"].to(device).eval()
    classifier = (modules["classifier"].to(device).eval()
                  if (want_cls and modules["classifier"] is not None) else None)

    rep_list, cls_list, lbl_list, pred_list = [], [], [], []
    total = 0
    for batch in loader:
        images, labels = batch[0], batch[1]
        if isinstance(images, (list, tuple)):   # safety: pick the first view
            images = images[0]
        images = images.to(device, non_blocking=True)

        feat = backbone(images)
        rep = projector(feat)
        rep_list.append(rep.float().cpu())
        lbl_list.append(torch.as_tensor(labels).long().cpu())
        if classifier is not None:
            logits = classifier(rep)
            cls_list.append(logits.float().cpu())
            pred_list.append(logits.argmax(1).cpu())

        total += images.shape[0]
        if max_samples and total >= max_samples:
            break

    rep = torch.cat(rep_list, 0).numpy()
    labels = torch.cat(lbl_list, 0).numpy()
    cls = torch.cat(cls_list, 0).numpy() if cls_list else None
    preds = torch.cat(pred_list, 0).numpy() if pred_list else None

    if max_samples and rep.shape[0] > max_samples:
        rep = rep[:max_samples]; labels = labels[:max_samples]
        if cls is not None:
            cls = cls[:max_samples]
        if preds is not None:
            preds = preds[:max_samples]
    return {"rep": rep, "cls": cls, "labels": labels, "preds": preds}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Visualize the Poincaré ball(s) of a trained HypCD model "
                    "(HypGCD / HypSimGCD / HypSelEx).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # what / where
    p.add_argument("--method", required=True, choices=["simgcd", "gcd", "selex"],
                   help="Which model family produced the checkpoint.")
    p.add_argument("--dataset_name", required=True,
                   help="cub | scars | aircraft | cifar10 | cifar100 | imagenet_100 | pets | herbarium_19")
    p.add_argument("--dino", default="v1", choices=["v1", "v2"],
                   help="DINO backbone version used in training.")
    p.add_argument("--model_path", required=True,
                   help="Path to the backbone checkpoint (e.g. .../model_best.pt). "
                        "Sibling proj-head/classifier files are derived automatically.")
    p.add_argument("--proj_head_path", default=None,
                   help="Override path to the projection-head checkpoint.")
    p.add_argument("--classifier_path", default=None,
                   help="Override path to the hyperbolic classifier checkpoint (SimGCD).")

    # geometry / architecture
    p.add_argument("--c", type=float, default=0.1, help="Ball curvature used in training.")
    p.add_argument("--cr", type=float, default=0.0,
                   help="Feature-clipping radius used in training (0 / <=0 = no clipping).")
    p.add_argument("--feat_dim", type=int, default=768, help="Backbone feature dim.")
    p.add_argument("--hyp_dim", type=int, default=256, help="GCD: ToPoincare ball dim (DINOHead2 bottleneck).")
    p.add_argument("--mlp_out_dim", type=int, default=8192, help="SelEx: DINOHead output (= ball) dim.")
    p.add_argument("--num_mlp_layers", type=int, default=3, help="DINOHead/DINOHead2 layers.")
    p.add_argument("--hidden_dim", type=int, default=2048, help="DINOHead/DINOHead2 hidden dim.")
    p.add_argument("--num_classes", type=int, default=None,
                   help="Total classes (auto-set from the data split; only needed if you "
                        "skip data loading).")

    # data split selection (mirrors the trainers)
    p.add_argument("--split", default="train_unlabelled",
                   choices=["train_unlabelled", "test"],
                   help="Which split to embed.")
    p.add_argument("--use_ssb_splits", action="store_true", default=True)
    p.add_argument("--prop_train_labels", type=float, default=0.5)
    p.add_argument("--transform", type=str, default="imagenet")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)

    # analysis controls
    p.add_argument("--out_dir", default="./poincare_viz", help="Where to write outputs.")
    p.add_argument("--embed", default="pca", choices=["pca", "tsne"],
                   help="2D embedding for the disk plots (pca is deterministic & origin-preserving).")
    p.add_argument("--max_samples", type=int, default=4000,
                   help="Cap on #images to embed (0 = all).")
    p.add_argument("--embed_sample", type=int, default=4000,
                   help="Cap on #points drawn in the 2D scatter.")
    p.add_argument("--metric_sample", type=int, default=2000,
                   help="Cap on #points for O(n^2) metrics (pairwise dist, delta, angular).")
    p.add_argument("--delta_tries", type=int, default=8,
                   help="Random sub-samples averaged for delta-hyperbolicity.")
    p.add_argument("--delta_batch", type=int, default=512,
                   help="Per-try sub-sample size for delta-hyperbolicity "
                        "(memory ~ batch^3; 512 -> ~0.5GB).")
    p.add_argument("--uniformity_t", type=float, default=1.0,
                   help="Temperature in the hyperbolic uniformity loss.")
    p.add_argument("--no_classification_ball", action="store_true", default=False,
                   help="Skip the SimGCD classification-space ball even if a classifier exists.")
    p.add_argument("--save_embeddings", action="store_true", default=True,
                   help="Also dump the raw points/labels as .npz.")

    # runtime
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto).")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    log(f"[visualize_poincare] starting | method={args.method} "
        f"dataset={args.dataset_name} dino={args.dino}")
    if _REPO_ROOT:
        log(f"[info] HypCD repo root: {_REPO_ROOT}")
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    # Resolve all user paths to absolute BEFORE we may change directory, so
    # outputs land where the user asked and checkpoints are found regardless
    # of CWD.
    args.out_dir = os.path.abspath(args.out_dir)
    args.model_path = os.path.abspath(args.model_path)
    if args.proj_head_path:
        args.proj_head_path = os.path.abspath(args.proj_head_path)
    if args.classifier_path:
        args.classifier_path = os.path.abspath(args.classifier_path)
    os.makedirs(args.out_dir, exist_ok=True)
    log(f"[info] device = {device}")
    log(f"[info] outputs -> {args.out_dir}")

    # ---- resolve sibling checkpoints ----
    proj_path = args.proj_head_path or _derive_sibling(args.model_path, "proj_head")
    cls_path = args.classifier_path or _derive_sibling(args.model_path, "hyp_cls")

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"backbone checkpoint not found: {args.model_path}")
    backbone_sd = _load_ckpt(args.model_path)

    proj_sd = None
    if os.path.isfile(proj_path):
        proj_sd = _load_ckpt(proj_path)
        log(f"[info] projection head : {proj_path}")
    else:
        log(f"[warn] projection-head checkpoint not found at {proj_path}. "
            f"The projector will use freshly-initialised weights "
            f"(fine for SimGCD, whose ToPoincare has no parameters; "
            f"NOT fine for GCD/SelEx).")

    want_cls = (args.method.lower().startswith("simgcd")
                and not args.no_classification_ball)
    cls_sd = None
    if want_cls:
        if os.path.isfile(cls_path):
            cls_sd = _load_ckpt(cls_path)
            log(f"[info] classifier      : {cls_path}")
        else:
            log(f"[warn] classifier checkpoint not found at {cls_path}; "
                f"the classification ball will be skipped.")
            want_cls = False

    # ---- data ----
    # The repo's data utilities use repo-root-relative paths (notably
    # `config.osr_split_dir = 'data/ssb_splits'`), so they only work when the
    # CWD is the repo root.  We switch into it here (after resolving the user's
    # paths to absolute above).  Dataset image roots in config.py are absolute,
    # so they are unaffected.
    if _REPO_ROOT and os.path.abspath(os.getcwd()) != os.path.abspath(_REPO_ROOT):
        log(f"[info] chdir -> {_REPO_ROOT} (so the repo's relative data paths resolve)")
        os.chdir(_REPO_ROOT)
    loader, num_labeled, num_classes = build_dataloader(args)
    args.num_classes = num_classes
    log(f"[info] dataset={args.dataset_name} split={args.split} "
        f"classes={num_classes} (known={num_labeled})")

    # ---- model ----
    modules = build_modules(args, proj_sd=proj_sd, cls_sd=cls_sd)
    c = modules["c"]
    log(f"[info] resolved: rep_dim={modules['rep_dim']} "
        f"cls_dim={modules['cls_dim']} c={c:g} cr={args.cr:g}")

    _robust_load(modules["backbone"], backbone_sd, "backbone")
    if proj_sd is not None:
        _robust_load(modules["projector"], proj_sd, "projection head")
    if want_cls and cls_sd is not None and modules["classifier"] is not None:
        _robust_load(modules["classifier"], cls_sd, "classifier")

    # ---- extract ----
    log("[info] extracting features ...")
    data = extract_features(modules, loader, device,
                            max_samples=args.max_samples, want_cls=want_cls)
    labels = data["labels"]
    is_old = labels < num_labeled
    log(f"[info] collected {labels.shape[0]} points "
        f"({int(is_old.sum())} known / {int((~is_old).sum())} novel)")

    # ---- analyse representation ball ----
    all_metrics = {}
    all_metrics["rep"] = analyze_ball(
        "rep", data["rep"], labels, is_old, c, num_labeled,
        os.path.join(args.out_dir), args,
        preds=None, title_extra="representation space")

    # ---- analyse classification ball (SimGCD) ----
    if want_cls and data["cls"] is not None:
        all_metrics["cls"] = analyze_ball(
            "cls", data["cls"], labels, is_old, c, num_labeled,
            os.path.join(args.out_dir), args,
            preds=data["preds"], title_extra="classification space")

    with open(os.path.join(args.out_dir, "all_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    log(f"\n[done] all outputs in: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()