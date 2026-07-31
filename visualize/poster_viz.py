#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poster_viz.py — HiFi-GCD A0 poster figures (data-driven set: F3a, F3b, F5, F6)
==============================================================================
Four publication-grade matplotlib figures sharing one design system:

  f3a   Curvature tug-of-war       (DIAGNOSIS · MEASUREMENT I)   — inset-ready
  f3b   Shell distribution         (DIAGNOSIS · MEASUREMENT II)
  f5    Depth stratifies, off→on   (VERDICT · P1)
  f6    Direction selectivity      (VERDICT · P2)

Quick preview with synthetic data (exact final styling):
  python poster_viz.py all --demo -o preview/

Real data:
  python poster_viz.py f3a --table mc_cub.csv --x-col epoch \
         --sup-col c_sup --unsup-col c_unsup --unbind 25 \
         --probe-rep 1.10 --probe-cls 0.47 -o figs/F3a
  python poster_viz.py f3b --radii radii_base.npy -o figs/F3b
  python poster_viz.py f5  --off-object off_obj.npy --off-image off_img.npy \
         --on-object on_obj.npy  --on-image on_img.npy \
         --anchor-object 0.56 --anchor-image 0.78 --feas-on 41.7 -o figs/F5
  python poster_viz.py f6  --obj ent_obj_parent.csv --obj-col feasible_pct \
         --img ent_img_parent.csv --img-col feasible_pct --x-col epoch -o figs/F6

Data formats
------------
Tables  (.csv / .tsv / .txt with a header row, .json dict-of-lists or
        list-of-dicts, .npz): columns selected by name
        (--x-col / --sup-col / --obj-col ...).
Arrays  (.npy; .npz with --key; .csv/.tsv/.txt with --key = column name,
        default first column): 1-D per-sample embedding norms.
        Values auto-normalise to r/R via --R, or by the array max if max>1.05.

Every subcommand accepts: -o BASE (file base or directory),
--formats pdf,png[,svg], --dpi, --size WxH (mm), --font-scale,
--no-title, --kicker/--title/--sub/--foot overrides ("" hides an element),
--transparent, --demo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.transforms import blended_transform_factory as blend  # noqa: E402

# ----------------------------------------------------------------------------
# design system
# ----------------------------------------------------------------------------
INK = "#1B2437"      # near-black slate
MUTED = "#8A93A6"    # secondary text
FAINT = "#C7CDD9"    # hairlines
GRID = "#E8ECF3"     # gridlines
PAPER = "#FFFFFF"

SUP = "#2563EB"      # supervised / known   (blue)
DISC = "#F97316"     # discovery / novel    (orange)
DEAD = "#DC2626"     # shell, dead radial axis (red accent)
OK = "#0E9F6E"       # verdict green (sparing)
AMBER = "#D97706"    # pending / demo tags

LEVEL = {            # hierarchy: one hue, centre → rim
    "part": "#A5B4FC",
    "object": "#6366F1",
    "image": "#3730A3",
}
LEVEL_NAME = {"part": "parts", "object": "object view", "image": "image view"}
LEVEL_SUB = {"part": "part", "object": "obj", "image": "img"}


def mix(c1, c2, t):
    a = np.array(mcolors.to_rgb(c1))
    b = np.array(mcolors.to_rgb(c2))
    return mcolors.to_hex((1 - t) * a + t * b)


def setup(fs=1.0):
    plt.rcParams.update({
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "Inter", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": MUTED, "axes.linewidth": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": "#4A5468", "ytick.labelcolor": "#4A5468",
        "xtick.labelsize": 11.5 * fs, "ytick.labelsize": 11.5 * fs,
        "axes.labelsize": 13.5 * fs, "font.size": 12.5 * fs,
        "xtick.major.size": 3.4, "ytick.major.size": 3.4,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def track(s, n=1):
    """Letterspace a short tag with thin spaces."""
    return ("\u2009" * n).join(list(s))


def _dy(fig, pts):
    return pts / 72.0 / fig.get_size_inches()[1]


def _dx(fig, pts):
    return pts / 72.0 / fig.get_size_inches()[0]


def header(fig, x0, kicker=None, title=None, sub=None, check=False, fs=1.0):
    """Draw kicker/title/sub at figure top-left; return y for axes top."""
    y = 1.0 - _dy(fig, 9)
    if kicker:
        fig.text(x0, y, kicker, size=10.8 * fs, color=MUTED,
                 weight="bold", va="top")
        y -= _dy(fig, 10.8) + _dy(fig, 7)
    if title:
        xt = x0
        if check:
            fig.text(x0, y, "\u2713", size=19.0 * fs, color=OK,
                     weight="bold", va="top")
            xt = x0 + _dx(fig, 24 * fs)
        fig.text(xt, y, title, size=19.0 * fs, color=INK, weight="bold",
                 va="top", linespacing=1.18)
        y -= _dy(fig, 19.0) * (title.count("\n") + 1) * 1.18 + _dy(fig, 6)
    if sub:
        fig.text(x0, y, sub, size=11.8 * fs, color=MUTED, va="top",
                 linespacing=1.3)
        y -= _dy(fig, 11.8) * (sub.count("\n") + 1) * 1.3 + _dy(fig, 4)
    return y - _dy(fig, 8)


def footer(fig, x0, text=None, fs=1.0, demo=False):
    """Draw footnote at figure bottom; return y just above it."""
    yb = _dy(fig, 8)
    top = yb
    if text:
        fig.text(x0, yb, text, size=9.8 * fs, color=MUTED, va="bottom",
                 linespacing=1.3)
        top = yb + _dy(fig, 9.8) * (text.count("\n") + 1) * 1.3
    if demo:
        fig.text(0.985, yb, "synthetic preview data", size=9.2 * fs,
                 color=AMBER, ha="right", va="bottom", style="italic")
        top = max(top, yb + _dy(fig, 9.2) * 1.3)
    return top + _dy(fig, 4)


def halo(lw=3.0):
    return [pe.withStroke(linewidth=lw, foreground="white")]


def strip_spines(ax, keep=("left", "bottom")):
    for s in ax.spines:
        ax.spines[s].set_visible(s in keep)


# ----------------------------------------------------------------------------
# small numerics
# ----------------------------------------------------------------------------
def smooth(y, w):
    y = np.asarray(y, float)
    if w is None or w <= 1:
        return y
    k = np.ones(int(w))
    num = np.convolve(y, k, "same")
    den = np.convolve(np.ones_like(y), k, "same")
    return num / den


def kde(v, grid, bw=None, floor=0.0, max_n=8000, seed=0):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size > max_n:
        v = np.random.default_rng(seed).choice(v, max_n, replace=False)
    n = v.size
    s = v.std()
    q75, q25 = np.percentile(v, [75, 25])
    sig = min(s, (q75 - q25) / 1.349) if (q75 - q25) > 0 else s
    if bw is None:
        bw = 0.9 * sig * n ** (-0.2) if sig > 0 else 0.0
    bw = max(bw, floor, 1e-6)
    z = (grid[:, None] - v[None, :]) / bw
    d = np.exp(-0.5 * z * z).sum(axis=1)
    return d / (n * bw * np.sqrt(2 * np.pi))


# ----------------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------------
def die(msg):
    sys.exit("[poster_viz] " + msg)


def load_table(path):
    p = Path(path)
    if not p.exists():
        die(f"file not found: {path}")
    e = p.suffix.lower()
    if e == ".npz":
        d = np.load(p)
        return {k: np.atleast_1d(np.asarray(d[k], float)).ravel()
                for k in d.files}
    if e == ".json":
        obj = json.load(open(p))
        if isinstance(obj, list):
            keys = list(obj[0].keys())
            return {k: np.array([row[k] for row in obj], float) for k in keys}
        return {k: np.atleast_1d(np.asarray(v, float)).ravel()
                for k, v in obj.items()}
    delim = "," if e == ".csv" else ("\t" if e == ".tsv" else None)
    arr = np.genfromtxt(p, names=True, delimiter=delim, dtype=float,
                        encoding="utf-8")
    arr = np.atleast_1d(arr)
    return {name: np.asarray(arr[name], float).ravel()
            for name in arr.dtype.names}


def getcol(tbl, name, path):
    if name in tbl:
        return tbl[name]
    for k in tbl:
        if k.lower() == name.lower():
            return tbl[k]
    die(f"column '{name}' not found in {path}; available: {list(tbl)}")


def load_array(path, key=None):
    p = Path(path)
    if not p.exists():
        die(f"file not found: {path}")
    e = p.suffix.lower()
    if e == ".npy":
        return np.asarray(np.load(p), float).ravel()
    if e == ".npz":
        d = np.load(p)
        k = key if key else d.files[0]
        if k not in d.files:
            die(f"key '{k}' not in {path}; available: {d.files}")
        return np.asarray(d[k], float).ravel()
    t = load_table(path)
    if key:
        return getcol(t, key, path)
    return t[list(t)[0]]


def norm_radii(r, R=None, tag=""):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if R:
        return r / R
    if r.size and np.nanmax(r) > 1.05:
        m = float(np.nanmax(r))
        print(f"[poster_viz] {tag}: normalising by max norm {m:.4g}")
        return r / m
    return r


# ----------------------------------------------------------------------------
# demo data
# ----------------------------------------------------------------------------
def demo_f3a():
    rng = np.random.default_rng(7)
    e = np.arange(0.0, 121.0)
    unb = 25.0
    base = 0.78
    cs = np.where(e < unb, base, 0.045 + (base - 0.045) * np.exp(-(e - unb) / 16.0))
    cu = np.where(e < unb, base, 1.385 - (1.385 - base) * np.exp(-(e - unb) / 20.0))
    jit = np.where(e < unb, 0.30, 1.0)
    cs = np.clip(cs + rng.normal(0, 0.013, e.size) * jit, 0.0, None)
    cu = cu + rng.normal(0, 0.017, e.size) * jit
    return e, cs, cu, unb


def _needle(rng, n):
    core = 1.0 - np.abs(rng.normal(0, 2.4e-4, int(n * 0.997)))
    tail = rng.uniform(0.985, 0.9995, n - core.size)
    r = np.concatenate([core, tail])
    rng.shuffle(r)
    return np.clip(r, 0, 1.0)


def demo_f3b():
    return _needle(np.random.default_rng(11), 24310)


def demo_f5():
    rng = np.random.default_rng(13)
    off = {"object": _needle(rng, 6000), "image": _needle(rng, 6000)}
    on = {
        "object": np.clip(rng.normal(0.560, 0.033, 6000), 0.05, 0.995),
        "image": np.clip(rng.normal(0.775, 0.046, 6000), 0.05, 0.998),
    }
    anchors = {"object": 0.56, "image": 0.78}
    return off, on, anchors


def demo_f6():
    rng = np.random.default_rng(17)
    e = np.arange(0.0, 91.0)
    obj = 62.0 * (1 - np.exp(-e / 13.5)) + rng.normal(0, 1.2, e.size)
    rise = 34.0 / (1 + np.exp(-(e - 16) / 5.5))
    decay = np.where(e > 34, np.exp(-(e - 34) / 18.0), 1.0)
    img = rise * decay + 6.5 * (1 - decay) + rng.normal(0, 1.1, e.size)
    return e, np.clip(obj, 0, None), e, np.clip(img, 0, None)


# ----------------------------------------------------------------------------
# F3a — curvature tug-of-war
# ----------------------------------------------------------------------------
def fig_f3a(ns):
    fs = ns.font_scale
    setup(fs)
    if ns.demo:
        x, cs, cu, unb = demo_f3a()
        unbind = unb if ns.unbind is None else ns.unbind
    else:
        if not ns.table:
            die("f3a needs --table (or --demo)")
        t = load_table(ns.table)
        x = getcol(t, ns.x_col, ns.table)
        cs = getcol(t, ns.sup_col, ns.table)
        cu = getcol(t, ns.unsup_col, ns.table)
        unbind = ns.unbind
    o = np.argsort(x)
    x, cs, cu = x[o], cs[o], cu[o]
    css, cus = smooth(cs, ns.smooth), smooth(cu, ns.smooth)
    probe = (ns.probe_rep is not None) and (ns.probe_cls is not None)

    W, H = ns.size or (170, 100)
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    x0 = 0.055
    kick = track("DIAGNOSIS") + "   ·   MEASUREMENT I" \
        if ns.kicker is None else ns.kicker
    ttl = None if ns.no_title else (
        "The loss that memorizes flattens space;\n"
        "the loss that discovers bends it." if ns.title is None else ns.title)
    sub = None if ns.no_title else (
        "Each objective is handed its own learnable curvature $c$ — and pulls\n"
        "the shared space toward the opposite geometry."
        if ns.sub is None else ns.sub)
    foot = ns.foot
    if foot is None:
        foot = (f"solid: moving average (w={ns.smooth}) over the raw trace "
                f"(faint)   ·   converging evidence")
        if probe:
            foot += ("\nprobe: converged curvature when representation / "
                     "classifier spaces are unbound separately")
    top = header(fig, x0, kick, ttl, sub, fs=fs)
    ftop = footer(fig, x0, foot, fs=fs, demo=ns.demo)
    yb = ftop + _dy(fig, 30)
    hgt = top - yb
    left = 0.085
    main_r = 0.795 if probe else 0.955
    ax = fig.add_axes([left, yb, main_r - left, hgt])

    ymax = max(float(np.nanmax(cus)), float(np.nanmax(css)))
    ylo, yhi = -0.075 * ymax, 1.16 * ymax
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(x[0], x[-1])
    yr = yhi - ylo
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    strip_spines(ax)

    if unbind is not None:
        m = x >= unbind
        ax.fill_between(x[m], css[m], cus[m], color=FAINT, alpha=0.28,
                        lw=0, zorder=0)
        ax.axvline(unbind, color=FAINT, lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.text(unbind + 0.008 * (x[-1] - x[0]), 0.975, "curvatures unbound",
                transform=blend(ax.transData, ax.transAxes), va="top",
                ha="left", size=9.8 * fs, color=MUTED)

    ax.axhline(0, color=FAINT, lw=1.0, zorder=1)
    ax.text(0.012, 0.0, "flat  (Euclidean)", transform=blend(ax.transAxes,
            ax.transData), va="bottom", ha="left", size=9.8 * fs, color=MUTED)

    for raw, sm, col in [(cs, css, SUP), (cu, cus, DISC)]:
        ax.plot(x, raw, color=col, lw=1.1, alpha=0.26, zorder=2)
        ax.plot(x, sm, color=col, lw=2.9, zorder=3,
                solid_capstyle="round")
        ax.plot([x[-1]], [sm[-1]], "o", ms=5.5, color=col, zorder=5)

    xr = x[-1] - x[0]
    ax.text(x[-1] - 0.012 * xr, css[-1] + 0.075 * yr,
            f"supervised  ·  $c$ → {css[-1]:.2f}", color=SUP, ha="right",
            va="center", size=12.2 * fs, weight="bold", path_effects=halo(),
            zorder=6)
    ax.text(x[-1] - 0.012 * xr, cus[-1] - 0.085 * yr,
            f"discovery  ·  $c$ → {cus[-1]:.2f}", color=DISC, ha="right",
            va="center", size=12.2 * fs, weight="bold", path_effects=halo(),
            zorder=6)

    ax.set_xlabel("training epoch")
    ax.set_ylabel("curvature  $c$")

    if probe:
        axp = fig.add_axes([main_r + 0.052, yb, 0.955 - main_r - 0.052, hgt],
                           sharey=ax)
        axp.tick_params(labelleft=False, left=False)
        axp.grid(axis="y", color=GRID, lw=0.9)
        axp.set_axisbelow(True)
        strip_spines(axp, keep=("bottom",))
        vals = [("rep.", ns.probe_rep), ("cls.", ns.probe_cls)]
        for i, (lab, v) in enumerate(vals):
            axp.plot([i, i], [0, v], color=FAINT, lw=1.4, zorder=2)
            axp.plot([i], [v], "o", ms=6.5, color=INK, zorder=3)
            axp.text(i, v + 0.05 * yr, f"{v:.2f}", ha="center", va="bottom",
                     size=10.8 * fs, weight="bold", color=INK,
                     path_effects=halo())
        axp.set_xlim(-0.7, 1.7)
        axp.set_xticks([0, 1], [v[0] for v in vals])
        axp.set_title("probe:\nconverged $c$", size=9.4 * fs, color=MUTED,
                      pad=5)
    return fig


# ----------------------------------------------------------------------------
# F3b — shell distribution + aperture mechanism
# ----------------------------------------------------------------------------
def fig_f3b(ns):
    fs = ns.font_scale
    setup(fs)
    if ns.demo:
        r = demo_f3b()
    else:
        if not ns.radii:
            die("f3b needs --radii (or --demo)")
        r = norm_radii(load_array(ns.radii, ns.key), ns.R, "radii")
    frac = float((r >= 0.999).mean() * 100)
    med = float(np.median(r))
    n = r.size

    grid = np.linspace(0.0, 1.03, 1600)
    d = kde(r, grid, bw=ns.bw, floor=ns.bw_floor)
    d = d / d.max()

    W, H = ns.size or (200, 118)
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    x0 = 0.055
    kick = track("DIAGNOSIS") + "   ·   MEASUREMENT II" \
        if ns.kicker is None else ns.kicker
    ttl = None if ns.no_title else (
        "A ball is not yet a tree." if ns.title is None else ns.title)
    sub = None if ns.no_title else (
        "Post-LN norms all exceed the clip radius, pinning every embedding to "
        "one shell —\nwhere the entailment cone is ≈17° wide and no pair fits "
        "inside." if ns.sub is None else ns.sub)
    foot = ns.foot
    if foot is None:
        foot = ("fixed clip  ⇒  z = exp₀(R·x/‖x‖),  ∂z/∂x ∝ I − uu$^\\top$: "
                "the radius receives zero gradient\n"
                "feasibility criterion:  ext∠(u→v) ≤ ω(u) = "
                "arcsin(2K/(√c‖u‖))")
        if ns.no_footnote:
            foot = ""
    top = header(fig, x0, kick, ttl, sub, fs=fs)
    ftop = footer(fig, x0, foot, fs=fs, demo=ns.demo)
    yb = ftop + _dy(fig, 30)
    ax = fig.add_axes([0.075, yb, 0.855 - 0.075, top - yb])

    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.30)
    ax.set_yticks([])
    ax.set_ylabel("density  (scaled)", size=11.0 * fs, color=MUTED)
    ax.set_xlabel("relative depth   r / R")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    strip_spines(ax)
    ax.grid(axis="x", color=GRID, lw=0.9)
    ax.set_axisbelow(True)

    ax.fill_between(grid, 0, d, color=DEAD, alpha=0.90, lw=0, zorder=3)
    ax.plot(grid, d, color=mix(DEAD, INK, 0.25), lw=1.1, zorder=4)

    if ns.extra:
        r2 = norm_radii(load_array(ns.extra, ns.extra_key), ns.R, "extra")
        d2 = kde(r2, grid, bw=ns.bw, floor=ns.bw_floor)
        d2 = d2 / d2.max() * 0.92
        ax.plot(grid, d2, color=MUTED, lw=1.6, zorder=3)
        i = int(np.argmax(d2))
        ax.text(grid[i] - 0.012, d2[i] + 0.03, ns.extra_label, color=MUTED,
                ha="right", va="bottom", size=10.2 * fs)

    ax.axvline(1.0, color=DEAD, ls=(0, (4, 3)), lw=1.3, zorder=2)
    ax.text(0.994, 1.285, "clip radius $R$", ha="right", va="top",
            size=10.6 * fs, color=DEAD, weight="bold")

    # big-stat block in the empty left field
    ax.text(0.045, 0.925, f"{frac:.1f}%", transform=ax.transAxes,
            ha="left", va="top", size=30 * fs, weight="bold", color=INK)
    ax.text(0.045, 0.735, "of embeddings sit within 0.1 % of the boundary",
            transform=ax.transAxes, ha="left", va="top", size=12.3 * fs,
            color=INK)
    ax.text(0.045, 0.635, f"median r/R = {med:.4f}      ·      n = {n:,}",
            transform=ax.transAxes, ha="left", va="top", size=10.6 * fs,
            color=MUTED)
    ax.text(0.045, 0.535, "→  the radial axis carries no learning signal",
            transform=ax.transAxes, ha="left", va="top", size=11.4 * fs,
            color=DEAD, style="italic")

    # aperture mechanism (right axis)
    if not ns.no_aperture:
        ax2 = ax.twinx()
        K, c = ns.K, ns.c
        arg0 = 2 * K / np.sqrt(c)
        rr = np.linspace(max(arg0, 0.12), 1.03, 400)
        ap = np.degrees(np.arcsin(np.clip(arg0 / rr, 0, 1)))
        ax2.plot(rr, ap, color=LEVEL["object"], lw=2.3, zorder=3)
        ax2.set_ylim(0, 105)
        ax2.set_yticks([0, 30, 60, 90])
        ax2.set_ylabel("entailment half-aperture  ω(r)   (deg)",
                       color=LEVEL["object"], size=12.0 * fs,
                       rotation=270, labelpad=18)
        ax2.tick_params(colors=MUTED)
        for s in ("top", "left", "bottom"):
            ax2.spines[s].set_visible(False)
        ax2.spines["right"].set_color(MUTED)
        ax2.axhline(90, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1)
        ap1 = float(np.degrees(np.arcsin(min(arg0 / 1.0, 1))))
        ax2.plot([1.0], [ap1], "o", ms=5.5, color=LEVEL["object"], zorder=5)
        ax2.text(0.985, ap1 - 5, f"ω(shell) ≈ {ap1:.0f}°", ha="right",
                 va="top", size=10.8 * fs, color=LEVEL["object"],
                 weight="bold", path_effects=halo())
        # infeasibility band: required exterior angle vs available aperture
        ax2.plot([1.017, 1.017], [ap1, 90], color=DEAD, lw=3.2,
                 solid_capstyle="butt", zorder=4)
        for yv in (ap1, 90):
            ax2.plot([1.017], [yv], marker="_", ms=9, color=DEAD, zorder=4)
        ax2.text(1.006, 91.5, "same-shell pairs:  ext∠ ≳ 90°", ha="right",
                 va="bottom", size=9.8 * fs, color=MUTED)
        ax2.annotate(f"entailment-feasible pairs:  {ns.feasible:.0f}%",
                     xy=(1.006, (ap1 + 90) / 2), ha="right", va="center",
                     size=11.4 * fs, color="white", weight="bold",
                     bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.9",
                               fc=DEAD, ec="none"), zorder=6)
    return fig


# ----------------------------------------------------------------------------
# F5 — depth stratifies (ridgeline, gate off → on)
# ----------------------------------------------------------------------------
def fig_f5(ns):
    fs = ns.font_scale
    setup(fs)
    if ns.demo:
        off, on, anchors0 = demo_f5()
        anchors = {
            "part": ns.anchor_part,
            "object": ns.anchor_object or anchors0["object"],
            "image": ns.anchor_image or anchors0["image"],
        }
        feas_off = 0.0 if ns.feas_off is None else ns.feas_off
        feas_on = 41.7 if ns.feas_on is None else ns.feas_on
    else:
        off, on = {}, {}
        for lvl, p in [("object", ns.off_object), ("image", ns.off_image)]:
            if p:
                off[lvl] = norm_radii(load_array(p, ns.key), ns.R, f"off-{lvl}")
        for lvl, p in [("part", ns.on_part), ("object", ns.on_object),
                       ("image", ns.on_image)]:
            if p:
                on[lvl] = norm_radii(load_array(p, ns.key), ns.R, f"on-{lvl}")
        if not off or not on:
            die("f5 needs at least one --off-* and one --on-* array "
                "(or --demo)")
        anchors = {"part": ns.anchor_part, "object": ns.anchor_object,
                   "image": ns.anchor_image}
        feas_off = 0.0 if ns.feas_off is None else ns.feas_off
        feas_on = ns.feas_on

    show_vals = not ns.schematic
    if ns.schematic:
        POS = {"part": 0.62 if ns.pos_part is None else ns.pos_part,
               "object": 0.84 if ns.pos_object is None else ns.pos_object,
               "image": 1.06 if ns.pos_image is None else ns.pos_image}
        SD = {"part": 0.034 if ns.sd_part is None else ns.sd_part,
              "object": 0.032 if ns.sd_object is None else ns.sd_object,
              "image": 0.042 if ns.sd_image is None else ns.sd_image}
        rng = np.random.default_rng(7)
        for l in list(on):
            on[l] = rng.normal(POS[l], SD[l], 4000)
            anchors[l] = POS[l]

    order = [l for l in ("part", "object", "image")]
    rows_off = [(l, off[l]) for l in order if l in off]
    rows_on = [(l, on[l]) for l in order if l in on]
    learned = all(anchors.get(l) is not None for l, _ in rows_on)
    for l, v in rows_on:
        if anchors.get(l) is None:
            anchors[l] = float(np.median(v))

    W, H = ns.size or ((168, 96) if ns.schematic else (192, 128))
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    x0 = 0.055
    kick = track("P1") + "   ·   RADIAL GATE, OFF → ON" \
        if ns.kicker is None else ns.kicker
    ttl = None if ns.no_title else (
        ("Depth stratifies in the predicted order." if ns.schematic else
         "Depth stratifies — in the predicted order.")
        if ns.title is None else ns.title)
    if ns.sub is not None:
        sub = None if ns.no_title else ns.sub
    elif ns.schematic:
        sub = None if ns.no_title else (
            "Gate off: every level pinned to one shell.\n"
            "Gate on: levels open into ordered annuli.")
    else:
        sub = None if ns.no_title else (
            "Same pipeline, radial gate off vs on: needles pinned at the "
            "shell open into\nlevel-wise annuli around learnable depth "
            "anchors $s_\\ell$.")
    foot = ns.foot
    if foot is None:
        if ns.schematic:
            foot = ("depth axis schematic — order and shell relation as "
                    "measured; spacing not to scale")
        else:
            src = ("learned $s_\\ell$" if learned else "per-level medians")
            foot = ("per-row densities scaled to equal height   ·   depth "
                    f"anchors shown at {src}   ·   off / on share the "
                    "backbone; only the gate differs")
    top = header(fig, x0, kick, ttl, sub, check=True, fs=fs)
    ftop = footer(fig, x0, foot, fs=fs, demo=ns.demo)
    yb = ftop + _dy(fig, 30)
    ax = fig.add_axes([0.145, yb, 0.955 - 0.145, top - yb])

    gap = 0.48 if ns.schematic else 0.62
    n_on = len(rows_on)
    bases_on = {l: float(n_on - 1 - i) for i, (l, _) in enumerate(rows_on)}
    bases_off = {l: n_on + gap + float(len(rows_off) - 1 - i)
                 for i, (l, _) in enumerate(rows_off)}
    top_row = max(bases_off.values()) if rows_off else max(bases_on.values())

    xmax = 1.05
    xlo = 0.0
    for _, v in rows_off + rows_on:
        if v.size:
            xmax = max(xmax, float(np.nanmax(v)) + 0.03)
    if ns.schematic:
        lo = min([float(np.nanmin(v)) for _, v in rows_off + rows_on
                  if v.size] + [1.0])
        xlo = max(0.0, lo - 0.10)
    grid = np.linspace(xlo, xmax - 0.02, 1400)
    trans = ax.get_yaxis_transform()

    def draw_row(lvl, data, base, dead):
        col = LEVEL[lvl]
        dd = kde(data, grid, bw=ns.bw, floor=ns.bw_floor)
        dd = dd / dd.max() * 0.86
        ax.plot([xlo, xmax - 0.02], [base, base], color=FAINT, lw=0.8,
                zorder=1)
        if dead:
            ax.fill_between(grid, base, base + dd, color=mix(col, "#9AA3B5",
                            0.55), alpha=0.55, lw=0, zorder=2)
            ax.plot(grid, base + dd, color=DEAD, lw=1.0, zorder=3)
        else:
            ax.fill_between(grid, base, base + dd, color=col, alpha=0.95,
                            lw=0, zorder=2)
            ax.plot(grid, base + dd, color="white", lw=0.8, zorder=3)
        ax.plot([-0.028], [base + 0.30], marker="o", ms=5.2, color=col,
                transform=trans, clip_on=False, zorder=4)
        nm = lvl if ns.schematic else LEVEL_NAME[lvl]
        ax.text(-0.042, base + 0.30, nm, transform=trans,
                ha="right", va="center", size=11.4 * fs, color=INK)

    for l, v in rows_off:
        draw_row(l, v, bases_off[l], dead=True)
    for l, v in rows_on:
        draw_row(l, v, bases_on[l], dead=False)

    # group headers
    hdr_dy = 0.90 if ns.schematic else 1.02
    off_hdr = track("OFF") if ns.schematic else track("RADIAL GATE") + "  —  OFF"
    on_hdr = track("ON") if ns.schematic else track("RADIAL GATE") + "  —  ON"
    if rows_off:
        ax.text(0.012, max(bases_off.values()) + hdr_dy, off_hdr,
                transform=trans, ha="left", va="bottom", size=10.2 * fs,
                color=MUTED, weight="bold")
    ax.text(0.012, max(bases_on.values()) + hdr_dy, on_hdr, transform=trans,
            ha="left", va="bottom", size=10.2 * fs, color=INK, weight="bold")

    # shell line + pinned annotation
    ax.axvline(1.0, color=DEAD, ls=(0, (4, 3)), lw=1.3, zorder=1)
    if not ns.schematic:
        ax.text(1.014,
                (min(bases_on.values()) + max(bases_on.values())) / 2 + 0.4,
                "shell  (clip radius)", rotation=90, va="center", ha="left",
                size=9.8 * fs, color=DEAD)
    if rows_off:
        ytop_off = max(bases_off.values())
        ax.annotate("pinned at the shell",
                    xy=(0.995, ytop_off + 0.55),
                    xytext=(0.80, ytop_off + 0.62), ha="right", va="center",
                    size=10.6 * fs, color=DEAD,
                    arrowprops=dict(arrowstyle="-", color=DEAD, lw=1.0))

    # anchors + order verdict
    present = [l for l, _ in rows_on]
    y_arrow = -1.55 if show_vals else -0.92
    for l in present:
        a = anchors[l]
        ax.plot([a], [-0.42], marker="^", ms=7.5, color=LEVEL[l],
                clip_on=False, zorder=5)
        if show_vals:
            ax.text(a, -0.78, f"$s_{{{LEVEL_SUB[l]}}}$ = {a:.2f}",
                    ha="center", va="top",
                    size=10.8 * fs, weight="bold", color=LEVEL[l])
    if len(present) >= 2:
        a_min = min(anchors[l] for l in present)
        a_max = max(anchors[l] for l in present)
        ax.annotate("", xy=(a_max, y_arrow), xytext=(a_min, y_arrow),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
        mid = (a_min + a_max) / 2
        order_str = "  <  ".join(f"$s_{{{LEVEL_SUB[l]}}}$" for l in present)
        a_seq = [anchors[l] for l in present]
        ordered = all(b > a for a, b in zip(a_seq, a_seq[1:]))
        if show_vals:
            ax.text(mid, y_arrow - 0.23, order_str, ha="center", va="top",
                    size=12.0 * fs, weight="bold", color=INK)
            if ordered:
                ax.text(mid, y_arrow - 0.73, "\u2713  order as predicted",
                        ha="center", va="top", size=11.0 * fs, weight="bold",
                        color=OK)
            else:
                ax.text(mid, y_arrow - 0.73,
                        "order not yet satisfied at this epoch",
                        ha="center", va="top", size=10.6 * fs, color=AMBER,
                        style="italic")
        elif ordered:
            ax.text(mid, y_arrow - 0.12,
                    "\u2713  " + order_str + "  —  order as predicted",
                    ha="center", va="top", size=11.4 * fs, weight="bold",
                    color=OK)
        else:
            ax.text(mid, y_arrow - 0.12, order_str + "  —  not yet satisfied",
                    ha="center", va="top", size=10.8 * fs, color=AMBER,
                    style="italic")

    # feasibility badge (skippable when F6 already carries this number)
    if not ns.no_feas:
        fe_on = f"{feas_on:g}%" if feas_on is not None else "\u25A1"
        ax.annotate(f"entailment-feasible pairs    {feas_off:g}%  →  {fe_on}",
                    xy=(0.985, 0.985), xycoords="axes fraction", ha="right",
                    va="top", size=11.2 * fs, color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.9",
                              fc=INK, ec="none"), zorder=6)

    ax.set_xlim(xlo, xmax)
    ax.set_ylim(y_arrow - (1.20 if show_vals else 0.90),
                top_row + (1.32 if show_vals else 1.16))
    ax.set_yticks([])
    if ns.schematic:
        ax.set_xticks([1.0])
        ax.set_xticklabels(["shell"])
        for t in ax.get_xticklabels():
            t.set_color(DEAD)
    else:
        ax.set_xticks([t for t in (0, 0.25, 0.5, 0.75, 1.0) if t >= xlo])
    strip_spines(ax, keep=("bottom",))
    ax.set_xlabel("depth" if ns.schematic else "relative depth   r / R")
    return fig


# ----------------------------------------------------------------------------
# F6 — direction selectivity
# ----------------------------------------------------------------------------
def fig_f6(ns):
    fs = ns.font_scale
    setup(fs)
    if ns.demo:
        xo, yo, xi, yi = demo_f6()
    else:
        if not ns.obj:
            die("f6 needs --obj (or --demo)")
        t = load_table(ns.obj)
        xo = getcol(t, ns.x_col, ns.obj)
        yo = getcol(t, ns.obj_col, ns.obj)
        xi = yi = None
        if ns.img:
            t2 = load_table(ns.img)
            xi = getcol(t2, ns.x_col, ns.img)
            yi = getcol(t2, ns.img_col or ns.obj_col, ns.img)
    o = np.argsort(xo)
    xo, yo = xo[o], yo[o]
    yos = smooth(yo, ns.smooth)
    dual = yi is not None
    flat = False
    if dual:
        o = np.argsort(xi)
        xi, yi = xi[o], yi[o]
        yis = smooth(yi, ns.smooth)
        flat = float(np.nanmax(yis)) < 1.0

    W, H = ns.size or (180, 92)
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    x0 = 0.055
    kick = track("P2") + "   ·   WHICH PARENT DOES THE GEOMETRY KEEP?" \
        if ns.kicker is None else ns.kicker
    ttl = None if ns.no_title else (
        "The geometry votes for the semantics."
        if ns.title is None else ns.title)
    if ns.sub is not None:
        sub = None if ns.no_title else ns.sub
    elif dual and flat:
        sub = ("Only the object-as-parent direction becomes — and stays — "
               "feasible.")
    elif dual:
        sub = ("Two runs, identical objectives — only the assumed parent–child "
               "direction differs.\nObject-as-parent consolidates; the reverse "
               "ordering rises, then dissolves.")
    else:
        sub = ("Entailment imposed with the object as parent consolidates "
               "and holds.")
    if ns.no_title:
        sub = None
    foot = ns.foot
    if foot is None:
        foot = ("feasibility = share of (parent, child) pairs inside the "
                f"entailment cone\nmoving average (w={ns.smooth}) over the "
                "raw trace")
    top = header(fig, x0, kick, ttl, sub, check=True, fs=fs)
    ftop = footer(fig, x0, foot, fs=fs, demo=ns.demo)
    yb = ftop + _dy(fig, 30)
    ax = fig.add_axes([0.10, yb, 0.955 - 0.10, top - yb])

    ymax = float(np.nanmax(yos))
    if dual:
        ymax = max(ymax, float(np.nanmax(yis)))
    ylo = -ymax * 0.05 if (dual and flat) else 0.0
    ax.set_ylim(ylo, ymax * 1.12)
    ax.set_xlim(min(xo[0], xi[0] if dual else xo[0]),
                max(xo[-1], xi[-1] if dual else xo[-1]))
    yr = ymax * 1.12 - ylo
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    strip_spines(ax)

    OBJ = LEVEL["object"]
    ax.fill_between(xo, 0, yos, color=OBJ, alpha=0.07, lw=0, zorder=1)
    ax.plot(xo, yo, color=OBJ, lw=1.1, alpha=0.25, zorder=2)
    ax.plot(xo, yos, color=OBJ, lw=3.0, zorder=4, solid_capstyle="round")
    ax.plot([xo[-1]], [yos[-1]], "o", ms=5.5, color=OBJ, zorder=5)
    ax.text(xo[-1] - 0.012 * xr, yos[-1] - 0.085 * yr,
            f"object as parent  —  holds at {yos[-1]:.0f}%", color=OBJ,
            ha="right", va="center", size=12.2 * fs, weight="bold",
            path_effects=halo(), zorder=6)

    if dual:
        ax.plot(xi, yi, color=MUTED, lw=1.0, alpha=0.30, zorder=2)
        ax.plot(xi, yis, color=MUTED, lw=2.3, ls=(0, (5, 3)), zorder=3)
        ax.plot([xi[-1]], [yis[-1]], "o", ms=5, color=MUTED, zorder=5)
        if not flat:
            ipk = int(np.argmax(yis))
            ax.plot([xi[ipk]], [yis[ipk]], "o", ms=6, mfc="white", mec=MUTED,
                    mew=1.6, zorder=5)
            ax.text(xi[ipk], yis[ipk] + 0.05 * yr, "transient", ha="center",
                    va="bottom", size=10.2 * fs, color=MUTED,
                    path_effects=halo())
        lab = ("image as parent  —  stays at 0%" if flat else
               f"image as parent  —  decays to {yis[-1]:.0f}%")
        ax.text(xi[-1] - 0.012 * xr, yis[-1] + 0.075 * yr, lab, color=MUTED,
                ha="right", va="center", size=11.6 * fs, weight="bold",
                path_effects=halo(), zorder=6)
    elif ns.reverse_note:
        ax.text(0.985, 0.055, "the reverse ordering did not persist",
                transform=ax.transAxes, ha="right", va="bottom",
                size=10.6 * fs, color=MUTED, style="italic")

    if ns.mark:
        try:
            ep, lab = ns.mark.split(":", 1)
            ep = float(ep)
        except ValueError:
            ep, lab = float(ns.mark), ""
        ax.axvline(ep, color=FAINT, lw=1.1, ls=(0, (4, 3)), zorder=1)
        if lab:
            ax.text(ep + 0.008 * xr, 0.975, lab,
                    transform=blend(ax.transData, ax.transAxes), va="top",
                    ha="left", size=9.8 * fs, color=MUTED)

    ax.set_xlabel("training epoch")
    ax.set_ylabel(ns.metric_label)
    return fig


# ----------------------------------------------------------------------------
# R4 — geometry scoreboard (table)
# ----------------------------------------------------------------------------
def demo_r4():
    return {
        "columns": ["at init", "gate only", "image parent", "object parent"],
        "col_notes": ["clipped shell", "no entailment", "reverse", "ours"],
        "highlight": 3,
        "rows": [
            {"label": "exterior angle  \u2220(parent\u2192child)",
             "values": ["93.6\u00b0", "51\u2013144\u00b0", "94.6\u00b0", "12.2\u00b0"]},
            {"label": "cone half-aperture  \u03c9(parent)",
             "values": ["17.1\u00b0", "14.8\u201316.7\u00b0", "14.8\u00b0", "57.5\u00b0"]},
            {"label": "entailment-feasible pairs",
             "values": ["0.0%", "0\u201310%", "0.0%", "96.8%"], "strong": True},
            {"label": "depth gap  $(s_{img}-s_{obj})/c_r$",
             "values": ["0.08", "0.10", "0.00", "0.75"]},
            {"label": "clustering ACC (All)",
             "values": ["\u2014", "78.9", "79.5", "79.1"], "dim": True},
            {"label": "hierarchy usable", "kind": "verdict",
             "values": ["no", "no", "no", "yes"]},
        ],
        "foot": None,
    }


def _spec_load(path, need=("columns", "rows")):
    obj = json.load(open(path, encoding="utf-8"))
    missing = [k for k in need if k not in obj]
    if missing:
        die(f"{path}: JSON is missing {missing}")
    return obj


def fig_r4(ns):
    fs = ns.font_scale
    setup(fs)
    spec = demo_r4() if ns.demo else (
        _spec_load(ns.spec) if ns.spec else die("r4 needs --spec (or --demo)"))
    cols = spec["columns"]
    notes = spec.get("col_notes") or [""] * len(cols)
    rows = spec["rows"]
    hi = spec.get("highlight", len(cols) - 1)

    ncol = len(cols)
    W, H = ns.size or (196, 22 + 15.5 * len(rows))
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    x0 = 0.055
    kick = (track("SCOREBOARD") + "   \u00b7   ONE RECIPE, ONE KNOB: THE PARENT"
            if ns.kicker is None else ns.kicker)
    ttl = None if ns.no_title else (
        "Semantics, not the gate, opens the cone."
        if ns.title is None else ns.title)
    sub = None if ns.no_title else (
        "Identical pipeline and schedule; the columns differ only in what the "
        "entailment term is\nasked to do. Geometry read out at the final epoch."
        if ns.sub is None else ns.sub)
    foot = ns.foot if ns.foot is not None else spec.get("foot")
    top = header(fig, x0, kick, ttl, sub, check=True, fs=fs)
    ftop = footer(fig, x0, foot, fs=fs, demo=ns.demo)
    ax = fig.add_axes([x0, ftop, 0.955 - x0, top - ftop])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    lab_w = float(spec.get("label_width", 0.36))
    cw = (1.0 - lab_w) / ncol
    cx = [lab_w + cw * (i + 0.5) for i in range(ncol)]

    n = len(rows)
    head_h = 0.155 if any(notes) else 0.115
    row_h = (1.0 - head_h) / max(n, 1)
    ytop = 1.0 - head_h

    # highlight column band
    if 0 <= hi < ncol:
        ax.add_patch(plt.Rectangle((lab_w + cw * hi, 0.0), cw, 1.0,
                                   facecolor=mix(LEVEL["object"], "#FFFFFF",
                                                 0.93),
                                   edgecolor="none", zorder=0))
    # column headers
    for i, c in enumerate(cols):
        strong = (i == hi)
        ax.text(cx[i], 1.0 - 0.035, c, ha="center", va="top",
                size=(11.8 if strong else 11.2) * fs,
                weight="bold" if strong else "normal",
                color=INK if strong else "#4A5468", zorder=3)
        if notes[i]:
            ax.text(cx[i], 1.0 - head_h + 0.030, notes[i], ha="center",
                    va="bottom", size=9.6 * fs,
                    color=LEVEL["object"] if strong else MUTED,
                    style="italic", zorder=3)
    ax.plot([0, 1], [ytop, ytop], color=INK, lw=1.2, zorder=2)

    for r, row in enumerate(rows):
        yc = ytop - row_h * (r + 0.5)
        if r:
            ax.plot([0, 1], [ytop - row_h * r] * 2, color=GRID, lw=0.9,
                    zorder=1)
        ax.text(0.0, yc, row["label"], ha="left", va="center",
                size=11.4 * fs, color=INK if not row.get("dim") else "#4A5468",
                zorder=3)
        verdict = row.get("kind") == "verdict"
        for i, v in enumerate(row["values"]):
            strong = (i == hi)
            if verdict:
                yes = str(v).strip().lower() in ("yes", "true", "1", "\u2713")
                ax.text(cx[i], yc, "\u2713" if yes else "\u2717", ha="center",
                        va="center", size=15.5 * fs, weight="bold",
                        color=OK if yes else FAINT, zorder=3)
                continue
            size = (13.2 if (strong and row.get("strong")) else
                    12.2 if strong else 11.6) * fs
            col = (INK if strong else ("#6B7488" if row.get("dim")
                                       else "#3A4256"))
            if strong and row.get("strong"):
                col = LEVEL["object"]
            ax.text(cx[i], yc, str(v), ha="center", va="center", size=size,
                    weight="bold" if strong else "normal", color=col, zorder=3)
    ax.plot([0, 1], [0, 0], color=FAINT, lw=1.0, zorder=2)
    return fig


# ----------------------------------------------------------------------------
# F8 — induced tree (dendrogram over class means, hyperbolic metric)
# ----------------------------------------------------------------------------
def average_linkage(D):
    """Average-linkage agglomeration on a square distance matrix.

    Returns merges [[a, b, height], ...] in scipy-linkage index convention
    (leaves 0..n-1, merge k creates node n+k). Pure numpy, n <= ~1000.
    """
    D = np.array(D, float)
    n = D.shape[0]
    np.fill_diagonal(D, np.inf)
    size = np.ones(n)
    ids = list(range(n))
    active = np.ones(n, bool)
    merges = []
    for k in range(n - 1):
        idx = np.where(active)[0]
        sub = D[np.ix_(idx, idx)]
        p = int(np.argmin(sub))
        i, j = idx[p // len(idx)], idx[p % len(idx)]
        h = float(D[i, j])
        merges.append([ids[i], ids[j], h])
        ni, nj = size[i], size[j]
        newrow = (ni * D[i, :] + nj * D[j, :]) / (ni + nj)
        newrow[i] = newrow[j] = np.inf
        D[i, :] = newrow
        D[:, i] = newrow
        D[i, i] = np.inf
        active[j] = False
        D[j, :] = np.inf
        D[:, j] = np.inf
        size[i] = ni + nj
        ids[i] = n + k
    return merges


def _tree_layout(merges, n):
    """Return (xy per node, leaf order). x = -height, y = slot index."""
    kids = {}
    height = {}
    for k, (a, b, h) in enumerate(merges):
        kids[n + k] = (int(a), int(b))
        height[n + k] = float(h)
    root = n + len(merges) - 1 if merges else 0

    order = []

    def walk(node):
        if node < n:
            order.append(node)
            return
        a, b = kids[node]
        # keep the bigger subtree on the outside for a calmer silhouette
        sa = 1 if a < n else 2
        walk(a)
        walk(b)
        _ = sa

    walk(root)
    ypos = {leaf: i for i, leaf in enumerate(order)}
    for k in range(len(merges)):
        node = n + k
        a, b = kids[node]
        ypos[node] = 0.5 * (ypos[a] + ypos[b])
    xpos = {i: 0.0 for i in range(n)}
    for k in range(len(merges)):
        xpos[n + k] = -height[n + k]
    return xpos, ypos, kids, order, root


def demo_f8(seed=3):
    rng = np.random.default_rng(seed)
    groups = [("Warbler", 4), ("Sparrow", 4), ("Tern", 3), ("Vireo", 3),
              ("Woodpecker", 3), ("Oriole", 3)]
    labels, parents, is_new = [], [], []
    for g, k in groups:
        for i in range(k):
            labels.append(f"{g} {i + 1}")
            parents.append(g)
            is_new.append(int(rng.random() < 0.42))
    n = len(labels)
    # latent 2-D "semantic" coordinates: tight genus clusters, wide spread
    ctr = {g: rng.normal(0, 1.6, 2) for g, _ in groups}
    X = np.array([ctr[p] + rng.normal(0, 0.30, 2) for p in parents])
    for i, j in ((1, 7), (12, 17)):        # two genuinely hard classes:
        X[i] = X[j] + rng.normal(0, 0.22, 2)   # land next to another genus
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    merges = average_linkage(D)
    nn = np.argsort(D + np.eye(n) * 1e9, axis=1)[:, 0]
    correct = [int(parents[i] == parents[nn[i]]) if is_new[i] else -1
               for i in range(n)]
    shown = [i for i in range(n) if is_new[i]]
    rate = 100.0 * np.mean([correct[i] for i in shown]) if shown else 0.0
    return {"labels": labels, "is_new": is_new, "correct": correct,
            "merges": merges,
            "stats": {"rate_shown": rate, "n_new_shown": len(shown),
                      "n_leaves": n, "dataset": "CUB", "groups": len(groups)}}


def fig_f8(ns):
    fs = ns.font_scale
    setup(fs)
    spec = demo_f8() if ns.demo else (
        _spec_load(ns.spec, ("labels", "merges")) if ns.spec
        else die("f8 needs --spec (or --demo)"))
    labels = list(spec["labels"])
    is_new = list(spec["is_new"])
    correct = list(spec.get("correct", [-1] * len(labels)))
    merges = [[int(a), int(b), float(h)] for a, b, h in spec["merges"]]
    st = spec.get("stats", {})
    n = len(labels)
    if len(merges) != n - 1:
        die(f"f8: got {len(merges)} merges for {n} leaves (need n-1)")

    xpos, ypos, kids, order, root = _tree_layout(merges, n)
    hmax = max(-xpos[k] for k in xpos) or 1.0

    W, H = ns.size or (196, max(102.0, 26.0 + 6.4 * n))
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    x0 = 0.055
    kick = (track("P3") + "   \u00b7   DOES THE TREE MEAN ANYTHING?"
            if ns.kicker is None else ns.kicker)
    ttl = None if ns.no_title else (
        "Novel classes attach where they belong."
        if ns.title is None else ns.title)
    sub = None if ns.no_title else (
        "Class means agglomerated under the learned hyperbolic metric.\n"
        "No taxonomy was supervised; orange leaves were never labelled."
        if ns.sub is None else ns.sub)
    foot = ns.foot
    if foot is None:
        ds = st.get("dataset", "")
        foot = ("average linkage on Poincar\u00e9 distances between class "
                "means" + (f"   \u00b7   {ds}" if ds else "")
                + "\na novel leaf counts as correctly attached when its "
                  "nearest class shares the ground-truth genus")
    top = header(fig, x0, kick, ttl, sub, check=True, fs=fs)
    ftop = footer(fig, x0, foot, fs=fs, demo=ns.demo)
    yb = ftop + _dy(fig, 26)
    ax = fig.add_axes([0.045, yb, 0.575, top - yb])

    lw = 1.5
    for k in range(len(merges)):
        node = n + k
        a, b = kids[node]
        xp, yp = xpos[node], ypos[node]
        for ch in (a, b):
            ax.plot([xpos[ch], xp], [ypos[ch]] * 2, color=FAINT, lw=lw,
                    solid_capstyle="round", zorder=1)
        ax.plot([xp, xp], [ypos[a], ypos[b]], color=FAINT, lw=lw,
                solid_capstyle="round", zorder=1)

    pad = 0.045 * hmax
    for leaf in order:
        y = ypos[leaf]
        new = bool(is_new[leaf])
        col = DISC if new else SUP
        ok = correct[leaf] == 1
        if new and ok:
            ax.plot([0.0], [y], marker="o", ms=9.5, mfc="none", mec=OK,
                    mew=1.8, zorder=3, clip_on=False)
        ax.plot([0.0], [y], marker="o", ms=4.6, color=col, zorder=4,
                clip_on=False)
        ax.text(pad, y, labels[leaf], ha="left", va="center",
                size=10.6 * fs, color=col if new else "#3A4256",
                weight="bold" if new else "normal", zorder=4, clip_on=False)

    ax.set_xlim(-hmax * 1.06, hmax * 0.62)
    ax.set_ylim(-0.9, n - 0.1)
    ax.set_yticks([])
    ticks = [-hmax, -hmax / 2, 0.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):.1f}" for t in ticks])
    strip_spines(ax, keep=("bottom",))
    ax.spines["bottom"].set_bounds(-hmax, 0.0)
    ax.set_xlabel("hyperbolic distance between class means")

    # legend + rate badge, right column
    lx = 0.665
    ly = top - _dy(fig, 6)
    rate = st.get("rate_shown")
    if rate is not None:
        fig.text(lx, ly, f"{rate:.0f}%", size=32 * fs, weight="bold",
                 color=OK, va="top", ha="left")
        ly -= _dy(fig, 32) * 1.05
        fig.text(lx, ly, "of the novel classes\nshown attach under\nthe "
                         "right genus", size=10.8 * fs, color=INK, va="top",
                 ha="left", linespacing=1.35)
        ly -= _dy(fig, 10.8) * 3 * 1.35 + _dy(fig, 12)
    for tag, col, txt in ((None, SUP, "labelled (known)"),
                          (None, DISC, "novel \u2014 discovered"),
                          ("ring", OK, "right genus")):
        if tag == "ring":
            fig.text(lx + 0.006, ly, "\u25cb", size=13.5 * fs, color=OK,
                     va="top", ha="left", weight="bold")
        else:
            fig.text(lx + 0.008, ly, "\u25cf", size=11.0 * fs, color=col,
                     va="top", ha="left")
        fig.text(lx + 0.030, ly, txt, size=10.6 * fs, color="#4A5468",
                 va="top", ha="left")
        ly -= _dy(fig, 11.0) * 1.9
    extra = st.get("note")
    if extra:
        fig.text(lx, ly - _dy(fig, 4), extra, size=9.8 * fs, color=MUTED,
                 va="top", ha="left", linespacing=1.35, style="italic")
    return fig


# ----------------------------------------------------------------------------
# saving + CLI
# ----------------------------------------------------------------------------
DEFAULT_NAME = {
    "f3a": "F3a_curvature_tug",
    "f3b": "F3b_shell",
    "f5": "F5_depth_stratifies",
    "f6": "F6_direction_selectivity",
    "f8": "F8_induced_tree",
    "r4": "R4_geometry_scoreboard",
}


def save(fig, ns, cmd):
    out = ns.out or ("figs/" + DEFAULT_NAME[cmd])
    p = Path(out)
    if out.endswith(("/", os.sep)) or p.is_dir():
        p = p / DEFAULT_NAME[cmd]
    p.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for f in [s.strip().lower() for s in ns.formats.split(",") if s.strip()]:
        fp = p.parent / (p.name + "." + f)
        fig.savefig(fp, dpi=ns.dpi, transparent=ns.transparent)
        written.append(str(fp))
    plt.close(fig)
    print("[poster_viz] " + "   ".join("→ " + w for w in written))
    return written


def parse_size(s):
    try:
        w, h = s.lower().replace("mm", "").split("x")
        return (float(w), float(h))
    except Exception:
        raise argparse.ArgumentTypeError("size must look like 200x120 (mm)")


def add_common(p):
    p.add_argument("-o", "--out", help="output base path or directory")
    p.add_argument("--formats", default="pdf,png")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--size", type=parse_size, default=None,
                   help="figure size in mm, e.g. 200x120")
    p.add_argument("--font-scale", type=float, default=1.0)
    p.add_argument("--demo", action="store_true",
                   help="render with synthetic preview data")
    p.add_argument("--no-title", action="store_true",
                   help="drop title+subtitle (poster supplies them)")
    p.add_argument("--kicker", default=None)
    p.add_argument("--title", default=None)
    p.add_argument("--sub", default=None)
    p.add_argument("--foot", default=None)
    p.add_argument("--transparent", action="store_true")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="poster_viz.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("f3a", help="curvature tug-of-war")
    add_common(p)
    p.add_argument("--table", help="csv/tsv/json/npz with the trajectories")
    p.add_argument("--x-col", default="epoch")
    p.add_argument("--sup-col", default="c_sup")
    p.add_argument("--unsup-col", default="c_unsup")
    p.add_argument("--smooth", type=int, default=7)
    p.add_argument("--unbind", type=float, default=None,
                   help="epoch at which curvatures were unbound")
    p.add_argument("--probe-rep", type=float, default=None)
    p.add_argument("--probe-cls", type=float, default=None)

    p = sub.add_parser("f3b", help="shell distribution")
    add_common(p)
    p.add_argument("--radii", help="array of per-sample norms")
    p.add_argument("--key", default=None, help="npz key / csv column")
    p.add_argument("--R", type=float, default=None, help="clip radius")
    p.add_argument("--extra", default=None,
                   help="optional second array (e.g. representation space)")
    p.add_argument("--extra-key", default=None)
    p.add_argument("--extra-label", default="representation space")
    p.add_argument("--K", type=float, default=0.145)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--bw", type=float, default=None)
    p.add_argument("--bw-floor", type=float, default=0.0035)
    p.add_argument("--feasible", type=float, default=0.0)
    p.add_argument("--no-aperture", action="store_true")
    p.add_argument("--no-footnote", action="store_true")

    p = sub.add_parser("f5", help="depth stratifies (gate off → on)")
    add_common(p)
    for a in ("off-object", "off-image", "on-part", "on-object", "on-image"):
        p.add_argument("--" + a)
    p.add_argument("--key", default=None)
    p.add_argument("--R", type=float, default=None)
    for a in ("anchor-part", "anchor-object", "anchor-image"):
        p.add_argument("--" + a, type=float, default=None)
    p.add_argument("--feas-off", type=float, default=None)
    p.add_argument("--feas-on", type=float, default=None)
    p.add_argument("--bw", type=float, default=None)
    p.add_argument("--bw-floor", type=float, default=0.0035)
    p.add_argument("--schematic", action="store_true",
                   help="qualitative depth axis: keep the learned order but "
                        "draw the ON annuli at display positions; no anchor "
                        "values, single tick at the shell")
    p.add_argument("--no-feas", action="store_true",
                   help="drop the feasibility badge (avoid duplicating F6)")
    for a in ("pos-part", "pos-object", "pos-image",
              "sd-part", "sd-object", "sd-image"):
        p.add_argument("--" + a, type=float, default=None)

    p = sub.add_parser("f6", help="direction selectivity")
    add_common(p)
    p.add_argument("--obj", help="table for the object-as-parent run")
    p.add_argument("--obj-col", default="feasible_pct")
    p.add_argument("--img", default=None,
                   help="table for the image-as-parent run (optional)")
    p.add_argument("--img-col", default=None)
    p.add_argument("--x-col", default="epoch")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--metric-label",
                   default="feasible pairs  (%)")
    p.add_argument("--mark", default=None, help="EPOCH[:label] vertical mark")
    p.add_argument("--no-reverse-note", dest="reverse_note",
                   action="store_false")

    p = sub.add_parser("f8", help="induced tree (dendrogram of class means)")
    add_common(p)
    p.add_argument("--spec", help="tree JSON from poster_extract.py f8")

    p = sub.add_parser("r4", help="geometry scoreboard table")
    add_common(p)
    p.add_argument("--spec", help="table JSON from poster_extract.py r4")

    p = sub.add_parser("all", help="render all (typically with --demo)")
    add_common(p)
    return ap


FIG_FN = {"f3a": fig_f3a, "f3b": fig_f3b, "f5": fig_f5, "f6": fig_f6,
          "f8": fig_f8, "r4": fig_r4}


def run(argv):
    ns = build_parser().parse_args(argv)
    if ns.cmd == "all":
        outdir = ns.out or "preview/"
        if not outdir.endswith(("/", os.sep)):
            outdir += os.sep
        extra = []
        if ns.demo:
            extra.append("--demo")
        for k, v in (("--formats", ns.formats), ("--dpi", str(ns.dpi)),
                     ("--font-scale", str(ns.font_scale))):
            extra += [k, v]
        if ns.transparent:
            extra.append("--transparent")
        if ns.no_title:
            extra.append("--no-title")
        demo_extras = {"f3a": ["--probe-rep", "1.10", "--probe-cls", "0.47"],
                       "f3b": [], "f5": [], "f6": [], "f8": [], "r4": []}
        for cmd in ("f3a", "f3b", "f5", "f6", "f8", "r4"):
            run([cmd, "-o", outdir] + extra + demo_extras[cmd])
        return
    fig = FIG_FN[ns.cmd](ns)
    save(fig, ns, ns.cmd)


if __name__ == "__main__":
    run(sys.argv[1:])
