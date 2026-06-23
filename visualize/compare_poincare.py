#!/usr/bin/env python3
"""Compare Poincaré-ball metrics across several experiments.

Reads the ``{rep,cls}_metrics.json`` files produced by ``visualize_poincare.py``
for a set of experiment folders and prints a side-by-side table plus a grouped
bar chart.

The metrics are split into two groups, because **only some are comparable
across experiments that used different curvature c**:

  SCALE-FREE  (safe to compare across different c / cr)
    - delta_rel       : relative Gromov delta (lower = more tree-like)
    - cophenetic_corr : how clean a tree the class-means form (higher = better)
    - MRL             : angular mean-resultant length (0 = uniform directions)
    - cos_abs_mean    : mean |cos| between directions

  C-DEPENDENT  (do NOT compare raw across different c -- the ball radius
                R = 1/sqrt(c) and the tanh squashing change the scale)
    - rel_norm_mean / frac>0.9R : how far out points sit (radial extension)
    - depth_mean                : absolute geodesic depth
    - pair_dist_mean            : absolute pairwise hyperbolic distance
    - sep_over_align            : ratio of hyperbolic distances (mostly, but not
                                  perfectly, c-invariant)

To compare radial extension fairly across different c, re-run the visualizer
with the SAME --c for the experiments being compared (a diagnostic geometry,
not each model's trained geometry), or rely on the scale-free block above.

Usage:
  python compare_poincare.py EXP_DIR1 EXP_DIR2 ...            # explicit folders
  python compare_poincare.py --glob "/path/to/pic/*"          # all subfolders
  python compare_poincare.py EXP1 EXP2 --ball cls --out cmp.png
"""
import os, sys, json, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCALE_FREE = [
    ("delta_rel",       ("delta_hyperbolicity", "delta_rel_mean"), "lower=tree-like"),
    ("cophenetic_corr", ("hierarchy", "cophenetic_corr"),          "higher=clean tree"),
    ("MRL",             ("directional_uniformity", "mean_resultant_length"), "0=uniform dir"),
    ("cos_abs_mean",    ("directional_uniformity", "cos_abs_mean"), "0=uniform dir"),
]
C_DEPENDENT = [
    ("rel_norm_mean",   ("radial", "rel_norm_mean"),    "frac of radius"),
    ("frac>0.9R",       ("radial", "frac_near_boundary_0.9R"), "near-boundary frac"),
    ("depth_mean",      ("radial", "depth_mean"),       "abs depth"),
    ("pair_dist_mean",  ("distance_uniformity", "pair_dist_mean"), "abs hyp dist"),
    ("sep_over_align",  ("alignment_separation", "separation_over_alignment"), "higher=separated"),
]
# known/novel split helps answer "does novel extend outward as much as known?"
KNOWN_NOVEL = [
    ("cmean_depth_known", ("hierarchy", "class_mean_depth_known_mean")),
    ("cmean_depth_novel", ("hierarchy", "class_mean_depth_novel_mean")),
]


def _dig(d, path):
    for k in path:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def load_exp(folder, ball):
    p = os.path.join(folder, f"{ball}_metrics.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def short_name(folder):
    return os.path.basename(os.path.normpath(folder))


def fmt(v):
    if v is None:
        return "   --   "
    if isinstance(v, float):
        return f"{v:8.3f}"
    return f"{str(v):>8}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare Poincaré metrics across experiments.")
    ap.add_argument("folders", nargs="*", help="experiment folders (each with *_metrics.json)")
    ap.add_argument("--glob", default=None, help="glob pattern for folders, e.g. '/path/pic/*'")
    ap.add_argument("--ball", choices=["rep", "cls", "both"], default="both")
    ap.add_argument("--out", default="poincare_comparison.png")
    args = ap.parse_args(argv)

    folders = list(args.folders)
    if args.glob:
        folders += [g for g in sorted(glob.glob(args.glob)) if os.path.isdir(g)]
    folders = [f for f in folders if os.path.isdir(f)]
    if not folders:
        ap.error("no experiment folders given (use positional paths or --glob)")

    balls = ["rep", "cls"] if args.ball == "both" else [args.ball]

    for ball in balls:
        exps = [(short_name(f), load_exp(f, ball)) for f in folders]
        exps = [(n, m) for n, m in exps if m is not None]
        if not exps:
            print(f"\n[{ball}] no metrics found in any folder, skipping.")
            continue

        names = [n for n, _ in exps]
        print("\n" + "=" * 78)
        print(f" BALL = {ball.upper()}     ({len(exps)} experiments)")
        print("=" * 78)
        # header row: curvature c first
        print(f"{'metric':>18} | " + " | ".join(f"{n[:14]:>14}" for n in names))
        print(f"{'c (curvature)':>18} | " + " | ".join(fmt(m.get('curvature_c')).rjust(14) for _, m in exps))
        print(f"{'dim':>18} | " + " | ".join(fmt(m.get('dim')).rjust(14) for _, m in exps))
        print("-" * 78)
        print(" SCALE-FREE  (comparable across different c):")
        for label, path, hint in SCALE_FREE:
            vals = [_dig(m, path) for _, m in exps]
            print(f"{label:>18} | " + " | ".join(fmt(v).rjust(14) for v in vals) + f"    # {hint}")
        print("-" * 78)
        print(" C-DEPENDENT  (do NOT compare raw across different c):")
        for label, path, hint in C_DEPENDENT:
            vals = [_dig(m, path) for _, m in exps]
            print(f"{label:>18} | " + " | ".join(fmt(v).rjust(14) for v in vals) + f"    # {hint}")
        print("-" * 78)
        print(" KNOWN vs NOVEL class-mean depth (within-experiment; gap is informative):")
        for label, path in KNOWN_NOVEL:
            vals = [_dig(m, path) for _, m in exps]
            print(f"{label:>18} | " + " | ".join(fmt(v).rjust(14) for v in vals))
        gaps = []
        for _, m in exps:
            k = _dig(m, ("hierarchy", "class_mean_depth_known_mean"))
            nv = _dig(m, ("hierarchy", "class_mean_depth_novel_mean"))
            gaps.append((k - nv) if (k is not None and nv is not None) else None)
        print(f"{'known-novel gap':>18} | " + " | ".join(fmt(v).rjust(14) for v in gaps))

    # ---- figure: grouped bars for the scale-free metrics on the cls ball ----
    fig_balls = balls
    fig, axes = plt.subplots(len(fig_balls), 1, figsize=(max(7, 1.6 * len(folders) + 4),
                                                          4.2 * len(fig_balls)), squeeze=False)
    for bi, ball in enumerate(fig_balls):
        ax = axes[bi, 0]
        exps = [(short_name(f), load_exp(f, ball)) for f in folders]
        exps = [(n, m) for n, m in exps if m is not None]
        if not exps:
            ax.set_visible(False)
            continue
        names = [n for n, _ in exps]
        metrics = SCALE_FREE
        x = np.arange(len(metrics))
        w = 0.8 / max(1, len(exps))
        for ei, (n, m) in enumerate(exps):
            vals = [(_dig(m, p) or 0.0) for _, p, _ in metrics]
            ax.bar(x + ei * w, vals, w, label=f"{n[:18]} (c={m.get('curvature_c')})")
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels([lbl for lbl, _, _ in metrics])
        ax.set_title(f"[{ball}] scale-free metrics (comparable across c)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"\n[figure] grouped bar chart -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()