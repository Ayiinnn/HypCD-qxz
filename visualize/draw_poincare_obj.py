#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poincaré-ball visualizer for the **object-branch** SimGCD variant
(``train_HypSimGCD_org_det_ab_obj.py``).

That variant adds a second, *object-level* branch that shares the backbone,
projector and classifier with the image branch: each image is turned into a
single foreground crop by ``models.foreground.ForegroundCropper`` and pushed
through the **same** modules, so the image features and the object (foreground)
features live in the **same** Poincaré ball.  Three losses tie the branches
together (``models.object_branch``):
  * entailment cone   (rep space) -- child must lie inside the parent's cone,
  * InfoNCE hyp-dist  (rep space) -- image_i pulled close to its own crop obj_i,
  * classification closeness (logit space) -- same label distribution.

This script (run it from anywhere inside the repo)::

    python visualize_poincare_obj.py --dataset_name cub --dino v1 \
        --model_path .../checkpoints/model_best_acc_all.pt --c 0.05 \
        --out_dir ./pic_obj

produces, for the SAME checkpoint:
  1.  the object branch on its own            (obj_rep_*, obj_cls_* : reuses the
      standard overview/hierarchy from ``visualize_poincare``);
  2.  an image-vs-object comparison in the ball (twobranch_rep.png / .json):
      overlaid depth-faithful disk, depth histograms, paired-distance vs
      shuffled baseline, entailment-cone satisfaction, per-class mean offsets;
  3.  an image<->object correspondence figure (correspondence_rep.png): a sample
      of image_i--obj_i pairs joined in the disk + a depth-shift scatter showing
      whether the crop moves deeper (as the entailment loss intends).

The image (original) branch by itself is already covered by ``visualize_poincare``
(``--method simgcd``); this file is only for the object branch and the two-branch
relationships.  Poincaré ball is used for all spatial plots; the entailment
diagnostic reuses the model's own Lorentz-bridge primitives
(``hyptorch.entailment``).
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Reuse everything from the base visualizer (geometry, embedding, module
# reconstruction, robust checkpoint loading, dataloader, analyze_ball, ...).
import draw_poincare as V
import hyptorch.pmath as pmath
from hyptorch.entailment import (
    poincare_to_lorentz, half_aperture, oxy_angle,
)

import torch

log = V.log


# --------------------------------------------------------------------------- #
# Foreground cropper + dual (image / object) feature extraction
# --------------------------------------------------------------------------- #
def build_fg_cropper(backbone, args):
    """Construct the same ForegroundCropper the trainer uses."""
    from models.foreground import ForegroundCropper
    return ForegroundCropper(
        backbone, model_name=args.dino, source=args.obj_fg_source,
        keep=args.obj_fg_keep, box_pad=args.obj_fg_pad, out_size=args.image_size,
    )


@torch.no_grad()
def extract_dual(modules, loader, device, fg_cropper, want_cls=True,
                 max_samples=4000):
    """Run each batch through the image path AND the foreground-crop path.

    Returns dict of numpy arrays, all index-aligned (row i of *_obj is the crop
    of row i of *_img): rep_img, rep_obj, cls_img, cls_obj, labels, preds_img,
    preds_obj.
    """
    backbone = modules["backbone"].to(device).eval()
    projector = modules["projector"]
    classifier = modules["classifier"]
    if projector is not None:
        projector = projector.to(device).eval()
    if classifier is not None:
        classifier = classifier.to(device).eval()

    rep_i, rep_o, cls_i, cls_o, labs = [], [], [], [], []
    pred_i, pred_o = [], []
    n = 0
    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        # ---- image branch ----
        feat_i = backbone(images)
        ri = projector(feat_i) if projector is not None else feat_i
        # ---- object branch (foreground crop through the SAME modules) ----
        obj_images = fg_cropper(images)
        feat_o = backbone(obj_images)
        ro = projector(feat_o) if projector is not None else feat_o

        rep_i.append(V.to_np(ri)); rep_o.append(V.to_np(ro))
        labs.append(V.to_np(labels).reshape(-1))
        if want_cls and classifier is not None:
            li = classifier(ri); lo = classifier(ro)
            cls_i.append(V.to_np(li)); cls_o.append(V.to_np(lo))
            pred_i.append(V.to_np(li.argmax(1)).reshape(-1))
            pred_o.append(V.to_np(lo.argmax(1)).reshape(-1))
        n += images.shape[0]
        if max_samples and n >= max_samples:
            break

    out = dict(
        rep_img=np.concatenate(rep_i, 0).astype(np.float64),
        rep_obj=np.concatenate(rep_o, 0).astype(np.float64),
        labels=np.concatenate(labs, 0).astype(np.int64),
        cls_img=np.concatenate(cls_i, 0).astype(np.float64) if cls_i else None,
        cls_obj=np.concatenate(cls_o, 0).astype(np.float64) if cls_o else None,
        preds_img=np.concatenate(pred_i, 0).astype(np.int64) if pred_i else None,
        preds_obj=np.concatenate(pred_o, 0).astype(np.int64) if pred_o else None,
    )
    if max_samples and out["rep_img"].shape[0] > max_samples:
        for k, v in out.items():
            if v is not None:
                out[k] = v[:max_samples]
    return out


# --------------------------------------------------------------------------- #
# Geometry helpers for two-branch plots
# --------------------------------------------------------------------------- #
def rel_norm_from_depth(d, c):
    """rel-norm = ||x||/R recovered from geodesic depth d and curvature c."""
    return np.tanh(math.sqrt(c) * np.asarray(d) / 2.0)


def paired_hyp_distance(A, B, c):
    """Row-wise Poincaré distance between matched rows of A and B -> (n,)."""
    At = pmath.project(V.as_tensor(A), c=c)
    Bt = pmath.project(V.as_tensor(B), c=c)
    d = pmath.dist(At, Bt, c=c)
    return V.to_np(d).reshape(-1)


def shared_disk_embedding(P_img, P_obj, c, seed=0):
    """Depth-faithful 2D coords for both branches in ONE shared frame.

    radius = true rel-norm of each point; angle from an origin-preserving SVD of
    the *union* of tangent vectors (so the two clouds share orientation).
    """
    from sklearn.decomposition import TruncatedSVD
    Ti = V.tangent_at_origin(P_img, c)
    To = V.tangent_at_origin(P_obj, c)
    k = min(2, Ti.shape[1])
    svd = TruncatedSVD(n_components=max(k, 1), random_state=seed)
    svd.fit(np.concatenate([Ti, To], axis=0))
    ui = svd.transform(Ti); uo = svd.transform(To)
    if ui.shape[1] < 2:
        ui = np.concatenate([ui, np.zeros((ui.shape[0], 1))], 1)
        uo = np.concatenate([uo, np.zeros((uo.shape[0], 1))], 1)

    def _polar(u, P):
        ang = np.arctan2(u[:, 1], u[:, 0])
        rel = np.linalg.norm(V.to_np(P), axis=1) / V.ball_radius(c)
        return np.stack([rel * np.cos(ang), rel * np.sin(ang)], axis=1)

    return _polar(ui, P_img), _polar(uo, P_obj)


def entailment_diagnostics(parent, child, c, aperture_scale, min_radius):
    """Per-pair entailment: exterior angle vs cone half-aperture (full-D, exact).

    Returns (angle, aperture_scale*aper, violation) in radians, where
    violation = angle - aperture_scale*aper  (<= 0  => child inside the cone).
    """
    Pp = pmath.project(V.as_tensor(parent), c=c)
    Cp = pmath.project(V.as_tensor(child), c=c)
    Pl = poincare_to_lorentz(Pp, c)
    Cl = poincare_to_lorentz(Cp, c)
    ang = oxy_angle(Pl, Cl, curv=c)
    aper = half_aperture(Pl, curv=c, min_radius=min_radius) * aperture_scale
    ang = V.to_np(ang).reshape(-1)
    aper = V.to_np(aper).reshape(-1)
    return ang, aper, ang - aper


# --------------------------------------------------------------------------- #
# (b) two-branch comparison
# --------------------------------------------------------------------------- #
def plot_two_branch(data, c, num_labeled, out_dir, args):
    rep_i, rep_o = data["rep_img"], data["rep_obj"]
    labels = data["labels"]
    is_old = labels < num_labeled
    R = V.ball_radius(c)
    rng = np.random.default_rng(args.seed)

    d_i = V.to_np(pmath.dist0(V.as_tensor(rep_i), c=c)).reshape(-1)
    d_o = V.to_np(pmath.dist0(V.as_tensor(rep_o), c=c)).reshape(-1)
    reln_i, reln_o = rel_norm_from_depth(d_i, c), rel_norm_from_depth(d_o, c)

    # paired vs shuffled hyperbolic distance (InfoNCE pulls matched pairs close)
    d_pair = paired_hyp_distance(rep_i, rep_o, c)
    perm = rng.permutation(len(rep_o))
    d_shuf = paired_hyp_distance(rep_i, rep_o[perm], c)

    # entailment cone satisfaction (parent/child per training config)
    if args.obj_entail_parent == "image":
        parent, child = rep_i, rep_o
    else:
        parent, child = rep_o, rep_i
    ang, aper, viol = entailment_diagnostics(
        parent, child, c, args.obj_aperture_scale, args.obj_min_radius)
    sat = float(np.mean(viol <= 0))

    # cls agreement (do image and crop predict the same class?)
    agree = None
    if data["preds_img"] is not None and data["preds_obj"] is not None:
        agree = float(np.mean(data["preds_img"] == data["preds_obj"]))

    xy_i, xy_o = shared_disk_embedding(rep_i, rep_o, c, seed=args.seed)

    fig, ax = plt.subplots(2, 3, figsize=(16.5, 11))

    # (0,0) overlaid depth-faithful disk
    a = ax[0, 0]; V._draw_disk_axes(a)
    a.scatter(xy_i[:, 0], xy_i[:, 1], s=6, c="#1f77b4", alpha=0.45,
              linewidths=0, label="image", zorder=3)
    a.scatter(xy_o[:, 0], xy_o[:, 1], s=6, c="#ff7f0e", alpha=0.45,
              linewidths=0, label="object (fg)", zorder=3)
    a.legend(fontsize=9, loc="upper right", framealpha=0.7)
    a.set_title("Both branches in the same ball\n(radius = TRUE depth)", fontsize=11)

    # (0,1) depth histogram image vs object
    a = ax[0, 1]
    bins = np.linspace(min(d_i.min(), d_o.min()), max(d_i.max(), d_o.max()) + 1e-9, 40)
    a.hist(d_i, bins=bins, alpha=0.55, color="#1f77b4", density=True, label="image")
    a.hist(d_o, bins=bins, alpha=0.55, color="#ff7f0e", density=True, label="object")
    a.axvline(d_i.mean(), color="#1f77b4", ls="--", lw=1)
    a.axvline(d_o.mean(), color="#ff7f0e", ls="--", lw=1)
    a.set_xlabel("geodesic depth  d(0,x)"); a.set_ylabel("density")
    a.set_title(f"Depth: image={d_i.mean():.2f}  object={d_o.mean():.2f}\n"
                f"(deeper object ⇒ more specific / entailment child)", fontsize=10)
    a.legend(fontsize=9)

    # (0,2) paired vs shuffled distance
    a = ax[0, 2]
    bb = np.linspace(0, max(d_pair.max(), np.percentile(d_shuf, 99)) + 1e-9, 40)
    a.hist(d_shuf, bins=bb, alpha=0.5, color="#999999", density=True,
           label=f"shuffled (mean={d_shuf.mean():.2f})")
    a.hist(d_pair, bins=bb, alpha=0.7, color="#4c72b0", density=True,
           label=f"matched i↔i (mean={d_pair.mean():.2f})")
    a.set_xlabel("hyperbolic distance  d(image_i, object_j)")
    a.set_ylabel("density")
    a.set_title("Image↔its-own-crop distance vs random pairs\n"
                "(matched ≪ shuffled ⇒ InfoNCE worked)", fontsize=10)
    a.legend(fontsize=8)

    # (1,0) entailment: angle vs aperture
    a = ax[1, 0]
    a.hist(viol, bins=40, color="#55a868", alpha=0.85, density=True)
    a.axvline(0.0, color="k", ls="--", lw=1)
    a.set_xlabel("oxy_angle(parent,child) − scale·half_aperture(parent)  [rad]")
    a.set_ylabel("density")
    a.set_title(f"Entailment cone (parent={args.obj_entail_parent})\n"
                f"{sat*100:.1f}% of children inside the cone (≤0)", fontsize=10)

    # (1,1) per-class mean offset: image-mean vs object-mean distance
    a = ax[1, 1]
    Mi, ids_i = V.poincare_class_means(rep_i, labels, c)
    Mo, ids_o = V.poincare_class_means(rep_o, labels, c)
    # both use np.unique(labels) in the same order, so rows are class-aligned
    off = paired_hyp_distance(V.to_np(Mi), V.to_np(Mo), c)
    if off.size:
        a.hist(off, bins=30, color="#c44e52", alpha=0.85, density=True)
        a.axvline(off.mean(), color="k", ls="--", lw=1, label=f"mean={off.mean():.2f}")
        a.legend(fontsize=9)
    a.set_xlabel("d(image class-mean, object class-mean)")
    a.set_ylabel("density")
    a.set_title("Per-class branch offset\n(small ⇒ branches agree per class)", fontsize=10)

    # (1,2) text summary
    a = ax[1, 2]; a.axis("off")
    lines = [
        "[image vs object — representation ball]",
        f"  curvature c        : {c:g}   (R={R:.3f})",
        f"  points / dim        : {len(rep_i)} / {rep_i.shape[1]}",
        f"  depth  image        : {d_i.mean():.3f} ± {d_i.std():.3f}",
        f"  depth  object       : {d_o.mean():.3f} ± {d_o.std():.3f}",
        f"  Δdepth (obj−img)    : {(d_o-d_i).mean():+.3f}  "
        f"(obj deeper for {np.mean(d_o>d_i)*100:.0f}% of pairs)",
        f"  rel-norm img / obj  : {reln_i.mean():.3f} / {reln_o.mean():.3f}",
        f"  matched dist        : {d_pair.mean():.3f}",
        f"  shuffled dist       : {d_shuf.mean():.3f}",
        f"  matched < shuffled  : {np.mean(d_pair < d_shuf)*100:.1f}% of rows",
        f"  entailment parent   : {args.obj_entail_parent}",
        f"  cone satisfied      : {sat*100:.1f}%",
        f"  per-class offset    : {off.mean():.3f}" if off.size else "  per-class offset    : --",
    ]
    if agree is not None:
        lines.append(f"  cls agree (img=obj) : {agree*100:.1f}%")
    a.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
           fontsize=10, transform=a.transAxes)

    fig.suptitle(f"[two-branch] image vs object · {args.dataset_name}/dino-{args.dino} · "
                 f"c={c:g}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(out_dir, "twobranch_rep.png")
    fig.savefig(p, dpi=140); plt.close(fig)

    metrics = dict(
        c=c, n_points=int(len(rep_i)), dim=int(rep_i.shape[1]),
        depth_image_mean=float(d_i.mean()), depth_object_mean=float(d_o.mean()),
        depth_delta_obj_minus_img=float((d_o - d_i).mean()),
        frac_obj_deeper=float(np.mean(d_o > d_i)),
        rel_norm_image=float(reln_i.mean()), rel_norm_object=float(reln_o.mean()),
        matched_dist_mean=float(d_pair.mean()), shuffled_dist_mean=float(d_shuf.mean()),
        frac_matched_lt_shuffled=float(np.mean(d_pair < d_shuf)),
        entail_parent=args.obj_entail_parent, entail_cone_satisfied=sat,
        per_class_offset_mean=float(off.mean()) if off.size else None,
        cls_agreement_img_obj=agree,
    )
    with open(os.path.join(out_dir, "twobranch_rep_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"[two-branch] wrote {p}")
    return metrics


# --------------------------------------------------------------------------- #
# (c) image <-> object correspondence
# --------------------------------------------------------------------------- #
def plot_correspondence(data, c, num_labeled, out_dir, args):
    rep_i, rep_o = data["rep_img"], data["rep_obj"]
    labels = data["labels"]
    rng = np.random.default_rng(args.seed)

    n = len(rep_i)
    n_show = min(args.n_pairs, n)
    sel = rng.choice(n, size=n_show, replace=False)

    xy_i, xy_o = shared_disk_embedding(rep_i, rep_o, c, seed=args.seed)
    d_i = V.to_np(pmath.dist0(V.as_tensor(rep_i), c=c)).reshape(-1)
    d_o = V.to_np(pmath.dist0(V.as_tensor(rep_o), c=c)).reshape(-1)

    K = int(np.unique(labels).size)
    pal = V._class_palette(K)
    cl_to_i = {cl: i for i, cl in enumerate(np.unique(labels))}

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))

    # left: disk with sampled image--object segments
    a = ax[0]; V._draw_disk_axes(a)
    segs = [[(xy_i[k, 0], xy_i[k, 1]), (xy_o[k, 0], xy_o[k, 1])] for k in sel]
    seg_cols = [pal[cl_to_i[labels[k]]] for k in sel]
    a.add_collection(LineCollection(segs, colors=seg_cols, linewidths=0.6,
                                    alpha=0.4, zorder=2))
    a.scatter(xy_i[sel, 0], xy_i[sel, 1], s=22, c="#1f77b4",
              edgecolors="white", linewidths=0.4, label="image", zorder=4)
    a.scatter(xy_o[sel, 0], xy_o[sel, 1], s=22, marker="^", c="#ff7f0e",
              edgecolors="white", linewidths=0.4, label="object (fg)", zorder=4)
    a.legend(fontsize=9, loc="upper right", framealpha=0.7)
    a.set_title(f"image ↔ object correspondence ({n_show} sampled pairs)\n"
                "segment colour = class · radius = TRUE depth", fontsize=11)

    # right: depth-shift scatter (image depth -> object depth)
    a = ax[1]
    lo = min(d_i[sel].min(), d_o[sel].min()); hi = max(d_i[sel].max(), d_o[sel].max())
    a.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1, label="image=object")
    deeper = d_o[sel] > d_i[sel]
    a.scatter(d_i[sel][deeper], d_o[sel][deeper], s=20, c="#ff7f0e", alpha=0.7,
              label=f"object deeper ({deeper.mean()*100:.0f}%)")
    a.scatter(d_i[sel][~deeper], d_o[sel][~deeper], s=20, c="#4c72b0", alpha=0.7,
              label=f"object shallower ({(~deeper).mean()*100:.0f}%)")
    a.set_xlabel("image depth  d(0, image_i)")
    a.set_ylabel("object depth  d(0, object_i)")
    a.set_title("Per-pair depth shift\n(above the line ⇒ crop is deeper)", fontsize=11)
    a.legend(fontsize=9); a.grid(alpha=0.3)

    fig.suptitle(f"[correspondence] {args.dataset_name}/dino-{args.dino} · c={c:g}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, "correspondence_rep.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    log(f"[correspondence] wrote {p}")


# --------------------------------------------------------------------------- #
# argparser + main
# --------------------------------------------------------------------------- #
def build_argparser():
    p = V.build_argparser()
    # this variant *is* org-SimGCD on the image side -> method is fixed.
    for a in p._actions:
        if a.dest == "method":
            a.required = False
            a.default = "simgcd"
    # force the image-branch method to simgcd (this variant *is* org SimGCD)
    g = p.add_argument_group("object branch")
    g.add_argument("--obj_fg_source", default="auto",
                   choices=["auto", "attention", "cls_sim"],
                   help="foreground saliency source (match training; auto picks "
                        "attention for v1, cls_sim for v2).")
    g.add_argument("--obj_fg_keep", type=float, default=0.6)
    g.add_argument("--obj_fg_pad", type=float, default=0.1)
    g.add_argument("--obj_entail_parent", default="image",
                   choices=["image", "object"],
                   help="which branch is the cone apex (match training).")
    g.add_argument("--obj_aperture_scale", type=float, default=1.2)
    g.add_argument("--obj_min_radius", type=float, default=0.1)
    g.add_argument("--n_pairs", type=int, default=120,
                   help="image↔object pairs drawn in the correspondence figure.")
    g.add_argument("--skip_object_overview", action="store_true",
                   help="skip the standalone object-branch overview/hierarchy.")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    args.method = "simgcd"   # the object variant shares the org-SimGCD heads
    V.set_seed(args.seed)
    log(f"[visualize_poincare_obj] starting | dataset={args.dataset_name} "
        f"dino={args.dino} c={args.c}")
    if V._REPO_ROOT:
        log(f"[info] HypCD repo root: {V._REPO_ROOT}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir = os.path.abspath(args.out_dir)
    args.model_path = os.path.abspath(args.model_path)
    if args.proj_head_path:
        args.proj_head_path = os.path.abspath(args.proj_head_path)
    if args.classifier_path:
        args.classifier_path = os.path.abspath(args.classifier_path)
    os.makedirs(args.out_dir, exist_ok=True)
    log(f"[info] device = {device}")
    log(f"[info] outputs -> {args.out_dir}")

    # ---- checkpoints (same siblings as the image branch; _best_acc_all ok) ----
    proj_path = args.proj_head_path or V._derive_sibling(args.model_path, "proj_head")
    cls_path = args.classifier_path or V._derive_sibling(args.model_path, "hyp_cls")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"backbone checkpoint not found: {args.model_path}")
    backbone_sd = V._load_ckpt(args.model_path)
    proj_sd = V._load_ckpt(proj_path) if os.path.isfile(proj_path) else None
    cls_sd = V._load_ckpt(cls_path) if os.path.isfile(cls_path) else None
    log(f"[info] projection head : {proj_path if proj_sd else 'MISSING'}")
    log(f"[info] classifier      : {cls_path if cls_sd else 'MISSING'}")

    modules = V.build_modules(args, proj_sd=proj_sd, cls_sd=cls_sd)
    V._robust_load(modules["backbone"], backbone_sd, "backbone")
    if proj_sd is not None and modules["projector"] is not None:
        V._robust_load(modules["projector"], proj_sd, "projector")
    if cls_sd is not None and modules["classifier"] is not None:
        V._robust_load(modules["classifier"], cls_sd, "classifier")
    c = modules["c"]

    # ---- data (chdir into repo root so relative split paths resolve) ----
    if V._REPO_ROOT and os.path.abspath(os.getcwd()) != os.path.abspath(V._REPO_ROOT):
        log(f"[info] chdir -> {V._REPO_ROOT}")
        os.chdir(V._REPO_ROOT)
    loader, num_labeled, num_classes = V.build_dataloader(args)

    # ---- foreground cropper (shares the backbone) + dual extraction ----
    fg_cropper = build_fg_cropper(modules["backbone"], args)
    want_cls = modules["classifier"] is not None
    data = extract_dual(modules, loader, device, fg_cropper,
                        want_cls=want_cls, max_samples=args.max_samples or 4000)
    log(f"[extract] image rep={data['rep_img'].shape} object rep={data['rep_obj'].shape}")

    labels = data["labels"]; is_old = labels < num_labeled

    # (1)/(a) object branch on its own -> reuse the standard overview/hierarchy
    if not args.skip_object_overview:
        V.analyze_ball("obj_rep", data["rep_obj"], labels, is_old, c, num_labeled,
                       args.out_dir, args, preds=None,
                       title_extra="OBJECT branch · representation space")
        if data["cls_obj"] is not None:
            V.analyze_ball("obj_cls", data["cls_obj"], labels, is_old, c, num_labeled,
                           args.out_dir, args, preds=data["preds_obj"],
                           title_extra="OBJECT branch · classification space")

    # (2)/(b) two-branch comparison
    plot_two_branch(data, c, num_labeled, args.out_dir, args)

    # (3)/(c) image <-> object correspondence
    plot_correspondence(data, c, num_labeled, args.out_dir, args)

    # bundle the raw paired features for any further analysis
    np.savez_compressed(
        os.path.join(args.out_dir, "twobranch_embedding.npz"),
        rep_img=data["rep_img"], rep_obj=data["rep_obj"], labels=labels,
        is_old=is_old,
        cls_img=data["cls_img"] if data["cls_img"] is not None else np.zeros((0,)),
        cls_obj=data["cls_obj"] if data["cls_obj"] is not None else np.zeros((0,)),
        curvature_c=np.array([c]))
    log("[done] object-branch visualization complete.")


if __name__ == "__main__":
    main()