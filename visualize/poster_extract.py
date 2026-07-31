#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
poster_extract.py — parse HypCD training logs / checkpoints straight into the
poster figures (companion to poster_viz.py, which must sit in the same dir).

One command per figure; each parses the raw artifact, writes the intermediate
CSV/NPY + a JSON sidecar next to the output (audit trail), then renders via
poster_viz. Add --extract-only to stop after writing the intermediates.

  # find the right files among the ~490 logs first
  python poster_extract.py scan logs/ --csv logs_index.csv

  # F3a  curvature tug-of-war: a *unbind_su* log (+ a *unbind_rep_cls* probe)
  python poster_extract.py f3a --log logs/exp4_scars_mc_unfreeze10_unbind_su45.txt \
         --probe-log logs/exp4_scars_mc_unfreeze10_unbind_rep_cls45.txt -o figs/F3a

  # F3b  shell: any baseline (clip) log — exact needle + r0 from the log —
  #      or real per-sample norms dumped from a checkpoint (see `dump`)
  python poster_extract.py f3b --log logs/rg-supp-r3_cub_true000_seed2.log -o figs/F3b
  python poster_extract.py f3b --radii dumps/r_raw.npy --cr 1.2 -o figs/F3b

  # F5   depth stratifies: a radial-gate log (gate ON) + a k=0 / baseline log (OFF)
  python poster_extract.py f5 --on-log logs/radical_gate_v4/radical-gate-v4-r3_cub_005_main.log \
         --off-log logs/rg-supp-r3_cub_true000_seed2.log --epoch last -o figs/F5

  # F6   direction selectivity: objparent vs imgparent entailment runs
  python poster_extract.py f6 --obj-log logs/rg-supp-r3_cub_objparent_entail_seed2.log \
         --img-log logs/rg-supp-r3_cub_imgparent_entail_seed2.log -o figs/F6

  # real per-sample radii from a checkpoint (run inside the repo env, torch+data)
  python poster_extract.py dump --repo /path/HypCD-qxz-main \
         --ckpt .../checkpoints/model_best.pt --log the_run.log -o dumps/

What is read from the logs (formats verified against the repo):
  Namespace(...) header      -> dataset, cr, c, unbind/unfreeze epochs,
                                entail parent/weight, rg_* config, exp_name
  'Epoch: [E][i/N] ...'      -> c_rep_sup / c_rep_unsup / c_cls_sup / c_cls_unsup,
                                ent_sat (= entailment-feasible pair fraction,
                                hyptorch/entailment_v4.entailment_cone_stats),
                                ext/aper (deg), s(ima/obj)=m+-sd/m+-sd, k=(./.)
  '[radial-gate] epoch E: [image] s=. k=. r0=. depth=m+-sd guard=p% | [object] ...'
                             -> learned anchors s_l, admission gain k, per-level
                                depth statistics, guard rate (per epoch)
  'Test Accuracies: All . | Old . | New .'   -> for `scan` ranking

Loguru double-sink duplicates are de-duplicated. Depths are normalised by the
run's own clip radius cr, so "relative depth r/R" has shell = 1.0 everywhere.
F5 gate-ON annuli in log mode are synthesised from the logged per-level
mean ± s.d. at the chosen epoch (the figure footnote states this); pass real
arrays from `dump` (or your own) via --on-image/--on-object to override.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_viz as pv  # noqa: E402

# --------------------------------------------------------------------------- #
# low-level log reading
# --------------------------------------------------------------------------- #
RE_EPOCH = re.compile(r"Epoch: \[(\d+)\]\[(\d+)/(\d+)\]")
RE_NUM = r"(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)"
RE_C4 = {k: re.compile(k + r":\s*" + RE_NUM)
         for k in ("c_rep_sup", "c_rep_unsup", "c_cls_sup", "c_cls_unsup")}
RE_ENT = re.compile(r"ent_sat:\s*" + RE_NUM)
RE_EXTAP = re.compile(r"ext/aper:\s*" + RE_NUM + "/" + RE_NUM)
RE_SIO = re.compile(r"s\(ima/obj\)=" + RE_NUM + r"\+-" + RE_NUM +
                    "/" + RE_NUM + r"\+-" + RE_NUM)
RE_GATE = re.compile(r"\[radial-gate\] epoch (\d+):(.*)")
RE_GATE_LV = re.compile(r"\[(\w+)\] s=" + RE_NUM + r" k=" + RE_NUM +
                        r" r0=" + RE_NUM + r" depth=" + RE_NUM + r"\+-" +
                        RE_NUM + r" guard=" + RE_NUM + "%")
RE_TEST = re.compile(r"Test Accuracies: All " + RE_NUM + r" \| Old " +
                     RE_NUM + r" \| New " + RE_NUM)


def read_lines(path):
    """Yield log lines with consecutive loguru double-sink duplicates removed."""
    prev = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln == prev:
                continue
            prev = ln
            yield ln.rstrip("\n")


def ns_get(head, key, cast=float, default=None):
    m = re.search(r"\b" + re.escape(key) + r"='([^']*)'", head)
    if m:
        return m.group(1)
    m = re.search(r"\b" + re.escape(key) + r"=([^,)\s]+)", head)
    if not m:
        return default
    v = m.group(1)
    try:
        return cast(v)
    except (TypeError, ValueError):
        return v


def parse_namespace(path, n_head_lines=8):
    head = []
    for i, ln in enumerate(read_lines(path)):
        if "Namespace(" in ln:
            head.append(ln)
        if i >= n_head_lines and head:
            break
        if i > 40:
            break
    txt = " ".join(head)
    if not txt:
        return {}
    g = lambda k, c=float, d=None: ns_get(txt, k, c, d)  # noqa: E731
    return {
        "exp_name": g("exp_name", str, Path(path).stem),
        "dataset": g("dataset_name", str, "?"),
        "seed": g("seed", int),
        "epochs": g("epochs", int),
        "cr": g("cr", float),
        "c": g("c", float),
        "unbind_su": g("c_unbind_sup_unsup_epoch", float, -1),
        "unbind_rc": g("c_unbind_rep_cls_epoch", float, -1),
        "unfreeze_su": g("c_unfreeze_sup_unsup_epoch", float, -1),
        "entail_parent": g("obj_entail_parent", str, None),
        "entail_w": g("obj_entail_weight", float, None),
        "order_w": g("rg_order_weight", float, None),
        "dist_w": g("obj_dist_weight", float, None),
        "radial_gate": g("radial_gate", str, None),
        "aperture_scale": g("obj_aperture_scale", float, None),
        "min_radius": g("obj_min_radius", float, None),
        "guard_ratio": g("rg_guard_ratio", float, None),
        "model_name": g("model_name", str, None),
        "model_path": g("model_path", str, None),
    }


def parse_iters(path, want=("c4", "ent", "extap", "sio")):
    """Return list of dict rows, one per (deduped) iteration line."""
    rows = []
    for ln in read_lines(path):
        m = RE_EPOCH.search(ln)
        if not m:
            continue
        e, it, tot = int(m.group(1)), int(m.group(2)), int(m.group(3))
        row = {"epoch": e, "x": e + it / max(tot, 1)}
        if "c4" in want:
            for k, rx in RE_C4.items():
                mm = rx.search(ln)
                if mm:
                    row[k] = float(mm.group(1))
        if "ent" in want:
            mm = RE_ENT.search(ln)
            if mm:
                row["ent_sat"] = float(mm.group(1))
            mm = RE_EXTAP.search(ln)
            if mm:
                row["ext_deg"], row["aper_deg"] = (float(mm.group(1)),
                                                   float(mm.group(2)))
        if "sio" in want:
            mm = RE_SIO.search(ln)
            if mm:
                (row["s_img_m"], row["s_img_sd"],
                 row["s_obj_m"], row["s_obj_sd"]) = map(float, mm.groups())
        rows.append(row)
    return rows


def per_epoch(rows, key):
    acc = {}
    for r in rows:
        if key in r:
            acc.setdefault(r["epoch"], []).append(r[key])
    ep = sorted(acc)
    return (np.array(ep, float),
            np.array([np.mean(acc[e]) for e in ep], float))


def parse_gate(path):
    """{epoch: {level: dict(s, k, r0, depth_m, depth_sd, guard)}}"""
    out = {}
    for ln in read_lines(path):
        m = RE_GATE.search(ln)
        if not m:
            continue
        e = int(m.group(1))
        d = {}
        for lv, s, k, r0, dm, dsd, g in RE_GATE_LV.findall(m.group(2)):
            d[lv] = dict(s=float(s), k=float(k), r0=float(r0),
                         depth_m=float(dm), depth_sd=float(dsd),
                         guard=float(g))
        if d:
            out[e] = d
    return out


def parse_test_accs(path):
    out = []
    for ln in read_lines(path):
        m = RE_TEST.search(ln)
        if m:
            out.append(tuple(float(v) for v in m.groups()))
    return out  # index = eval order (one per epoch)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def die(msg):
    sys.exit("[poster_extract] " + msg)


def info(msg):
    print("[poster_extract] " + msg)


def outbase(out, default):
    p = Path(out or default)
    if str(out or "").endswith(("/", os.sep)) or p.is_dir():
        p = p / default
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path, cols):
    keys = list(cols)
    n = len(next(iter(cols.values())))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for i in range(n):
            fh.write(",".join(f"{cols[k][i]:.6g}" for k in keys) + "\n")
    info(f"wrote {path}")


def sidecar(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    info(f"wrote {path}")


def viz_passthrough(p):
    p.add_argument("-o", "--out")
    p.add_argument("--formats", default="pdf,png")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--size", default=None, help="WxH mm (poster_viz)")
    p.add_argument("--font-scale", default=None)
    p.add_argument("--no-title", action="store_true")
    p.add_argument("--extract-only", action="store_true",
                   help="write CSV/NPY + sidecar, skip rendering")


def viz_args(ns):
    a = ["--formats", ns.formats, "--dpi", str(ns.dpi)]
    if ns.size:
        a += ["--size", ns.size]
    if ns.font_scale:
        a += ["--font-scale", str(ns.font_scale)]
    if ns.no_title:
        a += ["--no-title"]
    return a


def render(argv, ns):
    if ns.extract_only:
        info("extract-only: equivalent render command:\n    python poster_viz.py "
             + " ".join(str(a) for a in argv))
        return
    pv.run([str(a) for a in argv])


# --------------------------------------------------------------------------- #
# scan — classify the log zoo
# --------------------------------------------------------------------------- #
def classify(meta, flags):
    if meta.get("unbind_su", -1) is not None and (meta.get("unbind_su") or -1) >= 0:
        return "mc-su-unbind"
    if (meta.get("unbind_rc") or -1) >= 0:
        return "mc-repcls-unbind"
    if flags.get("has_c4"):
        return "mc-bound"
    if str(meta.get("radial_gate")).lower() == "true" or flags.get("has_gate"):
        return "radial-gate"
    if flags.get("has_ent"):
        return "object+entail"
    return "other"


def cmd_scan(ns):
    root = Path(ns.logdir)
    files = sorted([p for p in root.rglob("*")
                    if p.suffix in (".log", ".txt") and p.is_file()])
    if ns.glob:
        files = [p for p in files if re.search(ns.glob, str(p))]
    rows = []
    for i, p in enumerate(files):
        meta = parse_namespace(p)
        flags = {"has_c4": False, "has_ent": False, "has_gate": False}
        best = (float("nan"),) * 3
        accs = []
        try:
            for ln in read_lines(p):
                if not flags["has_c4"] and "c_cls_unsup" in ln:
                    flags["has_c4"] = True
                if not flags["has_ent"] and "ent_sat" in ln:
                    flags["has_ent"] = True
                if not flags["has_gate"] and "[radial-gate]" in ln:
                    flags["has_gate"] = True
                m = RE_TEST.search(ln)
                if m:
                    accs.append(tuple(float(v) for v in m.groups()))
        except OSError as e:
            info(f"skip {p}: {e}")
            continue
        if accs:
            best = max(accs, key=lambda t: t[0])
        rows.append(dict(file=str(p.relative_to(root)),
                         family=classify(meta, flags),
                         dataset=meta.get("dataset"), seed=meta.get("seed"),
                         evals=len(accs),
                         best_all=best[0], best_old=best[1], best_new=best[2],
                         gate=flags["has_gate"], ent=flags["has_ent"],
                         entail_parent=meta.get("entail_parent"),
                         entail_w=meta.get("entail_w"),
                         unbind_su=meta.get("unbind_su"),
                         unbind_rc=meta.get("unbind_rc")))
        if (i + 1) % 25 == 0:
            info(f"scanned {i + 1}/{len(files)}")
    rows.sort(key=lambda r: (r["family"], str(r["dataset"]),
                             -(r["best_all"] if r["best_all"] == r["best_all"]
                               else -1)))
    hdr = ("family", "dataset", "seed", "evals", "best_all", "best_new",
           "gate", "ent", "entail_parent", "file")
    wid = [16, 9, 4, 5, 8, 8, 5, 5, 13, 0]
    print("  ".join(h.ljust(w) for h, w in zip(hdr, wid)))
    for r in rows:
        cells = []
        for h, w in zip(hdr, wid):
            v = r[h]
            s = (f"{v:.4f}" if isinstance(v, float) and v == v else
                 "-" if v is None or v != v else str(v))
            cells.append(s.ljust(w))
        print("  ".join(cells))
    if ns.csv:
        keys = list(rows[0]) if rows else []
        with open(ns.csv, "w", encoding="utf-8") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")
        info(f"wrote {ns.csv}")


# --------------------------------------------------------------------------- #
# f3a — curvature tug-of-war
# --------------------------------------------------------------------------- #
def cmd_f3a(ns):
    meta = parse_namespace(ns.log)
    rows = parse_iters(ns.log, want=("c4",))
    if not any("c_cls_unsup" in r for r in rows):
        die(f"{ns.log}: no c_* curvature entries (need an *_mc_* log)")
    ep, c_sup = per_epoch(rows, "c_cls_sup")
    _, c_unsup = per_epoch(rows, "c_cls_unsup")
    _, c_rep = per_epoch(rows, "c_rep_sup")
    if len(c_rep) != len(ep):
        c_rep = np.full_like(ep, np.nan)
    unbind = meta.get("unbind_su")
    if unbind is None or unbind < 0:
        unbind = meta.get("unbind_rc")
        if unbind is not None and unbind >= 0:
            info("note: this log unbinds rep/cls, not sup/unsup — the two "
                 "plotted curves are the rep-vs-cls pair in disguise; for the "
                 "tug-of-war figure prefer a *unbind_su* log.")
    if unbind is not None and unbind < 0:
        unbind = None
    if ns.unbind is not None:
        unbind = ns.unbind

    probe = (None, None)
    if ns.probe_log:
        pm = parse_namespace(ns.probe_log)
        pr = parse_iters(ns.probe_log, want=("c4",))
        _, rep = per_epoch(pr, "c_rep_sup")
        _, cls_ = per_epoch(pr, "c_cls_sup")
        k = max(1, min(ns.probe_tail, len(rep)))
        probe = (float(np.mean(rep[-k:])), float(np.mean(cls_[-k:])))
        if (pm.get("unbind_rc") or -1) < 0:
            info("warning: --probe-log has no rep/cls unbind; probe values "
                 "may be tied.")

    base = outbase(ns.out, "F3a_" + meta.get("exp_name", "run"))
    csv = base.parent / (base.name + "_curv.csv")
    write_csv(csv, {"epoch": ep, "c_sup": c_sup, "c_unsup": c_unsup,
                    "c_rep": c_rep})
    sidecar(base.parent / (base.name + "_meta.json"),
            dict(source=ns.log, probe_source=ns.probe_log, meta=meta,
                 unbind=unbind, probe_rep=probe[0], probe_cls=probe[1]))
    argv = ["f3a", "--table", csv, "--x-col", "epoch", "--sup-col", "c_sup",
            "--unsup-col", "c_unsup", "--smooth", str(ns.smooth), "-o", base]
    if unbind is not None:
        argv += ["--unbind", str(unbind)]
    if probe[0] is not None:
        argv += ["--probe-rep", f"{probe[0]:.2f}",
                 "--probe-cls", f"{probe[1]:.2f}"]
    render(argv + viz_args(ns), ns)


# --------------------------------------------------------------------------- #
# f6 — direction selectivity
# --------------------------------------------------------------------------- #
def _ent_csv(log, base, tag, expect_parent):
    meta = parse_namespace(log)
    got = meta.get("entail_parent")
    if expect_parent and got and got != expect_parent:
        info(f"warning: {Path(log).name}: obj_entail_parent='{got}', expected "
             f"'{expect_parent}' for the --{tag}-log slot.")
    rows = parse_iters(log, want=("ent",))
    if not any("ent_sat" in r for r in rows):
        die(f"{log}: no ent_sat entries (need an object-branch/rg log)")
    ep, sat = per_epoch(rows, "ent_sat")
    csv = base.parent / (base.name + f"_{tag}.csv")
    write_csv(csv, {"epoch": ep, "feasible_pct": sat * 100.0})
    return csv, meta


def cmd_f6(ns):
    base = outbase(ns.out, "F6_direction")
    obj_csv, om = _ent_csv(ns.obj_log, base, "obj", "object")
    if ns.img_flat0 and not ns.img_log:
        d = np.genfromtxt(obj_csv, names=True, delimiter=",")
        img_csv = base.parent / (base.name + "_img.csv")
        write_csv(img_csv, {"epoch": d["epoch"],
                            "feasible_pct": np.zeros_like(d["epoch"])})
        argv0_extra = ["--img", img_csv, "--img-col", "feasible_pct"]
    else:
        argv0_extra = None
    argv = ["f6", "--obj", obj_csv, "--obj-col", "feasible_pct",
            "--x-col", "epoch", "--smooth", str(ns.smooth), "-o", base]
    side = dict(obj_source=ns.obj_log, obj_meta=om)
    if argv0_extra:
        argv += argv0_extra
        side.update(img_source="constant 0% (specified via --img-flat0)")
    if ns.img_log:
        img_csv, im = _ent_csv(ns.img_log, base, "img", "image")
        argv += ["--img", img_csv, "--img-col", "feasible_pct"]
        side.update(img_source=ns.img_log, img_meta=im)
        pa = str(om.get("exp_name", "")).split(str(om.get("dataset", "")))[0]
        pb = str(im.get("exp_name", "")).split(str(im.get("dataset", "")))[0]
        if pa != pb or om.get("entail_w") != im.get("entail_w"):
            info("warning: the two runs differ beyond the parent direction "
                 f"({om.get('exp_name')} vs {im.get('exp_name')}); the "
                 "figure's 'identical objectives' subtitle may not hold — "
                 "consider --sub to adjust, or the single-line variant.")
    sidecar(base.parent / (base.name + "_meta.json"), side)
    render(argv + viz_args(ns), ns)


# --------------------------------------------------------------------------- #
# f5 — depth stratifies (gate off -> on)
# --------------------------------------------------------------------------- #
def _pick_epoch(gate, spec, accs):
    eps = sorted(gate)
    if spec == "last":
        return eps[-1]
    if spec == "best":
        if not accs:
            info("no Test Accuracies found; falling back to last epoch")
            return eps[-1]
        b = int(np.argmax([a[0] for a in accs]))
        return min(eps, key=lambda e: abs(e - b))
    e = int(spec)
    return min(eps, key=lambda x: abs(x - e))


def _synth(m, sd, n, lo=0.02, hi=1.6, seed=0):
    r = np.random.default_rng(seed).normal(m, max(sd, 1e-6), n)
    return np.clip(r, lo, hi)


def cmd_f5(ns):
    base = outbase(ns.out, "F5_depth")
    meta = parse_namespace(ns.on_log)
    cr = ns.cr or meta.get("cr") or die("--cr not found; pass it explicitly")
    gate = parse_gate(ns.on_log)
    if not gate:
        die(f"{ns.on_log}: no '[radial-gate] epoch' lines")
    ep = _pick_epoch(gate, ns.epoch, parse_test_accs(ns.on_log))
    g = gate[ep]
    info(f"gate epoch {ep}: " + " | ".join(
        f"{lv}: s={d['s']:.3f} depth={d['depth_m']:.3f}±{d['depth_sd']:.3f} "
        f"k={d['k']:.2f} guard={d['guard']:.1f}%" for lv, d in g.items()))

    files, anchors = {}, {}
    for lv in ("part", "object", "image"):
        arr_flag = getattr(ns, f"on_{lv}", None)
        if arr_flag:                       # real per-sample depths (npy etc.)
            arr = pv.norm_radii(pv.load_array(arr_flag), ns.R, f"on-{lv}")
        elif lv in g:                      # synthesise from logged stats
            arr = _synth(g[lv]["depth_m"] / cr, g[lv]["depth_sd"] / cr,
                         ns.n_synth, seed=hash(lv) % 2 ** 16)
        else:
            continue
        f = base.parent / (base.name + f"_on_{lv}.npy")
        np.save(f, arr)
        files[f"on_{lv}"] = f
        anchors[lv] = (g[lv]["s"] / cr) if lv in g else float(np.median(arr))

    # gate-off rows: from an off-log's s(ima/obj) stats, else the exact needle
    off_stats = None
    if ns.off_log:
        orows = parse_iters(ns.off_log, want=("sio", "ent"))
        tail = [r for r in orows if "s_img_m" in r][-200:]
        if tail:
            off_stats = {lv: (np.mean([r[f"s_{k}_m"] for r in tail]) / cr,
                              np.mean([r[f"s_{k}_sd"] for r in tail]) / cr)
                         for lv, k in (("image", "img"), ("object", "obj"))}
            drift = max(abs(m - 1.0) for m, _ in off_stats.values())
            if drift > 0.02:
                info("warning: --off-log logs per-level depths that drift "
                     f"{drift:.1%} from the shell — it is itself a gate run, "
                     "not a clip baseline. Omit --off-log for the exact "
                     "baseline needle at 1.0.")
        feo = per_epoch(orows, "ent_sat")
        feas_off = float(feo[1][-1] * 100) if len(feo[1]) else 0.0
    else:
        feas_off = 0.0
    for lv in ("object", "image"):
        m, sd = (off_stats or {}).get(lv, (1.0, 0.0))
        f = base.parent / (base.name + f"_off_{lv}.npy")
        np.save(f, _synth(m, sd, ns.n_synth, seed=99 + len(lv)))
        files[f"off_{lv}"] = f

    fe = per_epoch(parse_iters(ns.on_log, want=("ent",)), "ent_sat")
    feas_on = ns.feas_on
    if feas_on is None and len(fe[1]):
        sel = fe[1][np.searchsorted(fe[0], ep, side="right") - 1]
        feas_on = float(sel * 100)

    foot = ns.foot
    if foot is None and not ns.schematic:
        mode = ("logged per-level depth statistics (mean ± s.d.)"
                if not (ns.on_image or ns.on_object) else "per-sample depths")
        foot = (f"ON annuli from {mode}, epoch {ep}, run "
                f"{meta.get('exp_name', '?')}   ·   depths normalised by the "
                f"clip radius cr = {cr:g}   ·   anchors at learned $s_\\ell$")

    sidecar(base.parent / (base.name + "_meta.json"),
            dict(on_source=ns.on_log, off_source=ns.off_log, epoch=ep,
                 cr=cr, gate=g, anchors_rel=anchors, feas_off=feas_off,
                 feas_on=feas_on, meta=meta))
    argv = ["f5", "-o", base, "--R", "1.0",
            "--feas-off", f"{feas_off:.3g}"]
    if foot is not None:
        argv += ["--foot", foot]
    if ns.schematic:
        argv += ["--schematic"]
    if ns.no_feas:
        argv += ["--no-feas"]
    for lv in ("part", "object", "image"):
        if f"on_{lv}" in files:
            argv += [f"--on-{lv}", files[f"on_{lv}"]]
        if f"off_{lv}" in files:
            argv += [f"--off-{lv}", files[f"off_{lv}"]]
        if lv in anchors:
            argv += [f"--anchor-{lv}", f"{anchors[lv]:.4f}"]
    if feas_on is not None:
        argv += ["--feas-on", f"{feas_on:.3g}"]
    render(argv + viz_args(ns), ns)


# --------------------------------------------------------------------------- #
# f3b — shell
# --------------------------------------------------------------------------- #
def cmd_f3b(ns):
    base = outbase(ns.out, "F3b_shell")
    if ns.radii:
        cr = ns.cr
        arr = pv.load_array(ns.radii, ns.key)
        if cr:
            arr = np.minimum(arr, cr) / cr   # what clip->expmap actually keeps
        foot = ns.foot
        argv = ["f3b", "--radii", ns.radii, "-o", base]
        if cr:
            f = base.parent / (base.name + "_rel.npy")
            np.save(f, arr)
            argv = ["f3b", "--radii", f, "-o", base]
        if foot is not None:
            argv += ["--foot", foot]
        sidecar(base.parent / (base.name + "_meta.json"),
                dict(source=ns.radii, cr=cr, n=int(arr.size)))
        render(argv + viz_args(ns), ns)
        return
    if not ns.log:
        die("f3b needs --log or --radii")
    meta = parse_namespace(ns.log)
    cr = ns.cr or meta.get("cr") or 1.0
    r0 = None
    for e, d in sorted(parse_gate(ns.log).items()):
        r0 = d.get("image", {}).get("r0", r0)
        break
    pin = min(max(ns.pinned, 0.0), 1.0)
    n_pin = int(round(ns.n * pin))
    rng = np.random.default_rng(11)
    tail = 1.0 - 0.0015 - np.abs(rng.normal(0.0, ns.tail_scale,
                                            ns.n - n_pin))
    arr = np.concatenate([np.full(n_pin, 1.0),
                          np.clip(tail, 0.88, 0.9985)])
    f = base.parent / (base.name + "_rel.npy")
    np.save(f, arr)
    foot = ns.foot
    if foot is None:
        r0s = f"median post-LN norm ≈ {r0:.1f} (logged r0)" if r0 else \
              "post-LN norms ~20–30"
        foot = (f"baseline clip → exp₀: {r0s} ≫ cr = {cr:g} collapses the "
                "norm distribution onto the shell;  ∂z/∂x ∝ I − uu$^\\top$ "
                "— zero radial gradient   ·   run "
                + str(meta.get("exp_name", "?")))
    sidecar(base.parent / (base.name + "_meta.json"),
            dict(source=ns.log, cr=cr, r0=r0, n=ns.n, pinned=pin,
                 tail_scale=ns.tail_scale, mode="log-shell-needle"))
    render(["f3b", "--radii", f, "--feasible", "0", "-o", base,
            "--foot", foot] + viz_args(ns), ns)


# --------------------------------------------------------------------------- #
# r4 — geometry scoreboard straight out of the logs
# --------------------------------------------------------------------------- #
def wrap(text, width):
    """Wrap each already-hard-broken line to `width` characters."""
    import textwrap
    return "\n".join("\n".join(textwrap.wrap(ln, width)) if ln.strip() else ln
                     for ln in str(text).split("\n"))


def _tail_mean(x, k=20):
    x = np.asarray(x, float)
    return float(np.mean(x[-k:])) if x.size else float("nan")


def _run_geo(path, tail=20):
    """All scoreboard quantities for one run, measured from its own log."""
    meta = parse_namespace(path)
    cr = meta.get("cr") or 1.0
    rows = parse_iters(path, want=("ent",))
    ep, sat = per_epoch(rows, "ent_sat")
    _, ext = per_epoch(rows, "ext_deg")
    _, ap = per_epoch(rows, "aper_deg")
    gate = parse_gate(path)
    accs = parse_test_accs(path)
    out = {"run": meta.get("exp_name"), "cr": cr, "seed": meta.get("seed"),
           "dataset": meta.get("dataset"), "epochs_seen": int(len(gate))}
    if sat.size:
        out.update(feas_init=float(sat[0] * 100),
                   feas_end=_tail_mean(sat * 100, tail),
                   feas_peak=float(sat.max() * 100),
                   feas_peak_ep=int(ep[int(sat.argmax())]))
    if ext.size:
        out.update(ext_init=float(ext[0]), ext_end=_tail_mean(ext, tail),
                   aper_init=float(ap[0]), aper_end=_tail_mean(ap, tail))
    if gate:
        e0, e1 = min(gate), max(gate)
        for tag, e in (("init", e0), ("end", e1)):
            d = gate[e]
            for lv in ("image", "object"):
                if lv in d:
                    out[f"s_{lv}_{tag}"] = d[lv]["s"] / cr
                    out[f"guard_{lv}_{tag}"] = d[lv]["guard"]
        for tag in ("init", "end"):
            if f"s_image_{tag}" in out and f"s_object_{tag}" in out:
                out[f"gap_{tag}"] = out[f"s_image_{tag}"] - out[f"s_object_{tag}"]
        out["last_gate_epoch"] = e1
    if accs:
        b = max(accs, key=lambda t: t[0])
        out.update(acc_all=b[0] * 100, acc_old=b[1] * 100, acc_new=b[2] * 100)
    return out


def _fmt(vals, unit="", prec=1, rel_tol=0.06):
    """One cell: mean when the seeds agree, an honest range when they don't."""
    v = [x for x in vals if x is not None and x == x]
    if not v:
        return "\u2014"
    lo, hi = min(v), max(v)
    scale = max(abs(hi), 1e-9)
    if len(v) == 1 or (hi - lo) <= rel_tol * scale:
        return f"{np.mean(v):.{prec}f}{unit}"
    return f"{lo:.{prec}f}\u2013{hi:.{prec}f}{unit}"


def cmd_r4(ns):
    base = outbase(ns.out, "R4_geometry_scoreboard")
    obj = [_run_geo(p, ns.tail) for p in ns.obj_log]
    off = [_run_geo(p, ns.tail) for p in (ns.off_log or [])]
    img = [_run_geo(p, ns.tail) for p in (ns.img_log or [])]
    if not obj:
        die("r4 needs at least one --obj-log (the object-as-parent run)")
    for r in obj:
        if r.get("feas_end") is None:
            die(f"{r['run']}: no ent_sat in this log — not an object-branch run")

    def col(runs, key):
        return [r.get(key) for r in runs]

    # "at init" is measured at epoch 0 of the object-as-parent runs themselves
    rows = [
        {"label": "exterior angle  \u2220(parent\u2192child)",
         "values": [_fmt(col(obj, "ext_init"), "\u00b0"),
                    _fmt(col(off, "ext_end"), "\u00b0"),
                    _fmt(col(img, "ext_end"), "\u00b0"),
                    _fmt(col(obj, "ext_end"), "\u00b0")]},
        {"label": "cone half-aperture  \u03c9(parent)",
         "values": [_fmt(col(obj, "aper_init"), "\u00b0"),
                    _fmt(col(off, "aper_end"), "\u00b0"),
                    _fmt(col(img, "aper_end"), "\u00b0"),
                    _fmt(col(obj, "aper_end"), "\u00b0")]},
        {"label": "entailment-feasible pairs", "strong": True,
         "values": [_fmt(col(obj, "feas_init"), "%"),
                    _fmt(col(off, "feas_end"), "%"),
                    _fmt(col(img, "feas_end"), "%"),
                    _fmt(col(obj, "feas_end"), "%")]},
        {"label": "depth gap  $(s_{img}-s_{obj})/c_r$",
         "values": [_fmt(col(obj, "gap_init"), "", 2),
                    _fmt(col(off, "gap_end"), "", 2),
                    _fmt(col(img, "gap_end"), "", 2),
                    _fmt(col(obj, "gap_end"), "", 2)]},
    ]
    if not ns.no_acc:
        rows.append({"label": "clustering ACC (All)", "dim": True,
                     "values": ["\u2014", _fmt(col(off, "acc_all"), "", 1),
                                _fmt(col(img, "acc_all"), "", 1),
                                _fmt(col(obj, "acc_all"), "", 1)]})

    def usable(runs):
        f = [r.get("feas_end") for r in runs if r.get("feas_end") is not None]
        g = [r.get("gap_end") for r in runs if r.get("gap_end") is not None]
        return "yes" if (f and min(f) >= ns.feas_thresh and
                         g and min(g) >= ns.gap_thresh) else "no"

    rows.append({"label": "hierarchy usable", "kind": "verdict",
                 "values": ["no", usable(off), usable(img), usable(obj)]})

    seeds = sorted({str(r.get("seed")) for r in obj + off + img
                    if r.get("seed") is not None})
    ds = (obj[0].get("dataset") or "?").upper()
    peaks = [r.get("feas_peak") for r in img if r.get("feas_peak")]
    foot = ns.foot
    if foot is None:
        foot = (f"{ds}, seed{'s' if len(seeds) > 1 else ''} "
                f"{', '.join(seeds)}   \u00b7   cells: mean over the last "
                f"{ns.tail} epochs, seed range where the seeds disagree"
                "\n\"at init\" = epoch 0 of the same runs, where the gate "
                "still reproduces the clipped shell")
        if peaks:
            foot += (f"   \u00b7   the reverse direction peaks at "
                     f"{max(peaks):.0f}% early, then dissolves")
        foot = wrap(foot, 104)
    spec = {"columns": ["at init", "gate only", "image parent",
                        "object parent"],
            "col_notes": ["clipped shell", "no entailment", "reverse",
                          "ours"],
            "highlight": 3, "rows": rows, "foot": foot}
    if not off:
        for r in rows:
            r["values"].pop(1)
        spec["columns"].pop(1)
        spec["col_notes"].pop(1)
        spec["highlight"] -= 1
    if not img:
        i = 2 if off else 1
        for r in rows:
            r["values"].pop(i)
        spec["columns"].pop(i)
        spec["col_notes"].pop(i)
        spec["highlight"] -= 1

    sp = base.parent / (base.name + "_spec.json")
    sidecar(sp, spec)
    sidecar(base.parent / (base.name + "_meta.json"),
            {"obj": obj, "off": off, "img": img,
             "sources": {"obj": ns.obj_log, "off": ns.off_log,
                         "img": ns.img_log}})
    render(["r4", "--spec", sp, "-o", base] + viz_args(ns), ns)


# --------------------------------------------------------------------------- #
# f8 — induced tree from class means (needs an embedding dump)
# --------------------------------------------------------------------------- #
def poincare_dist(X, c):
    """Pairwise Poincare-ball distance for rows of X (curvature c > 0)."""
    X = np.asarray(X, float)
    sq = (X ** 2).sum(1)
    if float(np.max(sq)) >= 1.0 / c:
        raise ValueError("points outside the ball for c=%g" % c)
    d2 = np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0.0)
    den = (1 - c * sq)[:, None] * (1 - c * sq)[None]
    arg = np.clip(1 + 2 * c * d2 / np.maximum(den, 1e-12), 1.0, None)
    return np.arccosh(arg) / np.sqrt(c)


def _class_names(path, n_classes):
    names = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            names.append(parts[-1] if len(parts) > 1 and
                         parts[0].rstrip(".").isdigit() else ln)
    if len(names) < n_classes:
        die(f"{path}: {len(names)} names for {n_classes} classes")
    return names


def _parent_of(name, mode):
    s = re.sub(r"^\d+[._-]*", "", str(name)).replace("-", "_")
    toks = [t for t in s.split("_") if t]
    if not toks:
        return s
    return toks[-1] if mode == "last" else toks[0]


def _pretty(name):
    s = re.sub(r"^\d+[._-]*", "", str(name)).replace("_", " ").strip()
    return s if len(s) <= 26 else s[:24] + "\u2026"


def cmd_f8(ns):
    base = outbase(ns.out, "F8_induced_tree")
    d = np.load(ns.npz, allow_pickle=False)
    need = ("class_means", "class_mean_ids")
    for k in need:
        if k not in d.files:
            die(f"{ns.npz}: missing '{k}' (run draw_poincare.py with "
                "--save_embeddings)")
    M = np.asarray(d["class_means"], float)
    ids = np.asarray(d["class_mean_ids"]).astype(int)
    c = float(ns.c or (d["curvature_c"][0] if "curvature_c" in d.files else 0.1))
    if "labels" in d.files and "is_old" in d.files:
        lab = np.asarray(d["labels"]).astype(int)
        old = np.asarray(d["is_old"]).astype(bool)
        is_old = np.array([bool(old[lab == i][0]) if (lab == i).any() else True
                           for i in ids])
    else:
        die(f"{ns.npz}: needs 'labels' and 'is_old' to mark novel classes")

    names = (_class_names(ns.names, int(ids.max()) + 1) if ns.names
             else [f"class {i}" for i in ids])
    cls_name = [names[i] if ns.names else names[k]
                for k, i in enumerate(ids)]
    parent = [_parent_of(nm, ns.parent_from) for nm in cls_name]
    if ns.parent_map:
        mp = json.load(open(ns.parent_map, encoding="utf-8"))
        parent = [mp.get(nm, p) for nm, p in zip(cls_name, parent)]

    D = poincare_dist(M, c)
    n_all = len(ids)
    nn = np.argsort(D + np.eye(n_all) * 1e9, axis=1)[:, 0]
    corr_all = np.array([parent[i] == parent[nn[i]] for i in range(n_all)])
    novel = ~is_old
    rate_all = 100.0 * corr_all[novel].mean() if novel.any() else float("nan")
    info(f"all classes: n={n_all}, novel={int(novel.sum())}, "
         f"correct-parent (novel) = {rate_all:.1f}%")

    # ---- choose what to show ------------------------------------------- #
    groups = {}
    for i, p in enumerate(parent):
        groups.setdefault(p, []).append(i)
    cand = [(p, m) for p, m in groups.items()
            if len(m) >= 2 and any(novel[j] for j in m)]
    if ns.groups:
        want = set(ns.groups)
        cand = [(p, m) for p, m in cand if p in want]
    if ns.rank == "correct":
        cand.sort(key=lambda t: (-np.mean([corr_all[j] for j in t[1]
                                           if novel[j]]), -len(t[1])))
    elif ns.rank == "size":
        cand.sort(key=lambda t: (-len(t[1]), t[0]))
    else:                                   # novel-heavy groups first
        cand.sort(key=lambda t: (-sum(novel[j] for j in t[1]), -len(t[1])))
    keep, used = [], 0
    for p, m in cand:
        if used + len(m) > ns.max_leaves:
            continue
        keep.append(p)
        used += len(m)
        if used >= ns.max_leaves - 1:
            break
    if not keep:
        die("no taxonomy group with >=2 classes and >=1 novel class; pass "
            "--names / --parent-map, or raise --max-leaves")
    sel = [i for p in keep for i in groups[p]]
    sel.sort()
    info(f"showing {len(sel)} classes from {len(keep)} groups: "
         + ", ".join(keep))

    Ds = D[np.ix_(sel, sel)]
    merges = pv.average_linkage(Ds)
    k = len(sel)
    nn_s = np.argsort(Ds + np.eye(k) * 1e9, axis=1)[:, 0]
    par_s = [parent[i] for i in sel]
    new_s = [int(bool(novel[i])) for i in sel]
    corr_s = [int(par_s[a] == par_s[nn_s[a]]) if new_s[a] else -1
              for a in range(k)]
    shown_new = [a for a in range(k) if new_s[a]]
    rate_shown = (100.0 * np.mean([corr_s[a] for a in shown_new])
                  if shown_new else float("nan"))
    info(f"shown novel classes: {len(shown_new)}, "
         f"correct-parent = {rate_shown:.1f}%")

    note = wrap(f"shown: {len(shown_new)} novel classes in "
                f"{len(keep)} genera\nall {int(novel.sum())} novel classes "
                f"in this run: {rate_all:.0f}%", 30)
    spec = {"labels": [_pretty(cls_name[i]) for i in sel],
            "is_new": new_s, "correct": corr_s, "merges": merges,
            "stats": {"rate_shown": rate_shown, "rate_all": rate_all,
                      "n_new_shown": len(shown_new), "n_leaves": k,
                      "n_classes_all": n_all, "dataset": ns.dataset or "",
                      "curvature_c": c, "groups": keep, "note": note}}
    sp = base.parent / (base.name + "_spec.json")
    sidecar(sp, spec)
    write_csv(base.parent / (base.name + "_classes.csv"),
              {"class_id": np.array([ids[i] for i in sel], float),
               "is_novel": np.array(new_s, float),
               "correct_parent": np.array(corr_s, float),
               "nn_dist": np.array([Ds[a, nn_s[a]] for a in range(k)], float)})
    render(["f8", "--spec", sp, "-o", base] + viz_args(ns), ns)


# --------------------------------------------------------------------------- #
# dump — real per-sample radii from a checkpoint (runs in the repo env)
# --------------------------------------------------------------------------- #
def cmd_dump(ns):
    try:
        import torch
    except ImportError:
        die("`dump` needs torch — run it inside the training environment")
    repo = Path(ns.repo).resolve()
    if not (repo / "hyptorch").is_dir():
        die(f"--repo {repo} does not look like the HypCD repo")
    sys.path.insert(0, str(repo))
    meta = parse_namespace(ns.log) if ns.log else {}
    dataset = ns.dataset or meta.get("dataset") or die("--dataset required")
    cr = ns.cr or meta.get("cr") or die("--cr required")
    c = ns.c or meta.get("c") or 0.1
    model_name = ns.model_name or meta.get("model_name") or "v2"

    from types import SimpleNamespace
    from data.augmentations import get_transform
    from data.get_datasets import get_datasets, get_class_splits
    args = SimpleNamespace(dataset_name=dataset, prop_train_labels=0.5,
                           use_ssb_splits=True, image_size=224,
                           transform="imagenet", seed=meta.get("seed", 0),
                           n_views=2, interpolation=3, crop_pct=0.875)
    args = get_class_splits(args)
    _, test_tf = get_transform(args.transform, image_size=args.image_size,
                               args=args)
    from data.data_utils import MergedDataset  # noqa: F401  (repo import check)
    _, _, unlab, _ = get_datasets(args.dataset_name, test_tf, test_tf, args)
    loader = torch.utils.data.DataLoader(unlab, batch_size=ns.batch,
                                         num_workers=ns.workers, shuffle=False)
    # get_datasets remaps targets to 0..K-1: train_classes first, then the rest
    order = np.asarray(list(args.train_classes) + list(args.unlabeled_classes))
    n_known = len(args.train_classes)

    if model_name == "v2":
        from models import vision_transformer2 as vits
    else:
        from models import vision_transformer as vits
    backbone = vits.__dict__["vit_base"]()
    sd = torch.load(ns.ckpt, map_location="cpu")
    backbone.load_state_dict(sd)
    dev = ns.device
    backbone.to(dev).eval()

    proj_path = ns.proj or str(Path(ns.ckpt).with_name(
        Path(ns.ckpt).name.replace("model", "model_proj_head", 1)))
    gate = None
    if Path(proj_path).exists():
        psd = torch.load(proj_path, map_location="cpu")
        if any(k.startswith("alpha.") for k in psd):
            from hyptorch.radial_gate import RadialToPoincare
            gate = RadialToPoincare(c=c, cr=cr,
                                    levels=tuple(sorted({k.split(".")[1]
                                                for k in psd
                                                if k.startswith("alpha.")})))
            gate.load_state_dict(psd)
            gate.to(dev).eval()
            info(f"loaded radial gate from {proj_path}: levels {gate.levels}")
    else:
        info(f"projector state not found at {proj_path}; dumping raw norms "
             "and clip depths only")

    lv_mean = ns.mean_level if gate else None
    r_raw, depth, tangents, labels = [], {lv: [] for lv in
                                          (gate.levels if gate else ())}, [], []
    with torch.no_grad():
        for batch in loader:
            images = batch[0]
            if ns.class_means and len(batch) > 1:
                labels.append(torch.as_tensor(batch[1]).cpu())
            feat = backbone(images.to(dev))
            r = feat.norm(dim=-1)
            r_raw.append(r.cpu())
            for lv in depth:
                r0 = getattr(gate, "r0_" + lv).to(dev)
                t = torch.log(r / r0)
                t = gate.T * torch.tanh(t / gate.T)
                s = gate.depth(lv).to(dev) * torch.exp(
                    gate.kappa(lv).to(dev) * t)
                s_bar = torch.clamp(s, max=gate.r_guard)
                depth[lv].append(s_bar.cpu())
                if ns.class_means and lv == lv_mean:
                    tangents.append((feat * (s_bar / r).unsqueeze(-1)).cpu())
            if ns.class_means and not gate:
                scale = torch.clamp(torch.as_tensor(cr, device=dev) / r,
                                    max=1.0)
                tangents.append((feat * scale.unsqueeze(-1)).cpu())
    outd = Path(ns.out or "dumps")
    outd.mkdir(parents=True, exist_ok=True)
    rr = torch.cat(r_raw).numpy()
    np.save(outd / "r_raw.npy", rr)
    np.save(outd / "depth_clip_rel.npy", np.minimum(rr, cr) / cr)
    info(f"wrote {outd/'r_raw.npy'}  (n={rr.size}, median={np.median(rr):.2f})"
         f"  and depth_clip_rel.npy")
    for lv, chunks in depth.items():
        d = torch.cat(chunks).numpy()
        np.save(outd / f"depth_{lv}_rel.npy", d / cr)
        info(f"wrote {outd}/depth_{lv}_rel.npy  "
             f"(mean={d.mean()/cr:.3f}, sd={d.std()/cr:.3f})")
    if ns.class_means:
        if not tangents or not labels:
            die("--class-means needs a loader that yields (image, label, ...)")
        import hyptorch.pmath as pmath
        V = torch.cat(tangents).float()
        y = torch.cat(labels).long().numpy()          # remapped 0..K-1
        y_orig = order[y]                             # original class ids
        is_old = y < n_known
        ids = np.unique(y_orig)
        means = torch.stack([V[torch.as_tensor(y_orig == i)].mean(0)
                             for i in ids])
        Z = pmath.project(pmath.expmap0(means, c=torch.tensor(float(c))),
                          c=torch.tensor(float(c))).numpy()
        out = outd / "class_means.npz"
        np.savez_compressed(out, class_means=Z.astype(np.float32),
                            class_mean_ids=ids.astype(np.int64),
                            labels=y_orig.astype(np.int64), is_old=is_old,
                            curvature_c=np.array([c], dtype=np.float32))
        n_novel = int(sum(not is_old[y_orig == i][0] for i in ids))
        info(f"wrote {out}  ({len(ids)} classes, {n_novel} novel, level "
             f"'{lv_mean or 'clip'}')"
             "\n  feed it to:  python poster_extract.py f8 --npz "
             f"{out} --names <classes.txt> --dataset {dataset}")
    info("note: object-level depths computed here feed IMAGE features through "
         "the object gate — for exact object-view depths use the training "
         "pipeline's crops, or keep F5's log mode for that row.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build():
    ap = argparse.ArgumentParser(
        prog="poster_extract.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="index & classify a directory of logs")
    p.add_argument("logdir")
    p.add_argument("--glob", default=None, help="regex filter on paths")
    p.add_argument("--csv", default=None)

    p = sub.add_parser("f3a", help="curvature tug-of-war from an *_mc_* log")
    p.add_argument("--log", required=True, help="a *unbind_su* mc log")
    p.add_argument("--probe-log", default=None,
                   help="a *unbind_rep_cls* mc log for the rep/cls probe")
    p.add_argument("--probe-tail", type=int, default=5,
                   help="epochs averaged for the probe values")
    p.add_argument("--unbind", type=float, default=None)
    p.add_argument("--smooth", type=int, default=7)
    viz_passthrough(p)

    p = sub.add_parser("f6", help="direction selectivity from entail logs")
    p.add_argument("--obj-log", required=True)
    p.add_argument("--img-log", default=None)
    p.add_argument("--img-flat0", action="store_true",
                   help="draw the image-as-parent line as a constant 0% "
                        "(no log needed)")
    p.add_argument("--smooth", type=int, default=5)
    viz_passthrough(p)

    p = sub.add_parser("f5", help="depth stratification from a radial-gate log")
    p.add_argument("--on-log", required=True)
    p.add_argument("--off-log", default=None)
    p.add_argument("--epoch", default="last", help="last | best | <int>")
    p.add_argument("--n-synth", type=int, default=4000)
    p.add_argument("--cr", type=float, default=None)
    p.add_argument("--R", type=float, default=None)
    p.add_argument("--feas-on", type=float, default=None)
    p.add_argument("--foot", default=None)
    p.add_argument("--schematic", action="store_true",
                   help="qualitative depth axis (see poster_viz f5 --schematic)")
    p.add_argument("--no-feas", action="store_true",
                   help="drop the feasibility badge (F6 already carries it)")
    for lv in ("part", "object", "image"):
        p.add_argument(f"--on-{lv}", default=None,
                       help=f"real per-sample depths for the ON {lv} row")
    viz_passthrough(p)

    p = sub.add_parser("f3b", help="shell needle from a baseline log or dump")
    p.add_argument("--log", default=None)
    p.add_argument("--radii", default=None, help="r_raw.npy from `dump`")
    p.add_argument("--key", default=None)
    p.add_argument("--cr", type=float, default=None)
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--pinned", type=float, default=0.93,
                   help="share of samples drawn exactly on the shell "
                        "(rest form a thin inner tail)")
    p.add_argument("--tail-scale", type=float, default=0.018)
    p.add_argument("--foot", default=None)
    viz_passthrough(p)

    p = sub.add_parser("r4", help="geometry scoreboard table from logs")
    p.add_argument("--obj-log", nargs="+", required=True,
                   help="object-as-parent run(s), one per seed")
    p.add_argument("--off-log", nargs="+", default=None,
                   help="same recipe without the entailment term")
    p.add_argument("--img-log", nargs="+", default=None,
                   help="image-as-parent (reverse direction) run(s)")
    p.add_argument("--tail", type=int, default=20,
                   help="epochs averaged for the 'after training' cells")
    p.add_argument("--feas-thresh", type=float, default=50.0)
    p.add_argument("--gap-thresh", type=float, default=0.20)
    p.add_argument("--no-acc", action="store_true",
                   help="drop the accuracy row")
    p.add_argument("--foot", default=None)
    viz_passthrough(p)

    p = sub.add_parser("f8", help="induced tree from an embedding dump")
    p.add_argument("--npz", required=True,
                   help="*_embedding.npz from visualize/draw_poincare.py "
                        "--save_embeddings")
    p.add_argument("--names", default=None,
                   help="class-name list, e.g. CUB_200_2011/classes.txt")
    p.add_argument("--parent-from", default="last", choices=("last", "first"),
                   help="genus token: last word (CUB) or first (Cars make)")
    p.add_argument("--parent-map", default=None,
                   help="JSON {class_name: parent} overriding --parent-from")
    p.add_argument("--max-leaves", type=int, default=24)
    p.add_argument("--groups", nargs="+", default=None,
                   help="show exactly these genera")
    p.add_argument("--rank", default="novel",
                   choices=("novel", "size", "correct"),
                   help="which groups to keep when the leaf budget bites")
    p.add_argument("--dataset", default=None)
    p.add_argument("--c", type=float, default=None)
    viz_passthrough(p)

    p = sub.add_parser("dump", help="per-sample norms/depths from a checkpoint")
    p.add_argument("--repo", required=True)
    p.add_argument("--ckpt", required=True, help="backbone model[_best].pt")
    p.add_argument("--proj", default=None, help="projector state (auto)")
    p.add_argument("--log", default=None, help="run log to inherit config")
    p.add_argument("--dataset", default=None)
    p.add_argument("--model-name", default=None, choices=(None, "v1", "v2"))
    p.add_argument("--cr", type=float, default=None)
    p.add_argument("--c", type=float, default=None)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--class-means", action="store_true",
                   help="also export class means in the ball (input for f8)")
    p.add_argument("--mean-level", default="image",
                   choices=("image", "object"),
                   help="which gate level the class means go through")
    p.add_argument("-o", "--out", default="dumps")
    return ap


CMDS = {"scan": cmd_scan, "f3a": cmd_f3a, "f6": cmd_f6, "f5": cmd_f5,
        "f3b": cmd_f3b, "f8": cmd_f8, "r4": cmd_r4, "dump": cmd_dump}


def main(argv=None):
    ns = build().parse_args(argv)
    CMDS[ns.cmd](ns)


if __name__ == "__main__":
    main()
