"""
Object-level branch losses for HypCD.

Given image features and (foreground) object features that live in the *same*
Poincare ball, plus their classifier logits, this module computes the three
object-level losses requested for ``hypersimgcd_org_det_ab_obj``:

  1. entailment (feature space)  -- HyCoCLIP relu-cone form, direction configurable
  2. hyperbolic distance (feature space) -- InfoNCE in Poincare distance, image<->object
  3. classification closeness (logit space) -- object should carry the same label
     distribution as its source image (supervised CE on labelled samples +
     image->object distillation on all samples).

Each loss has its own weight. The module owns no parameters (the backbone,
projector and classifier are shared with the image branch).

Supervision structure (revised)
-------------------------------
The original implementation supervised every loss with the *same-view diagonal
only*: the single positive / entailment child / distillation teacher of image
row ``i`` was object row ``i``. That treats the object crop of the OTHER view
of the SAME instance -- and, for labelled samples, all GT same-class crops --
as negatives of the InfoNCE, which are textbook false negatives
(Khosla et al., NeurIPS 2020; Huynh et al., WACV 2022).

The revised supervision, built once per batch by :func:`build_supervision_masks`
and shared by the two feature-space losses:

* instance level (all samples). Both views of an instance are the same scene,
  so both object crops are positives of both image views
  (``--obj_dist_crossview``) and valid entailment children of both image views
  (``--obj_ent_crossview``). This follows the standard multi-view definition of
  a positive (SimCLR / SupCon) and HyCoCLIP's per-composition pairing.
* labelled same-class (GT). Following SupCon -- and mirroring the image
  branch's own ``SupConLoss`` -- object crops of all labelled samples with the
  same GT class are treated as positives of a labelled anchor
  (``--obj_dist_lab_mode pos``, weight ``--obj_dist_lab_weight``), and as
  weak (down-weighted) entailment children (``--obj_ent_lab_weight``).
  ``neutral`` removes them from the denominator instead (false-negative
  elimination); ``neg`` recovers the original behaviour.
* unlabelled same-pseudo-label. Pseudo-labels are the argmax of the
  view-averaged, detached image-teacher distribution, accepted only above a
  confidence threshold (``--obj_pl_thresh``, measured at the student
  temperature so acceptance ramps up naturally as the classifier sharpens --
  the incremental scheme of Chen et al., ICLR 2022). The default treatment is
  ``neutral`` (drop same-pseudo-label pairs from the negatives, i.e.
  false-negative *elimination*); ``pos`` additionally attracts them with weight
  ``--obj_dist_pl_weight``; ``neg`` recovers the original behaviour. Positives
  for unlabelled samples remain view-level by default, matching SimGCD's
  deliberate choice for unlabelled representation learning. Confident
  pseudo-labels also bridge labelled<->unlabelled pairs (CiPR-style), governed
  by the same pl mode. No pseudo-label pairs are ever fed to the entailment
  cone (hard geometric constraints are too brittle for noisy labels).
* classification space. Same-(pseudo-)label samples already share a prototype
  row of the shared classifier, so per-sample distillation keeps them
  implicitly attracted ("neutral") without pairwise terms; the revision only
  upgrades the teacher: ``--obj_cls_teacher both`` distills each object crop
  from the view-averaged image teacher of its instance (lower-variance,
  consistent with SimGCD's cross-view ``DistillLoss``).

``--obj_legacy_supervision`` routes every loss through the ORIGINAL code path
(bit-exact) for A/B comparison against previous runs.

It is written so the same primitives can later serve a true part-level branch:
just feed additional (parent, child) feature pairs / masks into
``weighted_entailment_cone_loss`` / ``masked_info_nce_hyp_dist``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from hyptorch.pmath import dist_matrix
from hyptorch.entailment import (
    entailment_cone_loss,
    entailment_cone_violation_pairwise,
)


# --------------------------------------------------------------------------- #
# Legacy (original) losses -- kept verbatim for --obj_legacy_supervision.
# --------------------------------------------------------------------------- #
def info_nce_hyp_dist(a: torch.Tensor, b: torch.Tensor, c, temperature: float = 0.1):
    """Symmetric InfoNCE that pulls index-matched ``a``/``b`` together and pushes
    mismatched pairs apart, using the (negative) Poincare distance as similarity.

    ``a`` and ``b`` are ``(N, D)`` Poincare-ball points; row ``i`` of ``a`` is the
    positive of row ``i`` of ``b`` (here: image view ``i`` <-> object crop ``i``).
    """
    n = a.shape[0]
    targets = torch.arange(n, device=a.device)
    logits_ab = -dist_matrix(a, b, c=c) / temperature
    logits_ba = -dist_matrix(b, a, c=c) / temperature
    return 0.5 * (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets))


# --------------------------------------------------------------------------- #
# Supervision masks shared by the two feature-space losses.
# --------------------------------------------------------------------------- #
def build_supervision_masks(
    class_labels: torch.Tensor,
    mask_lab: torch.Tensor,
    img_teacher_logits: torch.Tensor,
    n_views: int,
    pl_thresh: float,
    pl_temp: float = 0.1,
):
    """Build the pair-relation masks over the ``(N, N)`` grid, ``N = n_views*B``.

    Rows/columns follow the trainer's view-major layout (``torch.cat(views)``):
    rows ``[vB, (v+1)B)`` are view ``v`` of instances ``0..B-1``.

    Returns a dict of boolean masks (all symmetric):
      diag        -- same view of the same instance (the original positive).
      xview_inst  -- same instance, different view.
      lab_same    -- both labelled, same GT class, different instance.
      pl_same     -- same (confident pseudo-)label, different instance, at
                     least one side unlabelled. Labelled samples participate
                     with their GT label; unlabelled samples participate only
                     if the max prob of their view-averaged teacher
                     distribution (softmax at ``pl_temp``) exceeds
                     ``pl_thresh``.
    All teacher-derived quantities are detached (no gradient flows through the
    mask construction).
    """
    device = class_labels.device
    b = mask_lab.shape[0]
    n = n_views * b

    with torch.no_grad():
        inst_id = torch.arange(b, device=device).repeat(n_views)               # (N,)
        same_inst = inst_id[:, None] == inst_id[None, :]                       # (N, N)
        diag = torch.eye(n, dtype=torch.bool, device=device)
        xview_inst = same_inst & ~diag

        lab_full = mask_lab.repeat(n_views)                                    # (N,)
        labels_full = class_labels.repeat(n_views)                             # (N,)
        lab_pair = lab_full[:, None] & lab_full[None, :]
        same_cls = labels_full[:, None] == labels_full[None, :]
        lab_same = lab_pair & same_cls & ~same_inst

        # ---- confidence-gated pseudo-labels (view-averaged teacher) ----
        teacher_p = F.softmax(img_teacher_logits.detach() / pl_temp, dim=-1)   # (N, C)
        p_inst = teacher_p.view(n_views, b, -1).mean(dim=0)                    # (B, C)
        conf, pl = p_inst.max(dim=-1)                                          # (B,)
        pl_ok = (conf >= pl_thresh).repeat(n_views)                            # (N,)
        pl_full = pl.repeat(n_views)                                           # (N,)

        # labelled samples bridge with their GT label; unlabelled need a
        # confident pseudo-label to participate at all.
        bridge_label = torch.where(lab_full, labels_full, pl_full)
        bridge_ok = lab_full | (~lab_full & pl_ok)
        pl_same = (
            (bridge_label[:, None] == bridge_label[None, :])
            & bridge_ok[:, None] & bridge_ok[None, :]
            & ~same_inst & ~lab_pair
        )

    return {"diag": diag, "xview_inst": xview_inst, "lab_same": lab_same, "pl_same": pl_same}


def _compose_pos_and_valid(masks, crossview, lab_mode, lab_weight, pl_mode, pl_weight):
    """Turn the relation masks into (positive-weight, denominator-valid) matrices.

    ``*_mode`` in {'neg', 'neutral', 'pos'}:
      neg     -- pair stays a plain negative (original behaviour).
      neutral -- pair is removed from the InfoNCE denominator
                 (false-negative elimination).
      pos     -- pair becomes a positive with the given weight
                 (SupCon-style attraction; weight 1.0 = full positive,
                 (0, 1) = weak positive).
    """
    pos_w = masks["diag"].float()
    if crossview:
        pos_w = pos_w + masks["xview_inst"].float()

    valid = torch.ones_like(pos_w, dtype=torch.bool)
    for key, mode, weight in (("lab_same", lab_mode, lab_weight),
                              ("pl_same", pl_mode, pl_weight)):
        if mode == "pos":
            pos_w = pos_w + weight * masks[key].float()
        elif mode == "neutral":
            valid = valid & ~masks[key]
        # mode == "neg": leave as ordinary negatives.
    valid = valid | (pos_w > 0)  # positives always stay in the denominator (SupCon).
    return pos_w, valid


# --------------------------------------------------------------------------- #
# (2') masked / weighted hyperbolic-distance InfoNCE.
# --------------------------------------------------------------------------- #
def masked_info_nce_hyp_dist(
    a: torch.Tensor,
    b: torch.Tensor,
    c,
    temperature: float,
    pos_w: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-8,
):
    """Generalized symmetric InfoNCE between two point sets on the Poincare ball.

    ``pos_w`` (N, N) >= 0 are positive-pair weights; ``valid`` (N, N) bool marks
    the pairs kept in the softmax denominator. Per anchor the loss is the
    weighted mean over positives of ``-log p`` (the SupCon ``L_out`` form):

        L_i = - sum_j pos_w[i,j] * (s_ij - logsumexp_{k: valid[i,k]} s_ik)
                / sum_j pos_w[i,j]

    With ``pos_w = I`` and ``valid = all-ones`` this reduces exactly to the
    original diagonal InfoNCE.
    """
    def _one_direction(logits, w, v):
        masked = logits.masked_fill(~v, torch.finfo(logits.dtype).min)
        log_denom = torch.logsumexp(masked, dim=1, keepdim=True)
        log_prob = logits - log_denom
        w_sum = w.sum(dim=1).clamp_min(eps)
        return (-(w * log_prob).sum(dim=1) / w_sum).mean()

    logits_ab = -dist_matrix(a, b, c=c) / temperature
    logits_ba = -dist_matrix(b, a, c=c) / temperature
    return 0.5 * (
        _one_direction(logits_ab, pos_w, valid)
        + _one_direction(logits_ba, pos_w.t(), valid.t())
    )


# --------------------------------------------------------------------------- #
# (1') weighted entailment-cone loss over multiple (parent, child) pairs.
# --------------------------------------------------------------------------- #
def weighted_entailment_cone_loss(
    parent: torch.Tensor,
    child: torch.Tensor,
    c,
    pair_w: torch.Tensor,
    aperture_scale: float = 1.2,
    min_radius: float = 0.1,
    eps: float = 1e-8,
):
    """Weighted mean of pairwise cone violations; ``pair_w[i, j]`` weights the
    constraint "child j lies inside the cone of parent i"."""
    viol = entailment_cone_violation_pairwise(
        parent, child, c, aperture_scale=aperture_scale, min_radius=min_radius
    )
    return (pair_w * viol).sum() / pair_w.sum().clamp_min(eps)


# --------------------------------------------------------------------------- #
# (3') classification closeness.
# --------------------------------------------------------------------------- #
def classification_closeness(
    obj_logits: torch.Tensor,
    img_teacher_logits: torch.Tensor,
    class_labels: torch.Tensor,
    mask_lab: torch.Tensor,
    sup_weight: float,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
    teacher_mode: str = "same",
    n_views: int = 2,
):
    """Make the object's label distribution match the image's.

    * supervised: cross-entropy of object logits to the ground-truth label of its
      source image (labelled samples only), mirroring the image-branch ``cls_loss``.
      This is already class-level supervision: all same-class objects are pulled
      to the same prototype row of the shared classifier.
    * unsupervised: image -> object distillation. ``teacher_mode`` selects the
      teacher of object view ``v``:
        'same'  -- image teacher of view ``v`` (original behaviour);
        'cross' -- mean of the OTHER views' image teachers (SimGCD DistillLoss
                   convention);
        'both'  -- mean of ALL views' image teachers (view-ensembled, lowest
                   variance; default).
      Samples sharing a (pseudo-)label need no explicit pairwise term here:
      matching the same teacher prototype already attracts them ("neutral"),
      and pairwise attraction would double-count / amplify confirmation bias.

    ``obj_logits`` / ``img_teacher_logits`` are ``(n_views*B, C)`` in the usual
    view-major chunk layout; ``class_labels`` / ``mask_lab`` have length ``B``.
    """
    # supervised cross-entropy on labelled samples (both views).
    sup_logits = torch.cat([f[mask_lab] for f in (obj_logits / student_temp).chunk(n_views)], dim=0)
    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(n_views)], dim=0)
    if sup_logits.shape[0] > 0:
        loss_sup = F.cross_entropy(sup_logits, sup_labels)
    else:  # guard a batch without labelled samples (empty CE would be NaN).
        loss_sup = obj_logits.sum() * 0.0

    # image -> object consistency (teacher is detached upstream).
    student_logp = F.log_softmax(obj_logits / student_temp, dim=-1)
    teacher_p = F.softmax(img_teacher_logits / teacher_temp, dim=-1)
    if teacher_mode == "same":
        target_p = teacher_p
    else:
        views = teacher_p.chunk(n_views)                      # n_views x (B, C)
        mean_all = torch.stack(views, dim=0).mean(dim=0)      # (B, C)
        if teacher_mode == "both":
            target_p = mean_all.repeat(n_views, 1)
        elif teacher_mode == "cross":
            if n_views == 2:
                target_p = torch.cat([views[1], views[0]], dim=0)
            else:  # mean over the other views
                target_p = torch.cat(
                    [(mean_all * n_views - v) / (n_views - 1) for v in views], dim=0
                )
        else:
            raise ValueError(f"unknown obj_cls_teacher mode: {teacher_mode}")
    loss_con = -(target_p * student_logp).sum(dim=-1).mean()

    return sup_weight * loss_sup + (1.0 - sup_weight) * loss_con


class ObjectBranch:
    """Compute and combine the three object-level losses."""

    def __init__(self, args):
        self.c = args.c
        self.w_ent = args.obj_entail_weight
        self.w_dist = args.obj_dist_weight
        self.w_cls = args.obj_cls_weight
        self.entail_parent = args.obj_entail_parent      # 'image' or 'object'
        self.aperture_scale = args.obj_aperture_scale
        self.min_radius = args.obj_min_radius
        self.dist_temp = args.obj_dist_temp
        self.sup_weight = args.sup_weight
        self.student_temp = 0.1
        self.n_views = getattr(args, "n_views", 2)

        # ---- supervision structure (getattr => old scripts keep working) ----
        self.legacy = getattr(args, "obj_legacy_supervision", False)
        # feature-space InfoNCE
        self.dist_crossview = getattr(args, "obj_dist_crossview", True)
        self.dist_lab_mode = getattr(args, "obj_dist_lab_mode", "pos")
        self.dist_lab_weight = getattr(args, "obj_dist_lab_weight", 1.0)
        self.dist_pl_mode = getattr(args, "obj_dist_pl_mode", "neutral")
        self.dist_pl_weight = getattr(args, "obj_dist_pl_weight", 0.5)
        self.pl_thresh = getattr(args, "obj_pl_thresh", 0.7)
        # entailment cone
        self.ent_crossview = getattr(args, "obj_ent_crossview", True)
        self.ent_lab_weight = getattr(args, "obj_ent_lab_weight", 0.5)
        # classification space
        self.cls_teacher = getattr(args, "obj_cls_teacher", "both")
        if self.legacy:
            self.dist_crossview = False
            self.dist_lab_mode = "neg"
            self.dist_pl_mode = "neg"
            self.ent_crossview = False
            self.ent_lab_weight = 0.0
            self.cls_teacher = "same"

    def supervision_desc(self):
        if self.legacy:
            return "legacy(same-view diagonal only)"
        return (
            "dist[xview={} lab={}({}) pl={}({}) thr={}] "
            "ent[xview={} lab_w={}] cls[teacher={}]".format(
                self.dist_crossview, self.dist_lab_mode, self.dist_lab_weight,
                self.dist_pl_mode, self.dist_pl_weight, self.pl_thresh,
                self.ent_crossview, self.ent_lab_weight, self.cls_teacher,
            )
        )

    def __call__(
        self,
        img_feat: torch.Tensor,
        obj_feat: torch.Tensor,
        obj_logits: torch.Tensor,
        img_teacher_logits: torch.Tensor,
        class_labels: torch.Tensor,
        mask_lab: torch.Tensor,
        teacher_temp: float,
    ):
        # ---- pair-relation masks shared by the feature-space losses ----
        masks = None
        if not self.legacy:
            masks = build_supervision_masks(
                class_labels, mask_lab, img_teacher_logits,
                n_views=self.n_views, pl_thresh=self.pl_thresh,
                pl_temp=self.student_temp,
            )

        # (1) entailment cone --------------------------------------------------
        if self.entail_parent == "image":
            parent, child = img_feat, obj_feat
        else:
            parent, child = obj_feat, img_feat
        if self.legacy or (not self.ent_crossview and self.ent_lab_weight == 0.0):
            # bit-exact original elementwise path.
            loss_ent = entailment_cone_loss(
                parent, child, self.c,
                aperture_scale=self.aperture_scale, min_radius=self.min_radius,
            )
        else:
            ent_w = masks["diag"].float()
            if self.ent_crossview:
                ent_w = ent_w + masks["xview_inst"].float()
            if self.ent_lab_weight > 0.0:
                # class-level (approximate) hierarchy => weak weight;
                # never driven by pseudo-labels.
                ent_w = ent_w + self.ent_lab_weight * masks["lab_same"].float()
            loss_ent = weighted_entailment_cone_loss(
                parent, child, self.c, ent_w,
                aperture_scale=self.aperture_scale, min_radius=self.min_radius,
            )

        # (2) hyperbolic distance (feature space) ------------------------------
        if self.legacy:
            loss_dist = info_nce_hyp_dist(img_feat, obj_feat, self.c, self.dist_temp)
        else:
            pos_w, valid = _compose_pos_and_valid(
                masks, self.dist_crossview,
                self.dist_lab_mode, self.dist_lab_weight,
                self.dist_pl_mode, self.dist_pl_weight,
            )
            loss_dist = masked_info_nce_hyp_dist(
                img_feat, obj_feat, self.c, self.dist_temp, pos_w, valid,
            )

        # (3) classification closeness (logit space) ---------------------------
        loss_cls = classification_closeness(
            obj_logits, img_teacher_logits, class_labels, mask_lab,
            self.sup_weight, self.student_temp, teacher_temp,
            teacher_mode=self.cls_teacher, n_views=self.n_views,
        )

        total = self.w_ent * loss_ent + self.w_dist * loss_dist + self.w_cls * loss_cls
        logs = {"obj_entail": loss_ent, "obj_dist": loss_dist, "obj_cls": loss_cls}
        return total, logs