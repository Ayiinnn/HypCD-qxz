"""
Object-level branch losses for HypCD -- image<->object grounding (``_multi`` variant).

The object branch shares the backbone / projector / classifier with the image
branch (no new parameters) and adds three losses on the (foreground) object
crop:

  1. entailment (feature space)         -- image contains object   (relu cone)
  2. hyperbolic distance (feature space)-- image aligns with object (InfoNCE)
  3. classification closeness (logits)  -- image and object agree on the class

Two supervision modes, selected by ``--obj_sup_mode``:

* ``legacy`` (DEFAULT) -- the ORIGINAL same-view-diagonal supervision, bit-exact.
  Image view ``v`` of instance ``k`` is paired only with the SAME view's object
  crop: dist = single-positive InfoNCE on the diagonal, ent = elementwise cone
  (img_i -> obj_i), cls = supervised CE + same-view image->object distillation.
  With the object weights at 0 the branch vanishes and training reduces to the
  deterministic base pipeline (``000`` -> original HypCD); with non-zero weights
  this reproduces the original object-branch results.

* ``multi`` -- a conservative refinement, TAILORED PER LOSS, motivated by what
  the branch is for. Image<->object grounding is fundamentally INSTANCE-LEVEL
  (RegionCLIP / GLIP; HyCoCLIP pairs each image with its OWN box), so we keep a
  single strong positive everywhere and only remove obvious FALSE NEGATIVES.
  We deliberately do NOT add same-class / cross-view ATTRACTION: that is a
  class-level objective handled by the image branch's SupCon and by the shared
  classifier, and forcing it into this cross-modal InfoNCE (a) imposes an
  irreducible ``log(P)`` loss floor and (b) collapses an instance's two
  augmented views onto one point in the shared backbone -- which cratered
  New-class accuracy in the first ablation. Per loss:

    dist  -- ONE strong positive img_i<->obj_i. The pairs
             {cross-view same-instance, labelled same-class, confident pseudo
             same-class} are made NEUTRAL: removed from the InfoNCE denominator
             (false-negative elimination, Huynh et al. WACV'22) instead of being
             attracted OR repelled. One positive => no floor, no view collapse;
             neutral (vs. legacy's negative) merely stops the loss from pushing
             apart crops that are the same object / same class. Pseudo pairs are
             gated by teacher confidence (``--obj_pl_thresh``) so they only act
             once the classifier sharpens (incremental scheme, Chen et al.'22).
    ent   -- DIAGONAL ONLY, img_i -> obj_i (identical to legacy). "image
             contains object" is strictly true only for the same instance+view;
             cross-view or same-class entailment is geometrically wrong, so
             nothing is added to the cone.
    cls   -- supervised CE (unchanged) + image->object distillation from the
             VIEW-AVERAGED image teacher (lower variance, consistent with
             SimGCD's cross-view DistillLoss). ``--obj_cls_teacher same`` reverts
             to the original per-view teacher. Same-(pseudo-)label samples need
             no pairwise term here -- they already share a classifier prototype.

Design note: the richer multi-positive / part-hierarchy machinery is left for a
future PART-LEVEL branch. For image-object grounding the structure above is
deliberately simple, and the ONLY differences from ``legacy`` are (i) the dist
denominator (false-negative elimination) and (ii) the cls teacher (view
averaging) -- both principled, both low-risk, both individually switchable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from hyptorch.pmath import dist_matrix
from hyptorch.entailment_v4 import entailment_cone_loss, entailment_cone_stats


# --------------------------------------------------------------------------- #
# (2) single-positive hyperbolic-distance InfoNCE with an optional
#     denominator mask (false-negative elimination).
# --------------------------------------------------------------------------- #
def masked_info_nce_hyp_dist(a, b, c, temperature, neutral=None):
    """Symmetric single-positive InfoNCE on the Poincare ball.

    Row ``i`` of ``a`` is the positive of row ``i`` of ``b`` (image view ``i``
    <-> object crop ``i``), using the negative Poincare distance as similarity.
    ``neutral`` (optional ``(N, N)`` bool, diagonal False) marks pairs to REMOVE
    from the softmax denominator -- false negatives we neither attract nor
    repel. ``neutral=None`` (or all-False) gives the plain diagonal InfoNCE
    (bit-exact to the original), so ``legacy`` mode uses this same function.
    """
    n = a.shape[0]
    targets = torch.arange(n, device=a.device)

    def _one_direction(logits, mask):
        if mask is not None:
            # keep the diagonal (the positive); drop masked negatives from the denom.
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return F.cross_entropy(logits, targets)

    logits_ab = -dist_matrix(a, b, c=c) / temperature
    logits_ba = -dist_matrix(b, a, c=c) / temperature
    # neutral is symmetric at the instance level; use its transpose for the b->a direction.
    mask_ab = neutral
    mask_ba = neutral.t() if neutral is not None else None
    return 0.5 * (_one_direction(logits_ab, mask_ab) + _one_direction(logits_ba, mask_ba))


def build_neutral_mask(class_labels, mask_lab, img_teacher_logits, n_views, pl_thresh, pl_temp=0.1):
    """Boolean ``(N, N)`` mask of pairs to EXCLUDE from the dist denominator.

    ``N = n_views * B``; rows/cols follow the trainer's view-major layout
    (``torch.cat(views)``): rows ``[vB, (v+1)B)`` are view ``v`` of instances
    ``0..B-1``. The excluded (neutral) pairs are the union of:
      * cross-view, same instance   (the same object under another augmentation);
      * labelled same-class          (both labelled, same GT class, other instance);
      * confident pseudo same-class  (>=1 unlabelled side, teacher-argmax agrees
                                      and max prob >= ``pl_thresh``).
    The diagonal is NEVER excluded (it is the positive). All teacher-derived
    quantities are detached.
    """
    device = class_labels.device
    b = mask_lab.shape[0]
    n = n_views * b
    with torch.no_grad():
        inst = torch.arange(b, device=device).repeat(n_views)                 # (N,)
        same_inst = inst[:, None] == inst[None, :]
        eye = torch.eye(n, dtype=torch.bool, device=device)
        xview = same_inst & ~eye                                              # cross-view

        labf = mask_lab.repeat(n_views)                                       # (N,)
        lblf = class_labels.repeat(n_views)                                   # (N,)
        lab_pair = labf[:, None] & labf[None, :]
        lab_same = lab_pair & (lblf[:, None] == lblf[None, :]) & ~same_inst

        # confidence-gated pseudo-labels from the view-averaged image teacher.
        tp = F.softmax(img_teacher_logits.detach() / pl_temp, dim=-1)         # (N, C)
        p_inst = tp.view(n_views, b, -1).mean(dim=0)                          # (B, C)
        conf, pl = p_inst.max(dim=-1)                                         # (B,)
        ok = (conf >= pl_thresh).repeat(n_views)                             # (N,)
        plf = pl.repeat(n_views)                                              # (N,)
        bridge = torch.where(labf, lblf, plf)
        bok = labf | (~labf & ok)
        pl_same = ((bridge[:, None] == bridge[None, :])
                   & bok[:, None] & bok[None, :] & ~same_inst & ~lab_pair)

        neutral = (xview | lab_same | pl_same) & ~eye
    return neutral


# --------------------------------------------------------------------------- #
# (3) classification closeness.
# --------------------------------------------------------------------------- #
def classification_closeness(obj_logits, img_teacher_logits, class_labels, mask_lab, sup_weight,
                             student_temp=0.1, teacher_temp=0.04, teacher_mode="both", n_views=2):
    """Make the object's label distribution match the image's.

    * supervised: cross-entropy of object logits to the GT label of the source
      image (labelled samples, both views) -- mirrors the image-branch cls loss
      and is already class-level (same-class objects share a prototype row).
    * unsupervised: image -> object distillation. ``teacher_mode='same'`` uses
      the per-view image teacher (original); ``'both'`` uses the view-averaged
      image teacher (lower variance). Teacher logits are detached upstream.
    """
    sup_logits = torch.cat([f[mask_lab] for f in (obj_logits / student_temp).chunk(n_views)], dim=0)
    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(n_views)], dim=0)
    if sup_logits.shape[0] > 0:
        loss_sup = F.cross_entropy(sup_logits, sup_labels)
    else:  # guard a batch without labelled samples (empty CE would be NaN).
        loss_sup = obj_logits.sum() * 0.0

    student_logp = F.log_softmax(obj_logits / student_temp, dim=-1)
    teacher_p = F.softmax(img_teacher_logits / teacher_temp, dim=-1)
    if teacher_mode == "same":
        target_p = teacher_p
    elif teacher_mode == "both":
        views = teacher_p.chunk(n_views)
        target_p = torch.stack(views, dim=0).mean(dim=0).repeat(n_views, 1)
    else:
        raise ValueError(f"unknown obj_cls_teacher mode: {teacher_mode}")
    loss_con = -(target_p * student_logp).sum(dim=-1).mean()
    return sup_weight * loss_sup + (1.0 - sup_weight) * loss_con


class ObjectBranch:
    """Compute and combine the three object-level (grounding) losses."""

    def __init__(self, args):
        self.c = args.c
        self.w_ent = args.obj_entail_weight
        self.w_dist = args.obj_dist_weight
        self.w_cls = args.obj_cls_weight
        self.entail_parent = args.obj_entail_parent          # 'image' or 'object'
        self.aperture_scale = args.obj_aperture_scale
        self.min_radius = args.obj_min_radius
        self.dist_temp = args.obj_dist_temp
        self.sup_weight = args.sup_weight
        self.student_temp = 0.1
        self.n_views = getattr(args, "n_views", 2)

        # supervision mode (getattr => tolerant of old/other launch scripts)
        mode = getattr(args, "obj_sup_mode", "legacy")
        if getattr(args, "obj_legacy_supervision", False):
            mode = "legacy"
        self.mode = mode
        self.pl_thresh = getattr(args, "obj_pl_thresh", 0.7)
        self.cls_teacher = getattr(args, "obj_cls_teacher", "both")
        if self.mode == "legacy":
            # bit-exact original: no denominator masking, per-view cls teacher.
            self.cls_teacher = "same"

    def supervision_desc(self):
        if self.mode == "legacy":
            return "legacy (same-view diagonal: dist=1pos, ent=diag img->obj, cls=same-view teacher)"
        return ("multi grounding | dist=1pos + neutral[xview, lab-same, pl-same(thr={})] | "
                "ent=diag img->obj | cls=sup-CE + img->obj distill (teacher={})".format(
                    self.pl_thresh, self.cls_teacher))

    def __call__(self, img_feat, obj_feat, obj_logits, img_teacher_logits,
                 class_labels, mask_lab, teacher_temp):
        # (1) entailment cone -- DIAGONAL ONLY in both modes (img_i -> obj_i).
        if self.entail_parent == "image":
            parent, child = img_feat, obj_feat
        else:
            parent, child = obj_feat, img_feat
        loss_ent = entailment_cone_loss(
            parent, child, self.c,
            aperture_scale=self.aperture_scale, min_radius=self.min_radius,
        )
        # read-only cone diagnostics (sat rate / exterior angle / aperture);
        # no_grad, no RNG, does not alter the loss path.
        ent_stats = entailment_cone_stats(
            parent, child, self.c,
            aperture_scale=self.aperture_scale, min_radius=self.min_radius,
        )

        # (2) hyperbolic distance -- single positive; multi mode removes false
        #     negatives (cross-view / same-class / confident pseudo) from the denom.
        if self.mode == "legacy":
            neutral = None
        else:
            neutral = build_neutral_mask(
                class_labels, mask_lab, img_teacher_logits,
                self.n_views, self.pl_thresh, self.student_temp,
            )
        loss_dist = masked_info_nce_hyp_dist(img_feat, obj_feat, self.c, self.dist_temp, neutral)

        # (3) classification closeness.
        loss_cls = classification_closeness(
            obj_logits, img_teacher_logits, class_labels, mask_lab,
            self.sup_weight, self.student_temp, teacher_temp,
            teacher_mode=self.cls_teacher, n_views=self.n_views,
        )

        total = self.w_ent * loss_ent + self.w_dist * loss_dist + self.w_cls * loss_cls
        logs = {"obj_entail": loss_ent, "obj_dist": loss_dist, "obj_cls": loss_cls}
        logs.update(ent_stats)
        return total, logs