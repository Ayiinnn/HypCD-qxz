"""
Object-level branch for HypSelEx -- image<->object grounding.

This is the SelEx counterpart of ``models/object_branch_multi.py`` (the SimGCD
``_multi`` branch). It shares the backbone / projection head with the image
branch (no new parameters) and adds object-level grounding losses on the
(foreground) object crop.

Why SelEx keeps only TWO of the SimGCD branch's three losses
------------------------------------------------------------
The SimGCD branch has three losses:

  1. entailment (feature space)          -- image contains object   (relu cone)
  2. hyperbolic distance (feature space) -- image aligns with object (InfoNCE)
  3. classification closeness (logits)   -- image and object agree on the class

Losses (1) and (2) need ONLY Poincare features, which SelEx has (both image and
object crops go through the *same* projection head -> same Poincare ball), so
they are ported verbatim (the primitives are imported, not re-implemented).

Loss (3) distils the object's *classifier logits* from the image's. SelEx is
NON-PARAMETRIC -- there is no hyperbolic classifier, only per-epoch (semi-sup)
KMeans prototypes used for evaluation and for the SupCon hierarchy. A prototype
"soft-classifier" surrogate was considered and deliberately REJECTED: it would
(i) live in the raw backbone feature space, not the Poincare space the other two
losses use; (ii) rely on KMeans cluster IDs that are re-fit with a random init
each epoch (``random_state=None``), so the distillation *target* would drift in
meaning epoch-to-epoch; and (iii) pull all same-cluster object crops toward one
prototype -- a class-level attraction of exactly the kind that collapsed
New-class accuracy in the SimGCD ablation. So obj_cls is DROPPED for SelEx (the
class-level objective is already carried by SelEx's own hierarchical SupCon).

Supervision modes (mirror the SimGCD branch, minus obj_cls)
-----------------------------------------------------------
* ``legacy`` (DEFAULT) -- plain same-view diagonal. ``dist`` is single-positive
  InfoNCE with every off-diagonal a NEGATIVE; ``ent`` is the elementwise cone
  ``img_i -> obj_i``. Conservative baseline: with the object weights at 0 the
  branch vanishes and training reduces to the base SelEx pipeline.
* ``multi`` -- ``dist`` keeps ONE strong positive ``img_i<->obj_i`` and merely
  REMOVES false negatives from the InfoNCE denominator (cross-view same-instance,
  labelled same-class, and same-cluster pseudo pairs) -- neutral, never attracted
  (no ``log(P)`` floor, no view collapse). The pseudo-label source is SelEx's OWN
  finest hierarchical cluster assignment (``preds_ind_list[0]``); for labelled
  samples the semi-supervised KMeans constrains them to their GT-class cluster,
  so no softmax/confidence gate is needed (SelEx already trusts these labels for
  its SupCon positives, and neutral-masking is a strictly weaker use). ``ent``
  stays diagonal (cross-view / same-class entailment is geometrically wrong).

Never adds same-class / cross-view ATTRACTION.
"""
from __future__ import annotations

import torch

# Reuse, do not re-implement: the feature-space primitives are shared with the
# SimGCD branch / the hyperbolic library.
from hyptorch.entailment import entailment_cone_loss
from models.object_branch_multi import masked_info_nce_hyp_dist


# --------------------------------------------------------------------------- #
# Neutral (false-negative) mask for the distance InfoNCE -- SelEx analog of
# object_branch_multi.build_neutral_mask, with the classifier softmax argmax
# replaced by SelEx's finest hierarchical cluster assignment.
# --------------------------------------------------------------------------- #
def build_neutral_mask_selex(class_labels, mask_lab, cluster_labels, n_views):
    """Boolean ``(N, N)`` mask of pairs to EXCLUDE from the dist denominator.

    ``N = n_views * B``; rows/cols follow the trainer's view-major layout
    (``torch.cat(views)``). Excluded (neutral) pairs are the union of:
      * cross-view, same instance   (same scene under another augmentation);
      * labelled same-class          (both labelled, same GT class, other instance);
      * same-cluster pseudo          (>=1 unlabelled side, SelEx finest cluster
                                      assignment agrees, other instance).
    The diagonal is NEVER excluded (it is the positive). ``cluster_labels`` is
    the per-sample finest cluster id (``preds_ind_list[0]`` gathered for the
    batch); for labelled samples it already equals the GT class, so ``bridge``
    below uses the GT label for labelled rows and the cluster id otherwise.
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

        clf = cluster_labels.to(device).long().repeat(n_views)               # (N,)
        bridge = torch.where(labf, lblf, clf)              # GT for labelled, cluster otherwise
        pl_same = (bridge[:, None] == bridge[None, :]) & ~same_inst & ~lab_pair

        neutral = (xview | lab_same | pl_same) & ~eye
    return neutral


class ObjectBranch:
    """Compute and combine the two object-level (grounding) losses for SelEx."""

    def __init__(self, args):
        self.c = args.c
        self.w_ent = args.obj_entail_weight
        self.w_dist = args.obj_dist_weight
        self.entail_parent = args.obj_entail_parent          # 'image' or 'object'
        self.aperture_scale = args.obj_aperture_scale
        self.min_radius = args.obj_min_radius
        self.dist_temp = args.obj_dist_temp
        self.n_views = getattr(args, "n_views", 2)

        # supervision mode (getattr => tolerant of old/other launch scripts)
        mode = getattr(args, "obj_sup_mode", "legacy")
        if getattr(args, "obj_legacy_supervision", False):
            mode = "legacy"
        self.mode = mode

    def supervision_desc(self):
        if self.mode == "legacy":
            return ("legacy (same-view diagonal: dist=1pos [off-diag negative], "
                    "ent=diag img->obj) | obj_cls DROPPED (SelEx is non-parametric)")
        return ("multi grounding | dist=1pos + neutral[xview, lab-same, "
                "cluster-same(preds_ind_list[0])] | ent=diag img->obj | "
                "obj_cls DROPPED (SelEx is non-parametric)")

    def __call__(self, img_feat, obj_feat, class_labels, mask_lab, cluster_labels=None):
        # (1) entailment cone -- DIAGONAL ONLY in both modes (img_i -> obj_i).
        if self.entail_parent == "image":
            parent, child = img_feat, obj_feat
        else:
            parent, child = obj_feat, img_feat
        loss_ent = entailment_cone_loss(
            parent, child, self.c,
            aperture_scale=self.aperture_scale, min_radius=self.min_radius,
        )

        # (2) hyperbolic distance -- single positive; multi mode removes false
        #     negatives (cross-view / same-class / same-cluster) from the denom.
        if self.mode == "legacy" or cluster_labels is None:
            neutral = None
        else:
            neutral = build_neutral_mask_selex(
                class_labels, mask_lab, cluster_labels, self.n_views,
            )
        loss_dist = masked_info_nce_hyp_dist(img_feat, obj_feat, self.c, self.dist_temp, neutral)

        total = self.w_ent * loss_ent + self.w_dist * loss_dist
        # logs are detached (the epoch-level print holds them past the batch's
        # backward; keeping the graph alive would waste memory).
        logs = {"obj_entail": loss_ent.detach(), "obj_dist": loss_dist.detach()}
        return total, logs