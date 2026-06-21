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
projector and classifier are shared with the image branch). It is written so the
same primitives can later serve a true part-level branch: just feed additional
(parent, child) feature pairs into ``entailment_cone_loss`` / ``info_nce_hyp_dist``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from hyptorch.pmath import dist_matrix
from hyptorch.entailment import entailment_cone_loss


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


def classification_closeness(
    obj_logits: torch.Tensor,
    img_teacher_logits: torch.Tensor,
    class_labels: torch.Tensor,
    mask_lab: torch.Tensor,
    sup_weight: float,
    student_temp: float = 0.1,
    teacher_temp: float = 0.04,
):
    """Make the object's label distribution match the image's.

    * supervised: cross-entropy of object logits to the ground-truth label of its
      source image (labelled samples only), mirroring the image-branch ``cls_loss``.
    * unsupervised: image -> object distillation (object student distribution is
      pulled toward the detached image teacher distribution, per matching view).

    ``obj_logits`` / ``img_teacher_logits`` are ``(2N, C)`` with the usual two-view
    chunk layout; ``class_labels`` / ``mask_lab`` have length ``N``.
    """
    # supervised cross-entropy on labelled samples (both views).
    sup_logits = torch.cat([f[mask_lab] for f in (obj_logits / student_temp).chunk(2)], dim=0)
    sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
    loss_sup = F.cross_entropy(sup_logits, sup_labels)

    # image -> object consistency (teacher is detached upstream).
    student_logp = F.log_softmax(obj_logits / student_temp, dim=-1)
    teacher_p = F.softmax(img_teacher_logits / teacher_temp, dim=-1)
    loss_con = -(teacher_p * student_logp).sum(dim=-1).mean()

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
        # (1) entailment cone --------------------------------------------------
        if self.entail_parent == "image":
            parent, child = img_feat, obj_feat
        else:
            parent, child = obj_feat, img_feat
        loss_ent = entailment_cone_loss(
            parent, child, self.c,
            aperture_scale=self.aperture_scale, min_radius=self.min_radius,
        )

        # (2) hyperbolic distance (feature space) ------------------------------
        loss_dist = info_nce_hyp_dist(img_feat, obj_feat, self.c, self.dist_temp)

        # (3) classification closeness (logit space) ---------------------------
        loss_cls = classification_closeness(
            obj_logits, img_teacher_logits, class_labels, mask_lab,
            self.sup_weight, self.student_temp, teacher_temp,
        )

        total = self.w_ent * loss_ent + self.w_dist * loss_dist + self.w_cls * loss_cls
        logs = {"obj_entail": loss_ent, "obj_dist": loss_dist, "obj_cls": loss_cls}
        return total, logs
