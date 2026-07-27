"""[hab] Clean A+B trainer: HypCD's Hyp-SimGCD (A, verbatim) + PartCo's part
branch (B, verbatim), joined ONLY through the shared backbone.

A = train_HypCD_org_det_ab.py, byte-identical outside the [hab]-marked blocks
    (`diff train_HypCD_org_det_ab.py train_hab.py` is the complete change list).
B = PartCo's part branch, plugged in EXACTLY the way PartCo plugs into SimGCD
    (paper's plug-and-play contract, two-view layout):
      * views = [aligned_view, contrastive_view] -- PartCo's own pair; the
        contrastive view is byte-identical to HypSimGCD's train transform;
      * patch tokens come from a second backbone pass over view 0 (aligned,
        crop/flip-free -> the 16x16 part-label grid stays token-aligned);
      * pseudo-labels for the unsup part loss = (student_out/0.1).chunk(2)[0],
        i.e. A's classifier logits on view 0 -- PartCo's verbatim line
        (PatchUnConLoss wraps them in no_grad internally);
      * patch_projector (LN + Dropout + Patch_Projection) and the losses
        PatchSupConLoss / PatchUnConLoss imported from models/partco_loss.py,
        a file-level copy of the PartCo repo (zero edits);
      * PartCo's warmup / 0.5-weighting assembly, verbatim.
    Temperature follows HypCD's system: part losses run at 0.07 *
    hyper_temp_scale by default (= the E-series sup setting; override with
    --part_temp 0.07 for PartCo's native value).  No dualization, no gate
    rewrites, no third view -- r1-r5 legacy carries nothing into this file.

The CLEAN BASELINE (CB) is --part_mode sup: PartCo minus the unsup part loss
on top of the untouched Hyp-SimGCD.  The ablation ladder adds the verbatim
unsup loss back and then isolates one variable per switch:
  --part_mode {none,sup,full}          none = pure-A control, sup = CB,
                                       full = + PartCo verbatim unsup
  --part_unsup_backprop {full,stopgrad} stopgrad = unsup trains projector only
  --part_pseudo_source {hyp,probe,oracle}
                                       probe = detached Euclidean linear head
                                       (H3: INVALID, old-biased -- kept for the
                                       record); oracle = ground-truth pseudo-
                                       labels (HO: max dose, perfect labels)
  --part_unsup_scale s                 S-series dose-response: multiplies ONLY
                                       the injected unsup term (ep0-29 bitwise
                                       unaffected)
  --hyper_kill_epoch E                 lambda_d = 0 from epoch E on (H2P: the
                                       clean-prefix D2 cell; H2 = --hyper_max_
                                       weight 0 has a DIFFERENT ep0-29 prefix)
  --hyper_max_weight 0                 (existing A knob) = distance leg off
                                       from epoch 0 (prefix differs)
  --part_temp                          part-loss temperature override (acts on
                                       the SUP part loss from epoch 0 too, so
                                       it changes the prefix)
  --part_unsup_ramp_epochs N           R-series: linear 1/N -> 1 ramp of the
                                       unsup multiplier over N epochs after
                                       onset (same steady state, slow approach)
  --eval_kmeans                        H1K probe: per-eval K-means on Euclidean
                                       backbone features (representation-level
                                       acc, bypassing the hyperbolic head)
  --stop_after_epoch E                 break after epoch E, schedules untouched
                                       (cheap cliff-window runs on the 200ep
                                       trajectory)
  --diag_grad                          per-print backbone-gradient diagnostics:
                                       3-way cosines + ABSOLUTE norms gnA/gnU/
                                       gnPS (cross-trainer comparable)
Round-2 note: the sampler now uses its own seeded generator under
--deterministic, so batch order is invariant to module construction.  Round-2
cells therefore need their own H1 re-anchor (H1R) and are NOT bitwise
comparable to the round-1 logs; conclusions remain comparable.
"""
import argparse

import copy
import sys
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# [hab] B uses PartCo's ported pipeline: its two-view transform triple
# (aligned / contrastive / test; the contrastive one is byte-identical to
# HypCD's own train transform) and its datasets (sync flip + part-label grid).
from data.partco.augmentations import get_transform
from data.partco.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path, dinov2_pretrain_path
from models.model import DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from models import vision_transformer as vits1
from models import vision_transformer2 as vits2
# [hab] PartCo losses, verbatim file copy (models/partco_loss.py).
from models.partco_loss import PatchSupConLoss, PatchUnConLoss
# [hab-M] method-stage extensions (models/partco_loss.py stays verbatim)
from models.partco_loss_ext import PatchUnConLossAllMin, PatchUnConLossMinDenom, PatchUnConLossSurgery
from models.part_grad_decomp import decompose_unsup_channels
from hyptorch.entailment_v4 import entailment_cone_loss


# [hab] PartCo's Patch_Projection, verbatim (partco models/model.py).
class Patch_Projection(torch.nn.Module):
    def __init__(self):
        super(Patch_Projection, self).__init__()

        self.linear_projection = nn.Sequential(
            nn.Linear(768, 128), # 256
        )
        self.non_linear_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 128),
        )
    def forward(self, x):
        return self.linear_projection(x) + self.non_linear_projection(x)


# [hab] second backbone pass over view 0 -> patch tokens (part losses) and the
# view-0 CLS (probe input / norm diagnostic).  This mirrors PartCo's own
# repeated forward `student[0].forward_features(images.chunk(2)[0])`; total
# backbone cost is 3B exactly as in PartCo (2B A-views + 1B repeat).
def extract_patch_and_cls(backbone, images, model_name):
    if model_name == 'v2':
        feats = backbone.forward_features(images)
        return feats['x_norm_patchtokens'], feats['x_norm_clstoken']
    raise ValueError(f'[hab] model_name {model_name!r} unsupported: the 16x16 '
                     'part-label grid matches DINOv2/vitb14 (256 tokens); v1 '
                     'would need label downsampling and is out of scope here.')

# hyperbolic
import geoopt.optim.radam as radam_
import hyptorch.nn as hypnn
from hyptorch.pmath import dist_matrix


def set_random_seed(seed: int, deterministic: bool = False, strict: bool = False) -> None:
    # [determinism] CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first cuBLAS call
    # (any GPU matmul). We set it at the very top, before torch.cuda.* touches the
    # CUDA context. The bulletproof alternative is to export it in the shell:
    #     export CUBLAS_WORKSPACE_CONFIG=:4096:8
    if deterministic or strict:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic or strict:
        # strict (warn_only=False) RAISES on the first op without a deterministic
        # implementation -> use it to LOCATE the kernel that breaks reproducibility.
        # warn_only=True lets training finish, using deterministic kernels where they exist.
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        print('[determinism] use_deterministic_algorithms(True, warn_only={}) | '
              'CUBLAS_WORKSPACE_CONFIG={}'.format(
                  not strict, os.environ.get('CUBLAS_WORKSPACE_CONFIG')))


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, hyp_c = 0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.hyp_c = hyp_c

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        if self.hyp_c == 0:
            anchor_dot_contrast = torch.div(torch.matmul(F.normalize(anchor_feature, dim=-1, p=2) , F.normalize(contrast_feature, dim=-1, p=2).T), self.temperature)
        else:
            anchor_dot_contrast = torch.div(-dist_matrix(anchor_feature, contrast_feature, c=self.hyp_c), self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda', hyp_c=0, normalize=True):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    if normalize:
        features = F.normalize(features, dim=1)

    if hyp_c == 0:
        similarity_matrix = torch.matmul(F.normalize(features, dim=-1, p=2), F.normalize(features, dim=-1, p=2).T)
    else:
        similarity_matrix = -dist_matrix(features, features, c=hyp_c)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def train(student, train_loader, test_loader, unlabelled_train_loader, args, hyperbolic_projector, hyperbolic_classifier, patch_projector=None, probe=None, ema_backbone=None, ema_classifier=None):
    params_groups = get_params_groups(student)
    # [hab] PartCo convention: the patch projector joins the same SGD group.
    if patch_projector is not None:
        params_groups += get_params_groups(patch_projector)
    # [hab] pseudo-label probe (ablation): its own optimizer, its loss never
    # enters the main objective -- it only *reads* detached part-view CLS.
    probe_optimizer = SGD(probe.parameters(), lr=args.lr, momentum=args.momentum) if probe is not None else None
    optimizer_hyper = radam_.RiemannianAdam([
        {'params': hyperbolic_projector.parameters()},
        {'params': hyperbolic_classifier.parameters()}
    ], lr=0.01, stabilize=10)

    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    cluster_criterion = DistillLoss(args.warmup_teacher_temp_epochs, args.epochs, args.n_views, args.warmup_teacher_temp, args.teacher_temp, hyp_c=0)

    # [hab] PartCo losses, verbatim classes; temperature follows HypCD's
    # system (base 0.07 x hyper_temp_scale, the E-series sup setting) unless
    # overridden by --part_temp.  All other constructor arguments mirror the
    # PartCo trainer call (PatchUnConLoss(dynamic_threshold=True), library
    # defaults elsewhere).  Known, non-causal side effect on record: with
    # temperature != base_temperature (0.07), the class's log-softmax term is
    # rescaled by T/bT while its hard-negative term is not (r2's finding; hn
    # was cleared as a cliff cause by the r2 ablation).
    patch_sup_criterion = PatchSupConLoss(temperature=args.part_temp)
    # [hab-M] --part_unsup_neg_agg min swaps the unsup objective for the APL
    # all-min form (same gate/threshold/temperature; only the aggregation
    # structure changes -- negatives repel through their min-sim part only,
    # shrinking the false-negative repulsion channel of wrong pseudo-labels).
    if getattr(args, 'part_unsup_surgery', 'none') != 'none':
        # [hab-X] oracle surgery cells (XFP / XFN): pure diagnostics at full
        # dose on the H1R trajectory prefix; GT reaches only the two masked
        # lines inside the class.
        patch_unsup_criterion = PatchUnConLossSurgery(temperature=args.part_temp, dynamic_threshold=True,
                                                      drop={'drop_fp': 'fp', 'drop_fn': 'fn'}[args.part_unsup_surgery])
    elif args.part_unsup_neg_agg == 'min':
        patch_unsup_criterion = PatchUnConLossAllMin(temperature=args.part_temp, dynamic_threshold=True)
    elif args.part_unsup_neg_agg == 'min2':
        # [hab-AMv2] controlled all-min (log-softmax kept, negatives via
        # per-image min part) -- the F27 fix; enable after D verdicts gnFN.
        patch_unsup_criterion = PatchUnConLossMinDenom(temperature=args.part_temp, dynamic_threshold=True)
    else:
        patch_unsup_criterion = PatchUnConLoss(temperature=args.part_temp, dynamic_threshold=True)

    best_test_acc_lab = 0
    best_train_acc_all = 0
    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        if patch_projector is not None:
            patch_projector.train()
        for batch_idx, batch in enumerate(train_loader):
            # [hab] PartCo datasets yield the extra part-label grid.
            images, class_labels, patch_label_1, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            patch_label_1 = patch_label_1.cuda(non_blocking=True)
            # [hab] PartCo's two-view layout, verbatim: views = [aligned,
            # contrastive]; A consumes both concatenated, exactly as PartCo
            # feeds SimGCD.
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_out = student(images)
                student_proj = hyperbolic_projector(student_out)
                image_hyp_all = student_proj  # [hab-M] snapshot before the labeled-subset reshape below (cone child)
                student_out = hyperbolic_classifier(student_proj)
                teacher_out = student_out.detach()

                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # clustering, unsup
                cluster_loss = cluster_criterion(student_out, teacher_out, epoch)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj, hyp_c=args.c, normalize=False)
                contrastive_loss_distance = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)

                # euc unsup loss
                contrastive_logits_angle, contrastive_labels_angle = info_nce_logits(features=student_proj, hyp_c=0, normalize=False, temperature=args.hyper_temp_scale * 1.0)
                contrastive_loss_angle = torch.nn.CrossEntropyLoss()(contrastive_logits_angle, contrastive_labels_angle)

                # representation learning, sup
                student_proj = torch.cat([f[mask_lab].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss_distance = SupConLoss(hyp_c=args.c)(student_proj, labels=sup_con_labels)

                sup_con_loss_angle = SupConLoss(hyp_c=0, temperature=0.07 * args.hyper_temp_scale)(student_proj, labels=sup_con_labels)

                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss

                loss_distance = (1 - args.sup_weight) * contrastive_loss_distance + args.sup_weight * sup_con_loss_distance
                loss_angle = (1 - args.sup_weight) * contrastive_loss_angle + args.sup_weight * sup_con_loss_angle

                lambda_distance = (epoch - (args.hyper_start_epoch - 1)) / ((args.hyper_end_epoch - 1) - (args.hyper_start_epoch - 1))
                lambda_distance = torch.max(torch.tensor([0, lambda_distance])).item()
                lambda_distance = torch.min(torch.tensor([1, lambda_distance])).item()
                lambda_distance = lambda_distance * args.hyper_max_weight
                # [hab] --hyper_kill_epoch E (>=0): force lambda_distance = 0
                # from epoch E on.  With E = warmup_teacher_temp_epochs this is
                # the clean-prefix D2 cell (H2P): ep0..E-1 bitwise identical to
                # H1, distance leg removed exactly when the unsup loss enters
                # (unlike H2 = --hyper_max_weight 0, whose prefix differs).
                if args.hyper_kill_epoch >= 0 and epoch >= args.hyper_kill_epoch:
                    lambda_distance = 0.0
                
                loss_rep = (1 - lambda_distance) * loss_angle + lambda_distance * loss_distance
                loss += loss_rep

                # ---------------------------------------------------------- #
                # [hab] B: PartCo part branch, verbatim form.  A above is
                # untouched; the ONLY seam between A and B is the shared
                # backbone gradient (plus the no_grad pseudo-label read).
                # ---------------------------------------------------------- #
                loss_A = loss  # [hab] A-side total, kept for --diag_grad
                part_unsup_active = False
                loss_part_unsup_w = None
                if args.part_mode != 'none':
                    # [hab] PartCo verbatim: patch tokens from a second backbone
                    # pass over view 0 (the aligned view -- crop/flip-free, so
                    # the part-label grid stays token-aligned).
                    patch_features, view0_cls = extract_patch_and_cls(student, images.chunk(2)[0], args.model_name)  # [B,256,768], [B,768]
                    patch_out = patch_projector(patch_features)  # [B, 256, 128]
                    patch_out = torch.nn.functional.normalize(patch_out, dim=-1)

                    patch_sup_con_loss = patch_sup_criterion(patch_out[mask_lab], patch_label_1[mask_lab], sup_con_labels)

                    part_unsup_active = (args.part_mode == 'full') and (epoch >= args.warmup_teacher_temp_epochs)
                    if part_unsup_active:
                        # [hab] pseudo-labels: PartCo's verbatim line -- A's
                        # classifier logits on view 0 (here HypLinear output;
                        # PatchUnConLoss wraps them in no_grad internally).
                        # --part_pseudo_source probe swaps in the detached
                        # Euclidean linear head for the source ablation.
                        if args.part_pseudo_source == 'hyp':
                            # [hab-E] EMA pseudo-label source (--part_pl_ema m > 0):
                            # a slow copy of backbone + hyperbolic classifier
                            # (projector is parameter-free and shared) provides
                            # the view-0 logits.  Motivation (F5): the injected
                            # gradient collapses the LIVE head's confidence
                            # within ~30 batches (gate 0.59 -> 0.17), poisoning
                            # its own pseudo-label source; the EMA head is
                            # outside that loop.  eval() forward, no dropout /
                            # droppath, no RNG use -- the ep0-29 trajectory is
                            # bitwise unchanged (EMA is updated but unused).
                            if args.part_pl_ema > 0:
                                with torch.no_grad():
                                    _ema_feat = ema_backbone(images.chunk(2)[0])
                                    _ema_logits = ema_classifier(hyperbolic_projector(_ema_feat)) / 0.1
                                class_logits = _ema_logits
                            # [hab-M] pseudo-label quality levers (the 2x2 showed
                            # label quality x dose dominates the steady-state
                            # cost).  --part_pl_ensemble: average the two view
                            # logits (variance reduction).  --part_pl_agree:
                            # zero the logits of view-disagreeing samples --
                            # uniform softmax => conf 1/K => rejected by the
                            # existing gate, with zero loss-code intrusion.
                            _v0, _v1 = (student_out / 0.1).chunk(2)
                            if args.part_pl_ema > 0:
                                pass  # EMA source already set above; ensemble/agree stay live-model levers
                            else:
                                class_logits = 0.5 * (_v0 + _v1) if args.part_pl_ensemble else _v0
                            if args.part_pl_agree:
                                class_logits = class_logits.clone()
                                class_logits[_v0.argmax(1) != _v1.argmax(1)] = 0.0
                        elif args.part_pseudo_source == 'oracle':
                            # [hab] ORACLE cell: ground-truth labels as pseudo-
                            # labels (diagnostic only, obviously not a method).
                            # conf -> 1, gate fully open, pseudo == true: the
                            # pseudo-label axis (quality AND distribution) is
                            # removed entirely while the injection *dose* is
                            # maximal.  Constructed without any RNG use, so the
                            # ep0-29 prefix stays bitwise identical to H1.
                            class_logits = F.one_hot(class_labels, num_classes=args.num_labeled_classes + args.num_unlabeled_classes).float() * 100.0
                        else:
                            with torch.no_grad():
                                class_logits = probe(view0_cls.detach()) / 0.1
                        if args.part_unsup_backprop == 'stopgrad':
                            # [hab] unsup trains the projector only; the sup path
                            # above keeps full backprop (log-verified harmless).
                            patch_out_u = torch.nn.functional.normalize(patch_projector(patch_features.detach()), dim=-1)
                        else:
                            patch_out_u = patch_out
                        # [hab-T] tiered dose.  Old-class pseudo-labels are
                        # near-oracle (plod 0.9-1.0 across rounds) while errors
                        # concentrate on novel classes; HO showed full dose x
                        # true labels is a large win.  --part_unsup_scale_old s
                        # (>=0) therefore runs the unsup loss at dose s on the
                        # pseudo-OLD subset and at --part_unsup_scale on the
                        # pseudo-NOVEL subset.  Implementation: two masked calls
                        # -- the other tier is presented as 'labeled' so it is
                        # excluded (the verbatim loss class is untouched).  Side
                        # effect kept on record: cross-tier negatives drop out
                        # of each denominator (which also removes old<->novel
                        # false-negative repulsion).  Tiers are disjoint, so
                        # the intra-image consistency term is not double-counted.
                        if args.part_unsup_scale_old >= 0:
                            with torch.no_grad():
                                _is_old = class_logits.argmax(dim=1) < args.num_labeled_classes
                            patch_unsup_old = patch_unsup_criterion(patch_out_u, patch_label_1, class_logits, mask_lab | ~_is_old, epoch=epoch)
                            patch_unsup_novel = patch_unsup_criterion(patch_out_u, patch_label_1, class_logits, mask_lab | _is_old, epoch=epoch)
                            patch_unsup_con_loss = None  # tiered path assembles below
                        elif args.part_unsup_surgery != 'none':
                            patch_unsup_con_loss = patch_unsup_criterion(patch_out_u, patch_label_1, class_logits, mask_lab, class_labels, epoch=epoch)
                        else:
                            patch_unsup_con_loss = patch_unsup_criterion(patch_out_u, patch_label_1, class_logits, mask_lab, epoch=epoch)

                    # [hab-M] part->image entailment cone (HyCoCLIP relu-cone,
                    # hyptorch/entailment_v4 verbatim lib).  Parts pooled from
                    # the aligned view's 768-d tokens by the TRUE part-label
                    # grid (no pseudo-labels involved), lifted by the shared
                    # (parameter-free) hyperbolic projector into A's ball; the
                    # child is the view-0 image point.  Under cr=1.2 the clip
                    # is always active, so radii are pinned and the cone acts
                    # as a soft angular-alignment band (~17 deg at defaults) --
                    # a mild, label-free part-image coherence signal.  Starts
                    # with the unsup loss (same onset epoch, prefix preserved);
                    # weight 0 disables it entirely.
                    loss_part_entail_w = None
                    if args.part_entail_weight > 0 and epoch >= args.warmup_teacher_temp_epochs:
                        _plf = patch_label_1.reshape(patch_label_1.size(0), -1)
                        _Tm = int(_plf.max().item())
                        if _Tm >= 1:
                            _oh = torch.nn.functional.one_hot(_plf.clamp_min(0), num_classes=_Tm + 1)[..., 1:].float()  # [B,256,T]
                            _cnt = _oh.sum(dim=1)                                            # [B,T]
                            _pooled = torch.einsum('bnt,bnd->btd', _oh, patch_features) / _cnt.clamp_min(1).unsqueeze(-1)
                            _pv = _cnt > 0                                                   # [B,T]
                            if _pv.any():
                                _parent_e = _pooled[_pv]                                     # [M,768] euclidean pooled parts
                                _img_idx = torch.arange(_plf.size(0), device=_plf.device).unsqueeze(1).expand_as(_pv)[_pv]
                                _parent_h = hyperbolic_projector(_parent_e)
                                _child_h = image_hyp_all.chunk(2)[0][_img_idx]
                                if args.part_entail_parent == 'image':
                                    _parent_h, _child_h = _child_h, _parent_h
                                loss_part_entail_w = args.part_entail_weight * entailment_cone_loss(
                                    _parent_h, _child_h, args.c,
                                    aperture_scale=args.part_entail_aperture, min_radius=0.1)
                                loss = loss + loss_part_entail_w

                    # [hab] PartCo assembly, verbatim weights.  --part_unsup_scale
                    # multiplies ONLY the unsup term (S-series dose-response):
                    # scale=1 is H1; ep0-29 is bitwise unaffected.  The scaled
                    # term is the same tensor the gradient diagnostics measure,
                    # so gnormR reports the ACTUAL injected magnitude.
                    # [hab] --part_unsup_ramp_epochs N (R-series): the multiplier
                    # additionally rises linearly from 1/N to 1 over the N epochs
                    # after unsup onset (epoch-level, no RNG use, ep0-29 bitwise
                    # unaffected): the 0 -> scale step is replaced by a slow
                    # approach to the SAME steady state.
                    if part_unsup_active:
                        ramp_mult = 1.0
                        if args.part_unsup_ramp_epochs > 0:
                            ramp_mult = min(1.0, (epoch - args.warmup_teacher_temp_epochs + 1) / args.part_unsup_ramp_epochs)
                        if args.part_unsup_scale_old >= 0:
                            # [hab-T] tiered: novel tier at --part_unsup_scale, old tier at --part_unsup_scale_old
                            loss_part_unsup_w = ramp_mult * 0.5 * (1 - args.sup_weight) * (
                                args.part_unsup_scale_old * patch_unsup_old + args.part_unsup_scale * patch_unsup_novel)
                        else:
                            loss_part_unsup_w = ramp_mult * args.part_unsup_scale * 0.5 * (1 - args.sup_weight) * patch_unsup_con_loss
                        loss_part_sup_w = 0.5 * args.sup_weight * patch_sup_con_loss
                        loss = loss + loss_part_unsup_w + loss_part_sup_w
                    else:
                        loss = loss + args.sup_weight * patch_sup_con_loss

                    # [hab] cheap no_grad diagnostics (printed with unsup):
                    #   part_gate_frac    : accepted fraction of unlabeled samples
                    #   part_n_acc       : accepted-sample count (dose density)
                    #   part_pospair_frac: fraction of accepted samples having at
                    #                      least one accepted peer with the SAME
                    #                      pseudo-label -- an upper-bound proxy
                    #                      for the loss being active at all
                    #                      (Pdiag@ep30 logs unsup=0.0000 on most
                    #                      batches: the Euclidean side idles).
                    #   part_plod_acc    : pseudo-label acc on accepted OLD-class samples
                    #   part_pairprec    : precision of accepted pseudo-positive pairs
                    #   part_conf_mean   : mean max-softmax confidence on unlabeled
                    # (part_cls_norm was dropped: x_norm_clstoken is post-LayerNorm,
                    #  its norm is ~constant and carries no radial-drift signal.)
                    if part_unsup_active:
                        with torch.no_grad():
                            _thr = 0.8 - 0.5 * min(1.0, epoch / patch_unsup_criterion.total_epochs)
                            _probs = class_logits.softmax(dim=1)
                            _conf, _pseudo = _probs.max(dim=1)
                            part_conf_mean = _conf[~mask_lab].mean().item()
                            _acc_mask = (~mask_lab) & (_conf > _thr)
                            part_n_acc = int(_acc_mask.sum().item())
                            part_gate_frac = part_n_acc / max(1, (~mask_lab).sum().item())
                            part_plod_acc, part_pairprec, part_pospair_frac = -1.0, -1.0, -1.0
                            if _acc_mask.any():
                                _t, _p = class_labels[_acc_mask], _pseudo[_acc_mask]
                                _cnt = torch.bincount(_p, minlength=int(_p.max().item()) + 1)
                                part_pospair_frac = (_cnt[_p] > 1).float().mean().item()
                                _old = _t < args.num_labeled_classes
                                if _old.any():
                                    part_plod_acc = (_p[_old] == _t[_old]).float().mean().item()
                                _peq = (_p.unsqueeze(0) == _p.unsqueeze(1)) & ~torch.eye(len(_p), dtype=torch.bool, device=_p.device)
                                if _peq.any():
                                    part_pairprec = ((_t.unsqueeze(0) == _t.unsqueeze(1)) & _peq).float().sum().item() / _peq.float().sum().item()

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'distance sup_con_loss: {sup_con_loss_distance.item():.4f} '
                pstr += f'distance contrastive_loss: {contrastive_loss_distance.item():.4f} '
                pstr += f'angle sup_con_loss: {sup_con_loss_angle.item():.4f} '
                pstr += f'angle contrastive_loss: {contrastive_loss_angle.item():.4f} '
                # [hab] B-side log fields (names match PartCo's trainer).
                if args.part_mode != 'none':
                    pstr += f'patch_sup_con_loss: {patch_sup_con_loss.item():.4f} '
                    if loss_part_entail_w is not None:
                        pstr += f'patch_entail: {loss_part_entail_w.item():.4f} '
                    if part_unsup_active:
                        if args.part_unsup_scale_old >= 0:
                            pstr += f'patch_unsup_old: {patch_unsup_old.item():.4f} '
                            pstr += f'patch_unsup_novel: {patch_unsup_novel.item():.4f} '
                        else:
                            pstr += f'patch_unsup_con_loss: {patch_unsup_con_loss.item():.4f} '
                        pstr += f'part_gate_frac: {part_gate_frac:.3f} '
                        pstr += f'part_n_acc: {part_n_acc} '
                        pstr += f'part_pospair_frac: {part_pospair_frac:.3f} '
                        pstr += f'part_plod_acc: {part_plod_acc:.3f} '
                        pstr += f'part_pairprec: {part_pairprec:.3f} '
                        pstr += f'part_conf_mean: {part_conf_mean:.3f} '
                        if args.part_unsup_ramp_epochs > 0:
                            pstr += f'ramp_mult: {ramp_mult:.3f} '

            # [hab] --diag_grad: before the main backward, measure -- on the
            # trainable backbone parameters -- the cosine between the A-side
            # gradient and the (weighted) unsup-part gradient, and their norm
            # ratio.  Turns "conflict vs magnitude" into two numbers per print.
            # fp16-off only (the baseline runs fp16-off); retain_graph keeps the
            # graph alive for the real backward below.
            # [hab] --diag_grad, on the trainable backbone parameters:
            #   gcos(A,unsup), gcos(psup,unsup), gcos(A+psup,unsup)  direction
            #   gnA / gnU / gnPS                                     ABSOLUTE norms
            #   gnorm(unsup/A)                                       legacy ratio
            # Absolute norms make ||g_u|| comparable ACROSS trainers (same loss,
            # same backbone architecture) -- the ratio alone cannot separate
            # "g_u is large" from "g_A is small".  Three grad calls per print,
            # fp16-off only; retain_graph keeps the graph for the real backward.
            if args.diag_grad and part_unsup_active and (loss_part_unsup_w is not None) \
                    and (batch_idx % args.print_freq == 0) and fp16_scaler is None:
                bb_params = [p for p in student.parameters() if p.requires_grad]
                _gA = torch.autograd.grad(loss_A, bb_params, retain_graph=True, allow_unused=True)
                _gU = torch.autograd.grad(loss_part_unsup_w, bb_params, retain_graph=True, allow_unused=True)
                _gPS = torch.autograd.grad(loss_part_sup_w, bb_params, retain_graph=True, allow_unused=True)
                _flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                              for g, p in zip(gs, bb_params)])
                _va, _vu, _vps = _flat(_gA), _flat(_gU), _flat(_gPS)
                _cosf = lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0).item()
                pstr += f'gcos(A,unsup): {_cosf(_va, _vu):.3f} '
                pstr += f'gcos(psup,unsup): {_cosf(_vps, _vu):.3f} '
                pstr += f'gcos(A+psup,unsup): {_cosf(_va + _vps, _vu):.3f} '
                pstr += f'gnA: {_va.norm().item():.3e} gnU: {_vu.norm().item():.3e} gnPS: {_vps.norm().item():.3e} '
                if loss_part_entail_w is not None:
                    _gE = torch.autograd.grad(loss_part_entail_w, bb_params, retain_graph=True, allow_unused=True)
                    pstr += f'gnE: {_flat(_gE).norm().item():.3e} '
                pstr += f'gnorm(unsup/A): {(_vu.norm() / (_va.norm() + 1e-12)).item():.3f} '
                # [hab-D] GT-quadrant channel decomposition (statistics + gradient
                # meters only; GT never reaches the optimizer).  Channels are an
                # exact additive split of the MAIN SupCon term (completeness
                # verified offline, rel err ~1e-8).
                if getattr(args, 'diag_decomp', False):
                    _ch, _st = decompose_unsup_channels(
                        patch_out, patch_label_1, class_logits, mask_lab, class_labels,
                        epoch=epoch, temperature=args.part_temp, base_temperature=0.07,  # [F29] trainer constructs the
                        # verbatim loss WITHOUT base_temperature -> bT=0.07; align the meters.
                        dynamic_threshold=True)
                    if _st['n_groups'] > 0:
                        _gn, _cs = {}, {}
                        for _k in ('pull_T', 'pull_F', 'push_FN', 'push_R'):
                            _g = torch.autograd.grad(_ch[_k], bb_params, retain_graph=True, allow_unused=True)
                            _v = _flat(_g)
                            _gn[_k] = _v.norm().item()
                            _cs[_k] = torch.nn.functional.cosine_similarity(_va, _v, dim=0).item() if _v.norm() > 0 else 0.0
                        pstr += (f"dcmp[nTP:{_st['n_TP']} nFP:{_st['n_FP']} nFN:{_st['n_FN']}] "
                                 f"gnPT: {_gn['pull_T']:.2e} gnPF: {_gn['pull_F']:.2e} "
                                 f"gnFN: {_gn['push_FN']:.2e} gnPR: {_gn['push_R']:.2e} "
                                 f"cos(PF,A): {_cs['pull_F']:.3f} cos(FN,A): {_cs['push_FN']:.3f} "
                                 f"tp_cos: {_st['tp_cos']:.3f} fp_cos: {_st['fp_cos']:.3f} "
                                 f"tp_kl: {_st['tp_kl']:.3f} fp_kl: {_st['fp_kl']:.3f} ")


            # [hab] probe training step (pseudo-label-source ablation): reads
            # detached CLS only -- an isolated graph, so its backward neither
            # touches nor precedes the main one in any interacting way.
            if probe is not None and args.part_mode != 'none':
                probe_logits = probe(view0_cls.detach()) / 0.1
                probe_loss = nn.CrossEntropyLoss()(probe_logits[mask_lab], class_labels[mask_lab])
                probe_optimizer.zero_grad()
                probe_loss.backward()
                probe_optimizer.step()

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            optimizer_hyper.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
                optimizer_hyper.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.step(optimizer_hyper)
                fp16_scaler.update()

            # [hab-E] EMA update (trainable backbone params + classifier; the
            # frozen blocks are identical by construction, no buffers to sync).
            if ema_backbone is not None:
                with torch.no_grad():
                    _m = args.part_pl_ema
                    for p_e, p in zip([q for q in ema_backbone.parameters()], [q for q in student.parameters()]):
                        if p.requires_grad:
                            p_e.mul_(_m).add_(p.detach(), alpha=1.0 - _m)
                    for p_e, p in zip(ema_classifier.parameters(), hyperbolic_classifier.parameters()):
                        p_e.mul_(_m).add_(p.detach(), alpha=1.0 - _m)

            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'.format(epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        # Step schedule
        exp_lr_scheduler.step()

        if epoch:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch, 'Train ACC Unlabelled', args, hyperbolic_projector, hyperbolic_classifier)
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch, 'Test ACC', args, hyperbolic_projector, hyperbolic_classifier)

            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

            # save the model
            torch.save(student.state_dict(), args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))
            torch.save(hyperbolic_projector.state_dict(), args.model_path[:-3] + f'_proj_head.pt')
            torch.save(hyperbolic_classifier.state_dict(), args.model_path[:-3] + f'_hyp_cls.pt')

            if old_acc_test > best_test_acc_lab:
                best_test_acc_lab = old_acc_test

                args.logger.info(f'Exp Name: {args.exp_name}')
                args.logger.info(f'Metrics with best model on test set: All: {all_acc:.4f} Old: {old_acc:.4f} New: {new_acc:.4f}')

                # save the model with the best acc on train data
                torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
                torch.save(hyperbolic_projector.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
                torch.save(hyperbolic_classifier.state_dict(), args.model_path[:-3] + f'_hyp_cls_best.pt')

            # ---- additionally: save the best model by ALL accuracy on the (unlabelled) train set ----
            # NOTE: `all_acc` is the primary (first) eval-func metric; with the default
            #       `--eval_funcs v2 v2b` this is the UNBALANCED (split_cluster_acc_v2) all-accuracy.
            if all_acc > best_train_acc_all:
                best_train_acc_all = all_acc

                args.logger.info(f'Exp Name: {args.exp_name}')
                args.logger.info(f'Metrics with best (train all-acc) model: All: {all_acc:.4f} Old: {old_acc:.4f} New: {new_acc:.4f}')

                # save the model with the best ALL acc on (unlabelled) train data
                torch.save(student.state_dict(), args.model_path[:-3] + f'_best_acc_all.pt')
                args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best_acc_all.pt'))
                torch.save(hyperbolic_projector.state_dict(), args.model_path[:-3] + f'_proj_head_best_acc_all.pt')
                torch.save(hyperbolic_classifier.state_dict(), args.model_path[:-3] + f'_hyp_cls_best_acc_all.pt')

        # [hab] cheap cliff-window runs: stop after this epoch, schedules untouched.
        if args.stop_after_epoch >= 0 and epoch >= args.stop_after_epoch:
            args.logger.info(f'[hab] --stop_after_epoch {args.stop_after_epoch} reached, stopping.')
            break


def test(model, test_loader, epoch, save_name, args, hyperbolic_projector, hyperbolic_classifier):
    model.eval()
    hyperbolic_projector.eval()
    hyperbolic_classifier.eval()

    preds, targets = [], []
    feats_bank = []
    mask = np.array([])
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        # [hab] partco datasets return (img, label, patch_label, uq_idx);
        # only image and label are needed at eval time.
        images, label = batch[0], batch[1]
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            ec_feat = model(images)

            hyp_feat = hyperbolic_projector(ec_feat)
            logits = hyperbolic_classifier(hyp_feat)

            preds.append(logits.argmax(1).cpu().numpy())
            if getattr(args, 'eval_kmeans', False):
                feats_bank.append(ec_feat.cpu())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)

    # [hab] --eval_kmeans: representation-level probe (H1K cell).  K-means on
    # L2-normalized EUCLIDEAN backbone features -- the GCD-native protocol,
    # bypassing the hyperbolic head entirely.  Purely eval-side: sklearn uses
    # its own seeded RNG, so the training trajectory stays bitwise identical
    # to the same cell without this flag.  Read head-acc vs KMeans-acc through
    # the cliff: head collapses while KMeans holds => backbone-head mismatch;
    # both collapse => the representation itself is disrupted.
    if getattr(args, 'eval_kmeans', False) and 'Train' in save_name:
        from sklearn.cluster import KMeans
        km_feats = torch.nn.functional.normalize(torch.cat(feats_bank, dim=0), dim=-1).numpy()
        km_preds = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes,
                          random_state=0, n_init=3).fit_predict(km_feats)
        log_accs_from_preds(y_true=targets, y_pred=km_preds, mask=mask, T=epoch,
                            eval_funcs=args.eval_funcs, save_name='KMeans ' + save_name, args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2b'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)

    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--seed', default=0, type=int)
    # [determinism] opt-in; default False -> byte-identical to the original behavior.
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enable deterministic algorithms (warn_only). Slower, but reproducible run-to-run on identical HW + library versions.')
    parser.add_argument('--strict_deterministic', action='store_true', default=False,
                        help='Like --deterministic but RAISES on the first non-deterministic op; use to find which kernel breaks reproducibility.')

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='simgcd', type=str)

    # hyperbolic
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_model_path', default=None, type=str)
    parser.add_argument('--model_name', default='vit_dino', type=str)
    parser.add_argument('--hyper_start_epoch', default=0, type=int)
    parser.add_argument('--hyper_end_epoch', default=200, type=int)
    parser.add_argument('--c', default=0.05, type=float)
    parser.add_argument('--cr', type=float, default=0)
    parser.add_argument('--riemannian', type=bool, default=False)
    parser.add_argument('--hyper_max_weight', type=float, default=1.0)
    parser.add_argument('--hyper_temp_scale', type=float, default=1.0)

    # [hab] ablation switches (see module docstring / README)
    parser.add_argument('--part_mode', type=str, default='full', choices=['none', 'sup', 'full'],
                        help="none = pure-A regression control; sup = H0 (sup part loss only); full = H1 (PartCo verbatim schedule).")
    parser.add_argument('--part_unsup_backprop', type=str, default='full', choices=['full', 'stopgrad'],
                        help="stopgrad: the unsup part loss trains the patch projector only (sup path keeps full backprop).")
    parser.add_argument('--part_pseudo_source', type=str, default='hyp', choices=['hyp', 'probe', 'oracle'],
                        help="hyp: A's classifier logits on view 0, PartCo's verbatim line. probe: detached Euclidean linear head trained by CE on labeled samples only.")
    parser.add_argument('--part_temp', type=float, default=None,
                        help='Part-loss temperature. Default None -> 0.07 * hyper_temp_scale (HypCD temperature system, = E-series sup). Pass 0.07 for PartCo native.')
    parser.add_argument('--hyper_kill_epoch', type=int, default=-1,
                        help='Force lambda_distance = 0 from this epoch on (-1 = off). Set to warmup_teacher_temp_epochs for the clean-prefix D2 cell (H2P).')
    parser.add_argument('--stop_after_epoch', type=int, default=-1,
                        help='Break the training loop after this epoch completes, KEEPING the --epochs-based schedules (lr cosine, lambda_d ramp, teacher temp). Cheap cliff-window runs that stay bitwise on the 200ep trajectory.')
    # [hab-M] method-stage switches
    parser.add_argument('--part_pl_ensemble', action='store_true', default=False,
                        help='Pseudo-labels from the MEAN of the two view logits (variance reduction) instead of view 0 alone.')
    parser.add_argument('--part_pl_agree', action='store_true', default=False,
                        help='Reject samples whose two views disagree on argmax (their logits are zeroed => uniform conf => gated out).')
    parser.add_argument('--part_unsup_neg_agg', type=str, default='full', choices=['full', 'min', 'min2'],
                        help="full = PartCo verbatim PatchUnConLoss; min = APL all-min form (negative images repel only through their least-similar part).")
    parser.add_argument('--part_entail_weight', type=float, default=0.0,
                        help='Weight of the label-free part->image entailment-cone loss (0 = off). Starts at unsup onset; prefix preserved.')
    parser.add_argument('--part_entail_parent', type=str, default='part', choices=['part', 'image'],
                        help="Cone apex. 'part' = HyCoCLIP box-as-parent convention (default).")
    parser.add_argument('--part_entail_aperture', type=float, default=1.2,
                        help='Aperture scale of the cone (HyCoCLIP intra-modal default).')
    parser.add_argument('--part_unsup_surgery', type=str, default='none', choices=['none', 'drop_fp', 'drop_fn'],
                        help='[X] Oracle surgery diagnostics: drop_fp removes false-positive ATTRACTION, drop_fn removes false-negative REPULSION (GT enters only these masks). Run at full dose, stop ~35.')
    parser.add_argument('--part_pl_ema', type=float, default=0.0,
                        help='[E] EMA momentum for the pseudo-label source (backbone + hyp classifier copies; 0 = off, typical 0.999). Decouples the source from the injection-collapse loop (F5). ep0-29 bitwise unchanged.')
    parser.add_argument('--part_unsup_scale', type=float, default=1.0, help='Multiplier applied only to the unsupervised part loss.')
    parser.add_argument('--part_unsup_scale_old', type=float, default=-1.0,
                        help='[T] Tiered dose: unsup scale for pseudo-OLD samples (near-oracle labels, plod 0.9-1.0). -1 = off (single scale). Pseudo-NOVEL samples keep --part_unsup_scale.')
    parser.add_argument('--part_unsup_ramp_epochs', type=int, default=0,
                        help='R-series: linear ramp of the unsup multiplier from 1/N to 1 over N epochs after unsup onset (0 = original step). Same steady state as the step; only the approach differs.')
    parser.add_argument('--eval_kmeans', action='store_true', default=False,
                        help='H1K cell: per-eval K-means on L2-normalized Euclidean backbone features (unlabelled-train split), logged as "KMeans Train ACC ...". Eval-side only; training bitwise unaffected.')
    parser.add_argument('--diag_decomp', action='store_true', default=False,
                        help='GT-quadrant decomposition of the unsup part gradient (pull_T/pull_F/push_FN/push_R) on trainable backbone params at each print. GT is used for statistics and gradient meters ONLY. Requires --diag_grad, fp16 off.')
    parser.add_argument('--diag_grad', action='store_true', default=False,
                        help='Log cos(g_A, g_unsup) and ||g_unsup||/||g_A|| on trainable backbone params at each print (fp16-off only).')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    # [hab] part-loss temperature follows HypCD's system unless overridden.
    if args.part_unsup_surgery != 'none':
        assert args.part_unsup_scale_old < 0 and args.part_unsup_neg_agg == 'full', \
            '[hab-X] surgery cells must run on the plain verbatim path'
    if args.part_temp is None:
        args.part_temp = 0.07 * args.hyper_temp_scale
    print(args)
    set_random_seed(args.seed, deterministic=args.deterministic, strict=args.strict_deterministic)
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'HypSimGCD_{args.dataset_name}'])
    args.logger.info(f'Using evaluation function {args.eval_funcs} to print results')
    # Add a handler for stdout and configure it to log to stdout as well
    args.logger.add(sys.stdout)

    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875

    if args.model_name == 'v1':
        backbone = vits1.__dict__['vit_base']()
        state_dict = torch.load(dino_pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
    elif args.model_name == 'v2':
        backbone = vits2.__dict__['vit_base']()
        state_dict = torch.load(dinov2_pretrain_path, map_location='cpu')
        backbone.load_state_dict(state_dict)
    else:
        raise ValueError('Invalid model name')

    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    # hyperbolic dimension
    args.mlp_out_dim = 256

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    # [hab] PartCo's transform triple and two-view generator, verbatim:
    # view 0 = aligned (Resize+ColorJitter, crop/flip-free), view 1 = the
    # contrastive transform (byte-identical to HypSimGCD's train transform).
    train_transform, contrastive_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=[train_transform, contrastive_transform],
        n_views=args.n_views
    )
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)
    # [hab] eval determinism (det_ab contract): the partco datasets carry a
    # sample-level sync flip; disable it on the evaluation copies.
    unlabelled_train_examples_test.random_hflip = False
    test_dataset.random_hflip = False

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    # [hab-fix] The sampler previously drew from the GLOBAL CPU RNG, so cells
    # that construct a different module set (A0: no projector; H3: extra
    # probe) consumed the RNG differently and got DIFFERENT batch sequences
    # (verified in the 200e logs: H3 ep29 74.42 vs H1 74.82 despite identical
    # ep0-29 training semantics).  Under --deterministic the sampler now gets
    # its own seeded generator, making the batch sequence invariant to module
    # construction: {CB, E3R, H1, H3, S*} share a bitwise-identical ep0-29
    # prefix.  Non-deterministic runs keep the original behavior.
    sampler_generator = None
    if args.deterministic or args.strict_deterministic:
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), generator=sampler_generator)

    # --------------------
    # DATALOADERS
    # --------------------
    # [determinism] reproducible data loading; generator/worker_init_fn = None is a no-op,
    # so default (non-deterministic) behavior is unchanged unless a switch is passed.
    loader_generator = None
    worker_init_fn = None
    if args.deterministic or args.strict_deterministic:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % (2 ** 32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True, pin_memory=True, generator=loader_generator, worker_init_fn=worker_init_fn)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    args.c = args.c
    if args.cr != 0:
        hyperbolic_projector = hypnn.ToPoincare(c=args.c, ball_dim=args.mlp_out_dim, riemannian=args.riemannian, clip_r=args.cr).to(device)
    else:
        hyperbolic_projector = hypnn.ToPoincare(c=args.c, ball_dim=args.mlp_out_dim, riemannian=args.riemannian).to(device)
    hyperbolic_classifier = hypnn.HypLinear(in_features=args.feat_dim, out_features=args.num_labeled_classes + args.num_unlabeled_classes, c=args.c).to(device)

    # [hab] B: PartCo's patch projection head, verbatim construction.
    patch_projector = None
    if args.part_mode != 'none':
        patch_projector = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.1),
            Patch_Projection(),
        ).to(device)
    # [hab] detached Euclidean pseudo-label probe (only for the source ablation).
    probe = None
    if args.part_mode == 'full' and args.part_pseudo_source == 'probe':
        probe = nn.Linear(args.feat_dim, args.num_labeled_classes + args.num_unlabeled_classes).to(device)

    model = backbone.to(device)
    # [hab-E] EMA pseudo-label source (deepcopy uses no RNG; eval + no-grad).
    # Must come AFTER `model = backbone.to(device)` (r6 fix: it previously
    # referenced `model` before that assignment -> NameError).
    ema_backbone = ema_classifier = None
    if args.part_mode == 'full' and args.part_pl_ema > 0:
        ema_backbone = copy.deepcopy(model).eval()
        ema_classifier = copy.deepcopy(hyperbolic_classifier).eval()
        for _p in list(ema_backbone.parameters()) + list(ema_classifier.parameters()):
            _p.requires_grad_(False)
    # ----------------------
    # TRAIN
    # ----------------------
    if args.eval_only:
        if args.eval_model_path is not None:
            print(f'Loading evaluation model weights from {args.eval_model_path}')
            model.load_state_dict(torch.load(args.eval_model_path, map_location='cpu'))
            hyperbolic_projector.load_state_dict(torch.load(args.eval_model_path.replace('model_', 'model_proj_head_'), map_location='cpu'))
            hyperbolic_classifier.load_state_dict(torch.load(args.eval_model_path.replace('model_', 'model_hyp_cls_'), map_location='cpu'))
            test(model, test_loader_unlabelled, 0, 'Train ACC Unlabelled', args, hyperbolic_projector, hyperbolic_classifier)
    else:
        train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args, hyperbolic_projector, hyperbolic_classifier, patch_projector=patch_projector, probe=probe, ema_backbone=ema_backbone, ema_classifier=ema_classifier)