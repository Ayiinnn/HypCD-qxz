"""HyPartCo = Hyp-SimGCD (org, det, ab) + PartCo part-level branch.

Strict A+B composition of two stably-running components:

  A  train_HypCD_org_det_ab.py    -- Hyp-SimGCD baseline: hyperbolic
     classifier + hybrid angle/distance representation losses with the
     linear a_d ramp (HypCD paper Sec. 3.4), deterministic switches, and
     best/best-all checkpointing.  All A code below is verbatim.

  B  partco repo train_partco_simgcd.py -- PartCo part branch: patch tokens
     of the aligned view-0 are projected (LayerNorm-Dropout-Patch_Projection),
     average-pooled per part id, and trained with the part-level
     correspondence losses (PartCo paper Eq. 6-7); supervised-only during the
     teacher-temperature warmup, then 0.5*((1-sw)*unsup + sw*sup).  The data
     layer (aligned view-0 transform + patch-label datasets) is B's and is
     ported verbatim under data/partco/.

  A applied to B (the "SimGCD -> HypSimGCD" mapping, per part loss):
     part loss L  ->  (1 - a_d) * L(cosine, 0.07 * hyper_temp_scale)
                      +     a_d * L(-D_H on ToPoincare(pooled), 0.07)
     with the SAME a_d ramp, curvature c, clip_r and ToPoincare settings as
     the image branch.  At a_d = 0 the part branch is numerically the stable
     PartCo loss (cosine is invariant under expmap0, HypCD Eq. 10).

  No new hyperparameters are introduced.

Rationale (papers): HypCD motivates hyperbolic space precisely by the
part-whole hierarchy of objects ("object parts ... reside within a
hierarchical structure") but never supervises parts; PartCo supervises parts
explicitly but in spherical/cosine geometry.  Each is the missing half of
the other, and PartCo is designed as a plug-in on top of parametric GCD
losses -- which is exactly how it is attached here.

Orthogonality: no changes to any mainline file; the object branch (obj_multi
v2) and the radial gate (v4) remain independently attachable -- the part
branch exposes its own ToPoincare operator as the natural gating hook.
"""
import argparse

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

# [hypartco] B's data layer: PartCo transforms (aligned view-0) + part-label datasets
from data.partco.augmentations import get_transform
from data.partco.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path, dinov2_pretrain_path
from models.model import DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from models import vision_transformer as vits1
from models import vision_transformer2 as vits2

# [hypartco] B's part branch, hyperbolized with A's recipe
from models.hyp_partco import (
    build_patch_projector,
    extract_patch_tokens,
    align_patch_labels,
    HypPatchSupConLoss,
    HypPatchUnConLoss,
)

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


def train(student, patch_projector, train_loader, test_loader, unlabelled_train_loader, args,
          hyperbolic_projector, hyperbolic_classifier, hyperbolic_part_projector):
    params_groups = get_params_groups(student)
    # [hypartco] the patch projector is Euclidean and joins the backbone's SGD,
    # exactly as in the partco repo trainer.
    params_groups += get_params_groups(patch_projector)
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

    # [hypartco] B's part losses under A's angle/distance dualization.
    #   angle    (hyp_c=0)      : byte-identical to the stable partco losses,
    #                             with A's hyper_temp_scale on the temperature
    #                             (the same scaling A applies to its own angle
    #                             losses: 0.07 -> 0.07 * hts).
    #   distance (hyp_c=args.c) : same losses with S = -D_H after ToPoincare.
    # Constructor arguments beyond these mirror the partco trainer call
    # PatchUnConLoss(dynamic_threshold=True) -- i.e. library defaults.
    part_sup_angle = HypPatchSupConLoss(temperature=0.07 * args.hyper_temp_scale, hyp_c=0)
    part_sup_dist = HypPatchSupConLoss(temperature=0.07, hyp_c=args.c, to_hyp=hyperbolic_part_projector)
    part_unsup_angle = HypPatchUnConLoss(temperature=0.07 * args.hyper_temp_scale, hyp_c=0,
                                         dynamic_threshold=True)
    # [hypartco-r3] The unsup part loss is no longer dualized (see the loss
    # assembly below); this distance instance is therefore not constructed.
    # part_unsup_dist = HypPatchUnConLoss(temperature=0.07, hyp_c=args.c, to_hyp=hyperbolic_part_projector,
    #                                     dynamic_threshold=True)

    best_test_acc_lab = 0
    best_train_acc_all = 0
    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        patch_projector.train()
        for batch_idx, batch in enumerate(train_loader):
            # [hypartco] partco datasets yield the extra patch label
            images, class_labels, patch_label_1, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            patch_label_1 = patch_label_1.cuda(non_blocking=True)
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_out = student(images)
                student_proj = hyperbolic_projector(student_out)
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

                # ------------------------------------------------------------------ #
                # [hypartco] part representation learning (B, partco trainer verbatim)
                # Patch tokens come from view 0 only: its transform is crop/flip-free,
                # so the part-label grid stays spatially aligned with the tokens.
                # ------------------------------------------------------------------ #
                patch_features = extract_patch_tokens(student, images.chunk(2)[0], args.model_name)  # [B, N, 768]
                patch_out = patch_projector(patch_features)  # [B, N, 128]
                patch_out = torch.nn.functional.normalize(patch_out, dim=-1)
                patch_label_grid = align_patch_labels(patch_label_1, patch_out.shape[1])  # no-op for v2

                if epoch < args.warmup_teacher_temp_epochs:
                    patch_sup_con_loss_angle = part_sup_angle(patch_out[mask_lab], patch_label_grid[mask_lab], sup_con_labels)
                    patch_sup_con_loss_dist = part_sup_dist(patch_out[mask_lab], patch_label_grid[mask_lab], sup_con_labels)
                else:
                    patch_sup_con_loss_angle = part_sup_angle(patch_out[mask_lab], patch_label_grid[mask_lab], sup_con_labels)
                    patch_sup_con_loss_dist = part_sup_dist(patch_out[mask_lab], patch_label_grid[mask_lab], sup_con_labels)
                    class_logits = (student_out / 0.1).chunk(2)[0]
                    # [hypartco-r4] Relative-age gating: the criterion's dynamic
                    # threshold (0.8 - 0.5*min(1, t/total_epochs), total_epochs=100)
                    # is evaluated at the unsup loss's own age t = epoch - 30 instead
                    # of the global epoch.  Passing the global epoch made the loss
                    # skip the strict 0.8 -> 0.65 phase and be born at tau = 0.65 --
                    # under A's sharper HypLinear head that admitted (nearly) the full
                    # unlabeled population at once (r3: every logged batch active from
                    # ep30; residual -15..-20pt cliff with distance-unsup already
                    # removed), whereas PartCo's gate produced an ~8-epoch sparse ->
                    # dense onset (2/5 -> 5/5 active batches) costing only -4.7pt.
                    # Relative age restores a strict-onset schedule that pointwise
                    # dominates PartCo's own (tau_rel(ep60) = 0.65 = PartCo's ep30
                    # value) using only existing constants -- zero new hyperparameters.
                    # `epoch` is consumed solely by the threshold inside the criterion.
                    part_unsup_age = epoch - args.warmup_teacher_temp_epochs
                    patch_unsup_con_loss_angle = part_unsup_angle(patch_out, patch_label_grid, class_logits, mask_lab, epoch=part_unsup_age)
                    # [hypartco-r3] no distance-unsup forward (see assembly below)
                    # [hypartco-r4] Diagnostics (logging only, no effect on training):
                    #   part_gate_frac : fraction of unlabeled samples above the
                    #                    (relative-age) threshold.  ~1.0 from the start
                    #                    means the head's confidence is saturated and
                    #                    gating is transparent -> see CHANGES_r4 for
                    #                    the pre-authorized short-ramp fallback.
                    #   part_plod_acc  : pseudo-label accuracy among accepted unlabeled
                    #                    samples whose true class is OLD (old prototypes
                    #                    are label-aligned by the sup CE, so raw
                    #                    agreement is meaningful; new ones are not,
                    #                    absent Hungarian matching).
                    #   part_pairprec  : precision of accepted pseudo-positive pairs,
                    #                    P(true_i == true_j | pseudo_i == pseudo_j) --
                    #                    Hungarian-free, meaningful for old AND new.
                    with torch.no_grad():
                        _thr = 0.8 - 0.5 * min(1.0, max(0, part_unsup_age) / 100.0)
                        _probs = class_logits.softmax(dim=1)
                        _conf, _pseudo = _probs.max(dim=1)
                        _acc_mask = (~mask_lab) & (_conf > _thr)
                        part_gate_frac = _acc_mask.float().sum().item() / max(1, (~mask_lab).sum().item())
                        part_plod_acc, part_pairprec = -1.0, -1.0
                        if _acc_mask.any():
                            _t, _p = class_labels[_acc_mask], _pseudo[_acc_mask]
                            _old = _t < args.num_labeled_classes
                            if _old.any():
                                part_plod_acc = (_p[_old] == _t[_old]).float().mean().item()
                            _peq = (_p.unsqueeze(0) == _p.unsqueeze(1)) & ~torch.eye(len(_p), dtype=torch.bool, device=_p.device)
                            if _peq.any():
                                part_pairprec = ((_t.unsqueeze(0) == _t.unsqueeze(1)) & _peq).float().sum().item() / _peq.float().sum().item()

                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss

                loss_distance = (1 - args.sup_weight) * contrastive_loss_distance + args.sup_weight * sup_con_loss_distance
                loss_angle = (1 - args.sup_weight) * contrastive_loss_angle + args.sup_weight * sup_con_loss_angle

                lambda_distance = (epoch - (args.hyper_start_epoch - 1)) / ((args.hyper_end_epoch - 1) - (args.hyper_start_epoch - 1))
                lambda_distance = torch.max(torch.tensor([0, lambda_distance])).item()
                lambda_distance = torch.min(torch.tensor([1, lambda_distance])).item()
                lambda_distance = lambda_distance * args.hyper_max_weight

                loss_rep = (1 - lambda_distance) * loss_angle + lambda_distance * loss_distance
                loss += loss_rep

                # [hypartco] part loss: B's warmup / 0.5-weighting verbatim, then
                # A's angle/distance blend with the SAME lambda_distance ramp.
                if epoch < args.warmup_teacher_temp_epochs:
                    part_loss_angle = args.sup_weight * patch_sup_con_loss_angle
                    part_loss_dist = args.sup_weight * patch_sup_con_loss_dist
                else:
                    # [hypartco-r3] The UNSUP part loss is NOT dualized: it enters both
                    # blend arms in its PartCo form (angle @ 0.07*hts), so its total
                    # weight is PartCo's constant 0.5*(1-sup_weight) and lambda_distance
                    # only dualizes the SUP part loss:
                    #   loss_part = 0.5(1-sw)*unsup_a + 0.5*sw*[(1-ld)*sup_a + ld*sup_d]
                    # Rationale (r2 logs): with the r2 hard-negative fix verified in the
                    # values (angle unsup settles in PartCo's 8.5-9.6 band, distance
                    # unsup at the bounded -4..-5.4 hn+CE level, no runaway), the ep30
                    # cliff persisted and its depth tracked lambda_distance almost
                    # linearly (-36.4pt at ld=0.155 vs -18.8pt at ld=0.078, while the
                    # angle-unsup coefficient *rose* 9%) -- convicting the
                    # distance-unsup composite (pseudo-label SupCon + consistency under
                    # -D_H).  That is also the one cell with no precedent in either
                    # paper: HypCD derives its substitution for true-label SupCon and
                    # augmentation-pair InfoNCE; PartCo's pseudo-label part loss exists
                    # only in cosine.  Sup part (true labels, the HypCD-covered form,
                    # log-verified healthy for 30 epochs in both geometries) stays dual.
                    part_loss_angle = 0.5 * ((1 - args.sup_weight) * patch_unsup_con_loss_angle + args.sup_weight * patch_sup_con_loss_angle)
                    part_loss_dist = 0.5 * ((1 - args.sup_weight) * patch_unsup_con_loss_angle + args.sup_weight * patch_sup_con_loss_dist)
                loss_part = (1 - lambda_distance) * part_loss_angle + lambda_distance * part_loss_dist
                loss += loss_part

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'distance sup_con_loss: {sup_con_loss_distance.item():.4f} '
                pstr += f'distance contrastive_loss: {contrastive_loss_distance.item():.4f} '
                pstr += f'angle sup_con_loss: {sup_con_loss_angle.item():.4f} '
                pstr += f'angle contrastive_loss: {contrastive_loss_angle.item():.4f} '
                pstr += f'angle patch_sup_con_loss: {patch_sup_con_loss_angle.item():.4f} '
                pstr += f'distance patch_sup_con_loss: {patch_sup_con_loss_dist.item():.4f} '
                if epoch >= args.warmup_teacher_temp_epochs:
                    pstr += f'angle patch_unsup_con_loss: {patch_unsup_con_loss_angle.item():.4f} '
                    # [hypartco-r3] no distance patch_unsup line (loss not computed)
                    pstr += f'part_gate_frac: {part_gate_frac:.3f} '
                    pstr += f'part_plod_acc: {part_plod_acc:.3f} '
                    pstr += f'part_pairprec: {part_pairprec:.3f} '

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
            torch.save(patch_projector.state_dict(), args.model_path[:-3] + f'_patch_proj.pt')

            if old_acc_test > best_test_acc_lab:
                best_test_acc_lab = old_acc_test

                args.logger.info(f'Exp Name: {args.exp_name}')
                args.logger.info(f'Metrics with best model on test set: All: {all_acc:.4f} Old: {old_acc:.4f} New: {new_acc:.4f}')

                # save the model with the best acc on train data
                torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
                torch.save(hyperbolic_projector.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')
                torch.save(hyperbolic_classifier.state_dict(), args.model_path[:-3] + f'_hyp_cls_best.pt')
                torch.save(patch_projector.state_dict(), args.model_path[:-3] + f'_patch_proj_best.pt')

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
                torch.save(patch_projector.state_dict(), args.model_path[:-3] + f'_patch_proj_best_acc_all.pt')


def test(model, test_loader, epoch, save_name, args, hyperbolic_projector, hyperbolic_classifier):
    model.eval()
    hyperbolic_projector.eval()
    hyperbolic_classifier.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        # [hypartco] partco datasets return (img, label, patch_label, uq_idx);
        # only image and label are needed at eval time.
        images, label = batch[0], batch[1]
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            ec_feat = model(images)

            hyp_feat = hyperbolic_projector(ec_feat)
            logits = hyperbolic_classifier(hyp_feat)

            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, args=args)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2b'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cub, scars, aircraft, pets (datasets with ported PartCo labels)')
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
    parser.add_argument('--exp_name', default='hypartco', type=str)

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

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    print(args)
    set_random_seed(args.seed, deterministic=args.deterministic, strict=args.strict_deterministic)
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'HyPartCo_{args.dataset_name}'])
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
    # [hypartco] PartCo's transform triple: view 0 = Resize+ColorJitter only
    # (no crop/flip -> patch labels stay aligned), view 1 = the standard
    # contrastive transform.  Verbatim B behavior.
    train_transform, contrastive_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(
        base_transform=[train_transform, contrastive_transform],
        n_views=args.n_views
    )
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # [hypartco] the partco datasets random-flip img+patch-label inside
    # __getitem__ (training-time part augmentation).  Disable it ONLY for the
    # two evaluation datasets: eval must stay deterministic for the det/ab
    # best-model selection.  Training datasets keep the original behavior.
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
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

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

    # ----------------------
    # [hypartco] PART BRANCH
    # ----------------------
    # Euclidean patch projector (B, verbatim) + a ToPoincare operator with the
    # SAME curvature / riemannian / clipping settings as the image branch (A):
    # the part branch lives on the same ball.  ToPoincare is parameter-free
    # here, so this adds no learnable state; it is a separate instance purely
    # as the future per-branch (radial-gate) hook.
    patch_projector = build_patch_projector().to(device)
    if args.cr != 0:
        hyperbolic_part_projector = hypnn.ToPoincare(c=args.c, ball_dim=128, riemannian=args.riemannian, clip_r=args.cr).to(device)
    else:
        hyperbolic_part_projector = hypnn.ToPoincare(c=args.c, ball_dim=128, riemannian=args.riemannian).to(device)

    model = backbone.to(device)
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
        train(model, patch_projector, train_loader, test_loader_labelled, test_loader_unlabelled, args,
              hyperbolic_projector, hyperbolic_classifier, hyperbolic_part_projector)
