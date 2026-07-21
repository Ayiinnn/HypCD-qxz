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
  --part_pseudo_source {hyp,probe}     probe = detached Euclidean linear head
  --hyper_max_weight 0                 (existing A knob) = distance leg off
  --part_temp                          part-loss temperature override
  --diag_grad                          per-print backbone-gradient diagnostics
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


def train(student, train_loader, test_loader, unlabelled_train_loader, args, hyperbolic_projector, hyperbolic_classifier, patch_projector=None, probe=None):
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
                            class_logits = (student_out / 0.1).chunk(2)[0]
                        else:
                            with torch.no_grad():
                                class_logits = probe(view0_cls.detach()) / 0.1
                        if args.part_unsup_backprop == 'stopgrad':
                            # [hab] unsup trains the projector only; the sup path
                            # above keeps full backprop (log-verified harmless).
                            patch_out_u = torch.nn.functional.normalize(patch_projector(patch_features.detach()), dim=-1)
                        else:
                            patch_out_u = patch_out
                        patch_unsup_con_loss = patch_unsup_criterion(patch_out_u, patch_label_1, class_logits, mask_lab, epoch=epoch)

                    # [hab] PartCo assembly, verbatim weights.
                    if part_unsup_active:
                        loss_part_unsup_w = 0.5 * (1 - args.sup_weight) * patch_unsup_con_loss
                        loss = loss + loss_part_unsup_w + 0.5 * args.sup_weight * patch_sup_con_loss
                    else:
                        loss = loss + args.sup_weight * patch_sup_con_loss

                    # [hab] cheap no_grad diagnostics (printed with unsup):
                    #   part_gate_frac : accepted fraction of unlabeled samples
                    #   part_plod_acc  : pseudo-label acc on accepted OLD-class samples
                    #   part_pairprec  : precision of accepted pseudo-positive pairs
                    #   part_conf_mean : mean max-softmax confidence on unlabeled
                    #   part_cls_norm  : mean view-0 CLS feature norm (the
                    #                    classifier-input Euclidean feature)
                    # The last two trace the head-response path (confidence /
                    # feature-norm drift) batch-by-batch through the cliff.
                    if part_unsup_active:
                        with torch.no_grad():
                            _thr = 0.8 - 0.5 * min(1.0, epoch / patch_unsup_criterion.total_epochs)
                            _probs = class_logits.softmax(dim=1)
                            _conf, _pseudo = _probs.max(dim=1)
                            part_conf_mean = _conf[~mask_lab].mean().item()
                            part_cls_norm = view0_cls[~mask_lab].norm(dim=-1).mean().item()
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
                    if part_unsup_active:
                        pstr += f'patch_unsup_con_loss: {patch_unsup_con_loss.item():.4f} '
                        pstr += f'part_gate_frac: {part_gate_frac:.3f} '
                        pstr += f'part_plod_acc: {part_plod_acc:.3f} '
                        pstr += f'part_pairprec: {part_pairprec:.3f} '
                        pstr += f'part_conf_mean: {part_conf_mean:.3f} '
                        pstr += f'part_cls_norm: {part_cls_norm:.2f} '

            # [hab] --diag_grad: before the main backward, measure -- on the
            # trainable backbone parameters -- the cosine between the A-side
            # gradient and the (weighted) unsup-part gradient, and their norm
            # ratio.  Turns "conflict vs magnitude" into two numbers per print.
            # fp16-off only (the baseline runs fp16-off); retain_graph keeps the
            # graph alive for the real backward below.
            if args.diag_grad and part_unsup_active and (loss_part_unsup_w is not None) \
                    and (batch_idx % args.print_freq == 0) and fp16_scaler is None:
                bb_params = [p for p in student.parameters() if p.requires_grad]
                _gA = torch.autograd.grad(loss_A, bb_params, retain_graph=True, allow_unused=True)
                _gU = torch.autograd.grad(loss_part_unsup_w, bb_params, retain_graph=True, allow_unused=True)
                _flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                              for g, p in zip(gs, bb_params)])
                _va, _vu = _flat(_gA), _flat(_gU)
                _cos = torch.nn.functional.cosine_similarity(_va, _vu, dim=0).item()
                _ratio = (_vu.norm() / (_va.norm() + 1e-12)).item()
                pstr += f'gcos(A,unsup): {_cos:.3f} gnorm(unsup/A): {_ratio:.3f} '

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


def test(model, test_loader, epoch, save_name, args, hyperbolic_projector, hyperbolic_classifier):
    model.eval()
    hyperbolic_projector.eval()
    hyperbolic_classifier.eval()

    preds, targets = [], []
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
    parser.add_argument('--part_pseudo_source', type=str, default='hyp', choices=['hyp', 'probe'],
                        help="hyp: A's classifier logits on view 0, PartCo's verbatim line. probe: detached Euclidean linear head trained by CE on labeled samples only.")
    parser.add_argument('--part_temp', type=float, default=None,
                        help='Part-loss temperature. Default None -> 0.07 * hyper_temp_scale (HypCD temperature system, = E-series sup). Pass 0.07 for PartCo native.')
    parser.add_argument('--diag_grad', action='store_true', default=False,
                        help='Log cos(g_A, g_unsup) and ||g_unsup||/||g_A|| on trainable backbone params at each print (fp16-off only).')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    # [hab] part-loss temperature follows HypCD's system unless overridden.
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
        train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args, hyperbolic_projector, hyperbolic_classifier, patch_projector=patch_projector, probe=probe)