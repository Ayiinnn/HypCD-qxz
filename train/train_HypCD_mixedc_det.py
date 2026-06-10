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

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path, dinov2_pretrain_path
from models.model import DistillLoss, ContrastiveLearningViewGenerator, get_params_groups
from models import vision_transformer as vits1
from models import vision_transformer2 as vits2

# hyperbolic / product-space
import geoopt.optim.radam as radam_
import hyptorch.nn_mc_mixedc as hypnn


# ============================================================================
#  HypSimGCD with mixed-curvature PRODUCT-SPACE factorization.
#
#  The single shared embedding is split into constant-curvature factors
#  (default: 1 Euclidean + 1 Poincare). Losses are routed to factors by *role*:
#     cls / me_max  -> role "cls" (flat-leaning)
#     contrastive / supcon (rep) -> role "rep" (curved-leaning)
#     alignment (anchor) -> role "align" (uniform, gate-free, full metric)
#  sup and unsup SHARE one embedding and one set of role weights -> no separate
#  curvature per branch -> the supervised-branch collapse is impossible.
#
#  Add factors by extending --factors (e.g. "E:256,P:384,P:128"); a part-level
#  factor + a "part" role can be wired at the HOOK marked below.
# ============================================================================


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------
#  Product-aware contrastive losses.
#  `features` is the on-manifold product embedding (concatenated coords).
#  mode="angle"   -> cosine in the flat factor (Euclidean end of curriculum)
#  mode="distance"-> negative product geodesic distance under `role` weights
# ----------------------------------------------------------------------------
def product_info_nce_logits(features, manifold, role, mode,
                            n_views=2, temperature=1.0, device='cuda'):
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    if mode == "angle":
        sim = manifold.angle_sim_matrix(features, features)
    else:
        sim = -manifold.dist_matrix(features, features, role=role)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim = sim[~mask].view(sim.shape[0], -1)

    positives = sim[labels.bool()].view(labels.shape[0], -1)
    negatives = sim[~labels.bool()].view(sim.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    targets = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    return logits, targets


class ProductSupConLoss(nn.Module):
    """SupCon on the product embedding. features: [B, n_views, total_dim]."""

    def __init__(self, manifold, role, mode, temperature=0.07):
        super().__init__()
        self.manifold = manifold
        self.role = role
        self.mode = mode
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        B, V = features.shape[0], features.shape[1]
        contrast = torch.cat(torch.unbind(features, dim=1), dim=0)  # [B*V, D]

        if self.mode == "angle":
            sim = self.manifold.angle_sim_matrix(contrast, contrast) / self.temperature
        else:
            sim = -self.manifold.dist_matrix(contrast, contrast, role=self.role) / self.temperature

        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(V, V)
        self_mask = torch.scatter(torch.ones_like(mask), 1,
                                  torch.arange(B * V).view(-1, 1).to(device), 0)
        mask = mask * self_mask

        exp_logits = torch.exp(logits) * self_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return (-mean_log_prob_pos).view(V, B).mean()


def train(student, train_loader, test_loader, unlabelled_train_loader, args,
          manifold, product_classifier):
    params_groups = get_params_groups(student)

    manifold.set_train_c(args.train_c)
    hyper_params = list(manifold.parameters()) + list(product_classifier.parameters())
    hyper_params = [p for p in hyper_params if p.requires_grad]

    optimizer_hyper = radam_.RiemannianAdam(hyper_params, lr=args.hyper_lr, stabilize=10)
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    cluster_criterion = DistillLoss(args.warmup_teacher_temp_epochs, args.epochs, args.n_views,
                                    args.warmup_teacher_temp, args.teacher_temp, hyp_c=0)

    best_test_acc_lab = 0
    for epoch in range(args.epochs):
        loss_record = AverageMeter()

        student.train()
        manifold.train()
        product_classifier.train()

        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels = class_labels.cuda(non_blocking=True)
            mask_lab = mask_lab.cuda(non_blocking=True).bool()
            images = torch.cat(images, dim=0).cuda(non_blocking=True)

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_out = student(images)               # [2B, feat_dim]
                z = manifold.project(student_out)           # [2B, total_dim] (SHARED embedding)

                # -------- classification (role="cls", flat-leaning) --------
                logits = product_classifier(z, role="cls")  # [2B, n_classes]
                teacher_logits = logits.detach()

                # supervised CE on labeled, both views
                sup_logits = torch.cat([f[mask_lab] for f in (logits / args.logit_temp).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # unsupervised self-distillation + me_max (acts on the FLAT factor,
                # so it no longer pushes the hyperbolic factor's curvature to zero)
                cluster_loss = cluster_criterion(logits, teacher_logits, epoch)
                avg_probs = (logits / args.logit_temp).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss

                # -------- representation (role="rep", curved-leaning) --------
                #          with the angle -> distance curriculum, which here is
                #          literally "start in the flat factor, ramp into the curved one".
                # unsup contrastive
                u_log_d, u_lab_d = product_info_nce_logits(z, manifold, role="rep", mode="distance")
                contrastive_loss_distance = nn.CrossEntropyLoss()(u_log_d, u_lab_d)
                u_log_a, u_lab_a = product_info_nce_logits(z, manifold, role="rep", mode="angle",
                                                           temperature=args.hyper_temp_scale * 1.0)
                contrastive_loss_angle = nn.CrossEntropyLoss()(u_log_a, u_lab_a)

                # sup contrastive
                z_sup = torch.cat([f[mask_lab].unsqueeze(1) for f in z.chunk(2)], dim=1)  # [n_lab, 2, D]
                sup_con_labels = class_labels[mask_lab]
                sup_con_loss_distance = ProductSupConLoss(manifold, role="rep", mode="distance")(z_sup, sup_con_labels)
                sup_con_loss_angle = ProductSupConLoss(manifold, role="rep", mode="angle",
                                                       temperature=0.07 * args.hyper_temp_scale)(z_sup, sup_con_labels)

                loss = 0.
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss

                loss_distance = (1 - args.sup_weight) * contrastive_loss_distance + args.sup_weight * sup_con_loss_distance
                loss_angle = (1 - args.sup_weight) * contrastive_loss_angle + args.sup_weight * sup_con_loss_angle

                lambda_distance = (epoch - (args.hyper_start_epoch - 1)) / ((args.hyper_end_epoch - 1) - (args.hyper_start_epoch - 1))
                lambda_distance = torch.max(torch.tensor([0, lambda_distance])).item()
                lambda_distance = torch.min(torch.tensor([1, lambda_distance])).item()
                lambda_distance = lambda_distance * args.hyper_max_weight

                loss_rep = (1 - lambda_distance) * loss_angle + lambda_distance * loss_distance
                loss += loss_rep

                # -------- alignment (LOAD-BEARING): full product metric, uniform & gate-free --------
                # ties the two views together across ALL factors -> keeps every
                # factor a consistent view of one representation (anti-degeneracy)
                # and keeps old/new comparable under a single metric (anchoring).
                a_log, a_lab = product_info_nce_logits(z, manifold, role="align", mode="distance",
                                                       temperature=args.hyper_temp_scale * 1.0)
                align_loss = nn.CrossEntropyLoss()(a_log, a_lab)
                loss += args.align_weight * align_loss

                # -------- homeostasis + anti-degeneracy regularizers --------
                radius_reg = torch.zeros((), device=images.device)
                if args.radius_weight > 0 and manifold.has_hyperbolic:
                    radius_reg = manifold.radius_penalty(args.target_radius)
                    loss += args.radius_weight * radius_reg

                degeneracy_reg = torch.zeros((), device=images.device)
                if args.degeneracy_weight > 0:
                    degeneracy_reg = manifold.degeneracy_penalty(z, args.degeneracy_floor)
                    loss += args.degeneracy_weight * degeneracy_reg

                # ============================================================
                #  HOOK: part-level supervision (future, >2 factors)
                #  e.g.:
                #    if args.use_part_loss:
                #        # add factor "P:128" via --factors and register a role once:
                #        #   manifold.add_role("part", init_logits=[...])
                #        part_logits = product_classifier(z, role="part")
                #        loss += args.part_weight * part_criterion(part_logits, part_targets)
                # ============================================================

                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'dist sup_con: {sup_con_loss_distance.item():.4f} '
                pstr += f'dist contrast: {contrastive_loss_distance.item():.4f} '
                pstr += f'angle sup_con: {sup_con_loss_angle.item():.4f} '
                pstr += f'angle contrast: {contrastive_loss_angle.item():.4f} '
                pstr += f'align: {align_loss.item():.4f} '
                pstr += f'radius_reg: {float(radius_reg):.4f} '
                pstr += f'degen_reg: {float(degeneracy_reg):.4f} '
                pstr += manifold.log_string() + ' '

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
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'.format(
                    epoch, batch_idx, len(train_loader), loss.item(), pstr))

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))

        exp_lr_scheduler.step()

        if epoch:
            args.logger.info('Testing on unlabelled examples in the training data...')
            all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch,
                                             'Train ACC Unlabelled', args, manifold, product_classifier)
            args.logger.info('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch,
                                                            'Test ACC', args, manifold, product_classifier)

            args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

            torch.save(student.state_dict(), args.model_path)
            args.logger.info("model saved to {}.".format(args.model_path))
            torch.save(manifold.state_dict(), args.model_path[:-3] + f'_manifold.pt')
            torch.save(product_classifier.state_dict(), args.model_path[:-3] + f'_prod_cls.pt')
            args.logger.info("product manifold/classifier saved.")

            if old_acc_test > best_test_acc_lab:
                best_test_acc_lab = old_acc_test
                args.logger.info(f'Exp Name: {args.exp_name}')
                args.logger.info(f'Metrics with best model on test set: All: {all_acc:.4f} Old: {old_acc:.4f} New: {new_acc:.4f}')
                torch.save(student.state_dict(), args.model_path[:-3] + f'_best.pt')
                torch.save(manifold.state_dict(), args.model_path[:-3] + f'_manifold_best.pt')
                torch.save(product_classifier.state_dict(), args.model_path[:-3] + f'_prod_cls_best.pt')
                args.logger.info("best product manifold/classifier saved.")


def get_det_ckpt_paths(model_path):
    if model_path.endswith('_best.pt'):
        base_path = model_path[:-len('_best.pt')]
        return {
            'model': model_path,
            'manifold': base_path + '_manifold_best.pt',
            'cls': base_path + '_prod_cls_best.pt',
        }
    base_path = model_path[:-3]
    return {
        'model': model_path,
        'manifold': base_path + '_manifold.pt',
        'cls': base_path + '_prod_cls.pt',
    }


def test(model, test_loader, epoch, save_name, args, manifold, product_classifier):
    model.eval()
    manifold.eval()
    product_classifier.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            feat = model(images)
            z = manifold.project(feat)
            logits = product_classifier(z, role=args.eval_role)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch,
                                                    eval_funcs=args.eval_funcs, save_name=save_name, args=args)
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

    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')

    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default='hypsimgcd_product', type=str)

    # base model / hyperbolic curriculum
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_model_path', default=None, type=str)
    parser.add_argument('--model_name', default='v1', type=str, choices=['v1', 'v2'])
    parser.add_argument('--hyper_start_epoch', default=0, type=int)
    parser.add_argument('--hyper_end_epoch', default=200, type=int)
    parser.add_argument('--hyper_max_weight', type=float, default=1.0)
    parser.add_argument('--hyper_temp_scale', type=float, default=1.0)
    parser.add_argument('--hyper_lr', type=float, default=0.01)
    parser.add_argument('--logit_temp', type=float, default=0.1)

    # ---------------- product-space (factor decomposition) ----------------
    # Add factors by extending this string, e.g. "E:256,P:384,P:128" (last for part-level).
    parser.add_argument('--factors', type=str, default='E:384,P:384',
                        help="E=euclidean, P=poincare, S=spherical. Per-factor overrides: P:384:c0.1:cr2.0")
    parser.add_argument('--c', type=float, default=0.1, help='default init curvature for hyperbolic factors')
    parser.add_argument('--cr', type=float, default=2.0, help='default clip radius for poincare factors')
    parser.add_argument('--train_c', action='store_true', default=True, help='learnable per-factor curvature')
    parser.add_argument('--learn_gates', action='store_true', default=True, help='learnable per-factor importance (dataset-adaptive)')
    parser.add_argument('--learn_role_weights', action='store_true', default=False, help='learnable cls/rep role weights (align stays fixed)')
    parser.add_argument('--role_init_strength', type=float, default=0.7, help='initial skew of cls(flat)/rep(curved) role weights')

    # alignment + regularizers
    parser.add_argument('--align_weight', type=float, default=1.0, help='full-metric anchor loss weight (load-bearing)')
    parser.add_argument('--radius_weight', type=float, default=0.1, help='effective-radius homeostasis penalty weight')
    parser.add_argument('--target_radius', type=float, default=0.85, help='target sqrt(c)*clip_r per poincare factor')
    parser.add_argument('--degeneracy_weight', type=float, default=0.0, help='per-factor variance-floor penalty weight (0=off)')
    parser.add_argument('--degeneracy_floor', type=float, default=0.1)

    parser.add_argument('--eval_role', default='cls', type=str, help='role used for classification at test time')

    # riemannian flag kept for parity (unused by product head)
    parser.add_argument('--riemannian', type=bool, default=False)

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    print(args)
    set_random_seed(args.seed)
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'HypSimGCD_{args.dataset_name}'])
    args.logger.info(f'Using evaluation function {args.eval_funcs} to print results')
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

    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = 256

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True

    args.logger.info('model build')

    # --------------------
    # TRANSFORMS / DATA
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)

    # ----------------------
    # PRODUCT SPACE  (manifold + classifier)
    # ----------------------
    n_classes = args.num_labeled_classes + args.num_unlabeled_classes
    manifold, product_classifier = hypnn.build_product(
        spec_str=args.factors,
        in_dim=args.feat_dim,
        n_classes=n_classes,
        default_c=args.c,
        default_clip_r=args.cr,
        train_c=args.train_c,
        learn_gates=args.learn_gates,
        learn_role_weights=args.learn_role_weights,
        role_init_strength=args.role_init_strength,
    )
    manifold = manifold.to(device)
    product_classifier = product_classifier.to(device)
    args.logger.info('Product factors: ' + ' | '.join(repr(s) for s in manifold.specs))
    args.logger.info('Initial geometry: ' + manifold.log_string())

    model = backbone.to(device)

    # ----------------------
    # TRAIN / EVAL
    # ----------------------
    if args.eval_only:
        if args.eval_model_path is not None:
            print(f'Loading evaluation model weights from {args.eval_model_path}')
            ckpt_paths = get_det_ckpt_paths(args.eval_model_path)
            model.load_state_dict(torch.load(ckpt_paths['model'], map_location='cpu'))
            if os.path.exists(ckpt_paths['manifold']):
                manifold.load_state_dict(torch.load(ckpt_paths['manifold'], map_location='cpu'))
            if os.path.exists(ckpt_paths['cls']):
                product_classifier.load_state_dict(torch.load(ckpt_paths['cls'], map_location='cpu'))
            eval_epoch = args.epochs
            test(model, test_loader_unlabelled, eval_epoch, 'Train ACC Unlabelled', args, manifold, product_classifier)
    else:
        train(model, train_loader, test_loader_labelled, test_loader_unlabelled, args, manifold, product_classifier)