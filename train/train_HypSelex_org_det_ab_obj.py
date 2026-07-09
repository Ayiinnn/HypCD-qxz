import argparse
import copy
import os
import random
import warnings
import math
import contextlib
import torch
import numpy as np
import torch.nn as nn
import hyptorch.nn as hypnn

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler
from sklearn.cluster import KMeans
from project_utils.cluster_utils import mixed_eval, AverageMeter
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights
from project_utils.cluster_and_log_utils import log_accs_from_preds
from matplotlib import pyplot as plt
from methods.clustering.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from models import vision_transformer as vits1
from models import vision_transformer2 as vits2
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from tqdm import tqdm

from config import exp_root, dino_pretrain_path, dinov2_pretrain_path

from kmeans_pytorch import kmeans
from hyptorch.pmath import dist_matrix

# object-level branch (shared backbone/projection-head; no new params).
# NOTE: SelEx is non-parametric (KMeans eval, no hyperbolic classifier), so the
# SimGCD branch's third loss (obj_cls / logit distillation) is dropped here --
# see models/object_branch_selex.py for the rationale.
from models.foreground import ForegroundCropper
from models.object_branch_selex import ObjectBranch

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def preserve_rng_state(ref_tensor):
    """Run a block without letting it advance the global RNG.

    The object branch issues extra backbone forwards (foreground cropper +
    object encoding). In training mode those forwards consume the DropPath /
    stochastic-depth RNG of the finetuned blocks, which would shift the random
    stream of the *image* branch on every subsequent batch and break bit-level
    reproducibility with the original (single-branch) run. Saving the CPU (and
    the relevant CUDA device) RNG state on entry and restoring it on exit makes
    the image branch byte-for-byte independent of the object branch: with the
    object weights at 0 (or ``--no_object_branch``) the variant reproduces the
    original exactly, and changing object hyper-parameters never moves the image
    trajectory.
    """
    cpu_state = torch.get_rng_state()
    device = ref_tensor.device if ref_tensor.is_cuda else None
    cuda_state = torch.cuda.get_rng_state(device) if device is not None else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, device)


def set_random_seed(seed: int, deterministic: bool = False, strict: bool = False) -> None:
    # [determinism] CUBLAS_WORKSPACE_CONFIG must be set BEFORE the first cuBLAS
    # call (any GPU matmul). The bulletproof alternative is to export it in the
    # shell: export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
        # warn_only=True lets training finish, using deterministic kernels where
        # they exist. NB: SelEx's SS-KMeans / kmeans_pytorch use non-deterministic
        # reductions, so warn_only=True is the practical setting here.
        torch.use_deterministic_algorithms(True, warn_only=not strict)
        print('[determinism] use_deterministic_algorithms(True, warn_only={}) | '
              'CUBLAS_WORKSPACE_CONFIG={}'.format(
                  not strict, os.environ.get('CUBLAS_WORKSPACE_CONFIG')))


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, epsilon=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes

    def forward(self, input, target, similarity, smoothing=0.5):
        target_smooth = F.one_hot(target, input.size(1)).float() * (1 - smoothing) + smoothing * similarity  # F.one_hot(similarity,input.size(1)).float()#s1/input.size(0)#coef# / self.num_classes
        return torch.nn.CrossEntropyLoss()(input, target_smooth)


class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07, hyper_c=0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.hyper_c = hyper_c

    def forward(self, features, labels=None, mask=None, is_code=False):  # , smoothing=None):
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
        if is_code:
            dist = torch.cdist(anchor_feature, contrast_feature)
            dist = -dist / (dist.sum(dim=1) + 1e-10)
        else:
            if self.hyper_c == 0:
                dist = -torch.cdist(anchor_feature, contrast_feature)
            else:
                dist = -dist_matrix(anchor_feature, contrast_feature, c=args.c)

        anchor_dot_contrast = torch.div(dist, self.temperature)

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
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def info_nce_logits(features, confusion_factor, args, is_code=False, hyper_c=0):
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(args.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    if is_code:
        dist = torch.cdist(features, features, p=2)
        similarity_matrix = -dist / (dist.sum(dim=1) + 1e-10)
    else:
        if hyper_c == 0:
            similarity_matrix = -torch.cdist(features, features)
        else:
            similarity_matrix = -dist_matrix(features, features, c=args.c)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    confusion_factor = confusion_factor[~mask].view(confusion_factor.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    pos_confs = confusion_factor[labels.bool()].view(confusion_factor.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    neg_confs = confusion_factor[~labels.bool()].view(confusion_factor.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    log_confs = torch.cat([pos_confs, neg_confs], dim=1)

    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / args.temperature
    return logits, labels, log_confs


def train(projection_head, model, train_loader, test_loader, unlabelled_train_loader, merge_train_loader, merge_train_loader_test, args):
    optimizer = SGD(list(projection_head.parameters()) + list(model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3)

    sup_con_crit = SupConLoss(hyper_c=args.c)
    sup_con_crit_angle = SupConLoss(hyper_c=0)

    best_epoch_lab, best_epoch_comb, best_epoch = 0, 0, 0
    strategy = args.strategy
    cluster_momentum = args.cluster_momentum

    best_stats = []
    Total_loss = []
    Contrastive_loss = []

    accuracy_old = []
    accuracy_new = []
    accuracy_all = []

    best_train_acc = 0
    best_test_acc_lab = 0
    best_train_acc_all = 0          # [all-best] best ALL-acc on the (unlabelled) train set
    unsupervised_smoothing = args.unsupervised_smoothing
    train_report_interval = args.train_report_interval
    prototype_extraction_interval = args.prototype_extraction_interval
    Distance = args.distance

    # ---- object-level branch (shared weights, no extra parameters) ----
    fg_cropper = None
    object_branch = None
    if args.use_object_branch:
        fg_cropper = ForegroundCropper(
            model, model_name=args.dino, source=args.obj_fg_source,
            keep=args.obj_fg_keep, box_pad=args.obj_fg_pad, out_size=args.image_size,
        )
        object_branch = ObjectBranch(args)
        print('[object-branch] enabled | fg_source={} parent={} w(ent/dist)={}/{}'.format(
            fg_cropper.source, args.obj_entail_parent, args.obj_entail_weight, args.obj_dist_weight))
        print('[object-branch] supervision: {}'.format(object_branch.supervision_desc()))

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        train_acc_record = AverageMeter()

        loss_cons_record = AverageMeter()

        with torch.no_grad():
            if epoch % prototype_extraction_interval == 0:
                uq_index, all_preds, cluster_protos_list, preds_ind_list, metrics = extract_labeled_protos(model, merge_train_loader, args=args)
                for i in range(len(preds_ind_list)):
                    preds_ind_list[i] = preds_ind_list[i].to(device).long()
                    cluster_protos_list[i] = cluster_protos_list[i].to(device)
                    if Distance == 'cosine':
                        cluster_protos_list[i] = cluster_protos_list[i] / torch.norm(cluster_protos_list[i], dim=1).unsqueeze(1)

                cluster_distances_list = []
                cluster_radius_list = []
                for i in range(len(preds_ind_list)):

                    if Distance == 'euclidean':
                        cluster_distances = torch.cdist(cluster_protos_list[i], cluster_protos_list[i])
                    else:
                        cluster_distances = torch.matmul(cluster_protos_list[i], cluster_protos_list[i].T)

                    cluster_distances_list.append(cluster_distances.clone())
                    cluster_radius = (cluster_distances + torch.eye(cluster_distances.shape[0]).to(device) * cluster_distances.max()).min(dim=1)[0] / 2
                    cluster_radius_list.append(cluster_radius.clone())


        projection_head.train()
        model.train()

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            loss_angle, loss_distance = 0.0, 0.0

            images, class_labels, uq_idxs, mask_lab = batch
            mask_lab = mask_lab[:, 0]

            class_labels, mask_lab = class_labels.to(device), mask_lab.to(device).bool()
            images = torch.cat(images, dim=0).to(device)

            # Extract features with base model
            features = model(images)

            # Pass features through projection head
            features_1, features_2 = features.detach().chunk(2)
            all_features = torch.cat([features_1, features_2], dim=0)
            features = projection_head(features)          # (2B, D) Poincare = img_feat
            # L2-normalize features
            # features = torch.nn.functional.normalize(features, dim=-1)

            # ---- object-level branch (shared backbone + projection head) ----
            # Runs BEFORE the image `features` are re-chunked below. The extra
            # backbone forwards are wrapped in preserve_rng_state so the image
            # branch's random stream is byte-for-byte unaffected (with obj weights
            # at 0 / --no_object_branch this reduces to the base SelEx run).
            obj_loss = None
            obj_logs = None
            if args.use_object_branch:
                with preserve_rng_state(images):
                    obj_images = fg_cropper(images)                       # (2B, 3, H, W)
                    obj_feat = projection_head(model(obj_images))         # (2B, D) Poincare
                cluster_labels_full = None
                if object_branch.mode != 'legacy':
                    # SelEx finest cluster assignment (pseudo-label) per sample,
                    # aligned to the batch exactly like the SupCon hierarchy below.
                    cluster_labels_full = preds_ind_list[0][np.argsort(uq_index)[uq_idxs]]
                obj_loss, obj_logs = object_branch(
                    img_feat=features,
                    obj_feat=obj_feat,
                    class_labels=class_labels,
                    mask_lab=mask_lab,
                    cluster_labels=cluster_labels_full,
                )

            with torch.no_grad():
                confusion_factor = 0
                if Distance == 'euclidean':
                    pair_dist = torch.cdist(all_features, all_features)
                else:
                    normalized_feats = all_features / torch.norm(all_features.unsqueeze(1))
                    pair_dist = torch.matmul(normalized_feats, normalized_feats.T)

                n_labeled = args.num_labeled_classes
                n_unlabeled = args.num_unlabeled_classes

                for i in range(len(preds_ind_list)):
                    cluster_labels = (preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]).clone()
                    cluster_indexer = F.one_hot(cluster_labels.long(), n_labeled + n_unlabeled).float().T
                    n_labeled = max(int(n_labeled / 2), 1)
                    n_unlabeled = max(int(n_unlabeled / 2), 1)

                    cluster_indexer = torch.cat([cluster_indexer, cluster_indexer], dim=1)
                    n_samples = torch.sum(cluster_indexer, dim=1).unsqueeze(1)
                    n_samples[n_samples == 0] = 1

                    if Distance == 'euclidean':
                        distance = torch.cdist(all_features, cluster_protos_list[i].float())
                    else:
                        normalized_feats = all_features / torch.norm(all_features.unsqueeze(1))
                        distance = torch.matmul(normalized_feats, cluster_protos_list[i].float().T)

                    cluster_radius_list[i] = (cluster_indexer * distance.T).sum(dim=1) / n_samples.squeeze()  * (1 - cluster_momentum) + cluster_radius_list[i] * cluster_momentum
                    cluster_labels = torch.cat([cluster_labels, cluster_labels])

                    if Distance == 'euclidean':
                        if strategy == 'zero_one':
                            confusion_factor += (pair_dist > 2 * cluster_radius_list[i][
                                cluster_labels]).float() / 2 ** i
                        elif strategy == 'pair_dist':
                            confusion_factor += pair_dist / 2 ** i
                        elif strategy == 'pair_cluster':
                            confusion_factor += distance[:, cluster_labels] / 2 ** i
                        else:
                            pass
                    else:
                        if strategy == 'zero_one':
                            confusion_factor += (pair_dist < distance[:, cluster_labels] / 2).float() / 2 ** i
                        elif strategy == 'pair_dist':
                            confusion_factor += -pair_dist / 2 ** i
                        elif strategy == 'pair_cluster':
                            confusion_factor += -distance[:, cluster_labels] / 2 ** i

            # Choose which instances to run the contrastive loss on
            if args.contrast_unlabel_only:
                # Contrastive loss only on unlabelled instances
                f1, f2 = [f[~mask_lab] for f in features.chunk(2)]
                con_feats = torch.cat([f1, f2], dim=0)
            else:
                # Contrastive loss for all examples
                con_feats = features
            confusion_factor = (confusion_factor - confusion_factor.min()) / (confusion_factor.max() - confusion_factor.min() + 0.0000001)
            confusion_factor = confusion_factor / confusion_factor.sum(dim=1)

            torch.cuda.empty_cache()

            contrastive_logits, contrastive_labels, similarity = info_nce_logits(features=con_feats, confusion_factor=confusion_factor, args=args, hyper_c=args.c)
            contrastive_loss = LabelSmoothingLoss()(contrastive_logits, contrastive_labels, similarity, unsupervised_smoothing)
            loss_distance += (1 - args.sup_con_weight) * contrastive_loss

            contrastive_logits_angle, contrastive_labels_angle, similarity_angle = info_nce_logits(features=con_feats, confusion_factor=confusion_factor, args=args, hyper_c=0)
            contrastive_loss_angle = LabelSmoothingLoss()(contrastive_logits_angle, contrastive_labels_angle, similarity_angle, unsupervised_smoothing)
            loss_angle += (1 - args.sup_con_weight) * contrastive_loss_angle
            
            f1n, f2n = features.chunk(2)
            semisup_con_feats = torch.cat([f1n.unsqueeze(1), f2n.unsqueeze(1)], dim=1)

            # Supervised contrastive loss
            f1, f2 = [f[mask_lab] for f in features.chunk(2)]
            sup_con_feats = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            sup_con_labels = class_labels[mask_lab]
            sup_con_loss = sup_con_crit(sup_con_feats, labels=sup_con_labels)
            sup_con_loss_angle = sup_con_crit_angle(sup_con_feats, labels=sup_con_labels)
            dimension = semisup_con_feats.shape[-1]
            for i in range(len(preds_ind_list)):
                sup_con_loss += sup_con_crit(semisup_con_feats[:, :, :int(dimension / 2 ** (i + 1))], labels=preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]) / 2 ** (i + 1)
                sup_con_loss_angle += sup_con_crit_angle(semisup_con_feats[:, :, :int(dimension / 2 ** (i + 1))], labels=preds_ind_list[i][np.argsort(uq_index)[uq_idxs]]) / 2 ** (i + 1)
            loss_distance += args.sup_con_weight * sup_con_loss / 2
            loss_angle += args.sup_con_weight * sup_con_loss_angle / 2
            # Total loss
            lambda_distance = (epoch - (args.hyper_start_epoch - 1)) / ((args.hyper_end_epoch - 1) - (args.hyper_start_epoch - 1))
            lambda_distance = torch.max(torch.tensor([0, lambda_distance])).item()
            lambda_distance = torch.min(torch.tensor([1, lambda_distance])).item()
            lambda_distance = lambda_distance * args.hyper_max_weight
            
            loss_in = (1 - lambda_distance) * loss_angle + lambda_distance * loss_distance

            # Train acc
            _, pred = contrastive_logits.max(1)
            acc = (pred == contrastive_labels).float().mean().item()
            train_acc_record.update(acc, pred.size(0))

            loss_cons_record.update(loss_in.item(), class_labels.size(0))
            loss = loss_in
            # Object-branch grounding losses are added at full weight, OUTSIDE the
            # angle/distance blend -- mirroring how the SimGCD obj variant adds
            # obj_loss on top of loss_rep (they are already hyperbolic, curvature c).
            if obj_loss is not None:
                loss = loss + obj_loss
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        obj_pstr = ''
        if obj_logs is not None:
            obj_pstr = ' | obj_ent: {:.4f} obj_dist: {:.4f}'.format(
                obj_logs["obj_entail"].item(), obj_logs["obj_dist"].item())
        print('Epoch: {} Avg Loss: {:.3f} | Constrastive: {:.3f}{} '.format(epoch, loss_record.avg, loss_cons_record.avg, obj_pstr))

        Total_loss.append(loss_record.avg)
        Contrastive_loss.append(loss_cons_record.avg)

        with torch.no_grad():
            if (epoch + 1) % train_report_interval == 0:
                print('Testing on unlabelled examples in the training data...')
                all_acc, old_acc, new_acc = test_kmeans(model, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
            else:
                all_acc, old_acc, new_acc = metrics["all_acc"], metrics["old_acc"], metrics["new_acc"]
            print('Testing on disjoint test set...')
            all_acc_test, old_acc_test, new_acc_test = test_kmeans(model, test_loader, epoch=epoch, save_name='Test ACC', args=args)

            accuracy_old.append(old_acc)
            accuracy_new.append(new_acc)
            accuracy_all.append(all_acc)

        # ----------------
        # LOG
        # ----------------
        args.writer.add_scalar('Loss', loss_record.avg, epoch)
        args.writer.add_scalar('Train Acc Labelled Data', train_acc_record.avg, epoch)
        args.writer.add_scalar('LR', get_mean_lr(optimizer), epoch)

        if (epoch + 1) % train_report_interval == 0:
            print('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            print('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))

        # Step schedule
        exp_lr_scheduler.step()

        torch.save(model.state_dict(), args.model_path)
        print("model saved to {}.".format(args.model_path))
        torch.save(projection_head.state_dict(), args.model_path[:-3] + '_proj_head.pt')

        if old_acc_test > best_test_acc_lab:
            print(f'Best ACC on new Classes on disjoint test set: {new_acc_test:.4f}...')
            #print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            best_stats = [all_acc, old_acc, new_acc]
            torch.save(model.state_dict(), args.model_path[:-3] + f'_best.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best.pt')

            best_test_acc_lab = old_acc_test
            best_epoch = epoch

        # ---- [all-best] additionally save the best model by ALL-acc on the
        # (unlabelled) train set. Mirrors the SimGCD obj variant's *_best_acc_all.pt.
        # `all_acc` is the primary eval-func metric (updated every epoch from the
        # per-epoch SS-KMeans prototype extraction when not a full-report epoch).
        if all_acc > best_train_acc_all:
            best_train_acc_all = all_acc
            print(f'Best (train all-acc) model: All: {all_acc:.4f} Old: {old_acc:.4f} New: {new_acc:.4f}')
            torch.save(model.state_dict(), args.model_path[:-3] + f'_best_acc_all.pt')
            print("model saved to {}.".format(args.model_path[:-3] + f'_best_acc_all.pt'))
            torch.save(projection_head.state_dict(), args.model_path[:-3] + f'_proj_head_best_acc_all.pt')

        print('Best Train Epochs:  {}'.format(best_epoch))

    print('############# Final Reports #############')
    print('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(best_stats[0], best_stats[1], best_stats[2]))
    print('Best Train Epochs:  {} '.format(best_epoch))


def extract_labeled_protos(model, train_loader, args):
    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])
    ids = np.array([])
    mask_cls = np.array([])
    metrics = dict()

    for batch_idx, (images, label, uq_idx, mask_lab_) in enumerate(tqdm(train_loader)):
        images = images[0].cuda()
        label, mask_lab_ = label.to(device), mask_lab_.to(device).bool()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        ids = np.append(ids, uq_idx.cpu().numpy())
        mask = np.append(mask, mask_lab_.cpu().bool().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    mask = mask.astype(bool)
    mask_cls = mask_cls.astype(bool)
    # -----------------------
    # K-MEANS
    # -----------------------
    # print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    l_feats = all_feats[mask]  # Get labelled set
    u_feats = all_feats[~mask]  # Get unlabelled set
    l_targets = targets[mask]  # Get labelled targets
    u_targets = targets[~mask]  # Get unlabelled targets
    n_samples = len(targets)

    if args.unbalanced:
        cluster_size = None
    else:
        cluster_size = math.ceil(n_samples / (args.num_labeled_classes + args.num_unlabeled_classes))
    kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                              max_iterations=10, init='k-means++',
                              n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                              mode=None, protos=None, cluster_size=cluster_size)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats, l_targets, u_targets))

    kmeanssem.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeanssem.labels_
    mask_cls = mask_cls[~mask]
    preds = all_preds.cpu().numpy()[~mask]
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls, eval_funcs=args.eval_funcs, save_name='SS-K-Means Train ACC Unlabelled', print_output=True)
    metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
    prototype_higher = []
    prototypes = kmeanssem.cluster_centers_
    prototype_higher.append(prototypes.clone())
    n_labeled = args.num_labeled_classes
    n_novel = args.num_unlabeled_classes
    label_proto = prototypes.cpu().numpy()[:args.num_labeled_classes, :]
    preds_higher = []

    preds_higher.append(all_preds.clone())
    print('Hierarchy clustering')
    mask_known = (all_preds < args.num_labeled_classes).cpu().numpy()
    l_feats = all_feats[mask_known]  # Get labelled set
    u_feats = all_feats[~mask_known]
    l_feats, u_feats = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats))

    while n_labeled > 1:
        n_labeled = max(int(n_labeled / 2), 1)
        n_novel = max(int(n_novel / 2), 1)
        # kmeans on labeled proto
        kmeans_l = KMeans(n_clusters=n_labeled, random_state=0).fit(label_proto)
        preds_labels = torch.from_numpy(kmeans_l.labels_).to(device)
        level_l_targets = preds_labels[all_preds[mask_known]]  # all labeled data at the new cluster level
        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil(n_samples / (n_labeled + n_novel))
        # sskmeans on features with new level labels
        kmeans_higher = SemiSupKMeans(k=n_labeled + n_novel, tolerance=1e-4,
                                      max_iterations=10, init='k-means++',
                                      n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                                      mode=None, protos=None, cluster_size=cluster_size)
        kmeans_higher.fit_mix(u_feats, l_feats, level_l_targets)
        preds_level = kmeans_higher.labels_
        prototypes_level = kmeans_higher.cluster_centers_
        prototype_higher.append(prototypes_level.clone())
        preds_higher.append(preds_level.to(device).clone())

    return ids, all_preds, prototype_higher, preds_higher, metrics


def test(model, test_loader, args):
    # test the model accuracy using BSSKMeans
    model.eval()

    with torch.no_grad():
        all_feats = []
        targets = np.array([])
        mask = np.array([])
        ids = np.array([])
        mask_cls = np.array([])
        metrics = dict()

        for batch_idx, (images, label, uq_idx, mask_lab_) in enumerate(tqdm(test_loader)):
            if isinstance(images, list):
                images = images[0]
            images = images.to(device)
            label, mask_lab_ = label.to(device), mask_lab_.to(device).bool()

            # Pass features through base model and then additional learnable transform (linear layer)
            feats = model(images)
            all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            ids = np.append(ids, uq_idx.cpu().numpy())
            mask = np.append(mask, mask_lab_.cpu().bool().numpy())
            mask_cls = np.append(mask_cls, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

        mask = mask.astype(bool)
        mask_cls = mask_cls.astype(bool)
        # -----------------------
        # K-MEANS
        # -----------------------
        # print('Fitting K-Means...')
        all_feats = np.concatenate(all_feats)
        l_feats = all_feats[mask]  # Get labelled set
        u_feats = all_feats[~mask]  # Get unlabelled set
        l_targets = targets[mask]  # Get labelled targets
        u_targets = targets[~mask]  # Get unlabelled targets
        n_samples = len(targets)

        if args.unbalanced:
            cluster_size = None
        else:
            cluster_size = math.ceil(n_samples / (args.num_labeled_classes + args.num_unlabeled_classes))
        kmeanssem = SemiSupKMeans(k=args.num_labeled_classes + args.num_unlabeled_classes, tolerance=1e-4,
                                  max_iterations=10, init='k-means++',
                                  n_init=1, random_state=None, n_jobs=None, pairwise_batch_size=1024,
                                  mode=None, protos=None, cluster_size=cluster_size)

        l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to(device) for x in (l_feats, u_feats, l_targets, u_targets))

        kmeanssem.fit_mix(u_feats, l_feats, l_targets)
        all_preds = kmeanssem.labels_
        mask_cls = mask_cls[~mask]
        preds = all_preds.cpu().numpy()[~mask]
        all_acc, old_acc, new_acc = log_accs_from_preds(y_true=u_targets.cpu().numpy(), y_pred=preds, mask=mask_cls, eval_funcs=args.eval_funcs, save_name='SS-K-Means Train ACC Unlabelled Test', print_output=True)
        metrics["all_acc"], metrics["old_acc"], metrics["new_acc"] = all_acc, old_acc, new_acc
        return metrics


def test_kmeans(model, test_loader, epoch, save_name, args, Use_GPU=True):
    model.eval()
    all_feats = []

    targets = np.array([])
    mask = np.array([])

    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda()
        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)
        all_feats.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))

    # Get portion of mask_cls which corresponds to the unlabelled set
    mask = mask.astype(bool)
    all_feats = np.concatenate(all_feats)
    # -----------------------
    # EVALUATE
    # -----------------------
    if Use_GPU:
        preds, prototypes = kmeans(X=torch.from_numpy(all_feats).to(device), num_clusters=args.num_unlabeled_classes + args.num_labeled_classes, distance='euclidean', device=device, tqdm_flag=False)
        preds, prototypes = preds.cpu().numpy(), prototypes.cpu().numpy()
    else:
        kmeanss = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
        preds = kmeanss.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask, T=epoch, eval_funcs=args.eval_funcs, save_name=save_name, writer=args.writer)

    return all_acc, old_acc, new_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, scars, aircraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=True)

    parser.add_argument('--grad_from_block', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)
    parser.add_argument('--mlp_out_dim', default=256, type=int)

    parser.add_argument('--strategy', type=str, default='zero_one')
    parser.add_argument('--cluster_momentum', type=float, default=1)

    parser.add_argument('--unsupervised_smoothing', type=float, default=1)
    parser.add_argument('--distance', type=str, default='euclidean', help='options: euclidean, cosine')

    parser.add_argument('--train_report_interval', default=200, type=int)
    parser.add_argument('--prototype_extraction_interval', default=1, type=int)

    parser.add_argument('--gpu_clustering', type=str2bool, default=True)
    parser.add_argument('--unbalanced', type=str2bool, default=False)

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--report', type=str2bool, default=True)

    # hyperbolic
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_model_path', default=None, type=str)
    parser.add_argument('--hyper_start_epoch', default=0, type=int)
    parser.add_argument('--hyper_end_epoch', default=200, type=int)
    parser.add_argument('--c', default=0.05, type=float)
    parser.add_argument('--cr', type=float, default=0)
    parser.add_argument('--riemannian', type=bool, default=False)
    parser.add_argument('--dino', default='v1', type=str)
    parser.add_argument('--hyper_max_weight', default=1.0, type=float)

    # ----------------------
    # DETERMINISM (det) -- opt-in; default False => unchanged behaviour.
    # ----------------------
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help='Enable deterministic algorithms (warn_only) + reproducible data loading. '
                             'Slower, but reproducible run-to-run on identical HW + library versions. '
                             'NB: SelEx KMeans steps use non-deterministic reductions, so warn_only is used.')
    parser.add_argument('--strict_deterministic', action='store_true', default=False,
                        help='Like --deterministic but RAISES on the first non-deterministic op; use to '
                             'locate which kernel breaks reproducibility.')

    # ----------------------
    # OBJECT-LEVEL BRANCH (HypSelEx_org_det_ab_obj)
    # Shared backbone + projection head; no new parameters. SelEx is non-parametric,
    # so only the two FEATURE-SPACE grounding losses are used (obj_cls is dropped;
    # see models/object_branch_selex.py).
    # ----------------------
    parser.add_argument('--use_object_branch', action='store_true', default=True,
                        help='enable the object-level (foreground) branch')
    parser.add_argument('--no_object_branch', dest='use_object_branch', action='store_false',
                        help='disable the object branch (recovers the base det+ab SelEx behaviour)')
    # loss weights (one per feature-space loss; obj_cls has no SelEx counterpart)
    parser.add_argument('--obj_entail_weight', type=float, default=0.1,
                        help='weight of the feature-space entailment-cone loss')
    parser.add_argument('--obj_dist_weight', type=float, default=0.1,
                        help='weight of the feature-space hyperbolic-distance (InfoNCE) loss')
    # entailment cone configuration
    parser.add_argument('--obj_entail_parent', type=str, default='image', choices=['image', 'object'],
                        help="which view is the cone apex (parent). 'image'=scene entails object (paper); "
                             "'object'=object entails image (HyCoCLIP box-as-parent).")
    parser.add_argument('--obj_aperture_scale', type=float, default=1.2,
                        help='aperture scale of the entailment cone (HyCoCLIP intra-modal default 1.2)')
    parser.add_argument('--obj_min_radius', type=float, default=0.1,
                        help='min-radius constant of the half-aperture')
    # distance loss configuration
    parser.add_argument('--obj_dist_temp', type=float, default=0.1,
                        help='temperature of the hyperbolic-distance InfoNCE')
    # ---- supervision structure of the obj losses (mirrors object_branch_multi) ----
    #   legacy (DEFAULT) -- same-view diagonal for both losses; off-diagonal = negative.
    #                       With obj weights at 0 this reduces to the base SelEx pipeline.
    #   multi            -- dist keeps ONE positive img_i<->obj_i and merely REMOVES false
    #                       negatives (cross-view, labelled same-class, same-CLUSTER pseudo
    #                       via SelEx's preds_ind_list[0]) from the InfoNCE denominator
    #                       (no attraction => no log(P) floor, no view collapse); ent stays
    #                       diagonal img->obj. See models/object_branch_selex.py.
    parser.add_argument('--obj_sup_mode', type=str, default='legacy', choices=['legacy', 'multi'],
                        help="object-branch supervision. 'legacy' (default) = same-view-diagonal "
                             "(off-diagonal negative). 'multi' = false-negative elimination on the "
                             "dist denominator using SelEx cluster pseudo-labels.")
    parser.add_argument('--obj_legacy_supervision', action='store_true', default=False,
                        help='force legacy supervision (equivalent to --obj_sup_mode legacy).')
    # foreground cropper configuration
    parser.add_argument('--obj_fg_source', type=str, default='auto', choices=['auto', 'attention', 'cls_sim'],
                        help="foreground saliency source. 'auto' -> attention for v1, cls_sim for v2.")
    parser.add_argument('--obj_fg_keep', type=float, default=0.6,
                        help='fraction of saliency mass kept as foreground (DINO default 0.6)')
    parser.add_argument('--obj_fg_pad', type=float, default=0.1,
                        help='relative padding added on each side of the foreground box')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    set_random_seed(args.seed, deterministic=args.deterministic, strict=args.strict_deterministic)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=[f'HypSelEx_{args.dataset_name}'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':
        args.interpolation = 3
        args.crop_pct = 0.875
        if args.dino == 'v1':
            model = vits1.__dict__['vit_base']()
            torch.cuda.empty_cache()
            state_dict = torch.load(dino_pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            model = vits2.__dict__['vit_base']()
            torch.cuda.empty_cache()
            state_dict = torch.load(dinov2_pretrain_path, map_location='cpu')
            model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir + 'model_best.pt', map_location='cpu'), strict=False)
        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        max_block = 0
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num > max_block:
                    max_block = block_num

                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:
        raise NotImplementedError
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name, train_transform, test_transform, args)

    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / (unlabelled_len + label_len) for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    # --------------------
    # DATALOADERS
    # --------------------
    train_dataset_test = copy.deepcopy(train_dataset)
    train_dataset_test.labelled_dataset.transform = test_transform
    train_dataset_test.unlabelled_dataset.transform = test_transform
    # [determinism] reproducible data loading; generator/worker_init_fn = None is a
    # no-op, so default (non-deterministic) behaviour is unchanged unless a switch is passed.
    loader_generator = None
    worker_init_fn = None
    if args.deterministic or args.strict_deterministic:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed)
        def worker_init_fn(worker_id):
            worker_seed = torch.initial_seed() % (2 ** 32)
            np.random.seed(worker_seed)
            random.seed(worker_seed)
    merge_train_loader_test = DataLoader(train_dataset_test, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    merge_train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, sampler=sampler, drop_last=True, generator=loader_generator, worker_init_fn=worker_init_fn)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers, batch_size=int(args.batch_size / 2), shuffle=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=int(args.batch_size / 2), shuffle=False)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits1.__dict__['DINOHead'](in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
    if args.cr != 0:
        hyperbolic_projector = hypnn.ToPoincare(c=args.c, ball_dim=args.mlp_out_dim, riemannian=args.riemannian, clip_r=args.cr).to(device)
    else:
        hyperbolic_projector = hypnn.ToPoincare(c=args.c, ball_dim=args.mlp_out_dim, riemannian=args.riemannian).to(device)
    if args.warmup_model_dir is not None:
        print(f'Loading projection head weights from {args.warmup_model_dir}')
        projection_head.load_state_dict(torch.load(args.warmup_model_dir + 'model_proj_head_best.pt', map_location='cpu'), strict=False)

    projection_head = nn.Sequential(projection_head, hyperbolic_projector)
    projection_head.to(device)

    # ----------------------
    # TRAIN
    # ----------------------
    if args.eval_only:
        if args.eval_model_path is not None:
            print(f'Loading evaluation model weights from {args.eval_model_path}')
            model.load_state_dict(torch.load(args.eval_model_path, map_location='cpu'))
            test(model, merge_train_loader_test, args)
    else:
        train(projection_head, model, train_loader, test_loader_labelled, test_loader_unlabelled, merge_train_loader, merge_train_loader_test, args)