"""Part-level (PartCo) branch, hyperbolized in the HypCD recipe.

Design (strict A+B, no new factors):
  A = Hyp-SimGCD baseline (train_HypCD_org_det_ab.py).  Its recipe maps a
      SimGCD loss L(cos) to the hybrid pair
          (1 - a_d) * L(cosine, temp * hyper_temp_scale)  +  a_d * L(-D_H, temp)
      with a_d ramping linearly 0 -> hyper_max_weight (HypCD paper Sec. 3.4,
      Eq. 8-11: S in {Sa = cos, Sd = -D_H}).
  B = PartCo part-level correspondence losses (partco repo,
      models/partco_loss.py), which are cosine SupCon losses over per-part
      average-pooled patch projections (PartCo paper Eq. 6-7).

This file applies A's loss mapping to B's two losses.  The classes below are
the partco repo's ``PatchSupConLoss`` / ``PatchUnConLoss`` copied verbatim,
generalized ONLY at the similarity computation with a ``hyp_c`` switch --
exactly how the baseline's own ``SupConLoss``/``info_nce_logits`` switch
between angle and distance:

  * ``hyp_c == 0``   -> original cosine path, numerically identical to the
    partco repo (the ANGLE branch).  Note cosine on exp-mapped points equals
    cosine on the pooled Euclidean features (expmap0 preserves directions;
    HypCD Eq. 10: "cosine similarity remains equivalent in both E^n and H^n"),
    so the angle branch of the hyperbolized part loss IS the stable PartCo
    loss.
  * ``hyp_c > 0``    -> per-part pooled features are mapped to the Poincare
    ball by the SAME ToPoincare operator (clipping + expmap0) used for the
    image branch, and the similarity becomes S = -D_H (HypCD Sd).  Pooling
    stays Euclidean (pre-expmap), matching how the baseline exp-maps the
    (Euclidean) backbone feature.

Future extension point (kept orthogonal, not implemented here): the ball
places generic concepts near the origin and specific ones near the boundary
(HyCoCLIP / entailment cones); a radial gate on the part vs. image radii
would slot in at the ``to_hyp`` operators without touching these losses.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from hyptorch.pmath import dist_matrix


# --------------------------------------------------------------------------- #
#  Patch projection head -- verbatim from partco repo, models/model.py
#  (https://arxiv.org/pdf/2212.04994)
# --------------------------------------------------------------------------- #
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


def build_patch_projector():
    """The exact patch projector stack used by the partco repo trainer."""
    return nn.Sequential(
        nn.LayerNorm(768),
        nn.Dropout(0.1),
        Patch_Projection(),
    )


# --------------------------------------------------------------------------- #
#  Patch-token extraction / part-label grid alignment
# --------------------------------------------------------------------------- #
def extract_patch_tokens(backbone, images, model_name):
    """Post-norm patch tokens [B, N, 768] for the two mainline backbones.

    v2 (DINOv2 ViT-B/14, models/vision_transformer2.py): the partco path,
        ``forward_features(x)['x_norm_patchtokens']`` (register tokens are
        already excluded by the slicing inside forward_features).
        224px -> N = 16*16 = 256, exactly the PartCo label grid.
    v1 (DINO ViT-B/16, models/vision_transformer.py):
        ``forward(x, return_all_patches=True)`` returns the post-norm token
        sequence; drop the CLS token.  224px -> N = 14*14 = 196.
    """
    if model_name == 'v2':
        return backbone.forward_features(images)['x_norm_patchtokens']
    elif model_name == 'v1':
        return backbone(images, return_all_patches=True)[:, 1:, :]
    raise ValueError(f'Unknown model_name {model_name!r} (expected v1/v2)')


def align_patch_labels(patch_label, num_patch_tokens):
    """Match the PartCo label grid to the backbone's patch-token grid.

    PartCo labels are stored on the DINOv2/14 grid (16x16 at 224px); for v2
    this is an exact no-op (labels pass through untouched -> byte-identical
    to the partco repo).  For a /16 backbone (v1: 14x14) the id map is
    nearest-resized -- ids stay valid part ids, no new labels are invented.
    """
    g = int(round(math.sqrt(num_patch_tokens)))
    assert g * g == num_patch_tokens, f'non-square patch grid: {num_patch_tokens}'
    if patch_label.dim() == 3 and patch_label.shape[-2] == g and patch_label.shape[-1] == g:
        return patch_label
    lab = patch_label.float().unsqueeze(1)
    lab = F.interpolate(lab, size=(g, g), mode='nearest')
    return lab.squeeze(1).long()


# --------------------------------------------------------------------------- #
#  Dual-space PartCo losses.
#  Bodies are verbatim from partco repo models/partco_loss.py; the ONLY edits
#  are the [hypartco] similarity blocks (angle <-> hyperbolic distance).
# --------------------------------------------------------------------------- #
class HypPatchSupConLoss(nn.Module):
    """partco ``PatchSupConLoss`` with the HypCD angle/distance switch."""

    def __init__(self, temperature=0.07, base_temperature=0.07, hyp_c=0, to_hyp=None):
        super(HypPatchSupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        # [hypartco] hyp_c == 0 -> original cosine (angle) loss;
        #            hyp_c  > 0 -> S = -D_H on to_hyp(-pooled features).
        self.hyp_c = hyp_c
        self.to_hyp = to_hyp
        if self.hyp_c != 0 and self.to_hyp is None:
            raise ValueError('hyp_c > 0 requires a ToPoincare operator (to_hyp)')

    def forward(self, sup_patch_out, patch_label, sup_con_labels):
        """
        Args:
            sup_patch_out: patch features of shape [B, 256, 512]
            patch_label: patch labels of shape [B, 16, 16]
            sup_con_labels: image labels of shape [B]
        Returns:
            loss: patch pooling supervised contrastive loss
        """
        batch_size = sup_patch_out.size(0)
        device = sup_patch_out.device

        # Reshape patch_label from [B, 16, 16] to [B, 256]
        # patch_label_flat = patch_label.reshape(batch_size, 256)
        patch_label_flat = patch_label.reshape(batch_size, -1)

        # Dictionary to store features by part type
        part_features_dict = {}
        part_labels_dict = {}

        # For each sample in the batch
        for i in range(batch_size):
            features = sup_patch_out[i]  # [256, proj feat dim]
            part_labels = patch_label_flat[i]  # [256]
            category_label = sup_con_labels[i]  # scalar

            # Get unique part labels
            unique_parts = torch.unique(part_labels)
            unique_parts = unique_parts[unique_parts > 0]  # Exclude background (0) if any

            # Average features for each part
            for part in unique_parts:
                part_mask = (part_labels == part)
                if part_mask.sum() > 0:
                    part_features = features[part_mask]
                    avg_feature = torch.mean(part_features, dim=0)  # [proj feat dim]

                    # Initialize dictionary entry if not exist
                    part_id = part.item()
                    if part_id not in part_features_dict:
                        part_features_dict[part_id] = []
                        part_labels_dict[part_id] = []

                    # Add feature and label to corresponding part type
                    part_features_dict[part_id].append(avg_feature)
                    part_labels_dict[part_id].append(category_label)

        if not part_features_dict:
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = 0.0
        valid_parts = 0

        # Process each part type separately
        for part_id in part_features_dict:
            features = part_features_dict[part_id]
            labels = part_labels_dict[part_id]

            # Skip if we don't have enough features for this part type
            if len(features) <= 1:
                continue

            # Stack features and labels
            features = torch.stack(features)  # [N, 512]
            labels = torch.stack(labels)      # [N]

            # Check if we have at least one positive pair (same category)
            unique_labels, counts = torch.unique(labels, return_counts=True)
            if (counts > 1).sum() == 0:
                continue  # No category appears more than once for this part

            # [hypartco] angle vs. hyperbolic-distance similarity (HypCD Eq. 8-10)
            if self.hyp_c == 0:
                # Normalize features
                features = F.normalize(features, dim=1)

                # Compute similarity matrix
                sim_matrix = torch.mm(features, features.transpose(0, 1)) / self.temperature
            else:
                features = self.to_hyp(features)
                sim_matrix = -dist_matrix(features, features, c=self.hyp_c) / self.temperature

            # Create mask based on labels
            labels_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # [N, N]

            # Remove self-contrast
            eye_mask = torch.eye(labels_equal.size(0), device=device)
            pos_mask = labels_equal - eye_mask

            # Check if we have any positive pairs
            if pos_mask.sum() == 0:
                continue

            # For numerical stability
            logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - logits_max.detach()

            # Compute exp_logits and mask out self-similarity
            exp_logits = torch.exp(sim_matrix)
            exp_logits = exp_logits * (1 - eye_mask)

            # Compute log_prob
            log_prob = sim_matrix - logits_max.detach() - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

            # Calculate mean of log-likelihood over positive pairs
            valid_anchors = pos_mask.sum(1) > 0
            if valid_anchors.sum() > 0:
                mean_log_prob_pos = (pos_mask * log_prob).sum(1)[valid_anchors] / pos_mask.sum(1).clamp(min=1)[valid_anchors]
                part_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
                total_loss += part_loss
                valid_parts += 1

        # Return average loss over valid part types
        if valid_parts > 0:
            return total_loss / valid_parts
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class HypPatchUnConLoss(nn.Module):
    """partco ``PatchUnConLoss`` with the HypCD angle/distance switch."""

    def __init__(self, temperature=0.07, base_temperature=0.07, confidence_threshold=0.7,
                 dynamic_threshold=False, total_epochs=100, class_balance_weight=0.3,
                 hard_negative_weight=0.5, consistency_weight=0.2, hyp_c=0, to_hyp=None):
        super(HypPatchUnConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.confidence_threshold = confidence_threshold
        self.dynamic_threshold = dynamic_threshold
        self.total_epochs = total_epochs
        self.class_balance_weight = class_balance_weight
        self.hard_negative_weight = hard_negative_weight
        self.consistency_weight = consistency_weight
        # [hypartco] hyp_c == 0 -> original cosine (angle) loss;
        #            hyp_c  > 0 -> S = -D_H on to_hyp(-pooled features).
        self.hyp_c = hyp_c
        self.to_hyp = to_hyp
        if self.hyp_c != 0 and self.to_hyp is None:
            raise ValueError('hyp_c > 0 requires a ToPoincare operator (to_hyp)')

    def forward(self, patch_out, patch_label, class_logits, mask_lab, epoch=None):
        """
        Args:
            patch_out: patch features of shape [B, 256, feat_dim]
            patch_label: patch labels of shape [B, 16, 16]
            class_logits: class predictions from model [B, num_classes]
            mask_lab: boolean mask indicating which samples are labeled [B]
            epoch: current training epoch (used for dynamic thresholding)
        Returns:
            loss: improved contrastive loss using only high-confidence unlabeled samples
        """
        batch_size = patch_out.size(0)
        device = patch_out.device

        # Check if we have any unlabeled data
        if (~mask_lab).sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Potentially adjust threshold based on training progress
        threshold = self.confidence_threshold
        if self.dynamic_threshold and epoch is not None:
            # Gradually decrease threshold from 0.8 to 0.3 or 0.6 over total training epochs
            threshold = 0.8 - 0.5 * min(1.0, epoch / self.total_epochs)

        # Reshape patch_label from [B, 16, 16] to [B, 256]
        # patch_label_flat = patch_label.reshape(batch_size, 256)
        patch_label_flat = patch_label.reshape(batch_size, -1)

        # Filter out labeled data and get pseudo-labels for unlabeled samples
        with torch.no_grad():
            probs = F.softmax(class_logits, dim=1)
            confidence, pseudo_labels = torch.max(probs, dim=1)

            # Create mask for high-confidence unlabeled samples only
            confidence_mask = torch.zeros_like(mask_lab, dtype=torch.bool)
            confidence_mask[~mask_lab] = confidence[~mask_lab] > threshold

            # Skip if no high-confidence unlabeled samples
            if confidence_mask.sum() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

            # Compute class weights for balancing (inverse frequency)
            unlabeled_pseudo_labels = pseudo_labels[~mask_lab & confidence_mask]
            if unlabeled_pseudo_labels.numel() > 0:
                class_counts = torch.bincount(unlabeled_pseudo_labels)
                class_weights = torch.ones(class_counts.size(0), device=device)
                class_weights = class_weights / class_counts.float()
                class_weights = class_weights / class_weights.sum() * len(class_counts)
            else:
                class_weights = None

        # Dictionary to store features by part type
        part_features_dict = {}
        part_labels_dict = {}
        part_confidence_dict = {}
        sample_indices_dict = {}

        # Process only high-confidence unlabeled samples
        for i in range(batch_size):
            if not confidence_mask[i]:
                continue

            features = patch_out[i]  # [256, feat_dim]
            part_labels = patch_label_flat[i]  # [256]
            category_label = pseudo_labels[i]  # pseudo label for this unlabeled sample
            sample_conf = confidence[i].item()  # confidence for this sample

            # Get unique part labels
            unique_parts = torch.unique(part_labels)
            unique_parts = unique_parts[unique_parts > 0]  # Exclude background (0) if any

            # Average features for each part
            for part in unique_parts:
                part_mask = (part_labels == part)
                if part_mask.sum() > 0:
                    part_features = features[part_mask]
                    avg_feature = torch.mean(part_features, dim=0)  # [feat_dim]

                    # Initialize dictionary entry if not exist
                    part_id = part.item()
                    if part_id not in part_features_dict:
                        part_features_dict[part_id] = []
                        part_labels_dict[part_id] = []
                        part_confidence_dict[part_id] = []
                        sample_indices_dict[part_id] = []

                    # Store pooled feature, pseudo-label, confidence, and sample index
                    part_features_dict[part_id].append(avg_feature)
                    part_labels_dict[part_id].append(category_label)
                    part_confidence_dict[part_id].append(sample_conf)
                    sample_indices_dict[part_id].append(i)

        # Skip if no valid parts found
        if not part_features_dict:
            return torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = 0.0
        valid_parts = 0

        # Cross-part consistency loss (for same sample, different parts should have consistent predictions)
        consistency_loss = 0.0
        num_consistency_pairs = 0

        # First, organize features by sample index for cross-part consistency
        sample_parts = {}
        for part_id in part_features_dict:
            features = part_features_dict[part_id]
            indices = sample_indices_dict[part_id]

            for idx, (feat, sample_idx) in enumerate(zip(features, indices)):
                if sample_idx not in sample_parts:
                    sample_parts[sample_idx] = []
                sample_parts[sample_idx].append((part_id, idx))

        # Process each part type separately
        for part_id in part_features_dict:
            features = part_features_dict[part_id]
            labels = part_labels_dict[part_id]
            confidences = part_confidence_dict[part_id]

            # Skip if we don't have enough features for this part type
            if len(features) <= 1:
                continue

            # Stack features, labels, and confidences
            features = torch.stack(features)  # [N, feat_dim]
            labels = torch.stack(labels)      # [N]
            confidences = torch.tensor(confidences, device=device)  # [N]

            # Apply class balancing weights if available
            if class_weights is not None and self.class_balance_weight > 0:
                # Get weights for each sample based on its pseudo-label
                sample_weights = torch.ones_like(confidences)
                for i, label in enumerate(labels):
                    if label < len(class_weights):
                        sample_weights[i] = 1.0 + self.class_balance_weight * (class_weights[label] - 1.0)

                # Combine confidence weights and class balance weights
                combined_weights = confidences * sample_weights
                combined_weights = combined_weights / combined_weights.sum() * len(combined_weights)
            else:
                # Use just the confidence as weight
                combined_weights = confidences / confidences.sum() * len(confidences)

            # Check if we have at least one positive pair (same category)
            unique_labels, counts = torch.unique(labels, return_counts=True)
            if (counts > 1).sum() == 0:
                continue  # No category appears more than once for this part

            # [hypartco] angle vs. hyperbolic-distance similarity (HypCD Eq. 8-10)
            if self.hyp_c == 0:
                # Normalize features
                features = F.normalize(features, dim=1)

                # Compute similarity matrix
                sim_matrix = torch.mm(features, features.transpose(0, 1)) / self.temperature
            else:
                features = self.to_hyp(features)
                sim_matrix = -dist_matrix(features, features, c=self.hyp_c) / self.temperature

            # Create mask based on labels
            labels_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # [N, N]

            # Remove self-contrast
            eye_mask = torch.eye(labels_equal.size(0), device=device)
            pos_mask = labels_equal - eye_mask
            neg_mask = 1.0 - labels_equal

            # Check if we have any positive pairs
            if pos_mask.sum() == 0:
                continue

            # For numerical stability
            logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - logits_max.detach()

            # Compute exp_logits and mask out self-similarity
            exp_logits = torch.exp(sim_matrix)
            exp_logits = exp_logits * (1 - eye_mask)

            # Hard negative mining - find samples with high similarity but different labels
            if self.hard_negative_weight > 0:
                # Hard negatives: high similarity but different class
                hard_negatives = sim_matrix * neg_mask
                # Get top-k hard negatives per sample
                k = max(1, int(0.2 * (len(features) - 1)))  # Use top 20% as hard negatives
                hard_neg_sim, _ = torch.topk(hard_negatives, k=k, dim=1)
                hard_negative_factor = self.hard_negative_weight * hard_neg_sim.mean()
            else:
                hard_negative_factor = 0.0

            # Compute log_prob
            log_prob = sim_matrix - logits_max.detach() - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

            # Calculate mean of log-likelihood over positive pairs
            valid_anchors = pos_mask.sum(1) > 0
            if valid_anchors.sum() > 0:
                mean_log_prob_pos = (pos_mask * log_prob).sum(1)[valid_anchors] / pos_mask.sum(1).clamp(min=1)[valid_anchors]

                # Weight the positive pairs by combined confidence and class weights
                anchor_weights = combined_weights[valid_anchors]
                weighted_mean_log_prob_pos = mean_log_prob_pos * anchor_weights
                weighted_mean_log_prob_pos = weighted_mean_log_prob_pos.sum() / anchor_weights.sum()

                # Final part loss with hard negative weighting
                part_loss = -(self.temperature / self.base_temperature) * weighted_mean_log_prob_pos
                part_loss = part_loss + hard_negative_factor

                total_loss += part_loss
                valid_parts += 1

        # Compute cross-part consistency loss
        if self.consistency_weight > 0:
            for sample_idx, parts in sample_parts.items():
                if len(parts) > 1:  # At least 2 parts needed for consistency
                    # Get all part features for this sample
                    sample_part_features = []
                    for part_id, idx in parts:
                        feat = part_features_dict[part_id][idx]
                        sample_part_features.append(feat)

                    if len(sample_part_features) > 1:
                        # Stack part features and normalize
                        sample_part_features = torch.stack(sample_part_features)  # [num_parts, feat_dim]
                        # [hypartco] same angle/distance switch as above; the original
                        # consistency term is a raw (untempered) cosine, its distance
                        # counterpart is the raw negative hyperbolic distance.
                        if self.hyp_c == 0:
                            sample_part_features = F.normalize(sample_part_features, dim=1)

                            # All part features from the same object should be consistent
                            sim_matrix = torch.mm(sample_part_features, sample_part_features.transpose(0, 1))
                        else:
                            sample_part_features = self.to_hyp(sample_part_features)
                            sim_matrix = -dist_matrix(sample_part_features, sample_part_features, c=self.hyp_c)

                        # Mask out self-similarity
                        eye_mask = torch.eye(len(sample_part_features), device=device)

                        # Get mean similarity between different parts of same object
                        consistency = (sim_matrix * (1 - eye_mask)).sum() / ((1 - eye_mask).sum() + 1e-8)

                        # Higher similarity means better consistency
                        consistency_loss += -consistency
                        num_consistency_pairs += 1

            if num_consistency_pairs > 0:
                consistency_loss = consistency_loss / num_consistency_pairs
            else:
                consistency_loss = torch.tensor(0.0, device=device)

        # Return average loss over valid part types + consistency regularization
        if valid_parts > 0:
            main_loss = total_loss / valid_parts
            final_loss = main_loss

            if self.consistency_weight > 0 and num_consistency_pairs > 0:
                final_loss = main_loss + self.consistency_weight * consistency_loss

            return final_loss
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)
