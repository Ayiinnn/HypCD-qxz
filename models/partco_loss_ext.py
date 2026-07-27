"""[hab] Extension losses for the method stage (round M).  models/partco_loss.py
stays a verbatim PartCo copy; everything new lives here.

PatchUnConLossAllMin
--------------------
APL's all-min contrastive objective (Dai et al., "Adaptive Part Learning for
Fine-Grained GCD", Eq. 5) transplanted onto PartCo's pooled-part structure:

    L = -1/N * sum_a log [ sum_{b in S+(a)} sum_{t in C_ab} exp(sim_t / tau)
                           / sum_{b in S-(a)} exp( min_{t in C_ab} sim_t / tau) ]

* positives (same pseudo-label): ALL corresponding parts attract;
* negatives (different pseudo-label): ONLY the least-similar corresponding
  part repels -- the remaining parts stay unconstrained, so shareable parts
  are not pushed apart by (possibly wrong) pseudo-labels.  Mechanistically
  this shrinks the false-negative repulsion channel, the toxicity path the
  R1/RO 2x2 identified (label errors x dose dominate the steady-state cost).

Kept identical to PartCo's PatchUnConLoss (single-variable change = the
aggregation structure): confidence gate with the same dynamic threshold,
background part-id 0 excluded, per-anchor confidence weighting, the 0.2 *
intra-image consistency term, pooled-mean features compared by raw inner
product (PartCo's convention -- no re-normalization after pooling).  Dropped:
the hard-negative term (measured inert in the verbatim implementation) and
the class-balance reweighting (tied to the per-part SupCon layout).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchUnConLossAllMin(nn.Module):
    def __init__(self, temperature=0.07, confidence_threshold=0.7,
                 dynamic_threshold=False, total_epochs=100,
                 consistency_weight=0.2):
        super().__init__()
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.dynamic_threshold = dynamic_threshold
        self.total_epochs = total_epochs
        self.consistency_weight = consistency_weight

    def forward(self, patch_out, patch_label, class_logits, mask_lab, epoch=None):
        B, device = patch_out.size(0), patch_out.device
        if (~mask_lab).sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        threshold = self.confidence_threshold
        if self.dynamic_threshold and epoch is not None:
            threshold = 0.8 - 0.5 * min(1.0, epoch / self.total_epochs)

        patch_label_flat = patch_label.reshape(B, -1)
        with torch.no_grad():
            probs = F.softmax(class_logits, dim=1)
            confidence, pseudo_labels = torch.max(probs, dim=1)
            confidence_mask = torch.zeros_like(mask_lab, dtype=torch.bool)
            confidence_mask[~mask_lab] = confidence[~mask_lab] > threshold
        if confidence_mask.sum() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        idxs = torch.nonzero(confidence_mask, as_tuple=False).flatten()
        T_max = int(patch_label_flat[idxs].max().item())
        if T_max < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # pooled parts [N, T, D] + validity [N, T]  (part ids 1..T_max; 0 = bg)
        N, D = len(idxs), patch_out.size(-1)
        parts = patch_out.new_zeros(N, T_max, D)
        valid = torch.zeros(N, T_max, dtype=torch.bool, device=device)
        for n, i in enumerate(idxs.tolist()):
            pl = patch_label_flat[i]
            for t in torch.unique(pl):
                t = int(t.item())
                if t <= 0:
                    continue
                m = (pl == t)
                parts[n, t - 1] = patch_out[i][m].mean(dim=0)
                valid[n, t - 1] = True

        pl_sel = pseudo_labels[idxs]
        conf_sel = confidence[idxs]
        same = pl_sel.unsqueeze(0) == pl_sel.unsqueeze(1)                       # [N, N]
        eye = torch.eye(N, dtype=torch.bool, device=device)
        common = valid.unsqueeze(1) & valid.unsqueeze(0)                        # [N, N, T]

        # part-wise inner-product sims (PartCo convention), masked to common parts
        sim = torch.einsum('atd,btd->abt', parts, parts) / self.temperature     # [N, N, T]
        NEG = torch.finfo(sim.dtype).min
        sim_c = sim.masked_fill(~common, NEG)

        # numerical anchor (detached row-max over everything in play)
        row_max = sim_c.reshape(N, -1).max(dim=1, keepdim=True).values.detach() # [N, 1]
        pos_exp = torch.exp((sim_c - row_max.unsqueeze(-1)).clamp_min(-60.0))
        pos_exp = pos_exp * common.float()

        pos_pair = same & ~eye
        neg_pair = ~same
        has_common = common.any(dim=-1)

        # numerator: sum over positive images and ALL their common parts
        numer = (pos_exp * pos_pair.unsqueeze(-1).float()).sum(dim=(1, 2))      # [N]
        # denominator: each negative image contributes ONLY its min-sim part
        min_sim = sim.masked_fill(~common, 1e9).min(dim=-1).values              # [N, N]
        neg_ok = neg_pair & has_common
        neg_exp = torch.exp((min_sim - row_max).clamp(min=-60.0, max=0.0)) * neg_ok.float()
        denom = neg_exp.sum(dim=1)                                              # [N]

        anchor_ok = (numer > 0) & (denom > 0)
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        if anchor_ok.any():
            log_ratio = torch.log(numer[anchor_ok] + 1e-8) - torch.log(denom[anchor_ok] + 1e-8)
            w = conf_sel[anchor_ok]
            loss = loss + (-(w * log_ratio).sum() / w.sum().clamp_min(1e-8))

        # intra-image consistency (verbatim spirit of the original 0.2 term):
        # parts of one image should stay coherent.
        if self.consistency_weight > 0:
            cons_vals = []
            for n in range(N):
                v = valid[n]
                if v.sum() >= 2:
                    P = parts[n][v]
                    S = P @ P.t()
                    off = ~torch.eye(len(P), dtype=torch.bool, device=device)
                    cons_vals.append(S[off].mean())
            if cons_vals:
                loss = loss - self.consistency_weight * torch.stack(cons_vals).mean()
        return loss


class PatchUnConLossMinDenom(nn.Module):
    """[hab-AMv2] CONTROLLED all-min: the fix for the runaway MC transplant
    (F27).  Keeps the SupCon log-softmax normalization (so gradients stay
    self-limited -- gnU is expected to DROP vs the verbatim loss, since the
    negative support shrinks) and applies APL's idea only where it belongs:
    each negative IMAGE enters every denominator through its single
    least-similar corresponding part,

        denom(a,t) = sum_{b pos, t common} exp(S_abt)
                   + sum_{b neg}           exp(min_{t'} S_abt')
        L = - mean_a w_a mean_{(t, b pos)} [ S_abt - log denom(a,t) ]

    so false-negative repulsion (pseudo-diff & GT-same pairs) acts through one
    part per image instead of all parts.  Gate / dynamic threshold /
    background exclusion / per-part renormalize / confidence weighting / the
    0.2 intra-image consistency term follow the verbatim class; the
    class-balance reweight is omitted (tied to the per-part layout).
    Enable only after the D-series verdicts gnFN as the dominant channel.
    """

    def __init__(self, temperature=0.07, base_temperature=0.07, confidence_threshold=0.7,
                 dynamic_threshold=False, total_epochs=100,
                 consistency_weight=0.2):
        super().__init__()
        self.temperature = temperature
        # (T/bT) factor kept for scale parity with the verbatim class, which the
        # trainer constructs WITHOUT base_temperature (so bT stays 0.07 and the
        # main term carries a 0.4 factor at part_temp 0.028 -- F29).
        self.base_temperature = base_temperature
        self.confidence_threshold = confidence_threshold
        self.dynamic_threshold = dynamic_threshold
        self.total_epochs = total_epochs
        self.consistency_weight = consistency_weight

    def forward(self, patch_out, patch_label, class_logits, mask_lab, epoch=None):
        B, device = patch_out.size(0), patch_out.device
        if (~mask_lab).sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        threshold = self.confidence_threshold
        if self.dynamic_threshold and epoch is not None:
            threshold = 0.8 - 0.5 * min(1.0, epoch / self.total_epochs)
        plf = patch_label.reshape(B, -1)
        with torch.no_grad():
            probs = F.softmax(class_logits, dim=1)
            confidence, pseudo = torch.max(probs, dim=1)
            cmask = torch.zeros_like(mask_lab, dtype=torch.bool)
            cmask[~mask_lab] = confidence[~mask_lab] > threshold
        idxs = torch.nonzero(cmask, as_tuple=False).flatten()
        if idxs.numel() < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)
        T_max = int(plf[idxs].max().item())
        if T_max < 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        N, D = len(idxs), patch_out.size(-1)
        parts = patch_out.new_zeros(N, T_max, D)
        valid = torch.zeros(N, T_max, dtype=torch.bool, device=device)
        for n, i in enumerate(idxs.tolist()):
            pl = plf[i]
            for t in torch.unique(pl):
                t = int(t.item())
                if t <= 0:
                    continue
                m = (pl == t)
                parts[n, t - 1] = F.normalize(patch_out[i][m].mean(dim=0), dim=0)
                valid[n, t - 1] = True

        y = pseudo[idxs]
        w = confidence[idxs]
        same = y.unsqueeze(0) == y.unsqueeze(1)
        eye = torch.eye(N, dtype=torch.bool, device=device)
        common = valid.unsqueeze(1) & valid.unsqueeze(0)                    # [N,N,T]
        S = torch.einsum('atd,btd->abt', parts, parts) / self.temperature   # [N,N,T]

        # per-image min over common corresponding parts (negatives' representative)
        m_ab = S.masked_fill(~common, 1e9).min(dim=-1).values               # [N,N]
        neg_pair = (~same) & ~eye & common.any(dim=-1)

        # numerically anchor everything per anchor-image a (detached)
        row_max = torch.maximum(S.masked_fill(~common, -1e9).reshape(N, -1).max(1).values,
                                m_ab.masked_fill(~neg_pair, -1e9).max(1).values).detach()  # [N]
        Sa = S - row_max.view(N, 1, 1)
        ma = m_ab - row_max.view(N, 1)

        pos_pair = same & ~eye                                              # [N,N]
        pos_mask3 = pos_pair.unsqueeze(-1) & common                        # [N,N,T]
        exp_pos = torch.exp(Sa.clamp(min=-60.0)) * pos_mask3.float()        # [N,N,T]
        exp_neg = torch.exp(ma.clamp(min=-60.0)) * neg_pair.float()         # [N,N]
        denom_at = exp_pos.sum(dim=1) + exp_neg.sum(dim=1, keepdim=True)    # [N,T] (+ broadcast neg)

        log_prob = Sa - torch.log(denom_at.unsqueeze(1) + 1e-8)             # [N,N,T]
        n_pos = pos_mask3.float().sum(dim=(1, 2))                           # [N]
        anchor_ok = n_pos > 0
        loss = torch.tensor(0.0, device=device, requires_grad=True)
        if anchor_ok.any():
            mlp = (log_prob * pos_mask3.float()).sum(dim=(1, 2))[anchor_ok] / n_pos[anchor_ok]
            wa = w[anchor_ok]
            loss = loss + (-(self.temperature / self.base_temperature) * (mlp * wa).sum() / wa.sum().clamp_min(1e-8))

        if self.consistency_weight > 0:
            cons_vals = []
            for n in range(N):
                v = valid[n]
                if v.sum() >= 2:
                    P = parts[n][v]
                    Sm = P @ P.t()
                    off = ~torch.eye(len(P), dtype=torch.bool, device=device)
                    cons_vals.append(Sm[off].mean())
            if cons_vals:
                loss = loss - self.consistency_weight * torch.stack(cons_vals).mean()
        return loss


class PatchUnConLossSurgery(nn.Module):
    """[hab-X] Oracle SURGERY variant of the verbatim PatchUnConLoss (pure
diagnostic: GT enters ONLY the two masked lines marked [hab-X]; everything
else -- gate, pooling, weighting, hard-negative, consistency, double-max,
T/bT convention -- is a mechanical copy of the verbatim class, verified by
the oracle-identity smoke test).

drop='fp': false-positive pairs (pseudo-same & GT-diff) stop ATTRACTING.
drop='fn': false-negative terms (pseudo-diff & GT-same) stop REPELLING.

Run at full dose on the H1R trajectory: if ONLY drop_fp removes the ep30
collapse, the FP-attraction channel is causally confirmed (upgrading the
D1 correlation), and its ep30-35 level bounds the recoverable gain.
"""
    def __init__(self, temperature=0.07, base_temperature=0.07, confidence_threshold=0.7,
                 dynamic_threshold=False, total_epochs=100, class_balance_weight=0.3,
                 hard_negative_weight=0.5, consistency_weight=0.2, drop='fp'):
        super(PatchUnConLossSurgery, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.confidence_threshold = confidence_threshold
        self.dynamic_threshold = dynamic_threshold
        self.total_epochs = total_epochs
        self.class_balance_weight = class_balance_weight
        self.hard_negative_weight = hard_negative_weight
        self.consistency_weight = consistency_weight
        assert drop in ('fp', 'fn', 'fp_neutral')
        self.drop = drop
        
    def forward(self, patch_out, patch_label, class_logits, mask_lab, gt_labels, epoch=None):
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
            gt_group = gt_labels[torch.tensor(sample_indices_dict[part_id], device=device)]  # [hab-X] GT per group member (surgery masks only)
            
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
            
            # Normalize features
            features = F.normalize(features, dim=1)
            
            # Compute similarity matrix
            sim_matrix = torch.mm(features, features.transpose(0, 1)) / self.temperature
            
            # Create mask based on labels
            labels_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()  # [N, N]
            
            # Remove self-contrast
            eye_mask = torch.eye(labels_equal.size(0), device=device)
            pos_mask = labels_equal - eye_mask
            neg_mask = 1.0 - labels_equal
            # [hab-X] oracle surgery masks.  GT enters ONLY here.
            gt_same = torch.eq(gt_group.unsqueeze(1), gt_group.unsqueeze(0)).float()
            if self.drop in ('fp', 'fp_neutral'):
                # remove FALSE-POSITIVE pairs from the positive set (pseudo-same
                # & GT-diff no longer attract; they stay in the denominator as
                # ordinary competitors) -- minimal intervention on the pull side.
                pos_mask = pos_mask * gt_same
            
            # Check if we have any positive pairs
            if pos_mask.sum() == 0:
                continue
            
            # For numerical stability
            logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
            sim_matrix = sim_matrix - logits_max.detach()
            
            # Compute exp_logits and mask out self-similarity
            exp_logits = torch.exp(sim_matrix)
            exp_logits = exp_logits * (1 - eye_mask)
            if self.drop == 'fn':
                # [hab-X] remove FALSE-NEGATIVE repulsion: pseudo-diff & GT-same
                # terms leave the denominator (same-class parts are no longer
                # pushed apart) -- minimal intervention on the push side.
                exp_logits = exp_logits * (1.0 - neg_mask * gt_same)
            if self.drop == 'fp_neutral':
                # [hab-X2] remove the ENTIRE FP edge: on top of dropping its
                # attraction (pos_mask above), the pair also leaves the
                # denominator.  Caveat pre-registered: this shares the
                # "denominator emptying" side effect suspected in the XFN
                # anomaly -- read jointly with fp_mass/fn_mass.
                exp_logits = exp_logits * (1.0 - labels_equal * (1.0 - gt_same))
            
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
                        sample_part_features = F.normalize(sample_part_features, dim=1)
                        
                        # All part features from the same object should be consistent
                        sim_matrix = torch.mm(sample_part_features, sample_part_features.transpose(0, 1))
                        
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
