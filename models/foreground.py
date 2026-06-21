"""
Pre-cutting foreground cropper for the object-level branch (HyCoCLIP-style:
crop first, then encode with the *shared* backbone).

The cropper produces exactly ONE foreground box per image with NO learnable
parameters, using ready-made, well-known recipes:

* ``attention``  (DINO ViT, ``model_name == 'v1'``): threshold the ``[CLS]``
  self-attention of the last block to keep a fraction of the attention mass and
  take the bounding box of the resulting mask. This is the recipe from the DINO
  repo (``visualize_attention.py``, Caron et al., 2021).
* ``cls_sim``    (DINOv2 ViT, ``model_name == 'v2'``, which does not expose
  attention weights): use the cosine similarity between the ``[CLS]`` token and
  each patch token of the last layer as a saliency map, then the same
  mask -> bbox routine.

The crop itself is done with ``torchvision.ops.roi_align`` (crop + resize to the
backbone resolution in a single call), so the object image lands in the exact
same input distribution as the full image and therefore the exact same Poincare
feature space after the shared projector.

The module is intentionally small and pluggable: to add a different foreground
source (TokenCut, a saliency network, ground-truth boxes, ...) implement a new
``_saliency`` branch or subclass ``ForegroundCropper`` and override ``saliency``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _import_roi_align():
    try:
        from torchvision.ops import roi_align
        return roi_align
    except Exception as e:  # pragma: no cover - torchvision is a repo dependency
        raise ImportError(
            "ForegroundCropper requires torchvision (torchvision.ops.roi_align). "
            "It is already a dependency of HypCD's data augmentations."
        ) from e


class ForegroundCropper:
    """Turn a batch of images into a batch of single-object foreground crops."""

    def __init__(
        self,
        backbone,
        model_name: str = "v1",
        source: str = "auto",
        keep: float = 0.6,
        box_pad: float = 0.1,
        out_size: int = 224,
        min_box: int = 8,
    ):
        """
        Args:
            backbone:   the shared ViT (same object passed as ``student``).
            model_name: 'v1' (DINO) or 'v2' (DINOv2); selects the default source.
            source:     'auto' | 'attention' | 'cls_sim'.
            keep:       fraction of saliency mass to keep as foreground (DINO uses 0.6).
            box_pad:    relative padding added on each side of the tight box.
            out_size:   side length of the resized crop (backbone input size).
            min_box:    minimum box side in pixels (guards against degenerate boxes).
        """
        self.backbone = backbone
        if source == "auto":
            source = "attention" if model_name == "v1" else "cls_sim"
        assert source in ("attention", "cls_sim"), f"unknown source {source}"
        self.source = source
        self.keep = float(keep)
        self.box_pad = float(box_pad)
        self.out_size = int(out_size)
        self.min_box = int(min_box)
        self._roi_align = _import_roi_align()

    # ------------------------------------------------------------------ #
    # Saliency map  (N, P)  with P = number of patch tokens.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def saliency(self, images: torch.Tensor) -> torch.Tensor:
        # Localisation only: run the backbone in eval mode so DropPath /
        # stochastic depth is disabled. This makes the boxes deterministic and
        # free of training-time noise; the backbone's training state is restored
        # afterwards so the *object encoding* forward still sees DropPath.
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            if self.source == "attention":
                # DINO v1 ViT: last-block CLS->patch attention, averaged over heads.
                out = self.backbone.get_last_selfattention(images)
                attn = out[1] if isinstance(out, (tuple, list)) else out
                sal = attn[:, :, 0, 1:].mean(dim=1)               # (N, P)
            else:
                # DINOv2 (or any ViT exposing forward_features): CLS<->patch cosine.
                feats = self.backbone.forward_features(images)
                cls = F.normalize(feats["x_norm_clstoken"], dim=-1)        # (N, D)
                pat = F.normalize(feats["x_norm_patchtokens"], dim=-1)     # (N, P, D)
                sal = torch.einsum("nd,npd->np", cls, pat).clamp_min(0)    # (N, P)
        finally:
            self.backbone.train(was_training)
        return sal

    # ------------------------------------------------------------------ #
    # Saliency map -> pixel-space boxes (x1, y1, x2, y2).
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _mask_to_boxes(self, sal: torch.Tensor, H: int, W: int) -> torch.Tensor:
        eps = 1e-8
        N, P = sal.shape
        g = int(round(P ** 0.5))
        sal = sal.reshape(N, g * g)

        # Keep the smallest set of patches holding `keep` fraction of the mass
        # (DINO `visualize_attention.py` thresholding). cumsum has no deterministic
        # CUDA kernel, so we run it on CPU: this keeps the foreground boxes
        # reproducible under torch.use_deterministic_algorithms(True) and is cheap
        # (one small no-grad reduction per step).
        prob = sal / sal.sum(dim=-1, keepdim=True).clamp_min(eps)
        vals, idx = torch.sort(prob, dim=-1, descending=True)
        cum = torch.cumsum(vals.cpu(), dim=-1).to(vals.device)
        keep_sorted = cum <= self.keep
        keep_sorted[:, 0] = True  # always keep the strongest patch
        mask = torch.zeros_like(prob, dtype=torch.bool)
        mask.scatter_(1, idx, keep_sorted)
        mask = mask.reshape(N, g, g)

        cell_h, cell_w = H / g, W / g
        pad_h, pad_w = self.box_pad * H, self.box_pad * W
        boxes = sal.new_zeros((N, 4))
        for n in range(N):
            ys, xs = torch.where(mask[n])
            if ys.numel() == 0:
                boxes[n] = torch.tensor([0, 0, W, H], device=sal.device)
                continue
            y1 = ys.min().item() * cell_h - pad_h
            y2 = (ys.max().item() + 1) * cell_h + pad_h
            x1 = xs.min().item() * cell_w - pad_w
            x2 = (xs.max().item() + 1) * cell_w + pad_w
            # clamp to image and enforce a minimum size
            x1 = max(0.0, min(x1, W - self.min_box))
            y1 = max(0.0, min(y1, H - self.min_box))
            x2 = min(float(W), max(x2, x1 + self.min_box))
            y2 = min(float(H), max(y2, y1 + self.min_box))
            boxes[n] = torch.tensor([x1, y1, x2, y2], device=sal.device)
        return boxes

    @torch.no_grad()
    def get_boxes(self, images: torch.Tensor) -> torch.Tensor:
        H, W = images.shape[-2:]
        sal = self.saliency(images)
        return self._mask_to_boxes(sal, H, W)

    # ------------------------------------------------------------------ #
    # Full pipeline: images -> object crops (resized to backbone resolution).
    # Everything runs under no_grad (boxes/pixels are non-differentiable); the
    # downstream ``backbone(crops)`` re-enables gradients to the shared weights.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        boxes = self.get_boxes(images)                                  # (N, 4)
        n = images.shape[0]
        batch_idx = torch.arange(n, device=images.device, dtype=images.dtype).view(-1, 1)
        rois = torch.cat([batch_idx, boxes.to(images.dtype)], dim=1)    # (N, 5)
        crops = self._roi_align(
            images, rois,
            output_size=(self.out_size, self.out_size),
            spatial_scale=1.0, aligned=True,
        )
        return crops
