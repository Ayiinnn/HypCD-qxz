"""
DINOv2 foreground cropper using last-block CLS-to-patch self-attention.

Only the DINOv2 saliency extraction is replaced. Mask thresholding, bounding
box construction, padding, and ROIAlign are inherited unchanged from the
existing DINOv1-style ForegroundCropper.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from models.foreground import ForegroundCropper


class DINOv2ForegroundCropper(ForegroundCropper):
    """Apply the DINOv1 attention-crop recipe to a DINOv2 backbone."""

    def __init__(
        self,
        backbone,
        model_name: str = "v2",
        source: str = "auto",
        keep: float = 0.6,
        box_pad: float = 0.1,
        out_size: int = 224,
        min_box: int = 8,
    ):
        if model_name != "v2":
            raise ValueError(
                f"DINOv2ForegroundCropper only supports model_name='v2', got {model_name!r}."
            )
        if source not in ("auto", "attention"):
            raise ValueError(
                "DINOv2ForegroundCropper always uses last-block attention; "
                f"source must be 'auto' or 'attention', got {source!r}."
            )
        super().__init__(
            backbone=backbone,
            model_name="v2",
            source="attention",
            keep=keep,
            box_pad=box_pad,
            out_size=out_size,
            min_box=min_box,
        )

    @staticmethod
    def _flatten_blocks(backbone) -> List[nn.Module]:
        """Return real transformer blocks for chunked and unchunked DINOv2."""
        if not getattr(backbone, "chunked_blocks", False):
            return list(backbone.blocks)

        blocks: List[nn.Module] = []
        for chunk in backbone.blocks:
            # DINOv2 chunked blocks contain leading Identity placeholders that
            # preserve global block indices; they must not be executed twice.
            blocks.extend(block for block in chunk if not isinstance(block, nn.Identity))
        return blocks

    @torch.no_grad()
    def saliency(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return last-block, head-averaged CLS-to-patch attention, shape (N, P).

        This is the DINOv2 counterpart of:
            get_last_selfattention(images)[:, :, 0, 1:].mean(dim=1)

        DINOv2's attention module does not expose its softmax weights, so the
        CLS row is reconstructed from the unchanged last-block qkv parameters.
        Only the CLS row is materialized; this is mathematically identical to
        taking row zero from the full attention matrix and avoids an O(T^2)
        allocation. Register-token columns are excluded from the patch map.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            if not hasattr(self.backbone, "prepare_tokens_with_masks"):
                raise TypeError(
                    "DINOv2ForegroundCropper expects a DINOv2 backbone exposing "
                    "prepare_tokens_with_masks()."
                )

            blocks = self._flatten_blocks(self.backbone)
            if not blocks:
                raise RuntimeError("DINOv2 backbone contains no transformer blocks.")

            tokens = self.backbone.prepare_tokens_with_masks(images, masks=None)
            for block in blocks[:-1]:
                tokens = block(tokens)

            last_block = blocks[-1]
            if not all(
                hasattr(last_block, name) for name in ("norm1", "attn")
            ) or not all(
                hasattr(last_block.attn, name)
                for name in ("qkv", "num_heads", "scale")
            ):
                raise TypeError(
                    "Unsupported DINOv2 last block: expected norm1 and an "
                    "attention module exposing qkv, num_heads, and scale."
                )

            x = last_block.norm1(tokens)
            batch_size, token_count, embed_dim = x.shape
            num_heads = int(last_block.attn.num_heads)
            if embed_dim % num_heads != 0:
                raise RuntimeError(
                    f"embed_dim={embed_dim} is not divisible by num_heads={num_heads}."
                )

            head_dim = embed_dim // num_heads
            qkv = (
                last_block.attn.qkv(x)
                .reshape(batch_size, token_count, 3, num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q_cls = qkv[0][:, :, :1] * last_block.attn.scale
            keys = qkv[1]

            cls_attention = torch.matmul(
                q_cls, keys.transpose(-2, -1)
            ).softmax(dim=-1)
            if hasattr(last_block.attn, "attn_drop"):
                cls_attention = last_block.attn.attn_drop(cls_attention)

            num_register_tokens = int(
                getattr(self.backbone, "num_register_tokens", 0)
            )
            patch_start = 1 + num_register_tokens
            saliency = cls_attention[:, :, 0, patch_start:].mean(dim=1)

            patch_count = saliency.shape[-1]
            grid_size = int(round(patch_count**0.5))
            if grid_size * grid_size != patch_count:
                raise RuntimeError(
                    "The inherited bbox routine requires a square patch grid, "
                    f"but received {patch_count} patch tokens."
                )

            return saliency
        finally:
            self.backbone.train(was_training)
