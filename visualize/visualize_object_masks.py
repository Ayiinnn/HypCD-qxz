#!/usr/bin/env python3
"""Visualize the online foreground localization used by the SimGCD object branch.

The object branch in ``train_HypSimGCD_org_det_ab_obj.py`` does not train a
separate segmentation network. It obtains patch saliency from the *shared,
trained* DINO/DINOv2 backbone, keeps the patches carrying ``obj_fg_keep`` of the
saliency mass, derives a padded bounding box, and feeds the corresponding
ROI-aligned crop back through the shared backbone.

For each selected image this script exports:

* ``*_original.png``: the deterministic 224x224 center-cropped model input;
* ``*_object_mask.png``: the thresholded patch mask used to derive the box;
* ``*_masked.png``: the model input with non-mask pixels removed;
* ``*_box_overlay.png``: the exact padded box used by the object branch;
* ``*_object_crop.png``: the exact ROI-aligned object crop;
* ``overview.png``: one row per selected image for quick inspection;
* ``manifest.json``: indices, labels, paths, boxes, and visualization settings.

Only the backbone checkpoint is needed. The hyperbolic projector and classifier
are irrelevant to foreground localization.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.ops import roi_align

# Allow execution from either the repository root or visualize/ itself.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import aircraft_root, car_root, cub_root  # noqa: E402
from data.augmentations import get_transform  # noqa: E402
from data.cub import CustomCub2011  # noqa: E402
from data.fgvc_aircraft import FGVCAircraft  # noqa: E402
from data.stanford_cars import CarsDataset  # noqa: E402
from models.foreground import ForegroundCropper  # noqa: E402

IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export object masks produced by the trained SimGCD obj backbone.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_name", required=True, choices=("cub", "aircraft", "scars"))
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "trainval", "test"),
        help="For CUB/SCars, trainval is treated as train. Aircraft supports all three.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Backbone checkpoint, e.g. checkpoints/model_best_acc_all.pt.",
    )
    parser.add_argument("--model_name", required=True, choices=("v1", "v2"))
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--output_dir", default="visualize/object_masks")
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument(
        "--indices",
        nargs="*",
        type=int,
        default=None,
        help="Exact dataset indices. When supplied, overrides --num_images/--seed.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random sample seed when --indices is omitted.")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--obj_fg_source",
        default="auto",
        choices=("auto", "attention", "cls_sim"),
        help="Use the same foreground source as training: auto -> attention(v1), cls_sim(v2).",
    )
    parser.add_argument(
        "--obj_fg_keep",
        type=float,
        default=0.6,
        help="Fraction of total saliency mass retained as foreground patches.",
    )
    parser.add_argument(
        "--obj_fg_pad",
        type=float,
        default=0.1,
        help="Relative padding added to each side of the tight foreground box.",
    )
    parser.add_argument("--min_box", type=int, default=8)
    parser.add_argument(
        "--background",
        default="black",
        choices=("black", "white"),
        help="Background used in *_masked.png.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.obj_fg_keep <= 1.0:
        raise ValueError("--obj_fg_keep must be in (0, 1].")
    if args.obj_fg_pad < 0.0:
        raise ValueError("--obj_fg_pad must be non-negative.")
    if args.num_images <= 0:
        raise ValueError("--num_images must be positive.")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")


def build_backbone(model_name: str) -> torch.nn.Module:
    # Lazy imports keep --help usable even when only one backbone dependency is installed.
    if model_name == "v1":
        from models import vision_transformer as vits1
        return vits1.__dict__["vit_base"]()
    if model_name == "v2":
        from models import vision_transformer2 as vits2
        return vits2.__dict__["vit_base"]()
    raise ValueError(f"Unsupported model_name: {model_name}")


def unwrap_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    """Accept the repository's raw state dict and common wrapped formats."""
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a dict, got {type(payload).__name__}.")

    state = payload
    for key in ("state_dict", "model", "student", "backbone"):
        value = state.get(key)
        if isinstance(value, dict) and value:
            state = value
            break

    if not state:
        raise ValueError("Checkpoint state dict is empty.")

    # DataParallel/DDP checkpoints commonly prefix every key with "module.".
    if all(isinstance(k, str) and k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    if not all(isinstance(k, str) for k in state.keys()):
        raise TypeError("Checkpoint contains non-string state-dict keys.")
    return state  # type: ignore[return-value]


def load_backbone_checkpoint(model: torch.nn.Module, checkpoint: str) -> None:
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state = unwrap_state_dict(payload)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Backbone checkpoint is incompatible with the selected --model_name. "
            "Use --model_name v1 for DINO-v1 checkpoints and v2 for DINOv2 checkpoints.\n"
            f"Original load error:\n{exc}"
        ) from exc


def build_raw_dataset(dataset_name: str, split: str):
    """Build a raw-PIL dataset using roots imported from config.py."""
    is_train = split in ("train", "trainval")

    if dataset_name == "cub":
        dataset = CustomCub2011(
            root=cub_root,
            train=is_train,
            transform=None,
            target_transform=None,
            download=False,
        )
        root = cub_root
    elif dataset_name == "aircraft":
        aircraft_split = split
        if split == "train":
            aircraft_split = "train"
        dataset = FGVCAircraft(
            root=aircraft_root,
            split=aircraft_split,
            transform=None,
            target_transform=None,
            download=False,
        )
        root = aircraft_root
    elif dataset_name == "scars":
        dataset = CarsDataset(
            data_dir=car_root,
            train=is_train,
            transform=None,
        )
        root = car_root
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset, root


def dataset_item_path(dataset_name: str, dataset, index: int) -> str:
    if dataset_name == "cub":
        row = dataset.data.iloc[index]
        return os.path.join(dataset.root, dataset.base_folder, row.filepath)
    if dataset_name == "aircraft":
        return dataset.samples[index][0]
    if dataset_name == "scars":
        return dataset.data[index]
    return ""


def choose_indices(length: int, explicit: Optional[Sequence[int]], num_images: int, seed: int) -> List[int]:
    if length <= 0:
        raise ValueError("Dataset is empty.")

    if explicit:
        indices = list(explicit)
    else:
        if num_images > length:
            raise ValueError(f"Requested {num_images} images, but dataset has only {length}.")
        indices = random.Random(seed).sample(range(length), num_images)

    bad = [idx for idx in indices if idx < 0 or idx >= length]
    if bad:
        raise IndexError(f"Dataset indices out of range [0, {length - 1}]: {bad}")
    if len(set(indices)) != len(indices):
        raise ValueError("--indices contains duplicates; provide distinct images.")
    return indices


def make_test_transform(image_size: int):
    # Matches train_HypSimGCD_org_det_ab_obj.py: interpolation=3, crop_pct=0.875.
    transform_args = SimpleNamespace(interpolation=3, crop_pct=0.875)
    _, test_transform = get_transform("imagenet", image_size=image_size, args=transform_args)
    return test_transform


def threshold_patch_mask(saliency: torch.Tensor, keep: float) -> torch.Tensor:
    """Exactly reproduce ForegroundCropper._mask_to_boxes thresholding."""
    eps = 1e-8
    n, p = saliency.shape
    grid = int(round(p ** 0.5))
    if grid * grid != p:
        raise ValueError(f"Expected a square patch grid, but received {p} patch tokens.")

    flat = saliency.reshape(n, grid * grid)
    prob = flat / flat.sum(dim=-1, keepdim=True).clamp_min(eps)
    vals, order = torch.sort(prob, dim=-1, descending=True)

    # Keep CPU cumsum to match models/foreground.py under deterministic mode.
    cumulative = torch.cumsum(vals.cpu(), dim=-1).to(vals.device)
    selected_sorted = cumulative <= keep
    selected_sorted[:, 0] = True

    patch_mask = torch.zeros_like(prob, dtype=torch.bool)
    patch_mask.scatter_(1, order, selected_sorted)
    return patch_mask.reshape(n, grid, grid)


def denormalize(batch: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=batch.device, dtype=batch.dtype)
    std = IMAGENET_STD.to(device=batch.device, dtype=batch.dtype)
    return (batch * std + mean).clamp(0.0, 1.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def mask_to_pil(mask: torch.Tensor) -> Image.Image:
    array = mask.detach().cpu().to(torch.uint8).mul(255).numpy()
    return Image.fromarray(array, mode="L")


def draw_box(image: Image.Image, box: Sequence[float], width: int = 3) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    x1, y1, x2, y2 = [float(x) for x in box]
    # PIL rectangle's maximum pixel coordinate is size-1.
    x2 = min(x2, output.width - 1)
    y2 = min(y2, output.height - 1)
    draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=width)
    return output


def make_overview(rows: Sequence[Sequence[Image.Image]], column_titles: Sequence[str]) -> Image.Image:
    if not rows:
        raise ValueError("Cannot create an overview without images.")

    cell_w = max(image.width for row in rows for image in row)
    cell_h = max(image.height for row in rows for image in row)
    header_h = 34
    canvas = Image.new("RGB", (cell_w * len(column_titles), header_h + cell_h * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)

    for col, title in enumerate(column_titles):
        draw.text((col * cell_w + 8, 9), title, fill="black")

    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            rgb = image.convert("RGB")
            x = col_idx * cell_w + (cell_w - rgb.width) // 2
            y = header_h + row_idx * cell_h + (cell_h - rgb.height) // 2
            canvas.paste(rgb, (x, y))
    return canvas


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False. Use --device cpu.")
    device = torch.device(args.device)

    dataset, dataset_root = build_raw_dataset(args.dataset_name, args.split)
    indices = choose_indices(len(dataset), args.indices, args.num_images, args.seed)
    transform = make_test_transform(args.image_size)

    backbone = build_backbone(args.model_name)
    load_backbone_checkpoint(backbone, args.checkpoint)
    backbone = backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    cropper = ForegroundCropper(
        backbone=backbone,
        model_name=args.model_name,
        source=args.obj_fg_source,
        keep=args.obj_fg_keep,
        box_pad=args.obj_fg_pad,
        out_size=args.image_size,
        min_box=args.min_box,
    )

    pil_images: List[Image.Image] = []
    labels: List[int] = []
    paths: List[str] = []
    input_tensors: List[torch.Tensor] = []

    for index in indices:
        image, label, _ = dataset[index]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Dataset item {index} did not return a PIL image.")
        pil_images.append(image.convert("RGB"))
        labels.append(int(label))
        paths.append(dataset_item_path(args.dataset_name, dataset, index))
        input_tensors.append(transform(image.convert("RGB")))

    inputs = torch.stack(input_tensors, dim=0).to(device)

    with torch.inference_mode():
        saliency = cropper.saliency(inputs)
        patch_masks = threshold_patch_mask(saliency, args.obj_fg_keep)
        boxes = cropper._mask_to_boxes(saliency, args.image_size, args.image_size)

        pixel_masks = F.interpolate(
            patch_masks.float().unsqueeze(1),
            size=(args.image_size, args.image_size),
            mode="nearest",
        ).squeeze(1).bool()

        batch_indices = torch.arange(
            inputs.shape[0], device=device, dtype=inputs.dtype
        ).view(-1, 1)
        rois = torch.cat((batch_indices, boxes.to(dtype=inputs.dtype)), dim=1)
        crops = roi_align(
            inputs,
            rois,
            output_size=(args.image_size, args.image_size),
            spatial_scale=1.0,
            aligned=True,
        )

    visible_inputs = denormalize(inputs)
    visible_crops = denormalize(crops)
    background_value = 0.0 if args.background == "black" else 1.0
    masked_inputs = torch.where(
        pixel_masks.unsqueeze(1),
        visible_inputs,
        torch.full_like(visible_inputs, background_value),
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_rows: List[List[Image.Image]] = []
    records = []

    for row_idx, dataset_index in enumerate(indices):
        prefix = f"sample_{row_idx:02d}_idx_{dataset_index:06d}"
        original = tensor_to_pil(visible_inputs[row_idx])
        object_mask = mask_to_pil(pixel_masks[row_idx])
        masked = tensor_to_pil(masked_inputs[row_idx])
        box = boxes[row_idx].detach().cpu().tolist()
        box_overlay = draw_box(original, box)
        object_crop = tensor_to_pil(visible_crops[row_idx])

        original.save(output_dir / f"{prefix}_original.png")
        object_mask.save(output_dir / f"{prefix}_object_mask.png")
        masked.save(output_dir / f"{prefix}_masked.png")
        box_overlay.save(output_dir / f"{prefix}_box_overlay.png")
        object_crop.save(output_dir / f"{prefix}_object_crop.png")

        overview_rows.append([original, object_mask.convert("RGB"), masked, box_overlay, object_crop])
        records.append(
            {
                "row": row_idx,
                "dataset_index": dataset_index,
                "label": labels[row_idx],
                "source_path": paths[row_idx],
                "box_xyxy": [round(float(value), 4) for value in box],
                "foreground_patch_fraction": round(float(patch_masks[row_idx].float().mean().item()), 6),
                "foreground_pixel_fraction": round(float(pixel_masks[row_idx].float().mean().item()), 6),
            }
        )

    overview = make_overview(
        overview_rows,
        ("Original", "Object mask", "Masked", "Branch box", "ROI crop"),
    )
    overview.save(output_dir / "overview.png")

    manifest = {
        "dataset_name": args.dataset_name,
        "dataset_root_from_config": dataset_root,
        "split": args.split,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "model_name": args.model_name,
        "foreground_source": cropper.source,
        "obj_fg_keep": args.obj_fg_keep,
        "obj_fg_pad": args.obj_fg_pad,
        "image_size": args.image_size,
        "background": args.background,
        "samples": records,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset_name} ({args.split}), root={dataset_root}")
    print(f"Foreground source: {cropper.source}, keep={args.obj_fg_keep}, pad={args.obj_fg_pad}")
    print(f"Selected indices: {indices}")
    print(f"Saved {len(indices)} samples to: {output_dir.resolve()}")
    print(f"Overview: {(output_dir / 'overview.png').resolve()}")


if __name__ == "__main__":
    main()