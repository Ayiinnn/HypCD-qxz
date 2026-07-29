#!/usr/bin/env python3
"""Export one CUB sample and the exact crops produced by ``gt_bbox_v2``.

The default path matches training with ``--gtbbox_mode view``:

1. load one random CUB image;
2. produce the recorded ImageNet training views;
3. map the GT box into every augmented view;
4. sample and ROI-align one GT-box crop per view.

Every image is saved separately (no overview/composite image):

* ``original.png``
* ``augmented_view_00.png``, ``augmented_view_01.png``, ...
* ``crop_view_00.png``, ``crop_view_01.png``, ...
* ``manifest.json``
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

# Allow execution from either the repository root or visualize/ itself.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
import data.cub as cub_module  # noqa: E402
from data.augmentations import get_transform  # noqa: E402
from data.cub import CustomCub2011  # noqa: E402
from models.gt_bbox_v2 import (  # noqa: E402
    GTBoxCropper,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RecordingViewGenerator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a random CUB image, its augmented views, and all gt_bbox_v2 crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cub_root", default=config.cub_root)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Exact dataset index. If omitted, one index is sampled using --seed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_views", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument(
        "--scale_min",
        type=float,
        default=0.7,
        help="Minimum sampled crop area relative to the padded, visible GT-box domain.",
    )
    parser.add_argument(
        "--bgswap_p",
        type=float,
        default=0.0,
        help="Background-swap probability passed directly to GTBoxCropper.",
    )
    parser.add_argument("--box_pad", type=float, default=0.15)
    parser.add_argument("--min_box", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--output_dir", default="visualize/gt_bbox_v2_sample")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_views <= 0:
        raise ValueError("--n_views must be positive.")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")
    if not 0.0 < args.scale_min <= 1.0:
        raise ValueError("--scale_min must be in (0, 1].")
    if not 0.0 <= args.bgswap_p <= 1.0:
        raise ValueError("--bgswap_p must be in [0, 1].")
    if args.box_pad < 0.0:
        raise ValueError("--box_pad must be non-negative.")
    if args.min_box <= 0:
        raise ValueError("--min_box must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")


def make_recording_transform(image_size: int, n_views: int) -> RecordingViewGenerator:
    # Matches train_HypCD_org_det_ab_obj_multi_v4.py.
    transform_args = SimpleNamespace(interpolation=3, crop_pct=0.875)
    train_transform, _ = get_transform(
        "imagenet",
        image_size=image_size,
        args=transform_args,
    )
    return RecordingViewGenerator(train_transform, n_views=n_views)


def choose_index(length: int, requested: Optional[int], seed: int) -> int:
    if length <= 0:
        raise ValueError("The selected CUB split is empty.")
    index = random.Random(seed).randrange(length) if requested is None else requested
    if not 0 <= index < length:
        raise IndexError(f"--index must be in [0, {length - 1}], got {index}.")
    return index


def choose_donor_index(length: int, target_index: int, seed: int) -> int:
    """Select the neighbour needed for GTBoxCropper's optional background swap."""
    if length < 2:
        raise ValueError("Background swap needs at least two dataset samples.")
    donor = random.Random(seed + 1).randrange(length - 1)
    return donor + int(donor >= target_index)


def unpack_views(item, n_views: int) -> Tuple[List[torch.Tensor], torch.Tensor, int, int]:
    packed, label, uq_idx = item
    if not isinstance(packed, (list, tuple)) or len(packed) != n_views + 1:
        raise RuntimeError(
            "RecordingViewGenerator must return n_views tensors followed by view_params."
        )
    views = list(packed[:-1])
    params = packed[-1]
    if params.shape != (n_views, 5):
        raise RuntimeError(
            f"Expected view_params shape {(n_views, 5)}, got {tuple(params.shape)}."
        )
    return views, params, int(label), int(uq_idx)


def make_view_major_batch(view_sets: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    """Match training's ``torch.cat(images, dim=0)``: view 0 batch, then view 1."""
    n_views = len(view_sets[0])
    return torch.cat(
        [torch.stack([sample[v] for sample in view_sets], dim=0) for v in range(n_views)],
        dim=0,
    )


def denormalize(image: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device, dtype=image.dtype).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        denormalize(image)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def main() -> None:
    args = parse_args()
    validate_args(args)

    # GTBoxCropper reads cub_root from config at construction time.
    config.cub_root = str(Path(args.cub_root).expanduser())
    cub_module.cub_root = config.cub_root

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = make_recording_transform(args.image_size, args.n_views)
    dataset = CustomCub2011(
        root=config.cub_root,
        train=args.split == "train",
        transform=transform,
        target_transform=None,
        download=False,
    )
    target_index = choose_index(len(dataset), args.index, args.seed)

    # A second sample is fed only when bgswap is enabled because the cropper uses
    # the same-view neighbour as the donor. Only the requested target is exported.
    selected_indices = [target_index]
    donor_index = None
    if args.bgswap_p > 0.0:
        donor_index = choose_donor_index(len(dataset), target_index, args.seed)
        selected_indices.append(donor_index)

    unpacked = [unpack_views(dataset[index], args.n_views) for index in selected_indices]
    view_sets = [entry[0] for entry in unpacked]
    view_params = torch.stack([entry[1] for entry in unpacked], dim=0)
    labels = [entry[2] for entry in unpacked]
    uq_idxs = [entry[3] for entry in unpacked]

    device = torch.device(args.device)
    images = make_view_major_batch(view_sets).to(device)

    # GTBoxCropper only needs the two dataset attributes below. Reusing the full
    # dataset in both slots is harmless: duplicate uq_idx entries are identical.
    merged_dataset = SimpleNamespace(
        labelled_dataset=dataset,
        unlabelled_dataset=dataset,
    )
    cropper = GTBoxCropper(
        merged_dataset=merged_dataset,
        dataset_name="cub",
        out_size=args.image_size,
        min_box=args.min_box,
        mode="view",
        scale_min=args.scale_min,
        box_pad=args.box_pad,
        bgswap_p=args.bgswap_p,
        seed=args.seed,
    )
    crops = cropper(uq_idxs, images, view_params)

    target_uq_idx = uq_idxs[0]
    source_path, gt_box = cropper.entries[target_uq_idx]
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        original = image.convert("RGB")
        original_size = original.size
        original.save(output_dir / "original.png")

    batch_size = len(selected_indices)
    output_files = {"original": "original.png", "augmented_views": [], "crops": []}
    for view_index in range(args.n_views):
        augmented_name = f"augmented_view_{view_index:02d}.png"
        crop_name = f"crop_view_{view_index:02d}.png"

        # View-major row for the target sample (batch position 0).
        row = view_index * batch_size
        tensor_to_pil(images[row]).save(output_dir / augmented_name)
        tensor_to_pil(crops[row]).save(output_dir / crop_name)
        output_files["augmented_views"].append(augmented_name)
        output_files["crops"].append(crop_name)

    manifest = {
        "dataset": "cub",
        "split": args.split,
        "dataset_index": target_index,
        "uq_idx": target_uq_idx,
        "label_zero_based": labels[0],
        "source_path": source_path,
        "class_folder": Path(source_path).parent.name,
        "original_size_wh": list(original_size),
        "gt_box_xyxy": None if gt_box is None else [float(value) for value in gt_box],
        "mode": "view",
        "n_views": args.n_views,
        "image_size": args.image_size,
        "scale_min": args.scale_min,
        "box_pad": args.box_pad,
        "bgswap_p": args.bgswap_p,
        "seed": args.seed,
        "donor_dataset_index": donor_index,
        "outputs": output_files,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(f"CUB sample: index={target_index}, uq_idx={target_uq_idx}, label={labels[0]}")
    print(f"Source: {source_path}")
    print(
        f"Cropper: mode=view, scale_min={args.scale_min}, "
        f"box_pad={args.box_pad}, bgswap_p={args.bgswap_p}"
    )
    print(f"Saved original + {args.n_views} views + {args.n_views} crops to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
