#!/usr/bin/env python3
"""
Select one visually similar but cross-class image pair from CUB, Stanford Cars,
and FGVC-Aircraft.

The default search is restricted to adjacent class IDs (|y_i - y_j| == 1).
Within that constraint, the pair with the largest cosine similarity between
lightweight CPU image descriptors is exported. No GPU, PyTorch, or pretrained
model is used.

Run from the repository root, for example:

    python visualize/select_similar_cross_class_pairs.py

Use all different-class pairs instead of adjacent classes:

    python visualize/select_similar_cross_class_pairs.py \
        --class-gap 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import io as mat_io


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import aircraft_root, car_root, cub_root


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    class_name: str


def _read_two_column_file(path: Path) -> Dict[int, str]:
    result: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            key, value = line.rstrip("\n").split(maxsplit=1)
            result[int(key)] = value
    return result


def load_cub_records(root: Path) -> List[ImageRecord]:
    base = root / "CUB_200_2011"
    if not base.is_dir():
        base = root

    image_names = _read_two_column_file(base / "images.txt")
    image_labels = {
        image_id: int(label)
        for image_id, label in _read_two_column_file(
            base / "image_class_labels.txt"
        ).items()
    }
    class_names = _read_two_column_file(base / "classes.txt")

    records = []
    for image_id in sorted(image_names):
        label_one_based = image_labels[image_id]
        records.append(
            ImageRecord(
                path=str(base / "images" / image_names[image_id]),
                label=label_one_based - 1,
                class_name=class_names[label_one_based]
                .split(".", maxsplit=1)[-1]
                .replace("_", " "),
            )
        )
    return records


def _mat_string(value) -> str:
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.flat[0]
    return str(value)


def load_scars_records(root: Path) -> List[ImageRecord]:
    devkit = root / "devkit"
    class_names_path = devkit / "cars_meta.mat"
    if not class_names_path.is_file():
        class_names_path = root / "cars_meta.mat"

    class_names_raw = mat_io.loadmat(class_names_path)["class_names"][0]
    class_names = [_mat_string(value) for value in class_names_raw]

    split_specs = [
        (
            devkit / "cars_train_annos.mat",
            root / "cars_train",
        ),
        (
            devkit / "cars_test_annos_withlabels.mat",
            root / "cars_test",
        ),
    ]

    records = []
    for annotations_path, image_dir in split_specs:
        annotations = mat_io.loadmat(annotations_path)["annotations"][0]
        for annotation in annotations:
            label_one_based = int(np.asarray(annotation[4]).squeeze())
            filename = _mat_string(annotation[5])
            records.append(
                ImageRecord(
                    path=str(image_dir / filename),
                    label=label_one_based - 1,
                    class_name=class_names[label_one_based - 1],
                )
            )
    return records


def _read_aircraft_split(path: Path) -> List[Tuple[str, str]]:
    items = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            image_id, class_name = line.rstrip("\n").split(maxsplit=1)
            items.append((image_id, class_name))
    return items


def load_aircraft_records(root: Path) -> List[ImageRecord]:
    data_dir = root / "data"
    items = []
    for split in ("trainval", "test"):
        items.extend(
            _read_aircraft_split(data_dir / f"images_variant_{split}.txt")
        )

    # Matches the alphabetical indexing produced by np.unique in the project.
    class_names = sorted({class_name for _, class_name in items})
    class_to_idx = {name: index for index, name in enumerate(class_names)}

    return [
        ImageRecord(
            path=str(data_dir / "images" / f"{image_id}.jpg"),
            label=class_to_idx[class_name],
            class_name=class_name,
        )
        for image_id, class_name in items
    ]


def validate_records(dataset_name: str, records: Sequence[ImageRecord]) -> None:
    if not records:
        raise RuntimeError(f"{dataset_name}: no images were found")
    missing = [record.path for record in records if not Path(record.path).is_file()]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"{dataset_name}: {len(missing)} image files are missing; first paths:\n"
            f"{preview}"
        )
    if len({record.label for record in records}) < 2:
        raise RuntimeError(f"{dataset_name}: at least two classes are required")


def subsample_per_class(
    records: Sequence[ImageRecord], max_per_class: int, seed: int
) -> List[ImageRecord]:
    if max_per_class <= 0:
        return list(records)

    by_class: Dict[int, List[ImageRecord]] = defaultdict(list)
    for record in records:
        by_class[record.label].append(record)

    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(by_class):
        candidates = sorted(by_class[label], key=lambda record: record.path)
        if len(candidates) > max_per_class:
            indices = np.sort(
                rng.choice(len(candidates), size=max_per_class, replace=False)
            )
            candidates = [candidates[index] for index in indices]
        selected.extend(candidates)
    return selected


def _normalized(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32, copy=False).reshape(-1)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def image_descriptor(path: str) -> np.ndarray:
    """A lightweight descriptor combining color, layout, and edge structure."""
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (64, 64),
            method=Image.Resampling.BILINEAR,
        )

    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )
    grad_y, grad_x = np.gradient(gray)
    edges = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge_scale = max(float(np.percentile(edges, 95)), 1e-6)
    edges = np.clip(edges / edge_scale, 0.0, 1.0)

    color_layout = np.asarray(
        image.resize((12, 12), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    gray_layout = np.asarray(
        Image.fromarray(np.uint8(gray * 255.0)).resize(
            (24, 24), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    edge_layout = np.asarray(
        Image.fromarray(np.uint8(edges * 255.0)).resize(
            (24, 24), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    color_histogram = np.concatenate(
        [
            np.histogram(rgb[:, :, channel], bins=16, range=(0.0, 1.0))[0]
            for channel in range(3)
        ]
    ).astype(np.float32)

    descriptor = np.concatenate(
        [
            0.35 * _normalized(color_layout),
            0.25 * _normalized(gray_layout),
            0.30 * _normalized(edge_layout),
            0.10 * _normalized(color_histogram),
        ]
    )
    return _normalized(descriptor)


def extract_features(
    records: Sequence[ImageRecord], num_workers: int
) -> np.ndarray:
    paths = [record.path for record in records]
    if num_workers == 0:
        features = [image_descriptor(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            features = list(executor.map(image_descriptor, paths))
    return np.stack(features).astype(np.float32, copy=False)


def find_best_pair(
    features: np.ndarray,
    records: Sequence[ImageRecord],
    class_gap: int,
    query_chunk_size: int,
) -> Tuple[int, int, float]:
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    all_indices = np.arange(len(records))

    best_score = float("-inf")
    best_i = -1
    best_j = -1

    for start in range(0, len(records), query_chunk_size):
        end = min(start + query_chunk_size, len(records))
        similarities = features[start:end] @ features.T

        query_labels = labels[start:end, None]
        valid = query_labels != labels[None, :]
        if class_gap > 0:
            valid &= np.abs(query_labels - labels[None, :]) <= class_gap

        # Keep only i < j, avoiding self-pairs and symmetric duplicates.
        query_indices = all_indices[start:end, None]
        valid &= query_indices < all_indices[None, :]
        similarities[~valid] = -np.inf

        flat_index = int(np.argmax(similarities))
        local_i = flat_index // similarities.shape[1]
        j = flat_index % similarities.shape[1]
        score = float(similarities[local_i, j])
        if score > best_score:
            best_score = score
            best_i = start + local_i
            best_j = j

    if best_i < 0 or not np.isfinite(best_score):
        constraint = (
            "any different classes"
            if class_gap == 0
            else f"different classes with ID gap <= {class_gap}"
        )
        raise RuntimeError(f"No valid pair found under constraint: {constraint}")
    return best_i, best_j, best_score


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return text.strip("_") or "class"


def _preview_panel(image: Image.Image, label: str) -> Image.Image:
    canvas_width, image_height, caption_height = 560, 420, 70
    contained = ImageOps.contain(
        image.convert("RGB"),
        (canvas_width, image_height),
        method=Image.Resampling.LANCZOS,
    )
    panel = Image.new(
        "RGB", (canvas_width, image_height + caption_height), "white"
    )
    x = (canvas_width - contained.width) // 2
    y = (image_height - contained.height) // 2
    panel.paste(contained, (x, y))
    ImageDraw.Draw(panel).multiline_text(
        (12, image_height + 10),
        label,
        fill="black",
        spacing=4,
    )
    return panel


def export_pair(
    dataset_name: str,
    first: ImageRecord,
    second: ImageRecord,
    similarity: float,
    output_root: Path,
    class_gap: int,
) -> Path:
    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    first_name = f"image_1_class_{first.label:03d}_{safe_name(first.class_name)}.jpg"
    second_name = (
        f"image_2_class_{second.label:03d}_{safe_name(second.class_name)}.jpg"
    )

    with Image.open(first.path) as image:
        first_image = image.convert("RGB")
        first_image.save(output_dir / first_name, quality=95)
    with Image.open(second.path) as image:
        second_image = image.convert("RGB")
        second_image.save(output_dir / second_name, quality=95)

    panel_1 = _preview_panel(
        first_image, f"Class {first.label}: {first.class_name}"
    )
    panel_2 = _preview_panel(
        second_image, f"Class {second.label}: {second.class_name}"
    )
    pair_preview = Image.new(
        "RGB", (panel_1.width + panel_2.width, panel_1.height), "white"
    )
    pair_preview.paste(panel_1, (0, 0))
    pair_preview.paste(panel_2, (panel_1.width, 0))
    pair_preview.save(output_dir / "pair.jpg", quality=95)

    manifest = {
        "dataset": dataset_name,
        "cosine_similarity": similarity,
        "class_gap_constraint": class_gap,
        "image_1": {
            **asdict(first),
            "exported_file": first_name,
        },
        "image_2": {
            **asdict(second),
            "exported_file": second_name,
        },
        "preview_file": "pair.jpg",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find a visually similar cross-class image pair in CUB, Stanford "
            "Cars, and FGVC-Aircraft using lightweight CPU descriptors."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["cub", "scars", "aircraft"],
        choices=["cub", "scar", "scars", "aircraft"],
    )
    parser.add_argument("--cub-root", type=Path, default=Path(cub_root))
    parser.add_argument("--scars-root", type=Path, default=Path(car_root))
    parser.add_argument(
        "--aircraft-root", type=Path, default=Path(aircraft_root)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "visualize" / "similar_cross_class_pairs",
    )
    parser.add_argument(
        "--class-gap",
        type=int,
        default=1,
        help=(
            "Maximum class-ID difference. The default 1 restricts the search "
            "to adjacent classes; 0 allows any two different classes."
        ),
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=20,
        help="Images sampled per class before search; 0 uses every image.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="CPU image-loading threads; 0 disables threading.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=512)
    args = parser.parse_args()

    if args.class_gap < 0:
        parser.error("--class-gap must be >= 0")
    if args.max_per_class < 0:
        parser.error("--max-per-class must be >= 0")
    if args.query_chunk_size <= 0:
        parser.error("--query-chunk-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    loaders = {
        "cub": lambda: load_cub_records(args.cub_root),
        "scars": lambda: load_scars_records(args.scars_root),
        "aircraft": lambda: load_aircraft_records(args.aircraft_root),
    }
    dataset_names = []
    for name in args.datasets:
        normalized = "scars" if name == "scar" else name
        if normalized not in dataset_names:
            dataset_names.append(normalized)

    summaries = []
    for offset, dataset_name in enumerate(dataset_names):
        all_records = loaders[dataset_name]()
        validate_records(dataset_name, all_records)
        records = subsample_per_class(
            all_records,
            max_per_class=args.max_per_class,
            seed=args.seed + offset,
        )
        print(
            f"[{dataset_name}] computing CPU descriptors for {len(records)} "
            f"images from {len({record.label for record in records})} classes ..."
        )
        features = extract_features(
            records=records,
            num_workers=args.num_workers,
        )
        first_idx, second_idx, similarity = find_best_pair(
            features=features,
            records=records,
            class_gap=args.class_gap,
            query_chunk_size=args.query_chunk_size,
        )
        first, second = records[first_idx], records[second_idx]
        manifest_path = export_pair(
            dataset_name=dataset_name,
            first=first,
            second=second,
            similarity=similarity,
            output_root=args.output_dir,
            class_gap=args.class_gap,
        )
        summaries.append(
            {
                "dataset": dataset_name,
                "cosine_similarity": similarity,
                "class_1": f"{first.label}: {first.class_name}",
                "class_2": f"{second.label}: {second.class_name}",
                "manifest": str(manifest_path),
            }
        )
        print(
            f"[{dataset_name}] {first.class_name} <-> {second.class_name}, "
            f"cosine={similarity:.6f}"
        )
        print(f"[{dataset_name}] saved to {manifest_path.parent}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
