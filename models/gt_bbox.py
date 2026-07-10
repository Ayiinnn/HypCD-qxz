"""
Ground-truth bounding-box cropper for the object-level branch (``--use_gtbbox``).

Drop-in alternative *box source* to :class:`models.foreground.ForegroundCropper`
(the docstring there lists "ground-truth boxes" as an intended extension):
instead of localising the foreground online from the backbone's saliency, the
object crop is taken with the dataset's human-annotated box (HyCoCLIP-style
"crop first, then encode", with GT boxes in place of grounding boxes).

Shape policy (crop -> network input)
------------------------------------
The box region is cropped and *warp-resized to the backbone resolution in a
single ``roi_align`` call* -- same op, same ``output_size``, ``aligned=True``
and bilinear kernel as the online ``ForegroundCropper``. This is the
mainstream treatment of region crops (R-CNN-style RoI warping; HyCoCLIP box
crops are likewise resized to the encoder input), and it keeps the GT crops in
the exact same input distribution as the saliency crops and the full images:
switching ``--use_gtbbox`` changes ONLY where the box comes from, so the two
modes stay directly A/B-comparable. No square-padding / letterboxing is used
anywhere in this pipeline.

Coordinates
-----------
GT boxes live in the ORIGINAL image frame, while the training tensors are
randomly cropped/flipped views whose geometry is not recorded, so the crop is
taken from the original file (re-loaded by ``uq_idx``) and normalised with the
same ImageNet statistics as the train transform. The crop is fully
deterministic (consumes no RNG) and is fed once per view (view-major
``repeat``) so every downstream shape matches the online cropper; the two
views of an instance therefore share one GT crop (they only differ through
DropPath in the shared backbone).

Supported datasets / annotation files (paths derived from ``config.py`` roots):
  cub      -> <cub_root>/CUB_200_2011/bounding_boxes.txt          (img_id x y w h)
  scars    -> <car_root>/devkit/cars_train_annos.mat and
              <car_root>/devkit/cars_test_annos_withlabels.mat    (x1 y1 x2 y2 ... fname)
  aircraft -> <aircraft_root>/data/images_box.txt                 (img_id x1 y1 x2 y2)
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from models.foreground import _import_roi_align

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Per-dataset builders: (subsampled) dataset copy -> {uq_idx: (path, box)}.
# ``box`` is (x1, y1, x2, y2) in original-image pixels, or None if unannotated.
# They read paths off the LIVE dataset objects (``.data`` / ``.samples`` are
# positionally aligned with ``.uq_idxs``), so no split logic is duplicated.
# --------------------------------------------------------------------------- #
def _cub_boxes(dataset):
    import pandas as pd  # already a dependency of data/cub.py
    from config import cub_root

    bbox_file = os.path.join(cub_root, 'CUB_200_2011', 'bounding_boxes.txt')
    boxes = pd.read_csv(bbox_file, sep=' ', names=['img_id', 'x', 'y', 'w', 'h'])
    by_id = {
        int(r.img_id): (float(r.x), float(r.y), float(r.x) + float(r.w), float(r.y) + float(r.h))
        for r in boxes.itertuples(index=False)
    }
    base = os.path.join(cub_root, 'CUB_200_2011', 'images')
    entries = {}
    for pos in range(len(dataset.uq_idxs)):
        row = dataset.data.iloc[pos]
        entries[int(dataset.uq_idxs[pos])] = (
            os.path.join(base, row.filepath), by_id.get(int(row.img_id)))
    return entries


def _scars_boxes(dataset):
    from scipy import io as mat_io  # already a dependency of data/stanford_cars.py
    from config import car_root

    path2box = {}
    for split, mat in (('cars_train', 'cars_train_annos.mat'),
                       ('cars_test', 'cars_test_annos_withlabels.mat')):
        mpath = os.path.join(car_root, 'devkit', mat)
        if not os.path.isfile(mpath):
            continue
        for img_ in mat_io.loadmat(mpath)['annotations'][0]:
            box = tuple(float(img_[i][0][0]) for i in range(4))     # bbox_x1..y2
            fname = str(img_[-1][0])                                # fname is last
            path2box[os.path.normpath(os.path.join(car_root, split, fname))] = box
    entries = {}
    for pos in range(len(dataset.uq_idxs)):
        path = dataset.data[pos]
        entries[int(dataset.uq_idxs[pos])] = (path, path2box.get(os.path.normpath(path)))
    return entries


def _aircraft_boxes(dataset):
    from config import aircraft_root

    bbox_file = os.path.join(aircraft_root, 'data', 'images_box.txt')
    by_id = {}
    with open(bbox_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                by_id[parts[0]] = tuple(float(v) for v in parts[1:5])
    entries = {}
    for pos in range(len(dataset.uq_idxs)):
        path = dataset.samples[pos][0]
        img_id = os.path.splitext(os.path.basename(path))[0]
        entries[int(dataset.uq_idxs[pos])] = (path, by_id.get(img_id))
    return entries


_BUILDERS = {'cub': _cub_boxes, 'scars': _scars_boxes, 'aircraft': _aircraft_boxes}


class GTBoxCropper:
    """Turn a batch of ``uq_idxs`` into a batch of GT-box object crops."""

    def __init__(self, merged_dataset, dataset_name, out_size: int = 224,
                 min_box: int = 8, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 num_threads: int = None):
        """
        Args:
            merged_dataset: the training ``MergedDataset`` (``train_loader.dataset``);
                            its labelled/unlabelled halves provide uq_idx -> path.
            dataset_name:   'cub' | 'scars' | 'aircraft'.
            out_size:       side length of the warped crop (backbone input size).
            min_box:        minimum box side in pixels (same guard as the online
                            cropper against degenerate annotations).
            mean, std:      normalisation stats of the train transform ('imagenet').
            num_threads:    JPEG-decode thread pool size (decode releases the GIL).
        """
        if dataset_name not in _BUILDERS:
            raise ValueError(
                "--use_gtbbox supports datasets {} (got '{}'): no GT box "
                "annotation is wired for this dataset.".format(sorted(_BUILDERS), dataset_name))
        self.entries = {}
        for ds in (merged_dataset.labelled_dataset, merged_dataset.unlabelled_dataset):
            self.entries.update(_BUILDERS[dataset_name](ds))
        n_missing = sum(1 for _, box in self.entries.values() if box is None)
        if n_missing:
            print('[gt-bbox] WARNING: {}/{} training images have no GT box; '
                  'falling back to the full image for those.'.format(n_missing, len(self.entries)))

        self.out_size = int(out_size)
        self.min_box = float(min_box)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self._roi_align = _import_roi_align()
        if num_threads is None:
            num_threads = min(8, os.cpu_count() or 1)
        self._pool = ThreadPoolExecutor(max_workers=num_threads) if num_threads > 1 else None

    # ------------------------------------------------------------------ #
    # One original image -> one normalised (3, S, S) crop (CPU, float32).
    # ------------------------------------------------------------------ #
    def _load_crop(self, uq_idx: int) -> torch.Tensor:
        path, box = self.entries[int(uq_idx)]
        with Image.open(path) as im:
            arr = np.array(im.convert('RGB'), dtype=np.uint8)       # (H, W, 3)
        H, W = arr.shape[:2]
        img = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)

        if box is None:
            x1, y1, x2, y2 = 0.0, 0.0, float(W), float(H)
        else:
            # clamp to the image and enforce a minimum size
            # (same guard as ForegroundCropper._mask_to_boxes).
            x1, y1, x2, y2 = (float(v) for v in box)
            x1 = max(0.0, min(x1, W - self.min_box))
            y1 = max(0.0, min(y1, H - self.min_box))
            x2 = min(float(W), max(x2, x1 + self.min_box))
            y2 = min(float(H), max(y2, y1 + self.min_box))

        rois = torch.tensor([[0.0, x1, y1, x2, y2]], dtype=img.dtype)
        crop = self._roi_align(
            img.unsqueeze(0), rois,
            output_size=(self.out_size, self.out_size),
            spatial_scale=1.0, aligned=True,
        )[0]
        return (crop - self.mean) / self.std

    # ------------------------------------------------------------------ #
    # Full pipeline: uq_idxs -> object crops, replicated per view so the
    # output aligns row-by-row with ``images = torch.cat(views)`` (view-major).
    # Everything runs under no_grad, exactly like the online cropper.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, uq_idxs, images: torch.Tensor) -> torch.Tensor:
        uq = [int(u) for u in uq_idxs]
        mapper = self._pool.map if self._pool is not None else map
        crops = torch.stack(list(mapper(self._load_crop, uq)), dim=0)   # (B, 3, S, S)
        n_views = max(1, images.shape[0] // len(uq))
        crops = crops.to(device=images.device, dtype=images.dtype)
        return crops.repeat(n_views, 1, 1, 1)                           # (n_views*B, ...)