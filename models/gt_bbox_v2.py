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
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from models.foreground import _import_roi_align

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =========================================================================== #
#  "Augment first, then crop" support (--gtbbox_mode view)
# =========================================================================== #
class RecordingViewGenerator:
    """Drop-in replacement for ContrastiveLearningViewGenerator that ALSO
    returns the geometric parameters of each view, so the GT box can be mapped
    into view coordinates and the object crop taken FROM THE AUGMENTED VIEW
    (online-cropper style: the crop inherits the view's geometry exactly).

    RNG contract (critical for the deterministic-ab framework): for the
    'imagenet' pipeline [Resize, RandomCrop, RandomHorizontalFlip, <tail...>]
    this class consumes the global RNG stream IDENTICALLY to the plain
    pipeline -- RandomCrop params come from the original
    ``transforms.RandomCrop.get_params`` (2x randint), the flip from one
    ``torch.rand(1)`` draw, and the tail transforms (ColorJitter/ToTensor/
    Normalize) are applied verbatim. The produced image views are therefore
    bit-identical to the non-recording pipeline under the same seed, and the
    000 zero-weight run still reduces exactly to the historical baseline.

    Returns per sample: ``[view_0, ..., view_{n-1}, params]`` with ``params``
    of shape (n_views, 5) = (sx, sy, top, left, flip) per view.
    """

    def __init__(self, base_transform, n_views=2):
        from torchvision import transforms as T
        tf = list(base_transform.transforms)
        assert isinstance(tf[0], T.Resize) and isinstance(tf[1], T.RandomCrop) \
            and isinstance(tf[2], T.RandomHorizontalFlip), \
            'RecordingViewGenerator expects the imagenet pipeline ' \
            '[Resize, RandomCrop, RandomHorizontalFlip, ...]'
        self.resize, self.crop, self.flip = tf[0], tf[1], tf[2]
        self.tail = T.Compose(tf[3:])
        self.n_views = n_views

    def _one(self, x):
        import torch
        from torchvision import transforms as T
        from torchvision.transforms import functional as F
        w0, h0 = x.size
        x = self.resize(x)                          # deterministic, no RNG
        w1, h1 = x.size
        i, j, th, tw = T.RandomCrop.get_params(x, self.crop.size)  # 2x randint
        x = F.crop(x, i, j, th, tw)
        flipped = bool(torch.rand(1) < self.flip.p)                # 1x rand
        if flipped:
            x = F.hflip(x)
        x = self.tail(x)                            # verbatim (incl. ColorJitter draw)
        params = torch.tensor([w1 / w0, h1 / h0, float(i), float(j), float(flipped)])
        return x, params

    def __call__(self, x):
        import torch
        outs = [self._one(x) for _ in range(self.n_views)]
        return [v for v, _ in outs] + [torch.stack([p for _, p in outs], 0)]


def _map_box_to_view(box, params, view_size, min_box):
    """GT box (original-image xyxy) -> view coordinates given (sx, sy, top, left, flip).
    Returns a clamped xyxy box in the view frame, or None if the object is
    (almost) entirely outside this view."""
    sx, sy, top, left, flip = params
    x1, y1, x2, y2 = box
    x1, x2 = x1 * sx - left, x2 * sx - left
    y1, y2 = y1 * sy - top, y2 * sy - top
    if flip >= 0.5:
        x1, x2 = view_size - x2, view_size - x1
    x1c, y1c = max(0.0, x1), max(0.0, y1)
    x2c, y2c = min(float(view_size), x2), min(float(view_size), y2)
    if (x2c - x1c) < min_box or (y2c - y1c) < min_box:
        return None
    return (x1c, y1c, x2c, y2c)


def _sample_rrc_box(domain, gen, scale_min, view_size, ratio=(3. / 4., 4. / 3.), tries=10):
    """RandomResizedCrop-style sub-box INSIDE ``domain`` (xyxy) using the
    private generator ``gen``: area fraction U(scale_min, 1.0) of the domain,
    aspect ratio log-uniform in ``ratio`` (bounds the square-warp to <=4/3,
    i.e. inside the backbone's RandomResizedCrop training distribution)."""
    import torch
    dx1, dy1, dx2, dy2 = domain
    dw, dh = dx2 - dx1, dy2 - dy1
    area = dw * dh
    logr = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(tries):
        a = area * (scale_min + (1.0 - scale_min) * torch.rand(1, generator=gen).item())
        r = math.exp(logr[0] + (logr[1] - logr[0]) * torch.rand(1, generator=gen).item())
        w, h = math.sqrt(a * r), math.sqrt(a / r)
        if w <= dw + 1e-6 and h <= dh + 1e-6:
            x1 = dx1 + (dw - w) * torch.rand(1, generator=gen).item()
            y1 = dy1 + (dh - h) * torch.rand(1, generator=gen).item()
            return (x1, y1, x1 + w, y1 + h)
    # fallback: largest centered box with clamped aspect
    r = min(max(dw / max(dh, 1e-6), ratio[0]), ratio[1])
    w = min(dw, dh * r); h = w / r
    cx, cy = (dx1 + dx2) / 2, (dy1 + dy2) / 2
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


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
                 num_threads: int = None, mode: str = 'orig',
                 scale_min: float = 0.5, box_pad: float = 0.15,
                 bgswap_p: float = 0.0, seed: int = 0):
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
            mode:           'orig' = v1 behaviour (deterministic full-box crop from
                            the ORIGINAL image, shared across views).
                            'view' = augment-first-then-crop: the GT box is mapped
                            into each augmented view (via RecordingViewGenerator
                            params) and a per-view stochastic RandomResizedCrop-style
                            sub-box inside the (padded) mapped box is taken FROM THE
                            VIEW TENSOR with one batched roi_align -- online-cropper
                            style, but guaranteed on-object.
            scale_min:      [view] lower bound of the sub-box area fraction.
            box_pad:        [view] relative context padding around the mapped box.
            bgswap_p:       [view] P2b: probability per row of replacing the crop by
                            a background-swap composite (the view's full GT-box
                            content pasted onto ANOTHER instance's view) -- a hard
                            "same object, different context" positive.
            seed:           seed of the PRIVATE torch.Generator used for all
                            stochastic choices here; the global RNG streams are
                            never touched, so the deterministic-ab property and the
                            000-baseline identity are preserved.
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
        assert mode in ('orig', 'view'), mode
        self.mode = mode
        self.scale_min = float(scale_min)
        self.box_pad = float(box_pad)
        self.bgswap_p = float(bgswap_p)
        self._gen = torch.Generator().manual_seed(int(seed) + 314159)  # private stream
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
    # Full pipeline.
    # mode 'orig': uq_idxs -> object crops from the ORIGINAL images,
    #   replicated per view (v1 behaviour, unchanged).
    # mode 'view': GT boxes are mapped into each augmented view via
    #   ``view_params`` (B, n_views, 5) from RecordingViewGenerator, a per-view
    #   stochastic sub-box is sampled inside the padded mapped box, and all
    #   2B crops are taken from ``images`` with ONE roi_align call -- exactly
    #   the online cropper's mechanism, but box-anchored. Optionally (P2b)
    #   some rows are replaced by background-swap composites.
    # Everything runs under no_grad and only the PRIVATE generator is used.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def __call__(self, uq_idxs, images: torch.Tensor, view_params: torch.Tensor = None) -> torch.Tensor:
        uq = [int(u) for u in uq_idxs]
        if self.mode == 'orig':
            mapper = self._pool.map if self._pool is not None else map
            crops = torch.stack(list(mapper(self._load_crop, uq)), dim=0)   # (B, 3, S, S)
            n_views = max(1, images.shape[0] // len(uq))
            crops = crops.to(device=images.device, dtype=images.dtype)
            return crops.repeat(n_views, 1, 1, 1)                           # (n_views*B, ...)

        # ---------------- mode 'view' ----------------
        assert view_params is not None, \
            "gtbbox_mode='view' needs the params from RecordingViewGenerator"
        B = len(uq)
        n_views = images.shape[0] // B
        S = images.shape[-1]
        vp = view_params.detach().float().cpu()                              # (B, n_views, 5)
        rois, mapped_full = [], []
        for v in range(n_views):
            for b in range(B):
                box = self.entries[uq[b]][1]
                m = None
                if box is not None:
                    m = _map_box_to_view(box, vp[b, v].tolist(), S, self.min_box)
                if m is None:                       # missing box / object outside view
                    m = (0.0, 0.0, float(S), float(S))
                mapped_full.append(m)
                # context padding, clamped to the view
                pw, ph = (m[2] - m[0]) * self.box_pad, (m[3] - m[1]) * self.box_pad
                dom = (max(0.0, m[0] - pw), max(0.0, m[1] - ph),
                       min(float(S), m[2] + pw), min(float(S), m[3] + ph))
                sub = _sample_rrc_box(dom, self._gen, self.scale_min, S)
                rois.append([float(v * B + b), *sub])
        rois = torch.tensor(rois, dtype=images.dtype, device=images.device)
        crops = self._roi_align(images, rois, output_size=(self.out_size,) * 2,
                                spatial_scale=1.0, aligned=True)             # (n_views*B, 3, S, S)

        # ---------------- P2b: background-swap hard positives ----------------
        if self.bgswap_p > 0 and B > 1:
            swap = torch.rand(n_views * B, generator=self._gen) < self.bgswap_p
            for r in torch.nonzero(swap).flatten().tolist():
                v, b = divmod(r, B)
                donor = images[v * B + (b + 1) % B].clone()                  # same-view neighbour
                x1, y1, x2, y2 = mapped_full[r]
                bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
                # aspect-preserving paste size: long side in [0.55, 0.85] * S
                frac = 0.55 + 0.30 * torch.rand(1, generator=self._gen).item()
                sc = frac * S / max(bw, bh)
                tw = max(int(round(bw * sc)), self.out_size // 8)
                th = max(int(round(bh * sc)), self.out_size // 8)
                tw, th = min(tw, S), min(th, S)
                roi = torch.tensor([[float(v * B + b), x1, y1, x2, y2]],
                                   dtype=images.dtype, device=images.device)
                patch = self._roi_align(images, roi, output_size=(th, tw),
                                        spatial_scale=1.0, aligned=True)[0]
                top = int(torch.randint(0, S - th + 1, (1,), generator=self._gen))
                left = int(torch.randint(0, S - tw + 1, (1,), generator=self._gen))
                donor[:, top:top + th, left:left + tw] = patch
                if donor.shape[-1] != self.out_size:                         # S != out_size guard
                    full = torch.tensor([[0.0, 0.0, 0.0, float(S), float(S)]],
                                        dtype=images.dtype, device=images.device)
                    donor = self._roi_align(donor.unsqueeze(0), full,
                                            output_size=(self.out_size,) * 2,
                                            spatial_scale=1.0, aligned=True)[0]
                crops[r] = donor
        return crops