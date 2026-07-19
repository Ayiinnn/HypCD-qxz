from torchvision import transforms

import torch

# [hypartco-r5] three-view support ------------------------------------------
import random

import numpy as np
from torchvision.transforms.functional import hflip


class PartCoThreeView:
    """[hypartco-r5] Transform container for the three-view A+B contract.

    views = [a_view_1, a_view_2, part_view]:
      * the two A views use ``a_transform`` -- byte-identical to HypCD's
        original train transform (Resize(size/crop_pct) -> RandomCrop ->
        RandomHorizontalFlip -> ColorJitter); their randomness lives inside
        the transform, exactly as in the A baseline, so A's input
        distribution is restored (r1-r4 replaced A's view 0 with the aligned
        part view, costing ~-2.8pt by ep29 vs the baseline);
      * the part view uses ``part_transform`` (PartCo's aligned
        Resize+ColorJitter) and receives PartCo's dataset-level sync flip
        together with the part-label grid (see ``three_view_apply``).
    """

    n_views = 3

    def __init__(self, a_transform, part_transform):
        self.a_transform = a_transform
        self.part_transform = part_transform


def three_view_apply(tv, img, patch_label, random_hflip=True):
    """[hypartco-r5] Apply ``PartCoThreeView`` with the sync flip on the part
    view only.  Returns ``([a1, a2, part_view], patch_label_tensor)``.

    Call order matters: the A views are drawn from the raw image BEFORE the
    sync flip (their own RandomHorizontalFlip supplies flip augmentation, as
    in the A baseline); the flip here exists solely to keep the part-label
    grid aligned with the part view, PartCo's original invariant.
    """
    a_views = [tv.a_transform(img), tv.a_transform(img)]
    if random_hflip and random.choice([0.0, 1.0]) == 1.0:
        img = hflip(img)
        patch_label = hflip(patch_label)
    patch_label = torch.tensor(np.array(patch_label))
    return a_views + [tv.part_transform(img)], patch_label
# ---------------------------------------------------------------------------


def get_transform(transform_type='imagenet', image_size=32, args=None):

    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        # we removed random crop from the train transform
        # to avoid inconsistent image sizes towards the patch label
        train_transform = transforms.Compose([
            transforms.Resize([image_size, image_size], interpolation),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        contrastive_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
    
    else:

        raise NotImplementedError

    return (train_transform, contrastive_transform, test_transform)