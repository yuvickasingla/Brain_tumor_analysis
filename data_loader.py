import os, shutil, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import config
import utils



def get_transforms(augment: bool = True):
   
    normalise = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),           # useful for MRI
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            normalise,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            normalise,
        ])

    val_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        normalise,
    ])

    return train_tf, val_tf


def _check_brisc_exists():
    """Raise a clear error if the user hasn't downloaded BRISC yet."""
    if not os.path.isdir(config.BRISC_TRAIN_DIR):
        raise FileNotFoundError(
            f"\n[ERROR] BRISC2025 dataset not found at:\n"
            f"  {config.BRISC_TRAIN_DIR}\n\n"
            f"Download steps:\n"
            f"  1. Go to https://www.kaggle.com/datasets/briscdataset/brisc2025\n"
            f"  2. Download & extract the archive\n"
            f"  3. Place the 'brisc2025' folder inside:  {config.DATASET_ROOT}\n"
            f"  OR set config.DATASET_ROOT to the correct path.\n"
        )


def _build_weighted_sampler(dataset):
    """Create WeightedRandomSampler to handle class imbalance."""
    targets = [dataset.targets[i] for i in range(len(dataset))]
    class_counts = np.bincount(targets, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


def create_dataloaders(batch_size: int = config.BATCH_SIZE,
                       num_workers: int = 2,
                       verbose: bool = True):
    """
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    _check_brisc_exists()
    utils.seed_everything()

    train_tf, val_tf = get_transforms(augment=True)

    full_train_ds = datasets.ImageFolder(config.BRISC_TRAIN_DIR,
                                         transform=train_tf)
    class_names   = full_train_ds.classes

    all_idx     = list(range(len(full_train_ds)))
    all_targets = full_train_ds.targets
    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=config.VAL_SPLIT,
        stratify=all_targets,
        random_state=config.SEED,
    )

    val_ds_base = datasets.ImageFolder(config.BRISC_TRAIN_DIR, transform=val_tf)

    train_subset = Subset(full_train_ds, train_idx)
    val_subset   = Subset(val_ds_base,   val_idx)

    train_targets = [all_targets[i] for i in train_idx]
    class_counts  = np.bincount(train_targets, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )

    test_ds = datasets.ImageFolder(config.BRISC_TEST_DIR, transform=val_tf)

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers,
                              pin_memory=True)

    val_loader   = DataLoader(val_subset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    test_loader  = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    if verbose:
        print(f"Classes  : {class_names}")
        print(f"Train    : {len(train_subset)} images")
        print(f"Val      : {len(val_subset)} images")
        print(f"Test     : {len(test_ds)} images")
        counts = np.bincount(all_targets, minlength=config.NUM_CLASSES)
        for i, c in enumerate(class_names):
            print(f"  {c:<15} {counts[i]} total images")

    return train_loader, val_loader, test_loader, class_names


if __name__ == '__main__':
    tr, va, te, classes = create_dataloaders()
    imgs, labels = next(iter(tr))
    print(f"\nBatch shape : {imgs.shape}")
    print(f"Label range : {labels.min().item()} – {labels.max().item()}")
    print("DataLoader check passed ✓")
