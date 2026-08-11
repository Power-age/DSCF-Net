import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_DATA_ROOT = os.environ.get("DSCF_DATA_ROOT", "Dataset")

DATASET_INFO = {
    "UCM": {"num_classes": 21, "image_size": 256},
    "AID": {"num_classes": 30, "image_size": 600},
    "NWPU45": {"num_classes": 45, "image_size": 256},
}

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SceneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted(d.name for d in self.root_dir.iterdir() if d.is_dir())
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in IMAGE_SUFFIXES:
                    self.samples.append((str(image_path), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, IOError):
            fallback_idx = (idx + 1) % len(self.samples)
            image_path, label = self.samples[fallback_idx]
            image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    dataset_name,
    train_ratio,
    batch_size=64,
    num_workers=4,
    data_root=None,
    seed=42,
):
    set_seed(seed)

    if data_root is None:
        data_root = DEFAULT_DATA_ROOT

    train_ratio = float(train_ratio)
    ratio_pct = int(round(train_ratio * 100))
    pre_split_name = f"{dataset_name}_{ratio_pct}"
    pre_split_path = os.path.join(data_root, pre_split_name)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if os.path.isdir(pre_split_path):
        train_dir = os.path.join(pre_split_path, "train")
        test_dir = os.path.join(pre_split_path, "test")
        if os.path.isdir(train_dir) and os.path.isdir(test_dir):
            train_dataset = SceneDataset(train_dir, transform=train_transform)
            val_dataset = SceneDataset(test_dir, transform=val_transform)
            if train_dataset.classes != val_dataset.classes:
                raise ValueError(
                    f"Class mismatch: train={len(train_dataset.classes)} vs test={len(val_dataset.classes)}"
                )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            info = DATASET_INFO.get(dataset_name, {"num_classes": len(train_dataset.classes)})
            print(f"Dataset: {pre_split_name} (pre-split)")
            print(f"  Classes: {len(train_dataset.classes)}")
            print(f"  Train samples: {len(train_dataset)} | Test samples: {len(val_dataset)}")
            return train_loader, val_loader, info["num_classes"], train_dataset.classes

    dataset_path = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {pre_split_path} or {dataset_path}")

    full_dataset = SceneDataset(dataset_path, transform=None)
    labels = np.array([sample[1] for sample in full_dataset.samples])
    train_idx, val_idx = _stratified_split(labels, train_ratio, seed)

    train_dataset = SceneDataset(dataset_path, transform=train_transform)
    train_dataset.samples = [full_dataset.samples[i] for i in train_idx]

    val_dataset = SceneDataset(dataset_path, transform=val_transform)
    val_dataset.samples = [full_dataset.samples[i] for i in val_idx]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    info = DATASET_INFO.get(dataset_name, {"num_classes": len(full_dataset.classes)})
    print(f"Dataset: {dataset_name} | Train ratio: {train_ratio}")
    print(f"  Classes: {len(full_dataset.classes)}")
    print(f"  Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    return train_loader, val_loader, info["num_classes"], full_dataset.classes


def _stratified_split(labels, train_ratio, seed=42):
    rng = np.random.RandomState(seed)
    unique_labels = np.unique(labels)
    train_idx = []
    val_idx = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        n_train = max(1, int(len(label_indices) * train_ratio))
        shuffled = label_indices[rng.permutation(len(label_indices))]
        train_idx.extend(shuffled[:n_train].tolist())
        val_idx.extend(shuffled[n_train:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx
