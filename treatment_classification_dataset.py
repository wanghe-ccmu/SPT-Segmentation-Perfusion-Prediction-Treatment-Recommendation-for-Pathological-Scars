"""Dataset utilities for Transformer-based keloid treatment classification."""

import random
from pathlib import Path

import PIL
import torch
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms


def build_transform(is_train, args):
    if is_train:
        raise ValueError("This repository entry point only supports inference transforms.")

    crop_pct = 224 / 256 if args.input_size <= 224 else 1.0
    resize_size = int(args.input_size / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=PIL.Image.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )


class KeloidTreatmentClassificationDataset(Dataset):
    """Pairs cropped lesion photographs with predicted blood perfusion images."""

    def __init__(self, cropped_lesion_dir, predicted_perfusion_dir, transform=None):
        self.cropped_lesion_dir = Path(cropped_lesion_dir)
        self.predicted_perfusion_dir = Path(predicted_perfusion_dir)
        self.transform = transform

        if not self.cropped_lesion_dir.exists():
            raise FileNotFoundError(f"Missing crop directory: {self.cropped_lesion_dir}")
        if not self.predicted_perfusion_dir.exists():
            raise FileNotFoundError(
                f"Missing predicted blood perfusion directory: {self.predicted_perfusion_dir}"
            )

        self.image_names = sorted(
            file.name
            for file in self.cropped_lesion_dir.iterdir()
            if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        if not self.image_names:
            raise FileNotFoundError(f"No images found in {self.cropped_lesion_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        cropped_lesion = Image.open(self.cropped_lesion_dir / image_name).convert("RGB")
        predicted_perfusion = Image.open(self.predicted_perfusion_dir / image_name).convert("RGB")

        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            torch.manual_seed(seed)
            cropped_lesion = self.transform(cropped_lesion)
            random.seed(seed)
            torch.manual_seed(seed)
            predicted_perfusion = self.transform(predicted_perfusion)

        paired_image = torch.cat([cropped_lesion, predicted_perfusion], dim=0)
        label_placeholder = 0
        return paired_image, label_placeholder, image_name
