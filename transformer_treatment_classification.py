"""Transformer-based conservative/aggressive treatment classification.

This script is inference-only. It loads two Vision Transformer feature
extractors and a fusion head, then classifies each case from two inputs:
the cropped lesion photograph and the predicted blood perfusion image.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import vision_transformer_backbone
from treatment_classification_dataset import KeloidTreatmentClassificationDataset, build_transform


def log(message):
    print(f"[Treatment Classification] {message}", flush=True)


def get_args_parser():
    parser = argparse.ArgumentParser("Keloid treatment classification", add_help=True)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--model", default="vit_base_patch16", type=str)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--drop_path", default=0.2, type=float)
    parser.add_argument("--nb_classes", default=2, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--crop-path", default="./data/crop_result")
    parser.add_argument("--perfusion-result-path", default="./data/perfusion_result")
    parser.add_argument(
        "--checkpoint1",
        default="./weights/treatment_classification/both4cls_model1.pth",
    )
    parser.add_argument(
        "--checkpoint2",
        default="./weights/treatment_classification/both4cls_model2.pth",
    )
    parser.add_argument(
        "--checkpoint3",
        default="./weights/treatment_classification/both4cls_model3.pth",
    )
    parser.add_argument("--output-xlsx", default="output_prediction.xlsx")
    parser.add_argument("--output-npy", default="all_target_multiple_prospective.npy")

    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for ViT features.",
    )
    return parser


def _weights_init(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight)


class TreatmentClassificationFusionHead(nn.Module):
    """Fusion head for cropped-lesion and blood-perfusion Transformer features."""

    def __init__(self):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()

        for _ in range(12):
            self.conv.append(
                nn.Sequential(
                    nn.Linear(768, 768),
                    nn.BatchNorm1d(768),
                    nn.ReLU(inplace=True),
                )
            )
            self.conv1.append(nn.Linear(768, 768))
            self.conv2.append(nn.Linear(768, 768))

        self.final_conv = nn.Sequential(
            nn.Linear(768, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Linear(192, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 2),
        )
        self.apply(_weights_init)

    def forward(self, x1, x2):
        x = None
        for index in range(len(x1)):
            fused = self.conv1[index](x1[index]) + self.conv2[index](x2[index])
            if x is not None:
                fused = fused + x
            x = self.conv[index](fused) + fused
        return self.final_conv(x)


def build_vit(args):
    return vision_transformer_backbone.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        img_size=args.input_size,
    )


def load_state_dict(model, checkpoint_path, device, name):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing {name} checkpoint: {checkpoint_path}")

    state_dict = torch.load(str(checkpoint_path), map_location=device)
    message = model.load_state_dict(state_dict, strict=False)
    log(f"Loaded {name}: {checkpoint_path}")
    if message.missing_keys:
        log(f"Missing keys in {name}: {message.missing_keys}")
    if message.unexpected_keys:
        log(f"Unexpected keys in {name}: {message.unexpected_keys}")


def build_dataset(args):
    transform = build_transform(is_train=False, args=args)
    return KeloidTreatmentClassificationDataset(
        cropped_lesion_dir=args.crop_path,
        predicted_perfusion_dir=args.perfusion_result_path,
        transform=transform,
    )


def build_data_loader(dataset, args):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.SequentialSampler(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )


@torch.no_grad()
def predict(data_loader, model1, model2, model3, device):
    model1.eval()
    model2.eval()
    model3.eval()

    probabilities = []
    image_names = []

    for images, _, names in data_loader:
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            output = model3(model1(images[:, :3]), model2(images[:, 3:]))
            probability = torch.softmax(output, dim=1)

        probabilities.append(probability.cpu().numpy())
        image_names.extend(list(names))

    return np.concatenate(probabilities, axis=0), np.array(image_names)


def save_predictions(probabilities, image_names, output_xlsx, output_npy):
    df = pd.DataFrame(
        {
            "name": image_names,
            "Conservative": probabilities[:, 0],
            "Aggressive": probabilities[:, 1],
        }
    )
    df.to_excel(output_xlsx, index=False)
    np.save(output_npy, probabilities)
    log(f"Prediction table: {output_xlsx}")
    log(f"Probability array: {output_npy}")


def main(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = build_dataset(args)
    data_loader = build_data_loader(dataset, args)
    log(f"Samples: {len(dataset)}")

    model1 = build_vit(args).to(device)
    model2 = build_vit(args).to(device)
    model3 = TreatmentClassificationFusionHead().to(device)

    load_state_dict(model1, args.checkpoint1, device, "cropped-lesion ViT")
    load_state_dict(model2, args.checkpoint2, device, "blood-perfusion ViT")
    load_state_dict(model3, args.checkpoint3, device, "treatment classification fusion head")

    probabilities, image_names = predict(data_loader, model1, model2, model3, device)
    save_predictions(probabilities, image_names, args.output_xlsx, args.output_npy)
    log("Completed")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
