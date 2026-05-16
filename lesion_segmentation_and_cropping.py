"""Automatic lesion segmentation and cropping for clinical keloid photographs.

This script is inference-only. It segments each input clinical photograph,
crops the predicted keloid region, saves visual quality-control outputs, and
prepares the paired input images used for blood perfusion prediction.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
from mit_semseg.lib.nn import async_copy_to, user_scattered_collate
from mit_semseg.lib.utils import as_numpy
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import setup_logger


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def log(message):
    print(f"[Segmentation] {message}", flush=True)


def collect_images(path):
    """Return sorted image paths from a file or directory."""
    image_path = Path(path)
    if image_path.is_file():
        return [str(image_path)]

    images = [
        str(file)
        for file in image_path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def get_single_loc(seg, label):
    label_pixels = np.where(seg == label)
    area = len(label_pixels[0])
    if area == 0:
        return 0, 0, 0, 0, 0

    ymin = np.percentile(label_pixels[1], 5)
    ymax = np.percentile(label_pixels[1], 95)
    xmin = np.percentile(label_pixels[0], 5)
    xmax = np.percentile(label_pixels[0], 95)

    center_x = np.array(label_pixels[0]).mean()
    center_y = np.array(label_pixels[1]).mean()
    length_x = 1.054 * (xmax - xmin)
    length_y = 1.054 * (ymax - ymin)
    return center_x, center_y, length_x, length_y, area


def crop_to_label(seg, image, label=1, crop_scale=1.5, output_size=256):
    """Crop around the predicted keloid label and resize image/mask."""
    center_x, center_y, length_x, length_y, area = get_single_loc(seg, label)

    if area == 0:
        cropped_image = image
        cropped_seg = seg
    else:
        crop_length = max(length_x, length_y)
        x0 = max(0, int(center_x - crop_scale * crop_length))
        x1 = min(image.shape[0], int(center_x + crop_scale * crop_length))
        y0 = max(0, int(center_y - crop_scale * crop_length))
        y1 = min(image.shape[1], int(center_y + crop_scale * crop_length))
        cropped_image = image[x0:x1, y0:y1, :]
        cropped_seg = seg[x0:x1, y0:y1]

    resized_image = Image.fromarray(cropped_image).resize(
        (output_size, output_size), Image.BILINEAR
    )
    resized_seg = Image.fromarray(cropped_seg.astype(image.dtype)).resize(
        (output_size, output_size), Image.NEAREST
    )
    return np.array(resized_image), np.array(resized_seg)


def save_segmentation_outputs(data, pred, paths):
    image, image_info = data
    crop_image, crop_seg = crop_to_label(pred, image)
    image_name = Path(image_info).with_suffix(".png").name

    blood_perfusion_input = np.concatenate([crop_image, crop_image], axis=1)
    Image.fromarray(crop_image).save(Path(paths.crop) / image_name)
    Image.fromarray(crop_seg).save(Path(paths.seg) / image_name)
    Image.fromarray(blood_perfusion_input).save(Path(paths.perfusion_pair_input) / image_name)


def run_inference(segmentation_module, loader, gpu, paths):
    segmentation_module.eval()

    for batch_data in tqdm(loader, total=len(loader)):
        batch_data = batch_data[0]
        seg_size = (
            batch_data["img_ori"].shape[0],
            batch_data["img_ori"].shape[1],
        )
        img_resized_list = batch_data["img_data"]

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, seg_size[0], seg_size[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict["img_data"] = img
                del feed_dict["img_ori"]
                del feed_dict["info"]
                feed_dict = async_copy_to(feed_dict, gpu)

                pred_tmp = segmentation_module(feed_dict, segSize=seg_size)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        try:
            save_segmentation_outputs(
                (batch_data["img_ori"], batch_data["info"]),
                pred,
                paths,
            )
        except Exception as exc:
            print(f"Failed to save {batch_data['info']}: {exc}")


def build_segmentation_module():
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder,
    )
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True,
    )
    criterion = nn.NLLLoss(ignore_index=-1)
    return SegmentationModule(net_encoder, net_decoder, criterion)


def create_loader(image_paths, num_workers):
    dataset = TestDataset([{"fpath_img": path} for path in image_paths], cfg.DATASET)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=num_workers,
        drop_last=True,
    )


def fuse_masks(seg_path, crop_path, fuse_path):
    import cv2

    from segmentation_overlay import my_generate_mask_over_img

    os.makedirs(fuse_path, exist_ok=True)
    for image_name in sorted(os.listdir(crop_path)):
        try:
            seg_file = Path(seg_path) / image_name
            crop_file = Path(crop_path) / image_name
            fused_image = my_generate_mask_over_img(str(seg_file), str(crop_file))
            cv2.imwrite(str(Path(fuse_path) / image_name), fused_image)
        except Exception as exc:
            print(f"Failed to fuse {image_name}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Segment clinical keloid photographs, crop the lesion region, "
            "and prepare inputs for blood perfusion prediction."
        )
    )
    parser.add_argument("--imgs", default="./data/raw_data", help="Image file or directory.")
    parser.add_argument(
        "--cfg",
        default="config/ade20k-hrnetv2.yaml",
        metavar="FILE",
        help="Path to the segmentation config file.",
    )
    parser.add_argument("--gpu", default=0, type=int, help="GPU id for inference.")
    parser.add_argument("--crop-path", default="./data/crop_result")
    parser.add_argument("--seg-path", default="./data/seg_result")
    parser.add_argument("--fuse-path", default="./data/fuse_result")
    parser.add_argument(
        "--perfusion-pair-input-path",
        default="./data/perfusion_pair_input/test",
        help="Output directory for paired images used by the blood perfusion model.",
    )
    parser.add_argument("--num-workers", default=20, type=int)
    parser.add_argument("--verbose-config", action="store_true")
    parser.add_argument(
        "opts",
        help="Override config options from the command line.",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.verbose_config:
        logger = setup_logger(distributed_rank=0)
        logger.info("Loaded configuration file %s", args.cfg)
        logger.info("Running with config:\n%s", cfg)

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()
    cfg.MODEL.weights_encoder = os.path.join(cfg.DIR, "encoder_" + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(cfg.DIR, "decoder_" + cfg.TEST.checkpoint)

    if not os.path.exists(cfg.MODEL.weights_encoder):
        raise FileNotFoundError(f"Missing encoder checkpoint: {cfg.MODEL.weights_encoder}")
    if not os.path.exists(cfg.MODEL.weights_decoder):
        raise FileNotFoundError(f"Missing decoder checkpoint: {cfg.MODEL.weights_decoder}")

    log(f"Config: {args.cfg}")
    log(f"Encoder checkpoint: {cfg.MODEL.weights_encoder}")
    log(f"Decoder checkpoint: {cfg.MODEL.weights_decoder}")

    paths = argparse.Namespace(
        crop=args.crop_path,
        seg=args.seg_path,
        fuse=args.fuse_path,
        perfusion_pair_input=args.perfusion_pair_input_path,
    )
    for output_path in (paths.crop, paths.seg, paths.fuse, paths.perfusion_pair_input):
        os.makedirs(output_path, exist_ok=True)

    image_paths = collect_images(args.imgs)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.imgs}")
    log(f"Input images: {len(image_paths)}")

    torch.cuda.set_device(args.gpu)
    segmentation_module = build_segmentation_module().cuda()
    loader = create_loader(image_paths, args.num_workers)

    run_inference(segmentation_module, loader, args.gpu, paths)
    fuse_masks(paths.seg, paths.crop, paths.fuse)
    log(f"Crop output: {paths.crop}")
    log(f"Mask output: {paths.seg}")
    log(f"Blood perfusion model input: {paths.perfusion_pair_input}")
    log("Completed")


if __name__ == "__main__":
    main()
