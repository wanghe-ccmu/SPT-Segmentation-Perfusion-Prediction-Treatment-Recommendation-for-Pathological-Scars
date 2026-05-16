import argparse
import os
import torch
import models


class BaseOptions:
    """Shared options for blood perfusion prediction inference."""

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        """Define common options for the blood perfusion prediction model."""
        parser.add_argument(
            "--dataroot",
            required=True,
            help="Directory containing the test folder for blood perfusion model inputs.",
        )
        parser.add_argument(
            "--name",
            type=str,
            default="blood_perfusion",
            help="Weight subfolder under --checkpoints_dir.",
        )
        parser.add_argument(
            "--checkpoints_dir",
            type=str,
            default="./weights",
            help="Root directory for model weights.",
        )
        parser.add_argument(
            "--gpu_ids",
            type=str,
            default="0",
            help="GPU ids, for example 0 or 0,1. Use -1 for CPU.",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="perfusion_gan",
            help="Internal blood perfusion model loader name.",
        )
        parser.add_argument("--num_threads", default=4, type=int, help="Data loading workers.")
        parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size.")
        parser.add_argument(
            "--max_dataset_size",
            type=int,
            default=float("inf"),
            help="Maximum number of images to process.",
        )
        parser.add_argument("--epoch", type=str, default="latest", help="Weight epoch to load.")
        parser.add_argument("--verbose", action="store_true", help="Print detailed network architecture.")

        self._add_hidden_model_defaults(parser)
        self._add_hidden_dataset_defaults(parser)
        self.initialized = True
        return parser

    def _add_hidden_model_defaults(self, parser):
        """Keep checkpoint-compatible generator settings out of the public CLI."""
        parser.add_argument("--input_nc", type=int, default=3, help=argparse.SUPPRESS)
        parser.add_argument("--output_nc", type=int, default=3, help=argparse.SUPPRESS)
        parser.add_argument("--ngf", type=int, default=64, help=argparse.SUPPRESS)
        parser.add_argument("--netG", type=str, default="unet_256", help=argparse.SUPPRESS)
        parser.add_argument("--norm", type=str, default="instance", help=argparse.SUPPRESS)
        parser.add_argument("--init_type", type=str, default="normal", help=argparse.SUPPRESS)
        parser.add_argument("--init_gain", type=float, default=0.02, help=argparse.SUPPRESS)
        parser.add_argument("--no_dropout", action="store_true", help=argparse.SUPPRESS)

    def _add_hidden_dataset_defaults(self, parser):
        """Keep paired-image loader settings stable while reducing help clutter."""
        parser.add_argument("--load_size", type=int, default=256, help=argparse.SUPPRESS)
        parser.add_argument("--crop_size", type=int, default=256, help=argparse.SUPPRESS)
        parser.add_argument("--preprocess", type=str, default="resize_and_crop", help=argparse.SUPPRESS)
        parser.add_argument("--no_flip", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--load_iter", type=int, default=0, help=argparse.SUPPRESS)

    def gather_options(self):
        """Initialize the parser and apply model/dataset defaults."""
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()

        model_option_setter = models.get_option_setter(opt.model)
        parser = model_option_setter(parser, self.isTrain)
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print a concise inference configuration summary."""
        summary = (
            f"[Blood Perfusion] Checkpoint: {os.path.join(opt.checkpoints_dir, opt.name)}\n"
            f"[Blood Perfusion] Input: {os.path.join(opt.dataroot, opt.phase)}\n"
            f"[Blood Perfusion] Model: {opt.model} ({opt.netG})"
        )
        print(summary)

    def parse(self):
        """Parse inference options and configure the target device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain

        self.print_options(opt)

        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
