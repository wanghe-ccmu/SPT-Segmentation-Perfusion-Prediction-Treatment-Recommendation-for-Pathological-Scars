import torch

from . import networks
from .base_model import BaseModel


class PerfusionGanModel(BaseModel):
    """Generator used for keloid blood perfusion prediction."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.visual_names = ["cropped_lesion", "predicted_perfusion", "paired_input"]
        self.model_names = ["G"]
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.norm,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
        )

    def set_input(self, input):
        self.cropped_lesion = input["A"].to(self.device)
        self.paired_input = input["B"].to(self.device)
        self.image_paths = input["A_paths"]

    def forward(self):
        self.predicted_perfusion = self.netG(self.cropped_lesion)

    def optimize_parameters(self):
        raise RuntimeError("PerfusionGanModel is inference-only in this repository.")
