from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """Inference options for keloid blood perfusion prediction."""

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--results_dir",
            type=str,
            default="./data/",
            help="Directory for predicted blood perfusion outputs.",
        )
        parser.add_argument("--phase", type=str, default="test", help="Input split name.")
        parser.add_argument("--eval", action="store_true", help="Use eval mode during inference.")
        parser.add_argument("--num_test", type=int, default=100000, help="Maximum number of images to process.")
        parser.set_defaults(model="perfusion_gan")
        parser.set_defaults(load_size=parser.get_default("crop_size"))
        self.isTrain = False
        return parser
