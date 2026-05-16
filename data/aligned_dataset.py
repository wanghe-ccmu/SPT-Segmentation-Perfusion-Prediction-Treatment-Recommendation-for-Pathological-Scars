import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


class AlignedDataset(BaseDataset):
    """Paired input dataset for blood perfusion prediction.

    Each image is split into left and right halves. The left half is the cropped
    lesion image. The right half preserves the paired-input format expected by
    the trained blood perfusion generator.
    """

    def __init__(self, opt):
        """Initialize the paired input dataset."""
        BaseDataset.__init__(self, opt)
        self.image_dir = os.path.join(opt.dataroot, opt.phase)
        self.image_paths = sorted(make_dataset(self.image_dir, opt.max_dataset_size))
        if self.opt.load_size < self.opt.crop_size:
            raise ValueError("load_size must be greater than or equal to crop_size.")

    def __getitem__(self, index):
        """Return the paired blood-perfusion input and source path."""
        image_path = self.image_paths[index]
        paired_image = Image.open(image_path).convert('RGB')
        # Split the paired input image into the two channels expected by the generator.
        w, h = paired_image.size
        w2 = int(w / 2)
        cropped_lesion = paired_image.crop((0, 0, w2, h))
        paired_context = paired_image.crop((w2, 0, w, h))

        transform_params = get_params(self.opt, cropped_lesion.size)
        transform = get_transform(self.opt, transform_params)

        cropped_lesion = transform(cropped_lesion)
        paired_context = transform(paired_context)
        return {
            "A": cropped_lesion,
            "B": paired_context,
            "A_paths": image_path,
            "B_paths": image_path,
        }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
