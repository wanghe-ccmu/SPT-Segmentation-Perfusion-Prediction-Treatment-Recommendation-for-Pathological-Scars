"""Data loading utilities for keloid blood perfusion prediction."""
import torch.utils.data

from data.aligned_dataset import AlignedDataset


def create_dataset(opt):
    """Create the dataset used by blood perfusion prediction."""
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Small wrapper around the configured dataset and PyTorch DataLoader."""

    def __init__(self, opt):
        """Create a dataset instance and dataloader."""
        self.opt = opt
        self.dataset = AlignedDataset(opt)
        print("[Blood Perfusion] Dataset: %s (%d images)" % (
            type(self.dataset).__name__, len(self.dataset)
        ))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.num_threads),
        )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of images in the dataset."""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Yield inference batches."""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
