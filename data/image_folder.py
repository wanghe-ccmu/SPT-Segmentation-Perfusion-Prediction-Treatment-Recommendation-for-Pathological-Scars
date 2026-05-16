"""Image discovery helpers for blood perfusion model inputs."""
import os

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"}


def is_image_file(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def make_dataset(image_dir, max_dataset_size=float("inf")):
    """Return sorted image paths from a directory and its subdirectories."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Input directory not found: {image_dir}")

    images = []
    for root, _, filenames in sorted(os.walk(image_dir)):
        for fname in sorted(filenames):
            if is_image_file(fname):
                images.append(os.path.join(root, fname))

    limit = min(int(max_dataset_size), len(images)) if max_dataset_size != float("inf") else len(images)
    return images[:limit]
