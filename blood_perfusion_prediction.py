"""Predict keloid blood perfusion images from cropped lesion photographs.

The module loads the trained perfusion generator and predicts a blood perfusion
image for each cropped lesion image. The predicted perfusion image is later
paired with the cropped lesion image for Transformer-based treatment
classification.
"""

from pathlib import Path

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
from util import util


def log(message):
    print(f"[Blood Perfusion] {message}", flush=True)


def configure_inference_options(opt):
    """Use deterministic inference settings for blood perfusion prediction."""
    opt.num_threads = 0
    opt.batch_size = 1
    opt.no_flip = True
    opt.display_id = -1
    return opt


def get_output_dir(opt):
    return Path(opt.results_dir) / "perfusion_result"


def get_output_name(image_paths):
    image_path = image_paths[0] if isinstance(image_paths, (list, tuple)) else image_paths
    return Path(image_path).with_suffix(".png").name


def save_predicted_blood_perfusion(visuals, image_paths, output_dir):
    if "predicted_perfusion" not in visuals:
        raise KeyError("Model output does not contain predicted blood perfusion image.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / get_output_name(image_paths)
    predicted_perfusion = util.tensor2im(visuals["predicted_perfusion"])
    util.save_image(predicted_perfusion, str(output_path))
    return output_path


def run_blood_perfusion_prediction(opt):
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)

    if opt.eval:
        model.eval()

    output_dir = get_output_dir(opt)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for index, data in enumerate(dataset):
        if index >= opt.num_test:
            break

        model.set_input(data)
        model.test()
        output_path = save_predicted_blood_perfusion(
            model.get_current_visuals(), model.get_image_paths(), output_dir
        )
        saved_count += 1

        if saved_count == 1 or saved_count % 5 == 0:
            log(f"Saved {saved_count:04d}: {output_path}")

    log(f"Output directory: {output_dir}")
    log(f"Completed; predicted perfusion images: {saved_count}")


def main():
    opt = configure_inference_options(TestOptions().parse())
    run_blood_perfusion_prediction(opt)


if __name__ == "__main__":
    main()
