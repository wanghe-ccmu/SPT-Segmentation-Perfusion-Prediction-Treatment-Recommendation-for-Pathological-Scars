"""Run the clinical-photo-to-treatment-classification pipeline for keloids."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_step(name, command):
    print(f"\n[Stage] {name}", flush=True)
    print(f"[Command] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"[Status] {name} completed", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run lesion segmentation/cropping, blood perfusion prediction, "
            "and Transformer-based treatment classification."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for all pipeline steps.",
    )
    parser.add_argument(
        "--perfusion-pair-dataroot",
        default="./data/perfusion_pair_input",
        help="Input directory passed to blood_perfusion_prediction.py.",
    )
    parser.add_argument(
        "--perfusion-name",
        default="blood_perfusion",
        help="Checkpoint name passed to blood_perfusion_prediction.py.",
    )
    parser.add_argument(
        "--perfusion-model",
        default="perfusion_gan",
        help="Model name passed to blood_perfusion_prediction.py.",
    )
    parser.add_argument(
        "--skip-segmentation",
        action="store_true",
        help="Skip lesion_segmentation_and_cropping.py if segmentation/crop outputs already exist.",
    )
    parser.add_argument(
        "--skip-perfusion",
        action="store_true",
        help="Skip blood_perfusion_prediction.py if predicted perfusion images already exist.",
    )
    parser.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip transformer_treatment_classification.py.",
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip automatic HTML report generation.",
    )
    args = parser.parse_args()

    print("Keloid Treatment Classification Pipeline", flush=True)
    print(f"[Workspace] {ROOT}", flush=True)

    if not args.skip_segmentation:
        run_step(
            "1/4 Lesion segmentation and crop preparation",
            [args.python, "lesion_segmentation_and_cropping.py"],
        )

    if not args.skip_perfusion:
        run_step(
            "2/4 Blood perfusion prediction",
            [
                args.python,
                "blood_perfusion_prediction.py",
                "--dataroot",
                args.perfusion_pair_dataroot,
                "--name",
                args.perfusion_name,
                "--model",
                args.perfusion_model,
            ],
        )

    if not args.skip_classification:
        run_step(
            "3/4 Transformer treatment classification",
            [args.python, "transformer_treatment_classification.py"],
        )

    if not args.skip_reports:
        run_step(
            "4/4 Per-case report generation",
            [args.python, "clinical_report_generation.py"],
        )

    print("\n[Done] Pipeline finished successfully.", flush=True)


if __name__ == "__main__":
    main()
