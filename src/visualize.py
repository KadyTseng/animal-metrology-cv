"""
Animal Metrology CV - Visualizer Utility
=======================================
Generates annotated metrology images overlaying:
1. Animal instance segmentation masks (YOLOv8-seg)
2. Left and Right eye keypoints from the fine-tuned YOLOv8-pose model
3. Task 3 and 4 measurement lines, with distance labels in a separate panel

Usage:
  python src/visualize.py                           # Process all images in data/raw/
  python src/visualize.py --image path/to/image.jpg # Process a specific image
  python src/visualize.py --input-dir path/to/images
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrology import AnimalMetrologyPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Visualizer")


def visualize_image(pipeline: AnimalMetrologyPipeline, image_path: Path, output_dir: Path) -> Path:
    """
    Run pipeline on a single image and save the annotated visualization.
    """
    logger.info("Processing visualization for: %s", image_path.name)
    results = pipeline.infer(image_path, image_id=image_path.name)

    output_path = output_dir / f"annotated_{image_path.name}"
    pipeline.annotate_image(image_path, results, output_path=output_path)

    logger.info(
        "Saved annotated visualization to: %s (Detected: %d animals)",
        output_path,
        results["detected_animals_count"],
    )
    logger.info(
        "Pose keypoints are model estimates; mask inclusion does not verify anatomical eye identity."
    )
    for animal in results["animals"]:
        logger.info(
            "  -> %s interocular distance (mask-checked) = %s mm",
            animal["animal_id"],
            animal.get("interocular_dist_mm"),
        )
    for p in results["pairwise_measurements"]:
        logger.info(
            "  -> Pair %s <-> %s: mask-checked R-R estimate = %s mm",
            p["animal_1_id"],
            p["animal_2_id"],
            p.get("right_eye_to_right_eye_dist_mm"),
        )
        candidate_distance = p.get("pose_candidate_right_eye_to_right_eye_dist_mm")
        if p.get("right_eye_to_right_eye_dist_mm") is None and candidate_distance is not None:
            logger.info("     Unverified pose candidate R-R distance = %.2f mm", candidate_distance)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize Animal Metrology CV outputs.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a specific animal image. If omitted, all images in data/raw/ are processed.",
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing images to process. Defaults to data/raw/ when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/annotated/seg_pose",
        help="Directory where annotated images will be saved.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.15,
        help="Metric conversion scale (mm/pixel). Defaults to 0.15.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device (e.g. 0 or cpu). Defaults to DEVICE in .env or automatic selection.",
    )

    args = parser.parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = AnimalMetrologyPipeline(pixel_to_mm_scale=args.scale, device=args.device)

    if args.image:
        target = Path(args.image)
        if not target.exists():
            target = PROJECT_ROOT / args.image
        if not target.exists():
            logger.error("Specified image does not exist: %s", args.image)
            sys.exit(1)
        visualize_image(pipeline, target, output_dir)
    else:
        raw_dir = (
            Path(args.input_dir)
            if args.input_dir
            else PROJECT_ROOT / "data" / "raw"
        )
        if not raw_dir.is_absolute():
            raw_dir = PROJECT_ROOT / raw_dir
        if not raw_dir.is_dir():
            logger.error("Input directory does not exist: %s", raw_dir)
            sys.exit(1)

        image_patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        images = sorted(path for pattern in image_patterns for path in raw_dir.glob(pattern))
        if not images:
            logger.error("No supported images found in %s", raw_dir)
            sys.exit(1)

        logger.info("Batch visualizing %d images from %s ...", len(images), raw_dir)
        for img in images:
            visualize_image(pipeline, img, output_dir)

    logger.info("\n[SUCCESS] All visualizations generated in: %s", output_dir)


if __name__ == "__main__":
    main()
