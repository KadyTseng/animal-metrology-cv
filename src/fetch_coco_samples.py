"""
COCO Dataset Sample Fetcher & Metrology Evaluator (10 Multi-Animal Samples)
==========================================================================
Fetches 10 verified sample images containing two or more animals from the official
MS COCO 2017 dataset (val2017) and executes the metrology pipeline.

Animal categories included:
- Cats (貓咪)
- Dogs (狗狗)
- Zebras (斑馬)
- Sheep (綿羊)
- Horses (駿馬)
- Cows / Cattle (乳牛)
- Elephants (大象)
- Giraffes (長頸鹿)
"""

import os
import sys
import urllib.request
import logging
from pathlib import Path
from typing import Optional, Union, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrology import AnimalMetrologyPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("COCOFetcher")

# 10 Official MS COCO 2017 validation images with 2+ animals
COCO_10_SAMPLES = [
    {
        "filename": "01_coco_two_cats_39769.jpg",
        "url": "http://images.cocodataset.org/val2017/000000039769.jpg",
        "animal": "Cats (兩隻貓)",
        "description": "Two cats lying side-by-side on couch with clear facial features",
    },
    {
        "filename": "02_coco_three_zebras_8842.jpg",
        "url": "http://images.cocodataset.org/val2017/000000008842.jpg",
        "animal": "Zebras (三隻斑馬)",
        "description": "Three zebras in grassland standing in diverse poses",
    },
    {
        "filename": "03_coco_two_sheep_73860.jpg",
        "url": "http://images.cocodataset.org/val2017/000000073860.jpg",
        "animal": "Sheep (兩隻綿羊)",
        "description": "Two white sheep grazing in pasture with high contrast background",
    },
    {
        "filename": "04_coco_two_horses_57027.jpg",
        "url": "http://images.cocodataset.org/val2017/000000057027.jpg",
        "animal": "Horses (兩匹馬)",
        "description": "Two horses standing side by side in field",
    },
    {
        "filename": "05_coco_two_dogs_8021.jpg",
        "url": "http://images.cocodataset.org/val2017/000000008021.jpg",
        "animal": "Dogs (兩隻狗)",
        "description": "Two dogs resting together with clearly visible facial anatomy",
    },
    {
        "filename": "06_coco_elephants_18837.jpg",
        "url": "http://images.cocodataset.org/val2017/000000018837.jpg",
        "animal": "Elephants (兩隻大象)",
        "description": "Mother and baby elephant walking together",
    },
    {
        "filename": "07_coco_cows_49961.jpg",
        "url": "http://images.cocodataset.org/val2017/000000049961.jpg",
        "animal": "Cows (牧場乳牛)",
        "description": "Multiple cows in meadow with distinct body contours",
    },
    {
        "filename": "08_coco_zebras_86956.jpg",
        "url": "http://images.cocodataset.org/val2017/000000086956.jpg",
        "animal": "Zebras (兩隻斑馬)",
        "description": "Two zebras grazing close to each other",
    },
    {
        "filename": "09_coco_dogs_playing_289594.jpg",
        "url": "http://images.cocodataset.org/val2017/000000289594.jpg",
        "animal": "Dogs (兩隻玩耍的狗)",
        "description": "Two dogs interactively standing together",
    },
    {
        "filename": "10_coco_giraffes_358757.jpg",
        "url": "http://images.cocodataset.org/val2017/000000358757.jpg",
        "animal": "Giraffes (兩隻長頸鹿)",
        "description": "Two giraffes side by side in savannah landscape",
    },
]


def download_10_coco_samples(target_dir: Optional[Union[str, Path]] = None) -> list[Path]:
    """
    Download 10 official MS COCO multi-animal images.
    """
    out_dir = Path(target_dir) if target_dir else (PROJECT_ROOT / "data" / "raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for item in COCO_10_SAMPLES:
        dest = out_dir / item["filename"]
        if not dest.exists():
            logger.info("Downloading [%s] %s ...", item["animal"], item["filename"])
            try:
                urllib.request.urlretrieve(item["url"], dest)
                logger.info("Saved %s (size: %d bytes)", dest.name, dest.stat().st_size)
            except Exception as e:
                logger.warning("Download failed for %s: %s", item["url"], e)
                continue
        else:
            logger.info("Already exists: %s", dest.name)
        downloaded.append(dest)

    logger.info("Total available COCO images: %d / %d", len(downloaded), len(COCO_10_SAMPLES))
    return downloaded


def run_batch_evaluation():
    """
    Process all downloaded COCO images with the metrology pipeline.
    """
    pipeline = AnimalMetrologyPipeline()
    images = download_10_coco_samples()

    all_pairs = []
    ann_dir = PROJECT_ROOT / "outputs" / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        logger.info("\n=== Processing: %s ===", img_path.name)
        res = pipeline.infer(img_path, image_id=img_path.name)
        logger.info("Detected animals: %d", res["detected_animals_count"])

        for a in res["animals"]:
            logger.info(
                "  [%s] L-Eye: %s, R-Eye: %s, EyeDist: %s mm",
                a["animal_id"],
                a.get("left_eye"),
                a.get("right_eye"),
                a.get("interocular_dist_mm"),
            )

        for p in res["pairwise_measurements"]:
            logger.info(
                "  Pair (%s <-> %s) Cross R-R Dist: %s mm",
                p["animal_1_id"],
                p["animal_2_id"],
                p.get("right_eye_to_right_eye_dist_mm"),
            )
            all_pairs.append(p)

        # Save visual annotations
        ann_path = ann_dir / f"annotated_{img_path.name}"
        pipeline.annotate_image(img_path, res, output_path=ann_path)

    if all_pairs:
        csv_file = pipeline.export_to_csv(
            all_pairs,
            output_path=PROJECT_ROOT / "outputs" / "measurement_results.csv",
            append=True,
        )
        logger.info("\n[SUCCESS] Exported all batch measurements to: %s", csv_file)


if __name__ == "__main__":
    run_batch_evaluation()
