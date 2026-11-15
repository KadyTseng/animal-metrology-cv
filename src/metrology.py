"""
Animal Metrology Pipeline
=========================
Two-Stage Computer Vision Metrology Pipeline:
1. Stage 1 (Segmentation): Utilize YOLOv8-seg (Instance Segmentation) to extract
   individual animal contours, bounding boxes, and binary mask polygons.
2. Stage 2 (Eye Localization & Metrology): Localize left and right eyes strictly
   within each segmented animal mask, calculating morphological distances.

Tasks:
- Task 3: Single-animal interocular distance (pixel to physical mm).
- Task 4: Cross-individual distance between right eyes of arbitrary animal pairs.
"""

from __future__ import annotations

import os
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AnimalMetrology")

# COCO Animal Category mapping (80-class COCO IDs)
COCO_ANIMAL_CLASSES = {
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}


class AnimalMetrologyPipeline:
    """
    Two-Stage Animal Metrology Pipeline:
    - Step 1: Semantic / Instance Segmentation with YOLOv8-seg.
    - Step 2: Anatomical Eye Keypoint Localization & Metric Metrology.
    """

    def __init__(
        self,
        seg_model_weights: Optional[str] = None,
        pose_model_weights: Optional[str] = None,
        pixel_to_mm_scale: Optional[float] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        pose_animal_conf_threshold: Optional[float] = None,
        pose_keypoint_conf_threshold: Optional[float] = None,
        pose_crop_margin: Optional[float] = None,
        device: Optional[Union[str, int]] = None,
    ):
        """
        Initialize the Two-Stage Metrology Pipeline.

        :param seg_model_weights: YOLOv8 Segmentation weights (default: yolov8n-seg.pt).
        :param pose_model_weights: Custom two-keypoint YOLOv8 Pose weights.
        :param pixel_to_mm_scale: Physical conversion ratio (mm/pixel, default: 0.15).
        :param pose_keypoint_conf_threshold: Minimum confidence for an eye keypoint.
        :param pose_crop_margin: Fraction of the animal box added around the pose crop.
        """
        self.seg_model_weights = (
            seg_model_weights
            or os.getenv("SEG_MODEL_WEIGHTS_PATH")
            or os.getenv("MODEL_WEIGHTS_PATH", "yolov8n-seg.pt")
        )
        self.pose_model_weights = (
            pose_model_weights
            or os.getenv(
                "POSE_MODEL_WEIGHTS_PATH",
                "models/animal_eye_pose.pt",
            )
        )
        self.pixel_to_mm_scale = float(
            pixel_to_mm_scale
            if pixel_to_mm_scale is not None
            else os.getenv("PIXEL_TO_MM_SCALE", "0.15")
        )
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.pose_animal_conf_threshold = float(
            pose_animal_conf_threshold
            if pose_animal_conf_threshold is not None
            else os.getenv("POSE_ANIMAL_CONF_THRESHOLD", "0.4")
        )
        self.pose_keypoint_conf_threshold = float(
            pose_keypoint_conf_threshold
            if pose_keypoint_conf_threshold is not None
            else os.getenv("POSE_KEYPOINT_CONF_THRESHOLD", "0.1")
        )
        self.pose_crop_margin = float(
            pose_crop_margin
            if pose_crop_margin is not None
            else os.getenv("POSE_CROP_MARGIN", "0.2")
        )
        self.device = device if device is not None else os.getenv("DEVICE") or None

        self.seg_model = None
        self.pose_model = None

        self._load_models()

    def _load_models(self) -> None:
        """
        Load the animal segmentation model and custom cat/dog eye-pose model.
        """
        from ultralytics import YOLO

        logger.info("Loading YOLOv8-seg segmentation model from: %s", self.seg_model_weights)
        self.seg_model = YOLO(self.seg_model_weights)

        logger.info("Loading custom YOLOv8-pose model from: %s", self.pose_model_weights)
        self.pose_model = YOLO(self.pose_model_weights)
        pose_names = {str(name).lower() for name in self.pose_model.names.values()}
        if not {"cat", "dog"}.issubset(pose_names):
            raise ValueError(
                "The eye pose model must contain both 'cat' and 'dog' classes; "
                f"found {sorted(pose_names)}."
            )

        pose_head = getattr(self.pose_model.model.model[-1], "kpt_shape", None)
        if pose_head is None or list(pose_head) != [2, 3]:
            raise ValueError(
                "The eye pose model must predict two keypoints with visibility/confidence; "
                f"found {pose_head}."
            )
        logger.info("Segmentation and two-eye pose models loaded successfully.")

    @staticmethod
    def calculate_box_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(box1_area + box2_area - intersection_area)

        if union_area <= 0:
            return 0.0
        return intersection_area / union_area

    @staticmethod
    def euclidean_distance(
        pt1: Union[Tuple[float, float], List[float], np.ndarray],
        pt2: Union[Tuple[float, float], List[float], np.ndarray],
    ) -> float:
        """
        Calculate 2D Euclidean distance:
            d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
        """
        x1, y1 = float(pt1[0]), float(pt1[1])
        x2, y2 = float(pt2[0]), float(pt2[1])
        return math.hypot(x2 - x1, y2 - y1)

    def pixel_to_mm(self, pixel_dist: float) -> float:
        """
        Convert pixel distance to physical millimeters:
            D_mm = D_pixel * scale
        """
        return round(pixel_dist * self.pixel_to_mm_scale, 3)

    def read_image(self, image_source: Union[str, Path, np.ndarray, bytes]) -> np.ndarray:
        """
        Robustly decode image from path, ndarray, or bytes buffer.
        """
        if isinstance(image_source, np.ndarray):
            return image_source

        if isinstance(image_source, bytes):
            nparr = np.frombuffer(image_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image from provided byte buffer.")
            return img

        path = Path(image_source)
        if not path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {path}")

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"OpenCV could not read image at: {path}")
        return img

    def _detect_eyes_within_mask(
        self,
        image: np.ndarray,
        bbox: List[float],
        polygon: Optional[List[List[float]]],
        class_name: str,
        animal_confidence: float = 1.0,
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        """
        Detect the two eye keypoints in an expanded animal crop and keep only points
        that fall inside that animal's segmentation mask.
        """
        image_h, image_w = image.shape[:2]
        if class_name.lower() not in {"cat", "dog"}:
            return None, None
        if animal_confidence < self.pose_animal_conf_threshold:
            return None, None
        if self.pose_model is None:
            raise RuntimeError("The trained cat/dog eye pose model is not loaded.")

        x1 = max(0, min(image_w, int(math.floor(float(bbox[0])))))
        y1 = max(0, min(image_h, int(math.floor(float(bbox[1])))))
        x2 = max(0, min(image_w, int(math.ceil(float(bbox[2])))))
        y2 = max(0, min(image_h, int(math.ceil(float(bbox[3])))))

        if x2 <= x1 or y2 <= y1:
            logger.warning("Skipping eye localization for an invalid animal bounding box.")
            return None, None
        if polygon is None:
            logger.warning("Skipping eye localization because the instance mask is unavailable.")
            return None, None

        polygon_array = np.asarray(polygon, dtype=np.float32)
        if polygon_array.ndim != 2 or polygon_array.shape[0] < 3 or polygon_array.shape[1] < 2:
            logger.warning("Skipping eye localization because the instance mask polygon is invalid.")
            return None, None

        instance_mask = np.zeros((image_h, image_w), dtype=np.uint8)
        polygon_points = np.rint(polygon_array[:, :2]).astype(np.int32)
        cv2.fillPoly(instance_mask, [polygon_points], 255)
        if cv2.countNonZero(instance_mask[y1:y2, x1:x2]) == 0:
            logger.warning("Skipping eye localization because the mask does not overlap its animal box.")
            return None, None

        box_width = x2 - x1
        box_height = y2 - y1
        crop_x1 = max(0, int(math.floor(x1 - box_width * self.pose_crop_margin)))
        crop_y1 = max(0, int(math.floor(y1 - box_height * self.pose_crop_margin)))
        crop_x2 = min(image_w, int(math.ceil(x2 + box_width * self.pose_crop_margin)))
        crop_y2 = min(image_h, int(math.ceil(y2 + box_height * self.pose_crop_margin)))
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            logger.warning("Skipping eye localization because the expanded animal crop is empty.")
            return None, None

        def is_inside_instance(x: float, y: float) -> bool:
            px, py = int(round(x)), int(round(y))
            return (
                x1 <= px < x2
                and y1 <= py < y2
                and instance_mask[py, px] > 0
            )

        pose_result = self.pose_model.predict(
            source=crop,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )[0]
        pose_boxes = getattr(pose_result, "boxes", None)
        pose_keypoints = getattr(pose_result, "keypoints", None)
        if pose_boxes is None or pose_keypoints is None or len(pose_boxes) == 0:
            return None, None

        predicted_boxes = pose_boxes.xyxy.cpu().numpy()
        predicted_classes = pose_boxes.cls.cpu().numpy().astype(int)
        keypoints = pose_keypoints.data.cpu().numpy()
        names = pose_result.names

        best_index = None
        best_iou = 0.0
        for index, (pose_box, class_id) in enumerate(zip(predicted_boxes, predicted_classes)):
            predicted_class_name = str(names[int(class_id)]).lower()
            if predicted_class_name != class_name.lower():
                continue

            full_image_box = [
                float(pose_box[0] + crop_x1),
                float(pose_box[1] + crop_y1),
                float(pose_box[2] + crop_x1),
                float(pose_box[3] + crop_y1),
            ]
            overlap = self.calculate_box_iou(bbox, full_image_box)
            if overlap > best_iou:
                best_index = index
                best_iou = overlap

        if best_index is None or best_iou < 0.1 or len(keypoints[best_index]) < 2:
            return None, None

        def make_eye(keypoint: np.ndarray) -> Optional[Dict[str, float]]:
            x = float(keypoint[0] + crop_x1)
            y = float(keypoint[1] + crop_y1)
            confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
            if confidence < self.pose_keypoint_conf_threshold or not is_inside_instance(x, y):
                return None
            return {"x": round(x, 2), "y": round(y, 2), "conf": round(confidence, 3)}

        left_eye = make_eye(keypoints[best_index][0])
        right_eye = make_eye(keypoints[best_index][1])
        return left_eye, right_eye

    def infer(
        self,
        image_source: Union[str, Path, np.ndarray, bytes],
        image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full Two-Stage inference pipeline:
        1. Stage 1: Segment individual animal contours (YOLOv8-seg) with Cross-Class NMS.
        2. Stage 2: Localize dual-eye keypoints strictly within each segmented animal contour.
        3. Task 3: Compute single-animal interocular distance (px & mm).
        4. Task 4: Compute cross-individual right-eye distance (px & mm).
        """
        image = self.read_image(image_source)
        h, w = image.shape[:2]
        img_id = image_id or ("image_demo" if not isinstance(image_source, (str, Path)) else Path(image_source).name)

        if self.seg_model is None:
            raise RuntimeError("The animal segmentation model is not loaded.")

        detected_animals: List[Dict[str, Any]] = []

        # Stage 1: Run instance segmentation to obtain each animal mask and box.
        seg_results = self.seg_model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            agnostic_nms=True,
            device=self.device,
            verbose=False,
        )[0]

        boxes = getattr(seg_results, "boxes", None)
        masks = getattr(seg_results, "masks", None)

        if boxes is not None and len(boxes) > 0:
            candidates = []
            for idx in range(len(boxes)):
                cls_id = int(boxes[idx].cls[0].cpu().numpy())
                cls_name = str(seg_results.names.get(cls_id, cls_id)).lower()
                if cls_name not in COCO_ANIMAL_CLASSES.values():
                    continue

                box = boxes[idx].xyxy[0].cpu().numpy().tolist()
                confidence = float(boxes[idx].conf[0].cpu().numpy())
                candidates.append(
                    {"idx": idx, "cls_name": cls_name, "box": box, "conf": confidence}
                )

            candidates.sort(key=lambda item: item["conf"], reverse=True)
            kept_candidates = []
            for candidate in candidates:
                if any(
                    self.calculate_box_iou(candidate["box"], kept["box"]) > 0.40
                    for kept in kept_candidates
                ):
                    continue
                kept_candidates.append(candidate)

            for animal_number, candidate in enumerate(kept_candidates, start=1):
                idx = candidate["idx"]
                class_name = candidate["cls_name"]
                box = candidate["box"]

                mask_polygon = None
                mask_area_px = 0.0
                if masks is not None and len(masks) > idx:
                    polygon = masks.xy[idx]
                    if len(polygon) > 0:
                        mask_polygon = polygon.tolist()
                        mask_area_px = float(
                            cv2.contourArea(np.asarray(polygon, dtype=np.float32))
                        )

                if mask_area_px == 0.0:
                    mask_area_px = float(
                        (box[2] - box[0]) * (box[3] - box[1]) * 0.75
                    )

                # Stage 2: Run the cat/dog pose model on this mask's expanded box.
                left_eye, right_eye = self._detect_eyes_within_mask(
                    image, box, mask_polygon, class_name, candidate["conf"]
                )

                interocular_px = None
                interocular_mm = None
                if left_eye is not None and right_eye is not None:
                    interocular_px = round(
                        self.euclidean_distance(
                            (left_eye["x"], left_eye["y"]),
                            (right_eye["x"], right_eye["y"]),
                        ),
                        2,
                    )
                    interocular_mm = self.pixel_to_mm(interocular_px)

                detected_animals.append(
                    {
                        "animal_id": f"animal_{animal_number}",
                        "class_name": class_name,
                        "bbox": [round(coordinate, 2) for coordinate in box],
                        "bbox_confidence": round(candidate["conf"], 3),
                        "mask_polygon": mask_polygon,
                        "mask_area_px": round(mask_area_px, 2),
                        "left_eye": left_eye,
                        "right_eye": right_eye,
                        "interocular_dist_px": interocular_px,
                        "interocular_dist_mm": interocular_mm,
                    }
                )

        # Task 4: Cross-individual distance between right eyes of arbitrary animal pairs
        pairwise_measurements: List[Dict[str, Any]] = []
        n_animals = len(detected_animals)

        for i in range(n_animals):
            for j in range(i + 1, n_animals):
                a1 = detected_animals[i]
                a2 = detected_animals[j]

                r1 = a1.get("right_eye")
                r2 = a2.get("right_eye")

                if r1 and r2 and r1.get("x") is not None and r2.get("x") is not None:
                    cross_dist_px = round(
                        self.euclidean_distance((r1["x"], r1["y"]), (r2["x"], r2["y"])), 2
                    )
                    cross_dist_mm = self.pixel_to_mm(cross_dist_px)
                else:
                    cross_dist_px = None
                    cross_dist_mm = None

                pairwise_measurements.append(
                    {
                        "image_id": img_id,
                        "animal_1_id": a1["animal_id"],
                        "animal_2_id": a2["animal_id"],
                        "animal_1_left_eye": f"({a1['left_eye']['x']},{a1['left_eye']['y']})" if a1.get("left_eye") else "N/A",
                        "animal_1_right_eye": f"({r1['x']},{r1['y']})" if r1 else "N/A",
                        "animal_1_interocular_dist_mm": a1.get("interocular_dist_mm"),
                        "animal_2_left_eye": f"({a2['left_eye']['x']},{a2['left_eye']['y']})" if a2.get("left_eye") else "N/A",
                        "animal_2_right_eye": f"({r2['x']},{r2['y']})" if r2 else "N/A",
                        "animal_2_interocular_dist_mm": a2.get("interocular_dist_mm"),
                        "right_eye_to_right_eye_dist_px": cross_dist_px,
                        "right_eye_to_right_eye_dist_mm": cross_dist_mm,
                    }
                )

        return {
            "image_id": img_id,
            "image_shape": {"height": h, "width": w, "channels": image.shape[2] if len(image.shape) > 2 else 1},
            "pixel_to_mm_scale": self.pixel_to_mm_scale,
            "detected_animals_count": len(detected_animals),
            "animals": detected_animals,
            "pairwise_measurements": pairwise_measurements,
        }

    def export_to_csv(
        self,
        measurements: List[Dict[str, Any]],
        output_path: Optional[Union[str, Path]] = None,
        append: bool = False,
    ) -> Path:
        """
        Export measurement results to CSV matching standard specification.
        """
        csv_file = Path(output_path or os.getenv("OUTPUT_CSV_PATH", "outputs/measurement_results.csv"))
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        columns = [
            "image_id",
            "animal_1_id",
            "animal_2_id",
            "animal_1_left_eye",
            "animal_1_right_eye",
            "animal_1_interocular_dist_mm",
            "animal_2_interocular_dist_mm",
            "right_eye_to_right_eye_dist_mm",
        ]

        formatted_rows = []
        for item in measurements:
            row = {
                "image_id": item.get("image_id", "N/A"),
                "animal_1_id": item.get("animal_1_id", "N/A"),
                "animal_2_id": item.get("animal_2_id", "N/A"),
                "animal_1_left_eye": item.get("animal_1_left_eye", "N/A"),
                "animal_1_right_eye": item.get("animal_1_right_eye", "N/A"),
                "animal_1_interocular_dist_mm": item.get("animal_1_interocular_dist_mm", 0.0),
                "animal_2_interocular_dist_mm": item.get("animal_2_interocular_dist_mm", 0.0),
                "right_eye_to_right_eye_dist_mm": item.get("right_eye_to_right_eye_dist_mm", 0.0),
            }
            formatted_rows.append(row)

        df = pd.DataFrame(formatted_rows, columns=columns)

        if append and csv_file.exists():
            df.to_csv(csv_file, mode="a", index=False, header=False)
        else:
            df.to_csv(csv_file, index=False)

        logger.info("Successfully exported %d measurement records to: %s", len(df), csv_file)
        return csv_file

    def annotate_image(
        self,
        image_source: Union[str, Path, np.ndarray],
        inference_results: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> np.ndarray:
        """
        Draw YOLOv8-seg animal contour masks, individual eye bounding boxes,
        measurement lines, and a separate panel for measurement labels.
        """
        img = self.read_image(image_source).copy()
        image_height, image_width = img.shape[:2]
        animals = inference_results.get("animals", [])
        pairs = inference_results.get("pairwise_measurements", [])

        # 1. Draw each segmented animal contour and its individual eyes
        for animal in animals:
            aid = animal["animal_id"]
            cname = animal.get("class_name", "animal")
            box = animal.get("bbox")
            polygon = animal.get("mask_polygon")

            # 1a. Draw Semantic / Instance Segmentation Contour (YOLOv8-seg Mask)
            if polygon and len(polygon) > 0:
                pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                # Distinct contour outline
                cv2.polylines(img, [pts], isClosed=True, color=(0, 230, 115), thickness=3)
                # Translucent mask overlay
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], color=(0, 230, 115))
                cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
            elif box:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 115), 2)

            if box:
                x1, y1, x2, y2 = map(int, box)
                cv2.putText(
                    img,
                    f"{aid}:{cname} (Conf: {animal.get('bbox_confidence', 0):.2f})",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 230, 115),
                    2,
                )

            # 1b. Individual eye keypoints
            le = animal.get("left_eye")
            re = animal.get("right_eye")

            if le and le.get("x") is not None:
                lx, ly = int(le["x"]), int(le["y"])
                cv2.circle(img, (lx, ly), 5, (255, 80, 0), -1)
                cv2.putText(
                    img,
                    "L",
                    (lx + 6, ly - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 80, 0),
                    2,
                )

            if re and re.get("x") is not None:
                rx, ry = int(re["x"]), int(re["y"])
                cv2.circle(img, (rx, ry), 5, (0, 80, 255), -1)
                cv2.putText(
                    img,
                    "R",
                    (rx + 6, ry - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 80, 255),
                    2,
                )

            # Task 3: Draw single-animal interocular distance line
            if le and re and animal.get("interocular_dist_mm") is not None:
                lx, ly = int(le["x"]), int(le["y"])
                rx, ry = int(re["x"]), int(re["y"])
                cv2.line(img, (lx, ly), (rx, ry), (0, 255, 255), 2)

        # 2. Task 4: Draw cross-individual right-eye distance lines
        for pair in pairs:
            dist_mm = pair.get("right_eye_to_right_eye_dist_mm")
            try:
                r1_str = pair.get("animal_1_right_eye", "").strip("()")
                r2_str = pair.get("animal_2_right_eye", "").strip("()")
                if "," in r1_str and "," in r2_str and dist_mm is not None:
                    p1 = tuple(map(int, map(float, r1_str.split(","))))
                    p2 = tuple(map(int, map(float, r2_str.split(","))))
                    cv2.line(img, p1, p2, (255, 0, 255), 2)
            except Exception as e:
                logger.debug("Annotation parse error: %s", e)

        valid_pairs = [
            pair for pair in pairs
            if pair.get("right_eye_to_right_eye_dist_mm") is not None
        ]
        panel_width = 360
        row_height = 24
        panel_height = max(
            image_height,
            180 + (len(animals) + len(valid_pairs)) * row_height,
        )
        canvas = np.full(
            (panel_height, image_width + panel_width, 3),
            (245, 245, 245),
            dtype=np.uint8,
        )
        canvas[:image_height, :image_width] = img
        cv2.rectangle(
            canvas,
            (image_width, 0),
            (image_width + panel_width - 1, panel_height - 1),
            (35, 35, 35),
            thickness=-1,
        )

        text_x = image_width + 16
        text_y = 32
        cv2.putText(
            canvas,
            "Measurement Summary",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        text_y += 30
        cv2.putText(
            canvas,
            "Interocular distance",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
        text_y += 22
        for animal in animals:
            distance = animal.get("interocular_dist_mm")
            value = f"{distance:.2f} mm" if distance is not None else "N/A"
            label = f"{animal['animal_id']} ({animal.get('class_name', 'animal')}): {value}"
            cv2.putText(
                canvas,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            text_y += row_height

        text_y += 4
        cv2.line(
            canvas,
            (text_x, text_y),
            (image_width + panel_width - 16, text_y),
            (95, 95, 95),
            1,
        )
        text_y += 24
        cv2.putText(
            canvas,
            "Right-eye pair distance",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 100, 255),
            1,
            cv2.LINE_AA,
        )
        text_y += 22
        if valid_pairs:
            for pair in valid_pairs:
                distance = pair["right_eye_to_right_eye_dist_mm"]
                label = (
                    f"{pair['animal_1_id']} - {pair['animal_2_id']}: "
                    f"{distance:.2f} mm"
                )
                cv2.putText(
                    canvas,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                text_y += row_height
        else:
            cv2.putText(
                canvas,
                "No valid pairwise measurements",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (190, 190, 190),
                1,
                cv2.LINE_AA,
            )
            text_y += row_height

        cv2.putText(
            canvas,
            f"Scale: {self.pixel_to_mm_scale:g} mm/px",
            (text_x, panel_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_p), canvas)
            logger.info("Saved annotated image to: %s", out_p)

        return canvas


if __name__ == "__main__":
    logger.info("Executing Two-Stage Animal Metrology Pipeline demo...")
    pipeline = AnimalMetrologyPipeline()

    dummy_img = np.full((720, 1280, 3), 245, dtype=np.uint8)
    demo_results = pipeline.infer(dummy_img, image_id="demo_two_stage.jpg")

    logger.info("Detected animals: %d", demo_results["detected_animals_count"])
    for a in demo_results["animals"]:
        logger.info(
            "Animal: %s (%s), Mask Area: %s px, Eye Distance: %s mm",
            a["animal_id"],
            a.get("class_name"),
            a["mask_area_px"],
            a["interocular_dist_mm"],
        )

    csv_out = pipeline.export_to_csv(demo_results["pairwise_measurements"])
    print(f"\n[OK] CSV updated at: {csv_out}")
