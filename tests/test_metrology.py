import unittest
from types import SimpleNamespace

import numpy as np

from src.metrology import AnimalMetrologyPipeline


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, boxes, classes):
        self.xyxy = FakeTensor(boxes)
        self.cls = FakeTensor(classes)

    def __len__(self):
        return len(self.xyxy.values)


class FakePoseModel:
    def __init__(self, keypoints, boxes=None, classes=None, names=None):
        self.keypoints = keypoints
        self.boxes = boxes or [[12, 12, 72, 72]]
        self.classes = classes or [0]
        self.names = names or {0: "cat", 1: "dog"}
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        result = SimpleNamespace(
            boxes=FakeBoxes(self.boxes, self.classes),
            keypoints=SimpleNamespace(data=FakeTensor(self.keypoints)),
            names=self.names,
        )
        return [result]


class EyeMaskTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = object.__new__(AnimalMetrologyPipeline)
        self.pipeline.pose_model = None
        self.pipeline.conf_threshold = 0.25
        self.pipeline.pose_animal_conf_threshold = 0.4
        self.pipeline.pose_keypoint_conf_threshold = 0.1
        self.pipeline.pose_crop_margin = 0.2
        self.pipeline.device = "cpu"
        self.image = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.bbox = [20, 20, 80, 80]
        self.square_mask = [[20, 20], [80, 20], [80, 80], [20, 80]]

    def test_pose_points_are_mapped_from_expanded_animal_crop(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[30, 30, 0.9], [70, 30, 0.8]]], dtype=float)
        )

        left_eye, right_eye = self.pipeline._detect_eyes_within_mask(
            self.image, self.bbox, self.square_mask, "cat"
        )

        self.assertEqual((left_eye["x"], left_eye["y"]), (38.0, 38.0))
        self.assertEqual((right_eye["x"], right_eye["y"]), (78.0, 38.0))
        crop = self.pipeline.pose_model.calls[0]["source"]
        self.assertEqual(crop.shape[:2], (84, 84))

    def test_points_outside_instance_mask_are_rejected_without_fallback(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[42, 62, 0.9], [70, 14, 0.9]]], dtype=float)
        )
        triangle = [[50, 20], [80, 80], [20, 80]]

        left_eye, right_eye = self.pipeline._detect_eyes_within_mask(
            self.image, self.bbox, triangle, "cat"
        )

        self.assertIsNotNone(left_eye)
        self.assertIsNone(right_eye)

    def test_pose_detection_must_match_segmented_animal_class(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[30, 30, 0.9], [70, 30, 0.9]]], dtype=float),
            classes=[1],
        )

        left_eye, right_eye = self.pipeline._detect_eyes_within_mask(
            self.image, self.bbox, self.square_mask, "cat"
        )

        self.assertIsNone(left_eye)
        self.assertIsNone(right_eye)

    def test_non_cat_dog_animals_do_not_get_eye_predictions(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[30, 30, 0.9], [70, 30, 0.9]]], dtype=float)
        )

        result = self.pipeline._detect_eyes_within_mask(
            self.image, self.bbox, self.square_mask, "horse"
        )

        self.assertEqual(result, (None, None))
        self.assertEqual(self.pipeline.pose_model.calls, [])

    def test_low_confidence_segmented_cat_does_not_get_eye_predictions(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[30, 30, 0.9], [70, 30, 0.9]]], dtype=float)
        )

        result = self.pipeline._detect_eyes_within_mask(
            self.image, self.bbox, self.square_mask, "cat", animal_confidence=0.32
        )

        self.assertEqual(result, (None, None))
        self.assertEqual(self.pipeline.pose_model.calls, [])

    def test_missing_mask_returns_no_eye_coordinates(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[30, 30, 0.9], [70, 30, 0.9]]], dtype=float)
        )

        self.assertEqual(
            self.pipeline._detect_eyes_within_mask(
                self.image, self.bbox, None, "cat"
            ),
            (None, None),
        )
        self.assertEqual(self.pipeline.pose_model.calls, [])


class AnnotationLayoutTests(unittest.TestCase):
    def test_measurement_labels_are_drawn_in_a_separate_panel(self):
        pipeline = object.__new__(AnimalMetrologyPipeline)
        pipeline.pixel_to_mm_scale = 0.15
        image = np.full((100, 160, 3), 255, dtype=np.uint8)
        pipeline.read_image = lambda _: image.copy()
        results = {
            "animals": [
                {
                    "animal_id": "animal_1",
                    "class_name": "cat",
                    "bbox": [10, 10, 80, 90],
                    "bbox_confidence": 0.9,
                    "left_eye": {"x": 30, "y": 35},
                    "right_eye": {"x": 50, "y": 35},
                    "interocular_dist_mm": 3.0,
                },
                {
                    "animal_id": "animal_2",
                    "class_name": "dog",
                    "bbox": [80, 10, 150, 90],
                    "bbox_confidence": 0.9,
                    "left_eye": {"x": 100, "y": 35},
                    "right_eye": {"x": 120, "y": 35},
                    "interocular_dist_mm": 3.0,
                },
            ],
            "pairwise_measurements": [
                {
                    "animal_1_id": "animal_1",
                    "animal_2_id": "animal_2",
                    "animal_1_right_eye": "(50,35)",
                    "animal_2_right_eye": "(120,35)",
                    "right_eye_to_right_eye_dist_mm": 10.5,
                }
            ],
        }

        annotated = pipeline.annotate_image(image, results)

        self.assertGreaterEqual(annotated.shape[0], image.shape[0])
        self.assertGreater(annotated.shape[1], image.shape[1])
        self.assertFalse(np.array_equal(annotated[99, 160], image[99, 159]))


if __name__ == "__main__":
    unittest.main()
