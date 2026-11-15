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

    def __getitem__(self, index):
        return FakeTensor(self.values[index])


class FakeBoxes:
    def __init__(self, boxes, classes, confidences=None):
        self.xyxy = FakeTensor(boxes)
        self.cls = FakeTensor(classes)
        self.conf = FakeTensor(confidences or [0.9] * len(boxes))

    def __len__(self):
        return len(self.xyxy.values)


class FakeSegBox:
    def __init__(self, box, class_id, confidence):
        self.xyxy = FakeTensor([box])
        self.cls = FakeTensor([class_id])
        self.conf = FakeTensor([confidence])

    def __len__(self):
        return 1


class FakeSegBoxes(list):
    pass


class FakeMasks:
    def __init__(self, polygons):
        self.xy = [np.asarray(polygon, dtype=np.float32) for polygon in polygons]

    def __len__(self):
        return len(self.xy)


class FakePoseModel:
    def __init__(self, keypoints, boxes=None, classes=None, names=None, confidences=None):
        self.keypoints = keypoints
        self.boxes = boxes or [[12, 12, 72, 72]]
        self.classes = classes or [0]
        self.confidences = confidences
        self.names = names or {0: "cat", 1: "dog"}
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        result = SimpleNamespace(
            boxes=FakeBoxes(self.boxes, self.classes, self.confidences),
            keypoints=SimpleNamespace(data=FakeTensor(self.keypoints)),
            names=self.names,
        )
        return [result]


class EyeMaskTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = object.__new__(AnimalMetrologyPipeline)
        self.pipeline.pose_model = None
        self.pipeline.conf_threshold = 0.25
        self.pipeline.pose_keypoint_conf_threshold = 0.1
        self.pipeline.pose_crop_margin = 0.08
        self.pipeline.device = "cpu"
        self.image = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.bbox = [20, 20, 80, 80]
        self.square_mask = [[20, 20], [80, 20], [80, 80], [20, 80]]

    def test_pose_points_are_mapped_from_masked_contour_crop(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[25, 30, 0.9], [55, 30, 0.8]]], dtype=float)
        )

        left_eye, right_eye = self.pipeline._predict_eyes_in_segmented_roi(
            self.image, self.square_mask
        )

        self.assertEqual((left_eye["x"], left_eye["y"]), (41.0, 46.0))
        self.assertEqual((right_eye["x"], right_eye["y"]), (71.0, 46.0))
        self.assertTrue(left_eye["inside_mask"])
        self.assertTrue(right_eye["inside_mask"])
        crop = self.pipeline.pose_model.calls[0]["source"]
        self.assertEqual(crop.shape[:2], (69, 69))

    def test_candidate_outside_mask_is_returned_and_marked_unverified(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[42, 62, 0.9], [70, 14, 0.9]]], dtype=float)
        )
        triangle = [[50, 20], [80, 80], [20, 80]]

        left_eye, right_eye = self.pipeline._predict_eyes_in_segmented_roi(
            self.image, triangle
        )

        self.assertIsNotNone(left_eye)
        self.assertIsNotNone(right_eye)
        self.assertTrue(left_eye["inside_mask"])
        self.assertFalse(right_eye["inside_mask"])

    def test_pose_class_is_not_filtered_against_segmentation_class(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[30, 30, 0.9], [70, 30, 0.9]]], dtype=float),
            classes=[1],
        )

        left_eye, right_eye = self.pipeline._predict_eyes_in_segmented_roi(
            self.image, self.square_mask
        )

        self.assertIsNotNone(left_eye)
        self.assertIsNotNone(right_eye)

    def test_missing_mask_returns_no_eye_coordinates(self):
        self.pipeline.pose_model = FakePoseModel(
            np.array([[[30, 30, 0.9], [70, 30, 0.9]]], dtype=float)
        )

        self.assertEqual(
            self.pipeline._predict_eyes_in_segmented_roi(
                self.image, None
            ),
            (None, None),
        )
        self.assertEqual(self.pipeline.pose_model.calls, [])


class SegmentedPosePipelineTests(unittest.TestCase):
    def test_runs_pose_for_non_catdog_segments_and_computes_both_pair_distances(self):
        pipeline = object.__new__(AnimalMetrologyPipeline)
        pipeline.pose_model = FakePoseModel(
            np.array([[[25, 30, 0.9], [55, 30, 0.8]]], dtype=float),
            classes=[1],
        )
        pipeline.seg_model = SimpleNamespace(
            predict=lambda **_: [
                SimpleNamespace(
                    boxes=FakeSegBoxes(
                        [
                            FakeSegBox([20, 20, 80, 80], 17, 0.32),
                            FakeSegBox([110, 20, 170, 80], 23, 0.29),
                        ]
                    ),
                    masks=FakeMasks(
                        [
                            [[20, 20], [80, 20], [80, 80], [20, 80]],
                            [[110, 20], [170, 20], [170, 80], [110, 80]],
                        ]
                    ),
                    names={17: "horse", 23: "giraffe"},
                )
            ]
        )
        pipeline.conf_threshold = 0.25
        pipeline.iou_threshold = 0.45
        pipeline.pose_keypoint_conf_threshold = 0.1
        pipeline.pose_crop_margin = 0.08
        pipeline.device = "cpu"
        pipeline.pixel_to_mm_scale = 0.15
        image = np.full((100, 200, 3), 255, dtype=np.uint8)

        result = pipeline.infer(image, image_id="two_non_catdog.jpg")

        self.assertEqual([animal["class_name"] for animal in result["animals"]], ["horse", "giraffe"])
        self.assertEqual(result["animals"][0]["interocular_dist_px"], 30.0)
        self.assertEqual(result["animals"][0]["pose_candidate_interocular_dist_mm"], 4.5)
        self.assertEqual(
            result["pairwise_measurements"][0][
                "pose_candidate_right_eye_to_right_eye_dist_px"
            ],
            90.0,
        )
        self.assertEqual(
            result["pairwise_measurements"][0][
                "pose_candidate_right_eye_to_right_eye_dist_mm"
            ],
            13.5,
        )


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
