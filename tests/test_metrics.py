import unittest

from backend.domain import Box
from backend.metrics import evaluate_detection_dataset


class MetricTests(unittest.TestCase):
    def sample(self, predictions):
        return [{
            "name": "example.jpg",
            "width": 200,
            "height": 200,
            "groundTruth": [Box(20, 20, 100, 100, "car", "large")],
            "predictions": predictions,
        }]

    def test_perfect_prediction_has_perfect_coco_ap(self):
        result = evaluate_detection_dataset(self.sample([Box(20, 20, 100, 100, "car", "large", 0.99)]))
        self.assertAlmostEqual(result["map"], 1.0, places=5)
        self.assertAlmostEqual(result["ap50"], 1.0, places=5)
        self.assertEqual(result["failures"]["missed"], 0)

    def test_duplicate_prediction_is_false_positive(self):
        result = evaluate_detection_dataset(self.sample([
            Box(20, 20, 100, 100, "car", "large", 0.99),
            Box(20, 20, 100, 100, "car", "large", 0.50),
        ]))
        self.assertEqual(result["failures"]["falsePositive"], 1)

    def test_missing_prediction_returns_zero(self):
        result = evaluate_detection_dataset(self.sample([]))
        self.assertEqual(result["map"], 0.0)
        self.assertEqual(result["failures"]["missed"], 1)


if __name__ == "__main__":
    unittest.main()
