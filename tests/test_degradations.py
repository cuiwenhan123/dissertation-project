import unittest

import numpy as np
from PIL import Image

from backend.images import degradation_parameters, degrade


class DegradationTests(unittest.TestCase):
    def setUp(self):
        array = np.zeros((40, 60, 3), dtype=np.uint8)
        array[:, 20:40] = 255
        self.image = Image.fromarray(array)

    def test_zero_is_identity(self):
        output = degrade(self.image, "blur", 0)
        self.assertTrue(np.array_equal(np.asarray(output), np.asarray(self.image)))

    def test_lowlight_is_seeded_and_darker(self):
        first = degrade(self.image, "lowlight", 3, seed=42)
        second = degrade(self.image, "lowlight", 3, seed=42)
        self.assertTrue(np.array_equal(np.asarray(first), np.asarray(second)))
        self.assertLess(float(np.asarray(first).mean()), float(np.asarray(self.image).mean()))

    def test_motion_blur_and_parameters(self):
        output = degrade(self.image, "blur", 5)
        self.assertFalse(np.array_equal(np.asarray(output), np.asarray(self.image)))
        self.assertEqual(degradation_parameters("blur", 5)["kernelLength"], 21)


if __name__ == "__main__":
    unittest.main()
