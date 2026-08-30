import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_environment_overrides_runtime_and_asset_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = {
                "ROBUSTNESS_HOST": "0.0.0.0",
                "ROBUSTNESS_PORT": "9010",
                "ROBUSTNESS_USE_REAL_MODELS": "false",
                "ROBUSTNESS_ALLOW_MODEL_DOWNLOAD": "yes",
                "ROBUSTNESS_COCO_VAL2017_ROOT": str(root / "coco"),
                "ROBUSTNESS_DETR_MODEL_PATH": str(root / "detr"),
                "ROBUSTNESS_FASTER_RCNN_MODEL_PATH": str(root / "faster.pth"),
                "ROBUSTNESS_BUILTIN_DATASET_PATH": str(root / "pilot.zip"),
            }
            with patch.dict(os.environ, environment, clear=True):
                settings = Settings.from_environment(root)

        self.assertEqual(settings.host, "0.0.0.0")
        self.assertEqual(settings.port, 9010)
        self.assertFalse(settings.use_real_models)
        self.assertTrue(settings.allow_model_download)
        self.assertEqual(settings.detr_model_path, root / "detr")
        self.assertEqual(settings.faster_rcnn_model_path, root / "faster.pth")
        self.assertEqual(settings.builtin_dataset_path, root / "pilot.zip")


if __name__ == "__main__":
    unittest.main()
