import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from backend.domain import Box
from backend import studies


def tiny_dataset() -> bytes:
    image_buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(image_buffer, format="JPEG")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("images/example.jpg", image_buffer.getvalue())
        archive.writestr("labels/example.txt", "0 0.5 0.5 0.5 0.5\n")
    return archive_buffer.getvalue()


class StudyRunnerTests(unittest.TestCase):
    def test_full_matrix_uses_real_runner_contract(self):
        study_id = "study-test"
        studies.JOBS[study_id] = {"id": study_id, "status": "queued"}
        studies.JOB_CANCEL[study_id] = __import__("threading").Event()

        def detector(_model, _image):
            return [Box(16, 16, 32, 32, "person", "medium", 0.99)], "test-backend"

        with patch.object(studies, "run_uploaded_detector", detector), \
             patch.object(studies, "_file_hash", return_value="hash"), \
             patch.object(studies, "save_study"):
            studies._run_study(
                study_id,
                studies.DatasetSource("zip", "tiny.zip", raw=tiny_dataset()),
                1,
                2026,
            )

        job = studies.get_study_status(study_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(len(job["result"]["rows"]), 36)
        self.assertEqual(job["completedTasks"], 32)
        self.assertEqual(job["result"]["config"]["evaluator"], "pycocotools COCOeval bbox")
        studies.JOBS.pop(study_id, None)
        studies.JOB_CANCEL.pop(study_id, None)

    def test_balanced_selection_is_deterministic(self):
        candidates = [
            {"name": f"image-{index}.jpg", "labels": {"common", f"class-{index % 4}"}}
            for index in range(30)
        ]
        first = studies._balanced_select(candidates, 10, 2026)
        second = studies._balanced_select(candidates, 10, 2026)
        self.assertEqual([item["name"] for item in first], [item["name"] for item in second])
        self.assertEqual(len({label for item in first for label in item["labels"]}), 5)

    def test_coco_directory_loader_keeps_images_lazy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            Image.new("RGB", (64, 48), "white").save(image_root / "sample.jpg")
            annotation_path = root / "instances.json"
            annotation_path.write_text(json.dumps({
                "images": [{"id": 1, "file_name": "sample.jpg", "width": 64, "height": 48}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [8, 8, 20, 20]}],
                "categories": [{"id": 1, "name": "person"}],
            }))
            samples, info = studies.load_dataset_samples(
                studies.DatasetSource("directory", "test", image_root=image_root, annotation_path=annotation_path),
                5000,
                2026,
            )
            self.assertEqual(len(samples), 1)
            self.assertNotIn("image", samples[0])
            self.assertEqual(info["samplingMethod"], "all eligible images")

    def test_directory_image_reader_retries_transient_decode_error(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            Image.new("RGB", (32, 24), "white").save(image_path)
            source = studies.DatasetSource("directory", "test", image_root=Path(directory))
            sample = {"name": "sample.jpg", "imagePath": str(image_path)}
            original_open = studies.Image.open
            attempts = {"count": 0}

            def flaky_open(value):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise OSError("transient decode failure")
                return original_open(value)

            with patch.object(studies.Image, "open", side_effect=flaky_open):
                image = studies._load_sample_image(source, sample, None)
            self.assertEqual(image.size, (32, 24))
            self.assertEqual(attempts["count"], 2)


if __name__ == "__main__":
    unittest.main()
