from __future__ import annotations

import unittest

from backend.study_import import validate_completed_archive


def valid_archive() -> dict:
    row = {
        "model": "transformer",
        "degradation": "blur",
        "severity": 0,
        "map": 0.5,
        "ap50": 0.7,
        "smallAP": 0.2,
        "mediumAP": 0.4,
        "largeAP": 0.6,
        "retention": 1.0,
        "classMetrics": {},
    }
    rows = [dict(row, severity=index % 6) for index in range(36)]
    return {
        "schemaVersion": 1,
        "study": {
            "id": "chapter4-coco-500-seed-2026",
            "status": "completed",
            "config": {"dataset": {"evaluatedImages": 500}},
            "result": {
                "id": "chapter4-coco-500-seed-2026",
                "rows": rows,
                "summary": {"runCount": 36},
            },
        },
    }


class StudyImportTests(unittest.TestCase):
    def test_valid_completed_archive(self) -> None:
        validate_completed_archive(valid_archive())

    def test_rejects_wrong_image_count(self) -> None:
        archive = valid_archive()
        archive["study"]["config"]["dataset"]["evaluatedImages"] = 128
        with self.assertRaisesRegex(ValueError, "500-image"):
            validate_completed_archive(archive)

    def test_rejects_incomplete_rows(self) -> None:
        archive = valid_archive()
        del archive["study"]["result"]["rows"][0]["classMetrics"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_completed_archive(archive)


if __name__ == "__main__":
    unittest.main()
