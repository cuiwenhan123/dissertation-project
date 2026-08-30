import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import study_storage


class StudyStorageTests(unittest.TestCase):
    def test_completed_study_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "studies.sqlite3"
            with patch.object(study_storage, "STUDY_DB", database), patch.object(study_storage, "RUNS_DIR", Path(directory)):
                study_storage.save_study("study-1", "completed", {"seed": 1}, {"rows": []})
                stored = study_storage.latest_completed_study()
                self.assertEqual(stored["id"], "study-1")
                self.assertEqual(stored["config"]["seed"], 1)


if __name__ == "__main__":
    unittest.main()
