import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACKED_SOURCE_PATTERNS = (
    "*.md",
    "*.py",
    "*.json",
    "*.js",
    "*.html",
    "*.css",
    "*.toml",
    "*.yml",
    "*.yaml",
)


class ReleaseHygieneTests(unittest.TestCase):
    def test_documented_clone_url_is_the_public_repository(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("https://github.com/cuiwenhan123/dissertation-project.git", readme)
        self.assertNotIn("YOUR_USERNAME", readme)

    def test_tracked_text_has_no_machine_specific_home_path(self):
        offenders = []
        for pattern in TRACKED_SOURCE_PATTERNS:
            for path in ROOT.rglob(pattern):
                if any(part in {".git", ".venv", "build", "tmp"} for part in path.parts):
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if re.search(r"/(?:Users|home)/[^/]+/", text):
                    offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
