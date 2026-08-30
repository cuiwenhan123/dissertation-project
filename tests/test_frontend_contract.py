import re
import unittest
from pathlib import Path

from backend.config import STATIC_DIR
from backend.routes import GET_API_ROUTES, POST_ROUTES, resolve_static_asset


VIEW_NAMES = {
    "overview",
    "dataset",
    "classAnalysis",
    "failures",
    "transitions",
    "comparison",
    "curves",
    "benchmark",
    "report",
    "log",
    "methodology",
}


class FrontendContractTests(unittest.TestCase):
    def test_navigation_targets_have_one_view_fragment(self):
        index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        targets = set(re.findall(r'data-target-page="([^"]+)"', index))
        fragments = {path.stem for path in (STATIC_DIR / "views").glob("*.html")}
        self.assertEqual(targets, VIEW_NAMES)
        self.assertEqual(fragments, VIEW_NAMES)
        for name in VIEW_NAMES:
            self.assertIsNotNone(resolve_static_asset(f"/views/{name}.html"))

    def test_view_ids_are_unique_across_fragments(self):
        identifiers = []
        for path in sorted((STATIC_DIR / "views").glob("*.html")):
            identifiers.extend(re.findall(r'id="([^"]+)"', path.read_text(encoding="utf-8")))
        duplicates = sorted({value for value in identifiers if identifiers.count(value) > 1})
        self.assertEqual(duplicates, [])

    def test_frontend_api_references_exist_in_backend_contract(self):
        references = set()
        get_references = set()
        post_references = set()
        for path in (STATIC_DIR / "js").rglob("*.js"):
            source = path.read_text(encoding="utf-8")
            references.update(re.findall(r'["`](/api/[A-Za-z0-9_/-]+)', source))
            get_references.update(
                re.findall(r'(?:getJson|requestJson)\(\s*["`](/api/[A-Za-z0-9_/-]+)', source)
            )
            post_references.update(
                re.findall(r'postJson\(\s*["`](/api/[A-Za-z0-9_/-]+)', source)
            )
        self.assertTrue(references)
        self.assertEqual(references - (GET_API_ROUTES | POST_ROUTES), set())
        self.assertEqual(get_references - GET_API_ROUTES, set())
        self.assertEqual(post_references - POST_ROUTES, set())

    def test_entrypoint_references_packaged_assets(self):
        index = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assets = re.findall(r'(?:href|src)="([^"?]+)', index)
        for asset in assets:
            with self.subTest(asset=asset):
                self.assertIsNotNone(resolve_static_asset(f"/{asset}"))


if __name__ == "__main__":
    unittest.main()
