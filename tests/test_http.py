import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from backend.application import create_server
from backend.routes import resolve_static_asset


class HttpApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server = create_server("127.0.0.1", 0)
        except PermissionError as exc:
            raise unittest.SkipTest("Local socket binding is disabled in this environment") from exc
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address[:2]
        cls.base_url = f"http://{host}:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_status_endpoint_returns_json(self):
        with urlopen(f"{self.base_url}/api/status", timeout=2) as response:
            payload = json.load(response)
        self.assertIn("python", payload)
        self.assertIn("localModels", payload)

    def test_static_server_exposes_only_frontend_assets(self):
        with urlopen(f"{self.base_url}/", timeout=2) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/html", response.headers["Content-Type"])

        for path in ("/server.py", "/runs/experiments.sqlite3", "/../server.py"):
            with self.subTest(path=path), self.assertRaises(HTTPError) as raised:
                urlopen(f"{self.base_url}{path}", timeout=2)
            self.assertEqual(raised.exception.code, 404)


class StaticAssetPolicyTests(unittest.TestCase):
    def test_frontend_assets_are_allow_listed(self):
        self.assertEqual(resolve_static_asset("/").name, "index.html")
        self.assertEqual(resolve_static_asset("/js/app.js").name, "app.js")
        self.assertEqual(resolve_static_asset("/css/base.css").name, "base.css")
        self.assertEqual(resolve_static_asset("/views/overview.html").name, "overview.html")

    def test_private_project_files_are_not_static(self):
        for path in ("/server.py", "/runs/experiments.sqlite3", "/../server.py", "/.env"):
            with self.subTest(path=path):
                self.assertIsNone(resolve_static_asset(path))


if __name__ == "__main__":
    unittest.main()
