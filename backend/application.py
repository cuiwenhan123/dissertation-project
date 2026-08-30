from __future__ import annotations

import logging
import warnings
from http.server import ThreadingHTTPServer

from .config import SETTINGS
from .routes import Handler
from .runtime import runtime_status


LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = SETTINGS.log_level) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    warnings.filterwarnings("ignore", message="A NumPy version .* is required for this version of SciPy.*")
    warnings.filterwarnings("ignore", message="for .* copying from a non-meta parameter .*")


def create_server(host: str = SETTINGS.host, port: int = SETTINGS.port) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    configure_logging()
    server = create_server()
    host, port = server.server_address[:2]
    LOGGER.info("Detection Robustness Workbench listening on http://%s:%s/", host, port)
    LOGGER.info("Runtime status: %s", runtime_status())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
