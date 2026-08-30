from __future__ import annotations

import argparse
from pathlib import Path

from backend.config import COMPLETED_STUDY_ARCHIVE
from backend.study_import import import_completed_archive


DEFAULT_ARCHIVE = COMPLETED_STUDY_ARCHIVE


def main() -> None:
    parser = argparse.ArgumentParser(description="Import the completed Chapter 4 study into the workbench database.")
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    imported = import_completed_archive(args.archive)
    print(
        f"Synced {imported['id']}: {imported['imageCount']} images, "
        f"{imported['rowCount']} reported condition rows."
    )


if __name__ == "__main__":
    main()
