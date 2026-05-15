"""Runtime helpers for importing the installed ultralytics package."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BUNDLED_ULTRALYTICS_DIR = PROJECT_ROOT / "ultralytics"
BUNDLED_ULTRALYTICS_COMPLETE = (BUNDLED_ULTRALYTICS_DIR / "data").exists()


def _is_project_path(path_entry: str) -> bool:
    candidate = Path(path_entry or os.getcwd()).resolve()
    return candidate == PROJECT_ROOT


def _import_yolo():
    if BUNDLED_ULTRALYTICS_COMPLETE:
        from ultralytics import YOLO as ultralytics_yolo

        return ultralytics_yolo

    removed_entries = [entry for entry in sys.path if _is_project_path(entry)]
    sys.path[:] = [entry for entry in sys.path if not _is_project_path(entry)]

    try:
        from ultralytics import YOLO as ultralytics_yolo
    finally:
        sys.path[:0] = removed_entries

    return ultralytics_yolo


YOLO = _import_yolo()
