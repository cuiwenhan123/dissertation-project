import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("ROBUSTNESS_USE_REAL_MODELS", "1")
os.environ.setdefault("ROBUSTNESS_ALLOW_MODEL_DOWNLOAD", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
runpy.run_path(str(ROOT / "server.py"), run_name="__main__")
