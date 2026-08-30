import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("ROBUSTNESS_USE_REAL_MODELS", "1")
os.environ.setdefault("ROBUSTNESS_ALLOW_MODEL_DOWNLOAD", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from backend.application import main


if __name__ == "__main__":
    main()
