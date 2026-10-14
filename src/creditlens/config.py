"""Central configuration. Values can be overridden with environment variables or a .env file."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

PROJECT_ROOT = Path(os.path.abspath(__file__)).parents[2]
KNOWLEDGE_BASE_DIR = Path(os.getenv("CREDITLENS_KB_DIR", PROJECT_ROOT / "knowledge_base"))
ARTIFACTS_DIR = Path(os.getenv("CREDITLENS_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

RANDOM_SEED = int(os.getenv("CREDITLENS_SEED", "42"))
TRAINING_ROWS = int(os.getenv("CREDITLENS_TRAINING_ROWS", "20000"))
CV_FOLDS = int(os.getenv("CREDITLENS_CV_FOLDS", "5"))

# LLM provider: "auto", "mock", "gemini", "azure_openai" or "openai".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
MAX_MEMO_ATTEMPTS = int(os.getenv("CREDITLENS_MAX_MEMO_ATTEMPTS", "2"))
