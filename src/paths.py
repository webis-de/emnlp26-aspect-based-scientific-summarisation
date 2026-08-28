"""Portable project paths with optional environment overrides."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("ABSS_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
RESULTS_ROOT = Path(os.getenv("ABSS_RESULTS_ROOT", PROJECT_ROOT / "results")).expanduser()
LOG_ROOT = Path(os.getenv("ABSS_LOG_ROOT", PROJECT_ROOT / "logs")).expanduser()
CACHE_ROOT = Path(os.getenv("ABSS_CACHE_ROOT", PROJECT_ROOT / ".cache")).expanduser()
TESTS_ROOT = PROJECT_ROOT / "tests"