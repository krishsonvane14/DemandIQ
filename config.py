# config.py
#
# Central configuration file for DemandIQ.
# Defines paths, model hyperparameters, and environment settings
# that are shared across the project.
#
# Usage: import config and access settings as attributes, e.g.
#   from config import RAW_DATA_DIR, MODEL_PARAMS

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")

# ── Model hyperparameters ──────────────────────────────────────────────────────
MODEL_PARAMS = {
    # TODO: populate with model-specific parameters
}

# ── Environment ────────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEMANDIQ_DEBUG", "false").lower() == "true"
