"""Core configuration and settings."""

import os
import torch
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API settings
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# Auto-detect best device: cuda > mps (Apple Silicon) > cpu
if os.getenv("DEVICE"):
    DEVICE = os.getenv("DEVICE")
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# FP16 only safe on CUDA
FP16 = os.getenv("FP16", "true").lower() == "true" if DEVICE == "cuda" else False

# Refinement settings
REFINE_SUBDIVISIONS = int(os.getenv("REFINE_SUBDIVISIONS", "1"))
TEXTURE_RESOLUTION = int(os.getenv("TEXTURE_RESOLUTION", "1024"))

# Performance
KEEP_MODELS_IN_MEMORY = os.getenv("KEEP_MODELS_IN_MEMORY", "true").lower() == "true"
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))

# Job cleanup
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))  # 1 hour
