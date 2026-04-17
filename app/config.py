from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
CONFIG_DIR = BASE_DIR / "configs"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

IMAGE_SIZE = int(os.getenv("DEFECT_IMAGE_SIZE", "640"))
PATCH_SIZE = int(os.getenv("DEFECT_PATCH_SIZE", "128"))
PATCH_STRIDE = int(os.getenv("DEFECT_PATCH_STRIDE", "96"))
VIDEO_SAMPLE_INTERVAL_SECONDS = float(os.getenv("DEFECT_VIDEO_INTERVAL", "0.75"))
BLUR_THRESHOLD = float(os.getenv("DEFECT_BLUR_THRESHOLD", "80"))
BRIGHTNESS_MIN = float(os.getenv("DEFECT_BRIGHTNESS_MIN", "40"))
BRIGHTNESS_MAX = float(os.getenv("DEFECT_BRIGHTNESS_MAX", "220"))

PRIMARY_CONFIDENCE = float(os.getenv("DEFECT_PRIMARY_CONF", "0.25"))
MERGE_CONFIDENCE = float(os.getenv("DEFECT_MERGE_CONF", "0.40"))
CLASSIFIER_THRESHOLD = float(os.getenv("DEFECT_CLASSIFIER_THRESHOLD", "0.70"))
VIDEO_MIN_FRAMES = int(os.getenv("DEFECT_VIDEO_MIN_FRAMES", "2"))

YOLO_PRIMARY_WEIGHTS = os.getenv("DEFECT_YOLO_WEIGHTS", str(MODEL_DIR / "yolov8m_defects.pt"))
YOLO_SEGMENT_WEIGHTS = os.getenv("DEFECT_YOLO_SEG_WEIGHTS", "")
CLASSIFIER_WEIGHTS = os.getenv("DEFECT_CLASSIFIER_WEIGHTS", "")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "crack",
    "hole",
    "dent",
    "rust",
    "corrosion",
    "paint_damage",
    "scratch",
    "leak",
]

SEVERITY_WEIGHTS = {
    "crack": 10.0,
    "hole": 10.0,
    "leak": 9.0,
    "dent": 6.0,
    "corrosion": 5.0,
    "rust": 4.0,
    "scratch": 3.0,
    "paint_damage": 2.0,
}

CRITICAL_CLASSES = {"crack", "hole", "leak"}

