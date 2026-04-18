from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np

from app.config import (
    BLUR_THRESHOLD,
    BRIGHTNESS_MAX,
    BRIGHTNESS_MIN,
    IMAGE_SIZE,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
)


@dataclass
class PreprocessedImage:
    image: np.ndarray
    original: np.ndarray
    scale: float
    pad_x: int
    pad_y: int


def is_supported_media(path: Path) -> bool:
    return (
        path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
    )


def is_video(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def read_image_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def letterbox(image: np.ndarray, size: int = IMAGE_SIZE) -> PreprocessedImage:
    original = image.copy()
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return PreprocessedImage(
        image=canvas, original=original, scale=scale, pad_x=pad_x, pad_y=pad_y
    )


def blur_score(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))


def is_low_quality(image_rgb: np.ndarray) -> bool:
    blur = blur_score(image_rgb)
    brightness = brightness_score(image_rgb)
    return (
        blur < BLUR_THRESHOLD
        or brightness < BRIGHTNESS_MIN
        or brightness > BRIGHTNESS_MAX
    )


def enhance_image(image_rgb: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, 4, 4, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    # BUG FIX E741: renamed ambiguous single-letter 'l' to 'l_channel'
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)


def preprocess_image(image_rgb: np.ndarray) -> PreprocessedImage:
    enhanced = enhance_image(image_rgb)
    return letterbox(enhanced, IMAGE_SIZE)


def video_frame_indices(
    total_frames: int, fps: float, interval_seconds: float
) -> Generator[int, None, None]:
    if total_frames <= 0 or fps <= 0:
        return
    step = max(int(round(fps * interval_seconds)), 1)
    for idx in range(0, total_frames, step):
        yield idx


def load_video_frames(
    path: Path, interval_seconds: float
) -> Generator[Tuple[int, np.ndarray, float], None, None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0
    for frame_idx in video_frame_indices(total_frames, fps, interval_seconds):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = capture.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp = frame_idx / fps
        yield frame_idx, frame_rgb, timestamp
    capture.release()


def save_rgb_image(image_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
