from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, UnidentifiedImageError


CLASS_NAMES = [
    "corrosion",
    "crack",
    "dent",
    "hole",
    "leak",
    "paint_damage",
    "rust",
    "scratch",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
YOLO_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SourceImage:
    split: str
    class_name: str
    class_id: int
    source_path: Path
    target_name: str


@dataclass
class ConversionStats:
    scanned: int = 0
    copied: int = 0
    labeled: int = 0
    refined: int = 0
    full_image_labels: int = 0
    skipped: int = 0
    skipped_files: List[Dict[str, str]] = field(default_factory=list)
    split_counts: Dict[str, int] = field(default_factory=lambda: {"train": 0, "val": 0})


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in YOLO_IMAGE_EXTENSIONS


def verify_image(path: Path) -> Tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid image: {path}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions for {path}: {width}x{height}")
    return width, height


def build_target_name(class_name: str, split: str, source_path: Path) -> str:
    suffix = source_path.suffix.lower()
    stem = source_path.stem.replace(" ", "_")
    return f"{class_name}__{split}__{stem}{suffix}"


def collect_source_images(source_root: Path) -> List[SourceImage]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    records: List[SourceImage] = []
    for split_dir in ("train", "val"):
        split_root = source_root / split_dir
        if not split_root.exists():
            raise FileNotFoundError(
                f"Expected split directory missing: {split_root}. "
                "The source dataset should look like dataset/classification/images/{train,val}/<class>/..."
            )

        for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            class_name = class_dir.name
            if class_name not in CLASS_TO_ID:
                continue

            for image_path in sorted(class_dir.iterdir()):
                if not is_supported_image(image_path):
                    continue
                records.append(
                    SourceImage(
                        split=split_dir,
                        class_name=class_name,
                        class_id=CLASS_TO_ID[class_name],
                        source_path=image_path,
                        target_name=build_target_name(class_name, split_dir, image_path),
                    )
                )

    if not records:
        raise ValueError(f"No supported images found under {source_root}")
    return records


def normalize_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    box_width = x2 - x1
    box_height = y2 - y1
    x_center = x1 + box_width / 2.0
    y_center = y1 + box_height / 2.0
    return (
        round(x_center / width, 6),
        round(y_center / height, 6),
        round(box_width / width, 6),
        round(box_height / height, 6),
    )


def format_yolo_label(class_id: int, box: Tuple[float, float, float, float]) -> str:
    x_center, y_center, box_width, box_height = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def write_data_yaml(path: Path, dataset_root: Path) -> None:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to write data.yaml. Install project dependencies first.") from exc

    data = {
        "path": str(dataset_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    ensure_directory(path.parent)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def load_yolo_refiner(model_path: str, device: Optional[str]) -> Any:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is required for optional refinement and training. Install dependencies first."
        ) from exc

    model = YOLO(model_path)
    if device is not None:
        # Ultralytics handles the device during predict/train calls; we keep the value for predict-time use.
        return model, device
    return model, None


def refine_labels_with_yolo(
    model: Any,
    image_path: Path,
    class_id: int,
    width: int,
    height: int,
    confidence_threshold: float,
    device: Optional[str] = None,
) -> List[str]:
    results = model.predict(source=str(image_path), conf=confidence_threshold, verbose=False, device=device)
    if not results:
        return []

    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    label_lines: List[str] = []
    xyxy = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else [1.0] * len(xyxy)

    for coords, confidence in zip(xyxy, confidences):
        if confidence < confidence_threshold:
            continue
        normalized = normalize_box(coords[0], coords[1], coords[2], coords[3], width, height)
        if normalized is None:
            continue
        # Keep the class from the folder name; the pretrained model is only used to improve box geometry.
        label_lines.append(format_yolo_label(class_id, normalized))
    return label_lines


def copy_and_label_dataset(
    source_root: Path,
    output_root: Path,
    refine: bool = False,
    refine_model: str = "yolov8m.pt",
    refine_confidence: float = 0.4,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    records = collect_source_images(source_root)

    images_root = output_root / "images"
    labels_root = output_root / "labels"
    for split in ("train", "val"):
        clear_directory(images_root / split)
        clear_directory(labels_root / split)

    data_yaml_path = output_root / "data.yaml"
    write_data_yaml(data_yaml_path, output_root)

    refiner: Optional[Any] = None
    refiner_device: Optional[str] = None
    if refine:
        refiner, refiner_device = load_yolo_refiner(refine_model, device)

    stats = ConversionStats()
    for record in records:
        stats.scanned += 1
        target_image_path = images_root / record.split / record.target_name
        target_label_path = labels_root / record.split / f"{Path(record.target_name).stem}.txt"

        try:
            width, height = verify_image(record.source_path)
        except ValueError as exc:
            stats.skipped += 1
            stats.skipped_files.append({"path": str(record.source_path), "reason": str(exc)})
            continue

        ensure_directory(target_image_path.parent)
        ensure_directory(target_label_path.parent)
        shutil.copy2(record.source_path, target_image_path)
        stats.copied += 1

        label_lines: List[str] = []
        if refiner is not None:
            label_lines = refine_labels_with_yolo(
                model=refiner,
                image_path=record.source_path,
                class_id=record.class_id,
                width=width,
                height=height,
                confidence_threshold=refine_confidence,
                device=refiner_device,
            )
            if label_lines:
                stats.refined += 1

        if not label_lines:
            label_lines = [f"{record.class_id} 0.500000 0.500000 1.000000 1.000000"]
            stats.full_image_labels += 1

        target_label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        stats.labeled += 1
        stats.split_counts[record.split] = stats.split_counts.get(record.split, 0) + 1

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "data_yaml": str(data_yaml_path),
        "class_names": CLASS_NAMES,
        "class_to_id": CLASS_TO_ID,
        "refine_enabled": refine,
        "refine_model": refine_model if refine else None,
        "refine_confidence": refine_confidence if refine else None,
        "stats": {
            "scanned": stats.scanned,
            "copied": stats.copied,
            "labeled": stats.labeled,
            "refined": stats.refined,
            "full_image_labels": stats.full_image_labels,
            "skipped": stats.skipped,
            "split_counts": stats.split_counts,
        },
        "skipped_files": stats.skipped_files,
    }
    (output_root / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_train_config(
    data_yaml: Path,
    model: str,
    epochs: int,
    imgsz: int,
    batch: int,
    optimizer: str,
    lr0: float,
    project: str,
    name: str,
    device: Optional[str],
    patience: Optional[int],
    pretrained: Optional[bool],
) -> Dict[str, Any]:
    train_cfg: Dict[str, Any] = {
        "model": model,
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": optimizer,
        "lr0": lr0,
        "save": True,
        "project": project,
        "name": name,
    }
    if device is not None:
        train_cfg["device"] = device
    if patience is not None:
        train_cfg["patience"] = patience
    if pretrained is not None:
        train_cfg["pretrained"] = pretrained
    return train_cfg


def run_training(train_kwargs: Dict[str, Any]) -> Any:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required for training. Install dependencies first.") from exc

    model = YOLO(str(train_kwargs.pop("model")))
    return model.train(**train_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a classification-folder defect dataset into YOLO detection format and train a detector."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("dataset/classification/images"),
        help="Source dataset root containing train/val class folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset"),
        help="YOLO dataset output root.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Use a pretrained YOLO model to replace full-image labels with detected boxes when available.",
    )
    parser.add_argument(
        "--refine-model",
        type=str,
        default="yolov8m.pt",
        help="Pretrained YOLO model used for optional box refinement.",
    )
    parser.add_argument(
        "--refine-confidence",
        type=float,
        default=0.4,
        help="Confidence threshold for refinement detections.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device for refinement and training, e.g. cpu or 0.")
    parser.add_argument("--skip-train", action="store_true", help="Prepare the dataset but do not start training.")
    parser.add_argument("--train-model", type=str, default="yolov8m.pt", help="Base model for detection training.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Training batch size.")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Training optimizer.")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Ultralytics project directory.")
    parser.add_argument("--name", type=str, default="defect_train", help="Training run name.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=None,
        help="Enable pretrained weights for training.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_false",
        dest="pretrained",
        help="Disable pretrained weights for training.",
    )
    args = parser.parse_args()

    manifest = copy_and_label_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        refine=args.refine,
        refine_model=args.refine_model,
        refine_confidence=args.refine_confidence,
        device=args.device,
    )
    print(json.dumps(manifest, indent=2))

    if args.skip_train:
        return

    train_kwargs = build_train_config(
        data_yaml=args.output_root / "data.yaml",
        model=args.train_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=args.patience,
        pretrained=args.pretrained,
    )
    result = run_training(train_kwargs)
    print(result)


if __name__ == "__main__":
    main()
