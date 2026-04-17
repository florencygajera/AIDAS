from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to read the training config. Install requirements first.") from exc

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def build_train_kwargs(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    train_cfg = {
        "model": args.model or config.get("model", "yolov8m.pt"),
        "data": args.data or config.get("data", "configs/data.yaml"),
        "epochs": args.epochs or config.get("epochs", 100),
        "imgsz": args.imgsz or config.get("imgsz", 640),
        "batch": args.batch or config.get("batch", 16),
        "optimizer": args.optimizer or config.get("optimizer", "AdamW"),
        "lr0": args.lr0 or config.get("lr0", 0.001),
        "cos_lr": True if args.cos_lr else bool(config.get("cos_lr", True)),
        "mosaic": config.get("mosaic", 1.0),
        "mixup": config.get("mixup", 0.15),
        "label_smoothing": config.get("label_smoothing", 0.0),
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.7),
        "hsv_v": config.get("hsv_v", 0.4),
        "degrees": config.get("degrees", 10.0),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.5),
        "shear": config.get("shear", 2.0),
        "fliplr": config.get("fliplr", 0.5),
        "save": True,
        "project": args.project,
        "name": args.name,
    }
    if args.patience is not None:
        train_cfg["patience"] = args.patience
    if args.device:
        train_cfg["device"] = args.device
    if args.pretrained is not None:
        train_cfg["pretrained"] = args.pretrained
    return train_cfg


def train_yolo(train_kwargs: Dict[str, Any]) -> Any:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ultralytics is required for training. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    model = YOLO(str(train_kwargs.pop("model")))
    return model.train(**train_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the industrial defect detector with YOLOv8.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"), help="Training config file.")
    parser.add_argument("--data", type=str, default=None, help="Path to YOLO data.yaml.")
    parser.add_argument("--model", type=str, default=None, help="Base model or checkpoint path.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=None, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size.")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer name.")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate.")
    parser.add_argument("--cos-lr", action="store_true", help="Enable cosine LR scheduler.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default=None, help="Device like cpu, 0, 0,1.")
    parser.add_argument("--pretrained", action="store_true", default=None, help="Use pretrained weights.")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Disable pretrained weights.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Ultralytics project directory.")
    parser.add_argument("--name", type=str, default="defect_train", help="Run name.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    train_kwargs = build_train_kwargs(config, args)
    result = train_yolo(train_kwargs)
    print(result)


if __name__ == "__main__":
    main()

