# Industrial Defect Detection System

Production-oriented FastAPI app for industrial visual inspection.

## What it does

- Accepts JPG, PNG, JPEG, BMP, MP4, AVI, and MOV uploads
- Preprocesses every input to RGB with 640x640 model input
- Filters low-quality frames using Laplacian blur and brightness checks
- Enhances imagery with CLAHE and noise reduction
- Runs a multi-stage defect pipeline:
  - Primary YOLOv8 detection
  - Patch-based micro-defect scanning
  - Optional CNN classification refinement
  - Optional segmentation support
- Aggregates video detections across frames
- Computes a risk score and audit decision
- Serves a simple browser upload UI

## Project structure

- `app/main.py` FastAPI entry point
- `app/services/preprocess.py` image/video loading and enhancement
- `app/services/detection.py` YOLO wrappers and patch scanning
- `app/services/classification.py` optional classifier refinement
- `app/services/scoring.py` risk scoring and decision rules
- `app/services/pipeline.py` end-to-end orchestration
- `templates/index.html` upload UI
- `static/styles.css` interface styling

## Running locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add weights:

- Primary detector: `models/yolov8m_defects.pt`
- Optional segmenter: set `DEFECT_YOLO_SEG_WEIGHTS`
- Optional classifier: set `DEFECT_CLASSIFIER_WEIGHTS`

3. Start the server:

```bash
uvicorn app.main:app --reload
```

4. Open `http://127.0.0.1:8000`

## Training

Run the bundled training entrypoint:

```powershell
python train.py --data dataset/data.yaml --model yolov8n.pt --epochs 30 --batch 16 --imgsz 512
```

Example with a larger model and GPU device:

```powershell
python train.py --data dataset/data.yaml --model yolov8m.pt --epochs 100 --batch 16 --imgsz 640 --device 0
```

The training script writes runs under `runs/detect/` by default. The best checkpoint is usually:

```text
runs/detect/defect_train/weights/best.pt
```

Copy that file to `models/yolov8m_defects.pt` or point `DEFECT_YOLO_WEIGHTS` to it for inference.

## Dataset Rearchitecture

If your source images are organized like `dataset/images/val/raw/<class>/...`, you can create a clean
80/20 train/val layout with renamed files using:

```powershell
python scripts/restructure_dataset.py --source-root dataset/images/val/raw --output-root dataset/classification --train-ratio 0.8
```

That produces:

- `dataset/classification/images/train/<class>/<class>1.jpg`
- `dataset/classification/images/val/<class>/<class>1.jpg`
- `dataset/classification/manifest.json`

For classification training, use:

```powershell
yolo classify train data=configs/classification_data.yaml model=yolov8m-cls.pt epochs=100 imgsz=224 batch=16
```

## Classification to YOLO Detection Pipeline

To convert the classification-style defect dataset into YOLO detection format, auto-label it, optionally
refine the boxes with a pretrained detector, and then train a detection model in one step, run:

```powershell
python scripts/classification_to_yolo_pipeline.py --refine
```

That script:

- Reads `dataset/classification/images/train/<class>/...` and `dataset/classification/images/val/<class>/...`
- Copies images into `dataset/images/train` and `dataset/images/val`
- Creates matching labels in `dataset/labels/train` and `dataset/labels/val`
- Writes `dataset/data.yaml`
- Logs skipped or corrupted images in `dataset/conversion_manifest.json`
- Starts YOLOv8 detection training with `yolov8m.pt`
- Keeps the folder name as the defect class and uses YOLO refinement only for box localization

If you want dataset preparation without training, add:

```powershell
python scripts/classification_to_yolo_pipeline.py --refine --skip-train
```

To run the same training command manually after conversion:

```powershell
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=30 imgsz=512 batch=16 optimizer=AdamW lr0=0.001
```

## Training configuration

The code is wired for production inference, but the model quality depends on trained weights. Use a dataset with:

- 5,000 to 10,000+ annotated images
- YOLO-format bounding boxes
- Balanced coverage across:
  - crack
  - hole
  - dent
  - rust
  - corrosion
  - paint_damage
  - scratch
  - leak
- Hard negatives:
  - dirt vs rust
  - shadows vs cracks
  - reflections vs leaks

Suggested training settings:

- YOLOv8m or YOLOv8l
- epochs: 100+
- batch: 8-16
- image size: 640
- optimizer: AdamW
- lr: 0.001
- augmentations enabled

## Notes

- Patch scanning is built in and always active.
- Classification refinement is optional and becomes active when classifier weights are provided.
- For safety-critical deployments, validate thresholds and retrain on your real factory data before production use.
- Use `scripts/build_balanced_manifest.py` to rebalance datasets and oversample rare classes.
- Use `scripts/evaluate_predictions.py` to track precision, recall, and false negatives for crack, hole, and leak.
- Use `scripts/mining_failures.py` to collect missed defects and hard negatives for the next retraining loop.
