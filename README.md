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

