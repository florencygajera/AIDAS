from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import OUTPUT_DIR, STATIC_DIR, TEMPLATE_DIR
from app.jinja import build_templates, validate_templates
from app.services.pipeline import DefectPipeline


app = FastAPI(
    title="Industrial Defect Detection",
    description="Multi-stage defect detection pipeline for industrial surfaces.",
    version="1.0.0",
)

templates = build_templates(TEMPLATE_DIR)
validate_templates(templates)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

pipeline = DefectPipeline()


def render_template(request: Request, *, name: str, context: dict | None = None):
    if not isinstance(name, str):
        raise TypeError(f"Template name must be str, got {type(name).__name__}")
    payload = {"request": request}
    if context:
        payload.update(context)
    return templates.TemplateResponse(request, name, payload)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_template(request, name="index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov"}:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    upload_dir = OUTPUT_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / (Path(file.filename).stem + suffix)
    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        response = pipeline.process(input_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse(content=response.model_dump())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="[IP_ADDRESS]", port=8000, reload=True)
