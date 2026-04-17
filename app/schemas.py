from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DefectBox(BaseModel):
    type: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[int]
    severity: str


class AuditResponse(BaseModel):
    status: str
    risk_score: float
    defects: List[DefectBox]
    annotated_image: str
    annotated_video: Optional[str] = None
    source_type: str
    frame_count: Optional[int] = None
    skipped_frames: Optional[int] = None

