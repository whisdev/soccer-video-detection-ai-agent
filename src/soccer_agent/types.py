"""Data models for soccer video detection."""

from typing import List, Tuple, Optional
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float
    track_id: Optional[int] = None


class TVFrameResult(BaseModel):
    frame_id: int
    boxes: list[BoundingBox]
    keypoints: List[Tuple[float, float]]  # [(x, y), ...] float coordinates
