"""Soccer Video Detection AI Agent — player detection, team classification, pitch keypoints."""

from .agent import AiAgent
from .types import BoundingBox, TVFrameResult

__all__ = ["AiAgent", "BoundingBox", "TVFrameResult"]
