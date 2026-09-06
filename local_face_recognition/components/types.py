"""기술 컴포넌트 사이에서만 교환하는 검출·추적·품질 결과다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple
import numpy as np

BBox = Tuple[int, int, int, int]

@dataclass(frozen=True)
class PersonDetection:
    bbox: BBox
    confidence: float

@dataclass(frozen=True)
class TrackedPerson:
    internal_id: int
    bbox: BBox
    confirmed: bool

@dataclass(frozen=True)
class TrackEvent:
    kind: str
    internal_id: int
    occurred_at: datetime

@dataclass(frozen=True)
class FaceCandidate:
    track_id: int
    bbox: BBox
    detection_score: float
    landmarks: np.ndarray
    yaw_proxy: float
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    captured_at: datetime

@dataclass(frozen=True)
class FaceQuality:
    accepted: bool
    score: float
    brightness: float
    sharpness: float
    face_size: int
    occlusion_probability: float
    reason: str

@dataclass(frozen=True)
class MemoryFaceSample:
    track_id: int
    embedding: np.ndarray
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    quality: FaceQuality
    captured_at: datetime
