"""영속 가능한 얼굴 표본 엔티티와 값 객체다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence
from uuid import UUID, uuid4

@dataclass(frozen=True)
class FacePose:
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float

@dataclass(frozen=True)
class FaceQualitySummary:
    score: float
    metrics: Mapping[str, float]

@dataclass(frozen=True)
class FaceSampleEmbedding:
    vector: Sequence[float]
    model_version: str
    created_at: datetime

@dataclass(frozen=True)
class FaceSample:
    id: UUID
    observation_session_id: UUID
    captured_at: datetime
    face_crop_storage_key: str
    quality: FaceQualitySummary
    pose: FacePose
    embedding: FaceSampleEmbedding
    created_at: datetime

    @classmethod
    def create(cls, observation_session_id: UUID, captured_at: datetime, face_crop_storage_key: str, quality: FaceQualitySummary, pose: FacePose, embedding: FaceSampleEmbedding, created_at: datetime) -> "FaceSample":
        if created_at < captured_at:
            raise ValueError("FaceSample creation precedes capture.")
        return cls(uuid4(), observation_session_id, captured_at, face_crop_storage_key, quality, pose, embedding, created_at)
