"""영속 가능한 얼굴 표본 엔티티와 값 객체다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence
from uuid import UUID, uuid4

@dataclass(frozen=True)
class FacePose:
    """한 얼굴 표본의 3축 자세 값 객체.

    Attributes:
        yaw_degrees: float. 좌우 회전 각도.
        pitch_degrees: float. 상하 회전 각도.
        roll_degrees: float. 기울기 각도.
    """
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float

@dataclass(frozen=True)
class FaceQualitySummary:
    """영속 FaceSample의 품질 점수와 수치 근거.

    Attributes:
        score: float. 종합 품질 점수.
        metrics: Mapping[str, float]. 선명도·밝기 등 평가 지표.
    """
    score: float
    metrics: Mapping[str, float]

@dataclass(frozen=True)
class FaceSampleEmbedding:
    """로컬 모델이 만든 정규화 얼굴 벡터와 생성 이력.

    Attributes:
        vector: Sequence[float]. 얼굴 임베딩 좌표.
        model_version: str. 벡터를 만든 모델 식별자.
        created_at: datetime. 임베딩 생성 시각.
    """
    vector: Sequence[float]
    model_version: str
    created_at: datetime

@dataclass(frozen=True)
class FaceSample:
    """신원 판단과 등록에 실제 사용되는 영속 얼굴 표본 엔티티.

    Attributes:
        id: UUID. 표본 식별자.
        observation_session_id: UUID. 표본이 속한 관찰 세션.
        captured_at: datetime. 카메라 캡처 시각.
        face_crop_storage_key: str. private crop 저장 위치.
        quality: FaceQualitySummary. 품질 평가 결과.
        pose: FacePose. 얼굴 자세.
        embedding: FaceSampleEmbedding. 신원 검색 벡터.
        created_at: datetime. 도메인 표본 생성 시각.
    """
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
        """유효한 시각 순서를 확인하고 새 FaceSample을 생성한다.

        Args:
            observation_session_id: UUID. 소속 관찰 세션 ID.
            captured_at: datetime. 얼굴이 캡처된 시각.
            face_crop_storage_key: str. crop 저장 위치.
            quality: FaceQualitySummary. 통과한 품질 결과.
            pose: FacePose. 캡처 당시 얼굴 자세.
            embedding: FaceSampleEmbedding. 검색용 얼굴 벡터.
            created_at: datetime. 엔티티 생성 시각.
        Returns:
            FaceSample. 새 UUID를 가진 불변 표본.
        Raises:
            ValueError: 생성 시각이 캡처 시각보다 이르면 발생.
        """
        if created_at < captured_at:
            raise ValueError("FaceSample creation precedes capture.")
        return cls(uuid4(), observation_session_id, captured_at, face_crop_storage_key, quality, pose, embedding, created_at)
