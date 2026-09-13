"""기술 컴포넌트 사이에서만 교환하는 검출·추적·품질 결과다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple
import numpy as np

BBox = Tuple[int, int, int, int]

@dataclass(frozen=True)
class PersonDetection:
    """YOLO가 한 프레임에서 검출한 사람 DTO.

    Attributes:
        bbox: BBox. (left, top, right, bottom) 픽셀 경계 상자.
        confidence: float. 사람 검출 신뢰도.
    """
    bbox: BBox
    confidence: float

@dataclass(frozen=True)
class TrackedPerson:
    """추적기가 내부 ID와 연결한 현재 프레임의 사람 DTO.

    Attributes:
        internal_id: int. 런타임 전용 Track ID.
        bbox: BBox. 현재 프레임의 사람 영역.
        confirmed: bool. 확인 프레임 수를 만족했는지 여부.
    """
    internal_id: int
    bbox: BBox
    confirmed: bool

@dataclass(frozen=True)
class TrackEvent:
    """추적기의 확정·상실 생명주기 이벤트 DTO.

    Attributes:
        kind: str. TRACK_CONFIRMED 또는 TRACK_LOST.
        internal_id: int. 이벤트 대상 런타임 Track ID.
        occurred_at: datetime. 이벤트 발생 시각.
    """
    kind: str
    internal_id: int
    occurred_at: datetime

@dataclass(frozen=True)
class FaceCandidate:
    """확정 사람 영역에서 검출한 품질 평가 전 얼굴 후보 DTO.

    Attributes:
        track_id: int. 소유 런타임 Track ID.
        bbox: BBox. 얼굴 픽셀 영역.
        detection_score: float. 얼굴 검출 신뢰도.
        landmarks: np.ndarray. 5점 얼굴 랜드마크, shape (5, 2).
        yaw_proxy: float. 방향 다양성 비교용 좌우 근사값.
        yaw_degrees: float. 3D yaw 각도.
        pitch_degrees: float. 3D pitch 각도.
        roll_degrees: float. 3D roll 각도.
        captured_at: datetime. 후보 생성 시각.
    """
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
    """한 FaceCandidate의 품질·가림·자세 평가 결과 DTO.

    Attributes:
        accepted: bool. 영속 FaceSample 생성 가능 여부.
        score: float. 화면·로그용 종합 점수.
        brightness: float. grayscale 평균 밝기.
        sharpness: float. Laplacian 분산 선명도.
        face_size: int. 최소 얼굴 변 길이.
        occlusion_probability: float. 가림 확률.
        reason: str. accepted 또는 거부 사유.
    """
    accepted: bool
    score: float
    brightness: float
    sharpness: float
    face_size: int
    occlusion_probability: float
    reason: str

@dataclass(frozen=True)
class MemoryFaceSample:
    """한 런타임 Track 안에서 중복 표본을 막기 위한 메모리 전용 기록.

    Attributes:
        track_id: int. 소유 Track ID.
        embedding: np.ndarray. L2 정규화 얼굴 벡터.
        yaw_degrees: float. 표본 yaw.
        pitch_degrees: float. 표본 pitch.
        roll_degrees: float. 표본 roll.
        quality: FaceQuality. 채택 당시 품질.
        captured_at: datetime. 채택 시각.
    """
    track_id: int
    embedding: np.ndarray
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    quality: FaceQuality
    captured_at: datetime
