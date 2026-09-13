"""얼굴 후보, 품질, 가림, 임베딩 및 세션 내 중복 판정 컴포넌트다."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align

from .types import FaceCandidate, FaceQuality, MemoryFaceSample, TrackedPerson


class FaceAnalyzer:
    """확정 사람 영역에서 가장 큰 얼굴·랜드마크·자세를 추출하는 로컬 InsightFace 컴포넌트."""

    def __init__(self, model_root: Path) -> None:
        """CPU 얼굴 검출 모델을 연다.

        Args: model_root: Path. `models/buffalo_l`을 포함한 로컬 모델 루트.
        Returns: None.
        Raises: FileNotFoundError. 모델 디렉터리가 없으면 발생.
        """
        model_directory = model_root / "models" / "buffalo_l"
        if not model_directory.is_dir():
            raise FileNotFoundError(f"InsightFace model directory is missing: {model_directory}")
        self._analysis = FaceAnalysis(
            name="buffalo_l",
            root=str(model_root),
            allowed_modules=["detection", "landmark_3d_68"],
            providers=["CPUExecutionProvider"],
        )
        self._analysis.prepare(ctx_id=0, det_size=(640, 640))

    def find_largest_face(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        captured_at: datetime,
    ) -> FaceCandidate | None:
        """사람 BBox 안의 가장 큰 얼굴을 FaceCandidate로 반환한다.

        Args: frame: np.ndarray BGR 프레임; person: TrackedPerson 대상; captured_at: datetime 관측 시각.
        Returns: FaceCandidate | None. 얼굴·랜드마크·자세 또는 미검출 None.
        """
        frame_height, frame_width = frame.shape[:2]
        left, top, right, bottom = self._clip_bbox(person.bbox, frame_width, frame_height)
        person_crop = frame[top:bottom, left:right]
        faces = self._analysis.get(person_crop) if person_crop.size else []
        if not faces:
            return None

        face = max(
            faces,
            key=lambda value: float((value.bbox[2] - value.bbox[0]) * (value.bbox[3] - value.bbox[1])),
        )
        face_left, face_top, face_right, face_bottom = (int(value) for value in face.bbox)
        bbox = self._clip_bbox(
            (left + face_left, top + face_top, left + face_right, top + face_bottom),
            frame_width,
            frame_height,
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None

        local_landmarks = np.asarray(face.kps, dtype=np.float32)
        yaw_proxy = self._yaw_proxy(local_landmarks)
        pose = np.asarray(face.pose, dtype=np.float32)
        pitch, yaw, roll = (float(value) for value in pose)
        global_landmarks = local_landmarks + np.array([left, top], dtype=np.float32)
        return FaceCandidate(
            person.internal_id,
            bbox,
            float(face.det_score),
            global_landmarks,
            yaw_proxy,
            yaw,
            pitch,
            roll,
            captured_at,
        )

    @staticmethod
    def _clip_bbox(
        bbox: Tuple[int, int, int, int],
        width: int,
        height: int,
    ) -> Tuple[int, int, int, int]:
        """BBox를 프레임 폭·높이 안으로 자른다.

        Args: bbox: Tuple[int, int, int, int]; width: int; height: int.
        Returns: Tuple[int, int, int, int]. 유효 범위로 제한된 BBox.
        """
        left, top, right, bottom = bbox
        return max(0, left), max(0, top), min(width, right), min(height, bottom)

    @staticmethod
    def _yaw_proxy(points: np.ndarray) -> float:
        """5점 랜드마크에서 방향 다양성용 좌우 근사값을 계산한다.

        Args: points: np.ndarray. shape (5, 2) 얼굴 랜드마크.
        Returns: float. 눈 간 거리로 정규화한 코 중심 오프셋.
        """
        eye_distance = float(np.linalg.norm(points[1] - points[0]))
        if eye_distance == 0:
            return 0.0
        eye_midpoint_x = float((points[0][0] + points[1][0]) / 2)
        return float((points[2][0] - eye_midpoint_x) / eye_distance)

class FaceEmbeddingComponent:
    """품질 통과 얼굴 crop을 로컬 ArcFace 임베딩으로 변환하는 컴포넌트."""

    def __init__(self, model_root: Path) -> None:
        """로컬 recognition ONNX 모델을 연다.

        Args: model_root: Path. `w600k_r50.onnx`를 포함한 모델 루트.
        Returns: None.
        Raises: FileNotFoundError. 가중치가 없으면 발생.
        """
        model_path = model_root / "models" / "buffalo_l" / "w600k_r50.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"Face recognition model is missing: {model_path}")
        self._model = get_model(str(model_path), providers=["CPUExecutionProvider"])
        self._model.prepare(ctx_id=0)

    def embed(self, frame: np.ndarray, candidate: FaceCandidate) -> np.ndarray:
        """얼굴을 랜드마크 정렬하고 L2 정규화된 벡터를 반환한다.

        Args: frame: np.ndarray BGR 프레임; candidate: FaceCandidate 얼굴 영역과 랜드마크.
        Returns: np.ndarray. shape (N,)의 정규화 얼굴 임베딩.
        Raises: ValueError. crop이 비었거나 벡터 norm이 0이면 발생.
        """
        left, top, right, bottom = candidate.bbox
        crop = frame[top:bottom, left:right]
        if not crop.size:
            raise ValueError("Face crop is empty.")
        landmarks = candidate.landmarks.copy()
        landmarks[:, 0] -= left
        landmarks[:, 1] -= top
        vector = np.asarray(
            self._model.get_feat(face_align.norm_crop(crop, landmark=landmarks))[0],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Face embedding is zero.")
        return vector / norm


class FaceOcclusionEvaluator:
    """로컬 ONNX 분류기로 얼굴 crop의 가림 확률을 계산하는 컴포넌트."""

    def __init__(self, model_path: Path) -> None:
        """가림 분류 ONNX 모델을 연다.

        Args: model_path: Path. 로컬 가림 분류 모델 파일.
        Returns: None.
        Raises: FileNotFoundError. 모델 파일이 없으면 발생.
        """
        if not model_path.is_file():
            raise FileNotFoundError(f"Face occlusion model is missing: {model_path}")
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def occlusion_probability(self, crop: np.ndarray) -> float:
        """BGR 얼굴 crop의 가림 클래스 확률을 반환한다.

        Args: crop: np.ndarray. BGR 얼굴 crop.
        Returns: float. 0.0~1.0 가림 확률.
        """
        resized = cv2.resize(crop, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        values = np.exp(logits - np.max(logits))
        return float(values[1] / np.sum(values))


class FaceQualityEvaluator:
    """크기·선명도·밝기·가림·자세 기준으로 FaceCandidate 표본 적합도를 평가한다."""

    def __init__(
        self,
        occlusion_evaluator: FaceOcclusionEvaluator,
        minimum_face_size: int = 56,
        minimum_sharpness: float = 10.0,
        brightness_range: Tuple[float, float] = (45.0, 210.0),
        minimum_detection_score: float = 0.375,
        maximum_abs_yaw_degrees: float = 65.0,
        maximum_abs_pitch_degrees: float = 50.0,
        maximum_abs_roll_degrees: float = 40.0,
        maximum_occlusion_probability: float = 0.5,
        enforce_pose_limits: bool = True,
    ) -> None:
        """FaceSample 채택에 사용할 품질 정책 값을 설정한다.

        Args: occlusion_evaluator: FaceOcclusionEvaluator; 나머지 float/int 인자는 각 품질 하한·상한.
        Returns: None.
        """
        self._occlusion_evaluator = occlusion_evaluator
        self._minimum_face_size = minimum_face_size
        self._minimum_sharpness = minimum_sharpness
        self._brightness_range = brightness_range
        self._minimum_detection_score = minimum_detection_score
        self._maximum_abs_yaw_degrees = maximum_abs_yaw_degrees
        self._maximum_abs_pitch_degrees = maximum_abs_pitch_degrees
        self._maximum_abs_roll_degrees = maximum_abs_roll_degrees
        self._maximum_occlusion_probability = maximum_occlusion_probability
        self._enforce_pose_limits = enforce_pose_limits

    def evaluate(self, frame: np.ndarray, candidate: FaceCandidate) -> FaceQuality:
        """한 얼굴 후보의 표본 저장 적합도를 평가한다.

        Args: frame: np.ndarray BGR 프레임; candidate: FaceCandidate.
        Returns: FaceQuality. 통과 여부·점수·수치·거부 사유.
        """
        left, top, right, bottom = candidate.bbox
        crop = frame[top:bottom, left:right]
        if not crop.size:
            return FaceQuality(False, 0.0, 0.0, 0.0, 0, 1.0, "empty_crop")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        face_size = min(crop.shape[:2])
        occlusion_probability = self._occlusion_evaluator.occlusion_probability(crop)
        checks = [
            (candidate.detection_score >= self._minimum_detection_score, "low_detection"),
            (face_size >= self._minimum_face_size, "small_face"),
            (sharpness >= self._minimum_sharpness, "blurred"),
            (self._brightness_range[0] <= brightness <= self._brightness_range[1], "bad_brightness"),
            (occlusion_probability <= self._maximum_occlusion_probability, "face_occluded"),
        ]
        if self._enforce_pose_limits:
            checks.extend(
                [
                    (abs(candidate.yaw_degrees) <= self._maximum_abs_yaw_degrees, "excessive_yaw"),
                    (abs(candidate.pitch_degrees) <= self._maximum_abs_pitch_degrees, "excessive_pitch"),
                    (abs(candidate.roll_degrees) <= self._maximum_abs_roll_degrees, "excessive_roll"),
                ]
            )
        reasons = [reason for passed, reason in checks if not passed]
        score = min(1.0, candidate.detection_score) * min(1.0, face_size / 160.0) * (1.0 - occlusion_probability)
        return FaceQuality(
            not reasons,
            score,
            brightness,
            sharpness,
            face_size,
            occlusion_probability,
            "accepted" if not reasons else ",".join(reasons),
        )


class InMemoryFaceSampleStore:
    """한 런타임 Track 안에서 임베딩·자세가 유사한 표본의 반복 저장을 막는 메모리 저장소."""

    def __init__(
        self,
        duplicate_similarity: float = 0.98,
        maximum_yaw_difference_degrees: float = 15.0,
        maximum_pitch_difference_degrees: float = 15.0,
        maximum_roll_difference_degrees: float = 15.0,
    ) -> None:
        """중복 임베딩·유사 자세 차단 기준을 설정한다.

        Args: duplicate_similarity: float; maximum_*_difference_degrees: float 자세 차이 허용값.
        Returns: None.
        """
        self._duplicate_similarity = duplicate_similarity
        self._maximum_yaw_difference_degrees = maximum_yaw_difference_degrees
        self._maximum_pitch_difference_degrees = maximum_pitch_difference_degrees
        self._maximum_roll_difference_degrees = maximum_roll_difference_degrees
        self._samples: Dict[int, List[MemoryFaceSample]] = {}

    def add_if_non_duplicate(
        self,
        candidate: FaceCandidate,
        embedding: np.ndarray,
        quality: FaceQuality,
    ) -> Tuple[bool, float | None, str | None]:
        """새 표본만 Track 메모리에 추가하고 중복 결과를 반환한다.

        Args: candidate: FaceCandidate; embedding: np.ndarray; quality: FaceQuality.
        Returns: Tuple[bool, float | None, str | None]. 저장 여부, 최대 유사도, 거부 사유.
        """
        samples = self._samples.setdefault(candidate.track_id, [])
        vector = embedding / np.linalg.norm(embedding)
        similarities = [float(np.dot(vector, sample.embedding)) for sample in samples]
        maximum_similarity = max(similarities) if similarities else None
        if maximum_similarity is not None and maximum_similarity >= self._duplicate_similarity:
            return False, maximum_similarity, "near_identical_embedding"
        if any(self._has_similar_pose(candidate, sample) for sample in samples):
            return False, maximum_similarity, "similar_pose"
        samples.append(
            MemoryFaceSample(
                candidate.track_id,
                vector,
                candidate.yaw_degrees,
                candidate.pitch_degrees,
                candidate.roll_degrees,
                quality,
                candidate.captured_at,
            )
        )
        return True, maximum_similarity, None

    def count(self, track_id: int) -> int:
        """현재 Track에서 채택한 메모리 표본 수를 반환한다.

        Args: track_id: int. 런타임 Track ID.
        Returns: int. 채택 표본 수.
        """
        return len(self._samples.get(track_id, []))

    def clear(self, track_id: int) -> None:
        """종료된 Track의 중복 판정 메모리를 폐기한다.

        Args: track_id: int. 런타임 Track ID.
        Returns: None.
        """
        self._samples.pop(track_id, None)

    def _has_similar_pose(self, candidate: FaceCandidate, sample: MemoryFaceSample) -> bool:
        """새 후보와 기존 표본의 세 자세 차이가 모두 허용 범위인지 반환한다.

        Args: candidate: FaceCandidate; sample: MemoryFaceSample.
        Returns: bool. yaw·pitch·roll 모두 유사하면 True.
        """
        return (
            abs(candidate.yaw_degrees - sample.yaw_degrees) <= self._maximum_yaw_difference_degrees
            and abs(candidate.pitch_degrees - sample.pitch_degrees) <= self._maximum_pitch_difference_degrees
            and abs(candidate.roll_degrees - sample.roll_degrees) <= self._maximum_roll_difference_degrees
        )
