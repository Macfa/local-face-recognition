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
    """확정된 사람 영역에서 가장 큰 얼굴과 랜드마크를 찾는다."""

    def __init__(self, model_root: Path) -> None:
        """로컬 InsightFace 얼굴 검출 모델을 CPU 실행으로 준비한다."""
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
        """사람 영역의 가장 큰 얼굴을 품질 평가 전 후보로 반환한다."""
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
        """경계 상자가 프레임 밖으로 나가지 않도록 잘라낸다."""
        left, top, right, bottom = bbox
        return max(0, left), max(0, top), min(width, right), min(height, bottom)

    @staticmethod
    def _yaw_proxy(points: np.ndarray) -> float:
        """등록 표본 방향 다양성에 쓰는 랜드마크 기반 좌우 방향 근사값을 계산한다."""
        eye_distance = float(np.linalg.norm(points[1] - points[0]))
        if eye_distance == 0:
            return 0.0
        eye_midpoint_x = float((points[0][0] + points[1][0]) / 2)
        return float((points[2][0] - eye_midpoint_x) / eye_distance)

class FaceEmbeddingComponent:
    """품질을 통과한 얼굴 후보를 로컬 ArcFace 임베딩으로 변환한다."""

    def __init__(self, model_root: Path) -> None:
        """자동 다운로드 없이 배치된 recognition ONNX 가중치를 연다."""
        model_path = model_root / "models" / "buffalo_l" / "w600k_r50.onnx"
        if not model_path.is_file():
            raise FileNotFoundError(f"Face recognition model is missing: {model_path}")
        self._model = get_model(str(model_path), providers=["CPUExecutionProvider"])
        self._model.prepare(ctx_id=0)

    def embed(self, frame: np.ndarray, candidate: FaceCandidate) -> np.ndarray:
        """후보의 얼굴 영역을 정렬해 L2 정규화된 임베딩으로 반환한다."""
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
    """로컬 ONNX 분류기로 얼굴 가림 확률을 계산한다."""

    def __init__(self, model_path: Path) -> None:
        """얼굴 가림 모델과 입력 이름을 준비한다."""
        if not model_path.is_file():
            raise FileNotFoundError(f"Face occlusion model is missing: {model_path}")
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def occlusion_probability(self, crop: np.ndarray) -> float:
        """얼굴 crop이 가려졌을 확률을 반환한다."""
        resized = cv2.resize(crop, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        values = np.exp(logits - np.max(logits))
        return float(values[1] / np.sum(values))


class FaceQualityEvaluator:
    """품질·자세·가림 기준을 적용해 FaceCandidate의 표본 적합도를 평가한다."""

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
        """표본 저장 전에 적용할 초기 품질 정책 값을 설정한다."""
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
        """후보의 촬영 상태가 영속 FaceSample을 만들기에 충분한지 판단한다."""
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
    """한 Track 안에서 거의 같은 표본이 반복 저장되는 것을 막는다."""

    def __init__(
        self,
        duplicate_similarity: float = 0.98,
        maximum_yaw_difference_degrees: float = 15.0,
        maximum_pitch_difference_degrees: float = 15.0,
        maximum_roll_difference_degrees: float = 15.0,
    ) -> None:
        """임베딩 기록과 같은 자세 표본의 차단 기준을 설정한다."""
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
        """표본이 새로울 때만 저장하고, 거부 시 근거 유사도와 사유를 반환한다."""
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
        """현재 Track에서 영속화 대상으로 채택한 표본 수를 반환한다."""
        return len(self._samples.get(track_id, []))

    def clear(self, track_id: int) -> None:
        """종료된 Track의 런타임 중복 판정 상태를 폐기한다."""
        self._samples.pop(track_id, None)

    def _has_similar_pose(self, candidate: FaceCandidate, sample: MemoryFaceSample) -> bool:
        """새 후보와 기존 표본의 yaw·pitch·roll 차이가 모두 허용 범위 안인지 확인한다."""
        return (
            abs(candidate.yaw_degrees - sample.yaw_degrees) <= self._maximum_yaw_difference_degrees
            and abs(candidate.pitch_degrees - sample.pitch_degrees) <= self._maximum_pitch_difference_degrees
            and abs(candidate.roll_degrees - sample.roll_degrees) <= self._maximum_roll_difference_degrees
        )
