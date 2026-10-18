"""YOLO 추론 결과를 프로젝트 내부의 사람 검출 결과로 변환한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cv2.typing import MatLike


PERSON_CLASS_ID = 0
DEFAULT_CONFIDENCE = 0.5


@dataclass(frozen=True)
class PersonDetection:
    """한 프레임에서 검출된 사람의 위치와 신뢰도다."""

    left: int
    top: int
    right: int
    bottom: int
    confidence: float


class PersonDetectionError(RuntimeError):
    """YOLO 사람 검출 처리에 실패했을 때 발생한다."""


class YoloPersonDetector:
    """프레임을 YOLO에 전달하고 사람 클래스의 박스와 신뢰도만 반환한다."""

    def __init__(self, model: Any, confidence: float = DEFAULT_CONFIDENCE) -> None:
        self._model = model
        self._confidence = confidence

    def detect(self, frame: MatLike) -> list[PersonDetection]:
        """프레임에서 사람을 검출하고 화면 좌표 기준 결과로 변환한다."""
        try:
            # COCO의 person 클래스(0)만 요청해 불필요한 객체 결과를 제외한다.
            results = self._model.predict(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=self._confidence,
                verbose=False,
            )
            if not results or results[0].boxes is None:
                return []

            # 장치 텐서를 일반 Python 값으로 바꿔 나머지 계층이 YOLO에 의존하지 않게 한다.
            coordinates = results[0].boxes.xyxy.cpu().tolist()
            confidences = results[0].boxes.conf.cpu().tolist()
            return [
                PersonDetection(
                    left=round(left),
                    top=round(top),
                    right=round(right),
                    bottom=round(bottom),
                    confidence=float(confidence),
                )
                for (left, top, right, bottom), confidence in zip(
                    coordinates, confidences
                )
            ]
        except Exception as error:
            raise PersonDetectionError("YOLO 사람 검출에 실패했습니다.") from error
