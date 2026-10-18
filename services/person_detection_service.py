"""사람 검출 실행 주기를 관리하고 다음 검출 전까지 최근 결과를 제공한다."""

from __future__ import annotations

import time
from collections.abc import Callable

from cv2.typing import MatLike

from components.person_detector import PersonDetection, YoloPersonDetector


class InvalidDetectionFrequencyError(ValueError):
    """사람 검출 주기 설정이 올바르지 않을 때 발생한다."""


class PersonDetectionService:
    """검출 주기 정책과 최근 결과 보관을 담당한다.

    target_fps가 None이면 들어오는 모든 프레임을 검출한다. 값이 있으면 해당
    시간 간격이 지난 프레임만 검출하고, 중간 프레임에는 최근 결과를 반환한다.
    """

    def __init__(
        self,
        detector: YoloPersonDetector,
        target_fps: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if target_fps is not None and target_fps <= 0:
            raise InvalidDetectionFrequencyError(
                "사람 검출 FPS는 0보다 커야 합니다."
            )
        self._detector = detector
        self._interval = None if target_fps is None else 1.0 / target_fps
        self._clock = clock
        self._next_detection_at = 0.0
        self._latest_detections: list[PersonDetection] = []

    def process(self, frame: MatLike) -> list[PersonDetection]:
        """현재 프레임의 검출 필요 여부를 판단해 최신 사람 목록을 반환한다."""
        now = self._clock()
        if self._interval is not None and now < self._next_detection_at:
            # 아직 검출 시점이 아니므로 YOLO를 호출하지 않고 최근 결과를 재사용한다.
            return list(self._latest_detections)

        # 실제 사람 검출 방법은 component에 위임한다.
        self._latest_detections = self._detector.detect(frame)
        if self._interval is not None:
            self._next_detection_at = now + self._interval
        return list(self._latest_detections)
