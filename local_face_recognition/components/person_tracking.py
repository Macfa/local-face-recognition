"""사람 검출과 기술 Track 확정·상실을 담당하는 컴포넌트다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from ultralytics import YOLO

from .types import BBox, PersonDetection, TrackEvent, TrackedPerson


def intersection_over_union(first: BBox, second: BBox) -> float:
    """두 사람 경계 상자의 IoU를 계산한다.

    Args:
        first: BBox. 첫 번째 (left, top, right, bottom) 상자.
        second: BBox. 두 번째 상자.
    Returns:
        float. 0.0~1.0 범위의 겹침 비율.
    """
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - overlap
    return overlap / union if union else 0.0


class PersonDetector:
    """로컬 YOLO 가중치로 ndarray 프레임에서 사람 검출 DTO 목록을 만든다."""

    def __init__(self, model_path: Path, confidence_threshold: float = 0.45) -> None:
        """사람 검출 모델과 최소 신뢰도를 준비한다.

        Args:
            model_path: Path. 로컬 YOLO 가중치 파일 경로.
            confidence_threshold: float. person 검출 허용 하한.
        Returns:
            None.
        """
        self._model = YOLO(str(model_path))
        self._confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """BGR 카메라 프레임에서 person 클래스만 검출한다.

        Args:
            frame: np.ndarray. OpenCV BGR 영상 프레임.
        Returns:
            List[PersonDetection]. 신뢰도 기준을 통과한 사람 상자 목록.
        """
        result = self._model(
            frame,
            classes=[0],
            conf=self._confidence_threshold,
            verbose=False,
        )[0]
        return [
            PersonDetection(
                tuple(int(value) for value in box.xyxy[0].tolist()),
                float(box.conf[0]),
            )
            for box in result.boxes
        ]


@dataclass
class _TrackRecord:
    """IoUPersonTracker가 프레임 간 연결에만 사용하는 비공개 상태.

    Attributes:
        bbox: BBox. 마지막 연결 사람 상자.
        last_seen_at: datetime. 마지막 검출 시각.
        consecutive_hits: int. 연속 연결 횟수.
        confirmed: bool. TRACK_CONFIRMED 발행 여부.
    """

    bbox: BBox
    last_seen_at: datetime
    consecutive_hits: int = 1
    confirmed: bool = False


class IoUPersonTracker:
    """사람 상자 IoU로 다중 대상을 연결하고 확정·상실 이벤트를 만드는 컴포넌트."""

    def __init__(
        self,
        confirmation_frames: int = 3,
        lost_after_seconds: float = 1.0,
        iou_threshold: float = 0.3,
    ) -> None:
        """추적 확인·상실·상자 연결 정책을 초기화한다.

        Args:
            confirmation_frames: int. TRACK_CONFIRMED 전 필요한 연속 연결 수.
            lost_after_seconds: float. 마지막 검출 후 TRACK_LOST까지 허용 시간.
            iou_threshold: float. 기존 Track 연결에 필요한 최소 IoU.
        Returns:
            None.
        """
        self._confirmation_frames = confirmation_frames
        self._lost_after_seconds = lost_after_seconds
        self._iou_threshold = iou_threshold
        self._next_id = 1
        self._records: Dict[int, _TrackRecord] = {}

    def update(
        self,
        detections: Sequence[PersonDetection],
        occurred_at: datetime,
    ) -> Tuple[List[TrackedPerson], List[TrackEvent]]:
        """한 프레임 검출을 반영해 현재 확정 대상과 생명주기 이벤트를 반환한다.

        Args:
            detections: Sequence[PersonDetection]. 현재 프레임의 사람 검출 결과.
            occurred_at: datetime. 프레임 관측 시각.
        Returns:
            Tuple[List[TrackedPerson], List[TrackEvent]]. 화면·얼굴 분석 대상과 확정/상실 이벤트.
        """
        remaining_detection_indexes = set(range(len(detections)))
        matched_track_ids: set[int] = set()
        events: List[TrackEvent] = []

        for track_id, record in list(self._records.items()):
            detection_index = self._find_best_match(record, detections, remaining_detection_indexes)
            if detection_index is None:
                continue
            record.bbox = detections[detection_index].bbox
            record.consecutive_hits += 1
            record.last_seen_at = occurred_at
            matched_track_ids.add(track_id)
            remaining_detection_indexes.remove(detection_index)

        for track_id, record in list(self._records.items()):
            if track_id in matched_track_ids:
                if not record.confirmed and record.consecutive_hits >= self._confirmation_frames:
                    record.confirmed = True
                    events.append(TrackEvent("TRACK_CONFIRMED", track_id, occurred_at))
                continue
            record.consecutive_hits = 0
            if (occurred_at - record.last_seen_at).total_seconds() < self._lost_after_seconds:
                continue
            if record.confirmed:
                events.append(TrackEvent("TRACK_LOST", track_id, occurred_at))
            del self._records[track_id]

        for detection_index in remaining_detection_indexes:
            self._records[self._next_id] = _TrackRecord(detections[detection_index].bbox, occurred_at)
            self._next_id += 1

        active_people = [
            TrackedPerson(track_id, record.bbox, record.confirmed)
            for track_id, record in self._records.items()
            if record.confirmed and record.last_seen_at == occurred_at
        ]
        return active_people, events

    def _find_best_match(
        self,
        record: _TrackRecord,
        detections: Sequence[PersonDetection],
        remaining_detection_indexes: set[int],
    ) -> int | None:
        """아직 배정되지 않은 검출 중 Track과 IoU가 가장 큰 인덱스를 찾는다.

        Args:
            record: _TrackRecord. 연결 기준이 되는 기존 Track 상태.
            detections: Sequence[PersonDetection]. 현재 프레임 검출 목록.
            remaining_detection_indexes: set[int]. 다른 Track에 아직 배정되지 않은 인덱스.
        Returns:
            int | None. 최소 IoU를 통과한 최적 검출 인덱스 또는 None.
        """
        best_index: int | None = None
        best_iou = self._iou_threshold
        for candidate_index in remaining_detection_indexes:
            overlap = intersection_over_union(record.bbox, detections[candidate_index].bbox)
            if overlap >= best_iou:
                best_index = candidate_index
                best_iou = overlap
        return best_index
