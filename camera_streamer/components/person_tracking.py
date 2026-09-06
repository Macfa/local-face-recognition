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
    """두 경계 상자의 IoU를 계산해 프레임 간 연결 근거로 사용한다."""
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - overlap
    return overlap / union if union else 0.0


class PersonDetector:
    """로컬 YOLO 가중치로 현재 프레임의 person 검출만 수행한다."""

    def __init__(self, model_path: Path, confidence_threshold: float = 0.45) -> None:
        """검출 모델과 person 검출 신뢰도 하한을 준비한다."""
        self._model = YOLO(str(model_path))
        self._confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """프레임에서 person 클래스 검출 결과를 기술 DTO로 반환한다."""
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
    """추적기 내부에서만 쓰는 프레임 간 연결 상태다."""

    bbox: BBox
    consecutive_hits: int = 1
    missed_frames: int = 0
    confirmed: bool = False


class IoUPersonTracker:
    """IoU로 검출을 연결하고 안정된 대상에만 기술 Track 이벤트를 낸다."""

    def __init__(
        self,
        confirmation_frames: int = 3,
        max_missed_frames: int = 30,
        iou_threshold: float = 0.3,
    ) -> None:
        """확정·상실·연결 기준을 초기화한다."""
        self._confirmation_frames = confirmation_frames
        self._max_missed_frames = max_missed_frames
        self._iou_threshold = iou_threshold
        self._next_id = 1
        self._records: Dict[int, _TrackRecord] = {}

    def update(
        self,
        detections: Sequence[PersonDetection],
        occurred_at: datetime,
    ) -> Tuple[List[TrackedPerson], List[TrackEvent]]:
        """한 프레임의 검출을 반영하고 확정 대상 및 생명주기 이벤트를 반환한다."""
        remaining_detection_indexes = set(range(len(detections)))
        matched_track_ids: set[int] = set()
        events: List[TrackEvent] = []

        for track_id, record in list(self._records.items()):
            detection_index = self._find_best_match(record, detections, remaining_detection_indexes)
            if detection_index is None:
                continue
            record.bbox = detections[detection_index].bbox
            record.consecutive_hits += 1
            record.missed_frames = 0
            matched_track_ids.add(track_id)
            remaining_detection_indexes.remove(detection_index)

        for track_id, record in list(self._records.items()):
            if track_id in matched_track_ids:
                if not record.confirmed and record.consecutive_hits >= self._confirmation_frames:
                    record.confirmed = True
                    events.append(TrackEvent("TRACK_CONFIRMED", track_id, occurred_at))
                continue
            record.missed_frames += 1
            record.consecutive_hits = 0
            if record.missed_frames <= self._max_missed_frames:
                continue
            if record.confirmed:
                events.append(TrackEvent("TRACK_LOST", track_id, occurred_at))
            del self._records[track_id]

        for detection_index in remaining_detection_indexes:
            self._records[self._next_id] = _TrackRecord(detections[detection_index].bbox)
            self._next_id += 1

        active_people = [
            TrackedPerson(track_id, record.bbox, record.confirmed)
            for track_id, record in self._records.items()
            if record.confirmed and record.missed_frames == 0
        ]
        return active_people, events

    def _find_best_match(
        self,
        record: _TrackRecord,
        detections: Sequence[PersonDetection],
        remaining_detection_indexes: set[int],
    ) -> int | None:
        """아직 연결되지 않은 검출 중 현재 Track과 가장 많이 겹치는 것을 찾는다."""
        best_index: int | None = None
        best_iou = self._iou_threshold
        for candidate_index in remaining_detection_indexes:
            overlap = intersection_over_union(record.bbox, detections[candidate_index].bbox)
            if overlap >= best_iou:
                best_index = candidate_index
                best_iou = overlap
        return best_index
