"""사람 검출과 기술 Track 확정·상실을 담당하는 컴포넌트다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
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
    missing: bool = False
    previous_bbox: BBox | None = None
    association_uncertain: bool = False


class IoUPersonTracker:
    """위치·이동 연속성으로 사람 상자를 전역 배정하고 기술 Track 이벤트를 만든다.

    내부 Track ID는 영상 상자를 잇는 기술 상태일 뿐 사람 신원을 뜻하지 않는다. 연결 후보가
    가까워 확신할 수 없으면 TRACK_ASSOCIATION_UNCERTAIN을 발행해 Application이 이름을
    숨기고 새 얼굴 검증을 시작할 수 있게 한다.
    """

    def __init__(
        self,
        confirmation_frames: int = 3,
        missing_after_seconds: float = 1.0,
        retain_after_missing_seconds: float = 30.0,
        iou_threshold: float = 0.3,
        association_margin: float = 0.08,
    ) -> None:
        """추적 확인·상실·상자 연결 정책을 초기화한다.

        Args:
            confirmation_frames: int. TRACK_CONFIRMED 전 필요한 연속 연결 수.
            missing_after_seconds: float. 마지막 검출 후 화면에서 잠시 사라졌다고 판단할 시간.
            retain_after_missing_seconds: float. MISSING 후 동일 기술 Track을 보관할 시간.
            iou_threshold: float. 기존 Track 연결에 필요한 최소 IoU.
            association_margin: float. 최상위·차선 연결 점수 차이의 불확실 경계.
        Returns:
            None.
        """
        self._confirmation_frames = confirmation_frames
        self._missing_after_seconds = missing_after_seconds
        self._retain_after_missing_seconds = retain_after_missing_seconds
        self._iou_threshold = iou_threshold
        self._association_margin = association_margin
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

        assignments = self._globally_assign(detections)
        uncertain_track_ids = {
            track_id
            for track_id, detection_index in assignments.items()
            if self._is_ambiguous_assignment(self._records[track_id], detections, detection_index)
        }
        for first_track_id, first_detection_index in assignments.items():
            for second_track_id, second_detection_index in assignments.items():
                if first_track_id >= second_track_id:
                    continue
                first_record, second_record = self._records[first_track_id], self._records[second_track_id]
                if first_record.confirmed and second_record.confirmed and intersection_over_union(
                    detections[first_detection_index].bbox,
                    detections[second_detection_index].bbox,
                ) >= 0.5:
                    uncertain_track_ids.update({first_track_id, second_track_id})
        for track_id, detection_index in assignments.items():
            record = self._records[track_id]
            uncertain = track_id in uncertain_track_ids
            was_missing = record.missing
            record.previous_bbox = record.bbox
            record.bbox = detections[detection_index].bbox
            record.consecutive_hits += 1
            record.last_seen_at = occurred_at
            if was_missing:
                record.missing = False
                if record.confirmed:
                    events.append(TrackEvent("TRACK_REAPPEARED", track_id, occurred_at))
            if record.confirmed and uncertain and not record.association_uncertain and not was_missing:
                events.append(TrackEvent("TRACK_ASSOCIATION_UNCERTAIN", track_id, occurred_at))
            record.association_uncertain = uncertain
            matched_track_ids.add(track_id)
            remaining_detection_indexes.remove(detection_index)

        for track_id, record in list(self._records.items()):
            if track_id in matched_track_ids:
                if not record.confirmed and record.consecutive_hits >= self._confirmation_frames:
                    record.confirmed = True
                    events.append(TrackEvent("TRACK_CONFIRMED", track_id, occurred_at))
                continue
            elapsed_seconds = (occurred_at - record.last_seen_at).total_seconds()
            if not record.missing and elapsed_seconds >= self._missing_after_seconds:
                record.missing = True
                record.consecutive_hits = 0
                if record.confirmed:
                    events.append(TrackEvent("TRACK_MISSING", track_id, occurred_at))
            if elapsed_seconds < self._retain_after_missing_seconds:
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

    def _globally_assign(self, detections: Sequence[PersonDetection]) -> Dict[int, int]:
        """기존 Track과 검출 상자의 총 연결 점수가 가장 큰 일대일 배정을 계산한다.

        Returns:
            Dict[int, int]. 내부 Track ID와 연결된 detection 인덱스의 매핑.
        """
        track_ids = tuple(self._records.keys())
        if not track_ids or not detections:
            return {}
        score_rows = tuple(
            tuple(self._association_score(self._records[track_id], detection.bbox) for detection in detections)
            for track_id in track_ids
        )

        @lru_cache(maxsize=None)
        def choose(track_index: int, used_mask: int) -> Tuple[float, Tuple[int | None, ...]]:
            if track_index == len(track_ids):
                return 0.0, ()
            best_score, tail = choose(track_index + 1, used_mask)
            best_assignment: Tuple[int | None, ...] = (None,) + tail
            for detection_index, score in enumerate(score_rows[track_index]):
                if score is None or used_mask & (1 << detection_index):
                    continue
                next_score, next_tail = choose(track_index + 1, used_mask | (1 << detection_index))
                total = score + next_score
                if total > best_score:
                    best_score = total
                    best_assignment = (detection_index,) + next_tail
            return best_score, best_assignment

        _, selected = choose(0, 0)
        return {
            track_id: detection_index
            for track_id, detection_index in zip(track_ids, selected)
            if detection_index is not None
        }

    def _association_score(self, record: _TrackRecord, detection_bbox: BBox) -> float | None:
        """IoU와 이전 이동 방향을 함께 반영한 Track-검출 연결 점수를 계산한다."""
        overlap = intersection_over_union(record.bbox, detection_bbox)
        if overlap < self._iou_threshold:
            return None
        predicted = _predicted_bbox(record)
        motion_distance = _normalized_center_distance(predicted, detection_bbox)
        motion_score = max(0.0, 1.0 - motion_distance)
        return overlap * 0.75 + motion_score * 0.25

    def _is_ambiguous_assignment(
        self,
        record: _TrackRecord,
        detections: Sequence[PersonDetection],
        selected_index: int,
    ) -> bool:
        """선택된 상자가 차선 후보와 구분되지 않는지 확인한다."""
        scores = sorted(
            (score for detection in detections if (score := self._association_score(record, detection.bbox)) is not None),
            reverse=True,
        )
        selected_score = self._association_score(record, detections[selected_index].bbox)
        if selected_score is None or len(scores) < 2:
            return False
        return scores[0] - scores[1] < self._association_margin


def _predicted_bbox(record: _TrackRecord) -> BBox:
    """직전 상자 이동량으로 현재 프레임의 중심 위치를 짧게 예측한다."""
    if record.previous_bbox is None:
        return record.bbox
    last_center = _bbox_center(record.bbox)
    previous_center = _bbox_center(record.previous_bbox)
    delta_x, delta_y = last_center[0] - previous_center[0], last_center[1] - previous_center[1]
    return tuple(
        int(value + offset)
        for value, offset in zip(record.bbox, (delta_x, delta_y, delta_x, delta_y))
    )  # type: ignore[return-value]


def _normalized_center_distance(first: BBox, second: BBox) -> float:
    """두 상자 중심 거리를 평균 상자 대각선으로 정규화한다."""
    first_center, second_center = _bbox_center(first), _bbox_center(second)
    distance = ((first_center[0] - second_center[0]) ** 2 + (first_center[1] - second_center[1]) ** 2) ** 0.5
    first_diagonal = ((first[2] - first[0]) ** 2 + (first[3] - first[1]) ** 2) ** 0.5
    second_diagonal = ((second[2] - second[0]) ** 2 + (second[3] - second[1]) ** 2) ** 0.5
    return distance / max(1.0, (first_diagonal + second_diagonal) / 2.0)


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    """사람 상자의 중심 좌표를 반환한다."""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
