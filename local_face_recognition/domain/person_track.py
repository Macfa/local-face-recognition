"""PersonTrack 생명주기 엔티티다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

class PersonTrackStatus(str, Enum):
    """PersonTrack의 도메인 생명주기 상태.

    Values:
        TRACKING: 현재 카메라에서 확인 중.
        LOST: 기술 추적은 끊겼고 종료 대기 중.
        ENDED: 10분 보관 조건을 만족해 종료됨.
    """
    TRACKING = "TRACKING"
    LOST = "LOST"
    ENDED = "ENDED"

@dataclass
class PersonTrack:
    """한 번의 확정 사람 추적을 보관하는 도메인 엔티티.

    Attributes:
        id: UUID. 도메인 Track ID.
        started_at: datetime. 확정 시작 시각.
        status: PersonTrackStatus. 현재 생명주기 상태.
        lost_at: datetime | None. 마지막 상실 시각.
        ended_at: datetime | None. 종료 시각.
    """
    id: UUID
    started_at: datetime
    status: PersonTrackStatus
    lost_at: datetime | None = None
    ended_at: datetime | None = None

    @classmethod
    def start(cls, started_at: datetime) -> "PersonTrack":
        """TRACKING 상태의 새 PersonTrack을 생성한다.

        Args:
            started_at: datetime. TRACK_CONFIRMED 발생 시각.
        Returns:
            PersonTrack. 새 UUID와 TRACKING 상태를 가진 Track.
        """
        return cls(uuid4(), started_at, PersonTrackStatus.TRACKING)

    def mark_lost(self, lost_at: datetime) -> None:
        """TRACKING Track을 LOST로 전이한다.

        Args:
            lost_at: datetime. 마지막 검출 뒤 상실로 판단된 시각.
        Returns:
            None.
        Raises:
            ValueError: TRACKING이 아니거나 시작 전 시각이면 발생.
        """
        if self.status is not PersonTrackStatus.TRACKING or lost_at < self.started_at:
            raise ValueError("Invalid PersonTrack lost transition.")
        self.status, self.lost_at = PersonTrackStatus.LOST, lost_at

    def can_end(self, evaluated_at: datetime) -> bool:
        """LOST 뒤 10분이 지나 종료 가능한지 반환한다.

        Args:
            evaluated_at: datetime. 종료 가능 여부 평가 시각.
        Returns:
            bool. 종료 조건 충족 여부.
        """
        return self.status is PersonTrackStatus.LOST and self.lost_at is not None and evaluated_at >= self.lost_at + timedelta(minutes=10)

    def end(self, ended_at: datetime) -> None:
        """종료 조건을 만족한 LOST Track을 ENDED로 전이한다.

        Args:
            ended_at: datetime. 종료 시각.
        Returns:
            None.
        Raises:
            ValueError: LOST 10분 조건을 만족하지 않으면 발생.
        """
        if not self.can_end(ended_at):
            raise ValueError("PersonTrack cannot end yet.")
        self.status, self.ended_at = PersonTrackStatus.ENDED, ended_at
