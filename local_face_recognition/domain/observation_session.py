"""ObservationSession 엔티티다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
from .identity import CurrentIdentityResult, CurrentIdentityStatus

class ObservationSessionStatus(str, Enum):
    """ObservationSession의 표본 수집 가능 상태.

    Values:
        ACTIVE: 신원 판단과 표본 수집을 진행할 수 있음.
        ENDED: 더 이상 신원 결과를 갱신할 수 없음.
    """
    ACTIVE = "ACTIVE"
    ENDED = "ENDED"

@dataclass
class ObservationSession:
    """한 PersonTrack에서 표본과 현재 신원 결론을 누적하는 엔티티.

    Attributes:
        id: UUID. 관찰 세션 ID.
        person_track_id: UUID. 소속 PersonTrack ID.
        started_at: datetime. 세션 시작 시각.
        status: ObservationSessionStatus. 활성 또는 종료 상태.
        ended_at: datetime | None. 종료 시각.
        current_identity: CurrentIdentityResult. 최신 누적 신원 판단.
    """
    id: UUID
    person_track_id: UUID
    started_at: datetime
    status: ObservationSessionStatus
    ended_at: datetime | None
    current_identity: CurrentIdentityResult

    @classmethod
    def start(cls, person_track_id: UUID, started_at: datetime) -> "ObservationSession":
        """ANALYZING 신원 상태를 가진 활성 관찰 세션을 생성한다.

        Args:
            person_track_id: UUID. 세션을 소유하는 Track ID.
            started_at: datetime. 세션 시작 시각.
        Returns:
            ObservationSession. ACTIVE 상태의 새 세션.
        """
        session_id = uuid4()
        return cls(session_id, person_track_id, started_at, ObservationSessionStatus.ACTIVE, None, CurrentIdentityResult(session_id, CurrentIdentityStatus.ANALYZING, None, None, started_at))

    def accepts_face_samples(self) -> bool:
        """현재 세션이 FaceSample을 더 받을 수 있는지 반환한다.

        Returns:
            bool. ACTIVE이며 신원이 ANALYZING일 때만 True. IDENTIFIED 또는 UNREGISTERED면 False.
        """
        return self.status is ObservationSessionStatus.ACTIVE and self.current_identity.status is CurrentIdentityStatus.ANALYZING

    def update_current_identity(self, result: CurrentIdentityResult) -> None:
        """같은 세션의 시간 역행 없는 신원 결과만 현재 값으로 반영한다.

        Args:
            result: CurrentIdentityResult. 적용할 누적 신원 판단.
        Returns:
            None.
        Raises:
            ValueError: 세션 불일치, 종료 상태, 과거 시각 결과면 발생.
        """
        if self.status is not ObservationSessionStatus.ACTIVE or result.observation_session_id != self.id or result.evaluated_at < self.current_identity.evaluated_at:
            raise ValueError("Invalid current identity update.")
        self.current_identity = result

    def end(self, ended_at: datetime) -> None:
        """활성 세션을 종료해 이후 표본·신원 갱신을 막는다.

        Args:
            ended_at: datetime. 세션 종료 시각.
        Returns:
            None.
        Raises:
            ValueError: 비활성 세션이거나 시작 전 시각이면 발생.
        """
        if self.status is not ObservationSessionStatus.ACTIVE or ended_at < self.started_at:
            raise ValueError("Invalid ObservationSession end transition.")
        self.status, self.ended_at = ObservationSessionStatus.ENDED, ended_at
