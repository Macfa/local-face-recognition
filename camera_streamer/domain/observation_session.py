"""ObservationSession 엔티티다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
from .identity import CurrentIdentityResult, CurrentIdentityStatus

class ObservationSessionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    ENDED = "ENDED"

@dataclass
class ObservationSession:
    id: UUID
    person_track_id: UUID
    started_at: datetime
    status: ObservationSessionStatus
    ended_at: datetime | None
    current_identity: CurrentIdentityResult

    @classmethod
    def start(cls, person_track_id: UUID, started_at: datetime) -> "ObservationSession":
        session_id = uuid4()
        return cls(session_id, person_track_id, started_at, ObservationSessionStatus.ACTIVE, None, CurrentIdentityResult(session_id, CurrentIdentityStatus.ANALYZING, None, None, started_at))

    def accepts_face_samples(self) -> bool:
        return self.status is ObservationSessionStatus.ACTIVE and self.current_identity.status is CurrentIdentityStatus.ANALYZING

    def update_current_identity(self, result: CurrentIdentityResult) -> None:
        if self.status is not ObservationSessionStatus.ACTIVE or result.observation_session_id != self.id or result.evaluated_at < self.current_identity.evaluated_at:
            raise ValueError("Invalid current identity update.")
        self.current_identity = result

    def end(self, ended_at: datetime) -> None:
        if self.status is not ObservationSessionStatus.ACTIVE or ended_at < self.started_at:
            raise ValueError("Invalid ObservationSession end transition.")
        self.status, self.ended_at = ObservationSessionStatus.ENDED, ended_at
