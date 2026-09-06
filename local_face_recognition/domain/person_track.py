"""PersonTrack 생명주기 엔티티다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

class PersonTrackStatus(str, Enum):
    TRACKING = "TRACKING"
    LOST = "LOST"
    ENDED = "ENDED"

@dataclass
class PersonTrack:
    id: UUID
    started_at: datetime
    status: PersonTrackStatus
    lost_at: datetime | None = None
    ended_at: datetime | None = None

    @classmethod
    def start(cls, started_at: datetime) -> "PersonTrack":
        return cls(uuid4(), started_at, PersonTrackStatus.TRACKING)

    def mark_lost(self, lost_at: datetime) -> None:
        if self.status is not PersonTrackStatus.TRACKING or lost_at < self.started_at:
            raise ValueError("Invalid PersonTrack lost transition.")
        self.status, self.lost_at = PersonTrackStatus.LOST, lost_at

    def can_end(self, evaluated_at: datetime) -> bool:
        return self.status is PersonTrackStatus.LOST and self.lost_at is not None and evaluated_at >= self.lost_at + timedelta(minutes=10)

    def end(self, ended_at: datetime) -> None:
        if not self.can_end(ended_at):
            raise ValueError("PersonTrack cannot end yet.")
        self.status, self.ended_at = PersonTrackStatus.ENDED, ended_at
