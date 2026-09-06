"""외부인 등록 제안 엔티티다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence
from uuid import UUID, uuid4

class RegistrationProposalStatus(str, Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class RegistrationProposal:
    id: UUID
    observation_session_id: UUID
    face_sample_ids: Sequence[UUID]
    created_at: datetime
    expires_at: datetime
    status: RegistrationProposalStatus
    accepted_name: str | None
    responded_at: datetime | None

    @classmethod
    def create(cls, session_id: UUID, face_sample_ids: Sequence[UUID], created_at: datetime) -> "RegistrationProposal":
        if not face_sample_ids:
            raise ValueError("RegistrationProposal requires FaceSamples.")
        return cls(uuid4(), session_id, tuple(face_sample_ids), created_at, created_at + timedelta(minutes=30), RegistrationProposalStatus.PENDING, None, None)

    def accept(self, name: str, responded_at: datetime) -> None:
        self._ensure_pending(responded_at)
        normalized = name.strip()
        if not normalized:
            raise ValueError("Registration name is required.")
        self.status, self.accepted_name, self.responded_at = RegistrationProposalStatus.ACCEPTED, normalized, responded_at

    def reject(self, responded_at: datetime) -> None:
        self._ensure_pending(responded_at)
        self.status, self.responded_at = RegistrationProposalStatus.REJECTED, responded_at

    def expire(self, now: datetime) -> None:
        if self.status is not RegistrationProposalStatus.PENDING or now < self.expires_at:
            raise ValueError("RegistrationProposal cannot expire yet.")
        self.status = RegistrationProposalStatus.EXPIRED

    def _ensure_pending(self, responded_at: datetime) -> None:
        if self.status is not RegistrationProposalStatus.PENDING or responded_at > self.expires_at:
            raise ValueError("RegistrationProposal is not answerable.")
