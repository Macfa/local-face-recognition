"""외부인 등록 제안 엔티티다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Sequence
from uuid import UUID, uuid4

class RegistrationProposalStatus(str, Enum):
    """외부인 표본을 등록할지 결정하는 제안의 생명 주기 상태다."""
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class RegistrationProposal:
    """외부인 판정에 사용한 FaceSample을 등록으로 전환하는 도메인 엔티티다.

    Attributes:
        id: UUID. 제안과 화면 외부인 코드를 만드는 식별자.
        observation_session_id: UUID. 제안의 근거가 된 관찰 세션.
        face_sample_ids: Sequence[UUID]. 등록 템플릿으로 쓸 검증 통과 표본 순서.
        created_at: datetime. 외부인 판정 시각.
        expires_at: datetime. 운영자 응답 마감 시각.
        status: RegistrationProposalStatus. 현재 제안 상태.
        accepted_name: str | None. 승인 시 입력된 이름.
        responded_at: datetime | None. 승인·거절 응답 시각.
    """
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
        """외부인 판정의 근거 표본으로 30분짜리 등록 제안을 생성한다.

        Args: session_id: UUID. 현재 관찰 세션. face_sample_ids: Sequence[UUID]. 검증된 표본들.
            created_at: datetime. 제안 생성 시각.
        Returns: RegistrationProposal. PENDING 상태의 새 제안.
        Raises: ValueError. 표본이 하나도 없을 때.
        """
        if not face_sample_ids:
            raise ValueError("RegistrationProposal requires FaceSamples.")
        return cls(uuid4(), session_id, tuple(face_sample_ids), created_at, created_at + timedelta(minutes=30), RegistrationProposalStatus.PENDING, None, None)

    def accept(self, name: str, responded_at: datetime) -> None:
        """유효 기한 안의 PENDING 제안을 이름과 함께 ACCEPTED로 전이한다.

        Raises: ValueError. 이름이 비었거나 이미 응답·만료된 제안일 때.
        """
        self._ensure_pending(responded_at)
        normalized = name.strip()
        if not normalized:
            raise ValueError("Registration name is required.")
        self.status, self.accepted_name, self.responded_at = RegistrationProposalStatus.ACCEPTED, normalized, responded_at

    def reject(self, responded_at: datetime) -> None:
        """유효 기한 안의 PENDING 제안을 REJECTED로 전이한다.

        Raises: ValueError. 이미 응답·만료된 제안일 때.
        """
        self._ensure_pending(responded_at)
        self.status, self.responded_at = RegistrationProposalStatus.REJECTED, responded_at

    def expire(self, now: datetime) -> None:
        """기한이 지난 PENDING 제안을 EXPIRED로 전이한다.

        Raises: ValueError. 아직 만료되지 않았거나 PENDING 상태가 아닐 때.
        """
        if self.status is not RegistrationProposalStatus.PENDING or now < self.expires_at:
            raise ValueError("RegistrationProposal cannot expire yet.")
        self.status = RegistrationProposalStatus.EXPIRED

    def _ensure_pending(self, responded_at: datetime) -> None:
        """응답 시점에 제안이 아직 응답 가능한 상태인지 검증한다.

        Raises: ValueError. PENDING이 아니거나 응답 기한을 넘겼을 때.
        """
        if self.status is not RegistrationProposalStatus.PENDING or responded_at > self.expires_at:
            raise ValueError("RegistrationProposal is not answerable.")
