"""등록 제안 응답과 프로필 등록 유스케이스다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from ..domain import CreatePersonProfile, PersonProfile
from .ports import RegistrationRepository

@dataclass(frozen=True)
class RegistrationOutcome:
    """등록 제안 응답 유스케이스의 결과 DTO.

    Attributes:
        status: str. REGISTERED, REJECTED, FAILED 중 처리 결과.
        name: str | None. 성공 등록된 프로필 이름.
        error: str | None. 실패 원인.
    """
    status: str
    name: str | None = None
    error: str | None = None

class RegistrationService:
    """채널과 무관하게 RegistrationProposal의 승인·거절·만료와 프로필 생성을 수행한다."""
    def __init__(self, repository: RegistrationRepository) -> None:
        """등록 제안과 프로필을 저장할 포트를 설정한다.

        Args: repository: RegistrationRepository. 등록 영속 포트.
        Returns: None.
        """
        self._repository = repository

    def respond(self, proposal_id: str, name: str, at: datetime) -> RegistrationOutcome:
        """proposal_id에 연결된 이름 또는 거절을 도메인 전이와 프로필 등록으로 처리한다.

        Args:
            proposal_id: str. RegistrationProposal UUID 문자열.
            name: str. 등록 이름, 빈 문자열이면 거절.
            at: datetime. 채널 응답 시각.
        Returns: RegistrationOutcome. 등록·거절·실패 결과.
        """
        proposal = None
        try:
            proposal = self._repository.load_registration_proposal(UUID(proposal_id))
            if not name.strip():
                proposal.reject(at)
                self._repository.save_registration_rejection(proposal)
                return RegistrationOutcome("REJECTED")
            proposal.accept(name, at)
            profile = PersonProfile.create(CreatePersonProfile(proposal.accepted_name or "", at))
            samples = self._repository.load_face_samples(proposal.face_sample_ids)
            if len(samples) != len(proposal.face_sample_ids):
                raise RuntimeError("Registration proposal face samples are unavailable.")
            templates = [profile.add_face_template(sample, at) for sample in samples]
            profile = self._repository.register_person_profile(proposal, profile, templates)
            return RegistrationOutcome("REGISTERED", profile.name)
        except Exception as error:
            if proposal is not None and proposal.status.value == "ACCEPTED":
                self._repository.record_registration_failure(proposal, str(error))
            return RegistrationOutcome("FAILED", error=str(error))

    def expire_pending(self, now: datetime) -> list[str]:
        """응답 기한이 지난 제안을 만료하고 조율기에서 제거할 ID를 반환한다.

        Args:
            now: datetime. 만료 여부를 판단할 UTC 시각.
        Returns:
            list[str]: EXPIRED로 저장된 RegistrationProposal UUID 문자열 목록.
        Raises:
            RuntimeError: 저장소의 만료 제안 조회 또는 상태 저장이 실패했을 때.
        """
        expired_proposal_ids = []
        for proposal in self._repository.load_expirable_registration_proposals(now):
            proposal.expire(now)
            self._repository.save_registration_expiration(proposal)
            expired_proposal_ids.append(str(proposal.id))
        return expired_proposal_ids
