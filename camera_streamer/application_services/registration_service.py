"""등록 제안 응답과 프로필 등록 유스케이스다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from ..domain import CreatePersonProfile, PersonProfile
from .ports import RegistrationRepository

@dataclass(frozen=True)
class RegistrationOutcome:
    status: str
    name: str | None = None
    error: str | None = None

class RegistrationService:
    """Terminal이 전달한 이름 또는 취소만 등록 저장소에 적용한다."""
    def __init__(self, repository: RegistrationRepository) -> None:
        self._repository = repository

    def respond(self, proposal_id: str, name: str, at: datetime) -> RegistrationOutcome:
        """Terminal 응답을 도메인 제안 전이와 프로필 등록으로 처리한다."""
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

    def expire_pending(self, now: datetime) -> int:
        """응답 기한이 지난 제안에 도메인 만료 전이를 적용한다."""
        expired_count = 0
        for proposal in self._repository.load_expirable_registration_proposals(now):
            proposal.expire(now)
            self._repository.save_registration_expiration(proposal)
            expired_count += 1
        return expired_count
