"""Application 서비스가 인프라에 요구하는 입력·출력 계약이다.

도메인과 유스케이스는 SQLite·터미널 구현을 직접 의존하지 않는다. 이 포트가
영속화와 등록 채널의 책임 경계를 선언하고, 인프라 어댑터가 이를 구현한다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, Sequence
from uuid import UUID

import numpy as np

from ..components.types import BBox
from ..domain import (
    CurrentIdentityResult,
    FaceSample,
    IdentityDecision,
    ObservationSession,
    PersonProfile,
    PersonProfileCandidate,
    PersonProfileFaceTemplate,
    PersonTrack,
    RegistrationProposal,
)
from .registration_coordinator import RegistrationRequest, RegistrationResponse


class ObservationRepository(Protocol):
    """관찰 Track·세션·현재 신원 상태를 영속화하는 계약이다."""

    def save_track_and_session(self, storage_track_id: int, track: PersonTrack, session: ObservationSession) -> None:
        """새 기술 Track 키와 도메인 Track·세션을 저장한다."""
        ...

    def save_lost_track(self, track: PersonTrack) -> None:
        """LOST로 전이한 PersonTrack을 저장한다."""
        ...

    def save_ended_track_and_session(self, track: PersonTrack, session: ObservationSession) -> None:
        """종료 가능한 Track과 세션의 종료 이력을 저장한다."""
        ...

    def save_domain_current_identity(self, identity: CurrentIdentityResult) -> None:
        """ObservationSession의 최신 신원 결과를 저장한다."""
        ...


class FaceSampleRepository(ObservationRepository, Protocol):
    """FaceSample 유스케이스가 영속화와 신원 후보 검색에 요구하는 계약이다."""

    def next_face_crop_path(self) -> Path:
        """새 얼굴 crop 파일에 사용할 로컬 전용 경로를 반환한다."""
        ...

    def save_domain_face_sample(
        self, sample: FaceSample, bbox: BBox, quality_metadata: dict[str, object], crop: np.ndarray
    ) -> None:
        """통과 표본의 crop·품질·임베딩을 같은 저장 단위로 기록한다."""
        ...

    def search_active_profile_candidates(self, embedding: np.ndarray) -> Sequence[PersonProfileCandidate]:
        """입력 임베딩과 저장 템플릿의 유사도 후보를 반환한다."""
        ...

    def save_domain_identity_decision(self, decision: IdentityDecision) -> None:
        """FaceSample 하나에 대한 불변 신원 판단 근거를 저장한다."""
        ...

    def load_domain_identity_decisions(self, session_id: object) -> list[IdentityDecision]:
        """세션 누적 판단에 필요한 판단 근거를 시간순으로 반환한다."""
        ...

    def find_active_profile_name(self, person_profile_id: object) -> str | None:
        """식별된 프로필의 화면 표시 이름을 반환하며 없으면 ``None``을 반환한다."""
        ...

    def save_registration_proposal(self, proposal: RegistrationProposal) -> bool:
        """임시 인물 등록 제안을 한 번만 저장하고 성공 여부를 반환한다."""
        ...


class RegistrationRepository(Protocol):
    """등록 응답 유스케이스가 요구하는 프로필·제안 저장 계약이다."""

    def load_registration_proposal(self, proposal_id: UUID) -> RegistrationProposal:
        """응답할 등록 제안을 선택 표본과 함께 복원한다."""
        ...

    def load_expirable_registration_proposals(self, now: datetime) -> list[RegistrationProposal]:
        """``now``까지 응답되지 않은 만료 대상 제안을 반환한다."""
        ...

    def save_registration_rejection(self, proposal: RegistrationProposal) -> None:
        """거절된 등록 제안과 처리 결과를 저장한다."""
        ...

    def save_registration_expiration(self, proposal: RegistrationProposal) -> None:
        """시간 초과된 등록 제안의 만료 전이를 저장한다."""
        ...

    def purge_expired_unregistered_data(self, now: datetime) -> int:
        """만료된 미등록 제안의 얼굴 표본과 임시 등록 정보를 완전 삭제한다."""
        ...

    def load_face_samples(self, face_sample_ids: Sequence[object]) -> list[FaceSample]:
        """등록 제안이 참조한 FaceSample을 입력 순서대로 반환한다."""
        ...

    def register_person_profile(
        self, proposal: RegistrationProposal, profile: PersonProfile, templates: Sequence[PersonProfileFaceTemplate]
    ) -> object:
        """승인된 제안으로 프로필과 템플릿을 생성하고 등록 결과를 반환한다."""
        ...

    def record_registration_failure(self, proposal: RegistrationProposal, message: str) -> None:
        """승인 후 영속화에 실패한 이력을 저장한다."""
        ...


class RegistrationChannel(Protocol):
    """등록 질문을 전달하고 proposal_id 기반 응답을 제공하는 운영 채널 포트다."""

    def present(self, request: RegistrationRequest) -> None:
        """한 개의 활성 요청을 운영자에게 제시한다."""
        ...

    def next_response(self, timeout_seconds: float) -> RegistrationResponse | None:
        """제한 시간 안에 도착한 응답을 반환하고 없으면 ``None``을 반환한다."""
        ...

    def close(self) -> None:
        """채널이 보유한 로컬 리소스를 종료한다."""
        ...
