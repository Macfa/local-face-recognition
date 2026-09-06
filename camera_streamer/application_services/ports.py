"""Application 서비스가 요구하는 저장소 계약이다."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..components.types import BBox
from ..domain import (
    CurrentIdentityResult,
    FaceSample,
    IdentityDecision,
    ObservationSession,
    PersonProfileCandidate,
    PersonProfile,
    PersonProfileFaceTemplate,
    PersonTrack,
    RegistrationProposal,
)

class ObservationRepository(Protocol):
    def save_track_and_session(self, storage_track_id: int, track: PersonTrack, session: ObservationSession) -> None: ...
    def save_lost_track(self, track: PersonTrack) -> None: ...
    def save_ended_track_and_session(self, track: PersonTrack, session: ObservationSession) -> None: ...
    def save_domain_current_identity(self, identity: CurrentIdentityResult) -> None: ...


class FaceSampleRepository(ObservationRepository, Protocol):
    """FaceSample 유스케이스가 영속화와 검색에 요구하는 계약이다."""

    def next_face_crop_path(self) -> Path: ...
    def save_domain_face_sample(
        self,
        sample: FaceSample,
        bbox: BBox,
        quality_metadata: dict[str, object],
        crop: np.ndarray,
    ) -> None: ...
    def search_active_profile_candidates(self, embedding: np.ndarray) -> Sequence[PersonProfileCandidate]: ...
    def save_domain_identity_decision(self, decision: IdentityDecision) -> None: ...
    def load_domain_identity_decisions(self, session_id: object) -> list[IdentityDecision]: ...
    def find_active_profile_name(self, person_profile_id: object) -> str | None: ...
    def save_registration_proposal(self, proposal: RegistrationProposal) -> bool: ...


class RegistrationRepository(Protocol):
    """등록 응답 유스케이스가 요구하는 프로필·제안 저장 계약이다."""

    def load_registration_proposal(self, proposal_id: UUID) -> RegistrationProposal: ...
    def load_expirable_registration_proposals(self, now: datetime) -> list[RegistrationProposal]: ...
    def save_registration_rejection(self, proposal: RegistrationProposal) -> None: ...
    def save_registration_expiration(self, proposal: RegistrationProposal) -> None: ...
    def load_face_samples(self, face_sample_ids: Sequence[object]) -> list[FaceSample]: ...
    def register_person_profile(
        self,
        proposal: RegistrationProposal,
        profile: PersonProfile,
        templates: Sequence[PersonProfileFaceTemplate],
    ): ...
    def record_registration_failure(self, proposal: RegistrationProposal, message: str) -> None: ...
