"""표본별 신원 결정과 세션 누적 판단 정책이다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence
from uuid import UUID, uuid4

class CurrentIdentityStatus(str, Enum):
    ANALYZING = "ANALYZING"
    IDENTIFIED = "IDENTIFIED"
    EXTERNAL = "EXTERNAL"

@dataclass(frozen=True)
class CurrentIdentityResult:
    observation_session_id: UUID
    status: CurrentIdentityStatus
    person_profile_id: UUID | None
    identity_decision_id: UUID | None
    evaluated_at: datetime

    def __post_init__(self) -> None:
        if self.status is CurrentIdentityStatus.IDENTIFIED and (self.person_profile_id is None or self.identity_decision_id is None):
            raise ValueError("IDENTIFIED requires decision evidence.")
        if self.status is not CurrentIdentityStatus.IDENTIFIED and self.person_profile_id is not None:
            raise ValueError("Only IDENTIFIED references a profile.")

@dataclass(frozen=True)
class PersonProfileCandidate:
    person_profile_id: UUID
    person_profile_face_template_id: UUID
    similarity: float

@dataclass(frozen=True)
class IdentityDecision:
    id: UUID
    face_sample_id: UUID
    candidate_person_profile_id: UUID | None
    candidate_face_template_id: UUID | None
    similarity: float | None
    decided_at: datetime

    @classmethod
    def decide(cls, face_sample_id: UUID, candidates: Sequence[PersonProfileCandidate], decided_at: datetime) -> "IdentityDecision":
        candidate = min(candidates, key=lambda value: (-value.similarity, str(value.person_profile_face_template_id)), default=None)
        if candidate is None:
            return cls(uuid4(), face_sample_id, None, None, None, decided_at)
        return cls(uuid4(), face_sample_id, candidate.person_profile_id, candidate.person_profile_face_template_id, candidate.similarity, decided_at)

class IdentityPolicy:
    def __init__(self, recognition_similarity: float = .60) -> None:
        self._recognition_similarity = recognition_similarity

    def evaluate(self, session_id: UUID, decisions: Sequence[IdentityDecision], evaluated_at: datetime) -> CurrentIdentityResult:
        grouped: dict[UUID, list[IdentityDecision]] = {}
        for decision in decisions:
            if decision.candidate_person_profile_id and decision.similarity is not None and decision.similarity >= self._recognition_similarity:
                grouped.setdefault(decision.candidate_person_profile_id, []).append(decision)
        supported = [values for values in grouped.values() if len(values) >= 2]
        if supported:
            strongest = max((item for values in supported for item in values), key=lambda item: float(item.similarity))
            return CurrentIdentityResult(session_id, CurrentIdentityStatus.IDENTIFIED, strongest.candidate_person_profile_id, strongest.id, evaluated_at)
        if len(decisions) >= 3 and not grouped:
            return CurrentIdentityResult(session_id, CurrentIdentityStatus.EXTERNAL, None, None, evaluated_at)
        return CurrentIdentityResult(session_id, CurrentIdentityStatus.ANALYZING, None, None, evaluated_at)
