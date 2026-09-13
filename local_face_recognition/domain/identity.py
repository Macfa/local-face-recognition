"""표본별 신원 결정과 세션 누적 판단 정책이다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Sequence
from uuid import UUID, uuid4

class CurrentIdentityStatus(str, Enum):
    """ObservationSession이 현재 내린 신원 판단 상태.

    Values:
        ANALYZING: 표본을 더 모아야 함.
        IDENTIFIED: 등록 PersonProfile로 확인됨.
        UNREGISTERED: 등록 프로필로 확인되지 않았으나 임시 코드로 보관할 대상.
    """
    ANALYZING = "ANALYZING"
    IDENTIFIED = "IDENTIFIED"
    UNREGISTERED = "UNREGISTERED"

@dataclass(frozen=True)
class CurrentIdentityResult:
    """한 관찰 세션의 최신 누적 신원 판단 값 객체.

    Attributes:
        observation_session_id: UUID. 판단 대상 세션.
        status: CurrentIdentityStatus. 판단 상태.
        person_profile_id: UUID | None. IDENTIFIED일 때의 등록 인물.
        identity_decision_id: UUID | None. 결론을 뒷받침하는 표본 판단.
        evaluated_at: datetime. 평가 시각.
    """
    observation_session_id: UUID
    status: CurrentIdentityStatus
    person_profile_id: UUID | None
    identity_decision_id: UUID | None
    evaluated_at: datetime

    def __post_init__(self) -> None:
        """상태와 근거 ID 조합이 유효한지 검증한다.

        Returns:
            None.
        Raises:
            ValueError: IDENTIFIED 근거가 없거나 비식별 상태가 프로필을 참조하면 발생.
        """
        if self.status is CurrentIdentityStatus.IDENTIFIED and (self.person_profile_id is None or self.identity_decision_id is None):
            raise ValueError("IDENTIFIED requires decision evidence.")
        if self.status is not CurrentIdentityStatus.IDENTIFIED and self.person_profile_id is not None:
            raise ValueError("Only IDENTIFIED references a profile.")

@dataclass(frozen=True)
class PersonProfileCandidate:
    """한 FaceSample 임베딩이 검색한 등록 얼굴 템플릿 후보.

    Attributes:
        person_profile_id: UUID. 후보 인물 프로필.
        person_profile_face_template_id: UUID. 일치한 템플릿.
        similarity: float. 코사인 유사도.
    """
    person_profile_id: UUID
    person_profile_face_template_id: UUID
    similarity: float

@dataclass(frozen=True)
class IdentityDecision:
    """단일 FaceSample과 가장 유사한 등록 템플릿의 불변 비교 결과.

    Attributes:
        id: UUID. 판단 ID.
        face_sample_id: UUID. 판단에 사용한 표본.
        candidate_person_profile_id: UUID | None. 최고 후보 인물.
        candidate_face_template_id: UUID | None. 최고 후보 템플릿.
        similarity: float | None. 최고 유사도.
        decided_at: datetime. 비교 시각.
    """
    id: UUID
    face_sample_id: UUID
    candidate_person_profile_id: UUID | None
    candidate_face_template_id: UUID | None
    similarity: float | None
    decided_at: datetime

    @classmethod
    def decide(cls, face_sample_id: UUID, candidates: Sequence[PersonProfileCandidate], decided_at: datetime) -> "IdentityDecision":
        """후보 중 유사도가 가장 높은 템플릿 하나를 표본 판단으로 고정한다.

        Args:
            face_sample_id: UUID. 비교한 FaceSample ID.
            candidates: Sequence[PersonProfileCandidate]. 저장소 검색 후보.
            decided_at: datetime. 판단 시각.
        Returns:
            IdentityDecision. 후보가 없으면 profile/template/similarity가 None인 결과.
        """
        candidate = min(candidates, key=lambda value: (-value.similarity, str(value.person_profile_face_template_id)), default=None)
        if candidate is None:
            return cls(uuid4(), face_sample_id, None, None, None, decided_at)
        return cls(uuid4(), face_sample_id, candidate.person_profile_id, candidate.person_profile_face_template_id, candidate.similarity, decided_at)

class IdentityPolicy:
    """세션의 여러 IdentityDecision을 IDENTIFIED·UNREGISTERED·ANALYZING으로 누적 판단한다."""

    def __init__(self, recognition_similarity: float = .60, maximum_samples: int = 5) -> None:
        """등록 인물로 인정할 최소 코사인 유사도를 설정한다.

        Args:
            recognition_similarity: float. 후보 판단을 지지 증거로 셀 하한.
            maximum_samples: int. 등록 근거가 없을 때 임시 코드로 전환할 최대 표본 수.
        Returns:
            None.
        """
        self._recognition_similarity = recognition_similarity
        self._maximum_samples = maximum_samples

    def evaluate(self, session_id: UUID, decisions: Sequence[IdentityDecision], evaluated_at: datetime) -> CurrentIdentityResult:
        """표본 판단 이력을 누적 규칙으로 현재 신원 결과로 변환한다.

        Args:
            session_id: UUID. 결과를 적용할 관찰 세션.
            decisions: Sequence[IdentityDecision]. 세션의 표본별 판단 이력.
            evaluated_at: datetime. 정책 평가 시각.
        Returns:
            CurrentIdentityResult. 같은 프로필 지지 2개면 IDENTIFIED, 고품질 비중복 표본 5개면 UNREGISTERED.
        """
        grouped: dict[UUID, list[IdentityDecision]] = {}
        for decision in decisions:
            if decision.candidate_person_profile_id and decision.similarity is not None and decision.similarity >= self._recognition_similarity:
                grouped.setdefault(decision.candidate_person_profile_id, []).append(decision)
        supported = [values for values in grouped.values() if len(values) >= 2]
        if supported:
            strongest = max((item for values in supported for item in values), key=lambda item: float(item.similarity))
            return CurrentIdentityResult(session_id, CurrentIdentityStatus.IDENTIFIED, strongest.candidate_person_profile_id, strongest.id, evaluated_at)
        if len(decisions) >= self._maximum_samples:
            return CurrentIdentityResult(session_id, CurrentIdentityStatus.UNREGISTERED, None, None, evaluated_at)
        return CurrentIdentityResult(session_id, CurrentIdentityStatus.ANALYZING, None, None, evaluated_at)
