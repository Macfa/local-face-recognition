"""교차·가림 뒤 현재 화면 신원을 조용히 재검증하는 유스케이스다."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from ..domain import PersonProfileCandidate


@dataclass(frozen=True)
class IdentityReverificationResult:
    """재검증 표본 하나를 반영한 결과 DTO.

    Attributes:
        status: str. INCONCLUSIVE, CONFIRMED, CONFLICT 중 하나.
        candidate_person_profile_id: UUID | None. 가장 강한 후보 프로필.
    """

    status: str
    candidate_person_profile_id: UUID | None


class IdentityReverificationService:
    """의심 Track의 비중복 얼굴 근거를 누적해 현재 이름 유지·교체 시점만 결정한다."""

    def __init__(self, recognition_similarity: float = .60, required_support: int = 2) -> None:
        """현재 프로필 확인과 충돌 선언에 필요한 최소 근거 수를 설정한다."""
        self._recognition_similarity = recognition_similarity
        self._required_support = required_support
        self._evidence: dict[int, list[UUID | None]] = {}

    def observe(
        self,
        track_id: int,
        current_person_profile_id: UUID,
        candidates: list[PersonProfileCandidate],
    ) -> IdentityReverificationResult:
        """한 비중복 고품질 얼굴의 검색 후보를 누적한다.

        Args:
            track_id: int. 런타임 기술 Track ID.
            current_person_profile_id: UUID. 화면에 표시 중인 등록 프로필.
            candidates: list[PersonProfileCandidate]. 현재 얼굴 벡터의 검색 결과.
        Returns:
            IdentityReverificationResult. 현재 이름 확인, 다른 사람 근거 누적, 또는 보류 결과.
        """
        strongest = max(candidates, key=lambda item: item.similarity, default=None)
        profile_id = (
            strongest.person_profile_id
            if strongest is not None and strongest.similarity >= self._recognition_similarity
            else None
        )
        evidence = self._evidence.setdefault(track_id, [])
        evidence.append(profile_id)
        counts = Counter(evidence)
        if counts[current_person_profile_id] >= self._required_support:
            self.clear(track_id)
            return IdentityReverificationResult("CONFIRMED", current_person_profile_id)
        if profile_id != current_person_profile_id and counts[profile_id] >= self._required_support:
            self.clear(track_id)
            return IdentityReverificationResult("CONFLICT", profile_id)
        return IdentityReverificationResult("INCONCLUSIVE", profile_id)

    def clear(self, track_id: int) -> None:
        """해결 또는 종료된 Track의 일시 재검증 근거를 폐기한다."""
        self._evidence.pop(track_id, None)
