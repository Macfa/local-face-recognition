"""기술 추적 이벤트를 도메인 관찰 세션으로 전환하는 유스케이스다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..domain import (
    CurrentIdentityResult,
    CurrentIdentityStatus,
    IdentityDecision,
    IdentityPolicy,
    ObservationSession,
    PersonTrack,
)
from .ports import ObservationRepository


@dataclass
class ObservationContext:
    """실행 Track 키와 이에 대응하는 도메인 Track·관찰 세션의 묶음이다.

    Attributes:
        storage_track_id: int. 실행 중 고유한 저장소 Track 키.
        person_track: PersonTrack. 시간 기반 생명 주기를 가진 추적 엔티티.
        session: ObservationSession. 표본·신원 판단이 귀속되는 관찰 단위.
    """

    storage_track_id: int
    person_track: PersonTrack
    session: ObservationSession


class ObservationService:
    """기술 Track 이벤트를 도메인 Track·세션 전이로 조율한다."""

    def __init__(self, repository: ObservationRepository, policy: IdentityPolicy) -> None:
        """관찰 저장 포트와 누적 신원 정책을 설정한다.

        Args:
            repository: ObservationRepository. Track·세션·신원 결과 영속화 포트.
            policy: IdentityPolicy. 세션 판단 근거를 현재 신원으로 해석하는 정책.
        Returns:
            None.
        """
        self._repository, self._policy, self._contexts = repository, policy, {}

    def start(self, storage_track_id: int, at: datetime) -> ObservationContext:
        """확정 Track에 새 관찰 세션을 시작하거나 기존 문맥을 반환한다."""
        context = self._contexts.get(storage_track_id)
        if context is None:
            track = PersonTrack.start(at)
            context = ObservationContext(storage_track_id, track, ObservationSession.start(track.id, at))
            self._repository.save_track_and_session(storage_track_id, context.person_track, context.session)
            self._contexts[storage_track_id] = context
        return context

    def context_for(self, storage_track_id: int) -> ObservationContext:
        """활성 관찰 문맥을 반환한다.

        Raises:
            RuntimeError: 활성 Track 키의 관찰 문맥이 없을 때.
        """
        if storage_track_id not in self._contexts:
            raise RuntimeError("Observation context is unavailable.")
        return self._contexts[storage_track_id]

    def mark_lost(self, storage_track_id: int, at: datetime) -> None:
        """화면에서 사라진 Track을 LOST로 전이해 보관 시간 계산을 시작한다."""
        context = self.context_for(storage_track_id)
        context.person_track.mark_lost(at)
        self._repository.save_lost_track(context.person_track)

    def accepts_face_samples(self, storage_track_id: int) -> bool:
        """현재 세션이 아직 FaceSample을 받아야 하는지 반환한다."""
        return self.context_for(storage_track_id).session.accepts_face_samples()

    def apply_identity(self, storage_track_id: int, decisions: list[IdentityDecision], at: datetime) -> CurrentIdentityResult:
        """누적 판단 근거를 정책에 적용하고 세션의 현재 신원 결과를 저장한다."""
        context = self.context_for(storage_track_id)
        result = self._policy.evaluate(context.session.id, decisions, at)
        context.session.update_current_identity(result)
        self._repository.save_domain_current_identity(result)
        return result

    def end_if_possible(self, storage_track_id: int, at: datetime) -> bool:
        """LOST 보관 시간이 끝난 문맥을 종료·정리하고 성공 여부를 반환한다."""
        context = self.context_for(storage_track_id)
        if not context.person_track.can_end(at):
            return False
        context.person_track.end(at)
        context.session.end(at)
        self._repository.save_ended_track_and_session(context.person_track, context.session)
        del self._contexts[storage_track_id]
        return True

    def current_status(self, storage_track_id: int) -> CurrentIdentityStatus:
        """화면 표시에 사용할 활성 세션의 현재 신원 상태를 반환한다."""
        return self.context_for(storage_track_id).session.current_identity.status
