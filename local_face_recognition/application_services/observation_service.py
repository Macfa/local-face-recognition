"""PersonTrack과 ObservationSession 유스케이스다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from .ports import ObservationRepository
from ..domain import CurrentIdentityResult, CurrentIdentityStatus, IdentityDecision, IdentityPolicy, ObservationSession, PersonTrack

@dataclass
class ObservationContext:
    storage_track_id: int
    person_track: PersonTrack
    session: ObservationSession

class ObservationService:
    """기술 Track 이벤트를 도메인 Track·세션 전이로 조율한다."""
    def __init__(self, repository: ObservationRepository, policy: IdentityPolicy) -> None:
        self._repository, self._policy, self._contexts = repository, policy, {}

    def start(self, storage_track_id: int, at: datetime) -> ObservationContext:
        context = self._contexts.get(storage_track_id)
        if context is None:
            track = PersonTrack.start(at)
            context = ObservationContext(storage_track_id, track, ObservationSession.start(track.id, at))
            self._repository.save_track_and_session(storage_track_id, context.person_track, context.session)
            self._contexts[storage_track_id] = context
        return context

    def context_for(self, storage_track_id: int) -> ObservationContext:
        if storage_track_id not in self._contexts:
            raise RuntimeError("Observation context is unavailable.")
        return self._contexts[storage_track_id]

    def mark_lost(self, storage_track_id: int, at: datetime) -> None:
        context = self.context_for(storage_track_id)
        context.person_track.mark_lost(at)
        self._repository.save_lost_track(context.person_track)

    def accepts_face_samples(self, storage_track_id: int) -> bool:
        return self.context_for(storage_track_id).session.accepts_face_samples()

    def apply_identity(self, storage_track_id: int, decisions: list[IdentityDecision], at: datetime) -> CurrentIdentityResult:
        context = self.context_for(storage_track_id)
        result = self._policy.evaluate(context.session.id, decisions, at)
        context.session.update_current_identity(result)
        self._repository.save_domain_current_identity(result)
        return result

    def end_if_possible(self, storage_track_id: int, at: datetime) -> bool:
        context = self.context_for(storage_track_id)
        if not context.person_track.can_end(at):
            return False
        context.person_track.end(at); context.session.end(at)
        self._repository.save_ended_track_and_session(context.person_track, context.session)
        del self._contexts[storage_track_id]
        return True

    def current_status(self, storage_track_id: int) -> CurrentIdentityStatus:
        return self.context_for(storage_track_id).session.current_identity.status
