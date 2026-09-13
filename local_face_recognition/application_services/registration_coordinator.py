"""등록 요청의 순서와 현재 응답 대상을 관리하는 애플리케이션 서비스다."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class RegistrationRequest:
    """하나의 임시 인물 등록 제안을 채널에 전달할 때 필요한 식별 정보.

    Attributes: proposal_id: str; display_code: str; track_id: int; storage_track_id: int.
    """

    proposal_id: str
    display_code: str
    track_id: int
    storage_track_id: int


@dataclass(frozen=True)
class RegistrationResponse:
    """채널이 proposal_id에 연결해 돌려주는 등록 의사와 이름.

    Attributes: proposal_id: str; status: str; name: str | None.
    """

    proposal_id: str
    status: str
    name: str | None = None


class RegistrationCoordinator:
    """FIFO 등록 대기열에서 한 번에 한 RegistrationRequest만 채널에 전달하는 서비스."""

    def __init__(self) -> None:
        """대기열과 현재 응답 대상을 빈 상태로 준비한다."""
        self._lock = Lock()
        self._pending: list[RegistrationRequest] = []
        self._active: RegistrationRequest | None = None

    def enqueue(self, request: RegistrationRequest) -> None:
        """임시 인물 전환 순서대로 중복 없는 요청을 대기열에 추가한다.

        Args: request: RegistrationRequest. proposal_id를 가진 신규 요청.
        Returns: None.
        """
        with self._lock:
            proposal_ids = {item.proposal_id for item in self._pending}
            if request.proposal_id in proposal_ids or (
                self._active is not None and self._active.proposal_id == request.proposal_id
            ):
                return
            self._pending.append(request)

    def activate_next(self) -> RegistrationRequest | None:
        """활성 요청이 없을 때만 FIFO 첫 요청을 활성화한다.

        Returns: RegistrationRequest | None. 새 활성 요청 또는 이미 활성/대기 없음이면 None.
        """
        with self._lock:
            if self._active is not None or not self._pending:
                return None
            self._active = self._pending.pop(0)
            return self._active

    def active_request(self) -> RegistrationRequest | None:
        """현재 채널이 응답해야 하는 단 하나의 요청을 반환한다.

        Returns: RegistrationRequest | None. 현재 활성 요청 또는 None.
        """
        with self._lock:
            return self._active

    def complete(self, proposal_id: str) -> bool:
        """응답 처리된 활성 요청을 비워 다음 FIFO 요청을 진행하게 한다.

        Args: proposal_id: str. 완료할 RegistrationProposal ID.
        Returns: bool. 현재 활성 요청과 일치해 제거했는지 여부.
        """
        with self._lock:
            if self._active is None or self._active.proposal_id != proposal_id:
                return False
            self._active = None
            return True

    def discard(self, proposal_id: str) -> None:
        """만료 요청을 대기열과 활성 대상에서 제거한다.

        Args: proposal_id: str. 제거할 RegistrationProposal ID.
        Returns: None.
        """
        with self._lock:
            self._pending = [item for item in self._pending if item.proposal_id != proposal_id]
            if self._active is not None and self._active.proposal_id == proposal_id:
                self._active = None
