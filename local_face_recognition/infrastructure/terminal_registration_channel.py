"""표준 입력과 출력을 사용하는 로컬 등록 채널 어댑터다."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread

from ..application_services.registration_coordinator import RegistrationRequest, RegistrationResponse


class TerminalRegistrationChannel:
    """활성 요청 하나의 Y/N·이름 입력을 proposal_id 기반 응답으로 반환하는 터미널 어댑터."""

    def __init__(self, stop_requested: Event) -> None:
        """종료 Event와 RegistrationResponse 대기열을 준비한다.

        Args: stop_requested: Event. 앱 종료 신호.
        Returns: None.
        """
        self._stop_requested = stop_requested
        self._responses: Queue[RegistrationResponse] = Queue()

    def present(self, request: RegistrationRequest) -> None:
        """활성 RegistrationRequest 하나를 별도 입력 작업에서 질문한다.

        Args: request: RegistrationRequest. 화면 코드와 proposal_id를 가진 현재 요청.
        Returns: None.
        """
        Thread(target=self._collect_response, args=(request,), daemon=True).start()

    def next_response(self, timeout_seconds: float) -> RegistrationResponse | None:
        """입력 작업이 만든 proposal_id 기반 응답을 제한 시간 내 반환한다.

        Args: timeout_seconds: float. Queue 대기 시간.
        Returns: RegistrationResponse | None. 응답 또는 시간 초과 None.
        """
        try:
            return self._responses.get(timeout=timeout_seconds)
        except Empty:
            return None

    def close(self) -> None:
        """채널 종료를 요청한다.

        표준 ``input``은 다른 스레드에서 안전하게 강제 중단할 수 없다. 앱의 종료
        Event를 존중하고 입력 스레드는 daemon으로 남겨 프로세스 종료를 막지 않는다.

        Returns:
            None.
        """

    def _collect_response(self, request: RegistrationRequest) -> None:
        """Y/N 확인 뒤 같은 코드의 이름을 받고 한 개의 응답으로 만든다.

        Args: request: RegistrationRequest. FIFO에서 활성화된 유일한 등록 요청.
        Returns: None. 결과는 내부 ``Queue[RegistrationResponse]``로 전달한다.
        """
        print(
            f"registration_prompt_active code={request.display_code} track_id={request.track_id}",
            flush=True,
        )
        answer = self._read_answer(request.display_code)
        if answer is None:
            self._responses.put(RegistrationResponse(request.proposal_id, "CANCELLED"))
            return
        if answer == "N":
            self._responses.put(RegistrationResponse(request.proposal_id, "REJECTED"))
            return
        name = self._read_name(request.display_code)
        if name is None:
            self._responses.put(RegistrationResponse(request.proposal_id, "CANCELLED"))
            return
        self._responses.put(RegistrationResponse(request.proposal_id, "REGISTER", name))

    def _read_answer(self, display_code: str) -> str | None:
        """대소문자와 무관한 Y/N 값이 입력될 때까지 현재 요청만 기다린다.

        Args: display_code: str. 운영자가 현재 요청을 구별할 임시 인물 코드.
        Returns: str | None. ``Y`` 또는 ``N``. stdin 종료·앱 종료면 ``None``.
        """
        while not self._stop_requested.is_set():
            try:
                answer = input(f"임시 인물 {display_code}의 이름을 등록하시겠습니까? (Y/N): ").strip().casefold()
            except EOFError:
                return None
            if answer in {"y", "n"}:
                return answer.upper()
            print("등록 입력이 올바르지 않습니다. Y 또는 N을 입력하세요.", flush=True)
        return None

    def _read_name(self, display_code: str) -> str | None:
        """공백이 아닌 이름이 입력될 때까지 같은 등록 요청을 유지한다.

        Args: display_code: str. 운영자가 현재 요청을 구별할 임시 인물 코드.
        Returns: str | None. 정규화 전 이름 또는 취소 시 ``None``.
        """
        while not self._stop_requested.is_set():
            try:
                name = input(f"임시 인물 {display_code}의 이름을 입력하세요: ").strip()
            except EOFError:
                return None
            if name:
                return name
            print("이름을 비워둘 수 없습니다. 다시 입력하세요.", flush=True)
        return None
