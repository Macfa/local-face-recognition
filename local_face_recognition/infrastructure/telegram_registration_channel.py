"""Telegram Bot API로 등록 요청과 운영자 응답을 교환하는 어댑터다."""

from __future__ import annotations

import json
import re
from queue import Empty, Queue
from threading import Event, Lock, Thread
from time import sleep
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ..application_services.registration_coordinator import RegistrationRequest, RegistrationResponse


class TelegramRegistrationChannel:
    """허용된 Telegram 운영자와 코드 기반 등록 응답만 교환하는 RegistrationChannel 어댑터.

    Telegram으로는 임시 코드와 운영자 응답·이름만 보낸다. 카메라 프레임, 얼굴 crop, 임베딩,
    유사도와 사람 프로필 정보는 전송하지 않는다.
    """

    def __init__(self, bot_token: str, allowed_chat_id: str, stop_requested: Event) -> None:
        """Bot API 인증 정보와 허용된 운영자 채팅을 설정하고 수신 작업자를 시작한다.

        Args:
            bot_token: str. Telegram BotFather가 발급한 비밀 토큰.
            allowed_chat_id: str. 등록 명령을 보낼 수 있는 단 하나의 Telegram chat ID.
            stop_requested: Event. 애플리케이션 종료 신호.
        Raises:
            ValueError. 토큰 또는 chat ID가 비어 있으면 발생.
        """
        if not bot_token.strip() or not allowed_chat_id.strip():
            raise ValueError("Telegram bot token and allowed chat ID are required.")
        self._base_url = f"https://api.telegram.org/bot{bot_token.strip()}"
        self._allowed_chat_id = allowed_chat_id.strip()
        self._stop_requested = stop_requested
        self._channel_stop_requested = Event()
        self._lock = Lock()
        self._responses: Queue[RegistrationResponse] = Queue()
        self._active_request: RegistrationRequest | None = None
        self._responding_proposal_id: str | None = None
        self._next_update_id: int | None = None
        self._polling_thread = Thread(target=self._poll_updates, name="telegram-registration-poll", daemon=True)
        self._polling_thread.start()

    def present(self, request: RegistrationRequest) -> None:
        """활성 임시 인물 코드의 등록·거절 명령 형식을 Telegram 운영자에게 전송한다."""
        with self._lock:
            self._active_request = request
            self._responding_proposal_id = None
        try:
            self._call_api(
                "sendMessage",
                {
                    "chat_id": self._allowed_chat_id,
                    "text": (
                        f"임시 인물 {request.display_code}의 등록이 필요합니다.\n"
                        f"등록: /register {request.display_code} 이름\n"
                        f"거절: /reject {request.display_code}"
                    ),
                },
            )
            print(f"telegram_registration_prompt_sent code={request.display_code}", flush=True)
        except (HTTPError, URLError, OSError, ValueError) as error:
            print(f"telegram_registration_prompt_failed error_type={type(error).__name__}", flush=True)

    def next_response(self, timeout_seconds: float) -> RegistrationResponse | None:
        """Telegram 수신 작업자가 만든 현재 활성 요청의 응답을 반환한다."""
        try:
            return self._responses.get(timeout=timeout_seconds)
        except Empty:
            return None

    def is_prompt_active(self) -> bool:
        """Telegram은 콘솔 입력을 점유하지 않으므로 항상 False를 반환한다."""
        return False

    def close(self) -> None:
        """수신 long polling 작업자에 종료를 요청한다."""
        self._channel_stop_requested.set()
        self._polling_thread.join(timeout=1)

    def _poll_updates(self) -> None:
        """허용된 chat ID의 명령만 읽어 활성 RegistrationRequest 응답으로 변환한다."""
        while not self._stop_requested.is_set() and not self._channel_stop_requested.is_set():
            try:
                parameters: dict[str, Any] = {"timeout": 20}
                if self._next_update_id is not None:
                    parameters["offset"] = self._next_update_id
                updates = self._call_api("getUpdates", parameters)
                for update in updates:
                    self._next_update_id = int(update["update_id"]) + 1
                    self._accept_update(update)
            except (HTTPError, URLError, OSError, ValueError, KeyError, TypeError) as error:
                print(f"telegram_registration_poll_failed error_type={type(error).__name__}", flush=True)
                sleep(1)

    def _accept_update(self, update: dict[str, Any]) -> None:
        """한 Bot API update가 현재 허용된 등록 응답인지 확인하고 큐에 넣는다."""
        message = update.get("message")
        if not isinstance(message, dict) or str(message.get("chat", {}).get("id")) != self._allowed_chat_id:
            return
        text = message.get("text")
        if not isinstance(text, str):
            return
        with self._lock:
            request = self._active_request
            if request is None or self._responding_proposal_id == request.proposal_id:
                return
            response = self._response_for_text(request, text)
            if response is not None:
                self._responding_proposal_id = request.proposal_id
                self._responses.put(response)

    @staticmethod
    def _response_for_text(request: RegistrationRequest, text: str) -> RegistrationResponse | None:
        """한 Telegram 명령을 활성 코드에 귀속된 RegistrationResponse로 변환한다."""
        register = re.fullmatch(r"/register(?:@\w+)?\s+(U-[0-9A-F]{8})\s+(.+)", text.strip(), flags=re.IGNORECASE)
        if register is not None and register.group(1).upper() == request.display_code.upper():
            name = register.group(2).strip()
            return RegistrationResponse(request.proposal_id, "REGISTER", name) if name else None
        reject = re.fullmatch(r"/reject(?:@\w+)?\s+(U-[0-9A-F]{8})", text.strip(), flags=re.IGNORECASE)
        if reject is not None and reject.group(1).upper() == request.display_code.upper():
            return RegistrationResponse(request.proposal_id, "REJECTED")
        return None

    def _call_api(self, method: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """Telegram Bot API 호출을 수행하고 성공 응답의 result 목록을 반환한다."""
        request = Request(
            f"{self._base_url}/{method}",
            data=urlencode(parameters).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not payload.get("ok"):
            raise ValueError("Telegram API returned an unsuccessful response.")
        result = payload.get("result")
        return result if isinstance(result, list) else []
