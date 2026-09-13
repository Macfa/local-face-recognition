"""운영 모드별 저장·검색 인프라 어댑터다."""

from .local_sqlite_repository import LocalSQLiteRepository
from .terminal_registration_channel import TerminalRegistrationChannel
from .telegram_registration_channel import TelegramRegistrationChannel

__all__ = ["LocalSQLiteRepository", "TelegramRegistrationChannel", "TerminalRegistrationChannel"]
