"""운영 모드별 저장·검색 인프라 어댑터다."""

from .local_sqlite_repository import LocalSQLiteRepository

__all__ = ["LocalSQLiteRepository"]
