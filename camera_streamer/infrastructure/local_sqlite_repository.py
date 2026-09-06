"""로컬 운영 모드에서 도메인 이력과 private 얼굴 crop을 SQLite에 저장한다."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import cv2
import numpy as np

from ..domain import (
    CurrentIdentityResult,
    FacePose,
    FaceSample,
    FaceSampleEmbedding,
    FaceQualitySummary,
    IdentityDecision,
    ObservationSession,
    PersonProfileCandidate as DomainPersonProfileCandidate,
    PersonProfile,
    PersonProfileFaceTemplate,
    PersonTrack,
    RegistrationProposal,
)


@dataclass(frozen=True)
class RegisteredPersonProfile:
    """SQLite 로컬 운영 저장소에 등록한 인물 프로필의 식별 정보다."""

    id: str
    name: str


class LocalSQLiteRepository:
    """로컬 운영 모드의 Repository 포트를 SQLite와 private 파일로 구현한다."""

    def __init__(self, database_path: Path, crop_directory: Path) -> None:
        """로컬 DB 파일과 private 얼굴 crop 디렉터리 위치를 설정한다."""
        self._database_path = database_path
        self._crop_directory = crop_directory

    def initialize(self) -> None:
        """로컬 운영에 필요한 SQLite 테이블과 private crop 디렉터리를 만든다."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._crop_directory.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS person_tracks (
                    id TEXT PRIMARY KEY,
                    internal_track_id INTEGER NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN ('TRACKING', 'LOST')),
                    started_at TEXT NOT NULL,
                    lost_at TEXT
                );

                CREATE TABLE IF NOT EXISTS observation_sessions (
                    id TEXT PRIMARY KEY,
                    person_track_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status = 'ACTIVE'),
                    started_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    id TEXT PRIMARY KEY,
                    observation_session_id TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    bbox_left INTEGER NOT NULL,
                    bbox_top INTEGER NOT NULL,
                    bbox_right INTEGER NOT NULL,
                    bbox_bottom INTEGER NOT NULL,
                    quality_score REAL NOT NULL,
                    quality_factors_json TEXT NOT NULL,
                    face_crop_path TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS face_sample_embeddings (
                    face_sample_id TEXT PRIMARY KEY,
                    embedding_vector BLOB NOT NULL,
                    embedding_model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS person_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL CHECK (trim(name) <> ''),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS person_profile_face_templates (
                    id TEXT PRIMARY KEY,
                    person_profile_id TEXT NOT NULL,
                    source_face_sample_id TEXT NOT NULL UNIQUE,
                    embedding_vector BLOB NOT NULL,
                    embedding_model_version TEXT NOT NULL,
                    enrolled_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS identity_decisions (
                    id TEXT PRIMARY KEY,
                    face_sample_id TEXT NOT NULL UNIQUE,
                    candidate_person_profile_id TEXT,
                    candidate_face_template_id TEXT,
                    similarity REAL,
                    decided_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS observation_session_current_identities (
                    observation_session_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('ANALYZING', 'IDENTIFIED', 'EXTERNAL')),
                    person_profile_id TEXT,
                    identity_decision_id TEXT,
                    evaluated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS track_endings (
                    person_track_id TEXT PRIMARY KEY,
                    observation_session_id TEXT NOT NULL UNIQUE,
                    ended_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS registration_proposals (
                    id TEXT PRIMARY KEY,
                    observation_session_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN ('PENDING', 'ACCEPTED', 'REJECTED', 'EXPIRED')),
                    accepted_name TEXT,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    responded_at TEXT
                );

                CREATE TABLE IF NOT EXISTS registration_handlings (
                    registration_proposal_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status IN ('REGISTERED', 'REJECTED', 'FAILED')),
                    person_profile_id TEXT,
                    completed_at TEXT NOT NULL,
                    failure_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS registration_proposal_face_samples (
                    registration_proposal_id TEXT NOT NULL,
                    face_sample_id TEXT NOT NULL,
                    selection_order INTEGER NOT NULL CHECK (selection_order > 0),
                    PRIMARY KEY (registration_proposal_id, face_sample_id),
                    UNIQUE (registration_proposal_id, selection_order)
                );

                CREATE INDEX IF NOT EXISTS face_samples_session_captured_at_idx
                    ON face_samples (observation_session_id, captured_at);
                """
            )
            self._ensure_column(connection, "person_profiles", "updated_at", "TEXT")
            connection.execute(
                "UPDATE person_profiles SET updated_at = created_at WHERE updated_at IS NULL"
            )
            self._drop_column_if_exists(connection, "person_profiles", "deleted_at")
            self._drop_column_if_exists(
                connection,
                "person_profile_face_templates",
                "deleted_at",
            )
            self._remove_legacy_foreign_keys(connection)
            self._ensure_column(
                connection,
                "registration_proposal_face_samples",
                "selection_order",
                "INTEGER",
            )
            proposal_ids = connection.execute(
                "SELECT DISTINCT registration_proposal_id FROM registration_proposal_face_samples"
            ).fetchall()
            for (proposal_id,) in proposal_ids:
                rows = connection.execute(
                    """
                    SELECT rowid FROM registration_proposal_face_samples
                    WHERE registration_proposal_id = ?
                    ORDER BY face_sample_id
                    """,
                    (proposal_id,),
                ).fetchall()
                for selection_order, (row_id,) in enumerate(rows, start=1):
                    connection.execute(
                        """
                        UPDATE registration_proposal_face_samples
                        SET selection_order = ?
                        WHERE rowid = ? AND selection_order IS NULL
                        """,
                        (selection_order, row_id),
                    )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    registration_proposal_face_samples_selection_order_idx
                ON registration_proposal_face_samples (registration_proposal_id, selection_order)
                """
            )

    def next_face_crop_path(self) -> Path:
        """FaceSample 생성 전에 사용할 충돌 없는 로컬 crop 저장 키를 만든다."""
        return self._crop_directory / f"{uuid4()}.jpg"

    def save_track_and_session(
        self,
        storage_track_id: int,
        track: PersonTrack,
        session: ObservationSession,
    ) -> None:
        """도메인이 시작한 Track과 ObservationSession을 SQLite에 그대로 저장한다."""
        if session.person_track_id != track.id:
            raise ValueError("ObservationSession must belong to the saved PersonTrack.")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO person_tracks (id, internal_track_id, status, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (str(track.id), storage_track_id, track.status.value, track.started_at.isoformat()),
            )
            connection.execute(
                """
                INSERT INTO observation_sessions (id, person_track_id, status, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (str(session.id), str(track.id), session.status.value, session.started_at.isoformat()),
            )
            self._save_current_identity(connection, session.current_identity)

    def save_lost_track(self, track: PersonTrack) -> None:
        """PersonTrack이 수행한 LOST 전이를 영속 상태에 반영한다."""
        if track.lost_at is None:
            raise ValueError("A lost track must have a lost_at value.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE person_tracks SET status = ?, lost_at = ? WHERE id = ?",
                (track.status.value, track.lost_at.isoformat(), str(track.id)),
            )

    def save_ended_track_and_session(self, track: PersonTrack, session: ObservationSession) -> None:
        """SQLite의 기존 종료 이력 테이블에 도메인의 ENDED 전이를 보존한다."""
        if track.ended_at is None or session.ended_at is None:
            raise ValueError("Ended track and session timestamps are required.")
        if session.person_track_id != track.id:
            raise ValueError("ObservationSession must belong to the ended PersonTrack.")
        with self._connect() as connection:
            self._require_exists(connection, "person_tracks", track.id)
            self._require_exists(connection, "observation_sessions", session.id)
            connection.execute(
                """
                INSERT INTO track_endings (person_track_id, observation_session_id, ended_at)
                VALUES (?, ?, ?)
                ON CONFLICT(person_track_id) DO NOTHING
                """,
                (str(track.id), str(session.id), track.ended_at.isoformat()),
            )

    def save_domain_face_sample(
        self,
        face_sample: FaceSample,
        bbox: tuple[int, int, int, int],
        quality_factors: dict[str, float | str],
        face_crop: np.ndarray,
    ) -> Path:
        """이미 생성된 FaceSample의 crop·품질·임베딩을 SQLite에 저장한다."""
        crop_path = Path(face_sample.face_crop_storage_key)
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(crop_path), face_crop):
            raise RuntimeError("Face crop file could not be written.")
        normalized_embedding = np.asarray(face_sample.embedding.vector, dtype=np.float32).copy()
        normalized_embedding /= np.linalg.norm(normalized_embedding)
        left, top, right, bottom = bbox
        try:
            with self._connect() as connection:
                self._require_exists(
                    connection,
                    "observation_sessions",
                    face_sample.observation_session_id,
                )
                connection.execute(
                    """
                    INSERT INTO face_samples (
                        id, observation_session_id, captured_at, bbox_left, bbox_top,
                        bbox_right, bbox_bottom, quality_score, quality_factors_json,
                        face_crop_path, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(face_sample.id),
                        str(face_sample.observation_session_id),
                        face_sample.captured_at.isoformat(),
                        left,
                        top,
                        right,
                        bottom,
                        face_sample.quality.score,
                        json.dumps(quality_factors),
                        str(crop_path),
                        face_sample.created_at.isoformat(),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO face_sample_embeddings (
                        face_sample_id, embedding_vector, embedding_model_version, created_at
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        str(face_sample.id),
                        normalized_embedding.tobytes(),
                        face_sample.embedding.model_version,
                        face_sample.embedding.created_at.isoformat(),
                    ),
                )
        except Exception:
            crop_path.unlink(missing_ok=True)
            raise
        return crop_path

    def search_active_profile_candidates(
        self,
        embedding: np.ndarray,
    ) -> list[DomainPersonProfileCandidate]:
        """활성 프로필의 모든 템플릿을 검색해 도메인 후보 목록으로 반환한다."""
        query_embedding = embedding.astype(np.float32, copy=True)
        norm = np.linalg.norm(query_embedding)
        if norm == 0:
            raise ValueError("A face embedding must not be a zero vector.")
        query_embedding /= norm
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT person_profiles.id, person_profile_face_templates.id,
                       person_profile_face_templates.embedding_vector
                FROM person_profile_face_templates
                JOIN person_profiles
                    ON person_profiles.id = person_profile_face_templates.person_profile_id
                """
            ).fetchall()
        candidates = []
        for profile_id, template_id, vector_bytes in rows:
            template_embedding = np.frombuffer(vector_bytes, dtype=np.float32)
            if template_embedding.shape == query_embedding.shape:
                candidates.append(
                    DomainPersonProfileCandidate(
                        person_profile_id=UUID(profile_id),
                        person_profile_face_template_id=UUID(template_id),
                        similarity=float(np.dot(query_embedding, template_embedding)),
                    )
                )
        return candidates

    def save_domain_identity_decision(self, decision: IdentityDecision) -> None:
        """도메인이 만든 불변 IdentityDecision을 그대로 저장한다."""
        with self._connect() as connection:
            self._require_exists(connection, "face_samples", decision.face_sample_id)
            self._validate_decision_candidate(connection, decision)
            connection.execute(
                """
                INSERT INTO identity_decisions (
                    id, face_sample_id, candidate_person_profile_id,
                    candidate_face_template_id, similarity, decided_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(decision.id),
                    str(decision.face_sample_id),
                    None if decision.candidate_person_profile_id is None else str(decision.candidate_person_profile_id),
                    None if decision.candidate_face_template_id is None else str(decision.candidate_face_template_id),
                    decision.similarity,
                    decision.decided_at.isoformat(),
                ),
            )

    def load_domain_identity_decisions(self, session_id: UUID) -> list[IdentityDecision]:
        """관찰 세션의 불변 신원 판단 이력을 도메인 객체로 복원한다."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT identity_decisions.id, identity_decisions.face_sample_id,
                       identity_decisions.candidate_person_profile_id,
                       identity_decisions.candidate_face_template_id,
                       identity_decisions.similarity, identity_decisions.decided_at
                FROM identity_decisions
                JOIN face_samples ON face_samples.id = identity_decisions.face_sample_id
                WHERE face_samples.observation_session_id = ?
                ORDER BY face_samples.captured_at, identity_decisions.id
                """,
                (str(session_id),),
            ).fetchall()
        return [
            IdentityDecision(
                id=UUID(row[0]),
                face_sample_id=UUID(row[1]),
                candidate_person_profile_id=None if row[2] is None else UUID(row[2]),
                candidate_face_template_id=None if row[3] is None else UUID(row[3]),
                similarity=row[4],
                decided_at=datetime.fromisoformat(row[5]),
            )
            for row in rows
        ]

    def save_domain_current_identity(self, identity: CurrentIdentityResult) -> None:
        """ObservationSession이 반영한 현재 신원 결과를 저장한다."""
        with self._connect() as connection:
            self._save_current_identity(connection, identity)

    def find_active_profile_name(self, person_profile_id: UUID) -> str | None:
        """화면 표시용 활성 PersonProfile 이름을 조회한다."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT name FROM person_profiles WHERE id = ?",
                (str(person_profile_id),),
            ).fetchone()
        return None if row is None else str(row[0])

    def register_person_profile(
        self,
        proposal: RegistrationProposal,
        profile: PersonProfile,
        templates: list[PersonProfileFaceTemplate],
    ) -> RegisteredPersonProfile:
        """승인된 제안의 선택 표본만 새 프로필 템플릿으로 등록한다."""
        if proposal.status.value != "ACCEPTED" or proposal.accepted_name is None:
            raise ValueError("An accepted RegistrationProposal is required.")
        with self._connect() as connection:
            proposal_row = connection.execute(
                "SELECT status FROM registration_proposals WHERE id = ?",
                (str(proposal.id),),
            ).fetchone()
            if proposal_row is None or proposal_row[0] != "PENDING":
                raise RuntimeError("The registration proposal is not pending.")
            expected_sample_ids = {str(face_sample_id) for face_sample_id in proposal.face_sample_ids}
            template_sample_ids = {str(template.source_face_sample_id) for template in templates}
            if (
                len(templates) != len(proposal.face_sample_ids)
                or template_sample_ids != expected_sample_ids
                or any(template.person_profile_id != profile.id for template in templates)
            ):
                raise RuntimeError("Registration proposal face samples are unavailable.")
            rows = connection.execute(
                """
                SELECT face_sample_id FROM registration_proposal_face_samples
                WHERE registration_proposal_id = ?
                """,
                (str(proposal.id),),
            ).fetchall()
            if {row[0] for row in rows} != expected_sample_ids:
                raise RuntimeError("Registration proposal sample relationship is invalid.")
            completed_at = datetime_now_iso()
            connection.execute(
                "INSERT INTO person_profiles (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (
                    str(profile.id),
                    profile.name,
                    profile.created_at.isoformat(),
                    profile.updated_at.isoformat(),
                ),
            )
            for template in templates:
                connection.execute(
                    """
                    INSERT INTO person_profile_face_templates (
                        id, person_profile_id, source_face_sample_id, embedding_vector,
                        embedding_model_version, enrolled_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(template.id),
                        str(template.person_profile_id),
                        str(template.source_face_sample_id),
                        np.asarray(template.embedding_vector, dtype=np.float32).tobytes(),
                        template.embedding_model_version,
                        template.enrolled_at.isoformat(),
                    ),
                )
            connection.execute(
                """
                UPDATE registration_proposals
                SET status = 'ACCEPTED', accepted_name = ?, responded_at = ?
                WHERE id = ?
                """,
                (proposal.accepted_name, proposal.responded_at.isoformat(), str(proposal.id)),
            )
            connection.execute(
                """
                INSERT INTO registration_handlings (
                    registration_proposal_id, status, person_profile_id, completed_at, failure_reason
                ) VALUES (?, 'REGISTERED', ?, ?, NULL)
                """,
                (str(proposal.id), str(profile.id), completed_at),
            )
        return RegisteredPersonProfile(str(profile.id), profile.name)

    def load_face_samples(self, face_sample_ids: list[UUID]) -> list[FaceSample]:
        """등록 제안이 참조한 FaceSample을 도메인 객체로 복원한다."""
        if not face_sample_ids:
            return []
        placeholders = ", ".join("?" for _ in face_sample_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT face_samples.id, face_samples.observation_session_id, face_samples.captured_at,
                       face_samples.face_crop_path, face_samples.quality_score,
                       face_samples.quality_factors_json, face_samples.created_at,
                       face_sample_embeddings.embedding_vector,
                       face_sample_embeddings.embedding_model_version,
                       face_sample_embeddings.created_at
                FROM face_samples
                JOIN face_sample_embeddings ON face_sample_embeddings.face_sample_id = face_samples.id
                WHERE face_samples.id IN ({placeholders})
                """,
                [str(face_sample_id) for face_sample_id in face_sample_ids],
            ).fetchall()
        samples_by_id = {}
        for row in rows:
            factors = json.loads(row[5])
            metrics = {key: float(value) for key, value in factors.items() if isinstance(value, (int, float))}
            samples_by_id[UUID(row[0])] = FaceSample(
                UUID(row[0]), UUID(row[1]), datetime.fromisoformat(row[2]), row[3],
                FaceQualitySummary(float(row[4]), metrics),
                FacePose(float(factors.get("yaw_degrees", 0.0)), float(factors.get("pitch_degrees", 0.0)), float(factors.get("roll_degrees", 0.0))),
                FaceSampleEmbedding(np.frombuffer(row[7], dtype=np.float32).tolist(), row[8], datetime.fromisoformat(row[9])),
                datetime.fromisoformat(row[6]),
            )
        return [samples_by_id[face_sample_id] for face_sample_id in face_sample_ids if face_sample_id in samples_by_id]

    def load_registration_proposal(self, proposal_id: UUID) -> RegistrationProposal:
        """응답 전이에 사용할 PENDING 등록 제안과 선택 표본을 도메인 객체로 복원한다."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT observation_session_id, status, accepted_name, created_at, expires_at, responded_at
                FROM registration_proposals WHERE id = ?
                """,
                (str(proposal_id),),
            ).fetchone()
            sample_rows = connection.execute(
                """
                SELECT face_sample_id FROM registration_proposal_face_samples
                WHERE registration_proposal_id = ? ORDER BY selection_order
                """,
                (str(proposal_id),),
            ).fetchall()
        if row is None:
            raise RuntimeError("Registration proposal is unavailable.")
        from ..domain import RegistrationProposalStatus
        return RegistrationProposal(
            proposal_id, UUID(row[0]), tuple(UUID(sample_row[0]) for sample_row in sample_rows),
            datetime.fromisoformat(row[3]), datetime.fromisoformat(row[4]),
            RegistrationProposalStatus(row[1]), row[2], None if row[5] is None else datetime.fromisoformat(row[5]),
        )

    def load_expirable_registration_proposals(self, now: datetime) -> list[RegistrationProposal]:
        """만료 시각을 지난 PENDING 제안을 도메인 전이용으로 복원한다."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM registration_proposals WHERE status = 'PENDING' AND expires_at <= ?",
                (now.isoformat(),),
            ).fetchall()
        return [self.load_registration_proposal(UUID(row[0])) for row in rows]

    def save_registration_rejection(self, proposal: RegistrationProposal) -> None:
        """도메인이 거절 전이한 제안과 처리 이력을 저장한다."""
        if proposal.status.value != "REJECTED" or proposal.responded_at is None:
            raise ValueError("A rejected RegistrationProposal is required.")
        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE registration_proposals SET status = ?, responded_at = ? WHERE id = ? AND status = 'PENDING'",
                (proposal.status.value, proposal.responded_at.isoformat(), str(proposal.id)),
            )
            if updated.rowcount != 1:
                raise RuntimeError("The registration proposal cannot be rejected.")
            connection.execute(
                """INSERT INTO registration_handlings
                (registration_proposal_id, status, person_profile_id, completed_at, failure_reason)
                VALUES (?, 'REJECTED', NULL, ?, NULL)""",
                (str(proposal.id), proposal.responded_at.isoformat()),
            )

    def save_registration_expiration(self, proposal: RegistrationProposal) -> None:
        """도메인이 만료 전이한 제안 상태를 저장한다."""
        if proposal.status.value != "EXPIRED":
            raise ValueError("An expired RegistrationProposal is required.")
        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE registration_proposals SET status = ? WHERE id = ? AND status = 'PENDING'",
                (proposal.status.value, str(proposal.id)),
            )
            if updated.rowcount != 1:
                raise RuntimeError("The registration proposal cannot be expired.")

    def save_registration_proposal(self, proposal: RegistrationProposal) -> bool:
        """도메인이 만든 등록 제안과 선택된 FaceSample 관계를 한 번만 저장한다."""
        with self._connect() as connection:
            self._require_exists(connection, "observation_sessions", proposal.observation_session_id)
            placeholders = ", ".join("?" for _ in proposal.face_sample_ids)
            rows = connection.execute(
                f"""
                SELECT id FROM face_samples
                WHERE observation_session_id = ? AND id IN ({placeholders})
                """,
                [str(proposal.observation_session_id), *map(str, proposal.face_sample_ids)],
            ).fetchall()
            if {row[0] for row in rows} != {str(face_sample_id) for face_sample_id in proposal.face_sample_ids}:
                raise RuntimeError("RegistrationProposal samples must belong to its ObservationSession.")
            existing = connection.execute(
                "SELECT id FROM registration_proposals WHERE observation_session_id = ?",
                (str(proposal.observation_session_id),),
            ).fetchone()
            if existing is not None:
                return False
            connection.execute(
                """
                INSERT INTO registration_proposals (
                    id, observation_session_id, status, accepted_name, created_at, expires_at, responded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(proposal.id),
                    str(proposal.observation_session_id),
                    proposal.status.value,
                    proposal.accepted_name,
                    proposal.created_at.isoformat(),
                    proposal.expires_at.isoformat(),
                    None if proposal.responded_at is None else proposal.responded_at.isoformat(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO registration_proposal_face_samples (
                    registration_proposal_id, face_sample_id, selection_order
                ) VALUES (?, ?, ?)
                """,
                [
                    (str(proposal.id), str(face_sample_id), index)
                    for index, face_sample_id in enumerate(proposal.face_sample_ids, start=1)
                ],
            )
        return True

    def record_registration_failure(
        self,
        proposal: RegistrationProposal,
        failure_reason: str,
    ) -> None:
        """승인된 도메인 제안과 프로필 생성 실패 이력을 함께 보존한다."""
        if proposal.status.value != "ACCEPTED" or proposal.accepted_name is None or proposal.responded_at is None:
            raise ValueError("An accepted RegistrationProposal is required.")
        with self._connect() as connection:
            updated = connection.execute(
                """UPDATE registration_proposals
                SET status = ?, accepted_name = ?, responded_at = ?
                WHERE id = ? AND status = 'PENDING'""",
                (
                    proposal.status.value,
                    proposal.accepted_name,
                    proposal.responded_at.isoformat(),
                    str(proposal.id),
                ),
            )
            if updated.rowcount != 1:
                return
            connection.execute(
                """
                INSERT INTO registration_handlings (
                    registration_proposal_id, status, person_profile_id, completed_at, failure_reason
                ) VALUES (?, 'FAILED', NULL, ?, ?)
                """,
                (str(proposal.id), proposal.responded_at.isoformat(), failure_reason),
            )

    @staticmethod
    def _remove_legacy_foreign_keys(connection: sqlite3.Connection) -> None:
        """기존 SQLite 파일의 FK 제약을 데이터 보존 방식으로 제거한다."""
        table_rows = connection.execute(
            """
            SELECT name, sql FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql LIKE '%REFERENCES%'
            """
        ).fetchall()
        if not table_rows:
            return

        connection.commit()
        connection.execute("PRAGMA foreign_keys = OFF")
        temporary_tables: list[tuple[str, str]] = []
        for table_name, table_sql in table_rows:
            temporary_name = f"{table_name}_without_fk"
            connection.execute(f"DROP TABLE IF EXISTS {temporary_name}")
            replacement_sql = re.sub(
                r"^CREATE TABLE(?: IF NOT EXISTS)?\s+[^\s(]+",
                f"CREATE TABLE {temporary_name}",
                table_sql,
                count=1,
                flags=re.IGNORECASE,
            )
            replacement_sql = re.sub(
                r"\s+REFERENCES\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)",
                "",
                replacement_sql,
                flags=re.IGNORECASE,
            )
            connection.execute(replacement_sql)
            connection.execute(f"INSERT INTO {temporary_name} SELECT * FROM {table_name}")
            temporary_tables.append((str(table_name), temporary_name))

        for table_name, _ in temporary_tables:
            connection.execute(f"DROP TABLE {table_name}")
        for table_name, temporary_name in temporary_tables:
            connection.execute(f"ALTER TABLE {temporary_name} RENAME TO {table_name}")
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS face_samples_session_captured_at_idx
            ON face_samples (observation_session_id, captured_at)
            """
        )

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ) -> None:
        """기존 로컬 운영 DB에 새 nullable 컬럼을 안전하게 추가한다."""
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}
        if column_name not in columns:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")

    @staticmethod
    def _drop_column_if_exists(
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
    ) -> None:
        """소프트 딜리트에서 전환한 기존 SQLite 파일의 레거시 열을 제거한다."""
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table_name})")}
        if column_name in columns:
            connection.execute(f"ALTER TABLE {table_name} DROP COLUMN {column_name}")

    @staticmethod
    def _find_session_id(
        connection: sqlite3.Connection,
        storage_track_id: int,
        *,
        require_active: bool,
    ) -> str | None:
        """실행 Track 키에 연결된 관찰 세션 ID를 조회한다."""
        status_condition = "AND person_tracks.status = 'TRACKING'" if require_active else ""
        row = connection.execute(
            f"""
            SELECT observation_sessions.id
            FROM observation_sessions
            JOIN person_tracks ON person_tracks.id = observation_sessions.person_track_id
            WHERE person_tracks.internal_track_id = ? {status_condition}
            """,
            (storage_track_id,),
        ).fetchone()
        return None if row is None else str(row[0])

    @staticmethod
    def _require_exists(
        connection: sqlite3.Connection,
        table_name: str,
        entity_id: UUID,
    ) -> None:
        """FK 대신 저장하려는 도메인 식별자의 존재를 명시적으로 검증한다."""
        row = connection.execute(
            f"SELECT 1 FROM {table_name} WHERE id = ?",
            (str(entity_id),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Required {table_name} record is unavailable.")

    @staticmethod
    def _validate_decision_candidate(
        connection: sqlite3.Connection,
        decision: IdentityDecision,
    ) -> None:
        """IdentityDecision의 후보 프로필·템플릿 소유 관계를 검증한다."""
        profile_id = decision.candidate_person_profile_id
        template_id = decision.candidate_face_template_id
        if profile_id is None and template_id is None:
            return
        if profile_id is None or template_id is None or decision.similarity is None:
            raise ValueError("IdentityDecision candidate fields must be provided together.")
        row = connection.execute(
            """
            SELECT 1 FROM person_profile_face_templates
            WHERE id = ? AND person_profile_id = ?
            """,
            (str(template_id), str(profile_id)),
        ).fetchone()
        if row is None:
            raise RuntimeError("IdentityDecision candidate relationship is unavailable.")

    @staticmethod
    def _save_current_identity(
        connection: sqlite3.Connection,
        identity: CurrentIdentityResult,
    ) -> None:
        """도메인 CurrentIdentityResult를 SQLite 현재 결과 행으로 기록한다."""
        LocalSQLiteRepository._require_exists(
            connection,
            "observation_sessions",
            identity.observation_session_id,
        )
        if identity.person_profile_id is not None:
            LocalSQLiteRepository._require_exists(
                connection,
                "person_profiles",
                identity.person_profile_id,
            )
        if identity.identity_decision_id is not None:
            row = connection.execute(
                """
                SELECT 1 FROM identity_decisions
                JOIN face_samples ON face_samples.id = identity_decisions.face_sample_id
                WHERE identity_decisions.id = ? AND face_samples.observation_session_id = ?
                """,
                (str(identity.identity_decision_id), str(identity.observation_session_id)),
            ).fetchone()
            if row is None:
                raise RuntimeError("Current identity evidence must belong to its ObservationSession.")
        connection.execute(
            """
            INSERT INTO observation_session_current_identities (
                observation_session_id, status, person_profile_id,
                identity_decision_id, evaluated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(observation_session_id) DO UPDATE SET
                status = excluded.status,
                person_profile_id = excluded.person_profile_id,
                identity_decision_id = excluded.identity_decision_id,
                evaluated_at = excluded.evaluated_at
            """,
            (
                str(identity.observation_session_id),
                identity.status.value,
                None if identity.person_profile_id is None else str(identity.person_profile_id),
                None if identity.identity_decision_id is None else str(identity.identity_decision_id),
                identity.evaluated_at.isoformat(),
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        """저장 단위마다 독립 트랜잭션을 사용하는 짧은 SQLite 연결을 연다."""
        return sqlite3.connect(self._database_path)


def datetime_now_iso() -> str:
    """등록 처리 완료 시각을 UTC ISO 형식으로 만든다."""
    return datetime.now(timezone.utc).isoformat()


def datetime_now() -> datetime:
    """등록 제안 만료 비교에 사용할 현재 UTC 시각을 만든다."""
    return datetime.now(timezone.utc)
