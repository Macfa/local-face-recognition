"""등록 인물과 얼굴 템플릿 엔티티다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
from uuid import UUID, uuid4
from .face_sample import FaceSample

@dataclass(frozen=True)
class CreatePersonProfile:
    """등록 제안 승인으로 새 PersonProfile을 만들기 위한 입력값이다."""

    name: str
    created_at: datetime


@dataclass
class PersonProfile:
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, command: CreatePersonProfile) -> "PersonProfile":
        """승인된 이름으로 등록 인물 프로필을 생성한다."""
        normalized = command.name.strip()
        if not normalized:
            raise ValueError("PersonProfile name is required.")
        return cls(uuid4(), normalized, command.created_at, command.created_at)

    def add_face_template(
        self,
        source_face_sample: FaceSample,
        added_at: datetime,
    ) -> "PersonProfileFaceTemplate":
        """승인 제안이 선택한 FaceSample의 기존 임베딩으로 템플릿을 만든다."""
        self.updated_at = added_at
        return PersonProfileFaceTemplate(
            uuid4(),
            self.id,
            source_face_sample.id,
            source_face_sample.embedding.vector,
            source_face_sample.embedding.model_version,
            added_at,
        )

@dataclass
class PersonProfileFaceTemplate:
    id: UUID
    person_profile_id: UUID
    source_face_sample_id: UUID
    embedding_vector: Sequence[float]
    embedding_model_version: str
    enrolled_at: datetime
