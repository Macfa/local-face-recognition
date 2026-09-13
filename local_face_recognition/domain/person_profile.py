"""등록 인물과 얼굴 템플릿 엔티티다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
from uuid import UUID, uuid4
from .face_sample import FaceSample

@dataclass(frozen=True)
class CreatePersonProfile:
    """등록 제안 승인으로 새 PersonProfile을 만들기 위한 입력값이다.

    Attributes:
        name: str. 운영자가 입력한 표시 이름.
        created_at: datetime. 승인·생성 시각.
    """

    name: str
    created_at: datetime


@dataclass
class PersonProfile:
    """등록 인물의 이름과 얼굴 템플릿을 소유하는 도메인 엔티티다.

    Attributes:
        id: UUID. 프로필 식별자.
        name: str. 공백을 제거한 표시 이름.
        created_at: datetime. 최초 등록 시각.
        updated_at: datetime. 마지막 템플릿 추가 시각.
    """
    id: UUID
    name: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, command: CreatePersonProfile) -> "PersonProfile":
        """승인된 이름으로 등록 인물 프로필을 생성한다.

        Args: command: CreatePersonProfile. 이름과 생성 시각.
        Returns: PersonProfile. 영속화 전의 새 프로필.
        Raises: ValueError. 이름이 공백일 때.
        """
        normalized = command.name.strip()
        if not normalized:
            raise ValueError("PersonProfile name is required.")
        return cls(uuid4(), normalized, command.created_at, command.created_at)

    def add_face_template(
        self,
        source_face_sample: FaceSample,
        added_at: datetime,
    ) -> "PersonProfileFaceTemplate":
        """승인 제안이 선택한 FaceSample의 기존 임베딩으로 템플릿을 만든다.

        Args:
            source_face_sample: FaceSample. 검증을 통과해 등록 제안에 포함된 표본.
            added_at: datetime. 템플릿 등록 시각.
        Returns: PersonProfileFaceTemplate. 복제 대신 표본의 임베딩을 참조한 템플릿.
        """
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
    """등록 프로필 검색에 사용하는 불변 얼굴 임베딩 템플릿이다.

    Attributes:
        id: UUID. 템플릿 식별자.
        person_profile_id: UUID. 템플릿 소유 프로필 식별자.
        source_face_sample_id: UUID. 근거가 된 원본 FaceSample 식별자.
        embedding_vector: Sequence[float]. 정규화된 얼굴 임베딩.
        embedding_model_version: str. 임베딩을 생성한 모델 버전.
        enrolled_at: datetime. 템플릿 등록 시각.
    """
    id: UUID
    person_profile_id: UUID
    source_face_sample_id: UUID
    embedding_vector: Sequence[float]
    embedding_model_version: str
    enrolled_at: datetime
