"""통과한 기술 얼굴 결과를 FaceSample·IdentityDecision으로 전환하는 유스케이스다."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from ..components.types import FaceCandidate, FaceQuality
from ..domain import FacePose, FaceQualitySummary, FaceSample, FaceSampleEmbedding, IdentityDecision, RegistrationProposal
from .observation_service import ObservationService
from .ports import FaceSampleRepository

@dataclass(frozen=True)
class FaceSampleResult:
    """FaceSample 처리 유스케이스의 호출자 반환 DTO.

    Attributes:
        status: str. FAILED, DISCARDED, ANALYZING, IDENTIFIED, UNREGISTERED 중 결과 상태.
        person_name: str | None. IDENTIFIED일 때 화면에 표시할 이름.
        proposal_id: str | None. UNREGISTERED일 때 생성된 임시 코드·등록 제안 ID.
        error: str | None. FAILED일 때 오류 설명.
    """
    status: str
    person_name: str | None
    proposal_id: str | None
    error: str | None = None

class FaceSampleService:
    """통과 얼굴 표본의 저장·템플릿 검색·신원 판단·등록 제안 생성을 한 순서로 조율한다."""
    def __init__(self, repository: FaceSampleRepository, observations: ObservationService) -> None:
        """표본 저장소와 관찰 세션 서비스를 연결한다.

        Args:
            repository: FaceSampleRepository. 표본·결정·프로필 검색 영속 포트.
            observations: ObservationService. 세션 상태와 신원 정책 적용 서비스.
        Returns: None.
        """
        self._repository, self._observations = repository, observations

    def process(self, storage_track_id: int, candidate: FaceCandidate, quality: FaceQuality, embedding: np.ndarray, crop: np.ndarray) -> FaceSampleResult:
        """한 채택 표본을 저장하고 현재 신원 결론 또는 등록 제안을 반환한다.

        Args:
            storage_track_id: int. 저장소용 관찰 Track 키.
            candidate: FaceCandidate. 품질 평가 전 얼굴 후보 메타데이터.
            quality: FaceQuality. 통과한 품질 결과.
            embedding: np.ndarray. L2 정규화 얼굴 벡터.
            crop: np.ndarray. private 저장용 BGR 얼굴 이미지.
        Returns:
            FaceSampleResult. 신원 상태·이름·등록 제안 또는 오류.
        """
        try:
            if not self._observations.accepts_face_samples(storage_track_id):
                return FaceSampleResult("DISCARDED", None, None)
            context = self._observations.context_for(storage_track_id)
            sample = FaceSample.create(context.session.id, candidate.captured_at, str(self._repository.next_face_crop_path()), FaceQualitySummary(quality.score, {"detection_score":candidate.detection_score,"brightness":quality.brightness,"sharpness":quality.sharpness,"face_size":float(quality.face_size),"occlusion_probability":quality.occlusion_probability}), FacePose(candidate.yaw_degrees,candidate.pitch_degrees,candidate.roll_degrees), FaceSampleEmbedding(embedding.tolist(),"buffalo_l/w600k_r50",candidate.captured_at), datetime.now(timezone.utc))
            self._repository.save_domain_face_sample(sample,candidate.bbox,{"detection_score":candidate.detection_score,"brightness":quality.brightness,"sharpness":quality.sharpness,"face_size":quality.face_size,"occlusion_probability":quality.occlusion_probability,"yaw_proxy":candidate.yaw_proxy,"yaw_degrees":candidate.yaw_degrees,"pitch_degrees":candidate.pitch_degrees,"roll_degrees":candidate.roll_degrees,"reason":quality.reason},crop)
            decision=IdentityDecision.decide(sample.id,self._repository.search_active_profile_candidates(embedding),candidate.captured_at)
            self._repository.save_domain_identity_decision(decision)
            decisions = self._repository.load_domain_identity_decisions(context.session.id)
            identity=self._observations.apply_identity(storage_track_id,decisions,candidate.captured_at)
            name=self._repository.find_active_profile_name(identity.person_profile_id) if identity.person_profile_id else None
            proposal_id = None
            if identity.status.value == "UNREGISTERED":
                proposal = RegistrationProposal.create(
                    context.session.id,
                    [decision.face_sample_id for decision in decisions],
                    candidate.captured_at,
                )
                if self._repository.save_registration_proposal(proposal):
                    proposal_id = str(proposal.id)
            return FaceSampleResult(identity.status.value,name,proposal_id)
        except Exception as error:
            return FaceSampleResult("FAILED",None,None,str(error))
