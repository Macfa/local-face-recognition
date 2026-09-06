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
    status: str
    person_name: str | None
    proposal_id: str | None
    error: str | None = None

class FaceSampleService:
    """표본 영속·검색·결정·세션 신원 반영의 순서를 단일 유스케이스로 보장한다."""
    def __init__(self, repository: FaceSampleRepository, observations: ObservationService) -> None:
        self._repository, self._observations = repository, observations

    def process(self, storage_track_id: int, candidate: FaceCandidate, quality: FaceQuality, embedding: np.ndarray, crop: np.ndarray) -> FaceSampleResult:
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
            if identity.status.value == "EXTERNAL":
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
