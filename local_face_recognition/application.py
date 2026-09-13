"""카메라 표시와 비동기 분석 작업자를 조율하는 애플리케이션 진입 흐름이다."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Dict, List, Set, Tuple
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .components import (
    FaceAnalyzer,
    FaceEmbeddingComponent,
    FaceOcclusionEvaluator,
    FaceQualityEvaluator,
    InMemoryFaceSampleStore,
    IoUPersonTracker,
    PersonDetector,
)
from .components.types import BBox, FaceCandidate, FaceQuality, TrackEvent, TrackedPerson
from .domain import CurrentIdentityStatus, IdentityPolicy
from .application_services.observation_service import ObservationService
from .application_services.registration_coordinator import RegistrationCoordinator, RegistrationRequest
from .application_services.registration_service import RegistrationService
from .application_services.face_sample_service import FaceSampleService
from .infrastructure.local_sqlite_repository import LocalSQLiteRepository
from .infrastructure.terminal_registration_channel import TerminalRegistrationChannel


_DISPLAY_FONT_PATHS = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
)


def _log_operational_failure(stage: str, error: Exception) -> None:
    """민감한 파일 경로나 SQL 내용을 노출하지 않고 작업 실패를 기록한다.

    Args:
        stage: str. 실패한 작업 경계의 고정된 식별자.
        error: Exception. 내부 예외이며 유형만 로그에 기록한다.
    Returns:
        None.
    """
    print(f"operation_failed stage={stage} error_type={type(error).__name__}", flush=True)


@lru_cache(maxsize=8)
def _display_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """현재 OS에서 한국어를 표시할 수 있는 화면 글꼴을 찾는다."""
    for path in _DISPLAY_FONT_PATHS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


@lru_cache(maxsize=128)
def _render_unicode_label(text: str, color: Tuple[int, int, int], size: int) -> np.ndarray:
    """같은 한글 라벨을 매 프레임 다시 렌더링하지 않도록 BGRA 비트맵으로 보관한다."""
    font = _display_font(size)
    left, top, right, bottom = font.getbbox(text)
    image = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)), (0, 0, 0, 0))
    ImageDraw.Draw(image).text((-left, -top), text, font=font, fill=(color[2], color[1], color[0], 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)


def _draw_display_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    *,
    font_scale: float,
) -> None:
    """영문은 OpenCV로, 한글이 포함된 라벨은 OS 글꼴 비트맵으로 프레임에 표시한다."""
    if text.isascii():
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        return

    bitmap = _render_unicode_label(text, color, max(14, round(font_scale * 32)))
    x, baseline_y = origin
    y = baseline_y - bitmap.shape[0]
    x0, y0 = max(0, x), max(0, y)
    x1 = min(frame.shape[1], x + bitmap.shape[1])
    y1 = min(frame.shape[0], y + bitmap.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    source = bitmap[y0 - y:y1 - y, x0 - x:x1 - x]
    target = frame[y0:y1, x0:x1]
    alpha = source[:, :, 3:4].astype(np.float32) / 255.0
    target[:] = (source[:, :, :3] * alpha + target * (1.0 - alpha)).astype(np.uint8)


@dataclass(frozen=True)
class _FaceOverlay:
    """분석 결과를 다음 카메라 프레임에 표시하기 위한 얼굴 오버레이다."""

    bbox: BBox
    label: str
    accepted: bool


@dataclass(frozen=True)
class _FaceSampleStorageRequest:
    """로컬 운영 저장 작업자에게 전달하는 통과 얼굴 표본과 crop 데이터다."""

    candidate: FaceCandidate
    embedding: np.ndarray
    quality: FaceQuality
    face_crop: np.ndarray
    storage_track_id: int
    sample_number: int


@dataclass(frozen=True)
class _TrackStorageRequest:
    """저장소용 실행 단위 Track 키와 추적 이벤트를 묶는다."""

    event: TrackEvent
    storage_track_id: int


@dataclass(frozen=True)
class _LostTrack:
    """화면에서 사라진 뒤 종료 대기 중인 추적의 시각과 저장 키다."""

    storage_track_id: int
    technical_track_id: int
    lost_at: datetime


class VisionApplication:
    """카메라 표시, 비동기 분석, 도메인 서비스 호출을 연결하는 대표 오케스트레이터.

    입력은 OpenCV 프레임과 등록 채널 응답이며, 출력은 화면 오버레이·SQLite 이력·등록 결과다.
    검출·품질·신원·등록 규칙은 각각의 컴포넌트와 서비스에 위임한다.
    """

    def __init__(self) -> None:
        """운영 컴포넌트·서비스·작업 큐를 조합한다.

        Args: 없음. 모델과 SQLite 위치는 로컬 실행 환경에서 계산한다.
        Returns: None.
        """
        root = _resource_root()
        self._detector = PersonDetector(root / "models" / "yolo11n.pt")
        self._tracker = IoUPersonTracker()
        insightface_model_root = _insightface_model_root(root)
        self._face_analyzer = FaceAnalyzer(insightface_model_root)
        self._face_embedding = FaceEmbeddingComponent(insightface_model_root)
        self._quality_evaluator = FaceQualityEvaluator(
            occlusion_evaluator=FaceOcclusionEvaluator(root / "models" / "face_occlusion.onnx"),
            enforce_pose_limits=True,
        )
        self._samples = InMemoryFaceSampleStore()
        data_root = _application_data_root()
        self._repository = LocalSQLiteRepository(
            data_root / "local_face_recognition.sqlite3",
            data_root / "face-crops",
        )
        self._repository.initialize()
        self._observation_service = ObservationService(self._repository, IdentityPolicy())
        self._registration_service = RegistrationService(self._repository)
        self._face_sample_service = FaceSampleService(self._repository, self._observation_service)
        self._last_face_attempt_at: Dict[int, float] = {}
        self._analysis_queue: Queue[Tuple[np.ndarray, datetime]] = Queue(maxsize=1)
        self._storage_queue: Queue[object] = Queue()
        self._stop_requested = Event()
        self._registration_coordinator = RegistrationCoordinator()
        self._registration_channel = TerminalRegistrationChannel(self._stop_requested)
        self._snapshot_lock = Lock()
        self._track_end_delay_seconds = 10 * 60
        self._latest_people: List[Tuple[TrackedPerson, int]] = []
        self._latest_faces: Dict[int, _FaceOverlay] = {}
        self._analysis_faces: Dict[int, _FaceOverlay] = {}
        self._storage_track_ids: Dict[int, int] = {}
        self._registration_requested_tracks: Set[int] = set()
        self._registration_pending_tracks: Set[int] = set()
        self._registration_codes: Dict[int, str] = {}
        self._registered_names: Dict[int, str] = {}
        self._lost_tracks: Dict[int, _LostTrack] = {}
        self._recovery_labels: Dict[int, str] = {}
        self._pending_storage_by_track: Dict[int, int] = {}
    def run(self) -> None:
        """카메라 표시 루프와 책임별 작업자를 시작하고 종료를 정리한다.

        Returns: None.
        Raises: RuntimeError. 카메라를 열거나 프레임을 읽지 못하면 발생.
        """
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Camera 0 could not be opened.")

        # 화면 루프와 무거운 분석·저장·등록 입력을 분리해 카메라 프레임을 보호한다.
        worker = Thread(target=self._run_analysis_worker, name="analysis-worker", daemon=True)
        storage_worker = Thread(target=self._run_storage_worker, name="storage-worker", daemon=True)
        registration_worker = Thread(
            target=self._run_registration_worker,
            name="registration-worker",
            daemon=True,
        )
        worker.start()
        storage_worker.start()
        registration_worker.start()
        lifecycle_worker = Thread(
            target=self._run_track_lifecycle_worker,
            name="track-lifecycle-worker",
            daemon=True,
        )
        lifecycle_worker.start()
        print("camera_opened=0", flush=True)
        print("analysis_worker_started=true", flush=True)
        print("registration_worker_started=true", flush=True)
        print("press_q_to_stop=true", flush=True)
        try:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("A camera frame could not be read.")
                # 1. 최신 프레임만 분석으로 넘기고, 이전 분석 결과를 즉시 화면에 투영한다.
                self._submit_latest_frame(frame, datetime.now(timezone.utc))
                self._draw_latest_annotations(frame)
                cv2.imshow("Camera Vision - q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            # 2. 새 작업을 중단한 뒤 작업자·카메라·창을 순서대로 정리한다.
            self._stop_requested.set()
            worker.join(timeout=3)
            storage_worker.join(timeout=3)
            lifecycle_worker.join(timeout=3)
            camera.release()
            cv2.destroyAllWindows()
            print("camera_closed=true", flush=True)

    def _submit_latest_frame(self, frame: np.ndarray, occurred_at: datetime) -> None:
        """분석 큐의 오래된 프레임을 최신 프레임으로 교체한다."""
        item = (frame.copy(), occurred_at)
        try:
            self._analysis_queue.put_nowait(item)
        except Full:
            # 영상 지연을 피하기 위해 분석 대기 중인 이전 프레임을 버린다.
            try:
                self._analysis_queue.get_nowait()
            except Empty:
                return
            try:
                self._analysis_queue.put_nowait(item)
            except Full:
                pass

    def _run_analysis_worker(self) -> None:
        """최신 프레임에서 사람 추적과 얼굴 후보·품질·임베딩 생성을 수행한다.

        입력: Queue[Tuple[np.ndarray, datetime]]. 출력: 화면 스냅샷과 표본 저장 요청.
        SQLite 접근과 신원 누적 판단은 수행하지 않는다. Returns: None.
        """
        while not self._stop_requested.is_set():
            try:
                frame, occurred_at = self._analysis_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                # 1. 사람 검출을 Track으로 연결하고 확정·상실 이벤트를 만든다.
                people, events = self._tracker.update(self._detector.detect(frame), occurred_at)
                self._log_events(events)
                for event in events:
                    if event.kind == "TRACK_CONFIRMED":
                        self._storage_queue.put(
                            _TrackStorageRequest(event, self._storage_track_id_for(event.internal_id))
                        )
                    elif event.kind == "TRACK_MISSING":
                        self._clear_active_track_projection(event.internal_id, preserve_recovery_label=True)
                        storage_track_id = self._storage_track_id_for(event.internal_id)
                        self._storage_queue.put(_TrackStorageRequest(event, storage_track_id))
                        self._mark_track_lost(event, storage_track_id)
                    elif event.kind == "TRACK_REAPPEARED":
                        storage_track_id = self._begin_reverification(event.internal_id)
                        self._storage_queue.put(_TrackStorageRequest(event, storage_track_id))
                # 2. 확정 대상별 얼굴 분석과 중복 차단을 수행하고 통과 표본만 저장으로 넘긴다.
                for person in people:
                    overlay = self._analyze_face_if_due(frame, person, occurred_at)
                    if overlay is not None:
                        self._analysis_faces[person.internal_id] = overlay
                snapshot = [(person, self._samples.count(person.internal_id)) for person in people]
                with self._snapshot_lock:
                    self._latest_people = snapshot
                    self._latest_faces = dict(self._analysis_faces)
            except Exception as error:
                _log_operational_failure("analysis_worker", error)

    def _run_storage_worker(self) -> None:
        """Track 전이와 FaceSample 저장·DB 신원 판단을 SQLite 작업자로 수행한다.

        입력: _TrackStorageRequest | _FaceSampleStorageRequest. 출력: DB 이력·신원 결과·등록 제안.
        Returns: None.
        """
        while not self._stop_requested.is_set() or not self._storage_queue.empty():
            try:
                item = self._storage_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                if isinstance(item, _TrackStorageRequest):
                    # 1. 기술 Track 이벤트를 도메인 Track·관찰 세션 전이로 저장한다.
                    if item.event.kind == "TRACK_CONFIRMED":
                        self._observation_service.start(item.storage_track_id, item.event.occurred_at)
                    elif item.event.kind == "TRACK_MISSING":
                        self._observation_service.mark_lost(item.storage_track_id, item.event.occurred_at)
                    elif item.event.kind == "TRACK_REAPPEARED":
                        self._observation_service.start(item.storage_track_id, item.event.occurred_at)
                    print(
                        f"observation_transition=stored event={item.event.kind} track_id={item.event.internal_id}",
                        flush=True,
                    )
                elif isinstance(item, _FaceSampleStorageRequest):
                    # 2. 표본을 저장하고 템플릿 검색과 누적 신원 정책을 적용한다.
                    result = self._face_sample_service.process(item.storage_track_id, item.candidate, item.quality, item.embedding, item.face_crop)
                    if result.status == "FAILED":
                        print("face_sample_error=processing_failed", flush=True)
                    elif result.status == "DISCARDED":
                        print(f"face_sample=discarded_final_identity track_id={item.candidate.track_id}", flush=True)
                    else:
                        self._apply_identity(item.candidate.track_id, result.status, result.person_name)
                        print(f"identity_result=applied track_id={item.candidate.track_id} status={result.status}", flush=True)
                        if result.proposal_id is not None:
                            # 3. 임시 인물 등록 제안은 FIFO 등록 조율기로 넘긴다.
                            self._request_registration(item.candidate.track_id, item.storage_track_id, result.proposal_id)
            except Exception as error:
                _log_operational_failure("storage_worker", error)
            finally:
                if isinstance(item, _FaceSampleStorageRequest):
                    self._complete_storage_work(item.candidate.track_id)

    def _run_track_lifecycle_worker(self) -> None:
        """LOST Track의 저장 작업 완료와 10분 종료 조건을 주기적으로 확인한다."""
        while not self._stop_requested.wait(timeout=0.5):
            now = datetime.now(timezone.utc)
            self._finish_expired_tracks(now)
            self._expire_registration_proposals(now)
        now = datetime.now(timezone.utc)
        self._finish_expired_tracks(now)
        self._expire_registration_proposals(now)

    def _run_registration_worker(self) -> None:
        """FIFO 활성 요청 하나를 채널에 제시하고 proposal_id 기반 응답을 반영한다.

        입력: RegistrationCoordinator의 요청과 RegistrationChannel 응답. 출력: 등록·거절 전이와
        화면 이름 투영. Returns: None.
        """
        while not self._stop_requested.is_set():
            request = self._registration_coordinator.activate_next()
            if request is not None:
                # 1. 현재 활성 요청만 채널에 보낸다. 뒤 요청은 FIFO 대기한다.
                self._registration_channel.present(request)
            response = self._registration_channel.next_response(timeout_seconds=0.2)
            if response is None:
                continue
            request = self._registration_coordinator.active_request()
            if request is None or response.proposal_id != request.proposal_id:
                print("registration_response_discarded reason=unexpected_proposal", flush=True)
                continue
            try:
                # 2. 응답 proposal_id가 활성 요청과 일치할 때만 도메인 등록을 수행한다.
                if response.status == "CANCELLED":
                    self._clear_registration_pending(request.track_id)
                    print(f"registration_cancelled track_id={request.track_id} reason=stdin_closed", flush=True)
                    continue
                if response.status == "REJECTED":
                    outcome = self._registration_service.respond(
                        request.proposal_id,
                        "",
                        datetime.now(timezone.utc),
                    )
                    if outcome.status == "FAILED":
                        print("registration_error=RegistrationFailed", flush=True)
                        continue
                    self._clear_registration_pending(request.track_id)
                    print(f"registration_cancelled track_id={request.track_id}", flush=True)
                    continue
                outcome = self._registration_service.respond(
                    request.proposal_id,
                    response.name or "",
                    datetime.now(timezone.utc),
                )
                if outcome.status == "REJECTED":
                    self._clear_registration_pending(request.track_id)
                    print(f"registration_cancelled track_id={request.track_id}", flush=True)
                    continue
                if outcome.status == "FAILED":
                    print("registration_error=RegistrationFailed", flush=True)
                    continue
                self._apply_identity(request.track_id, "IDENTIFIED", outcome.name)
                self._clear_registration_pending(request.track_id)
                print(
                    f"person_profile=registered track_id={request.track_id} registration_code={request.display_code} "
                    f"name={outcome.name} template_count=3",
                    flush=True,
                )
                print(
                    f"camera_overlay_name_applied track_id={request.track_id} name={outcome.name}",
                    flush=True,
                )
            except EOFError:
                print(f"registration_cancelled track_id={request.track_id} reason=stdin_closed", flush=True)
            except Exception as error:
                _log_operational_failure("registration_channel", error)
            finally:
                self._registration_coordinator.complete(request.proposal_id)
        self._registration_channel.close()

    def _analyze_face_if_due(
        self,
        frame: np.ndarray,
        person: TrackedPerson,
        occurred_at: datetime,
    ) -> _FaceOverlay | None:
        """일정 간격으로 한 추적 대상의 얼굴 후보와 표본 생성 가능 여부를 처리한다."""
        if self._is_identity_finalized(person.internal_id):
            return None
        now = monotonic()
        if now - self._last_face_attempt_at.get(person.internal_id, 0.0) < 0.8:
            return None
        self._last_face_attempt_at[person.internal_id] = now
        candidate = self._face_analyzer.find_largest_face(frame, person, occurred_at)
        if candidate is None:
            print(f"track_id={person.internal_id} face_candidate=not_detected", flush=True)
            return None
        print(
            f"track_id={person.internal_id} face_candidate=extracted "
            f"detection_score={candidate.detection_score:.3f} yaw_proxy={candidate.yaw_proxy:.3f} "
            f"yaw={candidate.yaw_degrees:.1f} pitch={candidate.pitch_degrees:.1f} "
            f"roll={candidate.roll_degrees:.1f} "
            f"bbox={candidate.bbox}",
            flush=True,
        )
        quality = self._quality_evaluator.evaluate(frame, candidate)
        print(
            f"track_id={person.internal_id} face_quality=evaluated accepted={quality.accepted} "
            f"score={quality.score:.3f} reason={quality.reason} size={quality.face_size} "
            f"sharpness={quality.sharpness:.1f} brightness={quality.brightness:.1f} "
            f"occlusion_probability={quality.occlusion_probability:.3f}",
            flush=True,
        )
        overlay = _FaceOverlay(candidate.bbox, quality.reason, quality.accepted)
        if not quality.accepted:
            print(f"track_id={person.internal_id} face_sample=not_created", flush=True)
            return overlay
        try:
            embedding = self._face_embedding.embed(frame, candidate)
        except Exception as error:
            print(
                f"track_id={person.internal_id} face_embedding=failed "
                f"error={type(error).__name__}",
                flush=True,
            )
            return overlay
        print(f"track_id={person.internal_id} face_embedding=created", flush=True)
        saved, similarity, duplicate_reason = self._samples.add_if_non_duplicate(candidate, embedding, quality)
        if not saved:
            print(
                f"track_id={person.internal_id} face_sample=duplicate "
                f"reason={duplicate_reason} similarity={similarity:.4f}",
                flush=True,
            )
            return overlay
        similarity_text = "none" if similarity is None else f"{similarity:.4f}"
        sample_number = self._samples.count(person.internal_id)
        print(
            f"track_id={person.internal_id} face_sample=extracted_success "
            f"sample_number={sample_number} quality={quality.score:.3f} "
            f"max_existing_similarity={similarity_text}",
            flush=True,
        )
        left, top, right, bottom = candidate.bbox
        self._start_storage_work(person.internal_id)
        self._storage_queue.put(
            _FaceSampleStorageRequest(
                candidate=candidate,
                embedding=embedding,
                quality=quality,
                face_crop=frame[top:bottom, left:right].copy(),
                storage_track_id=self._storage_track_id_for(person.internal_id),
                sample_number=sample_number,
            )
        )
        return overlay

    def _request_registration(self, track_id: int, storage_track_id: int, proposal_id: str) -> None:
        """한 미등록 Track에 불변 임시 코드를 붙여 이름 입력을 한 번만 요청한다."""
        display_code = self._registration_code_for(proposal_id)
        with self._snapshot_lock:
            if track_id in self._registration_requested_tracks:
                return
            self._registration_requested_tracks.add(track_id)
            self._registration_pending_tracks.add(track_id)
            self._registration_codes[track_id] = display_code
        self._registration_coordinator.enqueue(
            RegistrationRequest(proposal_id, display_code, track_id, storage_track_id)
        )

    def _clear_registration_pending(self, track_id: int) -> None:
        """등록 질문이 끝난 Track의 화면 대기 상태를 해제한다."""
        with self._snapshot_lock:
            self._registration_pending_tracks.discard(track_id)

    def _apply_identity(self, track_id: int, status: str, person_name: str | None) -> None:
        """식별 완료된 이름만 화면용 투영 값으로 반영한다."""
        with self._snapshot_lock:
            if status in {"IDENTIFIED", "UNREGISTERED"}:
                self._recovery_labels.pop(track_id, None)
            if status == "IDENTIFIED" and person_name is not None:
                self._registered_names[track_id] = person_name

    def _is_identity_finalized(self, track_id: int) -> bool:
        """현재 추적의 신원 판단이 더 이상 표본을 받지 않는 결론인지 확인한다."""
        with self._snapshot_lock:
            storage_track_id = self._storage_track_ids.get(track_id)
        if storage_track_id is None:
            return False
        try:
            return not self._observation_service.accepts_face_samples(storage_track_id)
        except RuntimeError:
            return False

    def _storage_track_id_for(self, track_id: int) -> int:
        """프로세스 재시작 후에도 충돌하지 않는 저장소용 Track 키를 만든다."""
        with self._snapshot_lock:
            storage_track_id = self._storage_track_ids.get(track_id)
            if storage_track_id is None:
                storage_track_id = uuid4().int & ((1 << 63) - 1)
                self._storage_track_ids[track_id] = storage_track_id
        return storage_track_id

    def _begin_reverification(self, track_id: int) -> int:
        """30초 보존 Track의 재등장에 새 관찰 세션과 얼굴 검증을 시작한다."""
        with self._snapshot_lock:
            storage_track_id = uuid4().int & ((1 << 63) - 1)
            self._storage_track_ids[track_id] = storage_track_id
            self._last_face_attempt_at.pop(track_id, None)
            self._registration_pending_tracks.discard(track_id)
            self._registration_requested_tracks.discard(track_id)
        self._samples.clear(track_id)
        return storage_track_id

    def _mark_track_lost(self, event: TrackEvent, storage_track_id: int) -> None:
        """MISSING 관찰 세션을 과거 이력의 10분 종료 대기 상태로 전환한다."""
        with self._snapshot_lock:
            self._lost_tracks[storage_track_id] = _LostTrack(
                storage_track_id,
                event.internal_id,
                event.occurred_at,
            )

    def _clear_active_track_projection(self, track_id: int, *, preserve_recovery_label: bool = False) -> None:
        """잠시 사라진 기술 Track의 화면 투영을 지우고 필요하면 재검증 라벨을 보존한다."""
        with self._snapshot_lock:
            self._analysis_faces.pop(track_id, None)
            self._latest_faces.pop(track_id, None)
            self._latest_people = [item for item in self._latest_people if item[0].internal_id != track_id]
            name = self._registered_names.pop(track_id, None)
            code = self._registration_codes.get(track_id)
            if preserve_recovery_label:
                if name is not None:
                    self._recovery_labels[track_id] = name
                elif code is not None:
                    self._recovery_labels[track_id] = f"임시 인물 | {code}"
            self._registration_pending_tracks.discard(track_id)

    def _start_storage_work(self, track_id: int) -> None:
        """종료 전에 마쳐야 하는 FaceSample 저장 작업 수를 증가시킨다."""
        with self._snapshot_lock:
            self._pending_storage_by_track[track_id] = self._pending_storage_by_track.get(track_id, 0) + 1

    def _complete_storage_work(self, track_id: int) -> None:
        """FaceSample 저장 작업 완료를 기록하고 종료 대기 조건을 갱신한다."""
        with self._snapshot_lock:
            pending = self._pending_storage_by_track.get(track_id, 0)
            if pending <= 1:
                self._pending_storage_by_track.pop(track_id, None)
            else:
                self._pending_storage_by_track[track_id] = pending - 1

    def _finish_expired_tracks(self, now: datetime) -> None:
        """10분 동안 LOST이고 저장 작업이 끝난 Track을 종료·정리한다."""
        with self._snapshot_lock:
            finishable = [
                (storage_track_id, lost_track)
                for storage_track_id, lost_track in self._lost_tracks.items()
                if (now - lost_track.lost_at).total_seconds() >= self._track_end_delay_seconds
                and self._pending_storage_by_track.get(track_id, 0) == 0
            ]
        for storage_track_id, lost_track in finishable:
            try:
                ended = self._observation_service.end_if_possible(lost_track.storage_track_id, now)
            except Exception as error:
                _log_operational_failure("track_end", error)
                continue
            if not ended:
                continue
            with self._snapshot_lock:
                self._lost_tracks.pop(storage_track_id, None)
                track_id = lost_track.technical_track_id
                if self._storage_track_ids.get(track_id) == storage_track_id:
                    self._last_face_attempt_at.pop(track_id, None)
                    self._registration_pending_tracks.discard(track_id)
                    self._registration_requested_tracks.discard(track_id)
                    self._registration_codes.pop(track_id, None)
                    self._registered_names.pop(track_id, None)
                    self._recovery_labels.pop(track_id, None)
                    self._storage_track_ids.pop(track_id, None)
                    self._samples.clear(track_id)
            print(f"event=TRACK_ENDED track_id={lost_track.technical_track_id} at={now.isoformat()}", flush=True)

    def _expire_registration_proposals(self, now: datetime) -> None:
        """응답 기한을 넘긴 등록 제안을 만료 상태로 전환한다."""
        expired_proposal_ids = self._registration_service.expire_pending(now)
        for proposal_id in expired_proposal_ids:
            self._registration_coordinator.discard(proposal_id)
        if expired_proposal_ids:
            print(f"registration_proposals_expired={len(expired_proposal_ids)}", flush=True)
        purged_count = self._registration_service.purge_expired_unregistered_data(now)
        if purged_count:
            print(f"unregistered_data_purged={purged_count}", flush=True)

    @staticmethod
    def _log_events(events: List[TrackEvent]) -> None:
        """추적 확정·상실 이벤트를 운영 확인용 터미널 로그로 남긴다."""
        for event in events:
            print(
                f"event={event.kind} track_id={event.internal_id} at={event.occurred_at.isoformat()}",
                flush=True,
            )

    def _draw_latest_annotations(self, frame: np.ndarray) -> None:
        """분석 작업자가 만든 최신 사람·얼굴 결과를 현재 카메라 프레임에 표시한다."""
        with self._snapshot_lock:
            people = list(self._latest_people)
            faces = dict(self._latest_faces)
            registered_names = dict(self._registered_names)
        for person, sample_count in people:
            left, top, right, bottom = person.bbox
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 200, 0), 2)
            _draw_display_label(
                frame,
                (
                    registered_names[person.internal_id]
                    if person.internal_id in registered_names
                    else self._person_overlay_label(person.internal_id, sample_count)
                ),
                (left, max(24, top - 10)),
                (0, 200, 0),
                font_scale=0.6,
            )
            overlay = faces.get(person.internal_id)
            if overlay is not None:
                self._draw_face(frame, overlay)

    def _person_overlay_label(self, track_id: int, sample_count: int) -> str:
        """관찰 단계·임시 인물 코드·식별 결과를 화면용 문구로 바꾼다."""
        with self._snapshot_lock:
            storage_track_id = self._storage_track_ids.get(track_id)
            registration_pending = track_id in self._registration_pending_tracks
            registration_code = self._registration_codes.get(track_id)
            recovery_label = self._recovery_labels.get(track_id)
        observation_code = self._observation_code_for(track_id, storage_track_id)
        if recovery_label is not None:
            return f"{recovery_label}?"
        if registration_pending:
            return f"임시 인물 | {registration_code}"
        if storage_track_id is not None:
            try:
                if self._observation_service.current_status(storage_track_id) is CurrentIdentityStatus.UNREGISTERED:
                    return f"임시 인물 | {registration_code or observation_code}"
            except RuntimeError:
                pass
            return f"표본 수집 중 | {observation_code}"
        return "추적 확인 중"

    @staticmethod
    def _observation_code_for(track_id: int, storage_track_id: int | None) -> str:
        """현재 관찰 세션을 화면에서만 구별할 짧은 임시 코드를 만든다."""
        if storage_track_id is None:
            return f"T-{track_id:04X}"
        return f"T-{storage_track_id & 0xFFFFFFFF:08X}"

    @staticmethod
    def _registration_code_for(proposal_id: str) -> str:
        """영속 RegistrationProposal ID에서 재사용되지 않는 화면·터미널용 코드를 만든다."""
        return f"U-{proposal_id.replace('-', '')[:8].upper()}"

    @staticmethod
    def _draw_face(frame: np.ndarray, overlay: _FaceOverlay) -> None:
        """얼굴 품질 결과에 맞는 색과 라벨로 얼굴 영역을 표시한다."""
        left, top, right, bottom = overlay.bbox
        color = (0, 200, 0) if overlay.accepted else (0, 0, 220)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        _draw_display_label(
            frame,
            overlay.label,
            (left, min(frame.shape[0] - 8, bottom + 20)),
            color,
            font_scale=0.5,
        )


def _resource_root() -> Path:
    """소스 실행과 PyInstaller 실행에서 공통으로 번들 리소스 위치를 찾는다."""
    import sys

    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))


def _insightface_model_root(resource_root: Path) -> Path:
    """번들 모델을 우선 사용하고 소스 개발 환경의 기존 모델 경로를 보조로 사용한다."""
    bundled_root = resource_root / "models" / "insightface"
    if (bundled_root / "models" / "buffalo_l").is_dir():
        return bundled_root
    return Path.home() / ".insightface"


def _application_data_root() -> Path:
    """OS별 사용자 전용 경로에 SQLite와 private 얼굴 crop을 저장한다."""
    import os
    import sys

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "local_face_recognition"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "local_face_recognition"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "local_face_recognition"
