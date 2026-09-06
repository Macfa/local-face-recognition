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
from .application_services.registration_service import RegistrationService
from .application_services.face_sample_service import FaceSampleService
from .infrastructure.local_sqlite_repository import LocalSQLiteRepository


_DISPLAY_FONT_PATHS = (
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
)


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
class _RegistrationRequest:
    """세 표본 저장을 마친 추적에 대해 이름 입력을 요청한다."""

    track_id: int
    storage_track_id: int
    proposal_id: str


@dataclass(frozen=True)
class _LostTrack:
    """화면에서 사라진 뒤 종료 대기 중인 추적의 시각과 저장 키다."""

    storage_track_id: int
    lost_at: datetime


class VisionApplication:
    """카메라 표시와 DB 비의존 분석 파이프라인을 조율한다."""

    def __init__(self) -> None:
        """로컬 모델 컴포넌트와 최신 프레임 분석 작업자를 준비한다."""
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
        self._registration_queue: Queue[_RegistrationRequest] = Queue()
        self._stop_requested = Event()
        self._snapshot_lock = Lock()
        self._track_end_delay_seconds = 10 * 60
        self._latest_people: List[Tuple[TrackedPerson, int]] = []
        self._latest_faces: Dict[int, _FaceOverlay] = {}
        self._analysis_faces: Dict[int, _FaceOverlay] = {}
        self._storage_track_ids: Dict[int, int] = {}
        self._registration_requested_tracks: Set[int] = set()
        self._registration_pending_tracks: Set[int] = set()
        self._registered_names: Dict[int, str] = {}
        self._lost_tracks: Dict[int, _LostTrack] = {}
        self._pending_storage_by_track: Dict[int, int] = {}
    def run(self) -> None:
        """카메라 표시 루프와 별도 분석 작업자를 시작하고 종료를 정리한다."""
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("Camera 0 could not be opened.")

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
                self._submit_latest_frame(frame, datetime.now(timezone.utc))
                self._draw_latest_annotations(frame)
                cv2.imshow("Camera Vision - q to stop", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
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
        """사람 추적부터 얼굴 표본 판정까지를 화면 루프와 분리해 실행한다."""
        while not self._stop_requested.is_set():
            try:
                frame, occurred_at = self._analysis_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                people, events = self._tracker.update(self._detector.detect(frame), occurred_at)
                self._log_events(events)
                for event in events:
                    self._storage_queue.put(
                        _TrackStorageRequest(event, self._storage_track_id_for(event.internal_id))
                    )
                    if event.kind == "TRACK_LOST":
                        self._analysis_faces.pop(event.internal_id, None)
                        self._mark_track_lost(event)
                for person in people:
                    overlay = self._analyze_face_if_due(frame, person, occurred_at)
                    if overlay is not None:
                        self._analysis_faces[person.internal_id] = overlay
                snapshot = [(person, self._samples.count(person.internal_id)) for person in people]
                with self._snapshot_lock:
                    self._latest_people = snapshot
                    self._latest_faces = dict(self._analysis_faces)
            except Exception as error:
                print(f"analysis_worker_error={type(error).__name__} message={error}", flush=True)

    def _run_storage_worker(self) -> None:
        """로컬 저장소와 private crop 파일 저장을 화면·분석 작업과 분리해 처리한다."""
        while not self._stop_requested.is_set() or not self._storage_queue.empty():
            try:
                item = self._storage_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                if isinstance(item, _TrackStorageRequest):
                    if item.event.kind == "TRACK_CONFIRMED":
                        self._observation_service.start(item.storage_track_id, item.event.occurred_at)
                    elif item.event.kind == "TRACK_LOST":
                        self._observation_service.mark_lost(item.storage_track_id, item.event.occurred_at)
                    print(
                        f"observation_transition=stored event={item.event.kind} track_id={item.event.internal_id}",
                        flush=True,
                    )
                elif isinstance(item, _FaceSampleStorageRequest):
                    result = self._face_sample_service.process(item.storage_track_id, item.candidate, item.quality, item.embedding, item.face_crop)
                    if result.status == "FAILED":
                        print(f"face_sample_error={result.error}", flush=True)
                    elif result.status == "DISCARDED":
                        print(f"face_sample=discarded_final_identity track_id={item.candidate.track_id}", flush=True)
                    else:
                        self._apply_identity(item.candidate.track_id, result.status, result.person_name)
                        print(f"identity_result=applied track_id={item.candidate.track_id} status={result.status}", flush=True)
                        if result.proposal_id is not None:
                            self._request_registration(item.candidate.track_id, item.storage_track_id, result.proposal_id)
            except Exception as error:
                print(f"storage_error={type(error).__name__} message={error}", flush=True)
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
        """외부인 등록 제안에 대해 Terminal 확인·이름 입력과 프로필 등록을 처리한다."""
        while not self._stop_requested.is_set() or not self._registration_queue.empty():
            try:
                request = self._registration_queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                print(
                    f"external_identity_confirmed track_id={request.track_id} "
                    "registration_name_required=true",
                    flush=True,
                )
                answer = self._read_registration_answer(request.track_id)
                if answer is None:
                    self._clear_registration_pending(request.track_id)
                    print(f"registration_cancelled track_id={request.track_id} reason=stdin_closed", flush=True)
                    continue
                if answer == "N":
                    outcome = self._registration_service.respond(
                        request.proposal_id,
                        "",
                        datetime.now(timezone.utc),
                    )
                    if outcome.status == "FAILED":
                        print(f"registration_error=RegistrationFailed message={outcome.error}", flush=True)
                        continue
                    self._clear_registration_pending(request.track_id)
                    print(f"registration_cancelled track_id={request.track_id}", flush=True)
                    continue
                name = self._read_registration_name(request.track_id)
                if name is None:
                    self._clear_registration_pending(request.track_id)
                    print(f"registration_cancelled track_id={request.track_id} reason=stdin_closed", flush=True)
                    continue
                outcome = self._registration_service.respond(
                    request.proposal_id,
                    name,
                    datetime.now(timezone.utc),
                )
                if outcome.status == "REJECTED":
                    self._clear_registration_pending(request.track_id)
                    print(f"registration_cancelled track_id={request.track_id}", flush=True)
                    continue
                if outcome.status == "FAILED":
                    print(f"registration_error=RegistrationFailed message={outcome.error}", flush=True)
                    continue
                self._apply_identity(request.track_id, "IDENTIFIED", outcome.name)
                self._clear_registration_pending(request.track_id)
                print(
                    f"person_profile=registered track_id={request.track_id} "
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
                print(f"registration_channel_error={type(error).__name__} message={error}", flush=True)

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
        """한 Track에서 이름 입력 요청을 한 번만 대기열에 넣는다."""
        with self._snapshot_lock:
            if track_id in self._registration_requested_tracks:
                return
            self._registration_requested_tracks.add(track_id)
            self._registration_pending_tracks.add(track_id)
        self._registration_queue.put(_RegistrationRequest(track_id, storage_track_id, proposal_id))

    def _read_registration_answer(self, track_id: int) -> str | None:
        """대소문자와 무관하게 유효한 등록 여부 응답이 올 때까지 다시 묻는다."""
        while not self._stop_requested.is_set():
            try:
                answer = input(
                    f"외부인으로 확인되었습니다. Track {track_id}의 이름을 등록하시겠습니까? (Y/N): "
                ).strip().casefold()
            except EOFError:
                return None
            if answer in {"y", "n"}:
                return answer.upper()
            print("등록 입력이 올바르지 않습니다. Y 또는 N을 입력하세요.", flush=True)
        return None

    def _read_registration_name(self, track_id: int) -> str | None:
        """공백이 아닌 이름이 입력될 때까지 등록 대상의 이름을 다시 묻는다."""
        while not self._stop_requested.is_set():
            try:
                name = input(f"Track {track_id}의 이름을 입력하세요: ").strip()
            except EOFError:
                return None
            if name:
                return name
            print("이름을 비워둘 수 없습니다. 다시 입력하세요.", flush=True)
        return None

    def _clear_registration_pending(self, track_id: int) -> None:
        """등록 질문이 끝난 Track의 화면 대기 상태를 해제한다."""
        with self._snapshot_lock:
            self._registration_pending_tracks.discard(track_id)

    def _apply_identity(self, track_id: int, status: str, person_name: str | None) -> None:
        """식별 완료된 이름만 화면용 투영 값으로 반영한다."""
        with self._snapshot_lock:
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

    def _mark_track_lost(self, event: TrackEvent) -> None:
        """TRACK_LOST를 화면 제거와 10분 종료 대기 상태로 전환한다."""
        with self._snapshot_lock:
            self._lost_tracks[event.internal_id] = _LostTrack(
                self._storage_track_ids[event.internal_id],
                event.occurred_at,
            )

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
                (track_id, lost_track)
                for track_id, lost_track in self._lost_tracks.items()
                if (now - lost_track.lost_at).total_seconds() >= self._track_end_delay_seconds
                and self._pending_storage_by_track.get(track_id, 0) == 0
            ]
        for track_id, lost_track in finishable:
            try:
                ended = self._observation_service.end_if_possible(lost_track.storage_track_id, now)
            except Exception as error:
                print(f"track_end_error={type(error).__name__} message={error}", flush=True)
                continue
            if not ended:
                continue
            with self._snapshot_lock:
                self._lost_tracks.pop(track_id, None)
                self._last_face_attempt_at.pop(track_id, None)
                self._registration_pending_tracks.discard(track_id)
                self._registered_names.pop(track_id, None)
            self._samples.clear(track_id)
            print(f"event=TRACK_ENDED track_id={track_id} at={now.isoformat()}", flush=True)

    def _expire_registration_proposals(self, now: datetime) -> None:
        """응답 기한을 넘긴 등록 제안을 만료 상태로 전환한다."""
        expired_count = self._registration_service.expire_pending(now)
        if expired_count:
            print(f"registration_proposals_expired={expired_count}", flush=True)

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
                    f"{registered_names[person.internal_id]} | samples {sample_count}"
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
        """도메인 세션의 현재 신원 결과를 화면용 짧은 문구로 바꾼다."""
        with self._snapshot_lock:
            storage_track_id = self._storage_track_ids.get(track_id)
            registration_pending = track_id in self._registration_pending_tracks
        if registration_pending:
            return f"등록대기 | samples {sample_count}"
        if storage_track_id is not None:
            try:
                if self._observation_service.current_status(storage_track_id) is CurrentIdentityStatus.EXTERNAL:
                    return f"External | samples {sample_count}"
            except RuntimeError:
                pass
        return f"Track {track_id} | samples {sample_count}"

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
