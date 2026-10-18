
"""애플리케이션에 필요한 객체를 생성·조립하고 오케스트레이션을 호출한다."""

from __future__ import annotations

import sys

import cv2

from components.person_detector import YoloPersonDetector
from infrastructure.yolo_model_loader import load_yolo_model
from orchestration.application import run as run_orchestration
from presentation.opencv_window import OpenCVWindow
from services.person_detection_service import PersonDetectionService


CAMERA_INDEX = 0
# 검출 주기는 실측 후 확정한다. None은 가능한 최대 속도로 검출한다.
PERSON_DETECTION_FPS = None


class CameraInitializationError(RuntimeError):
    """카메라 객체를 정상적으로 초기화하지 못했을 때 발생한다."""


class ApplicationExecutionError(RuntimeError):
    """오케스트레이션 실행에 실패했을 때 발생한다."""


def _create_camera() -> cv2.VideoCapture:
    """설정된 카메라를 열고 사용할 수 있는 상태인지 검증한다."""
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise CameraInitializationError(
            f"카메라를 열 수 없습니다. 카메라 번호: {CAMERA_INDEX}"
        )
    return camera


def main() -> None:
    """외부 객체와 책임 객체를 조립한 뒤 실행 흐름을 오케스트레이션에 위임한다."""
    # YOLO 모델은 외부 자원이므로 infrastructure의 로더를 통해 생성한다.
    model = load_yolo_model()

    # 검출기는 모델을 사용해 한 프레임에서 사람의 위치를 찾는다.
    detector = YoloPersonDetector(model)

    # 서비스는 YOLO를 언제 호출할지 결정하고 최근 검출 결과를 관리한다.
    detection_service = PersonDetectionService(
        detector,
        target_fps=PERSON_DETECTION_FPS,
    )

    # 카메라와 화면은 애플리케이션 실행에 필요한 외부 입출력 객체다.
    camera = _create_camera()
    window = OpenCVWindow()
    try:
        print(f"camera_ready={CAMERA_INDEX}", flush=True)

        # 실행 순서의 제어는 main이 직접 수행하지 않고 오케스트레이션에 맡긴다.
        run_orchestration(camera, detection_service, window)
    except Exception as error:
        raise ApplicationExecutionError("애플리케이션 실행에 실패했습니다.") from error
    finally:
        # 실행 성공 여부와 관계없이 화면과 카메라 자원을 반드시 반환한다.
        window.close()
        camera.release()


if __name__ == "__main__":
    try:
        main()
    except CameraInitializationError as error:
        print(f"camera_initialization_error={error}", file=sys.stderr)
        raise SystemExit(1)
    except ApplicationExecutionError as error:
        print(f"application_execution_error={error}", file=sys.stderr)
        raise SystemExit(1)
