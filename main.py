
from __future__ import annotations

import sys

import cv2

from orchestration.application import run as run_orchestration
from presentation.opencv_window import OpenCVWindow


CAMERA_INDEX = 0


class CameraInitializationError(RuntimeError):
    """카메라 객체를 정상적으로 초기화하지 못했을 때 발생한다."""


class ApplicationExecutionError(RuntimeError):
    """오케스트레이션 실행에 실패했을 때 발생한다."""


def _create_camera() -> cv2.VideoCapture:
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise CameraInitializationError(
            f"카메라를 열 수 없습니다. 카메라 번호: {CAMERA_INDEX}"
        )
    return camera


def main() -> None:
    camera = _create_camera()
    window = OpenCVWindow()
    try:
        print(f"camera_ready={CAMERA_INDEX}", flush=True)
        run_orchestration(camera, window)
    except Exception as error:
        raise ApplicationExecutionError("애플리케이션 실행에 실패했습니다.") from error
    finally:
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
