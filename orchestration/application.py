import cv2

from presentation.opencv_window import OpenCVWindow


class FrameReadError(RuntimeError):
    """카메라 프레임을 읽지 못했을 때 발생한다."""


def run(camera: cv2.VideoCapture, window: OpenCVWindow) -> None:
    """카메라 프레임을 화면에 표시하고 Esc 입력 시 종료한다."""
    while True:
        success, frame = camera.read()
        if not success:
            raise FrameReadError("카메라 프레임을 읽을 수 없습니다.")
        if not window.show(frame):
            return

