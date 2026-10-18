"""카메라 입력, 사람 검출 서비스, 화면 출력의 실행 순서를 조정한다."""

import cv2

from presentation.opencv_window import OpenCVWindow
from services.person_detection_service import PersonDetectionService


class FrameReadError(RuntimeError):
    """카메라 프레임을 읽지 못했을 때 발생한다."""


def run(
    camera: cv2.VideoCapture,
    detection_service: PersonDetectionService,
    window: OpenCVWindow,
) -> None:
    """프레임 처리 단계를 반복하고 사용자의 종료 요청이 오면 반환한다."""
    while True:
        # 1단계: 카메라에서 현재 프레임을 가져온다.
        success, frame = camera.read()
        if not success:
            raise FrameReadError("카메라 프레임을 읽을 수 없습니다.")

        # 2단계: 검출 주기 판단과 사람 검출은 서비스에 위임한다.
        detections = detection_service.process(frame)

        # 3단계: 프레임과 검출 결과를 표시하고 Esc 입력 시 반복을 끝낸다.
        if not window.show(frame, detections):
            return

