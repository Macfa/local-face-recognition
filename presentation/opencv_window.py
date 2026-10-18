"""카메라 프레임과 사람 검출 결과를 OpenCV 창으로 표현한다."""

import cv2
from cv2.typing import MatLike

from components.person_detector import PersonDetection


WINDOW_NAME = "Local Face Recognition"
ESC_KEY = 27
BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 255, 0)


class OpenCVWindow:
    """검출 박스를 그린 프레임을 표시하고 Esc 종료 입력을 처리한다."""

    def show(self, frame: MatLike, detections: list[PersonDetection]) -> bool:
        """원본 프레임을 보존한 채 검출 결과를 그리고 계속 실행할지 반환한다."""
        # 후속 처리에서 원본을 사용할 수 있으므로 표시용 복사본에만 그림을 그린다.
        displayed_frame = frame.copy()
        for detection in detections:
            start = (detection.left, detection.top)
            end = (detection.right, detection.bottom)
            cv2.rectangle(displayed_frame, start, end, BOX_COLOR, 2)
            cv2.putText(
                displayed_frame,
                f"person {detection.confidence:.2f}",
                (detection.left, max(detection.top - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                TEXT_COLOR,
                2,
            )

        cv2.imshow(WINDOW_NAME, displayed_frame)
        pressed_key = cv2.waitKey(1) & 0xFF
        return pressed_key != ESC_KEY

    def close(self) -> None:
        """OpenCV가 생성한 모든 화면 자원을 정리한다."""
        cv2.destroyAllWindows()
