import cv2
from cv2.typing import MatLike


WINDOW_NAME = "Local Face Recognition"
ESC_KEY = 27


class OpenCVWindow:
    """카메라 프레임을 표시하고 Esc 종료 입력을 처리한다."""

    def show(self, frame: MatLike) -> bool:
        cv2.imshow(WINDOW_NAME, frame)
        pressed_key = cv2.waitKey(1) & 0xFF
        return pressed_key != ESC_KEY

    def close(self) -> None:
        cv2.destroyAllWindows()
