from unittest import TestCase
from unittest.mock import patch

import numpy as np

from components.person_detector import PersonDetection
from presentation.opencv_window import OpenCVWindow


class OpenCVWindowTest(TestCase):
    @patch("presentation.opencv_window.cv2.waitKey", return_value=27)
    @patch("presentation.opencv_window.cv2.imshow")
    def test_escape_requests_shutdown(self, imshow: object, wait_key: object) -> None:
        window = OpenCVWindow()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = PersonDetection(10, 20, 50, 80, 0.9)

        self.assertFalse(window.show(frame, [detection]))

    @patch("presentation.opencv_window.cv2.destroyAllWindows")
    def test_close_destroys_windows(self, destroy_all_windows: object) -> None:
        OpenCVWindow().close()

        destroy_all_windows.assert_called_once_with()
