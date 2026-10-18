from unittest import TestCase
from unittest.mock import patch

from presentation.opencv_window import OpenCVWindow


class OpenCVWindowTest(TestCase):
    @patch("presentation.opencv_window.cv2.waitKey", return_value=27)
    @patch("presentation.opencv_window.cv2.imshow")
    def test_escape_requests_shutdown(self, imshow: object, wait_key: object) -> None:
        window = OpenCVWindow()

        self.assertFalse(window.show(object()))

    @patch("presentation.opencv_window.cv2.destroyAllWindows")
    def test_close_destroys_windows(self, destroy_all_windows: object) -> None:
        OpenCVWindow().close()

        destroy_all_windows.assert_called_once_with()
