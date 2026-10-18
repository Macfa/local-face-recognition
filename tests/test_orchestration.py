from unittest import TestCase
from unittest.mock import Mock

from orchestration.application import FrameReadError, run


class OrchestrationTest(TestCase):
    def test_escape_stops_camera_loop(self) -> None:
        frame = object()
        camera = Mock()
        window = Mock()
        camera.read.return_value = (True, frame)
        window.show.return_value = False

        run(camera, window)

        camera.read.assert_called_once_with()
        window.show.assert_called_once_with(frame)

    def test_frame_read_failure_raises_error(self) -> None:
        camera = Mock()
        window = Mock()
        camera.read.return_value = (False, None)

        with self.assertRaises(FrameReadError):
            run(camera, window)

        window.show.assert_not_called()
