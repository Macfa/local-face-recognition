from unittest import TestCase
from unittest.mock import Mock

from orchestration.application import FrameReadError, run


class OrchestrationTest(TestCase):
    def test_escape_stops_camera_loop(self) -> None:
        frame = object()
        camera = Mock()
        detection_service = Mock()
        window = Mock()
        camera.read.return_value = (True, frame)
        detections = [object()]
        detection_service.process.return_value = detections
        window.show.return_value = False

        run(camera, detection_service, window)

        camera.read.assert_called_once_with()
        detection_service.process.assert_called_once_with(frame)
        window.show.assert_called_once_with(frame, detections)

    def test_frame_read_failure_raises_error(self) -> None:
        camera = Mock()
        detection_service = Mock()
        window = Mock()
        camera.read.return_value = (False, None)

        with self.assertRaises(FrameReadError):
            run(camera, detection_service, window)

        detection_service.process.assert_not_called()
        window.show.assert_not_called()
