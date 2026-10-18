from unittest import TestCase
from unittest.mock import Mock, patch

import main


class MainTest(TestCase):
    @patch("main.run_orchestration")
    @patch("main.OpenCVWindow")
    @patch("main._create_camera")
    @patch("main.PersonDetectionService")
    @patch("main.YoloPersonDetector")
    @patch("main.load_yolo_model")
    def test_main_calls_orchestration_and_releases_camera(
        self,
        load_yolo_model: Mock,
        detector_type: Mock,
        detection_service_type: Mock,
        create_camera: Mock,
        window_type: Mock,
        run_orchestration: Mock,
    ) -> None:
        camera = Mock()
        model = Mock()
        detector = Mock()
        detection_service = Mock()
        window = Mock()
        load_yolo_model.return_value = model
        detector_type.return_value = detector
        detection_service_type.return_value = detection_service
        create_camera.return_value = camera
        window_type.return_value = window

        main.main()

        detector_type.assert_called_once_with(model)
        detection_service_type.assert_called_once_with(detector, target_fps=None)
        run_orchestration.assert_called_once_with(camera, detection_service, window)
        window.close.assert_called_once_with()
        camera.release.assert_called_once_with()

    @patch("main.run_orchestration", side_effect=RuntimeError("failure"))
    @patch("main.OpenCVWindow")
    @patch("main._create_camera")
    @patch("main.PersonDetectionService")
    @patch("main.YoloPersonDetector")
    @patch("main.load_yolo_model")
    def test_main_releases_camera_when_orchestration_fails(
        self,
        load_yolo_model: Mock,
        detector_type: Mock,
        detection_service_type: Mock,
        create_camera: Mock,
        window_type: Mock,
        run_orchestration: Mock,
    ) -> None:
        camera = Mock()
        detector = Mock()
        detection_service = Mock()
        window = Mock()
        detector_type.return_value = detector
        detection_service_type.return_value = detection_service
        create_camera.return_value = camera
        window_type.return_value = window

        with self.assertRaises(main.ApplicationExecutionError):
            main.main()

        run_orchestration.assert_called_once_with(camera, detection_service, window)
        window.close.assert_called_once_with()
        camera.release.assert_called_once_with()
