from unittest import TestCase
from unittest.mock import Mock, patch

import main


class MainTest(TestCase):
    @patch("main.run_orchestration")
    @patch("main.OpenCVWindow")
    @patch("main._create_camera")
    def test_main_calls_orchestration_and_releases_camera(
        self,
        create_camera: Mock,
        window_type: Mock,
        run_orchestration: Mock,
    ) -> None:
        camera = Mock()
        window = Mock()
        create_camera.return_value = camera
        window_type.return_value = window

        main.main()

        run_orchestration.assert_called_once_with(camera, window)
        window.close.assert_called_once_with()
        camera.release.assert_called_once_with()

    @patch("main.run_orchestration", side_effect=RuntimeError("failure"))
    @patch("main.OpenCVWindow")
    @patch("main._create_camera")
    def test_main_releases_camera_when_orchestration_fails(
        self,
        create_camera: Mock,
        window_type: Mock,
        run_orchestration: Mock,
    ) -> None:
        camera = Mock()
        window = Mock()
        create_camera.return_value = camera
        window_type.return_value = window

        with self.assertRaises(main.ApplicationExecutionError):
            main.main()

        run_orchestration.assert_called_once_with(camera, window)
        window.close.assert_called_once_with()
        camera.release.assert_called_once_with()
