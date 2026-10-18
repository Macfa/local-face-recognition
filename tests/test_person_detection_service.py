from unittest import TestCase
from unittest.mock import Mock

from services.person_detection_service import PersonDetectionService


class PersonDetectionServiceTest(TestCase):
    def test_no_target_fps_runs_detection_for_every_frame(self) -> None:
        detector = Mock()
        detector.detect.side_effect = [["first"], ["second"]]
        service = PersonDetectionService(detector, target_fps=None)
        first_frame = object()
        second_frame = object()

        self.assertEqual(service.process(first_frame), ["first"])
        self.assertEqual(service.process(second_frame), ["second"])

        self.assertEqual(detector.detect.call_count, 2)

    def test_target_fps_reuses_latest_result_until_next_interval(self) -> None:
        detector = Mock()
        detector.detect.side_effect = [["first"], ["second"]]
        clock = Mock(side_effect=[0.0, 0.1, 0.2])
        service = PersonDetectionService(detector, target_fps=5, clock=clock)

        self.assertEqual(service.process(object()), ["first"])
        self.assertEqual(service.process(object()), ["first"])
        self.assertEqual(service.process(object()), ["second"])

        self.assertEqual(detector.detect.call_count, 2)
