from unittest import TestCase
from unittest.mock import Mock

from components.person_detector import PersonDetection, YoloPersonDetector


class YoloPersonDetectorTest(TestCase):
    def test_detect_returns_person_boxes(self) -> None:
        boxes = Mock()
        boxes.xyxy.cpu.return_value.tolist.return_value = [[10.2, 20.4, 50.6, 80.8]]
        boxes.conf.cpu.return_value.tolist.return_value = [0.91]
        result = Mock()
        result.boxes = boxes
        model = Mock()
        model.predict.return_value = [result]
        frame = object()

        detections = YoloPersonDetector(model).detect(frame)

        self.assertEqual(detections, [PersonDetection(10, 20, 51, 81, 0.91)])
        model.predict.assert_called_once_with(
            source=frame,
            classes=[0],
            conf=0.5,
            verbose=False,
        )

    def test_detect_returns_empty_list_when_no_person_was_found(self) -> None:
        result = Mock()
        result.boxes = None
        model = Mock()
        model.predict.return_value = [result]

        self.assertEqual(YoloPersonDetector(model).detect(object()), [])
