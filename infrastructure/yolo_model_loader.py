"""Ultralytics YOLO 모델 파일을 애플리케이션 객체로 불러온다."""

from __future__ import annotations

from typing import Any


DEFAULT_MODEL_NAME = "yolo26n.pt"


class YoloModelLoadError(RuntimeError):
    """YOLO 모델을 불러오지 못했을 때 발생한다."""


def load_yolo_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """사전 학습된 YOLO 모델을 불러온다.

    모델 파일이 없으면 Ultralytics가 공식 가중치를 최초 한 번 다운로드한다.
    외부 라이브러리 import와 모델 생성 실패는 프로젝트 오류로 변환한다.
    """
    try:
        from ultralytics import YOLO

        return YOLO(model_name)
    except Exception as error:
        raise YoloModelLoadError(
            f"YOLO 모델을 불러올 수 없습니다. 모델: {model_name}"
        ) from error
