"""local_face_recognition의 실행 환경을 준비하고 애플리케이션을 시작한다."""
from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIRECTORY = PROJECT_ROOT / ".venv"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
OCCLUSION_MODEL_URL = "https://huggingface.co/Jacky622/face_occlusion/resolve/main/face_occlusion.onnx?download=true"
INSIGHTFACE_MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
INSIGHTFACE_MODEL_DIRECTORY = MODELS_DIRECTORY / "insightface" / "models" / "buffalo_l"
REQUIRED_INSIGHTFACE_FILES = (
    "det_10g.onnx",
    "1k3d68.onnx",
    "w600k_r50.onnx",
)


def main() -> None:
    """실행 환경과 로컬 모델을 준비한 후 대표 애플리케이션을 실행한다."""
    python = _prepare_runtime_environment()
    _prepare_models()
    _run([str(python), str(PROJECT_ROOT / "local_face_recognition.py")])


def _prepare_runtime_environment() -> Path:
    """프로젝트 전용 가상환경을 만들고 필요한 Python 패키지를 설치한다."""
    if not VENV_DIRECTORY.exists():
        print("virtual_environment_create_started=true")
        _run([sys.executable, "-m", "venv", str(VENV_DIRECTORY)])
    python = VENV_DIRECTORY / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])
    return python


def _prepare_models() -> None:
    """각 모델의 공개 원본에서 누락된 로컬 가중치만 자동으로 준비한다."""
    _download_file(MODELS_DIRECTORY / "yolo11n.pt", YOLO_MODEL_URL)
    _download_file(MODELS_DIRECTORY / "face_occlusion.onnx", OCCLUSION_MODEL_URL)
    if all((INSIGHTFACE_MODEL_DIRECTORY / name).is_file() for name in REQUIRED_INSIGHTFACE_FILES):
        return

    archive_path = PROJECT_ROOT / ".insightface-buffalo-l.zip"
    print("model_download_started=insightface_buffalo_l")
    try:
        urllib.request.urlretrieve(INSIGHTFACE_MODEL_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            root = INSIGHTFACE_MODEL_DIRECTORY.resolve()
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if root not in destination.parents and destination != root:
                    raise RuntimeError("Invalid InsightFace model archive path.")
            archive.extractall(root)
    finally:
        archive_path.unlink(missing_ok=True)

    missing = [name for name in REQUIRED_INSIGHTFACE_FILES if not (INSIGHTFACE_MODEL_DIRECTORY / name).is_file()]
    if missing:
        raise RuntimeError(f"InsightFace model archive is incomplete: {', '.join(missing)}")


def _download_file(destination: Path, source_url: str) -> None:
    """한 모델 파일이 없을 때만 원본 배포처에서 해당 파일을 내려받는다."""
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"model_download_started={destination.name}")
    urllib.request.urlretrieve(source_url, destination)


def _run(command: list[str]) -> None:
    """하위 명령 실패를 즉시 상위 실행자에게 전달한다."""
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
