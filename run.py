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
MODEL_BUNDLE_URL = (
    "https://github.com/Macfa/local-face-recognition/releases/download/"
    "models-v1/local-face-recognition-models.zip"
)
REQUIRED_MODEL_FILES = (
    "yolo11n.pt",
    "face_occlusion.onnx",
    "insightface/models/buffalo_l/det_10g.onnx",
    "insightface/models/buffalo_l/1k3d68.onnx",
    "insightface/models/buffalo_l/w600k_r50.onnx",
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
    """누락된 로컬 모델을 GitHub Release 번들에서 한 번만 준비한다."""
    if all((MODELS_DIRECTORY / relative_path).is_file() for relative_path in REQUIRED_MODEL_FILES):
        return

    archive_path = PROJECT_ROOT / ".model-bundle.zip"
    print("models_download_started=true")
    try:
        urllib.request.urlretrieve(MODEL_BUNDLE_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            root = MODELS_DIRECTORY.resolve()
            for member in archive.infolist():
                destination = (root / member.filename).resolve()
                if root not in destination.parents and destination != root:
                    raise RuntimeError("Invalid model bundle path.")
            archive.extractall(MODELS_DIRECTORY)
    finally:
        archive_path.unlink(missing_ok=True)

    missing = [path for path in REQUIRED_MODEL_FILES if not (MODELS_DIRECTORY / path).is_file()]
    if missing:
        raise RuntimeError(f"Model bundle is incomplete: {', '.join(missing)}")


def _run(command: list[str]) -> None:
    """하위 명령 실패를 즉시 상위 실행자에게 전달한다."""
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
