"""현재 OS용 local_face_recognition 실행 파일을 만든다."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_VENV = PROJECT_ROOT / ".build-venv"
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
    """독립 빌드 환경, 모델 번들, PyInstaller 실행 파일을 순서대로 준비한다."""
    python = _prepare_build_environment()
    _prepare_models()
    _build_executable(python)
    print(f"build_complete={PROJECT_ROOT / 'dist'}")


def _prepare_build_environment() -> Path:
    """프로젝트와 분리된 빌드 가상환경에 배포 의존성을 설치한다."""
    if not BUILD_VENV.exists():
        _run([sys.executable, "-m", "venv", str(BUILD_VENV)])
    python = BUILD_VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt"), "pyinstaller"])
    return python


def _prepare_models() -> None:
    """GitHub Release 모델 번들을 한 번만 내려받아 빌드 입력으로 준비한다."""
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


def _build_executable(python: Path) -> None:
    """현재 OS에서 실행 가능한 PyInstaller 번들을 dist에 생성한다."""
    separator = ";" if os.name == "nt" else ":"
    _run(
        [
            str(python), "-m", "PyInstaller", "--noconfirm", "--clean",
            "--name", "local_face_recognition",
            "--add-data", f"{MODELS_DIRECTORY}{separator}models",
            "--collect-all", "insightface",
            "--collect-all", "onnxruntime",
            "--collect-all", "ultralytics",
            str(PROJECT_ROOT / "local_face_recognition.py"),
        ]
    )


def _run(command: list[str]) -> None:
    """빌드 실패를 즉시 호출자에게 전달한다."""
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
