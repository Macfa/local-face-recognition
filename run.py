"""local_face_recognition의 실행 환경을 준비하고 애플리케이션을 시작한다."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIRECTORY = PROJECT_ROOT / ".venv"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
MINIMUM_PYTHON_VERSION = (3, 9)
MAXIMUM_PYTHON_VERSION = (3, 12)
DOWNLOAD_ATTEMPTS = 3
DOWNLOAD_TIMEOUT_SECONDS = 90
YOLO_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
OCCLUSION_MODEL_URL = "https://huggingface.co/Jacky622/face_occlusion/resolve/main/face_occlusion.onnx?download=true"
INSIGHTFACE_MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
INSIGHTFACE_MODEL_DIRECTORY = MODELS_DIRECTORY / "insightface" / "models" / "buffalo_l"
REQUIRED_INSIGHTFACE_FILES = (
    "det_10g.onnx",
    "1k3d68.onnx",
    "w600k_r50.onnx",
)


class StartupError(RuntimeError):
    """다른 컴퓨터에서도 원인을 바로 알 수 있게 시작 준비 실패를 구분한다."""


def main() -> None:
    """실행 환경과 로컬 모델을 준비한 후 대표 애플리케이션을 실행한다.

    Returns: None.
    Raises: StartupError. Python·의존성·모델 준비가 불가능할 때.
    """
    _ensure_supported_python()
    python = _prepare_runtime_environment()
    _prepare_models()
    _run([str(python), str(PROJECT_ROOT / "local_face_recognition.py")])


def _prepare_runtime_environment() -> Path:
    """프로젝트 전용 가상환경을 만들고 필요한 Python 패키지를 설치한다.

    Returns: Path. 대표 앱을 실행할 가상환경 Python 경로.
    Raises: StartupError. venv 생성·의존성 설치 또는 Python 경로 검증에 실패할 때.
    """
    if not VENV_DIRECTORY.exists():
        print("virtual_environment_create_started=true")
        try:
            _run([sys.executable, "-m", "venv", str(VENV_DIRECTORY)])
        except subprocess.CalledProcessError as error:
            raise StartupError(
                "Virtual environment creation failed. Reinstall Python 3.12 with the venv module enabled."
            ) from error
    python = VENV_DIRECTORY / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        raise StartupError(f"Virtual environment Python is missing: {python}")
    try:
        _run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")])
    except subprocess.CalledProcessError as error:
        raise StartupError(
            "Dependency installation failed. Check the network, proxy, disk space, and Python 3.12 compatibility."
        ) from error
    return python


def _ensure_supported_python() -> None:
    """고정된 의존성이 휠을 제공하는 Python 범위를 설치 전에 검증한다.

    Raises: StartupError. 현재 인터프리터가 Python 3.9~3.12 범위를 벗어날 때.
    """
    current = sys.version_info[:2]
    if MINIMUM_PYTHON_VERSION <= current <= MAXIMUM_PYTHON_VERSION:
        return
    supported = f"{MINIMUM_PYTHON_VERSION[0]}.{MINIMUM_PYTHON_VERSION[1]}~{MAXIMUM_PYTHON_VERSION[0]}.{MAXIMUM_PYTHON_VERSION[1]}"
    actual = f"{current[0]}.{current[1]}"
    raise StartupError(
        f"Unsupported Python {actual}. This project requires Python {supported}. "
        "Install Python 3.12 and run this file with that interpreter."
    )


def _prepare_models() -> None:
    """각 모델의 공개 원본에서 누락된 로컬 가중치만 자동으로 준비한다.

    다운로드는 최초 준비 단계에만 일어나며, 실행 중 카메라 프레임·얼굴 crop·
    임베딩을 외부로 전송하지 않는다.

    Raises: StartupError. 모델 다운로드·검증·압축 해제에 실패할 때.
    """
    _download_file(MODELS_DIRECTORY / "yolo11n.pt", YOLO_MODEL_URL)
    _download_file(MODELS_DIRECTORY / "face_occlusion.onnx", OCCLUSION_MODEL_URL)
    if all((INSIGHTFACE_MODEL_DIRECTORY / name).is_file() for name in REQUIRED_INSIGHTFACE_FILES):
        return

    archive_path = PROJECT_ROOT / ".insightface-buffalo-l.zip"
    try:
        _download_file(archive_path, INSIGHTFACE_MODEL_URL, "insightface_buffalo_l")
        with TemporaryDirectory(prefix="insightface-extract-", dir=PROJECT_ROOT) as temporary_directory:
            extraction_root = Path(temporary_directory).resolve()
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    destination = (extraction_root / member.filename).resolve()
                    if extraction_root not in destination.parents and destination != extraction_root:
                        raise StartupError("InsightFace model archive contains an invalid path.")
                archive.extractall(extraction_root)
            source_files = {
                path.name: path
                for path in extraction_root.rglob("*.onnx")
                if path.is_file()
            }
            missing = [name for name in REQUIRED_INSIGHTFACE_FILES if name not in source_files]
            if missing:
                raise StartupError(f"InsightFace model archive is incomplete: {', '.join(missing)}")
            INSIGHTFACE_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
            for name, source in source_files.items():
                shutil.copy2(source, INSIGHTFACE_MODEL_DIRECTORY / name)
    except (OSError, zipfile.BadZipFile) as error:
        raise StartupError(
            "InsightFace model preparation failed. Delete the models directory and retry after checking the network and disk space."
        ) from error
    finally:
        archive_path.unlink(missing_ok=True)

    missing = [name for name in REQUIRED_INSIGHTFACE_FILES if not (INSIGHTFACE_MODEL_DIRECTORY / name).is_file()]
    if missing:
        raise StartupError(f"InsightFace model archive is incomplete: {', '.join(missing)}")


def _download_file(destination: Path, source_url: str, model_name: str | None = None) -> None:
    """모델을 재시도·임시 파일·원자 교체 방식으로 안전하게 내려받는다.

    Args:
        destination: Path. 완전한 파일을 둘 프로젝트 로컬 경로.
        source_url: str. 모델 공급자가 제공한 HTTPS 다운로드 URL.
        model_name: str | None. 운영 로그용 모델명. 없으면 파일명을 사용한다.
    Returns: None.
    Raises: StartupError. 재시도 후에도 모델을 준비하지 못할 때.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    name = model_name or destination.name
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        print(f"model_download_started={name} attempt={attempt}", flush=True)
        try:
            request = urllib.request.Request(source_url, headers={"User-Agent": "local-face-recognition/1.0"})
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if temporary_path.stat().st_size == 0:
                raise OSError("Downloaded file is empty.")
            os.replace(temporary_path, destination)
            print(f"model_download_completed={name}", flush=True)
            return
        except (HTTPError, URLError, OSError, TimeoutError) as error:
            temporary_path.unlink(missing_ok=True)
            print(
                f"model_download_failed={name} attempt={attempt} error={type(error).__name__} "
                f"url={source_url}",
                flush=True,
            )
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(attempt)
                continue
            raise StartupError(
                f"Model download failed: {name}. Check the network, proxy, antivirus, firewall, and free disk space, then retry. "
                f"Source: {source_url}"
            ) from error


def _run(command: list[str]) -> None:
    """하위 명령 실패를 즉시 상위 실행자에게 전달한다.

    Args: command: list[str]. shell 없이 실행할 명령과 인자.
    Returns: None.
    Raises: subprocess.CalledProcessError. 하위 명령의 종료 코드가 0이 아닐 때.
    """
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except (StartupError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"startup_error={error}", flush=True)
        raise SystemExit(1)
