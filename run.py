"""실행 환경을 검증하고 가상환경의 Python으로 애플리케이션을 시작한다."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIRECTORY = PROJECT_ROOT / ".venv"
VENV_PYTHON = VENV_DIRECTORY / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
APPLICATION_FILE = PROJECT_ROOT / "main.py"
MINIMUM_PYTHON_VERSION = (3, 9)


class StartupError(RuntimeError):
    """실행 환경 준비 단계에서 발생한 오류."""


def _version_text(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]}"


def _check_current_python() -> None:
    current = sys.version_info[:2]
    if current < MINIMUM_PYTHON_VERSION:
        raise StartupError(
            f"Python {_version_text(MINIMUM_PYTHON_VERSION)} 이상이 필요합니다. "
            f"현재 버전: {_version_text(current)}"
        )
    print(f"python_version={sys.version.split()[0]}", flush=True)


def _check_virtual_environment() -> None:
    if not VENV_DIRECTORY.is_dir():
        raise StartupError(
            ".venv 가상환경이 없습니다. 'python3 -m venv .venv'를 먼저 실행하세요."
        )
    if not VENV_PYTHON.is_file():
        raise StartupError(
            ".venv의 Python 실행 파일이 없습니다. 가상환경을 삭제한 뒤 다시 생성하세요."
        )

    try:
        result = subprocess.run(
            [
                str(VENV_PYTHON),
                "-c",
                (
                    "import sys; "
                    f"raise SystemExit(0 if sys.version_info[:2] >= {MINIMUM_PYTHON_VERSION!r} else 1)"
                ),
            ],
            cwd=PROJECT_ROOT,
            check=False,
        )
    except OSError as error:
        raise StartupError(".venv의 Python을 실행하지 못했습니다.") from error
    if result.returncode != 0:
        raise StartupError(
            f".venv는 Python {_version_text(MINIMUM_PYTHON_VERSION)} 이상으로 다시 생성해야 합니다."
        )
    print(f"virtual_environment={VENV_DIRECTORY}", flush=True)


def _ensure_requirements() -> None:
    if not REQUIREMENTS_FILE.is_file():
        raise StartupError("requirements.txt 파일이 없습니다.")

    try:
        subprocess.run(
            [
                str(VENV_PYTHON),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "-r",
                str(REQUIREMENTS_FILE),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise StartupError(
            "requirements 설치에 실패했습니다. 네트워크와 requirements.txt를 확인하세요."
        ) from error
    print("requirements_ready=true", flush=True)


def _run_application() -> None:
    if not APPLICATION_FILE.is_file():
        raise StartupError("main.py 파일이 없습니다.")

    try:
        subprocess.run(
            [str(VENV_PYTHON), str(APPLICATION_FILE)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except OSError as error:
        raise StartupError("main.py를 실행하지 못했습니다.") from error
    except subprocess.CalledProcessError as error:
        raise StartupError(
            f"main.py 실행에 실패했습니다. 종료 코드: {error.returncode}"
        ) from error


def main() -> None:
    """환경 준비 단계를 순서대로 통과한 경우에만 main.py를 실행한다."""
    # 1단계: 현재 Python이 프로젝트의 최소 실행 버전을 만족하는지 확인한다.
    _check_current_python()

    # 2단계: 프로젝트 가상환경과 가상환경 Python의 유효성을 확인한다.
    _check_virtual_environment()

    # 3단계: requirements.txt에 선언된 정확한 버전을 설치·확인한다.
    _ensure_requirements()
    print("runtime_ready=true", flush=True)

    # 4단계: 모든 준비가 끝난 뒤 실제 애플리케이션 진입점을 호출한다.
    _run_application()


if __name__ == "__main__":
    try:
        main()
    except StartupError as error:
        print(f"startup_error={error}", file=sys.stderr)
        raise SystemExit(1)
