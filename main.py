"""로컬 카메라 비전 애플리케이션을 시작하는 대표 실행 파일이다."""

from camera_streamer.application import VisionApplication


if __name__ == "__main__":
    try:
        VisionApplication().run()
    except KeyboardInterrupt:
        pass
