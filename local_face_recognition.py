"""local_face_recognition 애플리케이션의 대표 실행 파일이다."""

from local_face_recognition.application import VisionApplication


if __name__ == "__main__":
    try:
        VisionApplication().run()
    except KeyboardInterrupt:
        pass
