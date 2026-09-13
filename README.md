# local_face_recognition

실시간 카메라 영상에서 사람을 검출·추적하고, 얼굴 품질 평가와 임베딩 분석을
수행하는 Python 비전 시스템이다. 로컬 운영 모드에서는 등록 인물의 얼굴
템플릿을 검색해 새 관찰에서 이름을 표시한다.

## 기술 스택

- Python 3.9~3.12 (Python 3.13 이상, 특히 3.14는 현재 고정 의존성을 지원하지 않음)
- OpenCV
- Ultralytics YOLO
- InsightFace와 ONNX Runtime
- ONNX 얼굴 가림 분류기
- NumPy
- 자체 운영 PostgreSQL + pgvector
- 자체 운영 MinIO 객체 저장소

운영 모드는 두 가지다. **대규모 운영**은 PostgreSQL + pgvector와 MinIO를 사용한다.
**로컬 운영**은 설치 없이 SQLite와 로컬 private 얼굴 crop 파일을 사용한다. 두 모드는
동일한 도메인·유스케이스를 실행하며, 저장·검색 인프라 어댑터만 다르다.

## 로컬 처리 경계

기본 실행에서는 카메라 프레임, 얼굴 crop, 임베딩, 신원 검색과 SQLite 데이터가 모두 이 PC
안에서 처리된다. 모델 추론은 로컬 ONNX Runtime CPU 실행이며, 외부 API·클라우드 추론·원격
저장소로 얼굴 정보를 전송하지 않는다. Telegram 등록 채널을 명시적으로 선택한 경우에만 임시
코드와 운영자가 입력한 이름이 Telegram Bot API로 전송되며, 얼굴 이미지와 임베딩은 전송하지
않는다.

## 실행

macOS·Linux:

```bash
python3.12 run.py
```

Windows:

```powershell
py -3.12 run.py
```

`run.py`는 프로젝트 전용 가상환경을 만들고 Python 의존성을 설치한다. 이어 YOLO,
InsightFace, 얼굴 가림 모델을 각 공개 원본에서 자동으로 내려받은 뒤 애플리케이션을 실행한다.
이미 준비된 항목은 다시 내려받지 않는다. 다운로드는 3회 재시도하며, 최종 실패 시 모델명과
원본 URL, 네트워크·프록시·보안 프로그램 확인 안내를 출력한다. 영상 창에서 `q`를 누르면
종료된다. 현재 실행은 SQLite 기반 로컬 운영 모드다.
등록 프로필로 확인되지 않으면 임시 코드로 보관하고 터미널에서 이름을 입력해 등록할 수 있으며, 이후 새 Track에서 같은
인물의 표본 두 개가 확인되면 카메라에 저장된 이름을 표시한다.

### Telegram 등록 채널

기본 등록 채널은 Terminal이다. Telegram Bot을 사용하려면 BotFather에서 받은 토큰과, 등록
명령을 보낼 운영자 chat ID를 실행 환경변수로만 설정한다. 토큰은 소스·SQLite·Git에 저장하지
않는다.

```bash
export LOCAL_FACE_RECOGNITION_REGISTRATION_CHANNEL=telegram
export LOCAL_FACE_RECOGNITION_TELEGRAM_BOT_TOKEN='BotFather token'
export LOCAL_FACE_RECOGNITION_TELEGRAM_ALLOWED_CHAT_ID='authorized chat ID'
python3.12 run.py
```

Bot은 `임시 인물 U-XXXXXXXX` 요청을 보내며, 운영자는 아래 중 하나로 응답한다.

```text
/register U-XXXXXXXX 이름
/reject U-XXXXXXXX
```

다른 chat ID의 메시지와 현재 활성 코드에 맞지 않는 명령은 무시한다. Telegram은 외부 서비스이므로
이 모드는 운영자가 명시적으로 선택했을 때만 사용한다.

## 프로젝트 문서

- [시스템 설계](docs/system-design.md): 전체 구조, 책임 경계, 실시간 실행 원칙, 기술 스택, 논리 DB 스키마
- [도메인 설계](docs/domain-design.md): 핵심 도메인, PersonProfile, 개념적 타입과 계약
- [처리 흐름](docs/processing-flow.md): 분석·추적·신원 상태 흐름과 다이어그램
- [미결 설계 항목](docs/open-design-items.md): DB, 등록 업무, 정책, 운영 설계의 선행 결정
