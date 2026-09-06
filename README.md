# Camera Streamer

실시간 카메라 영상에서 사람을 검출·추적하고, 얼굴 품질 평가와 임베딩 분석을
수행하는 Python 비전 시스템이다. 로컬 운영 모드에서는 등록 인물의 얼굴
템플릿을 검색해 새 관찰에서 이름을 표시한다.

## 기술 스택

- Python 3.9 이상
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

## 설치

사람 검출 모델 `models/yolo11n.pt`와 얼굴 가림 품질 모델
`models/face_occlusion.onnx`가 로컬에 있어야 한다. 두 모델은 실행 중 내려받지
않으며, 배포 또는 개발 환경 준비 단계에서 고정된 가중치로 배치한다.

가림 모델은 MIT 라이선스의 `Jacky622/face_occlusion` ONNX export이며, 현재 배치한
파일의 SHA-256은 `82b8bda0bf6a29161316618653d2a1351f04e17b6ee71073eeb21b98a727ea1a`다.

## 로컬 처리 경계

실행 중 카메라 프레임, 얼굴 crop, 임베딩, 신원 검색과 SQLite 데이터는 모두 이 PC 안에서만
처리한다. 모델 추론은 로컬 ONNX Runtime CPU 실행이며, 외부 API·클라우드 추론·원격 저장소로
얼굴 정보를 전송하지 않는다.

## 실행

```bash
.venv/bin/python3 main.py
```

영상 창에서 `q`를 누르면 종료된다. 현재 실행은 SQLite 기반 로컬 운영 모드다.
외부인으로 확정되면 터미널에서 이름을 입력해 등록할 수 있고, 이후 새 Track에서 같은
인물의 표본 두 개가 확인되면 카메라에 저장된 이름을 표시한다.

## 프로젝트 문서

- [시스템 설계](docs/system-design.md): 전체 구조, 책임 경계, 실시간 실행 원칙, 기술 스택, 논리 DB 스키마
- [도메인 설계](docs/domain-design.md): 핵심 도메인, PersonProfile, 개념적 타입과 계약
- [처리 흐름](docs/processing-flow.md): 분석·추적·신원 상태 흐름과 다이어그램
- [미결 설계 항목](docs/open-design-items.md): DB, 등록 업무, 정책, 운영 설계의 선행 결정
