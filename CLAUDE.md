# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPE(개인보호장비) 실시간 객체 탐지 및 추적 시스템. YOLOv8 + BoT-SORT를 사용하여 웹캠으로 사람과 안전장비(헬멧/조끼/장갑)를 탐지하고 작업구역(ROI) 내 PPE 착용 여부를 검사한다.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Dependencies: `ultralytics`, `opencv-python`, `torch`, `torchvision`, `numpy`, `matplotlib`

## Run Commands

```bash
# 기본 객체 추적 (person/cup/clock/cell phone/book 5개 클래스)
python src/track.py

# PPE 탐지 시스템 (helmet/vest/gloves + ROI 분석) — 주력 스크립트
python src/track_m.py

# 헬멧 탐지 모델 단독 테스트 (USB 웹캠 index=1)
python tests/webcam_test.py
```

실행 중 `q`를 누르면 종료.

## Architecture

```
src/track.py       — 단순 추적. YOLOv8n + ByteTrack. 하드코딩된 ROI(픽셀 좌표).
src/track_m.py     — PPE 추적. YOLOv8 + BoT-SORT. 2단계 검출:
                       1) 전체 프레임에서 person 탐지
                       2) person bbox를 crop하여 PPE 모델(best.pt) 실행
                     신체 구역별 PPE 위치 검증 (헬멧: 상단 0-25%, 조끼: 20-65%, 장갑: 55-100%)
                     상태: GREEN(정상) / RED(미착용 경고) / ORANGE(위치 불일치)
tests/webcam_test.py — models/best.pt를 로드해 USB 웹캠으로 헬멧 단독 탐지 테스트
configs/botsort.yaml — BoT-SORT 추적기 설정. ReID 활성화, track_buffer=60
models/best.pt       — 파인튜닝된 헬멧 탐지 모델 (git에서 제외됨 — .gitignore)
yolov8n.pt / yolov8m.pt — YOLOv8 사전학습 모델 (git에서 제외됨)
```

## Key Configuration (Hardcoded)

| 항목 | 파일 | 값 |
|------|------|----|
| 신뢰도 임계값 | track.py | 0.6 |
| 신뢰도 임계값 | track_m.py | 0.5 |
| PPE 클래스 | track_m.py | `{0: helmet, 1: vest, 2: gloves}` |
| ROI (픽셀) | track.py | (150,100) ~ (490,380) |
| ROI (비율) | track_m.py | 가로 23-77%, 세로 21-79% |
| 웹캠 해상도 | 공통 | 640×480 |
| 웹캠 인덱스 | webcam_test.py | 1 (USB 웹캠) |

## Known TODOs

- `src/track_m.py` 내부에 PPE 모델 로드 관련 TODO 존재 — 모델 학습 완료 후 `best.pt` 연동 필요
- 웹캠 인덱스가 환경에 따라 다를 수 있음 (0, 1, 2 중 선택)
- `data/` 디렉토리는 비어 있음 (데이터셋 별도 준비 필요)

## Branches

- `main` — 안정 브랜치
- `dev` — 현재 개발 브랜치 (활성)
- `feature/model-dev` — 모델 개발 전용
