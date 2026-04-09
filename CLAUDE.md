# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

작업장 안전 모니터링을 위한 실시간 PPE(개인 보호 장비) 감지 및 추적 시스템. YOLOv8 + BoT-SORT를 사용하여 사람을 감지하고 프레임 간 추적하며, 헬멧·조끼·장갑 등 필수 PPE 착용 여부를 검증한다.

## 설치 및 실행

> **주의:** 아래 명령은 반드시 가상환경(venv)을 활성화한 상태에서 실행해야 한다.
> ```bash
> # Windows
> .venv\Scripts\activate
> ```

```bash
# 의존성 설치
pip install -r requirements.txt

# 헬멧만 감지 (models/best.pt + 웹캠 인덱스 1)
python tests/botsort+helmet.py

# ROI 구역 포함 전체 PPE 시스템 (PPE 모델 미학습 — ppe_model = None)
python tests/track_botsort.py

# 전체 화면 PPE 추적 (ROI 없음)
python tests/botsort_m.py

# 외부 웹캠 사용
python tests/ppe_helmet.py
```

실행 중 `q` 키를 누르면 종료된다.

## 아키텍처

모든 스크립트는 동일한 3단계 파이프라인을 따른다:

1. **사람 감지 + 추적** — `YOLO("yolov8n.pt").track(tracker="configs/botsort.yaml", persist=True)`로 사람을 감지하고 프레임 간 일관된 트랙 ID를 유지한다.
2. **PPE 감지** — 사람 바운딩 박스를 20px 패딩을 포함해 crop하고 별도 PPE 모델(`YOLO("models/best.pt")`)에 전달한다. crop 좌표는 원본 프레임 좌표로 변환한다.
3. **신체 영역 검증** — `check_ppe_in_zone()`으로 감지된 PPE 중심점이 해당 신체 영역 비율 안에 있는지 확인한다 (헬멧: 상단 25%, 조끼: 20~65%, 장갑: 55~100%).

### 주요 파일

| 파일 | 역할 |
|------|------|
| `tests/track_botsort.py` | 메인 스크립트 — ROI 작업 구역 포함 전체 PPE 시스템 |
| `tests/botsort+helmet.py` | 현재 개발 중인 변형 — `models/best.pt`로 헬멧만 감지 |
| `tests/botsort_m.py` | 전체 화면 변형 (ROI 없음) |
| `tests/ppe_helmet.py` | 외부 USB 웹캠 변형 |
| `configs/botsort.yaml` | BoT-SORT 트래커 설정 — `track_buffer`, `with_reid`, 신뢰도 임계값 |
| `models/best.pt` | 학습된 헬멧 감지 모델 (YOLOv8 파인튜닝) |

### PPE 클래스 ID (models/best.pt)

```python
PPE_CLASSES = {0: "helmet", 1: "vest", 2: "gloves"}
```

현재 학습된 모델에서는 `helmet`(클래스 0)만 확인됨. `vest`와 `gloves`는 추후 학습 예정.

### 웹캠 인덱스

- `botsort+helmet.py`, `ppe_helmet.py` → 인덱스 `1` (외부 USB 카메라)
- `track_botsort.py`, `botsort_m.py` → 인덱스 `0` (내장 카메라)

## 현재 개발 상태

- `models/best.pt`는 헬멧 감지 전용 모델이며, 조끼·장갑 감지는 아직 미학습
- `track_botsort.py`에서 `ppe_model = None`으로 설정되어 있어 전체 PPE 감지 비활성 상태
- 현재 실제로 사용 중인 스크립트는 `botsort+helmet.py`
- `src/`, `notebooks/` 디렉토리는 미사용 플레이스홀더
