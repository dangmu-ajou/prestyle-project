"""
USB 카메라 단일 PPE 추적 스크립트 (신체 3존 버전)
사람(yolov8n) + 헬멧(helmet_best) + 조끼(vest_best1)

[기존 ppe_usb_cam.py 대비 개선점]
  - 팔 벌림 문제 해결: person bbox 너비 대신 신장(H) 기준 타이트 crop 사용
    → 팔을 아무리 벌려도 머리/몸통 존 크기는 고정
  - 이상행동 탐지 대비: 각 PPE를 해당 존에서만 감지하므로
    "PPE가 올바른 신체 위치에 없음" 판단 기반 마련
  - 화면에는 통합 bbox만 표시 (존 분할선 미표시)

[신체 존 정의]
  head  → 헬멧, 보안경, 마스크 (추후 모델 추가)
  torso → 조끼
  lower → 장갑, 안전화 (추후 모델 + 감지 전략 추가)

[장갑/안전화 이상행동 탐지 전략 (미구현, 모델 준비 후 추가)]
  장갑 : 시간 기반 — N프레임 연속 미감지 시 이상
  안전화: lower 존 하단 감지 (발은 손보다 위치 안정적)

[프레임 최적화]
  - 캡처 스레드 분리 + 큐 maxsize=1
  - PPE_INTERVAL 프레임마다 PPE 추론
  - PPE crop imgsz=160
  - 모델 warmup
  - lock 없음 (단일 추론 스레드)
  - _track_state 버퍼로 깜빡임 방지
"""

import cv2
import threading
import queue
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ============================================
# 설정
# ============================================

CAM_INDEX    = 1
CAM_W        = 640
CAM_H        = 480

PERSON_CONF  = 0.35
PPE_CONF     = 0.40

PPE_INTERVAL  = 3   # N프레임마다 PPE 추론 (높을수록 빠름, 낮을수록 정확)
TRACK_MAX_AGE = 8   # 감지 누락 허용 프레임 (깜빡임 방지)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# ── 신체 존 정의 (신장 H 기준 비율) ──
# y_top / y_bot : person bbox 상단에서의 비율  (0.0 = 머리 꼭대기)
# x_half        : 수평 중심에서 ±(비율 * H)   ← 팔 벌림 무관
# ppe           : 이 존에서 감지할 PPE 목록
BODY_ZONES = {
    "head": {
        "y_top":  0.00,
        "y_bot":  0.25,
        "x_half": 0.14,
        "ppe":    ["helmet"],       # 추후: "goggles", "mask"
    },
    "torso": {
        "y_top":  0.18,
        "y_bot":  0.65,
        "x_half": 0.22,
        "ppe":    ["vest"],
    },
    # "lower": 장갑·안전화 모델 준비 후 추가
}

# PPE 이름 → 모델 파일 경로
PPE_MODEL_PATHS = {
    "helmet": MODELS_DIR / "helmet_best.pt",
    "vest":   MODELS_DIR / "vest_best1.pt",
}

# 색상 (BGR)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)


# ============================================
# 신체 존 crop
# ============================================

def get_zone_crop(frame, person_box, zone_key):
    """
    신장(H) 기준 타이트 crop 반환.

    Returns:
        crop   : ndarray, 유효하지 않으면 None
        coords : (x1, y1, x2, y2) 원본 프레임 좌표
    """
    px1, py1, px2, py2 = person_box
    H  = py2 - py1
    if H < 20:
        return None, (0, 0, 0, 0)

    cx   = (px1 + px2) // 2
    zone = BODY_ZONES[zone_key]

    x1 = max(0,     cx  - int(H * zone["x_half"]))
    y1 = max(0,     py1 + int(H * zone["y_top"]))
    x2 = min(CAM_W, cx  + int(H * zone["x_half"]))
    y2 = min(CAM_H, py1 + int(H * zone["y_bot"]))

    if x2 <= x1 or y2 <= y1:
        return None, (0, 0, 0, 0)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ============================================
# 메인
# ============================================

def run():
    # ── 모델 로드 ──
    person_model = YOLO("yolov8n.pt")
    ppe_models   = {
        name: YOLO(str(path))
        for name, path in PPE_MODEL_PATHS.items()
    }

    # ── warmup (첫 프레임 JIT 지연 제거) ──
    dummy_full  = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    dummy_small = np.zeros((80, 80, 3),       dtype=np.uint8)
    person_model(dummy_full,  imgsz=320, verbose=False)
    for m in ppe_models.values():
        m(dummy_small, imgsz=160, verbose=False)
    print("모델 warmup 완료")

    # ── 카메라 ──
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"카메라({CAM_INDEX}) 열기 실패")
        return

    # ── 캡처 스레드 ──
    frame_queue = queue.Queue(maxsize=1)
    stop        = threading.Event()

    def capture_loop():
        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

    threading.Thread(target=capture_loop, daemon=True).start()

    print("=" * 45)
    print("PPE 추적 시작 (USB 카메라)  q → 종료")
    print("=" * 45)

    # track_id → {box, ppe_status, age}
    track_state: dict = {}
    frame_count = 0

    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_count += 1
        run_ppe = (frame_count % PPE_INTERVAL == 0)

        # ── 사람 감지 + 추적 (매 프레임) ──
        results = person_model.track(
            source=frame,
            classes=[0],
            conf=PERSON_CONF,
            iou=0.5,
            imgsz=320,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        seen_ids = set()

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue

                tid = int(box.id[0])
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                person_box = (px1, py1, px2, py2)
                seen_ids.add(tid)

                if run_ppe:
                    # ── 존 기반 PPE 감지 ──
                    ppe_status = {}
                    ppe_draws  = []

                    for zone_key, zone_cfg in BODY_ZONES.items():
                        crop, zone_coords = get_zone_crop(frame, person_box, zone_key)

                        for lbl in zone_cfg["ppe"]:
                            if crop is None:
                                ppe_status[lbl] = False
                                continue
                            model = ppe_models.get(lbl)
                            if model is None:
                                ppe_status[lbl] = False
                                continue

                            ppe_res  = model(
                                crop, conf=PPE_CONF, iou=0.5,
                                imgsz=160, verbose=False,
                            )
                            detected = False
                            zx1, zy1 = zone_coords[0], zone_coords[1]
                            for pr in ppe_res:
                                if pr.boxes is None:
                                    continue
                                for pb in pr.boxes:
                                    bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
                                    abs_box = (bx1+zx1, by1+zy1, bx2+zx1, by2+zy1)
                                    ppe_draws.append((abs_box, lbl, GREEN))
                                    detected = True
                            ppe_status[lbl] = detected

                    track_state[tid] = {
                        "box":        person_box,
                        "ppe_status": ppe_status,
                        "ppe_draws":  ppe_draws,
                        "age":        0,
                    }

                else:
                    # ── PPE 스킵 프레임: 위치만 갱신, 상태 버퍼 유지 ──
                    if tid in track_state:
                        track_state[tid]["box"] = person_box
                        track_state[tid]["age"] = 0
                    else:
                        all_labels = [l for z in BODY_ZONES.values() for l in z["ppe"]]
                        track_state[tid] = {
                            "box":        person_box,
                            "ppe_status": {l: False for l in all_labels},
                            "ppe_draws":  [],
                            "age":        0,
                        }

        # ── 미감지 트랙 age 처리 ──
        expired = [
            tid for tid, s in track_state.items()
            if tid not in seen_ids and s["age"] + 1 > TRACK_MAX_AGE
        ]
        for tid in expired:
            del track_state[tid]
        for tid, state in track_state.items():
            if tid not in seen_ids:
                state["age"] += 1

        # ── 렌더링 (통합 bbox만, 존 선 없음) ──
        for tid, state in track_state.items():
            px1, py1, px2, py2 = state["box"]
            ps      = state["ppe_status"]
            missing = [lbl for lbl, ok in ps.items() if not ok]
            worn    = [lbl for lbl, ok in ps.items() if ok]

            if not missing:
                box_color = GREEN
                status    = f"ID:{tid} OK"
            elif worn:
                box_color = ORANGE
                status    = f"ID:{tid} No {', '.join(missing)}!"
            else:
                box_color = RED
                status    = f"ID:{tid} No PPE!"

            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, status, (px1, max(py1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            for abs_box, lbl, color in state.get("ppe_draws", []):
                cv2.rectangle(frame,
                              (abs_box[0], abs_box[1]),
                              (abs_box[2], abs_box[3]),
                              color, 1)
                cv2.putText(frame, lbl,
                            (abs_box[0], max(abs_box[1] - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("PPE Zone Tracking (USB)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
