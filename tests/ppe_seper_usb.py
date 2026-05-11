"""
USB 외장 웹캠 PPE 추적 스크립트 (신체 5존 버전)
사람(yolov8n) + 헬멧 + 조끼 + 장갑(좌/우 분리) + 안전화 (Yolov8s 모델)

ppe_seper.py와 동일, CAM_INDEX = 1 (USB 외장 카메라)
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

CAM_INDEX    = 1        # USB 외장 웹캠
CAM_W        = 640
CAM_H        = 480

PERSON_CONF  = 0.35
PPE_CONF     = 0.50

GLOVE_CONF   = 0.40
PPE_INTERVAL       = 3
TRACK_MAX_AGE      = 8
GLOVE_GRACE_FRAMES = 5
BOOTS_GRACE_FRAMES = 5

MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

BODY_ZONES = {
    "head": {
        "y_top":   0.00,
        "y_bot":   0.28,
        "x_start": -0.13,
        "x_end":    0.13,
        "ppe":     ["helmet"],
        "imgsz":   320,
        "min_px":  22,
    },
    "torso": {
        "y_top":   0.30,   # 목/턱 아래부터 → 머리 vest 오탐 방지
        "y_bot":   0.68,
        "x_start": -0.16,
        "x_end":    0.16,
        "ppe":     ["vest"],
        "imgsz":   160,
        "min_px":  30,
    },
    "left_hand": {
        "y_top":   0.35,
        "y_bot":   1.00,
        "x_start": -0.38,
        "x_end":   -0.02,
        "ppe":     ["gloves"],
        "imgsz":   160,
        "min_px":  20,
    },
    "right_hand": {
        "y_top":   0.35,
        "y_bot":   1.00,
        "x_start":  0.02,
        "x_end":    0.38,
        "ppe":     ["gloves"],
        "imgsz":   160,
        "min_px":  20,
    },
    "feet": {
        "y_top":   0.72,
        "y_bot":   1.00,
        "x_start": -0.18,
        "x_end":    0.18,
        "ppe":     ["safety_boots"],
        "imgsz":   160,
        "min_px":  20,
    },
}

PPE_MODEL_PATHS = {
    "helmet":       MODELS_DIR / "Yolov8s_helmet_best.pt",
    "vest":         MODELS_DIR / "Yolov8s_vest_best.pt",
    "gloves":       MODELS_DIR / "Yolov8s_gloves_best.pt",
    "safety_boots": MODELS_DIR / "Yolov8s_safety_boots_best.pt",
}

PPE_CONF_MAP = {
    "helmet":       0.45,
    "vest":         0.65,
    "gloves":       GLOVE_CONF,
    "safety_boots": 0.55,
}

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
ORANGE = (0, 165, 255)
CYAN   = (255, 255, 0)
PURPLE = (255, 0, 180)

GLOVE_ZONES = {"left_hand", "right_hand"}


# ============================================
# 신체 존 crop
# ============================================

def get_zone_crop(frame, person_box, zone_key):
    px1, py1, px2, py2 = person_box
    H = py2 - py1
    if H < 20:
        return None, (0, 0, 0, 0)

    cx   = (px1 + px2) // 2
    zone = BODY_ZONES[zone_key]

    x1 = max(0,     cx  + int(H * zone["x_start"]))
    y1 = max(0,     py1 + int(H * zone["y_top"]))
    x2 = min(CAM_W, cx  + int(H * zone["x_end"]))
    y2 = min(CAM_H, py1 + int(H * zone["y_bot"]))

    if x2 <= x1 or y2 <= y1:
        return None, (0, 0, 0, 0)

    min_px = zone.get("min_px", 20)
    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return None, (0, 0, 0, 0)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ============================================
# 시간 기반 PPE 판정 버퍼
# ============================================

class GraceBuffer:
    def __init__(self, grace_frames=5):
        self._history: dict[int, list[bool]] = {}
        self._grace = grace_frames

    def update(self, tid: int, detected: bool):
        hist = self._history.setdefault(tid, [])
        hist.append(detected)
        if len(hist) > self._grace:
            self._history[tid] = hist[-self._grace:]

    def is_wearing(self, tid: int) -> bool:
        return any(self._history.get(tid, []))

    def remove(self, tid: int):
        self._history.pop(tid, None)


# ============================================
# 메인
# ============================================

def run():
    person_model = YOLO("yolov8n.pt")
    ppe_models: dict = {}
    for name, path in PPE_MODEL_PATHS.items():
        if path.exists():
            ppe_models[name] = YOLO(str(path))
            print(f"  [OK] {name}: {path.name}")
        else:
            print(f"  [!!] {name} 모델 없음: {path}")

    dummy_full = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    dummy_160  = np.zeros((80, 80, 3),        dtype=np.uint8)
    dummy_320  = np.zeros((160, 160, 3),      dtype=np.uint8)
    person_model(dummy_full, imgsz=640, verbose=False)
    for name, m in ppe_models.items():
        if name == "helmet":
            m(dummy_320, imgsz=320, verbose=False)
        else:
            m(dummy_160, imgsz=160, verbose=False)
    print("warmup 완료")

    glove_buffer = GraceBuffer(GLOVE_GRACE_FRAMES)
    boots_buffer = GraceBuffer(BOOTS_GRACE_FRAMES)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"카메라(index={CAM_INDEX}) 열기 실패 — USB 카메라 연결을 확인하세요.")
        return

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

    print("=" * 50)
    print("PPE 추적 시작 (USB 외장 카메라)  q → 종료")
    print(f"감지 PPE: {list(ppe_models.keys())}")
    print("=" * 50)

    WIN_NAME = "PPE Zone Tracking (USB)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    track_state: dict = {}
    frame_count = 0

    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_count += 1
        run_ppe = (frame_count % PPE_INTERVAL == 0)

        results = person_model.track(
            source=frame,
            classes=[0],
            conf=PERSON_CONF,
            iou=0.5,
            imgsz=320,
            tracker=str(CONFIGS_DIR / "botsort_no_reid.yaml"),
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
                    ppe_status: dict[str, bool] = {}
                    ppe_draws  = []
                    glove_detected = False

                    for zone_key, zone_cfg in BODY_ZONES.items():
                        crop, zone_coords = get_zone_crop(frame, person_box, zone_key)

                        for lbl in zone_cfg["ppe"]:
                            model = ppe_models.get(lbl)
                            if crop is None or model is None:
                                if zone_key not in GLOVE_ZONES:
                                    ppe_status[lbl] = False
                                continue

                            conf_val  = PPE_CONF_MAP.get(lbl, PPE_CONF)
                            imgsz_val = zone_cfg.get("imgsz", 160)
                            ppe_res   = model(crop, conf=conf_val, iou=0.4,
                                              imgsz=imgsz_val, verbose=False)

                            zx1, zy1 = zone_coords[0], zone_coords[1]
                            for pr in ppe_res:
                                if pr.boxes is None:
                                    continue
                                for pb in pr.boxes:
                                    bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
                                    abs_box = (bx1+zx1, by1+zy1, bx2+zx1, by2+zy1)
                                    color = (CYAN   if lbl == "gloves" else
                                             PURPLE if lbl == "safety_boots" else GREEN)
                                    ppe_draws.append((abs_box, lbl, color))
                                    if lbl == "gloves":
                                        glove_detected = True
                                    else:
                                        ppe_status[lbl] = True

                            if lbl != "gloves":
                                ppe_status.setdefault(lbl, False)

                    glove_buffer.update(tid, glove_detected)
                    ppe_status["gloves"] = glove_buffer.is_wearing(tid)

                    boots_buffer.update(tid, ppe_status.get("safety_boots", False))
                    ppe_status["safety_boots"] = boots_buffer.is_wearing(tid)

                    track_state[tid] = {
                        "box": person_box, "ppe_status": ppe_status,
                        "ppe_draws": ppe_draws, "age": 0,
                    }
                else:
                    if tid in track_state:
                        track_state[tid]["box"] = person_box
                        track_state[tid]["age"] = 0
                    else:
                        track_state[tid] = {
                            "box": person_box,
                            "ppe_status": {l: False for l in
                                           ["helmet", "vest", "gloves", "safety_boots"]},
                            "ppe_draws": [], "age": 0,
                        }

        expired = [
            tid for tid, s in track_state.items()
            if tid not in seen_ids and s["age"] + 1 > TRACK_MAX_AGE
        ]
        for tid in expired:
            del track_state[tid]
            glove_buffer.remove(tid)
            boots_buffer.remove(tid)
        for tid, state in track_state.items():
            if tid not in seen_ids:
                state["age"] += 1

        for tid, state in track_state.items():
            px1, py1, px2, py2 = state["box"]
            ps      = state["ppe_status"]
            missing = [lbl for lbl, ok in ps.items() if not ok]
            worn    = [lbl for lbl, ok in ps.items() if ok]

            if not missing:
                box_color, status = GREEN, "OK"
            elif worn:
                box_color = ORANGE
                status    = f"No {', '.join(missing)}!"
            else:
                box_color, status = RED, "No PPE!"

            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, f"ID:{tid} {status}", (px1, max(py1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            for abs_box, lbl, color in state.get("ppe_draws", []):
                cv2.rectangle(frame,
                              (abs_box[0], abs_box[1]),
                              (abs_box[2], abs_box[3]), color, 1)
                cv2.putText(frame, lbl,
                            (abs_box[0], max(abs_box[1] - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        cv2.imshow(WIN_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
