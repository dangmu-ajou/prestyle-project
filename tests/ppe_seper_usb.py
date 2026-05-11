'''
카메라 PPE 추적 스크립트 (신체 5존 버전)
사람(yolov8n) + 헬멧 + 조끼 + 장갑(좌/우 분리) + 안전화 (Yolov8s 모델)

GPU(CUDA) 감지 시: TensorRT 자동 변환 + FP16 추론
배치 추론: 같은 모델을 쓰는 crop 묶어서 한 번에 처리
'''

import cv2
import threading
import queue
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# ============================================
# 설정
# ============================================

CAM_INDEX    = 0        # 0: 내장 카메라  /  1: USB 외장 웹캠
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

# GPU 자동 감지
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = (DEVICE == "cuda")

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
        "y_top":   0.30,
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

# 모델별 추론 해상도 (BODY_ZONES에서 자동 유도)
MODEL_IMGSZ: dict[str, int] = {}
for _zk, _zc in BODY_ZONES.items():
    for _lbl in _zc["ppe"]:
        MODEL_IMGSZ[_lbl] = _zc.get("imgsz", 160)

GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
ORANGE = (0, 165, 255)
CYAN   = (255, 255, 0)
PURPLE = (255, 0, 180)

GLOVE_ZONES = {"left_hand", "right_hand"}


# ============================================
# TensorRT 자동 변환 (GPU 환경에서만)
# ============================================

def resolve_model(pt_path: Path, imgsz: int) -> str:
    """
    GPU 환경: .engine 파일 존재 시 로드, 없으면 TensorRT 변환 후 사용.
    CPU 환경: .pt 그대로 사용.
    """
    if DEVICE == "cpu":
        return str(pt_path)

    engine_path = pt_path.with_suffix(".engine")
    if engine_path.exists():
        print(f"  [TRT] {engine_path.name} 로드")
        return str(engine_path)

    print(f"  [TRT] {pt_path.name} 변환 중 → {engine_path.name}  (수 분 소요 가능)...")
    try:
        YOLO(str(pt_path)).export(format="engine", device=0, half=True, imgsz=imgsz)
        if engine_path.exists():
            print(f"  [TRT] 변환 완료: {engine_path.name}")
            return str(engine_path)
    except Exception as e:
        print(f"  [TRT] 변환 실패 → .pt 사용: {e}")

    return str(pt_path)


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

    x1 = max(0,     cx   + int(H * zone["x_start"]))
    y1 = max(0,     py1  + int(H * zone["y_top"]))
    x2 = min(CAM_W, cx   + int(H * zone["x_end"]))
    y2 = min(CAM_H, py1  + int(H * zone["y_bot"]))

    if x2 <= x1 or y2 <= y1:
        return None, (0, 0, 0, 0)

    min_px = zone.get("min_px", 20)
    if (x2 - x1) < min_px or (y2 - y1) < min_px:
        return None, (0, 0, 0, 0)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ============================================
# 배치 PPE 추론
# ============================================

def infer_ppe_batch(
    frame,
    active_persons: dict,       # {tid: person_box}
    ppe_models: dict,
    glove_buffer,
    boots_buffer,
) -> dict:
    """
    모든 사람의 zone crop을 모델별로 묶어 한 번에 추론.
    사람 N명 기준: 기존 N×5회 호출 → 모델 4회 호출(배치).
    Returns: {tid: {"ppe_status": {...}, "ppe_draws": [...]}}
    """
    # task: (tid, zone_key, crop, zone_coords)
    tasks: dict[str, list] = {name: [] for name in ppe_models}

    for tid, person_box in active_persons.items():
        for zone_key, zone_cfg in BODY_ZONES.items():
            crop, zone_coords = get_zone_crop(frame, person_box, zone_key)
            for lbl in zone_cfg["ppe"]:
                if lbl in tasks:
                    tasks[lbl].append((tid, zone_key, crop, zone_coords))

    # detection[lbl][tid] = True if any zone for this person detected the label
    detection: dict[str, dict[int, bool]] = {}
    draws:     dict[int, list]            = {tid: [] for tid in active_persons}

    for lbl, task_list in tasks.items():
        detection[lbl] = {}
        model     = ppe_models[lbl]
        conf_val  = PPE_CONF_MAP.get(lbl, PPE_CONF)
        imgsz_val = MODEL_IMGSZ[lbl]

        valid   = [(t, zk, c, zc) for t, zk, c, zc in task_list if c is not None]
        invalid = [(t, zk)        for t, zk, c, zc in task_list if c is None]

        for tid, _ in invalid:
            detection[lbl].setdefault(tid, False)

        if not valid:
            continue

        batch_res = model(
            [c for _, _, c, _ in valid],
            conf=conf_val,
            iou=0.4,
            imgsz=imgsz_val,
            half=USE_HALF,
            device=DEVICE,
            verbose=False,
        )

        for (tid, zone_key, _, zone_coords), res in zip(valid, batch_res):
            zx1, zy1 = zone_coords[0], zone_coords[1]
            found = False
            if res.boxes is not None:
                for pb in res.boxes:
                    bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
                    color = (CYAN   if lbl == "gloves" else
                             PURPLE if lbl == "safety_boots" else GREEN)
                    draws[tid].append(((bx1+zx1, by1+zy1, bx2+zx1, by2+zy1), lbl, color))
                    found = True
            # OR-accumulate: gloves는 left+right 두 zone 중 하나라도 감지되면 True
            detection[lbl][tid] = detection[lbl].get(tid, False) or found

    # ppe_status per person
    results = {}
    for tid in active_persons:
        ps: dict[str, bool] = {}

        glove_buffer.update(tid, detection.get("gloves", {}).get(tid, False))
        boots_buffer.update(tid, detection.get("safety_boots", {}).get(tid, False))

        ps["gloves"]       = glove_buffer.is_wearing(tid)
        ps["safety_boots"] = boots_buffer.is_wearing(tid)

        for lbl in ["helmet", "vest"]:
            if lbl in detection:
                ps[lbl] = detection[lbl].get(tid, False)

        results[tid] = {"ppe_status": ps, "ppe_draws": draws[tid]}

    return results


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
    print(f"실행 장치: {DEVICE.upper()}" + (" (FP16)" if USE_HALF else ""))

    # Person model (TensorRT 변환 포함)
    person_pt    = Path("yolov8n.pt").resolve()
    person_model = YOLO(resolve_model(person_pt, imgsz=320))

    # PPE models
    ppe_models: dict = {}
    for name, pt_path in PPE_MODEL_PATHS.items():
        if pt_path.exists():
            ppe_models[name] = YOLO(resolve_model(pt_path, MODEL_IMGSZ[name]))
            print(f"  [OK] {name}: {pt_path.name}")
        else:
            print(f"  [!!] {name} 모델 없음: {pt_path}")

    # Warmup
    dummy_full = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    dummy_160  = np.zeros((80,     80,    3), dtype=np.uint8)
    dummy_320  = np.zeros((160,    160,   3), dtype=np.uint8)
    person_model(dummy_full, imgsz=640, half=USE_HALF, device=DEVICE, verbose=False)
    for name, m in ppe_models.items():
        dummy = dummy_320 if name == "helmet" else dummy_160
        sz    = 320       if name == "helmet" else 160
        m(dummy, imgsz=sz, half=USE_HALF, device=DEVICE, verbose=False)
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
    cam_label = "내장 카메라" if CAM_INDEX == 0 else f"USB 외장 카메라 (index={CAM_INDEX})"
    print(f"PPE 추적 시작 ({cam_label})  q → 종료")
    print(f"감지 PPE: {list(ppe_models.keys())}")
    print("=" * 50)

    WIN_NAME = f"PPE Zone Tracking (CAM {CAM_INDEX})"
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
            half=USE_HALF,
            device=DEVICE,
            verbose=False,
        )

        seen_ids       = set()
        current_persons: dict[int, tuple] = {}  # tid → person_box

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0])
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                seen_ids.add(tid)
                current_persons[tid] = (px1, py1, px2, py2)

        # 현재 프레임에서 감지된 사람 위치 갱신 / 신규 등록
        for tid, person_box in current_persons.items():
            if tid in track_state:
                track_state[tid]["box"] = person_box
                track_state[tid]["age"] = 0
            else:
                track_state[tid] = {
                    "box":        person_box,
                    "ppe_status": {l: False for l in ["helmet", "vest", "gloves", "safety_boots"]},
                    "ppe_draws":  [],
                    "age":        0,
                }

        # 배치 PPE 추론 (PPE_INTERVAL 프레임마다)
        if run_ppe and current_persons and ppe_models:
            ppe_results = infer_ppe_batch(
                frame, current_persons, ppe_models, glove_buffer, boots_buffer
            )
            for tid, ppe_data in ppe_results.items():
                track_state[tid]["ppe_status"] = ppe_data["ppe_status"]
                track_state[tid]["ppe_draws"]  = ppe_data["ppe_draws"]

        # 오래된 트랙 제거
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

        # 화면 렌더링
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
