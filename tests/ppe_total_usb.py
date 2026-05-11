"""
카메라 PPE 추적 스크립트 — 통합 모델 버전 (ppe_total.pt)
단일 모델로 helmet·vest·gloves·safety_boots 동시 감지

GPU(CUDA) 감지 시: TensorRT 자동 변환 + FP16 추론
배치 추론: 모든 사람 crop을 한 번에 처리
"""

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
PPE_CONF     = 0.30     # ppe_total 추론 confidence
PAD          = 20       # person crop 패딩 (px)

PPE_INTERVAL       = 3
TRACK_MAX_AGE      = 8
GRACE_FRAMES       = 5  # 최근 N 프레임 union으로 착용 판정

MODELS_DIR  = Path(__file__).resolve().parent.parent / "models"
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

TOTAL_MODEL_PATH = MODELS_DIR / "ppe_total.pt"
REQUIRED_PPE     = {"helmet", "vest", "gloves", "safety_boots"}  # 필수 착용 항목

# GPU 자동 감지
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
USE_HALF = (DEVICE == "cuda")

COLORS = {
    "helmet":       (0, 255, 255),
    "vest":         (0, 165, 255),
    "gloves":       (255, 255, 0),
    "safety_boots": (255, 0, 255),
}
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
ORANGE = (0, 165, 255)


# ============================================
# TensorRT 자동 변환 (GPU 환경에서만)
# ============================================

def resolve_model(pt_path: Path, imgsz: int) -> str:
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
# 다중 PPE Grace 버퍼 (최근 N프레임 union)
# ============================================

class MultiPPEBuffer:
    def __init__(self, grace_frames: int = 5):
        self._history: dict[int, list[set]] = {}
        self._grace = grace_frames

    def update(self, tid: int, detected: set):
        hist = self._history.setdefault(tid, [])
        hist.append(detected)
        if len(hist) > self._grace:
            self._history[tid] = hist[-self._grace:]

    def wearing(self, tid: int) -> set:
        result: set = set()
        for s in self._history.get(tid, []):
            result |= s
        return result

    def remove(self, tid: int):
        self._history.pop(tid, None)


# ============================================
# 배치 PPE 추론
# ============================================

def infer_ppe_batch(
    frame,
    active_persons: dict,   # {tid: person_box}
    total_model,
    ppe_names: dict,        # model.names
    ppe_buffer: MultiPPEBuffer,
    cam_w: int,
    cam_h: int,
) -> dict:
    """
    모든 사람 crop을 한 번에 배치 추론.
    Returns: {tid: {"ppe_worn": set, "ppe_draws": [...]}}
    """
    # crop 수집
    crop_tasks = []   # (tid, crop, cx1, cy1)
    for tid, (px1, py1, px2, py2) in active_persons.items():
        cx1 = max(0,     px1 - PAD)
        cy1 = max(0,     py1 - PAD)
        cx2 = min(cam_w, px2 + PAD)
        cy2 = min(cam_h, py2 + PAD)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size > 0:
            crop_tasks.append((tid, crop, cx1, cy1))

    draws:   dict[int, list] = {tid: [] for tid in active_persons}
    detected_by_tid: dict[int, set] = {tid: set() for tid in active_persons}

    if not crop_tasks:
        for tid in active_persons:
            ppe_buffer.update(tid, set())
        return {
            tid: {"ppe_worn": ppe_buffer.wearing(tid), "ppe_draws": []}
            for tid in active_persons
        }

    batch_res = total_model(
        [c for _, c, _, _ in crop_tasks],
        conf=PPE_CONF,
        iou=0.4,
        imgsz=640,
        half=USE_HALF,
        device=DEVICE,
        verbose=False,
    )

    for (tid, _, cx1, cy1), res in zip(crop_tasks, batch_res):
        if res.boxes is None:
            continue
        for pb in res.boxes:
            cls_id   = int(pb.cls[0])
            cls_name = ppe_names.get(cls_id, f"cls{cls_id}")
            if cls_name not in REQUIRED_PPE:
                continue
            bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
            abs_box = (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1)
            draws[tid].append((abs_box, cls_name, COLORS.get(cls_name, (255, 255, 255))))
            detected_by_tid[tid].add(cls_name)

    results = {}
    for tid in active_persons:
        ppe_buffer.update(tid, detected_by_tid[tid])
        results[tid] = {
            "ppe_worn":  ppe_buffer.wearing(tid),
            "ppe_draws": draws[tid],
        }
    return results


# ============================================
# 메인
# ============================================

def run():
    print(f"실행 장치: {DEVICE.upper()}" + (" (FP16)" if USE_HALF else ""))

    # Person model
    person_pt    = Path("yolov8n.pt").resolve()
    person_model = YOLO(resolve_model(person_pt, imgsz=320))

    # PPE 통합 모델
    if not TOTAL_MODEL_PATH.exists():
        print(f"[ERROR] ppe_total.pt 없음: {TOTAL_MODEL_PATH}")
        return
    total_model = YOLO(resolve_model(TOTAL_MODEL_PATH, imgsz=640))
    ppe_names   = total_model.names
    print(f"  [OK] ppe_total.pt  classes={ppe_names}")

    # Warmup
    dummy_full = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    person_model(dummy_full, imgsz=640, half=USE_HALF, device=DEVICE, verbose=False)
    total_model(dummy_full,  imgsz=640, half=USE_HALF, device=DEVICE, verbose=False)
    print("warmup 완료")

    ppe_buffer = MultiPPEBuffer(GRACE_FRAMES)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print(f"카메라(index={CAM_INDEX}) 열기 실패")
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

    cam_label = "내장 카메라" if CAM_INDEX == 0 else f"USB 외장 카메라 (index={CAM_INDEX})"
    print("=" * 50)
    print(f"PPE 추적 시작 ({cam_label})  q → 종료")
    print(f"필수 PPE: {sorted(REQUIRED_PPE)}")
    print("=" * 50)

    WIN_NAME = f"PPE Total (CAM {CAM_INDEX})"
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

        seen_ids        = set()
        current_persons: dict[int, tuple] = {}

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

        for tid, person_box in current_persons.items():
            if tid in track_state:
                track_state[tid]["box"] = person_box
                track_state[tid]["age"] = 0
            else:
                track_state[tid] = {
                    "box":       person_box,
                    "ppe_worn":  set(),
                    "ppe_draws": [],
                    "age":       0,
                }

        if run_ppe and current_persons:
            ppe_results = infer_ppe_batch(
                frame, current_persons, total_model, ppe_names,
                ppe_buffer, CAM_W, CAM_H,
            )
            for tid, data in ppe_results.items():
                track_state[tid]["ppe_worn"]  = data["ppe_worn"]
                track_state[tid]["ppe_draws"] = data["ppe_draws"]

        expired = [
            tid for tid, s in track_state.items()
            if tid not in seen_ids and s["age"] + 1 > TRACK_MAX_AGE
        ]
        for tid in expired:
            del track_state[tid]
            ppe_buffer.remove(tid)
        for tid, state in track_state.items():
            if tid not in seen_ids:
                state["age"] += 1

        # 화면 렌더링
        for tid, state in track_state.items():
            px1, py1, px2, py2 = state["box"]
            worn    = state["ppe_worn"]
            missing = sorted(REQUIRED_PPE - worn)

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

            for abs_box, cls_name, color in state["ppe_draws"]:
                cv2.rectangle(frame,
                              (abs_box[0], abs_box[1]),
                              (abs_box[2], abs_box[3]), color, 1)
                cv2.putText(frame, cls_name,
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
