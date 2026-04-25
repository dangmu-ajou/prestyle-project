import cv2
import threading
import queue
import traceback
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ============================================
# 설정
# ============================================

PERSON_CLASS_ID = 0

MODELS_DIR   = Path(__file__).resolve().parent.parent / "models"
CONFIGS_DIR  = Path(__file__).resolve().parent.parent / "configs"
BOTSORT_YAML = str(CONFIGS_DIR / "botsort.yaml")

PPE_BODY_ZONES = {
    "helmet": {"top_ratio": 0.0,  "bottom_ratio": 0.25},
    "vest":   {"top_ratio": 0.20, "bottom_ratio": 0.65},
}

TRACK_MAX_AGE = 6   # 감지 누락 허용 프레임 수 (깜빡임 방지)
PAD = 20            # 사람 crop 패딩(px)

CAM_WIDTH  = 640
CAM_HEIGHT = 480

# 색상 (BGR)
COLOR_GREEN  = (0, 255, 0)
COLOR_RED    = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


# ============================================
# PPE 신체 영역 검증
# ============================================

def check_ppe_in_zone(ppe_box, person_box, ppe_class_name):
    px1, py1, px2, py2 = person_box
    person_height = py2 - py1
    ppe_cx = (ppe_box[0] + ppe_box[2]) / 2
    ppe_cy = (ppe_box[1] + ppe_box[3]) / 2
    if not (px1 <= ppe_cx <= px2):
        return False
    zone = PPE_BODY_ZONES.get(ppe_class_name)
    if zone is None:
        return False
    zone_top    = py1 + person_height * zone["top_ratio"]
    zone_bottom = py1 + person_height * zone["bottom_ratio"]
    return zone_top <= ppe_cy <= zone_bottom


# ============================================
# 카메라 스레드
# ============================================

class CameraThread(threading.Thread):
    """
    캡처 루프와 추론 루프를 분리하여 프레임 끊김 방지.
    YOLO 인스턴스는 카메라별로 유지하여 ByteTrack persist 상태를 격리하되,
    .track()/.predict() 호출은 inference_lock으로 직렬화하여 동시 추론 충돌을 방지한다.
    _track_state 버퍼로 순간 감지 누락 시 마지막 상태를 유지한다 (깜빡임 방지).
    """

    def __init__(
        self,
        cam_index: int,
        cam_label: str,
        stop_event: threading.Event,
        inference_lock: threading.Lock,
    ):
        super().__init__(daemon=True)
        self.cam_index      = cam_index
        self.cam_label      = cam_label
        self.stop_event     = stop_event
        self.inference_lock = inference_lock

        self.model        = YOLO("yolov8n.pt")
        self.helmet_model = YOLO(str(MODELS_DIR / "helmet_best.pt"))
        self.vest_model   = YOLO(str(MODELS_DIR / "vest_best1.pt"))

        # 깜빡임 방지 버퍼: track_id → {box, helmet, vest, ppe_draws, age}
        self._track_state: dict = {}

        self._capture_queue = queue.Queue(maxsize=1)
        self._result_frame  = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        self._lock          = threading.Lock()

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"[{self.cam_label}] 카메라({self.cam_index})를 열 수 없습니다. 빈 화면으로 대체합니다.")
            return

        print(f"[{self.cam_label}] 카메라({self.cam_index}) 시작")

        capture_thread = threading.Thread(target=self._capture_loop, args=(cap,), daemon=True)
        capture_thread.start()

        self._inference_loop()

        cap.release()
        capture_thread.join()
        print(f"[{self.cam_label}] 종료")

    def _capture_loop(self, cap):
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            if not self._capture_queue.empty():
                try:
                    self._capture_queue.get_nowait()
                except queue.Empty:
                    pass
            self._capture_queue.put(frame)

    def _inference_loop(self):
        while not self.stop_event.is_set():
            try:
                frame = self._capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                annotated = self._process(frame)
                with self._lock:
                    self._result_frame = annotated
            except Exception as e:
                print(f"[{self.cam_label}] 추론 오류: {e}")
                traceback.print_exc()
                self.stop_event.set()
                break

    def _process(self, frame):
        # ── 사람 감지 + 추적 ──
        with self.inference_lock:
            results = self.model.track(
                source=frame,
                classes=[PERSON_CLASS_ID],
                conf=0.4,
                iou=0.5,
                imgsz=320,
                tracker=BOTSORT_YAML,
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
                track_id = int(box.id[0])
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                person_box = (px1, py1, px2, py2)
                seen_ids.add(track_id)

                # ── 사람 영역 crop ──
                cx1 = max(0, px1 - PAD)
                cy1 = max(0, py1 - PAD)
                cx2 = min(frame.shape[1], px2 + PAD)
                cy2 = min(frame.shape[0], py2 + PAD)
                crop = frame[cy1:cy2, cx1:cx2]

                helmet_ok = False
                vest_ok   = False
                ppe_draws = []   # (abs_box, label_text, color)

                # ── PPE 감지 (헬멧 → 조끼 순서) ──
                for ppe_model, label in [
                    (self.helmet_model, "helmet"),
                    (self.vest_model,   "vest"),
                ]:
                    with self.inference_lock:
                        ppe_res = ppe_model(crop, conf=0.4, iou=0.5, verbose=False)

                    for pr in ppe_res:
                        if pr.boxes is None:
                            continue
                        for pb in pr.boxes:
                            bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
                            abs_box = (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1)
                            worn = check_ppe_in_zone(abs_box, person_box, label)
                            if label == "helmet" and worn:
                                helmet_ok = True
                            if label == "vest" and worn:
                                vest_ok = True
                            color = COLOR_GREEN if worn else COLOR_ORANGE
                            ppe_draws.append((abs_box, f"{label}:{'O' if worn else 'X'}", color))

                # ── 트랙 상태 갱신 ──
                self._track_state[track_id] = {
                    "box":       person_box,
                    "helmet":    helmet_ok,
                    "vest":      vest_ok,
                    "ppe_draws": ppe_draws,
                    "age":       0,
                }

        # ── 미감지 트랙 age 증가 / 만료 제거 ──
        expired = [
            tid for tid, s in self._track_state.items()
            if tid not in seen_ids and s["age"] + 1 > TRACK_MAX_AGE
        ]
        for tid in expired:
            del self._track_state[tid]
        for tid, state in self._track_state.items():
            if tid not in seen_ids:
                state["age"] += 1

        # ── 버퍼 기준 렌더링 ──
        for tid, state in self._track_state.items():
            px1, py1, px2, py2 = state["box"]
            helmet_ok = state["helmet"]
            vest_ok   = state["vest"]

            if helmet_ok and vest_ok:
                box_color = COLOR_GREEN
                status    = f"ID:{tid} OK"
            elif helmet_ok or vest_ok:
                box_color = COLOR_ORANGE
                missing   = "vest" if helmet_ok else "helmet"
                status    = f"ID:{tid} No {missing}!"
            else:
                box_color = COLOR_RED
                status    = f"ID:{tid} No PPE!"

            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, status, (px1, py1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            for abs_box, label_text, color in state["ppe_draws"]:
                cv2.rectangle(frame,
                              (abs_box[0], abs_box[1]),
                              (abs_box[2], abs_box[3]),
                              color, 1)
                cv2.putText(frame, label_text,
                            (abs_box[0], abs_box[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.putText(frame, self.cam_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
        return frame

    def get_frame(self):
        with self._lock:
            return self._result_frame.copy()


# ============================================
# 메인
# ============================================

def run():
    stop_event     = threading.Event()
    inference_lock = threading.Lock()

    cam0 = CameraThread(
        cam_index=0, cam_label="CAM 0 (USB)",
        stop_event=stop_event, inference_lock=inference_lock,
    )
    cam1 = CameraThread(
        cam_index=1, cam_label="CAM 1 (내장)",
        stop_event=stop_event, inference_lock=inference_lock,
    )

    cam0.start()
    cam1.start()

    print("=" * 50)
    print("듀얼 카메라 PPE 추적 시작 (사람 + 헬멧 + 조끼)")
    print("종료: q 키")
    print("=" * 50)

    WIN = "듀얼 카메라 PPE 모니터"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while not stop_event.is_set():
        frame0 = cam0.get_frame()
        frame1 = cam1.get_frame()

        # 두 프레임을 같은 높이로 맞춰 좌우 합치기
        if frame0.shape[0] != frame1.shape[0]:
            frame1 = cv2.resize(frame1, (frame1.shape[1], frame0.shape[0]))
        combined = np.hstack([frame0, frame1])

        cv2.imshow(WIN, combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()

    cam0.join()
    cam1.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
