"""
USB 카메라 단일 PPE 추적 스크립트
사람(yolov8n) + 헬멧(helmet_best) + 조끼(vest_best1)

[프레임 드랍 최소화 적용 목록]
  1. PPE 추론 매 PPE_INTERVAL 프레임마다만 실행 → 나머지 프레임은 person만 추론
  2. 캡처 전용 스레드 분리 + 큐 maxsize=1 → 추론 중에도 항상 최신 프레임 유지
  3. imgsz=320 (person), imgsz=160 (PPE crop) → 입력 해상도 최소화
  4. conf 임계값을 모델별 최적값으로 조정 (오탐/미탐 균형)
  5. 모델 warmup (첫 프레임 느린 JIT 초기화를 시작 전에 처리)
  6. _track_state 버퍼 → PPE 상태를 프레임 간 유지 (깜빡임 방지 + PPE 스킵 프레임 보완)
  7. lock 없음 (단일 추론 스레드 → 불필요한 동기화 제거)
  8. verbose=False, 불필요한 로그 출력 제거
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

CAM_INDEX    = 1        # USB 카메라
CAM_W        = 640
CAM_H        = 480

PERSON_CONF  = 0.35     # 낮출수록 놓침 줄고 오탐 늘어남
PPE_CONF     = 0.40

PPE_INTERVAL = 3        # N프레임마다 PPE 추론 (1이면 매 프레임, 높을수록 빠름)
TRACK_MAX_AGE = 8       # 감지 누락 허용 프레임 수 (깜빡임 방지)
PAD          = 15       # 사람 crop 패딩(px)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

PPE_BODY_ZONES = {
    "helmet": {"top_ratio": 0.0,  "bottom_ratio": 0.25},
    "vest":   {"top_ratio": 0.20, "bottom_ratio": 0.65},
}

# 색상 (BGR)
GREEN  = (0, 255, 0)
RED    = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE  = (255, 255, 255)


# ============================================
# PPE 신체 영역 검증
# ============================================

def check_ppe_in_zone(ppe_box, person_box, label):
    px1, py1, px2, py2 = person_box
    h = py2 - py1
    cx = (ppe_box[0] + ppe_box[2]) / 2
    cy = (ppe_box[1] + ppe_box[3]) / 2
    if not (px1 <= cx <= px2):
        return False
    zone = PPE_BODY_ZONES.get(label)
    if not zone:
        return False
    return (py1 + h * zone["top_ratio"]) <= cy <= (py1 + h * zone["bottom_ratio"])


# ============================================
# 메인
# ============================================

def run():
    # ── 모델 로드 ──
    person_model = YOLO("yolov8n.pt")
    helmet_model = YOLO(str(MODELS_DIR / "helmet_best.pt"))
    vest_model   = YOLO(str(MODELS_DIR / "vest_best1.pt"))

    # ── 모델 warmup (첫 프레임 JIT 초기화 시간 제거) ──
    dummy = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    person_model(dummy, imgsz=320, verbose=False)
    helmet_model(dummy[:120, :120], imgsz=160, verbose=False)
    vest_model(dummy[:120, :120],   imgsz=160, verbose=False)
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
    stop = threading.Event()

    def capture_loop():
        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            # 큐가 차 있으면 오래된 프레임 버리고 최신으로 교체
            if not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put(frame)

    threading.Thread(target=capture_loop, daemon=True).start()

    print("=" * 45)
    print("PPE 추적 시작 (USB 카메라, q 키로 종료)")
    print("=" * 45)

    track_state = {}   # track_id → {box, helmet, vest, ppe_draws, age}
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
                    # ── PPE 추론 프레임 ──
                    cx1 = max(0, px1 - PAD)
                    cy1 = max(0, py1 - PAD)
                    cx2 = min(CAM_W, px2 + PAD)
                    cy2 = min(CAM_H, py2 + PAD)
                    crop = frame[cy1:cy2, cx1:cx2]

                    helmet_ok = False
                    vest_ok   = False
                    ppe_draws = []

                    for ppe_model, label in [
                        (helmet_model, "helmet"),
                        (vest_model,   "vest"),
                    ]:
                        ppe_res = ppe_model(
                            crop, conf=PPE_CONF, iou=0.5,
                            imgsz=160,   # crop은 작으므로 160으로 충분
                            verbose=False,
                        )
                        for pr in ppe_res:
                            if pr.boxes is None:
                                continue
                            for pb in pr.boxes:
                                bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
                                abs_box = (bx1+cx1, by1+cy1, bx2+cx1, by2+cy1)
                                worn = check_ppe_in_zone(abs_box, person_box, label)
                                if label == "helmet" and worn:
                                    helmet_ok = True
                                if label == "vest" and worn:
                                    vest_ok = True
                                color = GREEN if worn else ORANGE
                                ppe_draws.append(
                                    (abs_box, f"{label}:{'O' if worn else 'X'}", color)
                                )

                    track_state[tid] = {
                        "box":       person_box,
                        "helmet":    helmet_ok,
                        "vest":      vest_ok,
                        "ppe_draws": ppe_draws,
                        "age":       0,
                    }

                else:
                    # ── PPE 스킵 프레임: 박스 위치만 갱신, PPE 상태는 버퍼 유지 ──
                    if tid in track_state:
                        track_state[tid]["box"] = person_box
                        track_state[tid]["age"] = 0
                    else:
                        track_state[tid] = {
                            "box":       person_box,
                            "helmet":    False,
                            "vest":      False,
                            "ppe_draws": [],
                            "age":       0,
                        }

        # ── 미감지 트랙 age 증가 / 만료 제거 ──
        expired = [
            tid for tid, s in track_state.items()
            if tid not in seen_ids and s["age"] + 1 > TRACK_MAX_AGE
        ]
        for tid in expired:
            del track_state[tid]
        for tid, state in track_state.items():
            if tid not in seen_ids:
                state["age"] += 1

        # ── 렌더링 ──
        for tid, state in track_state.items():
            px1, py1, px2, py2 = state["box"]
            h_ok = state["helmet"]
            v_ok = state["vest"]

            if h_ok and v_ok:
                box_color = GREEN
                status    = f"ID:{tid} OK"
            elif h_ok or v_ok:
                box_color = ORANGE
                missing   = "vest" if h_ok else "helmet"
                status    = f"ID:{tid} No {missing}!"
            else:
                box_color = RED
                status    = f"ID:{tid} No PPE!"

            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, status, (px1, max(py1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            for abs_box, label_text, color in state["ppe_draws"]:
                cv2.rectangle(frame,
                              (abs_box[0], abs_box[1]),
                              (abs_box[2], abs_box[3]),
                              color, 1)
                cv2.putText(frame, label_text,
                            (abs_box[0], max(abs_box[1] - 4, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imshow("PPE Tracking (USB)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    stop.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
