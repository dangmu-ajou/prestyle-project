"""
영상 파일 PPE 추적 스크립트 (신체 3존 버전)
사람(yolov8n) + 헬멧(helmet_best) + 조끼(vest_best1)

ppe_zone_track.py 기반 — 웹캠 대신 data/input/test.mp4 사용
전체 영상 분석 후 data/output/result.mp4 저장
"""

import cv2
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# ============================================
# 설정
# ============================================

ROOT_DIR   = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT_DIR / "data" / "input" / "test.mp4"
OUT_PATH   = ROOT_DIR / "data" / "output" / "result.mp4"
MODELS_DIR = ROOT_DIR / "models"
CONFIGS_DIR = ROOT_DIR / "configs"

PERSON_CONF  = 0.25   # 영상 환경에 맞게 낮춤
PPE_CONF     = 0.35

PPE_INTERVAL  = 3   # N프레임마다 PPE 추론
TRACK_MAX_AGE = 8   # 감지 누락 허용 프레임 (깜빡임 방지)

# ── 신체 존 정의 (신장 H 기준 비율) ──
BODY_ZONES = {
    "head": {
        "y_top":  0.00,
        "y_bot":  0.25,
        "x_half": 0.14,
        "ppe":    ["helmet"],
    },
    "torso": {
        "y_top":  0.18,
        "y_bot":  0.65,
        "x_half": 0.22,
        "ppe":    ["vest"],
    },
}

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

def get_zone_crop(frame, person_box, zone_key, frame_w, frame_h):
    """신장(H) 기준 타이트 crop 반환."""
    px1, py1, px2, py2 = person_box
    H = py2 - py1
    if H < 20:
        return None, (0, 0, 0, 0)

    cx   = (px1 + px2) // 2
    zone = BODY_ZONES[zone_key]

    x1 = max(0,       cx  - int(H * zone["x_half"]))
    y1 = max(0,       py1 + int(H * zone["y_top"]))
    x2 = min(frame_w, cx  + int(H * zone["x_half"]))
    y2 = min(frame_h, py1 + int(H * zone["y_bot"]))

    if x2 <= x1 or y2 <= y1:
        return None, (0, 0, 0, 0)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


# ============================================
# 메인
# ============================================

def run():
    if not VIDEO_PATH.exists():
        print(f"영상 파일 없음: {VIDEO_PATH}")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ──
    person_model = YOLO(str(MODELS_DIR / "yolov8n.pt"))
    ppe_models   = {
        name: YOLO(str(path))
        for name, path in PPE_MODEL_PATHS.items()
    }

    # ── 영상 열기 ──
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"영상 열기 실패: {VIDEO_PATH}")
        return

    frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── VideoWriter 설정 ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_PATH), fourcc, fps, (frame_w, frame_h))

    # ── warmup ──
    dummy_full  = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    dummy_small = np.zeros((80, 80, 3),           dtype=np.uint8)
    person_model(dummy_full,  imgsz=320, verbose=False)
    for m in ppe_models.values():
        m(dummy_small, imgsz=160, verbose=False)
    print("모델 warmup 완료")

    print("=" * 50)
    print(f"분석 시작: {VIDEO_PATH.name} ({total_frames}프레임, {fps:.1f}fps)")
    print(f"저장 경로: {OUT_PATH}")
    print("=" * 50)

    tracker_cfg = "bytetrack.yaml"
    track_state: dict = {}
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        run_ppe = (frame_count % PPE_INTERVAL == 0)

        # ── 사람 감지 + 추적 ──
        results = person_model.track(
            source=frame,
            classes=[0],
            conf=PERSON_CONF,
            iou=0.5,
            imgsz=640,
            tracker=tracker_cfg,
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
                    ppe_status = {}
                    ppe_draws  = []

                    for zone_key, zone_cfg in BODY_ZONES.items():
                        crop, zone_coords = get_zone_crop(
                            frame, person_box, zone_key, frame_w, frame_h
                        )

                        for lbl in zone_cfg["ppe"]:
                            if crop is None:
                                ppe_status[lbl] = False
                                continue
                            model = ppe_models.get(lbl)
                            if model is None:
                                ppe_status[lbl] = False
                                continue

                            ppe_res = model(
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

        # ── 렌더링 ──
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

        writer.write(frame)

        # ── 진행률 출력 ──
        if frame_count % 30 == 0 or frame_count == total_frames:
            pct = frame_count / total_frames * 100 if total_frames else 0
            print(f"  [{frame_count}/{total_frames}] {pct:.1f}%  감지 인원: {len(track_state)}명")

    cap.release()
    writer.release()
    print("=" * 50)
    print(f"분석 완료 → {OUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    run()
