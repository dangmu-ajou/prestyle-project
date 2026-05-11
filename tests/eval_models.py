"""
Person crop 파이프라인 — 4개 Yolov8s PPE 모델 통합 평가
  1) yolov8n으로 사람 탐지
  2) 사람 bbox crop → 각 PPE 모델 추론
  3) 결과 영상 저장 + 통계 출력
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

ROOT       = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / "data" / "test2(성능 탐지 모델 체크용-20초).mp4"
OUTPUT_DIR = ROOT / "data" / "output"
MODEL_DIR  = ROOT / "models"

PERSON_MODEL_PATH = ROOT / "yolov8n.pt"
PERSON_CONF = 0.35
PAD = 20  # person bbox 패딩

# (모델 파일, 라벨, confidence threshold)
PPE_MODELS = [
    ("Yolov8s_helmet_best.pt",       "helmet",       0.25),
    ("Yolov8s_vest_best.pt",         "vest",         0.45),
    ("Yolov8s_gloves_best.pt",       "gloves",       0.20),
    ("Yolov8s_safety_boots_best.pt", "safety_boots", 0.50),
]

COLORS = {
    "helmet":       (0, 255, 255),   # 노란색
    "vest":         (0, 165, 255),   # 주황색
    "gloves":       (255, 255, 0),   # 시안색
    "safety_boots": (255, 0, 255),   # 자홍색
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run():
    print("모델 로드 중...")
    person_model = YOLO(str(PERSON_MODEL_PATH))
    ppe_models = [
        (YOLO(str(MODEL_DIR / mfile)), label, conf)
        for mfile, label, conf in PPE_MODELS
    ]
    print("모델 로드 완료\n")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {VIDEO_PATH}")
        return

    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = str(OUTPUT_DIR / "person_crop_result.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    stats = defaultdict(lambda: {"det_frames": 0, "total_dets": 0, "confs": []})
    person_detected_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        annotated = frame.copy()

        # ── 1. 사람 탐지 ──
        person_res = person_model(frame, classes=[0], conf=PERSON_CONF, verbose=False)[0]

        if person_res.boxes is None or len(person_res.boxes) == 0:
            writer.write(annotated)
            if frame_idx % 30 == 0:
                print(f"  {frame_idx}/{total} frames", end="\r")
            continue

        person_detected_frames += 1

        for pbox in person_res.boxes:
            px1, py1, px2, py2 = map(int, pbox.xyxy[0])
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(annotated, "person", (px1, max(py1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ── 2. person crop (패딩 포함) ──
            cx1 = max(0, px1 - PAD)
            cy1 = max(0, py1 - PAD)
            cx2 = min(w, px2 + PAD)
            cy2 = min(h, py2 + PAD)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # ── 3. PPE 모델 각각 추론 ──
            for model, label, conf_thresh in ppe_models:
                ppe_res = model(crop, conf=conf_thresh, verbose=False)[0]
                if ppe_res.boxes is None or len(ppe_res.boxes) == 0:
                    continue

                detected_this = False
                for pb in ppe_res.boxes:
                    bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
                    # crop 좌표 → 원본 좌표 변환
                    ax1, ay1 = bx1 + cx1, by1 + cy1
                    ax2, ay2 = bx2 + cx1, by2 + cy1
                    conf_val = float(pb.conf[0])

                    color = COLORS[label]
                    cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), color, 2)
                    cv2.putText(annotated, f"{label} {conf_val:.2f}",
                                (ax1, max(ay1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                    stats[label]["confs"].append(conf_val)
                    detected_this = True

                if detected_this:
                    stats[label]["det_frames"] += 1
                    stats[label]["total_dets"] += len(ppe_res.boxes)

        writer.write(annotated)
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames", end="\r")

    cap.release()
    writer.release()
    print(f"\n\n저장 완료: {out_path}")

    # ── 통계 출력 ──
    print("\n" + "=" * 55)
    print("[Person Crop 파이프라인 통계]")
    print("=" * 55)
    print(f"  총 프레임          : {total}")
    print(f"  사람 탐지 프레임   : {person_detected_frames} ({100*person_detected_frames/total:.1f}%)")
    for label, s in [("helmet", stats["helmet"]), ("vest", stats["vest"]),
                     ("gloves", stats["gloves"]), ("safety_boots", stats["safety_boots"])]:
        arr = np.array(s["confs"]) if s["confs"] else np.array([])
        print(f"\n  [{label}]")
        print(f"    탐지 프레임 : {s['det_frames']} / {total} ({100*s['det_frames']/total:.1f}%)")
        print(f"    총 탐지 수  : {s['total_dets']}")
        if arr.size > 0:
            print(f"    Confidence  : avg={arr.mean():.4f}  max={arr.max():.4f}  min={arr.min():.4f}")
        else:
            print(f"    Confidence  : 탐지 없음")


if __name__ == "__main__":
    run()
