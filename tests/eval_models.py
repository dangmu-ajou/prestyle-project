"""
PPE 모델 통합 평가
  [A] 4개 개별 모델 (Yolov8s_*_best.pt)
  [B] 통합 모델   (ppe_total.pt)
  동일 영상·동일 person crop에 두 파이프라인을 병렬 실행 후 통계 비교
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

PERSON_MODEL_PATH  = ROOT / "yolov8n.pt"
TOTAL_MODEL_PATH   = MODEL_DIR / "ppe_total.pt"

PERSON_CONF = 0.35
PAD         = 20     # person bbox 패딩

# ── [A] 개별 모델 ──────────────────────────────────────────────
PPE_MODELS = [
    ("Yolov8s_helmet_best.pt",       "helmet",       0.25),
    ("Yolov8s_vest_best.pt",         "vest",         0.45),
    ("Yolov8s_gloves_best.pt",       "gloves",       0.20),
    ("Yolov8s_safety_boots_best.pt", "safety_boots", 0.50),
]

# ── [B] 통합 모델 ──────────────────────────────────────────────
TOTAL_CONF     = 0.25
PPE_CLASS_NAMES = {"helmet", "vest", "gloves", "safety_boots"}  # 'worker' 제외

# 색상: [A] 두꺼운 박스 / [B] 얇은 박스 + "(T)" 라벨
COLORS = {
    "helmet":       (0, 255, 255),
    "vest":         (0, 165, 255),
    "gloves":       (255, 255, 0),
    "safety_boots": (255, 0, 255),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


def print_stats(label: str, s: dict, total_frames: int):
    arr = np.array(s["confs"]) if s["confs"] else np.array([])
    print(f"\n  [{label}]")
    print(f"    탐지 프레임 : {s['det_frames']} / {total_frames} ({100*s['det_frames']/total_frames:.1f}%)")
    print(f"    총 탐지 수  : {s['total_dets']}")
    if arr.size > 0:
        print(f"    Confidence  : avg={arr.mean():.4f}  max={arr.max():.4f}  min={arr.min():.4f}")
    else:
        print(f"    Confidence  : 탐지 없음")


def run():
    print("모델 로드 중...")
    person_model = YOLO(str(PERSON_MODEL_PATH))

    # [A] 개별 모델
    sep_models = []
    for mfile, label, conf in PPE_MODELS:
        path = MODEL_DIR / mfile
        if path.exists():
            sep_models.append((YOLO(str(path)), label, conf))
            print(f"  [A] {mfile}")
        else:
            print(f"  [A] 없음: {mfile}")

    # [B] 통합 모델
    total_model = None
    total_names = {}
    if TOTAL_MODEL_PATH.exists():
        total_model = YOLO(str(TOTAL_MODEL_PATH))
        total_names = total_model.names
        print(f"  [B] ppe_total.pt  classes={total_names}")
    else:
        print(f"  [B] ppe_total.pt 없음: {TOTAL_MODEL_PATH}")
    print()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] 영상 열기 실패: {VIDEO_PATH}")
        return

    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = str(OUTPUT_DIR / "eval_comparison.mp4")
    writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    sep_stats   = defaultdict(lambda: {"det_frames": 0, "total_dets": 0, "confs": []})
    total_stats = defaultdict(lambda: {"det_frames": 0, "total_dets": 0, "confs": []})
    person_detected_frames = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        annotated = frame.copy()

        # ── 1. 사람 탐지 ──────────────────────────────────────
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

            # ── 2. person crop ────────────────────────────────
            cx1 = max(0, px1 - PAD)
            cy1 = max(0, py1 - PAD)
            cx2 = min(w, px2 + PAD)
            cy2 = min(h, py2 + PAD)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # ── 3-A. 개별 모델 추론 (두꺼운 박스) ───────────
            for model, label, conf_thresh in sep_models:
                res = model(crop, conf=conf_thresh, verbose=False)[0]
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                detected_this = False
                for pb in res.boxes:
                    bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
                    ax1, ay1 = bx1 + cx1, by1 + cy1
                    ax2, ay2 = bx2 + cx1, by2 + cy1
                    conf_val  = float(pb.conf[0])
                    color     = COLORS[label]
                    cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), color, 2)
                    cv2.putText(annotated, f"{label} {conf_val:.2f}",
                                (ax1, max(ay1 - 5, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                    sep_stats[label]["confs"].append(conf_val)
                    detected_this = True
                if detected_this:
                    sep_stats[label]["det_frames"] += 1
                    sep_stats[label]["total_dets"] += len(res.boxes)

            # ── 3-B. 통합 모델 추론 (얇은 박스 + "(T)") ─────
            if total_model is not None:
                res = total_model(crop, conf=TOTAL_CONF, verbose=False)[0]
                if res.boxes is not None:
                    detected_labels: set[str] = set()
                    for pb in res.boxes:
                        cls_id   = int(pb.cls[0])
                        cls_name = total_names.get(cls_id, f"cls{cls_id}")
                        if cls_name not in PPE_CLASS_NAMES:
                            continue
                        conf_val = float(pb.conf[0])
                        bx1, by1, bx2, by2 = map(int, pb.xyxy[0])
                        ax1, ay1 = bx1 + cx1, by1 + cy1
                        ax2, ay2 = bx2 + cx1, by2 + cy1
                        color    = COLORS.get(cls_name, (255, 255, 255))
                        cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), color, 1)
                        cv2.putText(annotated, f"{cls_name}(T) {conf_val:.2f}",
                                    (ax1, max(ay1 - 16, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
                        total_stats[cls_name]["confs"].append(conf_val)
                        detected_labels.add(cls_name)
                    for lbl in detected_labels:
                        total_stats[lbl]["det_frames"] += 1
                        total_stats[lbl]["total_dets"] += 1

        writer.write(annotated)
        if frame_idx % 30 == 0:
            print(f"  {frame_idx}/{total} frames", end="\r")

    cap.release()
    writer.release()
    print(f"\n\n저장 완료: {out_path}")

    ppe_labels = ["helmet", "vest", "gloves", "safety_boots"]

    # ── 통계: [A] 개별 모델 ───────────────────────────────────
    print("\n" + "=" * 55)
    print("[A] 개별 모델 4종 통계")
    print("=" * 55)
    print(f"  총 프레임        : {total}")
    print(f"  사람 탐지 프레임 : {person_detected_frames} ({100*person_detected_frames/total:.1f}%)")
    for lbl in ppe_labels:
        print_stats(lbl, sep_stats[lbl], total)

    # ── 통계: [B] 통합 모델 ──────────────────────────────────
    if total_model is not None:
        print("\n" + "=" * 55)
        print("[B] 통합 모델 (ppe_total.pt) 통계")
        print("=" * 55)
        print(f"  총 프레임        : {total}")
        print(f"  사람 탐지 프레임 : {person_detected_frames} ({100*person_detected_frames/total:.1f}%)")
        for lbl in ppe_labels:
            print_stats(lbl, total_stats[lbl], total)

    # ── 비교 요약 ─────────────────────────────────────────────
    if total_model is not None:
        print("\n" + "=" * 55)
        print("[A vs B] 탐지 프레임 비교")
        print("=" * 55)
        print(f"  {'항목':<14}  {'[A] 개별':>10}  {'[B] total':>10}")
        print(f"  {'-'*14}  {'-'*10}  {'-'*10}")
        for lbl in ppe_labels:
            a = sep_stats[lbl]["det_frames"]
            b = total_stats[lbl]["det_frames"]
            print(f"  {lbl:<14}  {a:>10}  {b:>10}")


if __name__ == "__main__":
    run()
