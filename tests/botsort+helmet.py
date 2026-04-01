import cv2
import os
from ultralytics import YOLO

# ============================================
# 설정
# ============================================
PERSON_CLASS_ID = 0

PPE_CLASSES = {
    0: "helmet",
}

PPE_BODY_ZONES = {
    "helmet": {"top_ratio": 0.0, "bottom_ratio": 0.25},
}

# 색상 (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)

# 입출력 경로
INPUT_VIDEO = "data/input/test.mp4"
OUTPUT_VIDEO = "data/output/result.mp4"


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
    zone_top = py1 + person_height * zone["top_ratio"]
    zone_bottom = py1 + person_height * zone["bottom_ratio"]
    return zone_top <= ppe_cy <= zone_bottom


def run():
    # 모델 로드
    person_model = YOLO("yolov8m.pt")
    ppe_model = YOLO("models/best.pt")

    # 영상 열기
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"영상을 열 수 없습니다: {INPUT_VIDEO}")
        return

    # 영상 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 폴더 생성
    os.makedirs("outputs", exist_ok=True)

    # 영상 저장 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("=" * 40)
    print("헬멧 감지 테스트 시작")
    print(f"입력: {INPUT_VIDEO}")
    print(f"출력: {OUTPUT_VIDEO}")
    print(f"해상도: {width}x{height}, FPS: {fps}")
    print(f"총 프레임: {total_frames}")
    print("=" * 40)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"처리 중... {frame_count}/{total_frames} ({int(frame_count/total_frames*100)}%)")

        # 사람 감지 + 추적
        results = person_model.track(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=0.6,
            iou=0.5,
            imgsz=640,
            tracker="configs/botsort.yaml",
            persist=True,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                person_box = (px1, py1, px2, py2)
                track_id = int(box.id[0]) if box.id is not None else -1

                # 사람 영역 crop -> 헬멧 감지
                pad = 20
                crop_y1 = max(0, py1 - pad)
                crop_y2 = min(frame.shape[0], py2 + pad)
                crop_x1 = max(0, px1 - pad)
                crop_x2 = min(frame.shape[1], px2 + pad)
                person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                ppe_results = ppe_model(
                    person_crop,
                    conf=0.5,
                    iou=0.5,
                    verbose=False,
                )

                helmet_detected = False

                for ppe_result in ppe_results:
                    ppe_boxes = ppe_result.boxes
                    if ppe_boxes is None:
                        continue

                    for ppe_box in ppe_boxes:
                        ppe_x1, ppe_y1, ppe_x2, ppe_y2 = map(int, ppe_box.xyxy[0])
                        ppe_class_id = int(ppe_box.cls[0])
                        ppe_class_name = PPE_CLASSES.get(ppe_class_id, "unknown")

                        if ppe_class_name == "unknown":
                            continue

                        abs_ppe_box = (
                            ppe_x1 + crop_x1,
                            ppe_y1 + crop_y1,
                            ppe_x2 + crop_x1,
                            ppe_y2 + crop_y1,
                        )

                        is_worn = check_ppe_in_zone(abs_ppe_box, person_box, ppe_class_name)
                        if is_worn:
                            helmet_detected = True

                        ppe_color = COLOR_GREEN if is_worn else COLOR_ORANGE
                        cv2.rectangle(frame,
                                    (abs_ppe_box[0], abs_ppe_box[1]),
                                    (abs_ppe_box[2], abs_ppe_box[3]),
                                    ppe_color, 2)
                        cv2.putText(frame,
                                   f"helmet: {'O' if is_worn else 'X'}",
                                   (abs_ppe_box[0], abs_ppe_box[1] - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ppe_color, 1)

                if helmet_detected:
                    box_color = COLOR_GREEN
                    status_text = f"ID:{track_id} Helmet OK"
                else:
                    box_color = COLOR_RED
                    status_text = f"ID:{track_id} No Helmet!"

                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                cv2.putText(frame, status_text, (px1, py1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                zone_25 = py1 + int((py2 - py1) * 0.25)
                cv2.line(frame, (px1, zone_25), (px2, zone_25), COLOR_YELLOW, 1)

        # 프레임 저장
        out.write(frame)

    cap.release()
    out.release()
    print("=" * 40)
    print(f"완료! 결과 영상: {OUTPUT_VIDEO}")
    print("=" * 40)


if __name__ == "__main__":
    run()
