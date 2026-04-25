import cv2
from ultralytics import YOLO

# 탐지할 클래스만 정의
TARGET_CLASSES = {
    0: "person",
    41: "cup",
    74: "clock",
    67: "cell phone",
    73: "book",
}

def run_tracking():
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ROI 구역 설정
    roi_x1, roi_y1 = 150, 100
    roi_x2, roi_y2 = 490, 380

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            classes=list(TARGET_CLASSES.keys()),  # 5가지 클래스만 탐지
            conf=0.6,
            iou=0.5,
            imgsz=640,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        # ROI 기본 색상 (파란색)
        roi_color = (255, 0, 0)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 클래스 ID & 이름
                class_id = int(box.cls[0])
                class_name = TARGET_CLASSES.get(class_id, "unknown")

                # 트래킹 ID
                track_id = int(box.id[0]) if box.id is not None else -1

                if class_id == 0:
                    # 사람 중심점 계산
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # ROI 안에 있는지 확인
                    in_roi = (roi_x1 < cx < roi_x2) and (roi_y1 < cy < roi_y2)

                    # 구역 밖 = 초록색, 구역 안 = 빨간색
                    box_color = (0, 0, 255) if in_roi else (0, 255, 0)

                    # ROI 안에 사람 있으면 ROI 색도 빨간색으로
                    if in_roi:
                        roi_color = (0, 0, 255)

                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # ID + 상태 표시
                    status = "Inside" if in_roi else "Outside"
                    cv2.putText(
                        frame,
                        f"ID:{track_id} {status}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2
                    )

                else:
                    # 일상 물체 → 노란색
                    box_color = (0, 255, 255)

                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                    # 물체 이름 + ID 표시
                    cv2.putText(
                        frame,
                        f"{class_name} ID:{track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2
                    )

        # ROI 구역 그리기
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)
        cv2.putText(
            frame,
            "Work Zone",
            (roi_x1, roi_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            roi_color,
            2
        )

        cv2.imshow("PPE Project - Object Tracking", frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking()