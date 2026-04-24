import cv2
from ultralytics import YOLO

def run_test():
    # 학습된 헬멧 탐지 모델 불러오기
    model = YOLO("models/best_helmet.pt")

    # 외장 USB 웹캠 연결 (안되면 2, 3으로 바꿔봐)
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("웹캠 연결 실패! 번호를 바꿔봐 (0, 1, 2...)")
        return

    print("웹캠 연결 성공! q 누르면 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            source=frame,
            conf=0.5,                  # 50% 이상 확신할 때만 탐지
            iou=0.5,
            imgsz=640,
            tracker="bytetrack.yaml",  # 가려져도 추적 유지
            persist=True,              # 프레임 간 ID 유지
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 클래스 이름
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # 신뢰도
                conf = float(box.conf[0])

                # 트래킹 ID
                track_id = int(box.id[0]) if box.id is not None else -1

                # 헬멧 착용 = 초록색, 미착용 = 빨간색
                if "helmet" in class_name.lower() or "hard hat" in class_name.lower():
                    box_color = (0, 255, 0)
                else:
                    box_color = (0, 0, 255)

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # 클래스 이름 + ID + 신뢰도 표시
                cv2.putText(
                    frame,
                    f"{class_name} ID:{track_id} {conf:.0%}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

        cv2.imshow("Helmet Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_test()