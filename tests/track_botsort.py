import cv2
import numpy as np
from ultralytics import YOLO

# ============================================
# 설정
# ============================================

# 사람 감지용 클래스
PERSON_CLASS_ID = 0

# PPE 클래스 정의 (PPE 전용 모델 학습 후 클래스 ID에 맞게 수정)
PPE_CLASSES = {
    0: "helmet",
    1: "vest",
    2: "gloves",
}

# PPE별 신체 영역 매핑 (사람 bbox 기준 비율)
# top_ratio: 상단 시작 비율, bottom_ratio: 하단 끝 비율
PPE_BODY_ZONES = {
    "helmet": {"top_ratio": 0.0, "bottom_ratio": 0.25},   # 상단 25% (머리)
    "vest":   {"top_ratio": 0.2, "bottom_ratio": 0.65},    # 중상단 20%~65% (상체)
    "gloves": {"top_ratio": 0.55, "bottom_ratio": 1.0},    # 하단 55%~100% (손/하체)
}

# 색상 정의 (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)


def check_ppe_in_zone(ppe_box, person_box, ppe_class_name):
    """
    PPE 바운딩 박스가 사람의 올바른 신체 영역에 있는지 확인
    
    Args:
        ppe_box: PPE 바운딩 박스 (x1, y1, x2, y2) - 원본 프레임 좌표
        person_box: 사람 바운딩 박스 (x1, y1, x2, y2)
        ppe_class_name: PPE 클래스 이름 (helmet, vest, gloves)
    
    Returns:
        bool: 올바른 영역에 있으면 True (착용), 아니면 False
    """
    px1, py1, px2, py2 = person_box
    person_height = py2 - py1

    # PPE 중심점 계산
    ppe_cx = (ppe_box[0] + ppe_box[2]) / 2
    ppe_cy = (ppe_box[1] + ppe_box[3]) / 2

    # PPE가 사람 bbox 안에 있는지 먼저 확인
    if not (px1 <= ppe_cx <= px2):
        return False

    # 신체 영역 범위 계산
    zone = PPE_BODY_ZONES.get(ppe_class_name)
    if zone is None:
        return False

    zone_top = py1 + person_height * zone["top_ratio"]
    zone_bottom = py1 + person_height * zone["bottom_ratio"]

    # PPE 중심점이 해당 영역 안에 있으면 착용으로 판단
    return zone_top <= ppe_cy <= zone_bottom


def run_tracking():
    # ============================================
    # 모델 로드
    # ============================================
    
    # 1) 사람 감지 + 추적용 모델 (사전학습 모델)
    person_model = YOLO("yolov8n.pt")

    # 2) PPE 감지용 모델 (fine-tuning 후 경로 수정)
    # TODO: PPE 데이터셋으로 학습 완료 후 아래 경로를 수정하세요
    # ppe_model = YOLO("models/ppe_best.pt")
    ppe_model = None  # PPE 모델 학습 전까지 None

    # ============================================
    # 카메라 설정
    # ============================================
    cap = cv2.VideoCapture(0)
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # ROI 구역 설정 (비율 기반 - 카메라 해상도 변경에도 대응 가능)
    roi_x1 = int(frame_width * 0.23)
    roi_y1 = int(frame_height * 0.21)
    roi_x2 = int(frame_width * 0.77)
    roi_y2 = int(frame_height * 0.79)

    print("=" * 50)
    print("PPE Tracking System 시작")
    print(f"해상도: {frame_width}x{frame_height}")
    print(f"ROI 영역: ({roi_x1},{roi_y1}) ~ ({roi_x2},{roi_y2})")
    print(f"PPE 모델: {'로드됨' if ppe_model else '미로드 (PPE 감지 비활성)'}")
    print("종료: q 키")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ============================================
        # 1단계: 사람 감지 + 추적 (BoT-SORT + ReID)
        # ============================================
        results = person_model.track(
            source=frame,
            classes=[PERSON_CLASS_ID],       # 사람만 감지
            conf=0.6,
            iou=0.5,
            imgsz=640,
            tracker="configs/botsort.yaml",          # BoT-SORT (ReID 지원)
            persist=True,
            verbose=False,
        )

        # ROI 기본 색상 (파란색)
        roi_color = COLOR_BLUE
        abnormal_detected = False

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # 사람 바운딩 박스 좌표
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                person_box = (px1, py1, px2, py2)

                # 트래킹 ID
                track_id = int(box.id[0]) if box.id is not None else -1

                # 사람 중심점 계산
                cx = (px1 + px2) // 2
                cy = (py1 + py2) // 2

                # ROI 안에 있는지 확인
                in_roi = (roi_x1 < cx < roi_x2) and (roi_y1 < cy < roi_y2)

                # ============================================
                # 2단계: PPE 감지 (사람 영역 crop → PPE 모델)
                # ============================================
                ppe_status = {}  # {"helmet": True/False, "vest": True/False, "gloves": True/False}
                
                if ppe_model is not None and in_roi:
                    # 사람 영역에 여유(padding)를 줘서 crop
                    pad = 20
                    crop_y1 = max(0, py1 - pad)
                    crop_y2 = min(frame.shape[0], py2 + pad)
                    crop_x1 = max(0, px1 - pad)
                    crop_x2 = min(frame.shape[1], px2 + pad)
                    person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    # PPE 모델로 감지
                    ppe_results = ppe_model(
                        person_crop,
                        conf=0.5,
                        iou=0.5,
                        verbose=False,
                    )

                    # 기본값: 모든 PPE 미착용
                    for ppe_name in PPE_CLASSES.values():
                        ppe_status[ppe_name] = False

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

                            # crop 좌표를 원본 프레임 좌표로 변환
                            abs_ppe_box = (
                                ppe_x1 + crop_x1,
                                ppe_y1 + crop_y1,
                                ppe_x2 + crop_x1,
                                ppe_y2 + crop_y1,
                            )

                            # ============================================
                            # 3단계: PPE가 올바른 신체 영역에 있는지 확인
                            # ============================================
                            is_worn = check_ppe_in_zone(abs_ppe_box, person_box, ppe_class_name)
                            ppe_status[ppe_class_name] = is_worn

                            # PPE 바운딩 박스 그리기
                            ppe_color = COLOR_GREEN if is_worn else COLOR_ORANGE
                            cv2.rectangle(frame, 
                                        (abs_ppe_box[0], abs_ppe_box[1]),
                                        (abs_ppe_box[2], abs_ppe_box[3]),
                                        ppe_color, 2)
                            cv2.putText(frame, 
                                       f"{ppe_class_name}: {'O' if is_worn else 'X'}",
                                       (abs_ppe_box[0], abs_ppe_box[1] - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, ppe_color, 1)

                # ============================================
                # 표시 로직
                # ============================================
                
                # 이상 행동 판단: ROI 안에 있는데 PPE 미착용
                missing_ppe = [name for name, worn in ppe_status.items() if not worn]
                is_abnormal = in_roi and len(missing_ppe) > 0 and ppe_model is not None

                if is_abnormal:
                    abnormal_detected = True

                # 사람 바운딩 박스 색상 결정
                if is_abnormal:
                    box_color = COLOR_RED          # 이상 행동 (PPE 미착용)
                elif in_roi:
                    box_color = COLOR_GREEN         # ROI 안 + PPE 정상
                else:
                    box_color = COLOR_WHITE          # ROI 밖

                # 사람 바운딩 박스 그리기
                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)

                # 상태 텍스트 구성
                if in_roi and ppe_model is not None:
                    if is_abnormal:
                        status_text = f"ID:{track_id} ALERT: {', '.join(missing_ppe)} missing"
                    else:
                        status_text = f"ID:{track_id} PPE OK"
                elif in_roi:
                    status_text = f"ID:{track_id} Inside (PPE모델 미로드)"
                else:
                    status_text = f"ID:{track_id} Outside"

                cv2.putText(frame, status_text, (px1, py1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # 신체 영역 가이드라인 표시 (디버그용, ROI 안에 있을 때만)
                if in_roi and ppe_model is not None:
                    person_h = py2 - py1
                    zone_25 = py1 + int(person_h * 0.25)
                    zone_65 = py1 + int(person_h * 0.65)

                    cv2.line(frame, (px1, zone_25), (px2, zone_25), COLOR_YELLOW, 1)
                    cv2.line(frame, (px1, zone_65), (px2, zone_65), COLOR_YELLOW, 1)

        # ROI 구역 그리기
        if abnormal_detected:
            roi_color = COLOR_RED
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)

        roi_label = "Work Zone - WARNING!" if abnormal_detected else "Work Zone"
        cv2.putText(frame, roi_label, (roi_x1, roi_y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

        # 화면 표시
        cv2.imshow("PPE Tracking System", frame)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tracking()