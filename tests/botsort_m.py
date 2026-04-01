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
PPE_BODY_ZONES = {
    "helmet": {"top_ratio": 0.0, "bottom_ratio": 0.25},
    "vest":   {"top_ratio": 0.2, "bottom_ratio": 0.65},
    "gloves": {"top_ratio": 0.55, "bottom_ratio": 1.0},
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
    """
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


def put_korean_text(frame, text, position, font_size=0.6, color=(255, 255, 255), thickness=2):
    """
    한글 깨짐 방지를 위한 텍스트 표시 함수
    한글이 포함된 경우 PIL을 사용하고, 영문만 있으면 cv2 사용
    """
    try:
        # 한글 포함 여부 확인
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in text)
        if has_korean:
            from PIL import ImageFont, ImageDraw, Image
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
            font = ImageFont.truetype(font_path, int(font_size * 30))
            # BGR -> RGB 색상 변환
            rgb_color = (color[2], color[1], color[0])
            draw.text(position, text, font=font, fill=rgb_color)
            result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            np.copyto(frame, result)
        else:
            cv2.putText(frame, text, position,
                       cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)
    except Exception:
        # PIL 실패 시 기본 cv2 사용
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, font_size, color, thickness)


def run_tracking():
    # ============================================
    # 모델 로드
    # ============================================
    
    # 1) 사람 감지 + 추적용 모델
    person_model = YOLO("yolov8n.pt")

    # 2) PPE 감지용 모델 (fine-tuning 후 경로 수정)
    # ppe_model = YOLO("models/ppe_best.pt")
    ppe_model = None

    # ============================================
    # 카메라 설정
    # ============================================
    cap = cv2.VideoCapture(0)
    frame_width = 640
    frame_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # ============================================
    # 전체 화면 설정
    # ============================================
    window_name = "PPE Tracking System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("=" * 50)
    print("PPE Tracking System 시작")
    print(f"해상도: {frame_width}x{frame_height}")
    print(f"작업 구역: 화면 전체")
    print(f"PPE 모델: {'로드됨' if ppe_model else '미로드 (PPE 감지 비활성)'}")
    print("종료: q 키 / ESC 키")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ============================================
        # 1단계: 사람 감지 + 추적 (BoT-SORT)
        # ============================================
        results = person_model.track(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=0.5,                              # 0.6 → 0.5로 낮춰서 더 먼 거리의 사람도 감지
            iou=0.5,
            imgsz=640,
            tracker="configs/botsort.yaml",
            persist=True,
            verbose=False,
        )

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

                # 신뢰도(confidence) 값
                conf = float(box.conf[0])

                # ============================================
                # 2단계: PPE 감지 (사람 영역 crop → PPE 모델)
                # ============================================
                ppe_status = {}
                
                if ppe_model is not None:
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

                            abs_ppe_box = (
                                ppe_x1 + crop_x1,
                                ppe_y1 + crop_y1,
                                ppe_x2 + crop_x1,
                                ppe_y2 + crop_y1,
                            )

                            # 3단계: PPE 착용 판단
                            is_worn = check_ppe_in_zone(abs_ppe_box, person_box, ppe_class_name)
                            ppe_status[ppe_class_name] = is_worn

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
                missing_ppe = [name for name, worn in ppe_status.items() if not worn]
                is_abnormal = len(missing_ppe) > 0 and ppe_model is not None

                if is_abnormal:
                    abnormal_detected = True

                # 사람 바운딩 박스 색상 결정
                if is_abnormal:
                    box_color = COLOR_RED
                else:
                    box_color = COLOR_GREEN

                # 사람 바운딩 박스 그리기
                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)

                # 상태 텍스트 구성
                if ppe_model is not None:
                    if is_abnormal:
                        status_text = f"ID:{track_id} ALERT: {', '.join(missing_ppe)} missing"
                    else:
                        status_text = f"ID:{track_id} PPE OK"
                else:
                    status_text = f"ID:{track_id} ({conf:.0%})"

                cv2.putText(frame, status_text, (px1, py1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                # 신체 영역 가이드라인 (디버그용, PPE 모델 있을 때만)
                if ppe_model is not None:
                    person_h = py2 - py1
                    zone_25 = py1 + int(person_h * 0.25)
                    zone_65 = py1 + int(person_h * 0.65)
                    cv2.line(frame, (px1, zone_25), (px2, zone_25), COLOR_YELLOW, 1)
                    cv2.line(frame, (px1, zone_65), (px2, zone_65), COLOR_YELLOW, 1)

        # 상단 상태바 표시
        person_count = sum(1 for r in results for _ in (r.boxes if r.boxes is not None else []))
        status_bar = f"Tracking: {person_count} person(s) detected"
        if abnormal_detected:
            status_bar += " | WARNING: PPE violation"
        cv2.putText(frame, status_bar, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)

        # 화면 표시
        cv2.imshow(window_name, frame)

        # q 또는 ESC 누르면 종료
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_tracking()