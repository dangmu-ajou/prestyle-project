import cv2
import threading
import queue
import numpy as np
from ultralytics import YOLO

# ============================================
# 설정
# ============================================

PERSON_CLASS_ID = 0
CAM_WIDTH  = 640
CAM_HEIGHT = 480

# 색상 (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)


# ============================================
# 카메라 스레드
# ============================================

class CameraThread(threading.Thread):
    """
    캡처 루프와 추론 루프를 분리하여 프레임 끊김 방지.
    - 캡처 루프: 항상 최신 프레임만 queue에 유지 (maxsize=1, 오래된 프레임 드롭)
    - 추론 루프: queue에서 프레임을 꺼내 ByteTrack 추적 수행
    """

    def __init__(self, cam_index: int, cam_label: str, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cam_index  = cam_index
        self.cam_label  = cam_label
        self.stop_event = stop_event

        # maxsize=1: 추론이 느려도 항상 최신 프레임만 유지
        self._capture_queue = queue.Queue(maxsize=1)

        self._result_frame = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
        self._lock = threading.Lock()

        self.model = YOLO("yolov8n.pt")

    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 버퍼 최소화

        if not cap.isOpened():
            print(f"[{self.cam_label}] 카메라({self.cam_index})를 열 수 없습니다.")
            self.stop_event.set()
            return

        print(f"[{self.cam_label}] 카메라({self.cam_index}) 시작")

        # 캡처 전용 스레드 (항상 최신 프레임 유지)
        capture_thread = threading.Thread(target=self._capture_loop, args=(cap,), daemon=True)
        capture_thread.start()

        # 추론 루프 (현재 스레드에서 실행)
        self._inference_loop()

        cap.release()
        capture_thread.join()
        print(f"[{self.cam_label}] 종료")

    def _capture_loop(self, cap):
        """캡처만 전담 — 최신 프레임을 queue에 유지"""
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            # 이전 프레임이 아직 처리 안 됐으면 버리고 최신 프레임으로 교체
            if not self._capture_queue.empty():
                try:
                    self._capture_queue.get_nowait()
                except queue.Empty:
                    pass
            self._capture_queue.put(frame)

    def _inference_loop(self):
        """추론 전담 — queue에서 최신 프레임을 꺼내 추적 수행"""
        while not self.stop_event.is_set():
            try:
                frame = self._capture_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            annotated = self._process(frame)
            with self._lock:
                self._result_frame = annotated

    def _process(self, frame):
        results = self.model.track(
            source=frame,
            classes=[PERSON_CLASS_ID],
            conf=0.5,
            iou=0.5,
            imgsz=320,          # 640 → 320으로 축소하여 추론 속도 향상
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                conf     = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id}  {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLOR_GREEN,
                    2,
                )

        cv2.putText(
            frame,
            self.cam_label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLOR_WHITE,
            2,
        )
        return frame

    def get_frame(self):
        with self._lock:
            return self._result_frame.copy()


# ============================================
# 메인
# ============================================

def run():
    stop_event = threading.Event()

    cam0 = CameraThread(cam_index=0, cam_label="CAM 0 (내장)", stop_event=stop_event)
    cam1 = CameraThread(cam_index=1, cam_label="CAM 1 (USB)",  stop_event=stop_event)

    cam0.start()
    cam1.start()

    print("=" * 50)
    print("듀얼 카메라 사람 추적 시작 (ByteTrack)")
    print("종료: q 키")
    print("=" * 50)

    while not stop_event.is_set():
        frame0 = cam0.get_frame()
        frame1 = cam1.get_frame()

        combined = np.hstack([frame0, frame1])
        cv2.imshow("Dual Camera Tracking", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()

    cam0.join()
    cam1.join()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
