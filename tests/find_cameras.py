"""
카메라 인덱스 탐색 스크립트.
0~5번 인덱스를 순서대로 열어 실제 연결된 카메라를 찾는다.
각 카메라 창을 2초씩 띄워주며, 어느 창이 내장/USB인지 확인한다.
"""
import cv2
import time

MAX_INDEX = 6
SHOW_SECONDS = 2  # 각 카메라를 몇 초 동안 보여줄지

print("=" * 50)
print("카메라 인덱스 스캔 시작 (0 ~", MAX_INDEX - 1, ")")
print("각 창을 확인한 뒤 아무 키나 누르면 다음으로 넘어갑니다.")
print("=" * 50)

found = []

for idx in range(MAX_INDEX):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DirectShow 백엔드 (Windows)
    if not cap.isOpened():
        print(f"[인덱스 {idx}] 카메라 없음 (열기 실패)")
        cap.release()
        continue

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"[인덱스 {idx}] 열렸지만 프레임 읽기 실패 — 허상 장치일 수 있음")
        cap.release()
        continue

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[인덱스 {idx}] 카메라 발견!  해상도: {w}x{h}")
    found.append(idx)

    # 창에 인덱스 번호를 크게 표시
    label = f"CAM INDEX = {idx}  ({w}x{h})"
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow(label, frame)
    cv2.waitKey(SHOW_SECONDS * 1000)
    cv2.destroyAllWindows()
    cap.release()

print()
print("=" * 50)
if found:
    print(f"발견된 카메라 인덱스: {found}")
    print("창에서 내장 카메라와 USB 웹캠을 확인하여 인덱스를 메모하세요.")
else:
    print("연결된 카메라를 찾지 못했습니다.")
print("=" * 50)
