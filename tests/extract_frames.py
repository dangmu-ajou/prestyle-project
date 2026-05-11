import cv2, os

OUTPUT_DIR = r"C:\Users\공다운\OneDrive\바탕 화면\공다운\prestyle-project\data\output"
FRAME_DIR  = os.path.join(OUTPUT_DIR, "frames")
os.makedirs(FRAME_DIR, exist_ok=True)

videos = [
    "gloves_result.mp4",
    "helmet_result.mp4",
    "safety_boots_result.mp4",
    "vest_result.mp4",
]

for vname in videos:
    vpath = os.path.join(OUTPUT_DIR, vname)
    label = vname.replace("_result.mp4", "")
    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{label}: total={total}")

    sample_positions = [int(total * r) for r in [0.1, 0.3, 0.5, 0.7, 0.9]]
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            print(f"  frame {pos}: read failed")
            continue
        out_path = os.path.join(FRAME_DIR, f"{label}_{pos:04d}.jpg")
        success, buf = cv2.imencode(".jpg", frame)
        if success:
            with open(out_path, "wb") as f:
                f.write(buf.tobytes())
            print(f"  saved: {out_path} ({os.path.getsize(out_path)} bytes)")
        else:
            print(f"  encode failed for frame {pos}")
    cap.release()
