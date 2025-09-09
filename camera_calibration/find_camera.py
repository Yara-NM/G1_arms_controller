import cv2

print("Probing /dev/video[0–7]...\n")
for i in range(8):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"✅ /dev/video{i} | shape: {frame.shape} | dtype: {frame.dtype}")
        else:
            print(f"⚠️  /dev/video{i} opened but failed to return frame")
        cap.release()
    else:
        print(f"❌ /dev/video{i} could neot be opened")
