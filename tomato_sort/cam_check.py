"""カメラ現在映像を1枚撮って保存するだけ。画角確認用。
usage: python tomato_sort/cam_check.py [camera_index]
"""
import sys
import time

import cv2

idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(idx)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise SystemExit(f"カメラ index={idx} を開けませんでした")
# ウォームアップ(露出・WB安定)
for _ in range(15):
    cap.read()
    time.sleep(0.03)
ok, frame = cap.read()
cap.release()
if not ok:
    raise SystemExit("フレーム取得に失敗しました")
out = "cam_check.jpg"
cv2.imwrite(out, frame)
print(f"saved: {out} ({frame.shape[1]}x{frame.shape[0]})")
