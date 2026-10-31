"""登録済みポーズをゆっくり順番に再生して目視確認する。
各ポーズの間は「真上に持ち上げてから移動」するので衝突しない。
おかしいポーズが見つかったら: .venv/bin/python tomato_sort/capture_pose.py <ポーズ名> で録り直し。
usage: .venv/bin/python tomato_sort/verify_poses.py            # 全ポーズ
       .venv/bin/python tomato_sort/verify_poses.py fuhin      # 指定ポーズだけ
"""
import json
import os
import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

HERE = os.path.dirname(__file__)
PORT = "/dev/cu.usbmodem5B7B0094901"  # 白フォロワー
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

with open(os.path.join(HERE, "poses.json")) as f:
    poses = json.load(f)

ORDER = ["home", "transit", "inspect", "seihin", "fuhin"]
only = sys.argv[1] if len(sys.argv) > 1 else None
targets = [only] if only else ORDER
for t in targets:
    if t not in poses:
        raise SystemExit(f"ポーズ未登録: {t}")

bus = FeetechMotorsBus(port=PORT, motors={n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES)
                                          for i, n in enumerate(NAMES)})
bus.connect(handshake=True)

# 起動ジャンプ対策: Goal:=現在位置にしてからトルクON
cur = bus.sync_read("Present_Position", normalize=False, num_retry=2)
for n, v in cur.items():
    bus.write("Goal_Position", n, int(v), normalize=False, num_retry=2)
bus.enable_torque(num_retry=3)
print("トルクON(現在姿勢で保持)")


def goto(target: dict, duration: float = 3.5, hz: int = 25):
    cur = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    steps = max(2, int(duration * hz))
    for s in range(1, steps + 1):
        a = s / steps
        for n in NAMES:
            val = int(round(cur[n] + (target[n] - cur[n]) * a))
            try:
                bus.write("Goal_Position", n, val, normalize=False, num_retry=1)
            except Exception:
                pass
        time.sleep(1.0 / hz)


def lift_up():
    cur = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    up = dict(poses["transit"])
    up["shoulder_pan"] = cur["shoulder_pan"]
    up["gripper"] = cur["gripper"]
    goto(up, duration=2.5)


for name in targets:
    input(f"\nEnterで【{name}】へゆっくり移動します(いつでもCtrl+Cで停止): ")
    lift_up()
    goto(poses[name], duration=3.5)
    print(f"→ {name} に到達。位置・高さ・向きが意図通りか目視してください")
    print(f"   (おかしければ後で: .venv/bin/python tomato_sort/capture_pose.py {name})")

print("\n確認終了。アームは最後のポーズで保持しています")
bus.disconnect(disable_torque=False)
