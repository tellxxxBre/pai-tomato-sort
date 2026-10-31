"""箱ポーズのマネキン記憶ツール。

トルクOFFのフォロワーアームを手で動かして、各ポーズでEnterを押すと関節角度を記憶する。
- 開始時: アームを手で支えてもらってからトルクを切る(落下防止)
- 終了時: トルクをONに戻し、アームは最後のポーズで固定される(手を離してOK)

記憶するポーズ (2026-10-27農園版・2分類):
  home    : 掴みに行く前の構え(開始位置)。グリッパは開いておく
  transit : 移動の中継点(箱の上空あたりの安全な高い位置)
  seihin  : 製品箱の真上(落とせる位置)
  fuhin   : 非製品箱の真上
  inspect : 判定ポーズ(カメラにトマトを見せる位置)

usage: python tomato_sort/capture_pose.py            # 全ポーズ
       python tomato_sort/capture_pose.py home       # 指定ポーズだけ録り直し
"""
import json
import os
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "/dev/cu.usbmodem5B7B0094901"  # ★2026-10-27実測: 白フォロワー=0094901 (以前と逆)
OUT = os.path.join(os.path.dirname(__file__), "poses.json")

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

bus = FeetechMotorsBus(
    port=PORT,
    motors={name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, name in enumerate(MOTORS)},
)
bus.connect(handshake=True)

print("=" * 60)
input("⚠️  これからトルクを切ります。アームが脱力します。\n"
      "   【片手でアームをしっかり支えてから】Enterを押してください: ")
bus.disable_torque(num_retry=5)
print("トルクOFF。アームを手で動かせます。\n")

poses = {}
if os.path.exists(OUT):
    with open(OUT) as f:
        poses = json.load(f)

steps = [
    ("home", "① 掴みに行く前の【構え(開始位置)】にして、グリッパを【開いて】ください"),
    ("transit", "② 【中継点】(箱の上空あたりの安全な高い位置)にしてください"),
    ("seihin", "③ 【製品箱の真上】(ここでグリッパを開けばトマトが箱に落ちる位置)にしてください"),
    ("fuhin", "④ 【非製品箱の真上】にしてください"),
    ("inspect", "⑤ 【判定ポーズ】カメラ画面の中央あたりで、グリッパ先端がカメラによく見える位置にしてください(トマトを見せるポーズ)"),
]
only = sys.argv[1] if len(sys.argv) > 1 else None
if only:
    steps = [s for s in steps if s[0] == only]
    if not steps:
        raise SystemExit(f"不明なポーズ名: {only} (home/transit/seihin/fuhin/inspect)")

for name, msg in steps:
    input(f"{msg}\n → できたらEnter: ")
    pos = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    poses[name] = pos
    print(f"   記憶しました: {pos}\n")

with open(OUT, "w") as f:
    json.dump(poses, f, indent=2, ensure_ascii=False)
print(f"保存完了: {OUT}")

# トルクを戻して現在位置で固定 → 手を離しても落ちない
cur = bus.sync_read("Present_Position", normalize=False, num_retry=2)
bus.enable_torque(num_retry=3)
for name, val in cur.items():
    bus.write("Goal_Position", name, int(val), normalize=False, num_retry=2)
print("✅ トルクONに戻しました。アームはこの姿勢で固定されています。手を離してOKです。")
