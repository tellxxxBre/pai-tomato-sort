"""掴んだ状態の判定テスト ＋ 正解ラベル付きベンチマーク収集(2026-10-27農園版)。

アームを判定ポーズに動かし、あなたが手でトマトをグリッパに挟ませ、
撮影→Claude判定→あなたの目視正解(s=正品/f=非正品)を記録する。
出来上がるもの: logs/benchmark_*.jpg (本番と同一条件のクロップ画像) + logs/benchmark_labels.csv
これが後のClaude vs Jev比較・ローカル判定器蒸留の正解データセットになる。

usage: python tomato_sort/check_inspect.py
繰り返し: 何個でも連続でテストできる(qで終了)
"""
import base64
import csv
import json
import os
import time

import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

HERE = os.path.dirname(__file__)
PORT = "/dev/cu.usbmodem5B7B0094901"  # ★2026-10-27実測: 白フォロワー=0094901 (以前と逆)
LOG_DIR = os.path.join(HERE, "logs")
CSV_PATH = os.path.join(LOG_DIR, "benchmark_labels.csv")

import sys
sys.path.insert(0, HERE)
from tomato_sort import VLM_PROMPT, goto, judge_with_vlm, set_gripper  # 再利用

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY を設定してください")
    import anthropic
    client = anthropic.Anthropic()

    with open(os.path.join(HERE, "poses.json")) as f:
        poses = json.load(f)
    if "inspect" not in poses:
        raise SystemExit("先に judge ポーズを登録してください: python tomato_sort/capture_pose.py inspect")
    gripper_open = poses["home"]["gripper"]
    os.makedirs(LOG_DIR, exist_ok=True)

    rcfg = SOFollowerRobotConfig(
        port=PORT,
        cameras={"overhead": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
        disable_torque_on_disconnect=False,
    )
    rcfg.id = "my_follower"
    robot = SOFollower(rcfg)
    robot.connect()
    print("判定ポーズへ移動します...")
    goto(robot, poses["inspect"], duration=2.5)
    set_gripper(robot, gripper_open)

    n = 0
    while True:
        ans = input("\nトマトをグリッパの指の間に持っていって Enter (qで終了): ")
        if ans.strip().lower() == "q":
            break
        n += 1
        # キャリブレーション基準でゆっくり閉じる (正規化値: 0=閉じ切り, 100=全開)
        print("  グリッパを閉じます(トマトは軽く支えたままで)")
        try:
            g = float(robot.bus.read("Present_Position", "gripper", normalize=True, num_retry=2))
        except Exception:
            g = 60.0
        while g > 2:
            g -= 2  # 小刻みに
            try:
                robot.bus.write("Goal_Position", "gripper", g, normalize=True, num_retry=1)
                time.sleep(0.10)
                c = robot.bus.sync_read("Present_Current", normalize=False, num_retry=1)["gripper"]
                if c > 15:  # ~100mA: 触れたら即停止(トマトを潰さない)
                    g += 2  # 少し緩めて保持
                    robot.bus.write("Goal_Position", "gripper", g, normalize=True, num_retry=1)
                    print(f"  掴みました (電流{c*6.5:.0f}mA, やさしく保持)")
                    break
            except Exception:
                pass
        time.sleep(0.5)
        print("  手を離してください。2秒後に撮影します")
        time.sleep(2.0)

        obs = robot.get_observation()
        img = obs["overhead"]
        stamp = time.strftime("%H%M%S")
        path = os.path.join(LOG_DIR, f"benchmark_{stamp}_{n:03d}.jpg")
        crop_path = os.path.join(LOG_DIR, f"benchmark_{stamp}_{n:03d}_crop.jpg")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        verdict, latency = {}, None
        try:
            t0 = time.time()
            verdict = judge_with_vlm(client, img, crop_save_path=crop_path)
            latency = time.time() - t0
            print(f"  Claude判定: {verdict} ({latency:.1f}秒)")
        except Exception as e:
            print(f"  VLM呼び出し失敗: {str(e)[:100]}")
        # あなたの目視正解を記録(これがground truth)
        gt = ""
        while gt not in ("seihin", "fuhin", "skip"):
            a = input("  あなたの目視判定は? [s=正品 / f=非正品 / x=スキップ]: ").strip().lower()
            gt = {"s": "seihin", "f": "fuhin", "x": "skip"}.get(a, "")
        if gt != "skip":
            new_file = not os.path.exists(CSV_PATH)
            with open(CSV_PATH, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["crop_path", "ground_truth", "claude_label", "claude_held", "claude_latency_s"])
                w.writerow([os.path.basename(crop_path), gt, verdict.get("label"), verdict.get("held"), f"{latency:.2f}" if latency else ""])
            match = "一致✅" if verdict.get("label") == gt else "不一致❌"
            print(f"  記録しました (Claude={verdict.get('label')} vs 正解={gt} → {match})")
        set_gripper(robot, gripper_open)
        print("  (グリッパを開きました。トマトを受け取ってください)")

    print("終了。アームは判定ポーズのまま固定です")
    robot.disconnect()


if __name__ == "__main__":
    main()
