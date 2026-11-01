#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手首カメラ検証の掴み評価。学習済みACTで「ばら積みから1個掴んで持ち上げ」をN回試行し、
成功/失敗を目視で記録して成功率を出す(判定・仕分けはしない)。

usage:
  python tomato_sort/eval_grasp.py --model overhead   # 俯瞰のみモデル
  python tomato_sort/eval_grasp.py --model 2cam       # 俯瞰+手首モデル
  （任意）--n 15 --grasp-seconds 12 --step 050000

公平性のため:
- 2カメラ/俯瞰のどちらでも必ず両方のカメラをUSBに挿したまま評価する(indexを固定するため)。
  俯瞰のみモデルは overhead(index1) だけ読む。手首を抜くとindexがズレるので抜かない。
- 同じ山・似た難度で毎回組み直し、A(2cam)とB(overhead)を同条件で回す。
操作: 各試行で山を整えてEnter → 約grasp_seconds秒ポリシーが動く → 持ち上げたら y、ダメなら n。
"""
import argparse, csv, json, os, re, select, sys, time
from copy import copy
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor.factory import make_default_processors
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tomato_sort import goto, lift_up_in_place, crane_goto, set_gripper, clear_overload  # noqa: E402

PORT = "/dev/cu.usbmodem5B7B0094901"  # フォロワー(シリアル固定)
TASK = "pick up a tomato from the green crate and lift it up"  # 学習時のsingle_taskと一致
LOG_DIR = os.path.join(HERE, "logs")

# ★家の実測index(2026-11-01): 俯瞰=1, 手首=0 (index2=MacBook内蔵)
def cameras_for(model):
    oh = OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)
    wr = OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)
    return {"overhead": oh} if model == "overhead" else {"overhead": oh, "wrist": wr}

def policy_path_for(model, step):
    return f"outputs/train/act_tomato_{model}_48ep/checkpoints/{step}/pretrained_model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["overhead", "2cam"], required=True)
    ap.add_argument("--n", type=int, default=15, help="試行回数")
    ap.add_argument("--grasp-seconds", type=float, default=30.0,
                    help="1試行の最大秒数(デモ中央値19s/最長45s)。掴めたらEnterで早期終了可")
    ap.add_argument("--step", default="050000", help="チェックポイントのstepフォルダ名")
    args = ap.parse_args()

    policy_path = policy_path_for(args.model, args.step)
    if not os.path.isdir(policy_path):
        raise SystemExit(f"チェックポイントが無い: {policy_path}\n  Driveからダウンロードして配置してください")
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(os.path.join(HERE, "poses.json")) as f:
        poses = json.load(f)
    gripper_open = poses["home"]["gripper"]

    # --- 起動ジャンプ対策(トルクON時に古い目標へ跳ねるのを防ぐ) ---
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    _names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    _bus = FeetechMotorsBus(port=PORT, motors={n: Motor(i + 1, "sts3215", MotorNormMode.DEGREES)
                                               for i, n in enumerate(_names)})
    _bus.connect(handshake=False)
    _cur = _bus.sync_read("Present_Position", normalize=False, num_retry=2)
    for _n, _v in _cur.items():
        _bus.write("Goal_Position", _n, int(_v), normalize=False, num_retry=2)
    _bus.disconnect(disable_torque=False)

    # --- ロボット ---
    rcfg = SOFollowerRobotConfig(port=PORT, cameras=cameras_for(args.model),
                                 disable_torque_on_disconnect=False)
    rcfg.id = "my_follower"
    robot = SOFollower(rcfg)
    robot.connect()
    print(f"robot connected (cameras={list(cameras_for(args.model).keys())})")

    # --- ポリシー ---
    print(f"掴みポリシー[{args.model}]: {policy_path}")
    pcfg = PreTrainedConfig.from_pretrained(policy_path)
    pcfg.pretrained_path = policy_path
    pcfg.device = "mps"
    policy_cls = get_policy_class(pcfg.type)
    policy = policy_cls.from_pretrained(policy_path, config=pcfg)
    policy.to("mps").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=policy_path, dataset_stats=None,
        preprocessor_overrides={"device_processor": {"device": "mps"}},
    )
    _t, robot_action_proc, robot_obs_proc = make_default_processors()
    features = combine_feature_dicts(
        hw_to_dataset_features(robot.observation_features, "observation", True),
        hw_to_dataset_features(robot.action_features, "action", True),
    )
    action_keys = features["action"]["names"]
    device = torch.device("mps")
    print("policy loaded")

    print("構え(home)へゆっくり移動します...")
    crane_goto(robot, poses, poses["home"])

    EVAL_CSV = os.path.join(LOG_DIR, "grasp_eval_runs.csv")
    results = []
    try:
        for i in range(1, args.n + 1):
            input(f"\n=== 試行 {i}/{args.n} [{args.model}] === 山を組み直して(毎回位置を散らす) Enter: ")
            print(f"  ポリシー開始(最大{args.grasp_seconds:.0f}秒)。掴んで持ち上げたら【Enter】で即終了:")
            # --- ACTで掴む(農園tomato_sortと同一の制御ループ) ---
            policy.reset(); pre.reset(); post.reset()
            t0 = time.time()
            FPS = 30.0
            SOFT_START_S = 2.0
            STALL_CURRENT_RAW = 70
            STALL_SECONDS = 1.0
            stall_since = None
            frame_i = 0
            while time.time() - t0 < args.grasp_seconds:
                tick = time.perf_counter()
                try:
                    obs = robot.get_observation()
                    obs_p = robot_obs_proc(obs)
                    frame = build_dataset_frame(features, obs_p, prefix=OBS_STR)
                    with torch.inference_mode():
                        ob = prepare_observation_for_inference(copy(frame), device, TASK, robot.name)
                        ob = pre(ob)
                        act = policy.select_action(ob)
                        act = post(act)
                    act = act.squeeze(0).cpu()
                    action_dict = {k: act[i2].item() for i2, k in enumerate(action_keys)}
                    t_since = time.time() - t0
                    if t_since < SOFT_START_S:
                        a = t_since / SOFT_START_S
                        for k in action_dict:
                            if k in obs:
                                action_dict[k] = obs[k] + a * (action_dict[k] - obs[k])
                    processed = robot_action_proc((action_dict, obs))
                    robot.send_action(processed)
                    frame_i += 1
                    if frame_i % 8 == 0:
                        try:
                            cur_all = robot.bus.sync_read("Present_Current", normalize=False, num_retry=1)
                            hot = [n for n, v in cur_all.items() if v > STALL_CURRENT_RAW]
                            if hot:
                                if stall_since is None:
                                    stall_since = time.time()
                                elif time.time() - stall_since > STALL_SECONDS:
                                    print(f"  ⚠️ 押し付け検知({','.join(hot)}) → 中断")
                                    break
                            else:
                                stall_since = None
                        except Exception:
                            pass
                except RuntimeError as e:
                    if "Overload" in str(e):
                        m = re.search(r"id_=(\d)", str(e))
                        if m:
                            clear_overload(robot, _names[int(m.group(1)) - 1])
                        time.sleep(0.1)
                    else:
                        raise
                # 掴んで持ち上げた瞬間にユーザーがEnter → その場で早期終了(収録の→と同じ)
                if select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    print("  (Enter: 掴みフェーズ終了)")
                    break
                elapsed = time.perf_counter() - tick
                time.sleep(max(0.0, 1.0 / FPS - elapsed))
            grasp_s = time.time() - t0

            # 掴んだものを真上に持ち上げて目視確認しやすくする
            hold_now = robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)["gripper"]
            lift_up_in_place(robot, poses, hold_gripper=hold_now, duration=1.2)

            a = ""
            while a not in ("y", "n"):
                a = input(f"  掴んで持ち上げた? [y=成功 / n=失敗]: ").strip().lower()
            ok = (a == "y")
            results.append(ok)
            new = not os.path.exists(EVAL_CSV)
            with open(EVAL_CSV, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["date", "model", "step", "trial", "success", "grasp_s"])
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), args.model, args.step, i,
                            "1" if ok else "0", f"{grasp_s:.1f}"])
            n_ok = sum(results)
            print(f"  → {'成功' if ok else '失敗'} (ここまで {n_ok}/{len(results)})")

            # 次の試行へ: グリッパを開いてhomeへ戻す
            set_gripper(robot, gripper_open, duration=0.8)
            crane_goto(robot, poses, poses["home"])
    finally:
        try:
            crane_goto(robot, poses, poses["home"])
        except Exception:
            pass
        robot.disconnect()

    n_ok = sum(results)
    n = len(results)
    print(f"\n==== 結果 [{args.model}] step={args.step} ====")
    print(f"成功率: {n_ok}/{n} = {100*n_ok/n:.0f}%" if n else "試行なし")
    print(f"ログ: {EVAL_CSV}")


if __name__ == "__main__":
    main()
