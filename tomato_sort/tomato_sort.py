"""トマト選果 統合ループ (Phase 3):
  ACTが掴んで持ち上げ → Claude(VLM)が判定 → ルール表で箱決定 → 記憶ポーズ再生で箱へ → 開いて戻る

前提:
  - tomato_sort/poses.json が capture_pose.py で作成済み
  - 環境変数 ANTHROPIC_API_KEY が設定済み
  - アームを home 付近の構えにしてから起動する

usage: python tomato_sort/tomato_sort.py [--cycles 5] [--grasp-seconds 15]
中断: Ctrl+C (アームはその場で固まる)
"""
import argparse
import base64
import csv
import json
import os
import threading
import time
from copy import copy
from contextlib import nullcontext

import cv2
import numpy as np
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

HERE = os.path.dirname(__file__)
PORT = "/dev/cu.usbmodem5B7B0094901"  # ★2026-10-27実測: 白フォロワー=0094901 (以前と逆)
POLICY_PATH = "outputs/train/act_tomato_crate_40ep/checkpoints/last/pretrained_model"  # 2026-10-27農園・コンテナ直取り40本学習
TASK = "pick up a tomato from the green crate and lift it up"  # 学習時のsingle_taskと一致させる
VLM_MODEL = "claude-fable-5"
LOG_DIR = os.path.join(HERE, "logs")

# 判定ラベル → 箱ポーズ名 (ルール表: ここを書き換えれば振り分け変更)
# 2026-10-27農園版: 2分類(製品/非製品)。グリーンバック検出は今回スコープ外
LABEL_TO_BOX = {
    "seihin": "seihin",  # 製品(出荷可) → 製品箱
    "fuhin": "fuhin",    # 非製品(未熟・傷み) → 非製品箱
}

VLM_PROMPT = """この画像はロボットアームのグリッパ周辺を切り出して拡大したものです。グリッパ(白い樹脂の指)が画像内にあり、ミニトマト(細長いプラム型)を掴んで持ち上げている場合があります。掴んでいるかを確認し、掴んでいる場合はそのトマトを2分類してください。トマトの一部がグリッパの指に隠れている場合もありますが、見えている部分で判断してください。

重要(このカメラ固有の色のクセ・2026-10-27実測21枚で確認済み):
- 本当に完熟した赤いトマトは【濃いオレンジ〜赤】(色が濃く飽和)に写る
- 実物が未熟(緑〜黄緑)のトマトは【黄色〜飴色〜淡いオレンジ】に写る
- つまり「オレンジっぽい=熟している」と判断してはいけない。色の濃さ・赤みで見分けること

分類基準(2択)。判断軸は【色の濃さではなく色相(黄色寄りか赤寄りか)】:
- seihin(製品): 色相がオレンジ〜赤(赤み寄り)。オレンジの濃淡は問わない(中くらいの明るさのオレンジも製品)
- fuhin(非製品): 色相が黄色寄り(黄色・マスタード・飴色・黄みの強い橙)、または一部にオリーブ色〜緑の混じりがある、または表面にシワ・黒ずみ・傷みがある
- 迷った時の規則: 「黄色かオレンジか」で迷ったら fuhin(出荷しない安全側)。「薄いオレンジか濃いオレンジか」で迷ったら(黄色味がないなら) seihin

必ず次のJSONだけを出力してください(説明文なし):
{"held": true/false, "label": "seihin"|"fuhin"|null, "reason": "10文字程度"}
heldは「グリッパがトマトを保持して宙に浮いているか」。箱の中に置かれているトマトは無視してください。"""


def clear_overload(robot, name: str):
    """過負荷保護でロックしたモーターをトルク入れ直しで復帰させる"""
    print(f"  (モーター{name}の過負荷を検知 → 自動復帰)")
    try:
        robot.bus.write("Torque_Enable", name, 0, normalize=False, num_retry=3)
        time.sleep(0.15)
        robot.bus.write("Torque_Enable", name, 1, normalize=False, num_retry=3)
        time.sleep(0.1)
    except Exception as e:
        print(f"  (復帰失敗: {str(e)[:60]})")


def safe_write_goal(robot, name: str, val: int):
    """Goal_Position書き込み。Overloadなら復帰させ、押し付けをやめて現在位置で保持"""
    try:
        robot.bus.write("Goal_Position", name, val, normalize=False, num_retry=1)
    except RuntimeError as e:
        if "Overload" in str(e):
            clear_overload(robot, name)
            try:
                # 同じ目標に再度押し付けない。今いる位置を目標にして無理をやめる
                cur = robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)[name]
                robot.bus.write("Goal_Position", name, int(cur), normalize=False, num_retry=2)
            except Exception:
                pass
        else:
            raise


def goto(robot, target: dict, duration: float = 2.0, hold_gripper: float | None = None, hz: int = 25):
    """現在位置から target(生tick) へゆっくり補間移動。hold_gripper指定時はグリッパ値を固定。"""
    cur = robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)
    tgt = dict(target)
    if hold_gripper is not None:
        tgt["gripper"] = hold_gripper
    steps = max(2, int(duration * hz))
    for s in range(1, steps + 1):
        a = s / steps
        for name in cur:
            val = int(round(cur[name] + (tgt[name] - cur[name]) * a))
            safe_write_goal(robot, name, val)
        time.sleep(1.0 / hz)


def lift_up_in_place(robot, poses: dict, hold_gripper: float | None = None, duration: float = 1.5):
    """土台(pan)は今の向きのまま、腕だけtransitの高さまで真上に持ち上げる。
    コンテナの中や低い位置から移動する前に呼ぶことで、縁や山を薙ぐ横移動を防ぐ。"""
    cur = robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)
    liftup = dict(poses["transit"])
    liftup["shoulder_pan"] = cur["shoulder_pan"]  # 向きは変えない
    goto(robot, liftup, duration=duration, hold_gripper=hold_gripper)


def crane_goto(robot, poses: dict, target: dict, hold_gripper: float | None = None):
    """クレーン式移動: ①その場で持ち上げ ②高い姿勢のまま土台(pan)だけ目的方向へ回す ③降ろす。
    台本移動はすべてこの型に統一(斜めに薙ぐ関節補間を構造的に排除)。"""
    lift_up_in_place(robot, poses, hold_gripper=hold_gripper, duration=1.2)
    high = dict(poses["transit"])
    high["shoulder_pan"] = target["shoulder_pan"]
    goto(robot, high, duration=1.2, hold_gripper=hold_gripper)
    goto(robot, target, duration=1.5, hold_gripper=hold_gripper)


def set_gripper(robot, value: float, duration: float = 0.8, hz: int = 25):
    cur = robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)["gripper"]
    steps = max(2, int(duration * hz))
    for s in range(1, steps + 1):
        a = s / steps
        val = int(round(cur + (value - cur) * a))
        safe_write_goal(robot, "gripper", val)
        time.sleep(1.0 / hz)


# 判定ポーズ到達時にグリッパ+トマトが写る領域 (640x480基準)
# 2026-10-27農園で再計測: inspectポーズ実写からグリッパ先端+余白で決定(背景=白箱と床、トマト山は除外)
CROP_X0, CROP_X1, CROP_Y0, CROP_Y1 = 170, 440, 60, 285


def judge_with_vlm(client, image_rgb: np.ndarray, crop_save_path: str | None = None) -> dict:
    """グリッパ周辺を切り出し拡大してClaudeに投げ {held, label, reason} を得る"""
    crop = image_rgb[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    if crop_save_path:
        cv2.imwrite(crop_save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64 = base64.standard_b64encode(buf.tobytes()).decode()
    msg = client.messages.create(
        model=VLM_MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"}, output_config={"effort": "high"},  # じっくり観察させる(最高精度)
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }],
    )
    text = next(b.text for b in msg.content if getattr(b, "type", "") == "text")
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


# --- Jev(TypeSafe)判定: --judge jev で使用。Claudeと同一クロップ・同一基準の英訳版 ---
JEV_INSTRUCTIONS = (
    "The image `img` shows a robot gripper (white plastic fingers) possibly holding a plum-type "
    "mini tomato lifted in the air. This camera desaturates colors: a truly ripe red tomato "
    "appears deep orange to red, while an unripe green-yellow tomato appears yellow, amber, or "
    "pale yellowish-orange. Judge by HUE (yellowish vs reddish), not by brightness. "
    "Ignore any tomatoes lying in crates or boxes in the background."
)
JEV_HELD_THRESHOLD = 0.35  # ベンチ21枚(全部保持あり)でheld確率0.38〜0.88だったため低めに設定
_jev_client = None


def judge_with_jev(image_rgb: np.ndarray, crop_save_path: str | None = None) -> dict:
    """Claudeと同じクロップをJevに投げ {held, label, reason} を返す"""
    from pathlib import Path
    from typesafe_sdk import Choice, Image, Noul, TypeSafeClient
    global _jev_client
    if _jev_client is None:
        _jev_client = TypeSafeClient()
        _jev_client.__enter__()
    crop = image_rgb[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    if crop_save_path:
        cv2.imwrite(crop_save_path, bgr)
    tmp = os.path.join(LOG_DIR, "_jev_live_tmp.jpg")
    cv2.imwrite(tmp, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    resp = _jev_client.system_one(
        model="jev-latest",
        state={"img": Image(Path(tmp))},
        questions={
            "held": Noul(instructions="Is the gripper in `img` holding a tomato lifted in the air? "
                                      "(not tomatoes lying in the crate or boxes)"),
            "label": Choice(
                instructions=JEV_INSTRUCTIONS + " Which class is the HELD tomato?",
                criteria={
                    "seihin": "Product grade: the held tomato's hue is orange to red (reddish). "
                              "Medium or light orange still counts if there is no yellow tint.",
                    "fuhin": "Non-product: the held tomato looks yellow, mustard, amber, or yellowish-orange, "
                             "or has olive/green patches, or shows wrinkles, dark spots, or damage. "
                             "If unsure between yellowish and orange, choose this.",
                },
            ),
        },
        extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
    )
    held_p = resp.nouls["held"].noul
    return {"held": held_p >= JEV_HELD_THRESHOLD,
            "label": resp.choices["label"].choice,
            "reason": f"jev held_p={held_p:.2f}"}


# --- HSVルール判定: --judge hsv。学習もAPIも不要の古典CV ---
# ベンチ21枚で実測校正(2026-10-27): トマト画素の中央色相が 正品10-17 / 非正品19-26 と完全分離
# 保持検出: トマト画素数が 空645 / 保持あり3484-5655 → しきい値2000
HSV_HUE_CUTOFF = 18.0
HSV_MIN_PIXELS = 2000


def judge_with_hsv(image_rgb: np.ndarray, crop_save_path: str | None = None) -> dict:
    """彩度の高い暖色画素をトマトとみなし、中央色相で2分類する(0.001秒・0円)"""
    crop = image_rgb[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    if crop_save_path:
        cv2.imwrite(crop_save_path, bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (s > 100) & (v > 60) & ((h < 40) | (h > 160))
    n = int(mask.sum())
    if n < HSV_MIN_PIXELS:
        return {"held": False, "label": None, "reason": f"hsv px={n}"}
    hues = h[mask].astype(float)
    hues[hues > 160] -= 180  # 赤の折り返し
    med = float(np.median(hues))
    label = "seihin" if med <= HSV_HUE_CUTOFF else "fuhin"
    return {"held": True, "label": label, "reason": f"hsv hue={med:.0f} px={n}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--grasp-seconds", type=float, default=15.0)
    ap.add_argument("--judge", choices=["claude", "jev", "hsv"], default="claude",
                    help="判定エンジン: claude(VLM) / jev(TypeSafe) / hsv(古典CVルール・API不要)")
    ap.add_argument("--policy-path", default=POLICY_PATH,
                    help="掴みポリシーのパス(ACT40/ACT100/SmolVLA切替用)")
    ap.add_argument("--model-name", default=None, help="評価ログに書くモデル名(省略時はパスから推定)")
    ap.add_argument("--eval", action="store_true",
                    help="評価モード: サイクルごとに目視正解を聞き logs/eval_runs.csv に記録")
    ap.add_argument("--task", default=TASK,
                    help="ポリシーに渡す言語指示(SmolVLAの言語反応実験用。省略時は学習時と同じ文)")
    args = ap.parse_args()
    if args.model_name is None:
        parts = [p for p in args.policy_path.split(os.sep) if p]
        args.model_name = next((p for p in parts if "crate" in p or "smolvla" in p or "act" in p), parts[-1])

    if args.judge == "hsv":
        judge_fn = lambda img, crop_path: judge_with_hsv(img, crop_save_path=crop_path)
        print("判定エンジン: HSVルール(古典CV)")
    elif args.judge == "jev":
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise SystemExit("環境変数 TYPESAFE_API_KEY を設定してください")
        judge_fn = lambda img, crop_path: judge_with_jev(img, crop_save_path=crop_path)
        print("判定エンジン: Jev (TypeSafe)")
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("環境変数 ANTHROPIC_API_KEY を設定してください (export ANTHROPIC_API_KEY=...)")
        import anthropic
        client = anthropic.Anthropic()
        judge_fn = lambda img, crop_path: judge_with_vlm(client, img, crop_save_path=crop_path)
        print("判定エンジン: Claude")

    with open(os.path.join(HERE, "poses.json")) as f:
        poses = json.load(f)
    gripper_open = poses["home"]["gripper"]
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- 起動ジャンプ対策 ---
    # トルクON瞬間にサーボが「前回の古い目標位置」へ跳ねる(ブイーン)のを防ぐ:
    # 接続前に Goal_Position := 現在位置 を全モーターに書き込んで同期しておく
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
    print("起動ジャンプ対策: 目標位置を現在位置に同期しました")

    # --- ロボット ---
    rcfg = SOFollowerRobotConfig(
        port=PORT,
        cameras={"overhead": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)},
        disable_torque_on_disconnect=False,
    )
    rcfg.id = "my_follower"
    robot = SOFollower(rcfg)
    robot.connect()
    print("robot connected")

    # --- ポリシー ---
    print(f"掴みポリシー: {args.model_name} ({args.policy_path})")
    pcfg = PreTrainedConfig.from_pretrained(args.policy_path)
    pcfg.pretrained_path = args.policy_path
    pcfg.device = "mps"
    policy_cls = get_policy_class(pcfg.type)
    if getattr(pcfg, "use_peft", False):
        # LoRA(PEFT)チェックポイント: 土台モデル(例: lerobot/smolvla_base)を読み込み、
        # アダプタを合成(merge)して素のポリシーに戻す(推論オーバーヘッドなし)
        from peft import PeftConfig, PeftModel
        peft_cfg = PeftConfig.from_pretrained(args.policy_path)
        print(f"  PEFTアダプタ検出 → 土台 {peft_cfg.base_model_name_or_path} に合成します")
        policy = policy_cls.from_pretrained(peft_cfg.base_model_name_or_path, config=pcfg)
        policy = PeftModel.from_pretrained(policy, args.policy_path, config=peft_cfg)
        policy = policy.merge_and_unload()
    else:
        policy = policy_cls.from_pretrained(args.policy_path, config=pcfg)
    policy.to("mps").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=pcfg, pretrained_path=args.policy_path, dataset_stats=None,
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

    # 起動時: クレーン式(持ち上げ→回転→降ろす)でゆっくり構え(home)へ
    print("構え(home)へゆっくり移動します...")
    crane_goto(robot, poses, poses["home"])
    print("構え完了")

    # 評価モード: 1サイクル1行で logs/eval_runs.csv に記録(モデル比較の生データ)
    EVAL_CSV = os.path.join(LOG_DIR, "eval_runs.csv")

    def log_eval(cycle, outcome, label, verdict, grasp_s, judge_s, cycle_s):
        if not args.eval:
            return
        a = input("  目視正解は? [s=正品/f=非正品/x=トマト無し/Enter=skip]: ").strip().lower()
        gt = {"s": "seihin", "f": "fuhin", "x": "none"}.get(a, "")
        new = not os.path.exists(EVAL_CSV)
        with open(EVAL_CSV, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "model", "judge", "cycle", "outcome", "grasp_s", "judge_s",
                            "held", "label", "gt", "reason", "cycle_s"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), args.model_name, args.judge, cycle,
                        outcome, f"{grasp_s:.1f}", f"{judge_s:.1f}" if judge_s is not None else "",
                        verdict.get("held"), label or "", gt, verdict.get("reason", ""), f"{cycle_s:.1f}"])

    results = []
    try:
        for cycle in range(1, args.cycles + 1):
            if args.eval:
                input(f"\n=== 評価サイクル {cycle}/{args.cycles} [{args.model_name} / {args.judge}] ===\n"
                      f" コンテナの状態を軽く変えて(回転 or かき混ぜ) Enter: ")
            else:
                input(f"\n=== サイクル {cycle}/{args.cycles} === トマトを得意ゾーンに置いて Enter: ")
            cycle_t0 = time.time()

            # --- ① ACTで掴む ---
            policy.reset(); pre.reset(); post.reset()
            t0 = time.time()
            FPS = 30.0  # 学習時と同じ制御周期に厳密に合わせる(速すぎると暴走する)
            SOFT_START_S = 2.0       # 開始直後はACT指令に徐々に主導権を渡す(いきなり飛ばない)
            STALL_CURRENT_RAW = 70   # 約455mA: これ以上の電流が続いたら押し付けと判定
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
                        ob = prepare_observation_for_inference(copy(frame), device, args.task, robot.name)
                        ob = pre(ob)
                        act = policy.select_action(ob)
                        act = post(act)
                    act = act.squeeze(0).cpu()
                    action_dict = {k: act[i].item() for i, k in enumerate(action_keys)}
                    # ソフトスタート: 最初の2秒は現在姿勢→ACT指令へ滑らかにブレンド
                    t_since = time.time() - t0
                    if t_since < SOFT_START_S:
                        a = t_since / SOFT_START_S
                        for k in action_dict:
                            if k in obs:
                                action_dict[k] = obs[k] + a * (action_dict[k] - obs[k])
                    processed = robot_action_proc((action_dict, obs))
                    robot.send_action(processed)
                    # 押し付け監視: 約0.27秒ごとに電流を見て、1秒以上の強い押し付けで中断
                    frame_i += 1
                    if frame_i % 8 == 0:
                        try:
                            cur_all = robot.bus.sync_read("Present_Current", normalize=False, num_retry=1)
                            hot = [n for n, v in cur_all.items() if v > STALL_CURRENT_RAW]
                            if hot:
                                if stall_since is None:
                                    stall_since = time.time()
                                elif time.time() - stall_since > STALL_SECONDS:
                                    print(f"  ⚠️ 押し付け検知({','.join(hot)}) → 掴みを中断して引き上げます")
                                    break
                            else:
                                stall_since = None
                        except Exception:
                            pass
                except RuntimeError as e:
                    if "Overload" in str(e):
                        # どのモーターか特定してその場で復帰
                        import re
                        m = re.search(r"id_=(\d)", str(e))
                        if m:
                            clear_overload(robot, ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll","gripper"][int(m.group(1))-1])
                        time.sleep(0.1)
                    else:
                        raise
                elapsed = time.perf_counter() - tick
                time.sleep(max(0.0, 1.0 / FPS - elapsed))
            grasp_s = time.time() - t0
            print(f"  掴みフェーズ終了({grasp_s:.1f}秒) → 判定ポーズへ移動して判定します")

            # --- ② 判定ポーズで撮影 → 判定と移動を並行 ---
            hold_now = robot.bus.sync_read("Present_Position", normalize=False, num_retry=2)["gripper"]
            # クレーン式でコンテナから真上に抜いて判定ポーズへ
            crane_goto(robot, poses, poses["inspect"], hold_gripper=hold_now)
            time.sleep(0.6)
            obs = robot.get_observation()
            img = obs["overhead"]
            stamp = time.strftime("%H%M%S")
            crop_path = os.path.join(LOG_DIR, f"cycle{cycle}_{stamp}_crop.jpg")

            # 判定APIを別スレッドで開始し、待たずに中継点へ動き出す
            holder = {}
            def _judge():
                try:
                    holder["v"] = judge_fn(img, crop_path)
                except Exception as e:
                    holder["e"] = e
            t_judge0 = time.time()
            th = threading.Thread(target=_judge, daemon=True)
            th.start()
            print("  判定中...(判定ポーズのまま待機)")
            th.join()  # 判定が終わるまでその場で待つ(余計な移動をしない)
            judge_s = time.time() - t_judge0
            if "e" in holder:
                print(f"  VLM呼び出し失敗: {str(holder['e'])[:100]} → 規格外扱い")
                verdict = {"held": True, "label": "fuhin", "reason": "judge error"}
            else:
                verdict = holder["v"]
            cv2.imwrite(os.path.join(LOG_DIR, f"cycle{cycle}_{stamp}_{verdict.get('label')}.jpg"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"  VLM判定({judge_s:.1f}秒): {verdict}")

            # --- ③ 箱へ (クレーン式) ---
            if not verdict.get("held"):
                print("  → 掴めていない。構えに戻って開きます")
                crane_goto(robot, poses, poses["home"], hold_gripper=hold_now)
                set_gripper(robot, gripper_open)
                results.append((cycle, "MISS", None))
                log_eval(cycle, "MISS", None, verdict, grasp_s, judge_s, time.time() - cycle_t0)
                continue

            label = verdict.get("label") or "fuhin"  # 判定不能は安全側(非製品=出荷しない)へ
            box = LABEL_TO_BOX.get(label, "fuhin")
            print(f"  → {label} と判定 → {box} 箱へ")
            crane_goto(robot, poses, poses[box], hold_gripper=hold_now)
            set_gripper(robot, gripper_open)   # 落とす
            time.sleep(0.4)
            crane_goto(robot, poses, poses["home"])
            results.append((cycle, "OK", label))
            log_eval(cycle, "OK", label, verdict, grasp_s, judge_s, time.time() - cycle_t0)

    except KeyboardInterrupt:
        print("\n中断。アームは保持状態のままです")
    finally:
        print("\n=== 結果 ===")
        for c, st, lb in results:
            print(f"  cycle{c}: {st} {lb or ''}")
        ok = sum(1 for _, st, _ in results if st == "OK")
        print(f"  通し成功: {ok}/{len(results)}")
        robot.disconnect()


if __name__ == "__main__":
    main()
