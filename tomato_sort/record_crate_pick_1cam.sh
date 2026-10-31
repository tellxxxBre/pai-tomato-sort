#!/bin/bash
# 農園収録(俯瞰1カメラ版): 緑コンテナ(山積み)からトマトを1個掴んで持ち上げる。
# 手首カメラは取付が不安定なため今回は不使用(2026-10-27決定)。
# ポート対応2026-10-27実測: 白フォロワー=0094901, 黒リーダー=0090111
# ★手首カメラのUSBを抜いた状態では俯瞰=index0(実行前にscan_cameras.pyで要確認)
# カバレッジ注意: 狙うトマトをコンテナ全域(左右・手前奥・山の高低)に毎回散らすこと。
# n=保存, r=録り直し。リセット中はキー触らない。
# 実行: bash tomato_sort/record_crate_pick_1cam.sh
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"  # リポジトリルートで実行
export PATH="./.venv/bin:$PATH"  # Rerunビューアを見つけるため
# 起動失敗が残した「エピソード0件の空データセット」だけ自動削除(実データがあれば絶対消さない)
ROOT=$HOME/.cache/huggingface/lerobot/local/so101_tomato_crate_1cam_20261027
if [ -d "$ROOT" ] && [ -z "$(find "$ROOT/data" -name '*.parquet' 2>/dev/null)" ] && [ -z "$(find "$ROOT/videos" -type f 2>/dev/null)" ]; then
  echo "空の残骸データセットを削除して再作成: $ROOT"
  rm -rf "$ROOT"
fi
.venv/bin/lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5B7B0094901 \
  --robot.id=my_follower \
  --robot.cameras="{overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5B7B0090111 \
  --teleop.id=my_leader \
  --dataset.repo_id=local/so101_tomato_crate_1cam \
  --dataset.root=$ROOT \
  --dataset.single_task="pick up a tomato from the green crate and lift it up" \
  --dataset.num_episodes=30 \
  --dataset.fps=30 \
  --dataset.episode_time_s=45 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --resume=true
