#!/bin/bash
# 手首カメラ検証用の2カメラ収録: 緑カゴ(ばら積み)からトマトを1個掴んで持ち上げる。掴みのみ(判定はしない)。
# ★カメラindexはOpenCVの列挙順で変わる。環境が変わったら必ず cam_check.py で中身を見てから直すこと。
#   （この検証を行った家の実測例: 俯瞰=index1, 手首=index0。別マシン/別ポートでは変わる）
# ★ロボットのポートはシリアル固定でどのUSBに挿しても不変: フォロワー=0094901, リーダー=0090111（各自の実測値に）
# カバレッジ注意: 狙うトマトをカゴ全域(左右・手前奥・山の高低)に毎回散らすこと。
# 操作(lerobot 0.6.0): →=このエピソードを終了して保存・次へ / ←=録り直し / ESC=収録を終了。リセット中は山を組み直す。
# 実行: bash tomato_sort/record_crate_pick_2cam_home.sh
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
STAMP=$(date +%Y%m%d)
ROOT=$HOME/.cache/huggingface/lerobot/local/so101_tomato_crate_2cam_home_$STAMP
# 起動失敗が残した「エピソード0件の空データセット」だけ自動削除(実データがあれば絶対消さない)
if [ -d "$ROOT" ] && [ -z "$(find "$ROOT/data" -name '*.parquet' 2>/dev/null)" ] && [ -z "$(find "$ROOT/videos" -type f 2>/dev/null)" ]; then
  echo "空の残骸データセットを削除して再作成: $ROOT"
  rm -rf "$ROOT"
fi
.venv/bin/lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5B7B0094901 \
  --robot.id=my_follower \
  --robot.cameras="{overhead: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}}" \
  --robot.disable_torque_on_disconnect=true \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5B7B0090111 \
  --teleop.id=my_leader \
  --dataset.repo_id=local/so101_tomato_crate_2cam_home \
  --dataset.root="$ROOT" \
  --dataset.single_task="pick up a tomato from the green crate and lift it up" \
  --dataset.num_episodes=50 \
  --dataset.fps=30 \
  --dataset.episode_time_s=45 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false
