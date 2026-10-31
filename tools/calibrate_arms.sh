#!/bin/bash
# フォロワー→リーダーの順に再キャリブレーションする。
# 各アームの手順: ①全関節を可動域の真ん中の姿勢にする → ENTER
#                ②全関節を1つずつ端から端までゆっくり動かす(数値が動くこと確認) → ENTER
# 実行: bash calibrate_arms.sh
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"  # リポジトリルートで実行
set -e

echo "=============================================="
echo "① フォロワー(白アーム)のキャリブレーション"
echo "=============================================="
# ★2026-10-27農園で実測: 0094901=白フォロワー, 0090111=黒リーダー (家の記録と逆!)
# キャリブ開始時に「白を動かして数値が動くか」で毎回確認すること
.venv/bin/lerobot-calibrate \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem5B7B0094901 \
  --robot.id=my_follower

echo ""
echo "=============================================="
echo "② リーダー(黒アーム)のキャリブレーション"
echo "=============================================="
.venv/bin/lerobot-calibrate \
  --teleop.type=so101_leader \
  --teleop.port=/dev/cu.usbmodem5B7B0090111 \
  --teleop.id=my_leader

echo ""
echo "✅ 両アームのキャリブレーション完了。record_crate_pick.sh で収録に進めます"
