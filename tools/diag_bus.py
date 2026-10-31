"""モーターバスの揺すりテスト: 60秒間pingし続け、どのモーターが落ちるかライブ表示する。
実行中にケーブルを1本ずつ軽く揺すって、落ちる瞬間のケーブルを特定する。
usage: python diag_bus.py
"""
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "/dev/cu.usbmodem5B7B0090111"  # follower

bus = FeetechMotorsBus(
    port=PORT,
    motors={f"m{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in range(1, 7)},
)
bus.connect(handshake=False)
print("接続OK。60秒間pingし続けます。ケーブルを1本ずつ軽く揺すってください。")
print("(全応答の間は行を上書き表示、落ちた瞬間だけ行が残ります)")

t0 = time.time()
fail_count = 0
try:
    while time.time() - t0 < 60:
        stamp = f"{time.time()-t0:5.1f}s"
        try:
            found = bus.broadcast_ping()
            ids = sorted(found.keys()) if found else []
            missing = [i for i in range(1, 7) if i not in ids]
            if missing:
                fail_count += 1
                print(f"{stamp}  ❌ 応答なし: {missing}   (応答あり: {ids})")
            else:
                print(f"{stamp}  ✅ 全モーター応答 {ids}   ", end="\r", flush=True)
        except Exception as e:
            fail_count += 1
            print(f"{stamp}  💥 バス全体エラー: {type(e).__name__}: {e}")
            time.sleep(0.3)
        time.sleep(0.2)
finally:
    print(f"\n終了。異常検出回数: {fail_count}")
    try:
        bus.disconnect(disable_torque=False)
    except Exception:
        pass
