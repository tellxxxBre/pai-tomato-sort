"""負荷テスト: トルクON(モーター通電)状態で60秒間読み続け、どこで落ちるか見る。
手順: 実行 → 最初の15秒は何も触らない → その後、電源のバレルジャック・
モーター4-5間・5-6間のケーブルを順に軽く揺する。
usage: python diag_bus_torque.py
"""
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

PORT = "/dev/cu.usbmodem5B7B0090111"  # follower

bus = FeetechMotorsBus(
    port=PORT,
    motors={f"m{i}": Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for i in range(1, 7)},
)
bus.connect(handshake=True)
print("接続OK。トルクをONにします（アームがその場で固まります）...")
try:
    bus.enable_torque(num_retry=2)
    print("トルクON成功。60秒間読み続けます。")
except Exception as e:
    print(f"💥 トルクONの時点で失敗: {type(e).__name__}: {e}")
    print("→ 通電の瞬間に落ちる＝電源系の接触不良がほぼ確定です")

t0 = time.time()
fail = 0
try:
    while time.time() - t0 < 60:
        stamp = f"{time.time()-t0:5.1f}s"
        try:
            pos = bus.sync_read("Present_Position", normalize=False, num_retry=0)
            print(f"{stamp}  ✅ 全モーター応答   ", end="\r", flush=True)
        except Exception as e:
            fail += 1
            print(f"{stamp}  ❌ {type(e).__name__}: {str(e)[:80]}")
            time.sleep(0.3)
        time.sleep(0.15)
finally:
    print(f"\n終了。異常検出回数: {fail}")
    try:
        bus.disable_torque(num_retry=5)
        print("トルクOFFにしました。")
    except Exception:
        print("(トルクOFF失敗 — 電源を切ってください)")
