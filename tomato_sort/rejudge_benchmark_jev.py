"""ベンチマーク21枚(正解ラベル付き)をJev(TypeSafe)で判定し、Claudeと精度・速度を比較する。

前提:
  1. https://console.typesafe.ai/ でサインアップしてAPIキーを取得
  2. export TYPESAFE_API_KEY=<キー>
  3. .venv/bin/pip install typesafe-sdk

usage: .venv/bin/python tomato_sort/rejudge_benchmark_jev.py

入力はClaude判定と完全同一(同じ元画像→同じクロップ→同じ2倍拡大)。
結果: logs/jev_results.csv + 画面にClaude(21/21・平均3.7秒)との比較サマリ。
"""
import csv
import os
import time
from pathlib import Path

import cv2

HERE = os.path.dirname(__file__)
LOG_DIR = os.path.join(HERE, "logs")
CSV_IN = os.path.join(LOG_DIR, "benchmark_labels.csv")
CSV_OUT = os.path.join(LOG_DIR, "jev_results.csv")

import sys
sys.path.insert(0, HERE)
from tomato_sort import CROP_X0, CROP_X1, CROP_Y0, CROP_Y1  # Claudeと同一クロップ

if not os.environ.get("TYPESAFE_API_KEY"):
    raise SystemExit("TYPESAFE_API_KEY を設定してください (https://console.typesafe.ai/ で取得)")

from typesafe_sdk import Choice, Image, Noul, TypeSafeClient

INSTRUCTIONS = (
    "The image `img` shows a robot gripper (white plastic fingers) possibly holding a plum-type "
    "mini tomato lifted in the air. This camera desaturates colors: a truly ripe red tomato "
    "appears deep orange to red, while an unripe green-yellow tomato appears yellow, amber, or "
    "pale yellowish-orange. Judge by HUE (yellowish vs reddish), not by brightness. "
    "Ignore any tomatoes lying in crates or boxes in the background."
)

QUESTIONS = {
    "held": Noul(
        instructions="Is the gripper in `img` holding a tomato lifted in the air? "
                     "(not tomatoes lying in the crate or boxes)"
    ),
    "label": Choice(
        instructions=INSTRUCTIONS + " Which class is the HELD tomato?",
        criteria={
            "seihin": (
                "Product grade: the held tomato's hue is orange to red (reddish). "
                "Medium or light orange still counts if there is no yellow tint."
            ),
            "fuhin": (
                "Non-product: the held tomato looks yellow, mustard, amber, or yellowish-orange, "
                "or has olive/green patches, or shows wrinkles, dark spots, or damage. "
                "If unsure between yellowish and orange, choose this."
            ),
        },
    ),
}


TMP_JPG = os.path.join(LOG_DIR, "_jev_tmp.jpg")


def crop_to_tmp(full_path: str) -> str:
    """Claudeのjudge_with_vlmと同一の前処理: クロップ→2倍拡大→JPEG(q92)を一時ファイルに書く"""
    img = cv2.imread(full_path)
    crop = img[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
    crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(TMP_JPG, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return TMP_JPG


rows = list(csv.DictReader(open(CSV_IN)))
results = []
with TypeSafeClient() as client:
    for i, r in enumerate(rows):
        full = os.path.join(LOG_DIR, r["crop_path"].replace("_crop", ""))
        if not os.path.exists(full):
            print(f"[{i+1}] 画像なし: {full}")
            continue
        tmp = crop_to_tmp(full)
        t0 = time.time()
        try:
            resp = client.system_one(
                model="jev-latest",
                state={"img": Image(Path(tmp))},
                questions=QUESTIONS,
                extra_headers={"TypeSafe-Danger-Zone-Images": "true"},
            )
        except Exception as e:
            print(f"[{i+1}] Jev呼び出し失敗: {str(e)[:120]}")
            continue
        lat = time.time() - t0
        held_p = resp.nouls["held"].noul
        label = resp.choices["label"].choice
        gt = r["ground_truth"]
        match = label == gt
        results.append({"crop_path": r["crop_path"], "ground_truth": gt,
                        "jev_label": label, "jev_held_p": f"{held_p:.2f}",
                        "jev_latency_s": f"{lat:.2f}", "match": match})
        print(f"[{i+1}/{len(rows)}] 正解={gt} Jev={label} {'✅' if match else '❌'} "
              f"(held確率{held_p:.2f}, {lat:.2f}秒)")

if not results:
    raise SystemExit("結果なし")
with open(CSV_OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

n = len(results)
acc = sum(r["match"] for r in results)
lats = [float(r["jev_latency_s"]) for r in results]
print("\n=== Claude vs Jev (同一21枚・同一入力) ===")
print(f"精度:      Claude 21/21 (100%)  |  Jev {acc}/{n} ({acc/n*100:.0f}%)")
print(f"レイテンシ: Claude 平均3.7秒      |  Jev 平均{sum(lats)/len(lats):.2f}秒 (最短{min(lats):.2f} 最長{max(lats):.2f})")
print(f"保存: {CSV_OUT}")
for r in results:
    if not r["match"]:
        print(f"  Jev不一致: {r['crop_path']} 正解={r['ground_truth']} Jev={r['jev_label']}")
