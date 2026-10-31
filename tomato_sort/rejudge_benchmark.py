"""ベンチマーク画像(benchmark_labels.csv)に現行プロンプトで再判定をかけ、正解率を測る。
プロンプト改良のたびに実行すれば、同じ21枚(同一の物差し)で前後比較ができる。
usage: .venv/bin/python tomato_sort/rejudge_benchmark.py
"""
import csv
import os
import time

import cv2

HERE = os.path.dirname(__file__)
LOG_DIR = os.path.join(HERE, "logs")
CSV_PATH = os.path.join(LOG_DIR, "benchmark_labels.csv")

import sys
sys.path.insert(0, HERE)
from tomato_sort import judge_with_vlm  # 現行プロンプト・現行クロップをそのまま使う

if not os.environ.get("ANTHROPIC_API_KEY"):
    raise SystemExit("ANTHROPIC_API_KEY を設定してください")
import anthropic
client = anthropic.Anthropic()

rows = list(csv.DictReader(open(CSV_PATH)))
ok = 0
results = []
for i, r in enumerate(rows):
    full = os.path.join(LOG_DIR, r["crop_path"].replace("_crop", ""))
    img_bgr = cv2.imread(full)
    if img_bgr is None:
        print(f"[{i+1}] 画像なし: {full}")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    try:
        v = judge_with_vlm(client, img_rgb)
    except Exception as e:
        print(f"[{i+1}] 判定失敗: {str(e)[:80]}")
        continue
    lat = time.time() - t0
    gt = r["ground_truth"]
    match = v.get("label") == gt
    ok += match
    results.append((r["crop_path"], gt, v.get("label"), match))
    print(f"[{i+1}/{len(rows)}] 正解={gt} 新判定={v.get('label')} {'✅' if match else '❌'} ({lat:.1f}s)")

n = len(results)
print(f"\n=== 新プロンプトの正解率: {ok}/{n} = {ok/n*100:.0f}% (旧: 14/21=67%) ===")
for name, gt, lb, m in results:
    if not m:
        print(f"  まだ不一致: {name} 正解={gt} 判定={lb}")
