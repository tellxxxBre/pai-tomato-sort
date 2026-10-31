"""ベンチマーク21枚をHSVルール判定器で判定し、Claude/Jevと比較する(API不要・ローカル完結)。
usage: .venv/bin/python tomato_sort/rejudge_benchmark_hsv.py
"""
import csv
import os
import time

import cv2

HERE = os.path.dirname(__file__)
LOG_DIR = os.path.join(HERE, "logs")
CSV_IN = os.path.join(LOG_DIR, "benchmark_labels.csv")

import sys
sys.path.insert(0, HERE)
from tomato_sort import judge_with_hsv  # 本番と同一実装・同一しきい値

rows = list(csv.DictReader(open(CSV_IN)))
ok, results, lats = 0, [], []
for i, r in enumerate(rows):
    full = os.path.join(LOG_DIR, r["crop_path"].replace("_crop", ""))
    img_bgr = cv2.imread(full)
    if img_bgr is None:
        print(f"[{i+1}] 画像なし: {full}")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    v = judge_with_hsv(img_rgb)
    lat = time.time() - t0
    lats.append(lat)
    gt = r["ground_truth"]
    match = v.get("label") == gt
    ok += match
    results.append((r["crop_path"], gt, v.get("label"), match))
    print(f"[{i+1}/{len(rows)}] 正解={gt} HSV={v.get('label')} {'✅' if match else '❌'} ({v['reason']}, {lat*1000:.1f}ms)")

n = len(results)
print(f"\n=== 3エンジン比較(同一21枚) ===")
print(f"精度:   Claude 21/21 (100%) | Jev 19/21 (90%) | HSV {ok}/{n} ({ok/n*100:.0f}%)")
print(f"速度:   Claude 3.7秒        | Jev 0.26秒      | HSV {sum(lats)/len(lats)*1000:.1f}ミリ秒")
print(f"コスト: Claude 数円/回      | Jev 0.1円未満   | HSV 0円")
for name, gt, lb, m in results:
    if not m:
        print(f"  HSV不一致: {name} 正解={gt} 判定={lb}")
