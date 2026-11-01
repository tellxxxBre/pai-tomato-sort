#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2カメラ(俯瞰+手首)データセットから「俯瞰のみ」コピーを作る。

lerobot 0.6.0 はデータセット内の全カメラ(features の dtype=video/image)を自動で学習に使う。
CLI でカメラを選ぶフラグは無いので、俯瞰だけで学習したいときはこのコピーを作る。
検証済み(PLAN_リストカメラ検証.md): info.json/stats.json から wrist を除き、俯瞰動画だけ
symlink すると lerobot が認識する camera_keys が ['observation.images.overhead'] だけになる。

使い方:
  python3 tomato_sort/make_overhead_only.py <2camデータセットのroot> [--keep overhead] [--drop wrist]
出力: 同じ階層に <root名>_overheadonly/ を作る。

注意: 動画ファイルは symlink。元データを消すとコピーも壊れる(学習が終わるまで元を消さない)。
"""
import argparse, json, os, shutil, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="2カメラ データセットの root (…/local/so101_..._2cam_YYYYMMDD)")
    ap.add_argument("--keep", default="overhead", help="残すカメラ名 (feature名の末尾。既定 overhead)")
    ap.add_argument("--drop", default="wrist", help="落とすカメラ名 (既定 wrist)")
    ap.add_argument("--out", default=None, help="出力root (既定: <root>_overheadonly)")
    a = ap.parse_args()

    root = os.path.abspath(a.root.rstrip("/"))
    if not os.path.isdir(os.path.join(root, "meta")):
        sys.exit(f"meta/ が無い。rootが違う: {root}")
    keep_key = f"observation.images.{a.keep}"
    drop_key = f"observation.images.{a.drop}"
    out = os.path.abspath(a.out) if a.out else root + "_overheadonly"
    if os.path.exists(out):
        sys.exit(f"出力先が既にある。消してから再実行: {out}")

    info = json.load(open(os.path.join(root, "meta/info.json"), encoding="utf-8"))
    feats = info["features"]
    if keep_key not in feats:
        sys.exit(f"残すカメラ {keep_key} が features に無い。features={list(feats)}")
    if drop_key not in feats:
        sys.exit(f"落とすカメラ {drop_key} が features に無い(既に俯瞰のみ?)。features={list(feats)}")

    os.makedirs(out)
    # data/ は映像を含まない(action/state/index類のみ)ので丸ごと symlink で共有
    os.symlink(os.path.join(root, "data"), os.path.join(out, "data"))
    # videos/ は残すカメラのサブツリーだけ symlink
    os.makedirs(os.path.join(out, "videos"))
    os.symlink(os.path.join(root, "videos", keep_key), os.path.join(out, "videos", keep_key))
    # meta/ は編集するのでコピー
    shutil.copytree(os.path.join(root, "meta"), os.path.join(out, "meta"))

    # info.json から落とすカメラを除去
    o_info = json.load(open(os.path.join(out, "meta/info.json"), encoding="utf-8"))
    del o_info["features"][drop_key]
    json.dump(o_info, open(os.path.join(out, "meta/info.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=4)
    # stats.json から落とすカメラを除去(存在すれば)
    sp = os.path.join(out, "meta/stats.json")
    if os.path.exists(sp):
        st = json.load(open(sp, encoding="utf-8"))
        st.pop(drop_key, None)
        json.dump(st, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

    print(f"作成: {out}")
    print(f"  残したカメラ: {keep_key}")
    print(f"  落としたカメラ: {drop_key}")

    # 検証: lerobot が認識する camera_keys を metadata だけで確認(動画デコードしない)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        meta = LeRobotDatasetMetadata(repo_id="local/_check", root=out)
        print("  camera_keys =", meta.camera_keys)
        if list(meta.camera_keys) == [keep_key]:
            print("  OK: 俯瞰のみになっている")
        else:
            print("  NG: 想定と違う。camera_keys を確認せよ")
    except Exception as e:
        print(f"  (metadata検証はスキップ: {type(e).__name__}: {e})")
        print(f"  手動確認: python3 -c \"import json;print(list(json.load(open('{out}/meta/info.json'))['features']))\"")

if __name__ == "__main__":
    main()
