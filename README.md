# pai-tomato-sort — SO-101によるトマト選果の自動化（PAI最終課題）

安価なロボットアーム SO-101（約7万円）＋AIで、ミニトマト農園の選果作業（正品/非正品の判定と振り分け）をどこまで自動化できるかを、実機・実トマト・実農園で検証した際のコード一式です。

詳細な経緯・実験結果は提出の説明資料と動画を参照してください。

## システム構成

全工程をend-to-end学習にせず、工程ごとに手法を割り当てるモジュール分離設計です:

```
[俯瞰カメラ] → ①掴む: 模倣学習ポリシー（ACT） → ②判定: AI（VLM等3エンジン） → ③振り分け: 記憶ポーズ再生
```

- ①把持: テレオペ実演データからの模倣学習（ACT / SmolVLA+LoRA）
- ②熟度判定: 差し替え可能な3エンジン — Claude（VLM）/ Jev（判定特化AI・TypeSafe）/ HSV色相しきい値（古典CV）
- ③箱への配置: 登録ポーズの再生（クレーン式移動: 上げる→回す→降ろす）

## ディレクトリ

| パス | 内容 |
|---|---|
| `tomato_sort/tomato_sort.py` | 本体。掴む→判定→振り分けの通しサイクル実行（把持モデル・判定エンジンをフラグで差し替え可能） |
| `tomato_sort/record_crate_pick_1cam.sh` | テレオペ実演データ収集（俯瞰1カメラ構成） |
| `tomato_sort/record_crate_pick.sh` | 同・2カメラ構成版（手首カメラあり） |
| `tomato_sort/capture_pose.py` / `poses.json` / `verify_poses.py` | 振り分け先ポーズの登録・保存・検品 |
| `tomato_sort/check_inspect.py` | 判定テスト＋本番同一条件の正解ラベル付きベンチマーク収集 |
| `tomato_sort/rejudge_benchmark.py` | ベンチマークに対するClaude（VLM）判定の再採点（プロンプト反復改善用） |
| `tomato_sort/rejudge_benchmark_jev.py` | 同・Jev（TypeSafe）版 |
| `tomato_sort/rejudge_benchmark_hsv.py` | 同・HSV色相しきい値（古典CV・0円）版 |
| `tomato_sort/cam_check.py` | カメラindexの実測確認（現場では毎セッション実測する） |
| `training/colab_train_act_tomato_crate*.ipynb` | ACTの学習ノートブック（Colab A100・デモ40本/100本） |
| `training/colab_finetune_smolvla_tomato_crate_100ep.ipynb` | SmolVLAのLoRAファインチューニング |
| `tools/calibrate_arms.sh` | フォロワー→リーダーの再キャリブレーション手順 |
| `tools/diag_bus.py` / `diag_bus_torque.py` | モーターバスの診断（過負荷ラッチからの復旧用） |

## セットアップ

前提: SO-101（リーダー＋フォロワー）、USBカメラ（俯瞰）、Python 3.10+、[lerobot](https://github.com/huggingface/lerobot)

```bash
python -m venv .venv && source .venv/bin/activate
pip install lerobot anthropic opencv-python

# 判定エンジンに応じて設定（HSVエンジンのみ使う場合は不要）
export ANTHROPIC_API_KEY=...   # Claude（VLM）判定
export TYPESAFE_API_KEY=...    # Jev判定
```

ポート名・カメラindexは環境で変わるため、**毎セッション実測**してからスクリプト内の値を合わせてください（`cam_check.py`、`lerobot-find-port`）。

## 現場で学んだ安全機構（コード化済み）

- 起動時に「目標位置 := 現在位置」を書いてからトルクON（古い目標への跳ね防止）
- ソフトスタート（2秒ブレンド）
- 把持中の関節電流監視（押し付け検知で退避）
- 台本移動はクレーン式（上げる→回す→降ろす）に統一
- 過負荷ラッチの解除はトルクを切ったまま行う

## ライセンス / 注意

実験・学習用途のコードです。実機を動かす際は周囲の安全を確保してください。
