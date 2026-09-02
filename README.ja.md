# 不良画像を 1 枚も使わない外観検査 — 何が効いて何が効かなかったか

ゴム製アヒルを題材に、**不良画像を 1 枚も学習に使わない**外観検査を実測したリポジトリです。
2024/07 の記事 [Amazon Titan Multimodal Embeddings で外観検査をやってみました](https://dev.classmethod.jp/articles/visual-inspection-with-amazon-titan-multimodal-embeddings/) の続編にあたります。

## 結論

> **ファインチューニングは割に合わなかった。撮影条件を揃える方が、他のどの工夫よりも効いた。**

| やったこと | 運用ウィンドウの変化 |
|:--|:--|
| **照明 3 条件を撮る** | **−5.25σ → +12.18σ**（破綻から完全分離へ） |
| モデルを Bedrock から DINOv2 に替える | −3.18σ → **+12.18σ** |
| パッチ化 + 近傍集約 | 破綻時に −5.25σ → −2.83σ（改善するが負のまま） |
| **ファインチューニング** | **+12.18σ → +1.00σ（悪化）** |

「運用ウィンドウ」は **過検出 0% かつ検出 100% を同時に満たす閾値の幅**（単位は正常のばらつき σ）。
正なら「その幅のどこに閾値を引いても成立する」ことを意味します。

**AUROC は主要な 4 条件すべてで 1.000 でした。** AUROC だけを見ていたら、これらの差には気づけません。

---

## 動作環境

| 項目 | バージョン |
|:--|:--|
| OS | macOS（Apple Silicon / MPS で確認） |
| Python | 3.10.13 |
| PyTorch | 2.9.1 |
| transformers | 5.16.1 |
| OpenCV | 4.10.0 |
| バックボーン | `facebook/dinov2-small`（ViT-S/14, Apache-2.0, 22.1M） |
| カメラ（撮影時） | Logicool HD Pro Webcam C920 |

Bedrock を使う実験のみ AWS 認証が必要です（コストは後述、約 3 円）。

## セットアップ

```bash
git clone https://github.com/furuya02/sagemaker-embedding-ft-visual-inspection.git
cd sagemaker-embedding-ft-visual-inspection

# システムの torch を使う場合は --system-site-packages を付ける
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

**アヒルのデータ（67 枚 / 64MB）はリポジトリに同梱されています。** clone するだけで実験を再現できます。

---

## 実験の再現

### 1. 3 手法の比較（撮影条件を揃えた場合）

```bash
.venv/bin/python scripts/run_duck.py
```

`1 画像 = 1 ベクトル` / `パッチ + メモリバンク` / `パッチ + 近傍集約` を比較します。
**この条件では 3 手法とも AUROC 1.000、過検出 0% で検出 100%** になります。

### 2. 学習構成を変えた比較（本命）

```bash
.venv/bin/python scripts/run_experiments.py
```

- **分布シフト**: 学習を `normal` 9 枚に限定 → 未知の照明で破綻する
- **枚数スイープ**: 実写を増やすのと拡張で水増しするのとどちらが効くか

### 3. Bedrock 3 モデル（要 AWS 認証・約 $0.02）

```bash
.venv/bin/python scripts/run_bedrock.py
```

| モデル | モデル ID | リージョン |
|:--|:--|:--|
| Titan MME | `amazon.titan-embed-image-v1` | us-east-1（**東京では使えない**） |
| Nova 2 MME | `amazon.nova-2-multimodal-embeddings-v1:0` | us-east-1（**東京では使えない**） |
| Cohere Embed v4 | `cohere.embed-v4:0` | ap-northeast-1（**東京で使える唯一**） |

前記事の再現（基準画像 1 枚 / コサイン類似度 0.9）も同時に実行します。
埋め込みは `outputs/bedrock_emb.json` にキャッシュされるので、再実行しても課金されません。

### 4. ファインチューニング

```bash
# 手段A: 正常のみの自己教師あり適応（SimSiam 風）
.venv/bin/python scripts/train_ft.py --method a --train-set all --epochs 30

# 手段B: CutPaste 相当の疑似欠陥合成
.venv/bin/python scripts/train_ft.py --method b --train-set normal --epochs 30
```

`--train-set normal` にすると分布シフト条件（学習 9 枚）になります。
**実際の不良画像は 1 枚も使いません。** 疑似欠陥は正常画像から機械的に生成します。

DINOv2 の最終 2 ブロック（3.6M / 22.1M）のみ更新します。**ローカルの MPS で 70〜220 秒**で終わります。

### 5. 図の生成

```bash
.venv/bin/python scripts/make_figures.py
```


---

## 撮影する場合

自分のデータで試す場合は、撮影アプリを使えます。

```bash
.venv/bin/python scripts/capture_app.py --list-devices   # カメラ番号を調べる
.venv/bin/python scripts/capture_app.py --device 1
```

| キー | 動作 |
|:--|:--|
| ドラッグ | ROI を指定（`capture_config.json` に保存） |
| Space / Enter | 撮影 → 次の個体番号へ自動で進む |
| a / d | 個体番号 |
| 1 / 2 / 3 | 照明条件 |
| o / n | OK（無傷）/ NG（傷あり） |
| q / ESC | 終了 |

**ピントを常時監視し、未合焦では撮影をブロックします。** C920 は macOS の OpenCV から
フォーカスを固定できず（`set()` が False を返す）、合焦に約 2 秒かかるためです。

撮影後の品質チェック:

```bash
.venv/bin/python scripts/check_dataset.py --sheet
```

サイズ・鮮鋭度・露出・余白・重複・個体差を一括で点検し、コンタクトシートを出力します。

---

## ディレクトリ構成

```
.
├── src/
│   ├── embed.py          DINOv2 の CLS / パッチ埋め込み、局所近傍集約
│   ├── memorybank.py     1画像1ベクトル / パッチ+メモリバンク（leave-one-out 対応）
│   ├── metrics.py        運用ウィンドウ、σ 正規化、AUROC、ヒストグラム
│   └── augment.py        学習用の拡張（左右反転は使わない）
├── scripts/
│   ├── capture_app.py    撮影アプリ（OpenCV GUI）
│   ├── check_dataset.py  撮影データの品質チェック
│   ├── run_duck.py       3 手法の比較
│   ├── run_experiments.py 学習構成を変えた比較
│   ├── run_bedrock.py    Bedrock 3 モデル
│   ├── train_ft.py       ファインチューニング
│   ├── make_figures.py   記事用の図
└── data/duck/            アヒル 67 枚（CC BY 4.0）
```

## データセット

`data/duck/` — **Duck Visual Inspection Dataset**（67 枚 / CC BY 4.0）

| 用途 | 枚数 | 個体 |
|:--|--:|:--|
| 学習用（正常のみ） | 27 | #01〜#09 × 照明 3 条件 |
| 評価用・正常 | 30 | #10〜#19 × 照明 3 条件 |
| 評価用・異常 | 10 | #10〜#19（難易度 3 段階） |

**学習用と評価用は個体レベルで分離**しています。`test_ok` と `test_ng` は同じ個体なので、
**同一個体の before / after** として比較できます。

詳細は [data/duck/README.md](data/duck/README.md) を参照してください。

## ライセンス

- コード: **MIT**（[LICENSE](LICENSE)）
- データ: **CC BY 4.0**（[data/duck/LICENSE](data/duck/LICENSE)）

## 参考

- 前記事: https://dev.classmethod.jp/articles/visual-inspection-with-amazon-titan-multimodal-embeddings/
- AnomalyDINO (WACV 2025): https://arxiv.org/abs/2405.14529
- PatchCore: https://arxiv.org/abs/2106.08265
- DINOv2: https://github.com/facebookresearch/dinov2
