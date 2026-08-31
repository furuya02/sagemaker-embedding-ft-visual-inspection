# Visual Inspection Without a Single Defect Image — What Worked and What Didn't

Measuring **visual inspection that uses zero defect images for training**, on rubber ducks.
A follow-up to [Visual inspection with Amazon Titan Multimodal Embeddings](https://dev.classmethod.jp/articles/visual-inspection-with-amazon-titan-multimodal-embeddings/) (July 2024).

## Conclusion

> **Fine-tuning did not pay off. Getting the capture conditions right mattered more than any other technique.**

| What we did | Change in operating window |
|:--|:--|
| **Capture 3 lighting conditions** | **−5.25σ → +12.18σ** (from broken to fully separable) |
| Switch from Bedrock embeddings to DINOv2 | −3.18σ → **+12.18σ** |
| Patch-level features + neighborhood aggregation | −5.25σ → −2.83σ under shift (better, still negative) |
| **Fine-tuning** | **+12.18σ → +1.00σ (worse)** |

The *operating window* is the **range of thresholds that satisfy 0% false positives AND 100% recall
simultaneously**, measured in units of the standard deviation of normal scores (σ).
A positive value means "any threshold within this range works".

**AUROC was 1.000 in all four main conditions.** Looking only at AUROC, none of these differences are visible.

---

## Environment

| Item | Version |
|:--|:--|
| OS | macOS (verified on Apple Silicon / MPS) |
| Python | 3.10.13 |
| PyTorch | 2.9.1 |
| transformers | 5.16.1 |
| OpenCV | 4.10.0 |
| Backbone | `facebook/dinov2-small` (ViT-S/14, Apache-2.0, 22.1M) |
| Camera (for capture) | Logicool HD Pro Webcam C920 |

AWS credentials are required only for the Bedrock experiment (about $0.02).

## Setup

```bash
git clone https://github.com/furuya02/sagemaker-embedding-ft-visual-inspection.git
cd sagemaker-embedding-ft-visual-inspection

# use --system-site-packages if you already have torch installed
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

**The duck dataset (67 images / 64MB) is bundled in this repository.** Just clone and run.

---

## Reproducing the experiments

### 1. Comparing three methods (capture conditions aligned)

```bash
.venv/bin/python scripts/run_duck.py
```

Compares `one vector per image` / `patch + memory bank` / `patch + neighborhood aggregation`.
**Under this condition all three reach AUROC 1.000 and 100% recall at 0% false positives.**

### 2. Varying the training set (the interesting part)

```bash
.venv/bin/python scripts/run_experiments.py
```

- **Distribution shift**: train on 9 `normal` images only, then unseen lighting breaks it
- **Sweep**: capturing more real images vs. augmenting a few

### 3. Bedrock, three models (needs AWS credentials, ~$0.02)

```bash
.venv/bin/python scripts/run_bedrock.py
```

| Model | Model ID | Region |
|:--|:--|:--|
| Titan MME | `amazon.titan-embed-image-v1` | us-east-1 (**not available in Tokyo**) |
| Nova 2 MME | `amazon.nova-2-multimodal-embeddings-v1:0` | us-east-1 (**not available in Tokyo**) |
| Cohere Embed v4 | `cohere.embed-v4:0` | ap-northeast-1 (**the only one available in Tokyo**) |

This also reproduces the previous article's setup (one reference image, cosine similarity 0.9).
Embeddings are cached in `outputs/bedrock_emb.json`, so re-runs cost nothing.

### 4. Fine-tuning

```bash
# Method A: self-supervised adaptation on normal images only (SimSiam-style)
.venv/bin/python scripts/train_ft.py --method a --train-set all --epochs 30

# Method B: CutPaste-style synthetic defects
.venv/bin/python scripts/train_ft.py --method b --train-set normal --epochs 30
```

`--train-set normal` switches to the distribution-shift condition (9 training images).
**No real defect image is used.** Synthetic defects are generated from normal images only.

Only the last 2 blocks of DINOv2 (3.6M of 22.1M) are updated.
**Runs in 70–220 seconds on local MPS.**

### 5. Generating figures

```bash
.venv/bin/python scripts/make_figures.py
```

### 6. Validation on VisA (optional, 3.7GB download)

```bash
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar data/
tar xf data/VisA_20220922.tar -C data/
.venv/bin/python scripts/run_visa.py --category candle --n-train 200 --n-test-ok -1 --n-test-ng -1
.venv/bin/python scripts/subsample_stability.py --category candle --trials 300
```

VisA is published on the AWS Open Data Registry and **can be downloaded without an AWS account**.

---

## Capturing your own data

```bash
.venv/bin/python scripts/capture_app.py --list-devices
.venv/bin/python scripts/capture_app.py --device 1
```

| Key | Action |
|:--|:--|
| Drag | Set ROI (saved to `capture_config.json`) |
| Space / Enter | Capture, then advance to the next object |
| a / d | Object number |
| 1 / 2 / 3 | Lighting condition |
| o / n | OK (intact) / NG (defective) |
| q / ESC | Quit |

**Focus is monitored continuously and capture is blocked when out of focus.**
The C920's focus cannot be locked through OpenCV on macOS (`set()` returns False)
and autofocus takes about 2 seconds to settle.

Quality check after capture:

```bash
.venv/bin/python scripts/check_dataset.py --sheet
```

Checks size, sharpness, exposure, margins, duplicates and per-object variation, and writes a contact sheet.

---

## Layout

```
.
├── src/
│   ├── embed.py          DINOv2 CLS / patch embeddings, neighborhood aggregation
│   ├── memorybank.py     Global scorer / patch memory bank (with leave-one-out)
│   ├── metrics.py        Operating window, σ normalization, AUROC, histograms
│   └── augment.py        Training-time augmentation (no horizontal flip)
├── scripts/
│   ├── capture_app.py    Capture app (OpenCV GUI)
│   ├── check_dataset.py  Dataset quality check
│   ├── run_duck.py       Three-method comparison
│   ├── run_experiments.py Varying the training set
│   ├── run_bedrock.py    Bedrock, three models
│   ├── train_ft.py       Fine-tuning
│   ├── make_figures.py   Figures for the article
│   └── run_visa.py       Validation on VisA
└── data/duck/            67 duck images (CC BY 4.0)
```

## Dataset

`data/duck/` — **Duck Visual Inspection Dataset** (67 images / CC BY 4.0)

| Split | Count | Objects |
|:--|--:|:--|
| Train (normal only) | 27 | #01–#09 × 3 lighting conditions |
| Test, normal | 30 | #10–#19 × 3 lighting conditions |
| Test, anomalous | 10 | #10–#19 (3 difficulty levels) |

**Training and test objects are disjoint.** `test_ok` and `test_ng` contain the same objects,
so they form **before / after pairs of the same individual**.

See [data/duck/README.md](data/duck/README.md) for details.

## License

- Code: **MIT** ([LICENSE](LICENSE))
- Dataset: **CC BY 4.0** ([data/duck/LICENSE](data/duck/LICENSE))

## References

- Previous article: https://dev.classmethod.jp/articles/visual-inspection-with-amazon-titan-multimodal-embeddings/
- VisA (Zou et al., ECCV 2022): https://github.com/amazon-science/spot-diff
- AnomalyDINO (WACV 2025): https://arxiv.org/abs/2405.14529
- PatchCore: https://arxiv.org/abs/2106.08265
- DINOv2: https://github.com/facebookresearch/dinov2
