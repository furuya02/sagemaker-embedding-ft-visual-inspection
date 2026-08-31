"""DINOv2 をアヒルの正常画像だけでファインチューニングする（記事の本題）。

条件を揃えれば DINOv2 は既に完璧（運用幅 +12.18σ）で、破綻するのは
**学習に無い照明が来たとき**だけ（−5.25σ）。FT がこの破綻を救えるかを測る。

  手段A  正常のみの自己教師あり適応（SimSiam 風。stop-gradient で埋め込み崩壊を防ぐ）
  手段B  CutPaste 相当の疑似欠陥合成（正常 vs 人工欠陥の2値分類）

**実際の不良画像は 1 枚も使わない。** 手段B の疑似欠陥は正常画像から機械的に作る
（実 NG を見て似せると漏洩するので、一般的な CutPaste に留める）。

  python scripts/train_ft.py --method b --train-set normal --epochs 30
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embed import Dinov2Encoder, aggregate_neighbors, MEAN, STD  # noqa: E402
from src.memorybank import GlobalScorer, PatchMemoryBank, l2norm     # noqa: E402
from src.metrics import evaluate, to_sigma, operating_window, roc_points  # noqa: E402

DATA = ROOT / "data" / "duck"
LIGHTS = ["normal", "side", "bright"]
LEVELS = ["easy", "medium", "hard"]


def paths(train_set):
    lights = ["normal"] if train_set == "normal" else LIGHTS
    tr = sorted(str(p) for c in lights for p in (DATA / "train_ok" / c).glob("*.png"))
    ok = sorted(str(p) for c in LIGHTS for p in (DATA / "test_ok" / c).glob("*.png"))
    ng = sorted(str(p) for l in LEVELS for p in (DATA / "test_ng" / l).glob("*.png"))
    return tr, ok, ng


def load_tensor(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1)


# ------------------------------------------------------------------ 拡張
def aug(x, rng):
    """学習時の拡張。位置・向き・軽微な明度（実写で確保できないものは作らない）。"""
    if rng.random() < 0.5:
        k = int(rng.integers(-2, 3))                      # ±約15度 相当の粗い回転
        x = torch.rot90(x, 0, (1, 2)) if k == 0 else x
    # シフト（±5%）
    s = int(x.shape[-1] * 0.05)
    dx, dy = int(rng.integers(-s, s + 1)), int(rng.integers(-s, s + 1))
    x = torch.roll(x, shifts=(dy, dx), dims=(1, 2))
    # 明度・コントラスト
    x = x * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.15, 0.15))
    return x


def cutpaste(x, rng):
    """CutPaste: 自分の中の矩形を切って別の場所に貼る。外部データは使わない。"""
    _, h, w = x.shape
    area = float(rng.uniform(0.02, 0.10)) * h * w
    ar = float(rng.uniform(0.3, 3.3))
    ph, pw = int(np.sqrt(area / ar)), int(np.sqrt(area * ar))
    ph, pw = max(4, min(ph, h // 3)), max(4, min(pw, w // 3))
    sy, sx = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
    dy, dx = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
    y = x.clone()
    patch = x[:, sy:sy + ph, sx:sx + pw].clone()
    if rng.random() < 0.5:                                 # 色を少しずらして「異物感」を出す
        patch = patch * float(rng.uniform(0.7, 1.3))
    y[:, dy:dy + ph, dx:dx + pw] = patch
    return y


# ------------------------------------------------------------------ 学習
class Projector(nn.Module):
    def __init__(self, d, hidden=512, out=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.BatchNorm1d(hidden),
                                 nn.ReLU(inplace=True), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def finetune(enc, tr_paths, method, epochs, lr, device, seed=0):
    """最終 2 ブロックのみ更新する。27 枚（分布シフト時は 9 枚）しかないため、
    全体を動かすと事前学習の特徴が壊れる（catastrophic forgetting）。"""
    model = enc.model
    for p in model.parameters():
        p.requires_grad = False
    tuned = []
    for blk in model.encoder.layer[-2:]:
        for p in blk.parameters():
            p.requires_grad = True
            tuned.append(p)
    print(f"  更新するパラメータ: {sum(p.numel() for p in tuned)/1e6:.1f}M / "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    d = model.config.hidden_size
    head = (Projector(d).to(device) if method == "a"
            else nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Linear(256, 2)).to(device))
    pred = Projector(256, 128, 256).to(device) if method == "a" else None
    params = tuned + list(head.parameters()) + (list(pred.parameters()) if pred else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    imgs = torch.stack([load_tensor(p, enc.size) for p in tr_paths])
    rng = np.random.default_rng(seed)
    model.train(); n = len(imgs); bs = 2

    for ep in range(epochs):
        idx = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = imgs[idx[i:i + bs]]
            if len(b) < 2:
                continue          # BatchNorm はバッチ 1 では動かない（9 枚 ÷ 2 の余り）
            if method == "a":
                # 同じ画像の 2 ビューを近づける。stop-gradient で崩壊を防ぐ（SimSiam）
                v1 = torch.stack([aug(x, rng) for x in b]).to(device)
                v2 = torch.stack([aug(x, rng) for x in b]).to(device)
                z1 = head(model(pixel_values=v1).last_hidden_state[:, 0])
                z2 = head(model(pixel_values=v2).last_hidden_state[:, 0])
                p1, p2 = pred(z1), pred(z2)
                loss = -(F.cosine_similarity(p1, z2.detach(), dim=-1).mean()
                         + F.cosine_similarity(p2, z1.detach(), dim=-1).mean()) / 2
            else:
                # 正常 vs 疑似欠陥の 2 値分類（実 NG は 1 枚も使わない）
                normal = torch.stack([aug(x, rng) for x in b])
                fake = torch.stack([cutpaste(aug(x, rng), rng) for x in b])
                xb = torch.cat([normal, fake]).to(device)
                yb = torch.cat([torch.zeros(len(b)), torch.ones(len(b))]).long().to(device)
                loss = F.cross_entropy(head(model(pixel_values=xb).last_hidden_state[:, 0]), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
        if (ep + 1) % max(1, epochs // 6) == 0:
            print(f"    epoch {ep+1:>3}/{epochs}  loss {tot/max(1,(n+bs-1)//bs):+.4f}", flush=True)
    model.eval()
    return enc


# ------------------------------------------------------------------ 評価
def report(tag, enc, tr, ok, ng, agg, results):
    cls_tr, pat_tr = enc.encode(tr)
    cls_ok, pat_ok = enc.encode(ok)
    cls_ng, pat_ng = enc.encode(ng)

    def summarize(name, cal, s_ok, s_ng):
        m = evaluate(s_ok, s_ng)
        z_ok, z_ng = to_sigma(s_ok, cal), to_sigma(s_ng, cal)
        w = operating_window(z_ok, z_ng, max_fp=0.0, min_recall=1.0)
        r = {"auroc": m["auroc"], "window": w["width"], "ok_max": float(z_ok.max()),
             "ng_min": float(z_ng.min()),
             "fp0": roc_points(z_ok, z_ng, (0.0,))[0]["recall"],
             "by_light": {c: float(np.mean([z for p, z in zip(ok, z_ok) if f"/{c}/" in p]))
                          for c in LIGHTS}}
        results[f"{tag}/{name}"] = r
        bl = "  ".join(f"{c[:4]}{r['by_light'][c]:+5.1f}" for c in LIGHTS)
        print(f"  {name:<22}{r['auroc']:>7.3f}{r['window']:>+9.2f}{r['ok_max']:>+8.2f}"
              f"{r['ng_min']:>+8.2f}{r['fp0']*100:>8.0f}%   {bl}")
        return r

    print(f"\n{tag}")
    print(f"  {'手法':<22}{'AUROC':>7}{'運用幅':>9}{'OK最大':>8}{'NG最小':>8}{'検出':>9}   照明別(σ)")
    g = GlobalScorer().fit(cls_tr)
    sims = l2norm(cls_tr) @ l2norm(cls_tr).T
    np.fill_diagonal(sims, -np.inf)
    summarize("1画像=1ベクトル", 1.0 - sims.max(axis=1), g.score(cls_ok), g.score(cls_ng))

    a = lambda p: aggregate_neighbors(p, enc.grid, agg) if agg > 1 else p  # noqa: E731
    mb = PatchMemoryBank(coreset_ratio=1.0).fit_loo(a(pat_tr))
    summarize("パッチ+近傍集約", mb.score_loo(), mb.score(a(pat_ok))[0], mb.score(a(pat_ng))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["a", "b"], default="b")
    ap.add_argument("--train-set", choices=["all", "normal"], default="normal")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--agg", type=int, default=3)
    args = ap.parse_args()

    tr, ok, ng = paths(args.train_set)
    name = {"a": "手段A 正常のみ自己教師あり適応", "b": "手段B CutPaste相当の疑似欠陥"}[args.method]
    print(f"■ {name}   学習 {len(tr)} 枚（{args.train_set}）  epochs={args.epochs} lr={args.lr}")

    results = {}
    enc = Dinov2Encoder(size=args.size)
    base = copy.deepcopy(enc.model.state_dict())
    report(f"FT前（{args.train_set} {len(tr)}枚）", enc, tr, ok, ng, args.agg, results)

    t0 = time.time()
    finetune(enc, tr, args.method, args.epochs, args.lr, enc.device)
    print(f"  学習時間 {time.time()-t0:.0f}s")
    report(f"FT後 手段{args.method.upper()}", enc, tr, ok, ng, args.agg, results)

    enc.model.load_state_dict(base)   # 念のため元に戻す
    out = ROOT / "outputs" / f"ft_{args.method}_{args.train_set}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
