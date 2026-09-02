"""ファインチューニングの改善版。

初版（train_ft.py）では FT すると運用ウィンドウが縮んだ。原因を 4 つに切り分け、
そのうち 3 つに対策を入れたのがこのスクリプト。

  A 分類ヘッドの出力を直接スコアにする
      初版は「正常 vs 疑似欠陥」を分類するよう学習しておきながら、
      評価ではヘッドを捨てて埋め込みだけ使っていた。学習目標と評価がズレていた。
  B 学習を弱くする
      27 枚に対して 3.6M パラメータ × 30 エポックは強すぎた。
      更新するブロック数・学習率・エポックを絞れるようにした。
  C CutPaste-Scar を混ぜる
      初版の疑似欠陥は矩形パッチの貼り付けのみ。実際の「難」は幅 1mm 以下の細い線で、
      形が違いすぎた。CutPaste 原論文（CVPR 2021）の Scar variant（細長い矩形）を加える。
      ※ 実際の傷を見て似せたわけではなく、一般的な合成手法。NG は 1 枚も使わない。

公平に比較するため、学習用の正常を「学習」と「閾値校正」に分け、
FT 前後で同じ校正データを使う。NG は評価にのみ使う。

  python scripts/train_ft2.py --score head --scar --blocks 1 --lr 1e-6 --epochs 5
"""
import argparse
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


def obj_id(p):
    return int(Path(p).stem.split("_")[-1])


def paths(train_set, _levels=LEVELS):
    lights = ["normal"] if train_set == "normal" else LIGHTS
    tr = sorted(str(p) for c in lights for p in (DATA / "train_ok" / c).glob("*.png"))
    ok = sorted(str(p) for c in LIGHTS for p in (DATA / "test_ok" / c).glob("*.png"))
    ng = sorted(str(p) for l in _levels for p in (DATA / "test_ng" / l).glob("*.png"))
    # 学習用の正常を「学習」と「閾値校正」に個体で分ける（NG は使わない）
    calib_ids = {8, 9}
    fit = [p for p in tr if obj_id(p) not in calib_ids]
    cal = [p for p in tr if obj_id(p) in calib_ids]
    return fit, cal, ok, ng



def load_tensor(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x).permute(2, 0, 1)


# ------------------------------------------------------------------ 疑似欠陥
def aug(x, rng):
    s = int(x.shape[-1] * 0.05)
    x = torch.roll(x, shifts=(int(rng.integers(-s, s + 1)), int(rng.integers(-s, s + 1))),
                   dims=(1, 2))
    return x * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.1, 0.1))


def _paste(x, ph, pw, rng):
    """returns (貼り付け後の画像, 貼り付け位置 (dy, dx, ph, pw))"""
    _, h, w = x.shape
    ph, pw = max(3, min(ph, h // 3)), max(3, min(pw, w // 3))
    sy, sx = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
    dy, dx = int(rng.integers(0, h - ph)), int(rng.integers(0, w - pw))
    y = x.clone()
    patch = x[:, sy:sy + ph, sx:sx + pw].clone()
    if rng.random() < 0.5:
        patch = patch * float(rng.uniform(0.6, 1.4))
    y[:, dy:dy + ph, dx:dx + pw] = patch
    return y, (dy, dx, ph, pw)


def cutpaste(x, rng):
    """矩形パッチの貼り付け（CutPaste）。"""
    _, h, w = x.shape
    area = float(rng.uniform(0.02, 0.10)) * h * w
    ar = float(rng.uniform(0.3, 3.3))
    return _paste(x, int(np.sqrt(area / ar)), int(np.sqrt(area * ar)), rng)


def cutpaste_scar(x, rng):
    """細長い矩形の貼り付け（CutPaste-Scar）。

    今回の「難」レベルは幅 1mm 以下の細い線で、矩形パッチとは形が違いすぎた。
    原論文の Scar variant は幅 2〜16px・長さ 10〜25px。
    入力 518px に合わせてスケールしている。
    """
    scale = x.shape[-1] / 256.0
    pw = int(rng.uniform(2, 16) * scale)
    ph = int(rng.uniform(10, 25) * scale)
    if rng.random() < 0.5:
        ph, pw = pw, ph          # 縦横をランダムに入れ替える
    return _paste(x, ph, pw, rng)


def make_fake(x, rng, use_scar):
    if use_scar and rng.random() < 0.5:
        return cutpaste_scar(x, rng)
    return cutpaste(x, rng)


def patch_label(box, size, grid, patch=14):
    """貼り付けた矩形と重なるパッチだけを 1 にしたラベルを作る。

    CLS（画像全体の要約）で判定すると、幅 1mm の傷は全体の中に埋もれる。
    パッチ単位なら「その場所が異常か」を直接学習できる。
    """
    dy, dx, ph, pw = box
    m = torch.zeros(grid, grid)
    y0, y1 = dy // patch, min(grid - 1, (dy + ph) // patch)
    x0, x1 = dx // patch, min(grid - 1, (dx + pw) // patch)
    m[y0:y1 + 1, x0:x1 + 1] = 1.0
    return m.reshape(-1)


# ------------------------------------------------------------------ 学習
def finetune(enc, fit_paths, args, device, seed=0):
    model = enc.model
    for p in model.parameters():
        p.requires_grad = False
    tuned = []
    for blk in model.encoder.layer[-args.blocks:]:
        for p in blk.parameters():
            p.requires_grad = True
            tuned.append(p)
    print(f"  更新: {sum(p.numel() for p in tuned)/1e6:.2f}M / "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M "
          f"（最終 {args.blocks} ブロック）")

    d = model.config.hidden_size
    head = nn.Sequential(nn.Linear(d, 256), nn.ReLU(inplace=True), nn.Linear(256, 2)).to(device)
    # バックボーンとヘッドで学習率を分ける。
    # ヘッドはゼロから学習するので大きな学習率が要るが、同じ値をバックボーンに適用すると
    # 事前学習の特徴が壊れる（初版はこれで運用ウィンドウが縮んだ）。
    opt = torch.optim.AdamW([
        {"params": tuned, "lr": args.lr},
        {"params": list(head.parameters()), "lr": args.head_lr},
    ], weight_decay=1e-4)
    print(f"  学習率: バックボーン {args.lr} / ヘッド {args.head_lr}")

    imgs = torch.stack([load_tensor(p, enc.size) for p in fit_paths])
    rng = np.random.default_rng(seed)
    model.train(); n, bs = len(imgs), 2

    for ep in range(args.epochs):
        idx = rng.permutation(n); tot, cnt = 0.0, 0
        for i in range(0, n, bs):
            b = imgs[idx[i:i + bs]]
            if len(b) < 2:
                continue
            normal = torch.stack([aug(x, rng) for x in b])
            fakes, boxes = zip(*[make_fake(aug(x, rng), rng, args.scar) for x in b])
            xb = torch.cat([normal, torch.stack(fakes)]).to(device)
            h = model(pixel_values=xb).last_hidden_state
            if args.patch_level:
                # パッチ単位で「その場所が異常か」を学習する
                g = enc.grid
                pl = torch.cat([torch.zeros(len(b), g * g),
                                torch.stack([patch_label(bx, enc.size, g) for bx in boxes])])
                logit = head(h[:, -g * g:])                      # [B, P, 2]
                loss = F.cross_entropy(logit.reshape(-1, 2), pl.reshape(-1).long().to(device))
            else:
                yb = torch.cat([torch.zeros(len(b)), torch.ones(len(b))]).long().to(device)
                loss = F.cross_entropy(head(h[:, 0]), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
        print(f"    epoch {ep+1:>2}/{args.epochs}  loss {tot/max(1,cnt):.4f}", flush=True)
    model.eval()
    return head


@torch.no_grad()
def head_score(enc, head, paths, patch_level=False, bs=4):
    """分類ヘッドが出す「異常である確率」をそのまま異常スコアにする。

    patch_level のときは、最も異常なパッチの確率を画像スコアにする。
    """
    out = []
    for i in range(0, len(paths), bs):
        xb = torch.stack([load_tensor(p, enc.size) for p in paths[i:i + bs]]).to(enc.device)
        h = enc.model(pixel_values=xb).last_hidden_state
        if patch_level:
            g = enc.grid
            prob = F.softmax(head(h[:, -g * g:]), dim=-1)[:, :, 1]   # [B, P]
            out.append(prob.max(dim=1).values.float().cpu().numpy())
        else:
            out.append(F.softmax(head(h[:, 0]), dim=-1)[:, 1].float().cpu().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------------ 評価
def summarize(name, cal, s_ok, s_ng, ok_paths, results):
    m = evaluate(s_ok, s_ng)
    z_ok, z_ng = to_sigma(s_ok, cal), to_sigma(s_ng, cal)
    w = operating_window(z_ok, z_ng, max_fp=0.0, min_recall=1.0)
    r = {"auroc": m["auroc"], "window": w["width"], "gap": m["gap"],
         "ok_max": float(z_ok.max()), "ng_min": float(z_ng.min()),
         "fp0": roc_points(z_ok, z_ng, (0.0,))[0]["recall"],
         "by_light": {c: float(np.mean([z for p, z in zip(ok_paths, z_ok) if f"/{c}/" in p]))
                      for c in LIGHTS if any(f"/{c}/" in p for p in ok_paths)}}
    results[name] = r
    bl = "  ".join(f"{c[:4]}{v:+5.1f}" for c, v in r["by_light"].items())
    print(f"  {name:<26}{r['auroc']:>7.3f}{r['window']:>+9.2f}{r['ok_max']:>+8.2f}"
          f"{r['ng_min']:>+8.2f}{r['fp0']*100:>7.0f}%   {bl}")
    return r


def eval_embed(tag, enc, fit, cal, ok, ng, agg, results):
    """returns (cal, ok, ng) のパッチスコア（アンサンブル用）。"""
    c_fit, p_fit = enc.encode(fit)
    c_cal, p_cal = enc.encode(cal)
    c_ok, p_ok = enc.encode(ok)
    c_ng, p_ng = enc.encode(ng)
    g = GlobalScorer().fit(c_fit)
    summarize(f"{tag} 1画像=1ベクトル", g.score(c_cal), g.score(c_ok), g.score(c_ng), ok, results)
    a = lambda p: aggregate_neighbors(p, enc.grid, agg) if agg > 1 else p  # noqa: E731
    mb = PatchMemoryBank(coreset_ratio=1.0).fit(a(p_fit))
    s_cal, s_ok, s_ng = (mb.score(a(x))[0] for x in (p_cal, p_ok, p_ng))
    summarize(f"{tag} パッチ+近傍集約", s_cal, s_ok, s_ng, ok, results)
    return s_cal, s_ok, s_ng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-set", choices=["all", "normal"], default="all")
    ap.add_argument("--levels", default="easy,medium,hard",
                    help="評価に使う異常の難易度（前記事の残課題は hard）")
    ap.add_argument("--score", choices=["head", "embed", "both"], default="both")
    ap.add_argument("--scar", action="store_true", help="CutPaste-Scar を混ぜる")
    ap.add_argument("--patch-level", action="store_true", help="パッチ単位で分類する")
    ap.add_argument("--blocks", type=int, default=1, help="更新する末尾ブロック数")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-6, help="バックボーンの学習率")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="分類ヘッドの学習率")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--agg", type=int, default=3)
    args = ap.parse_args()

    fit, cal, ok, ng = paths(args.train_set, args.levels.split(","))
    print(f"■ 改善版FT  学習 {len(fit)} 枚 / 校正 {len(cal)} 枚 / 評価 OK {len(ok)} NG {len(ng)}")
    print(f"  score={args.score}  scar={args.scar}  blocks={args.blocks} "
          f"lr={args.lr}/{args.head_lr} epochs={args.epochs} patch={args.patch_level}\n")

    results = {}
    enc = Dinov2Encoder(size=args.size)
    hdr = f"  {'手法':<26}{'AUROC':>7}{'運用幅':>9}{'OK最大':>8}{'NG最小':>8}{'検出':>8}   照明別(σ)"
    print("FT前"); print(hdr)
    eval_embed("FT前", enc, fit, cal, ok, ng, args.agg, results)

    t0 = time.time()
    head = finetune(enc, fit, args, enc.device, seed=args.seed)
    print(f"  学習時間 {time.time()-t0:.0f}s\n")

    print("FT後"); print(hdr)
    emb = eval_embed("FT後", enc, fit, cal, ok, ng, args.agg, results)
    h_cal, h_ok, h_ng = (head_score(enc, head, x, args.patch_level) for x in (cal, ok, ng))
    summarize("FT後 分類ヘッド出力", h_cal, h_ok, h_ng, ok, results)

    # アンサンブル: 埋め込み（傷に敏感）と分類ヘッド（照明に頑健）は得意分野が逆なので、
    # それぞれを校正データで σ 正規化してから足す。
    e_cal, e_ok, e_ng = emb
    def z(x, base):
        mu, sd = np.mean(base), np.std(base, ddof=1)
        return (np.asarray(x) - mu) / max(float(sd), 1e-3)   # sd≈0 での発散を防ぐ
    for w in [1.0, 1.5, 2.0, 3.0, 5.0]:
        mix = lambda e, h: z(e, e_cal) + w * z(h, h_cal)   # noqa: E731
        summarize(f"FT後 アンサンブル(w={w})", mix(e_cal, h_cal),
                  mix(e_ok, h_ok), mix(e_ng, h_ng), ok, results)

    tag = (f"{args.train_set}_{'scar' if args.scar else 'plain'}"
           f"_b{args.blocks}_e{args.epochs}_{'patch' if args.patch_level else 'cls'}")
    out = ROOT / "outputs" / f"ft2_{tag}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved: {out.name}")


if __name__ == "__main__":
    main()
