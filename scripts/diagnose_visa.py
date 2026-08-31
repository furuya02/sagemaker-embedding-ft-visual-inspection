"""「パッチにすれば勝てる」わけではなかったので、原因を切り分ける。

検証する仮説:
  H1: coreset が小さすぎてメモリバンクが正常のばらつきをカバーできていない
  H2: 正方形への強制リサイズで画像が歪んでいる（VisA は横長）
  H3: 欠陥が小さすぎてパッチ解像度に埋もれている（マスクから面積比を測る）

埋め込みは npz にキャッシュして使い回す。
"""
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.embed import Dinov2Encoder                        # noqa: E402
from src.memorybank import GlobalScorer, PatchMemoryBank   # noqa: E402
from src.metrics import evaluate                           # noqa: E402


def load_split(data_dir, category):
    rows = defaultdict(list)
    with open(Path(data_dir) / "split_csv" / "1cls.csv") as f:
        for r in csv.DictReader(f):
            if r["object"] == category:
                rows[(r["split"], r["label"])].append(
                    (str(Path(data_dir) / r["image"]), r["mask"]))
    return rows


def defect_area_ratio(data_dir, mask_rels):
    """H3: 欠陥がフレームの何 % を占めるか。パッチ解像度と比較するため。"""
    ratios = []
    for rel in mask_rels:
        if not rel:
            continue
        p = Path(data_dir) / rel
        if not p.exists():
            continue
        # VisA のマスクは 0/255 ではなく 0/1。>127 で二値化すると全て 0 になる（実際に踏んだ）
        m = np.asarray(Image.open(p).convert("L"))
        ratios.append((m > 0).mean())
    return np.array(ratios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--category", default="candle")
    ap.add_argument("--n-train", type=int, default=27)
    ap.add_argument("--n-test-ok", type=int, default=30)
    ap.add_argument("--n-test-ng", type=int, default=10)
    ap.add_argument("--size", type=int, default=518)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = load_split(args.data_dir, args.category)

    def pick(key, n):
        items = sorted(rows[key])
        idx = rng.choice(len(items), min(n, len(items)), replace=False)
        return [items[i] for i in idx]

    train = pick(("train", "normal"), args.n_train)
    t_ok = pick(("test", "normal"), args.n_test_ok)
    t_ng = pick(("test", "anomaly"), args.n_test_ng)

    # ---------------- H3: 欠陥の面積比 ----------------
    ratios = defect_area_ratio(args.data_dir, [m for _, m in t_ng])
    im0 = Image.open(train[0][0])
    grid = args.size // 14
    print(f"=== {args.category} ===")
    print(f"元画像サイズ: {im0.size}  アスペクト比 {im0.size[0]/im0.size[1]:.2f}")
    if len(ratios):
        print(f"[H3] 欠陥の面積比: mean={ratios.mean()*100:.2f}%  "
              f"min={ratios.min()*100:.2f}%  max={ratios.max()*100:.2f}%")
        print(f"     1 パッチ = 全体の {100/grid**2:.3f}%  "
              f"→ 欠陥は平均 {ratios.mean()*grid**2:.1f} パッチ分")

    # ---------------- 埋め込み（2 通りの前処理） ----------------
    results = {}
    for mode in ["squash", "aspect"]:
        enc = Dinov2Encoder(size=args.size)
        if mode == "aspect":
            # H2: アスペクト比を保って短辺を size に合わせ、中央を正方形にクロップ
            def _load(path, _enc=enc):
                im = Image.open(path).convert("RGB")
                w, h = im.size
                s = _enc.size / min(w, h)
                im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
                w, h = im.size
                l, t = (w - _enc.size) // 2, (h - _enc.size) // 2
                im = im.crop((l, t, l + _enc.size, t + _enc.size))
                from src.embed import MEAN, STD
                import torch
                x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
                return torch.from_numpy(x).permute(2, 0, 1)
            enc._load = _load

        t0 = time.time()
        cls_tr, pat_tr = enc.encode([p for p, _ in train])
        cls_ok, pat_ok = enc.encode([p for p, _ in t_ok])
        cls_ng, pat_ng = enc.encode([p for p, _ in t_ng])
        print(f"\n--- 前処理: {mode} ({time.time()-t0:.1f}s) ---")

        g = GlobalScorer().fit(cls_tr)
        m = evaluate(g.score(cls_ok), g.score(cls_ng))
        results[f"{mode}/global"] = m
        print(f"  1画像=1ベクトル(CLS)        AUROC={m['auroc']:.3f}  gap={m['gap']:+.4f}")

        # ---------------- H1: coreset を振る ----------------
        for cr in [0.1, 0.25, 0.5, 1.0]:
            mb = PatchMemoryBank(coreset_ratio=cr).fit(pat_tr)
            s_ok, _ = mb.score(pat_ok)
            s_ng, _ = mb.score(pat_ng)
            m = evaluate(s_ok, s_ng)
            results[f"{mode}/patch_cr{cr}"] = m
            print(f"  パッチ+バンク coreset={cr:<4}  AUROC={m['auroc']:.3f}  "
                  f"gap={m['gap']:+.4f}  bank={mb.bank.shape[0]}")

    Path(ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / f"diagnose_{args.category}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
