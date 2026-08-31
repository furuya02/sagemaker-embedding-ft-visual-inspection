"""撮影した画像の品質をまとめて点検する。

傷をつける前（＝取り返しがつくうち）に問題を見つけるのが目的。
Blog/撮影要領.md の第 8 章「OK 撮影の直後に必ずやるチェック」を自動化したもの。

  python scripts/check_dataset.py
  python scripts/check_dataset.py --sheet   # コンタクトシートも出す
"""
import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "duck"

MIN_SHARP = 0.10     # コントラスト正規化後の値。実測: 合焦 0.18〜0.27 / ボケ 0.006
MIN_MARGIN = 40      # 拡張のシフト量(ROI の 5%)に耐える余白
MAX_CLIP = 0.5       # 白飛び・黒つぶれの許容割合(%)


def _sharp(g):
    """コントラストで正規化したラプラシアン分散（明るさに依存しない）。"""
    gf = g.astype(np.float64); sd = gf.std()
    return 0.0 if sd < 1e-6 else float(cv2.Laplacian(gf / sd, cv2.CV_64F).var())


def duck_bbox(rgb):
    """アヒル(肌色〜オレンジ)の範囲。背景は緑系の布なので色相で分離できる。"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = ((hsv[:, :, 0] < 30) & (hsv[:, :, 1] > 60) & (hsv[:, :, 2] > 80)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    ys, xs = np.where(m > 0)
    return (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else None


def inspect(path):
    a = np.asarray(Image.open(path).convert("RGB"))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    h, w = g.shape
    r = {
        "path": str(path.relative_to(DATA)),
        "size": (w, h),
        "sharp": _sharp(g),
        "mean": g.mean(),
        "clip_hi": (g >= 250).mean() * 100,
        "clip_lo": (g <= 5).mean() * 100,
        "md5": hashlib.md5(a.tobytes()).hexdigest(),
        "warn": [],
    }
    bb = duck_bbox(a)
    if bb:
        x0, x1, y0, y1 = bb
        r["bbox"] = (x1 - x0, y1 - y0)
        r["margin"] = (x0, w - x1, y0, h - y1)
        if min(r["margin"]) < MIN_MARGIN:
            r["warn"].append(f"余白不足 {min(r['margin'])}px")
    else:
        r["bbox"], r["margin"] = None, None
        r["warn"].append("アヒルを検出できず")
    if r["sharp"] < MIN_SHARP:
        r["warn"].append(f"ピント甘い {r['sharp']:.3f}")
    if r["clip_hi"] > MAX_CLIP:
        r["warn"].append(f"白飛び {r['clip_hi']:.1f}%")
    if r["clip_lo"] > MAX_CLIP:
        r["warn"].append(f"黒つぶれ {r['clip_lo']:.1f}%")
    return r


def contact_sheet(paths, out, cols=7, cell=200):
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((cell - 4, cell - 4))
        sheet.paste(im, (i % cols * cell + 2, i // cols * cell + 2))
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", action="store_true", help="コンタクトシートを出力")
    args = ap.parse_args()

    paths = sorted(DATA.rglob("*.png"))
    if not paths:
        print("画像がありません"); return

    res = [inspect(p) for p in paths]

    print(f"{'ファイル':<40}{'サイズ':>10}{'鮮鋭度':>8}{'明るさ':>7}"
          f"{'アヒル':>11}{'最小余白':>9}  警告")
    for r in res:
        bb = f"{r['bbox'][0]}x{r['bbox'][1]}" if r["bbox"] else "—"
        mg = f"{min(r['margin'])}" if r["margin"] else "—"
        print(f"{r['path']:<40}{r['size'][0]}x{r['size'][1]:>4}{r['sharp']:>8.3f}"
              f"{r['mean']:>7.0f}{bb:>11}{mg:>9}  {' / '.join(r['warn'])}")

    # ---- 全体 ----
    print("\n" + "=" * 78)
    sharp = np.array([r["sharp"] for r in res])
    sizes = {r["size"] for r in res}
    dup = {}
    for r in res:
        dup.setdefault(r["md5"], []).append(r["path"])
    dups = [v for v in dup.values() if len(v) > 1]
    warns = [r for r in res if r["warn"]]

    print(f"枚数            : {len(res)}")
    print(f"サイズ          : {'すべて ' + str(sizes.pop()) if len(sizes) == 1 else '★不揃い ' + str(sizes)}")
    print(f"鮮鋭度          : min {sharp.min():.3f} / 平均 {sharp.mean():.3f} / max {sharp.max():.3f}  (必要 {MIN_SHARP} 以上)")
    if r["margin"]:
        mm = min(min(x["margin"]) for x in res if x["margin"])
        print(f"最小余白        : {mm}px  (必要 {MIN_MARGIN}px 以上)")
    print(f"完全に同一の画像: {'なし' if not dups else '★あり ' + str(dups)}")

    # 画像同士の似すぎ検出（同じ個体を撮り続けていないか）
    small = [cv2.resize(np.asarray(Image.open(p).convert("L")), (64, 64)).astype(np.float32)
             for p in paths]
    sims = []
    for i in range(len(small)):
        for j in range(i + 1, len(small)):
            d = np.abs(small[i] - small[j]).mean()
            sims.append((d, res[i]["path"], res[j]["path"]))
    if sims:
        sims.sort()
        print(f"最も似ている組  : 差 {sims[0][0]:.1f}  {sims[0][1]} vs {sims[0][2]}")
        print(f"                  (差が 2 未満だと同じ個体の可能性)")

    print(f"\n警告のある画像  : {len(warns)} 枚")
    for r in warns:
        print(f"  ★ {r['path']}  {' / '.join(r['warn'])}")
    if not warns:
        print("  問題なし")

    if args.sheet:
        out = ROOT / "outputs" / "contact_sheet.png"
        out.parent.mkdir(exist_ok=True)
        contact_sheet(paths, out)
        print(f"\nコンタクトシート: {out}")


if __name__ == "__main__":
    main()
