"""VisA（CC BY 4.0）で実装を検証する。

アヒルの撮影データが揃う前に、既知のベンチマークで
「1 画像 = 1 ベクトル」と「パッチ + メモリバンク」の差を確認しておくのが目的。
アヒルで数字が出なかったときに、実装ミスかデータの問題かを切り分けられる。

使い方:
  python scripts/run_visa.py --category candle --n-train 27
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embed import Dinov2Encoder, aggregate_neighbors  # noqa: E402
from src.memorybank import GlobalScorer, PatchMemoryBank  # noqa: E402
from src.metrics import (evaluate, fmt, plot_hist, roc_points,  # noqa: E402
                         threshold_from_normal, apply_threshold,
                         to_sigma, operating_window)


def load_split(data_dir, category):
    """公式の 1cls 分割（split_csv/1cls.csv）を使う。"""
    rows = defaultdict(list)
    with open(Path(data_dir) / "split_csv" / "1cls.csv") as f:
        for r in csv.DictReader(f):
            if r["object"] != category:
                continue
            rows[(r["split"], r["label"])].append(str(Path(data_dir) / r["image"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--category", default="candle")
    ap.add_argument("--n-train", type=int, default=27,
                    help="学習(=メモリバンク登録)に使う正常画像の枚数。アヒルの実写 27 枚に合わせた既定値")
    ap.add_argument("--n-test-ok", type=int, default=30)
    ap.add_argument("--n-test-ng", type=int, default=10)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--calib-ratio", type=float, default=0.25,
                    help="正常画像のうち閾値決定に回す割合。NG は使わない")
    ap.add_argument("--aggregate", type=int, default=3, help="局所近傍集約のカーネルサイズ。1で無効")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows = load_split(args.data_dir, args.category)
    train = sorted(rows[("train", "normal")])
    test_ok = sorted(rows[("test", "normal")])
    test_ng = sorted(rows[("test", "anomaly")])

    # n<=0 なら全量を使う。VisA は test に normal/anomaly が各 100 枚あるので、
    # 実装の妥当性検証では全量を使い、アヒルの規模を模す実験とは分けて考える。
    def sub(items, n):
        if n is None or n <= 0 or n >= len(items):
            return items
        return [items[i] for i in rng.choice(len(items), n, replace=False)]

    train = sub(train, args.n_train)
    test_ok = sub(test_ok, args.n_test_ok)
    test_ng = sub(test_ng, args.n_test_ng)

    # 正常画像を「バンク登録用」と「閾値決定(キャリブレーション)用」に分ける。
    # 閾値を決めるための正常スコアは、バンクに入っていない正常でなければ
    # 楽観的に低く出る（自分自身が最近傍になるため）。
    # NG は一切使わないので「準備は正常画像だけ」というコンセプトは保たれる。
    n_calib = max(1, int(len(train) * args.calib_ratio))
    calib, train = train[:n_calib], train[n_calib:]

    print(f"category={args.category}  bank={len(train)}  calib_ok={len(calib)}  "
          f"test_ok={len(test_ok)}  test_ng={len(test_ng)}")

    enc = Dinov2Encoder(size=args.size)
    print(f"encoder: dinov2-small  device={enc.device}  size={args.size}  "
          f"grid={enc.grid}x{enc.grid}={enc.grid**2} patches  dim={enc.dim}")

    t0 = time.time()
    cls_tr, pat_tr = enc.encode(train)
    cls_ca, pat_ca = enc.encode(calib)
    cls_ok, pat_ok = enc.encode(test_ok)
    cls_ng, pat_ng = enc.encode(test_ng)
    print(f"encoded {len(train)+len(calib)+len(test_ok)+len(test_ng)} images in {time.time()-t0:.1f}s")

    results = {}

    def report(tag, label, s_ca, s_ok, s_ng, png):
        m = evaluate(s_ok, s_ng)
        results[tag] = m
        print(fmt(label, m))

        # 正常のばらつきを 1 単位に標準化して、手法間で「閾値の余裕」を比較できるようにする
        z_ok = to_sigma(s_ok, s_ca)
        z_ng = to_sigma(s_ng, s_ca)

        # 運用の入り口は「過検出をどこまで許すか」なので、その視点で検出率を見る
        print("     過検出の許容量ごとの検出率")
        pts = roc_points(z_ok, z_ng)
        results[f"{tag}/roc_points"] = pts
        for p in pts:
            thr = f"{p['threshold']:+.2f}σ" if p["threshold"] is not None else "—"
            print(f"       過検出<={p['max_fp']*100:>4.0f}% : 検出 {p['recall']*100:5.1f}%"
                  f"   (閾値 {thr})")

        # 主指標: 閾値をどれだけ自由に選べるか（運用時に現場が決められる幅）
        print("     閾値の選べる幅（単位: 正常のばらつき σ。負は両立不可でその不足分）")
        for fp, rc in [(0.00, 1.00), (0.05, 0.90), (0.10, 0.80)]:
            w = operating_window(z_ok, z_ng, max_fp=fp, min_recall=rc)
            results[f"{tag}/window_fp{fp}_rc{rc}"] = w
            mark = "○" if w.get("feasible") else "×"
            print(f"       {mark} 過検出<={fp*100:>4.0f}% かつ 検出>={rc*100:>4.0f}% : "
                  f"幅 {w['width']:>+6.2f}σ")
        plot_hist(z_ok, z_ng, f"VisA/{args.category}  {label}", png)
        return m

    # (A) 1 画像 = 1 ベクトル（前記事の方式）
    g = GlobalScorer().fit(cls_tr)
    a = report("global_cls", "[A] 1画像=1ベクトル (CLS)",
               g.score(cls_ca), g.score(cls_ok), g.score(cls_ng),
               out / f"{args.category}_global.png")

    # (B) パッチ + メモリバンク（training-free、2026 年の定石）
    mb = PatchMemoryBank(coreset_ratio=args.coreset).fit(pat_tr)
    print(f"    memory bank: {mb.bank.shape[0]} patches (coreset={args.coreset})")
    b = report("patch_memorybank", "[B] パッチ+メモリバンク",
               mb.score(pat_ca)[0], mb.score(pat_ok)[0], mb.score(pat_ng)[0],
               out / f"{args.category}_patch.png")

    # (C) パッチ + 局所近傍集約（PatchCore の工夫）
    if args.aggregate > 1:
        ag = lambda p: aggregate_neighbors(p, enc.grid, args.aggregate)  # noqa: E731
        mb2 = PatchMemoryBank(coreset_ratio=args.coreset).fit(ag(pat_tr))
        c = report("patch_agg", f"[C] パッチ+近傍集約{args.aggregate}x{args.aggregate}",
                   mb2.score(ag(pat_ca))[0], mb2.score(ag(pat_ok))[0], mb2.score(ag(pat_ng))[0],
                   out / f"{args.category}_patch_agg.png")

    print("\n--- 差分 ---")
    print(f"AUROC  {a['auroc']:.3f} -> {b['auroc']:.3f}  ({b['auroc']-a['auroc']:+.3f})")
    print(f"gap    {a['gap']:+.4f} -> {b['gap']:+.4f}   (正なら完全分離)")

    import json
    (out / f"{args.category}_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
