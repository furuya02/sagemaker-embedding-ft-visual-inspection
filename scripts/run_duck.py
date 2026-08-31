"""アヒルの実データで「1画像=1ベクトル」と「パッチ単位」を比較する。

前記事は Titan Multimodal Embeddings で 1 画像を 1 本のベクトルにして
コサイン類似度 0.9 を閾値にしていた。その構成で微細な異常が拾えなかったのが残課題。
真因が「モデルの性能」ではなく「1 画像を 1 ベクトルに潰していること」なのかを確かめる。

  python scripts/run_duck.py
  python scripts/run_duck.py --size 616 --agg 3
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embed import Dinov2Encoder, aggregate_neighbors          # noqa: E402
from src.memorybank import GlobalScorer, PatchMemoryBank, l2norm  # noqa: E402
from src.metrics import (evaluate, to_sigma, operating_window,    # noqa: E402
                         roc_points, plot_hist)

DATA = ROOT / "data" / "duck"
LIGHTS = ["normal", "side", "bright"]
LEVELS = ["easy", "medium", "hard"]


def load():
    train = sorted(p for c in LIGHTS for p in (DATA / "train_ok" / c).glob("*.png"))
    test_ok = sorted(p for c in LIGHTS for p in (DATA / "test_ok" / c).glob("*.png"))
    test_ng = sorted(p for l in LEVELS for p in (DATA / "test_ng" / l).glob("*.png"))
    return train, test_ok, test_ng


def tag_of(p, keys):
    return next((k for k in keys if f"/{k}/" in str(p)), "?")


def report(name, s_cal, s_ok, s_ng, ok_paths, ng_paths, out_png, results):
    """s_cal は「バンクに入っている正常」を自分抜きで測ったスコア（NG は使わない）。"""
    m = evaluate(s_ok, s_ng)
    z_ok, z_ng = to_sigma(s_ok, s_cal), to_sigma(s_ng, s_cal)
    results[name] = {"auroc": m["auroc"], "gap": m["gap"]}

    print(f"\n{'='*74}\n{name}")
    print(f"  AUROC {m['auroc']:.3f}   gap {m['gap']:+.4f}   "
          f"OK {m['ok_mean']:.4f}±  NG {m['ng_mean']:.4f}")

    print("  過検出の許容量ごとの検出率")
    for p in roc_points(z_ok, z_ng):
        thr = f"{p['threshold']:+.2f}σ" if p["threshold"] is not None else "—"
        print(f"    過検出<={p['max_fp']*100:>4.0f}% : 検出 {p['recall']*100:5.1f}%  (閾値 {thr})")
        results[f"{name}/fp{p['max_fp']}"] = p["recall"]

    print("  閾値の選べる幅（σ。負は両立不可でその不足分）")
    for fp, rc in [(0.00, 1.00), (0.05, 0.90), (0.10, 0.80)]:
        w = operating_window(z_ok, z_ng, max_fp=fp, min_recall=rc)
        mark = "○" if w.get("feasible") else "×"
        print(f"    {mark} 過検出<={fp*100:>4.0f}% かつ 検出>={rc*100:>4.0f}% : {w['width']:>+6.2f}σ")
        results[f"{name}/win_{fp}_{rc}"] = w["width"]

    # 難易度別（記事の主眼。易は取れて当然、難が取れるか）
    print("  難易度別の異常スコア（σ）  ※OK の最大 = " f"{z_ok.max():+.2f}σ")
    by = defaultdict(list)
    for p, z in zip(ng_paths, z_ng):
        by[tag_of(p, LEVELS)].append(z)
    for lv in LEVELS:
        if lv in by:
            v = np.array(by[lv])
            over = (v > z_ok.max()).sum()
            print(f"    {lv:<7} n={len(v)}  平均 {v.mean():+6.2f}σ  "
                  f"最小 {v.min():+6.2f}σ   OK最大を超えた数 {over}/{len(v)}")
            results[f"{name}/lv_{lv}"] = float(v.mean())

    # 照明条件別の正常スコア（分布シフトの影響）
    by = defaultdict(list)
    for p, z in zip(ok_paths, z_ok):
        by[tag_of(p, LIGHTS)].append(z)
    print("  照明条件別の正常スコア（σ）  ※大きいほど異常寄り＝過検出の原因")
    for c in LIGHTS:
        if c in by:
            v = np.array(by[c])
            print(f"    {c:<7} n={len(v)}  平均 {v.mean():+6.2f}σ  最大 {v.max():+6.2f}σ")

    plot_hist(z_ok, z_ng, f"duck  {name}", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--agg", type=int, default=3, help="局所近傍集約のカーネル。1で無効")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train, test_ok, test_ng = load()
    print(f"train_ok {len(train)}  test_ok {len(test_ok)}  test_ng {len(test_ng)}")

    enc = Dinov2Encoder(size=args.size)
    print(f"dinov2-small  device={enc.device}  size={args.size}  "
          f"{enc.grid}x{enc.grid}={enc.grid**2} patches")
    t0 = time.time()
    cls_tr, pat_tr = enc.encode([str(p) for p in train])
    cls_ok, pat_ok = enc.encode([str(p) for p in test_ok])
    cls_ng, pat_ng = enc.encode([str(p) for p in test_ng])
    print(f"encoded {len(train)+len(test_ok)+len(test_ng)} images in {time.time()-t0:.0f}s")

    results = {}

    # (A) 1 画像 = 1 ベクトル（前記事の方式）
    g = GlobalScorer().fit(cls_tr)
    # 自分自身を除いた正常スコア（NG は使わない）
    sims = l2norm(cls_tr) @ l2norm(cls_tr).T
    np.fill_diagonal(sims, -np.inf)
    cal = 1.0 - sims.max(axis=1)
    report("[A] 1画像=1ベクトル (CLS)", cal, g.score(cls_ok), g.score(cls_ng),
           test_ok, test_ng, out / "duck_A_global.png", results)

    # (B) パッチ + メモリバンク
    mb = PatchMemoryBank(coreset_ratio=1.0).fit_loo(pat_tr)
    report("[B] パッチ+メモリバンク", mb.score_loo(), mb.score(pat_ok)[0], mb.score(pat_ng)[0],
           test_ok, test_ng, out / "duck_B_patch.png", results)

    # (C) パッチ + 局所近傍集約
    if args.agg > 1:
        ag = lambda p: aggregate_neighbors(p, enc.grid, args.agg)  # noqa: E731
        mb2 = PatchMemoryBank(coreset_ratio=1.0).fit_loo(ag(pat_tr))
        report(f"[C] パッチ+近傍集約{args.agg}x{args.agg}", mb2.score_loo(),
               mb2.score(ag(pat_ok))[0], mb2.score(ag(pat_ng))[0],
               test_ok, test_ng, out / "duck_C_agg.png", results)

    (out / "duck_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
