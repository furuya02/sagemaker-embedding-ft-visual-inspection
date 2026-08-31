"""「実 NG 10 枚で AUROC を語ってよいか」を実測で確かめる。

記事計画では「少数 NG では AUROC の信頼区間が広いので分離マージンを主指標にする」と
書いたが、それがどの程度なのかは数字で示さないと説得力がない。

VisA の test 全量（正常 100 / 異常 100）を先に埋め込んでおき、
そこから「正常 30 枚 / 異常 10 枚」をランダムに選ぶ試行を繰り返して、
AUROC がどれだけ振れるかを見る。
"""
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.embed import Dinov2Encoder                        # noqa: E402
from src.memorybank import GlobalScorer, PatchMemoryBank   # noqa: E402
from sklearn.metrics import roc_auc_score                  # noqa: E402


def load_split(data_dir, category):
    rows = defaultdict(list)
    with open(Path(data_dir) / "split_csv" / "1cls.csv") as f:
        for r in csv.DictReader(f):
            if r["object"] == category:
                rows[(r["split"], r["label"])].append(str(Path(data_dir) / r["image"]))
    return rows


def get_scores(args, cache):
    """全量のスコアを一度だけ計算してキャッシュする。"""
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    rows = load_split(args.data_dir, args.category)
    rng = np.random.default_rng(0)
    tr = sorted(rows[("train", "normal")])
    tr = [tr[i] for i in rng.choice(len(tr), min(args.n_train, len(tr)), replace=False)]
    ok, ng = sorted(rows[("test", "normal")]), sorted(rows[("test", "anomaly")])

    enc = Dinov2Encoder(size=args.size)
    print(f"encoding {len(tr)+len(ok)+len(ng)} images (device={enc.device}) ...")
    cls_tr, pat_tr = enc.encode(tr)
    cls_ok, pat_ok = enc.encode(ok)
    cls_ng, pat_ng = enc.encode(ng)

    g = GlobalScorer().fit(cls_tr)
    mb = PatchMemoryBank(coreset_ratio=args.coreset).fit(pat_tr)
    out = {
        "global_ok": g.score(cls_ok), "global_ng": g.score(cls_ng),
        "patch_ok": mb.score(pat_ok)[0], "patch_ng": mb.score(pat_ng)[0],
    }
    np.savez(cache, **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--category", default="candle")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()

    out_dir = ROOT / "outputs"; out_dir.mkdir(exist_ok=True)
    s = get_scores(args, out_dir / f"scores_{args.category}.npz")

    # アヒルの評価セット規模（OK 30 / NG 10）と、比較用にいくつかの規模で試す
    settings = [(30, 10), (30, 30), (100, 100)]
    rng = np.random.default_rng(0)
    report = {}

    print(f"\n=== {args.category}: 評価セットの規模と AUROC のばらつき（{args.trials} 試行）===")
    print(f"{'method':<22}{'n_ok/n_ng':>12}{'mean':>8}{'sd':>7}{'p5':>8}{'p95':>8}{'range':>8}")
    for name, ok_all, ng_all in [("1画像=1ベクトル(CLS)", s["global_ok"], s["global_ng"]),
                                 ("パッチ+メモリバンク", s["patch_ok"], s["patch_ng"])]:
        for n_ok, n_ng in settings:
            aucs = []
            for _ in range(args.trials):
                o = ok_all[rng.choice(len(ok_all), min(n_ok, len(ok_all)), replace=False)]
                n = ng_all[rng.choice(len(ng_all), min(n_ng, len(ng_all)), replace=False)]
                y = np.r_[np.zeros(len(o)), np.ones(len(n))]
                aucs.append(roc_auc_score(y, np.r_[o, n]))
            a = np.array(aucs)
            report[f"{name}/{n_ok}_{n_ng}"] = {
                "mean": float(a.mean()), "sd": float(a.std()),
                "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            }
            print(f"{name:<22}{f'{n_ok}/{n_ng}':>12}{a.mean():>8.3f}{a.std():>7.3f}"
                  f"{np.percentile(a,5):>8.3f}{np.percentile(a,95):>8.3f}"
                  f"{np.percentile(a,95)-np.percentile(a,5):>8.3f}")

    # 2 手法の優劣が n=10 で何回ひっくり返るか
    flips = 0
    for _ in range(args.trials):
        idx_ok = rng.choice(len(s["global_ok"]), 30, replace=False)
        idx_ng = rng.choice(len(s["global_ng"]), 10, replace=False)
        y = np.r_[np.zeros(30), np.ones(10)]
        a_g = roc_auc_score(y, np.r_[s["global_ok"][idx_ok], s["global_ng"][idx_ng]])
        a_p = roc_auc_score(y, np.r_[s["patch_ok"][idx_ok], s["patch_ng"][idx_ng]])
        flips += int(a_g >= a_p)
    print(f"\n全量では パッチ > CLS だが、OK30/NG10 で抽出すると "
          f"{flips}/{args.trials} 回（{flips/args.trials*100:.1f}%）で CLS が勝つ（優劣が逆転する）")
    report["flip_rate_ok30_ng10"] = flips / args.trials

    (out_dir / f"stability_{args.category}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
