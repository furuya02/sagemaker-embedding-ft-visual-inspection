"""学習セットの構成を変えて、どこで手法の差が出るかを調べる。

条件を固定した状態では「1 画像 = 1 ベクトル」でも完璧に分離できてしまった
（run_duck.py の結果）。そこで学習データの側を崩して比較する。

  実験1 分布シフト : 学習を normal だけにし、未知の照明(side/bright)で過検出が出るか
  実験2 枚数スイープ: 実写を増やすのと拡張で水増しするのとどちらが効くか

  python scripts/run_experiments.py
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

from src.augment import expand                                    # noqa: E402
from src.embed import Dinov2Encoder, aggregate_neighbors          # noqa: E402
from src.memorybank import GlobalScorer, PatchMemoryBank, l2norm  # noqa: E402
from src.metrics import evaluate, to_sigma, operating_window      # noqa: E402

DATA = ROOT / "data" / "duck"
LIGHTS = ["normal", "side", "bright"]
LEVELS = ["easy", "medium", "hard"]
CACHE = ROOT / "outputs" / "emb_cache.npz"


def obj_id(p):
    return int(Path(p).stem.split("_")[-1])


def train_paths(lights=LIGHTS, ids=range(1, 10)):
    return sorted(str(p) for c in lights for p in (DATA / "train_ok" / c).glob("*.png")
                  if obj_id(p) in ids)


def test_paths():
    ok = sorted(str(p) for c in LIGHTS for p in (DATA / "test_ok" / c).glob("*.png"))
    ng = sorted(str(p) for l in LEVELS for p in (DATA / "test_ng" / l).glob("*.png"))
    return ok, ng


class Encoder:
    """実写画像の埋め込みをキャッシュして使い回す。"""

    def __init__(self, size, agg):
        self.enc = Dinov2Encoder(size=size)
        self.agg = agg
        self.cls, self.pat = {}, {}
        if CACHE.exists():
            z = np.load(CACHE, allow_pickle=True)
            if int(z["size"]) == size:
                self.cls = dict(z["cls"].item())
                self.pat = dict(z["pat"].item())

    def get(self, paths):
        new = [p for p in paths if p not in self.cls]
        if new:
            print(f"    encoding {len(new)} images ...", flush=True)
            c, p = self.enc.encode(new)
            for i, path in enumerate(new):
                self.cls[path], self.pat[path] = c[i], p[i]
        return (np.stack([self.cls[p] for p in paths]),
                np.stack([self.pat[p] for p in paths]))

    def save(self):
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, size=self.enc.size, cls=self.cls, pat=self.pat)

    def agg_fn(self, pat):
        return aggregate_neighbors(pat, self.enc.grid, self.agg) if self.agg > 1 else pat


def run_one(enc, tr_paths, ok_paths, ng_paths):
    """1 つの学習構成について、CLS 版とパッチ版の成績を返す。"""
    cls_tr, pat_tr = enc.get(tr_paths)
    cls_ok, pat_ok = enc.get(ok_paths)
    cls_ng, pat_ng = enc.get(ng_paths)
    out = {}

    # --- [A] 1 画像 = 1 ベクトル
    g = GlobalScorer().fit(cls_tr)
    sims = l2norm(cls_tr) @ l2norm(cls_tr).T
    np.fill_diagonal(sims, -np.inf)
    cal = 1.0 - sims.max(axis=1)
    out["A"] = _score(cal, g.score(cls_ok), g.score(cls_ng), ok_paths, ng_paths)

    # --- [C] パッチ + 近傍集約
    a_tr, a_ok, a_ng = (enc.agg_fn(x) for x in (pat_tr, pat_ok, pat_ng))
    mb = PatchMemoryBank(coreset_ratio=1.0).fit_loo(a_tr)
    out["C"] = _score(mb.score_loo(), mb.score(a_ok)[0], mb.score(a_ng)[0], ok_paths, ng_paths)
    return out


def _score(cal, s_ok, s_ng, ok_paths, ng_paths):
    m = evaluate(s_ok, s_ng)
    z_ok, z_ng = to_sigma(s_ok, cal), to_sigma(s_ng, cal)
    w = operating_window(z_ok, z_ng, max_fp=0.0, min_recall=1.0)
    by_light = {c: float(np.mean([z for p, z in zip(ok_paths, z_ok) if f"/{c}/" in p]))
                for c in LIGHTS}
    by_lv = {l: float(np.mean([z for p, z in zip(ng_paths, z_ng) if f"/{l}/" in p]))
             for l in LEVELS}
    return {"auroc": m["auroc"], "window": w["width"], "ok_max": float(z_ok.max()),
            "ng_min": float(z_ng.min()), "by_light": by_light, "by_level": by_lv,
            "n_over": int((z_ok > z_ng.min()).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--agg", type=int, default=3)
    args = ap.parse_args()

    ok_paths, ng_paths = test_paths()
    enc = Encoder(args.size, args.agg)
    aug_dir = ROOT / "outputs" / "aug"

    configs = [
        ("全27枚（基準）",            train_paths(), 0),
        ("normal 9枚のみ",           train_paths(lights=["normal"]), 0),
        ("S1 実写9枚(3個体)",         train_paths(ids=[1, 2, 3]), 0),
        ("S2 実写27枚(9個体)",        train_paths(), 0),
        ("S3 実写9枚+拡張→27",        train_paths(ids=[1, 2, 3]), 27),
        ("S4 実写27枚+拡張→100",      train_paths(), 100),
    ]

    results = {}
    for name, paths, target in configs:
        t0 = time.time()
        use = paths
        if target:
            use = expand(paths, target, aug_dir / name.split()[0], seed=0)
        print(f"\n■ {name}   学習 {len(use)} 枚")
        results[name] = run_one(enc, use, ok_paths, ng_paths)
        print(f"    ({time.time()-t0:.0f}s)")
        enc.save()

    # ---------------- まとめ ----------------
    print("\n" + "=" * 92)
    print(f"{'構成':<24}{'手法':<6}{'AUROC':>7}{'運用幅':>9}{'OK最大':>8}{'NG最小':>8}"
          f"{'逆転':>6}   照明別の正常スコア(σ)")
    for name, r in results.items():
        for k, label in [("A", "1ベク"), ("C", "パッチ")]:
            v = r[k]
            bl = "  ".join(f"{c[:4]}{v['by_light'][c]:+5.1f}" for c in LIGHTS)
            print(f"{name:<24}{label:<6}{v['auroc']:>7.3f}{v['window']:>+9.2f}"
                  f"{v['ok_max']:>+8.2f}{v['ng_min']:>+8.2f}{v['n_over']:>6}   {bl}")

    print("\n難易度別の異常スコア(σ)")
    print(f"{'構成':<24}{'手法':<6}" + "".join(f"{l:>10}" for l in LEVELS))
    for name, r in results.items():
        for k, label in [("A", "1ベク"), ("C", "パッチ")]:
            print(f"{name:<24}{label:<6}"
                  + "".join(f"{r[k]['by_level'][l]:>+10.1f}" for l in LEVELS))

    (ROOT / "outputs" / "experiments.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
