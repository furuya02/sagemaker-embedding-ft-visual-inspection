"""評価指標。

実 NG が 10 枚しかないため AUROC は信頼区間が広い。
そこで主指標は「分離マージン」（OK と NG のスコア分布がどれだけ離れているか）とし、
AUROC は参考値として併記する。
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(ok_scores, ng_scores):
    """スコアは「大きいほど異常」。"""
    ok, ng = np.asarray(ok_scores), np.asarray(ng_scores)

    y = np.r_[np.zeros(len(ok)), np.ones(len(ng))]
    s = np.r_[ok, ng]

    # gap: NG の最小値 - OK の最大値。正なら完全に分離できている（閾値が引ける）
    gap = float(ng.min() - ok.max())

    # 平均差を、ばらつきで割って正規化したもの（Cohen's d）
    pooled = np.sqrt((ok.var(ddof=1) + ng.var(ddof=1)) / 2) if len(ok) > 1 and len(ng) > 1 else np.nan
    d = float((ng.mean() - ok.mean()) / pooled) if pooled and pooled > 0 else np.nan

    return {
        "auroc": float(roc_auc_score(y, s)),
        "gap": gap,                                  # 主指標: 正なら完全分離
        "margin": float(ng.mean() - ok.mean()),      # 平均の差
        "cohens_d": d,
        "ok_mean": float(ok.mean()), "ok_max": float(ok.max()),
        "ng_mean": float(ng.mean()), "ng_min": float(ng.min()),
        "n_ok": len(ok), "n_ng": len(ng),
    }


def to_sigma(scores, calib_normal):
    """正常スコアの分布で標準化する（正常の平均を 0、ばらつきを 1 とする）。

    手法ごとにスコアの絶対値のスケールが違う（CLS は 0.02 付近、パッチは 0.29 付近）ため、
    そのままでは「閾値の余裕」を比較できない。
    正常のばらつきを 1 単位に取れば「正常のブレ何個分の余裕があるか」として比較できる。
    """
    c = np.asarray(calib_normal)
    mu, sd = c.mean(), c.std(ddof=1)
    # 校正データが少ないと sd が極端に小さくなり、割り算で値が発散する。
    # 実測では 0.001 を下回ることはなかったので、そこで下限を切る。
    return (np.asarray(scores) - mu) / max(float(sd), 1e-3)


def operating_window(ok_scores, ng_scores, max_fp=0.05, min_recall=0.90):
    """**閾値をどれだけ自由に選べるか**を測る。

    閾値は「モデルを準備する時点」では決めなくてよく、運用に入る段階で現場が決めればよい。
    そのとき効いてくるのは「どこに引いても成立する幅がどれだけあるか」であって、
    特定の 1 点の成績ではない。幅が広いほど、現場での調整が楽になり、
    ライン変更や経時変化に対しても壊れにくい。

    「過検出 <= max_fp」かつ「検出率 >= min_recall」を同時に満たす閾値の範囲を返す。
    入力は to_sigma() で標準化済みのスコアを想定（幅の単位が「正常のばらつき何個分」になる）。

    閾値を上げると過検出は減り、検出率も下がる（どちらも単調）。したがって
      lo = 過検出 <= max_fp を満たす最小の閾値（これ以上に上げれば過検出は許容内）
      hi = 検出率 >= min_recall を満たす最大の閾値（これ以下に下げれば検出は足りる）
    となり、width = hi - lo が正なら「その幅のどこに引いても両立する」。
    **負でもそのまま返す**（両立不可のとき、あとどれだけ足りないかが分かるため）。
    """
    ok, ng = np.asarray(ok_scores), np.asarray(ng_scores)
    cands = np.unique(np.r_[ok, ng, ok.max() + 1.0, ng.min() - 1.0])

    fp_ok = [t for t in cands if (ok > t).mean() <= max_fp]
    rc_ok = [t for t in cands if (ng > t).mean() >= min_recall]
    if not fp_ok or not rc_ok:
        return {"width": float("nan"), "lo": None, "hi": None,
                "max_fp": max_fp, "min_recall": min_recall}

    lo, hi = float(min(fp_ok)), float(max(rc_ok))
    return {"width": hi - lo, "lo": lo, "hi": hi,
            "feasible": bool(hi > lo), "max_fp": max_fp, "min_recall": min_recall}


def roc_points(ok_scores, ng_scores, fp_targets=(0.0, 0.01, 0.05, 0.10)):
    """「過検出を X% 許容したとき、検出率はどこまで伸びるか」。

    運用では過検出の許容量から入ることが多いので、その視点での表。
    """
    ok, ng = np.asarray(ok_scores), np.asarray(ng_scores)
    out = []
    for fp in fp_targets:
        # 過検出が fp 以下になる最小の閾値 = 最も検出率が高くなる点
        cands = np.unique(np.r_[ok, ng, ok.max() + 1.0])
        feas = [t for t in cands if (ok > t).mean() <= fp]
        if not feas:
            out.append({"max_fp": fp, "threshold": None, "recall": 0.0})
            continue
        t = float(min(feas))
        out.append({"max_fp": fp, "threshold": t,
                    "recall": float((ng > t).mean()),
                    "actual_fp": float((ok > t).mean())})
    return out


def threshold_from_normal(normal_scores, method="p99", k=3.0):
    """**正常スコアだけ**から閾値を決める。

    この記事のコンセプトは「本番の準備に不良画像を 1 枚も使わない」なので、
    閾値を OK と NG の分布を見比べて引いてはいけない（それは NG を使った準備になる）。
    NG を使ってよいのは、決めた閾値が何枚取れたかを**後から評価する**ときだけ。

    前記事は「0.9」という固定値だったが、これは根拠のない数字だった。
      p99  : 正常スコアの 99 パーセンタイル（正常の 1% を過検出として許容する）
      max  : 正常スコアの最大値（過検出ゼロを狙う。外れ値に弱い）
      sigma: 平均 + k*標準偏差
    """
    s = np.asarray(normal_scores)
    if method == "p99":
        return float(np.percentile(s, 99))
    if method == "max":
        return float(s.max())
    if method == "sigma":
        return float(s.mean() + k * s.std(ddof=1))
    raise ValueError(method)


def apply_threshold(thr, ok_scores, ng_scores):
    """決めた閾値の実運用上の成績。ここで初めて NG を使う（＝試験）。"""
    ok, ng = np.asarray(ok_scores), np.asarray(ng_scores)
    return {
        "threshold": float(thr),
        "recall": float((ng > thr).mean()),        # 不良を何割拾えたか（見逃しの裏返し）
        "false_positive": float((ok > thr).mean()),  # 良品を何割はじいたか（過検出）
        "detected": int((ng > thr).sum()), "n_ng": len(ng),
        "over_rejected": int((ok > thr).sum()), "n_ok": len(ok),
    }


def fmt(name, m):
    return (f"{name:<28} AUROC={m['auroc']:.3f}  gap={m['gap']:+.4f}  "
            f"margin={m['margin']:.4f}  d={m['cohens_d']:.2f}  "
            f"(OK n={m['n_ok']} / NG n={m['n_ng']})")


def plot_hist(ok_scores, ng_scores, title, path, bins=30):
    """OK と NG のスコア分布を重ねて描く。少数サンプルでは AUROC より正直に伝わる。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # macOS の日本語フォント（無ければ既定のまま）
    matplotlib.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(ok_scores, bins=bins, alpha=0.6, label=f"OK (n={len(ok_scores)})")
    ax.hist(ng_scores, bins=bins, alpha=0.6, label=f"NG (n={len(ng_scores)})")
    ax.set_xlabel("anomaly score"); ax.set_ylabel("count")
    ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
