"""記事用の図を生成する。出力は ../Blog/0XX.png。

  006 照明 3 条件の模式図（影の向きが変わることを示す）
  007 3 条件の作例（同一個体で影の位置が違う）
  008 NG の before / after（hard 3 個）
  009 運用ウィンドウの比較グラフ（記事の主張そのもの）
  010 異常スコアのヒストグラム
  011 パッチ単位のヒートマップ
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from PIL import Image, ImageDraw         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BLOG = ROOT.parent / "Blog"
DATA = ROOT / "data" / "duck"
LIGHTS = ["normal", "side", "bright"]


# ------------------------------------------------------------ 006 照明の模式図
def fig_lighting():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    cfg = [("通常", "蛍光灯のみ", [(0.25, 0.92)], "左上", (0.30, 0.30)),
           ("側光", "デスクライトのみ", [(0.80, 0.80)], "右上", (0.72, 0.30)),
           ("明るい", "蛍光灯 + デスクライト", [(0.25, 0.92), (0.80, 0.80)], "打ち消し合う", None)]
    for ax, (name, src, lamps, shadow, spos) in zip(axes, cfg):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(plt.Rectangle((0.1, 0.12), 0.8, 0.42, fc="#3f6b57", ec="none"))  # 布
        if spos:                                                                       # 影
            ax.add_patch(plt.matplotlib.patches.Ellipse(spos, 0.30, 0.16, fc="#2a4a3b", ec="none"))
        ax.add_patch(plt.Circle((0.5, 0.33), 0.11, fc="#f0d0a8", ec="#c8a878", lw=1.5))  # アヒル
        ax.add_patch(plt.Circle((0.5, 0.44), 0.045, fc="#e88030", ec="none"))
        for lx, ly in lamps:                                                            # 光源と光線
            ax.add_patch(plt.Circle((lx, ly), 0.045, fc="#ffe680", ec="#d4a017", lw=1.5))
            ax.annotate("", xy=(0.5, 0.44), xytext=(lx, ly - 0.05),
                        arrowprops=dict(arrowstyle="->", color="#d4a017", lw=1.6, alpha=0.8))
        ax.set_title(f"{name}\n{src}", fontsize=12, pad=8)
        ax.text(0.5, 0.03, f"影: {shadow}", ha="center", fontsize=11, color="#333")
    fig.suptitle("照明 3 条件 — 明るさではなく「影の向き」を変えている", fontsize=13, y=0.99)
    fig.tight_layout(); fig.savefig(BLOG / "006.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------ 007 3条件の作例
def fig_conditions(obj=13):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    stats = json.loads((ROOT / "outputs" / "experiments.json").read_text()) if \
        (ROOT / "outputs" / "experiments.json").exists() else {}
    labels = {"normal": "通常（蛍光灯のみ）", "side": "側光（デスクライトのみ）",
              "bright": "明るい（両方）"}
    dark = {"normal": "左上が暗い", "side": "右上が暗い", "bright": "左右が同じ"}
    for ax, c in zip(axes, LIGHTS):
        p = DATA / "test_ok" / c / f"test_ok_{c}_{obj}.png"
        ax.imshow(Image.open(p)); ax.axis("off")
        ax.set_title(f"{labels[c]}\n{dark[c]}", fontsize=11)
    fig.suptitle(f"同一個体（#{obj}）を 3 条件で撮影 — 全体の明るさはほぼ同じ", fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    fig.savefig(BLOG / "007.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------ 008 before/after
def fig_before_after():
    ids = [17, 18, 19]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    for j, i in enumerate(ids):
        axes[0, j].imshow(Image.open(DATA / "test_ok" / "normal" / f"test_ok_normal_{i}.png"))
        axes[1, j].imshow(Image.open(DATA / "test_ng" / "hard" / f"ng_hard_{i}.png"))
        axes[0, j].set_title(f"#{i} 無傷", fontsize=11)
        axes[1, j].set_title(f"#{i} 傷あり（難）", fontsize=11)
        for k in (0, 1):
            axes[k, j].axis("off")
    fig.suptitle("同一個体の before / after — 差分は傷だけ（幅 1mm 以下の線）", fontsize=13)
    fig.tight_layout(); fig.savefig(BLOG / "008.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------ 009 運用ウィンドウ
def fig_window():
    rows = [
        ("前記事の方式\n(Titan MME)",        -3.18, "#c0504d"),
        ("Nova 2 MME",                       -5.03, "#c0504d"),
        ("Cohere Embed v4",                  -5.03, "#c0504d"),
        ("DINOv2 CLS\n(1画像=1ベクトル)",   12.18, "#4f81bd"),
        ("DINOv2 パッチ+近傍集約",            4.08, "#4f81bd"),
        ("DINOv2 + FT 手段A",                 1.00, "#9bbb59"),
        ("DINOv2 + FT 手段B",                 2.24, "#9bbb59"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.invert_yaxis(); ax.axvline(0, color="#333", lw=1.2)
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.2f}σ", va="center",
                ha="left" if v >= 0 else "right", fontsize=10.5, fontweight="bold")
    ax.set_xlim(-8.5, 15.5)
    ax.set_xlabel("運用ウィンドウ（過検出 0% かつ 検出 100% を満たす閾値の幅・σ）", fontsize=11)
    ax.set_title("同じ「条件を揃えたデータ」での比較\n"
                 "正なら閾値を引ける。FT すると余裕が減る", fontsize=12.5)
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.99, 0.012, "※ AUROC はこの 7 条件のうち 4 つで 1.000。AUROC では差が見えない",
             ha="right", fontsize=9.5, color="#666")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(BLOG / "009.png", dpi=140); plt.close(fig)


def fig_window_shift():
    """分布シフトの有無で比較（記事の主張の核心）。"""
    labels = ["1画像=1ベクトル", "パッチ+近傍集約"]
    before = [12.18, 4.08]      # 学習に 3 照明を含む
    after = [-5.25, -2.83]      # 学習が normal のみ
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.bar(x - w/2, before, w, label="学習に照明 3 条件を含む（27枚）", color="#4f81bd")
    ax.bar(x + w/2, after,  w, label="学習が normal のみ（9枚）",      color="#c0504d")
    for xi, v in zip(x - w/2, before):
        ax.text(xi, v + 0.4, f"{v:+.2f}σ", ha="center", fontsize=10.5, fontweight="bold")
    for xi, v in zip(x + w/2, after):
        ax.text(xi, v - 0.9, f"{v:+.2f}σ", ha="center", fontsize=10.5, fontweight="bold")
    ax.axhline(0, color="#333", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("運用ウィンドウ（σ）", fontsize=11)
    ax.set_title("最も効いたのは撮影条件だった\n"
                 "学習に無い照明が来ると、符号が反転して破綻する", fontsize=12.5)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(BLOG / "010.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------ 011 ヒートマップ
def fig_heatmap():
    from src.embed import Dinov2Encoder, aggregate_neighbors
    from src.memorybank import PatchMemoryBank
    enc = Dinov2Encoder(size=518)
    tr = sorted(str(p) for c in LIGHTS for p in (DATA / "train_ok" / c).glob("*.png"))
    ng = [str(DATA / "test_ng" / lv / f"ng_{lv}_{i}.png")
          for lv, i in [("easy", 10), ("medium", 13), ("hard", 17)]]
    ok = [str(DATA / "test_ok" / "normal" / "test_ok_normal_11.png")]
    _, pat_tr = enc.encode(tr)
    _, pat_q = enc.encode(ok + ng)
    ag = lambda p: aggregate_neighbors(p, enc.grid, 3)   # noqa: E731
    mb = PatchMemoryBank(coreset_ratio=1.0).fit(ag(pat_tr))
    _, pscore = mb.score(ag(pat_q))

    titles = ["正常（#11）", "易（#10）", "中（#13）", "難（#17）"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    vmax = float(pscore.max())
    for ax, p, t, s in zip(axes, ok + ng, titles, pscore):
        im = np.asarray(Image.open(p).convert("RGB").resize((518, 518)))
        hm = np.array(Image.fromarray((s.reshape(enc.grid, enc.grid) / vmax * 255)
                                      .astype(np.uint8)).resize((518, 518), Image.BICUBIC))
        ax.imshow(im); ax.imshow(hm, cmap="jet", alpha=0.45, vmin=0, vmax=255)
        ax.set_title(f"{t}\n最大スコア {s.max():.3f}", fontsize=11); ax.axis("off")
    fig.suptitle("パッチ単位の異常スコア — どこを異常と見たか（1 パッチ ≒ 実寸 1.3mm）",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(BLOG / "011.png", dpi=130); plt.close(fig)


# ------------------------------------------------------------ 012 FTが効く条件
def fig_ft_condition():
    """FT が効く条件と効かない条件（記事の新しい結論）。"""
    labels = ["条件を揃えた場合\n(27枚・3照明)", "分布シフト\n(学習9枚)",
              "分布シフト × 難易度hard\n(前記事の残課題)"]
    before = [11.63, -2.28, -2.09]
    after = [3.39, -0.30, 1.00]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    b1 = ax.bar(x - w/2, before, w, label="FT 前", color="#7f8c9b")
    b2 = ax.bar(x + w/2, after, w, label="FT 後",
                color=["#c0504d", "#c0504d", "#4f81bd"])
    for bars, vals in [(b1, before), (b2, after)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + (0.35 if v >= 0 else -0.9),
                    f"{v:+.2f}σ", ha="center", fontsize=10.5, fontweight="bold")
    ax.axhline(0, color="#333", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("運用ウィンドウ（σ）", fontsize=11)
    ax.set_title("ファインチューニングが効く条件・効かない条件\n"
                 "元の精度が高いほど FT の効果は小さく、むしろ余裕を減らす", fontsize=12.5)
    ax.legend(fontsize=10.5); ax.grid(axis="y", alpha=0.25)
    ax.annotate("負 → 正\n完全分離を達成", xy=(2 + w/2, 1.00), xytext=(2.05, 6.2),
                fontsize=10.5, color="#4f81bd", fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="#4f81bd", lw=1.6))
    fig.tight_layout(); fig.savefig(BLOG / "012.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------ 013 FT改善の内訳
def fig_ft_steps():
    """初版の失敗を切り分けて積み上げた改善（分布シフト条件）。"""
    steps = ["初版\n(CLSで分類・埋め込みで評価)", "A 分類ヘッドを\n直接スコアに",
             "D パッチ単位で\n分類", "アンサンブル\n(埋め込み+ヘッド)"]
    vals = [-2.43, -1.12, -0.95, -0.30]
    fig, ax = plt.subplots(figsize=(9.6, 5))
    bars = ax.bar(np.arange(len(steps)), vals, 0.6,
                  color=["#c0504d", "#d99694", "#9bbb59", "#4f81bd"])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v - 0.16, f"{v:+.2f}σ",
                ha="center", va="top", fontsize=11, fontweight="bold")
    ax.axhline(0, color="#333", lw=1.2)
    ax.set_xticks(np.arange(len(steps))); ax.set_xticklabels(steps, fontsize=10)
    ax.set_ylabel("運用ウィンドウ（σ）", fontsize=11)
    ax.set_ylim(-2.9, 0.35)
    ax.set_title("初版の失敗を 4 つに切り分けて改善した（分布シフト条件）\n"
                 "最大の原因は「評価が学習目標とズレていた」ことだった", fontsize=12.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(BLOG / "013.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    BLOG.mkdir(exist_ok=True)
    for name, fn in [("006 照明の模式図", fig_lighting), ("007 3条件の作例", fig_conditions),
                     ("008 before/after", fig_before_after), ("009 運用ウィンドウ", fig_window),
                     ("010 分布シフト", fig_window_shift), ("011 ヒートマップ", fig_heatmap),
                     ("012 FTが効く条件", fig_ft_condition), ("013 FT改善の内訳", fig_ft_steps)]:
        fn(); print(f"  {name} ... done", flush=True)
    print(f"\nsaved to {BLOG}")
