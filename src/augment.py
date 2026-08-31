"""学習用 OK 画像のデータ拡張。

撮影を 139 枚 → 67 枚に削減できたのは、ばらつきを
「実写でしか作れないもの（個体差・物理的な照明変化）」と
「拡張で作れるもの（位置・向き・軽微な明度差）」に切り分けたため。
ここで作るのは後者だけ。

**評価用画像には絶対に適用しない**（評価が実写でなくなると数字の意味がなくなる）。
"""
import numpy as np
from PIL import Image, ImageEnhance

# 左右反転は入れない。アヒルは左右非対称で、反転個体は検査対象として存在しないため、
# 「ありえない正常」を学習させてしまう。


def augment(im, rng, max_shift=0.05, max_deg=15.0, jitter=0.15):
    """1 枚の PIL 画像から拡張版を 1 枚作る。

    max_shift: 画像サイズに対する平行移動の最大割合（撮影時の置き位置のずれに相当）
    max_deg:   回転角の最大値（撮影時の向きのばらつきに相当。±15 度に留める）
    jitter:    明度・コントラスト・彩度の変動幅
    """
    w, h = im.size

    # 回転（背景色で埋める。背景紙が均一なのでこれで足りる）
    deg = rng.uniform(-max_deg, max_deg)
    bg = tuple(np.asarray(im).reshape(-1, 3).mean(axis=0).astype(int))
    im = im.rotate(deg, resample=Image.BICUBIC, fillcolor=bg)

    # 平行移動
    dx = int(rng.uniform(-max_shift, max_shift) * w)
    dy = int(rng.uniform(-max_shift, max_shift) * h)
    im = im.transform((w, h), Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                      resample=Image.BICUBIC, fillcolor=bg)

    # 明度・コントラスト・彩度のジッター
    # ※ これは「軽微な明度差」であって、照明条件の切り替え（影・ハイライトの変化）ではない。
    #    照明 3 条件は実写で確保している。
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        im = enhancer(im).enhance(1.0 + rng.uniform(-jitter, jitter))

    return im


def expand(paths, target_n, out_dir, seed=0):
    """実写 paths を拡張して target_n 枚に増やし、out_dir に書き出してパス一覧を返す。

    元画像はそのまま含める（実写 N 枚 + 拡張 (target_n - N) 枚）。
    """
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    out = []
    for i, p in enumerate(paths):  # 実写をそのままコピー
        dst = out_dir / f"real_{i:04d}.png"
        Image.open(p).convert("RGB").save(dst)
        out.append(str(dst))

    for i in range(max(0, target_n - len(paths))):
        src = paths[i % len(paths)]
        dst = out_dir / f"aug_{i:04d}.png"
        augment(Image.open(src).convert("RGB"), rng).save(dst)
        out.append(str(dst))

    return out
