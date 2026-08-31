"""training-free の異常検知: 正常パッチのメモリバンク + 最近傍距離。

AnomalyDINO（WACV 2025, Apache-2.0）の手法を参照した最小実装。
学習は一切しない。正常画像のパッチ埋め込みを貯めておき、
検査画像の各パッチについて「最も似ている正常パッチとの距離」を異常スコアにする。

この記事の対比軸:
  - GlobalScorer  … 1 画像 = 1 ベクトル（前記事の方式）
  - PatchMemoryBank … パッチ単位（2026 年の定石）
"""
import numpy as np


def l2norm(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


class GlobalScorer:
    """1 画像 = 1 ベクトル。正常ベクトル群との最大コサイン類似度から異常スコアを出す。

    前記事は「基準画像 1 枚とのコサイン類似度が 0.9 以下なら異常」だった。
    ここでは基準を複数枚に一般化している（1 枚でも動く）。
    """

    def fit(self, cls_vecs):
        self.bank = l2norm(cls_vecs)
        return self

    def score(self, cls_vecs):
        sims = l2norm(cls_vecs) @ self.bank.T  # [M, N]
        return 1.0 - sims.max(axis=1)  # 距離が大きいほど異常


class PatchMemoryBank:
    """パッチ単位のメモリバンク。

    coreset_ratio: メモリバンクに残すパッチの割合。
                   PatchCore は greedy coreset を使うが、ここでは
                   ランダムサブサンプルで十分（記事の主眼ではないため最小実装）。
    """

    def __init__(self, coreset_ratio=0.1, seed=0):
        self.coreset_ratio = coreset_ratio
        self.seed = seed

    def fit(self, patches):
        """patches: [N, P, D]（画像数 × パッチ数 × 次元）"""
        f = l2norm(patches.reshape(-1, patches.shape[-1]))
        if self.coreset_ratio < 1.0:
            rng = np.random.default_rng(self.seed)
            k = max(1, int(len(f) * self.coreset_ratio))
            f = f[rng.choice(len(f), k, replace=False)]
        self.bank = f
        return self

    def score(self, patches, chunk=16):
        """returns (image_scores[M], patch_scores[M, P])

        画像スコアは「最も異常なパッチの値」。微細な欠陥は画像全体の平均に埋もれるため、
        平均ではなく最大を取るのが定石。
        """
        img_scores, pat_scores = [], []
        for i in range(0, len(patches), chunk):
            q = l2norm(patches[i : i + chunk])  # [B, P, D]
            b, p, d = q.shape
            sims = q.reshape(-1, d) @ self.bank.T  # [B*P, K]
            dist = (1.0 - sims.max(axis=1)).reshape(b, p)
            pat_scores.append(dist)
            img_scores.append(dist.max(axis=1))
        return np.concatenate(img_scores), np.concatenate(pat_scores)

    def fit_loo(self, patches):
        """leave-one-out 用。画像ごとの区切りを保持したままバンクを作る（coreset なし）。

        学習用の正常画像が 27 枚しかないため、閾値校正用に切り分ける余裕がない。
        「自分自身をバンクから除いてスコアリングする」ことで、
        27 枚すべてをバンクに使いながら、正常スコアの分布も得られる。
        """
        self.n_img, self.n_pat = patches.shape[0], patches.shape[1]
        self.bank = l2norm(patches.reshape(-1, patches.shape[-1]))
        self.loo = True
        return self

    def score_loo(self, chunk=4):
        """バンクに入れた画像自身を、自分を除いたバンクでスコアリングする。"""
        assert getattr(self, "loo", False), "fit_loo() で作ったバンクが必要"
        out = []
        for i in range(self.n_img):
            s, e = i * self.n_pat, (i + 1) * self.n_pat
            q = self.bank[s:e]                       # 自分のパッチ（正規化済み）
            sims = q @ self.bank.T                   # [P, N*P]
            sims[:, s:e] = -np.inf                   # 自分由来を除外
            out.append((1.0 - sims.max(axis=1)).max())
        return np.array(out)
