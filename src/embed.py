"""DINOv2 の埋め込み抽出。

この記事の核心は「1 画像 = 1 ベクトル」と「パッチ単位」の比較なので、
同じモデルから CLS トークン（=画像全体を 1 本に潰したもの）と
パッチトークン（=局所特徴）の両方を取り出せるようにしている。
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

# ImageNet の正規化パラメータ（DINOv2 の学習時と同じ）
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def aggregate_neighbors(patches, grid, k=3):
    """パッチ特徴を k×k の近傍で平均する（PatchCore の局所近傍集約）。

    1 パッチ単体だと受容野が狭く、ノイズに振られやすい。
    近傍を混ぜると「その場所の周辺がどうなっているか」を含んだ特徴になる。
    入出力とも [N, grid*grid, D]。
    """
    import torch.nn.functional as F

    n, p, d = patches.shape
    x = torch.from_numpy(patches).view(n, grid, grid, d).permute(0, 3, 1, 2)
    x = F.avg_pool2d(x, k, stride=1, padding=k // 2)
    return x.permute(0, 2, 3, 1).reshape(n, p, d).numpy()


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Dinov2Encoder:
    """facebook/dinov2-small (ViT-S/14, Apache-2.0, 22.1M params)

    size=518 のとき 518/14 = 37 なので 37x37 = 1369 パッチ。
    実測: アヒル全長 36mm、ROI 900px 中に 660px で写るので 1 パッチ ≒ 1.3mm。
    """

    def __init__(self, model_id="facebook/dinov2-small", size=518, device=None):
        self.device = device or pick_device()
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.size = size
        self.patch_size = self.model.config.patch_size
        self.grid = size // self.patch_size
        self.dim = self.model.config.hidden_size

    def _load(self, path):
        im = Image.open(path).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x).permute(2, 0, 1)

    @torch.no_grad()
    def encode(self, paths, batch_size=8):
        """returns (cls[N, D], patches[N, grid*grid, D])"""
        cls_out, patch_out = [], []
        for i in range(0, len(paths), batch_size):
            xb = torch.stack([self._load(p) for p in paths[i : i + batch_size]]).to(self.device)
            h = self.model(pixel_values=xb).last_hidden_state  # [B, 1+N, D]
            cls_out.append(h[:, 0].float().cpu().numpy())
            # register token を持つ版に備え、末尾 grid*grid 個をパッチとして取る
            patch_out.append(h[:, -self.grid * self.grid :].float().cpu().numpy())
        return np.concatenate(cls_out), np.concatenate(patch_out)
