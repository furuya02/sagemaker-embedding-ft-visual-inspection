"""アヒル撮影用のキャプチャアプリ（macOS）。

Blog/撮影要領.md の 67 枚（学習用 OK 27 / 評価用 OK 30 / NG 10）を管理しながら、
ROI を固定したまま連続撮影する。

GUI は OpenCV で描画している。tkinter を使わないのは、pyenv の Python が
Tk 8.5 にリンクされており macOS 15 で "macOS 15 (1507) or later required" と
abort するため（実際に踏んだ）。

2 つのモード:
  camera : Web カメラから直接取得（推奨。C920 なら 1920x1080 の元解像度が使える）
  screen : 画面の指定領域をキャプチャ（QuickTime のプレビューを撮る）

使い方:
  .venv/bin/python scripts/capture_app.py --device 1
  .venv/bin/python scripts/capture_app.py --source screen
  .venv/bin/python scripts/capture_app.py --list-devices

操作:
  ドラッグ      ROI を指定
  Space/Enter  撮影
  a / d        個体番号を前後
  1 / 2 / 3    照明（通常 / 側光 / 明るい）
  4            未知光（評価用 #10〜#19 のみ。学習には使わない）
  o / n        OK（無傷）/ NG（傷あり）
  s            ROI 正方形の ON/OFF
  f            ピント未合焦でも撮る の ON/OFF
  q / ESC      終了
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "capture_config.json"
DATA = ROOT / "data" / "duck"

LIGHTS = [("normal", "通常"), ("side", "側光"), ("bright", "明るい")]
# 評価専用の照明。学習には一切使わないので「常に未知の照明」であり続ける。
# 実験で「学習に無い照明が来ると破綻する」ことが分かったため用意したが、
# 今回は時間の都合で撮影しなかった（side/bright で分布シフトの検証は可能）。
# 撮る場合は USE_EVAL_LIGHT を True にすると評価用 10 枚が計画に加わる。
EVAL_LIGHT = ("unseen", "未知光")
USE_EVAL_LIGHT = False
ALL_LIGHTS = LIGHTS + ([EVAL_LIGHT] if USE_EVAL_LIGHT else [])
NG_LEVEL = {**{i: "easy" for i in range(10, 13)},
            **{i: "medium" for i in range(13, 17)},
            **{i: "hard" for i in range(17, 20)}}

FONT_PATH = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
PREVIEW_W, PANEL_W = 960, 360
SHARP_OK = 0.10    # 実測: 合焦時 0.18〜0.27 / 暗所でボケ 0.006


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def plan():
    """撮るべき 67 枚の一覧 (rel_path, 説明)。"""
    items = []
    for i in range(1, 10):
        for c, _ in LIGHTS:
            items.append((f"train_ok/{c}/ok_{c}_{i:02d}.png", f"#{i:02d} 学習OK {c}"))
    for i in range(10, 20):
        for c, _ in ALL_LIGHTS:          # 評価用は未知光も含む
            items.append((f"test_ok/{c}/test_ok_{c}_{i:02d}.png", f"#{i:02d} 評価OK {c}"))
    for i in range(10, 20):
        lv = NG_LEVEL[i]
        items.append((f"test_ng/{lv}/ng_{lv}_{i:02d}.png", f"#{i:02d} NG {lv}"))
    return items


def target_path(kind, obj_id, light):
    if kind == "ng":
        lv = NG_LEVEL.get(obj_id)
        return f"test_ng/{lv}/ng_{lv}_{obj_id:02d}.png" if lv else None
    if obj_id <= 9:
        if light == EVAL_LIGHT[0]:
            return None                  # 未知光は評価専用（学習に混ぜない）
        return f"train_ok/{light}/ok_{light}_{obj_id:02d}.png"
    return f"test_ok/{light}/test_ok_{light}_{obj_id:02d}.png"


def sharpness(rgb):
    """コントラストで正規化したラプラシアン分散。

    素のラプラシアン分散は**暗いと値が下がる**ため、照明条件を変えると
    ピントが合っていても閾値を割ってしまう（実際に踏んだ）。
    輝度の標準偏差で割ることで、明るさ・コントラストに依存しない指標になる。

    実測値:
      通常照明・合焦   0.180 〜 0.274
      暗所・ブレてボケ  0.006
    """
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    sd = g.std()
    if sd < 1e-6:
        return 0.0
    return float(cv2.Laplacian(g / sd, cv2.CV_64F).var())


# ----------------------------------------------------------------- 取得元
class CameraSource:
    def __init__(self, device=0, width=1920, height=1080):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"カメラ {device} を開けません。--list-devices で番号を確認してください。")
        for _ in range(10):      # 露出とフォーカスが落ち着くまで捨てる
            self.cap.read()
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.live = True

    def frame(self):
        ok, f = self.cap.read()
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None

    def close(self):
        self.cap.release()


class ScreenSource:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp()) / "s.png"
        f = self.frame()
        if f is None:
            raise RuntimeError("画面をキャプチャできません。"
                               "システム設定 > プライバシーとセキュリティ > 画面収録 を許可してください。")
        self.size = (f.shape[1], f.shape[0])
        self.live = False

    def frame(self):
        subprocess.run(["screencapture", "-x", str(self.tmp)], check=False, capture_output=True)
        return np.asarray(Image.open(self.tmp).convert("RGB")) if self.tmp.exists() else None

    def close(self):
        pass


# ----------------------------------------------------------------- アプリ
class App:
    WIN = "duck capture"

    def __init__(self, src):
        self.src = src
        self.scale = src.size[0] / PREVIEW_W
        self.ph = int(src.size[1] / self.scale)
        self.roi = self.load_roi()
        self.drag = None
        self.kind, self.light, self.obj_id = "ok", "normal", 1
        self.square, self.ignore_focus = True, False
        self.msg, self.msg_color = "ドラッグで ROI を指定 / Space で撮影", (255, 255, 255)
        self.f_small, self.f_mid, self.f_big = font(15), font(17), font(20)

        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WIN, self.on_mouse)

    # ------------------------------------------------------------- ROI
    def load_roi(self):
        if CONF.exists():
            r = json.loads(CONF.read_text()).get("roi")
            if r:
                return tuple(r)
        w, h = self.src.size
        s = int(min(w, h) * 0.9)
        return ((w - s) // 2, (h - s) // 2, s, s)

    def on_mouse(self, event, x, y, flags, _):
        if x >= PREVIEW_W:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            self.drag = (self.drag[0], self.drag[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            x0, y0, x1, y1 = self.drag
            self.drag = None
            px, py = min(x0, x1), min(y0, y1)
            w, h = abs(x1 - x0), abs(y1 - y0)
            if w < 10 or h < 10:
                return
            if self.square:
                w = h = min(w, h)
            self.roi = (int(px * self.scale), int(py * self.scale),
                        int(w * self.scale), int(h * self.scale))
            CONF.write_text(json.dumps({"roi": list(self.roi)}, indent=2))
            self.msg, self.msg_color = f"ROI を更新 {self.roi[2]}x{self.roi[3]}", (120, 255, 120)

    def crop(self, f):
        x, y, w, h = self.roi
        return f[y:y + h, x:x + w]

    def jump_to_unshot(self):
        """いまの種別・照明で、まだ撮っていない最小の個体番号へ移動する。"""
        lo = 10 if self.kind == "ng" else 1
        for i in range(lo, 20):
            rel = target_path(self.kind, i, self.light)
            if rel and not (DATA / rel).exists():
                self.obj_id = i
                return
        self.obj_id = lo

    # ------------------------------------------------------------- 動作
    def shoot(self, frame):
        rel = target_path(self.kind, self.obj_id, self.light)
        if rel is None:
            self.msg = ("未知光は評価用(#10〜#19)のみです" if self.light == EVAL_LIGHT[0]
                        else "NG は #10〜#19 のみです")
            self.msg_color = (255, 90, 90)
            return
        img = self.crop(frame)
        if img.size == 0:
            self.msg, self.msg_color = "ROI が画面外です", (255, 90, 90)
            return
        s = sharpness(img)
        if s < SHARP_OK and not self.ignore_focus:
            self.msg = f"ピント未合焦 (鮮鋭度 {s:.3f} < {SHARP_OK:.2f})。数秒待って再度"
            self.msg_color = (255, 90, 90)
            return
        out = DATA / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out)
        self.msg = f"保存 {rel}  {img.shape[1]}x{img.shape[0]}"
        self.msg_color = (120, 255, 120)
        # OK は #01〜#19 を通しで撮る（#01-09 が学習用、#10-19 が評価用に自動で振り分く）。
        # NG は #10〜#19 のみ。
        if self.obj_id < 19:
            self.obj_id += 1
        else:
            self.msg += "  ← この条件は最後です"

    # ------------------------------------------------------------- 描画
    def render(self, frame):
        prev = cv2.resize(frame, (PREVIEW_W, self.ph))
        canvas = Image.new("RGB", (PREVIEW_W + PANEL_W, max(self.ph, 620)), (28, 28, 30))
        canvas.paste(Image.fromarray(prev), (0, 0))
        d = ImageDraw.Draw(canvas)

        x, y, w, h = [v / self.scale for v in self.roi]
        d.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=3)
        if self.drag:
            d.rectangle([min(self.drag[0], self.drag[2]), min(self.drag[1], self.drag[3]),
                         max(self.drag[0], self.drag[2]), max(self.drag[1], self.drag[3])],
                        outline=(60, 170, 255), width=2)

        # ピント状態（ROI 内で判定）
        s = sharpness(self.crop(frame))
        ok = s >= SHARP_OK
        col = (60, 220, 60) if ok else (255, 70, 70)
        d.rectangle([10, 10, 10 + min(260, s * 900), 30], fill=col)
        d.rectangle([10, 10, 270, 30], outline=(255, 255, 255))
        d.text((280, 9), f"ピント {s:>5.3f} / {SHARP_OK:.2f}  "
                         f"{'OK' if ok else '合焦待ち…'}", font=self.f_mid, fill=col)
        d.text((10, self.ph - 28), self.msg, font=self.f_mid, fill=self.msg_color)

        # 右パネル
        items = plan()
        done = {r for r, _ in items if (DATA / r).exists()}
        rel = target_path(self.kind, self.obj_id, self.light)
        px = PREVIEW_W + 16
        d.text((px, 14), f"進捗  {len(done)} / {len(items)} 枚", font=self.f_big, fill=(255, 255, 255))
        ty = 48
        for label, pref in [("学習OK", "train_ok/"), ("評価OK", "test_ok/"), ("NG", "test_ng/")]:
            tot = sum(1 for r, _ in items if r.startswith(pref))
            n = sum(1 for r in done if r.startswith(pref))
            d.text((px, ty), f"  {label:<6} {n:>2} / {tot}", font=self.f_mid, fill=(200, 200, 200))
            ty += 24

        ty += 14
        for line, val in [("種別  [o/n]", "OK（無傷）" if self.kind == "ok" else "NG（傷あり）"),
                          ("照明  [1/2/3/4]", dict(ALL_LIGHTS)[self.light]),
                          ("個体  [a/d]", f"#{self.obj_id:02d}")]:
            d.text((px, ty), line, font=self.f_small, fill=(150, 150, 150))
            d.text((px + 130, ty - 2), val, font=self.f_mid, fill=(255, 255, 255))
            ty += 28

        ty += 10
        d.text((px, ty), f"ROI  {self.roi[2]}x{self.roi[3]}"
                         f"  {'正方形' if self.square else '自由'}  [s]",
               font=self.f_small, fill=(150, 150, 150)); ty += 30

        d.text((px, ty), "次に撮るもの", font=self.f_small, fill=(150, 150, 150)); ty += 22
        d.text((px, ty), f"{rel or '—'}", font=self.f_small,
               fill=(255, 210, 90) if rel not in done else (120, 200, 120)); ty += 22
        d.text((px, ty), "（撮影済み・上書きになります）" if rel in done else "（未撮影）",
               font=self.f_small, fill=(150, 150, 150)); ty += 32

        d.text((px, ty), "未撮影の先頭", font=self.f_small, fill=(150, 150, 150)); ty += 22
        remain = [ds for r, ds in items if r not in done]
        for ds in (remain[:6] or ["（すべて撮影済み）"]):
            d.text((px, ty), f"  {ds}", font=self.f_small, fill=(190, 190, 190)); ty += 21

        ty += 12
        d.text((px, ty), "f: 未合焦でも撮る = " + ("ON" if self.ignore_focus else "OFF"),
               font=self.f_small, fill=(150, 150, 150)); ty += 21
        d.text((px, ty), "q / ESC: 終了", font=self.f_small, fill=(150, 150, 150))

        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    # ------------------------------------------------------------- ループ
    def run(self):
        frame = None
        while True:
            f = self.src.frame()
            if f is not None:
                frame = f
            if frame is None:
                continue
            cv2.imshow(self.WIN, self.render(frame))
            k = cv2.waitKey(30 if self.src.live else 300) & 0xFF
            if k == 255:
                if cv2.getWindowProperty(self.WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            if k in (ord('q'), 27):
                break
            elif k in (32, 13):
                self.shoot(frame)
            elif k == ord('a'):
                self.obj_id = max(1, self.obj_id - 1)
            elif k == ord('d'):
                self.obj_id = min(19, self.obj_id + 1)
            elif k in (ord('1'), ord('2'), ord('3'), ord('4')):
                self.light = ALL_LIGHTS[k - ord('1')][0]
                self.jump_to_unshot()   # 照明を変えたらその条件の未撮影の先頭へ
            elif k == ord('o'):
                self.kind = "ok"; self.jump_to_unshot()
            elif k == ord('n'):
                self.kind = "ng"; self.jump_to_unshot()
            elif k == ord('s'):
                self.square = not self.square
            elif k == ord('f'):
                self.ignore_focus = not self.ignore_focus
        cv2.destroyAllWindows()


def list_devices():
    print("カメラ番号を探索中...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"  device {i}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
                  f"x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["camera", "screen"], default="camera")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        list_devices(); return

    src = CameraSource(args.device) if args.source == "camera" else ScreenSource()
    print(f"source={args.source}  size={src.size[0]}x{src.size[1]}")
    app = App(src)
    try:
        app.run()
    finally:
        src.close()


if __name__ == "__main__":
    main()
