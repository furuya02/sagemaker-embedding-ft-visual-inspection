"""Bedrock のマルチモーダル埋め込み 3 モデルをアヒルの実データで比較する。

前記事は Titan Multimodal Embeddings で「基準画像 1 枚とのコサイン類似度が 0.9 以下なら異常」
という構成だった。その再現と、2 年後に選べるようになった 2 モデルを同じ土俵に乗せる。

  Titan MME  amazon.titan-embed-image-v1              us-east-1（東京では使えない）
  Nova 2 MME amazon.nova-2-multimodal-embeddings-v1:0 us-east-1（東京では使えない）
  Cohere v4  cohere.embed-v4:0                        ap-northeast-1（東京で使える唯一）

コスト: 画像 1 枚 $0.0001。67 枚 × 3 モデル ≒ $0.02。
埋め込みは npz にキャッシュするので、再実行しても課金されない。

  python scripts/run_bedrock.py
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memorybank import GlobalScorer, l2norm                  # noqa: E402
from src.metrics import evaluate, to_sigma, operating_window, roc_points  # noqa: E402

DATA = ROOT / "data" / "duck"
CACHE = ROOT / "outputs" / "bedrock_emb.json"
LIGHTS = ["normal", "side", "bright"]
LEVELS = ["easy", "medium", "hard"]
DIM = 1024   # 3 モデルで共通に指定できる唯一の次元


def paths():
    tr = sorted(str(p) for c in LIGHTS for p in (DATA / "train_ok" / c).glob("*.png"))
    ok = sorted(str(p) for c in LIGHTS for p in (DATA / "test_ok" / c).glob("*.png"))
    ng = sorted(str(p) for l in LEVELS for p in (DATA / "test_ng" / l).glob("*.png"))
    return tr, ok, ng


def b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


# ----------------------------------------------------------------- 各モデル
def titan(img_b64, rt):
    body = {"inputImage": img_b64, "embeddingConfig": {"outputEmbeddingLength": DIM}}
    r = rt.invoke_model(modelId="amazon.titan-embed-image-v1", body=json.dumps(body))
    return json.loads(r["body"].read())["embedding"]


def nova(img_b64, rt):
    body = {"taskType": "SINGLE_EMBEDDING", "singleEmbeddingParams": {
        "embeddingPurpose": "GENERIC_INDEX", "embeddingDimension": DIM,
        "image": {"format": "png", "source": {"bytes": img_b64}}}}
    r = rt.invoke_model(modelId="amazon.nova-2-multimodal-embeddings-v1:0", body=json.dumps(body))
    # Titan と違ってレスポンスが 1 段深い（前記事のコードをそのまま流用すると落ちる）
    return json.loads(r["body"].read())["embeddings"][0]["embedding"]


def cohere(img_b64, rt):
    body = {"model": "embed-v4.0", "input_type": "image", "embedding_types": ["float"],
            "output_dimension": DIM, "images": [f"data:image/png;base64,{img_b64}"]}
    r = rt.invoke_model(modelId="cohere.embed-v4:0", body=json.dumps(body))
    e = json.loads(r["body"].read())["embeddings"]
    return e["float"][0] if isinstance(e, dict) else e[0]


MODELS = {
    "Titan MME":  (titan,  "us-east-1"),
    "Nova 2 MME": (nova,   "us-east-1"),
    "Cohere v4":  (cohere, "ap-northeast-1"),
}


def embed_all(all_paths):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for name, (fn, region) in MODELS.items():
        rt = boto3.client("bedrock-runtime", region_name=region)
        todo = [p for p in all_paths if f"{name}|{p}" not in cache]
        if not todo:
            print(f"  {name:<12} キャッシュ済み"); continue
        print(f"  {name:<12} {region}  {len(todo)} 枚を埋め込み中 ...", flush=True)
        t0 = time.time()
        for i, p in enumerate(todo):
            for attempt in range(4):
                try:
                    cache[f"{name}|{p}"] = fn(b64(p), rt); break
                except Exception as e:
                    if attempt == 3:
                        print(f"    ! {Path(p).name}: {type(e).__name__}: {str(e)[:80]}")
                    time.sleep(1.5 * (attempt + 1))
            if (i + 1) % 20 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"    {i+1}/{len(todo)}  ({time.time()-t0:.0f}s)", flush=True)
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(cache))
        print(f"    完了 {time.time()-t0:.0f}s   概算コスト ${len(todo)*0.0001:.3f}")
    return cache


def vecs(cache, name, ps):
    return np.array([cache[f"{name}|{p}"] for p in ps], dtype=np.float32)


def score_set(tr, ok, ng, ok_paths, ng_paths):
    """1 画像 = 1 ベクトルの評価。閾値校正は学習データの leave-one-out（NG は使わない）。"""
    g = GlobalScorer().fit(tr)
    sims = l2norm(tr) @ l2norm(tr).T
    np.fill_diagonal(sims, -np.inf)
    cal = 1.0 - sims.max(axis=1)
    s_ok, s_ng = g.score(ok), g.score(ng)
    m = evaluate(s_ok, s_ng)
    z_ok, z_ng = to_sigma(s_ok, cal), to_sigma(s_ng, cal)
    w = operating_window(z_ok, z_ng, max_fp=0.0, min_recall=1.0)
    by_light = {c: float(np.mean([z for p, z in zip(ok_paths, z_ok) if f"/{c}/" in p]))
                for c in LIGHTS}
    by_lv = {l: float(np.mean([z for p, z in zip(ng_paths, z_ng) if f"/{l}/" in p]))
             for l in LEVELS}
    return {"auroc": m["auroc"], "window": w["width"], "ok_max": float(z_ok.max()),
            "ng_min": float(z_ng.min()), "by_light": by_light, "by_level": by_lv,
            "fp0_recall": roc_points(z_ok, z_ng, (0.0,))[0]["recall"]}


def prev_article_style(tr, ok, ng, thr=0.9):
    """前記事の再現: 基準画像 1 枚とのコサイン類似度が thr 以下なら異常。"""
    ref = l2norm(tr[:1])
    sim_ok = (l2norm(ok) @ ref.T).ravel()
    sim_ng = (l2norm(ng) @ ref.T).ravel()
    return {"threshold": thr,
            "ok_sim_min": float(sim_ok.min()), "ok_sim_mean": float(sim_ok.mean()),
            "ng_sim_max": float(sim_ng.max()), "ng_sim_mean": float(sim_ng.mean()),
            "detected": int((sim_ng <= thr).sum()), "n_ng": len(sim_ng),
            "false_positive": int((sim_ok <= thr).sum()), "n_ok": len(sim_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embed", action="store_true")
    args = ap.parse_args()

    tr, ok, ng = paths()
    print(f"train_ok {len(tr)}  test_ok {len(ok)}  test_ng {len(ng)}\n")
    cache = json.loads(CACHE.read_text()) if args.skip_embed and CACHE.exists() else embed_all(tr + ok + ng)

    tr_normal = [p for p in tr if "/normal/" in p]
    results = {}

    print("\n" + "=" * 96)
    print("■ 前記事の再現（基準画像 1 枚 / コサイン類似度 0.9 を閾値）")
    print(f"{'モデル':<12}{'OK類似度の最小':>14}{'NG類似度の最大':>14}{'検出':>10}{'過検出':>10}")
    for name in MODELS:
        r = prev_article_style(vecs(cache, name, tr), vecs(cache, name, ok), vecs(cache, name, ng))
        results[f"{name}/prev_style"] = r
        print(f"{name:<12}{r['ok_sim_min']:>14.4f}{r['ng_sim_max']:>14.4f}"
              f"{r['detected']:>7}/{r['n_ng']:<3}{r['false_positive']:>7}/{r['n_ok']:<3}")

    for label, trs in [("条件を揃えた場合（学習 27 枚 = 3 照明）", tr),
                       ("分布シフト（学習 9 枚 = normal のみ）", tr_normal)]:
        print("\n" + "=" * 96)
        print(f"■ {label}")
        print(f"{'モデル':<12}{'AUROC':>8}{'運用幅':>9}{'OK最大':>8}{'NG最小':>8}"
              f"{'過検出0%の検出率':>16}   照明別の正常スコア(σ)")
        for name in MODELS:
            r = score_set(vecs(cache, name, trs), vecs(cache, name, ok), vecs(cache, name, ng), ok, ng)
            results[f"{name}/{label}"] = r
            bl = "  ".join(f"{c[:4]}{r['by_light'][c]:+5.1f}" for c in LIGHTS)
            print(f"{name:<12}{r['auroc']:>8.3f}{r['window']:>+9.2f}{r['ok_max']:>+8.2f}"
                  f"{r['ng_min']:>+8.2f}{r['fp0_recall']*100:>15.0f}%   {bl}")

    (ROOT / "outputs" / "bedrock_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved: outputs/bedrock_results.json")


if __name__ == "__main__":
    main()
