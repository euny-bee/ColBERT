#!/usr/bin/env python3
"""
recon_error.py
--------------
방법 1: Reconstruction Error 측정.

두 인덱스는 동일한 centroids와 codes를 공유하므로,
각 임베딩에 대해:
  vec_f16  = normalize(centroid[code] + residual_f16)   ← ground truth
  vec_2bit = normalize(centroid[code] + decode(residual_2bit))

L2 error = ||vec_f16 - vec_2bit||₂  (unit vector 간 거리, 최대 2.0)
Angular error = arccos(vec_f16 · vec_2bit)  (라디안)

실행:
  python recon_error.py
"""

import os, sys, json, torch
import numpy as np

COLBERT_DIR  = os.path.expanduser("~/ColBERT")
ANALOG_DIR   = os.path.join(COLBERT_DIR, "experiments/msmarco/indexes/200k.analog")
BIT2_DIR     = os.path.join(COLBERT_DIR, "experiments/msmarco/indexes/200k.2bit")
RESULTS_DIR  = os.path.join(COLBERT_DIR, "experiments/msmarco")
N_CHUNKS     = 8

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)


def load_pt(path):
    return torch.load(path, map_location="cpu")


def main():
    from colbert.indexing.codecs.residual import ResidualCodec
    from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings

    print("Loading 2-bit codec...")
    codec = ResidualCodec.load(BIT2_DIR)

    centroids = load_pt(os.path.join(ANALOG_DIR, "centroids.pt"))  # (32768, 128)

    all_l2   = []
    all_ang  = []

    for chunk in range(N_CHUNKS):
        codes_f16 = load_pt(os.path.join(ANALOG_DIR, f"{chunk}.codes.pt"))       # int32
        res_f16   = load_pt(os.path.join(ANALOG_DIR, f"{chunk}.residuals.pt"))   # float16
        codes_2bt = load_pt(os.path.join(BIT2_DIR,   f"{chunk}.codes.pt"))       # int32
        res_2bt   = load_pt(os.path.join(BIT2_DIR,   f"{chunk}.residuals.pt"))   # uint8

        # codes가 같은지 확인
        assert torch.all(codes_f16 == codes_2bt), f"Chunk {chunk}: codes differ!"

        n = codes_f16.size(0)

        # float16 복원: centroid + residual → normalize
        c_vecs  = centroids[codes_f16.long()].float()          # (n, 128)
        vec_f16 = torch.nn.functional.normalize(
                      c_vecs + res_f16.float(), p=2, dim=-1)   # (n, 128)

        # 2-bit 복원: codec.decompress 사용 (centroid + decoded_residual → normalize)
        compressed = ResidualEmbeddings(codes_2bt.to(torch.int32), res_2bt)
        vec_2bt    = codec.decompress(compressed).float().cpu()  # (n, 128)

        # L2 error
        l2 = (vec_f16 - vec_2bt).norm(dim=-1)                  # (n,)

        # Angular error (clip으로 수치 오차 방지)
        cos_sim = (vec_f16 * vec_2bt).sum(dim=-1).clamp(-1.0, 1.0)
        ang     = torch.acos(cos_sim)                          # radians

        all_l2.extend(l2.tolist())
        all_ang.extend(ang.tolist())

        print(f"  chunk {chunk:02d} | n={n:,} | "
              f"L2 mean={l2.mean():.4f}  ang mean={ang.mean():.4f} rad "
              f"({ang.mean().item() * 180 / 3.14159:.2f}°)")

    all_l2  = np.array(all_l2)
    all_ang = np.array(all_ang)
    all_ang_deg = np.degrees(all_ang)

    print(f"\n{'='*58}")
    print(f"  방법 1: Reconstruction Error  (총 {len(all_l2):,} 임베딩)")
    print(f"{'='*58}")
    print(f"  L2 error   mean   : {all_l2.mean():.4f}  (max 2.0 = 완전 반대)")
    print(f"  L2 error   std    : {all_l2.std():.4f}")
    print(f"  L2 error   median : {np.median(all_l2):.4f}")
    print(f"  L2 error   p95    : {np.percentile(all_l2, 95):.4f}")
    print(f"  L2 error   max    : {all_l2.max():.4f}")
    print(f"  Angular    mean   : {all_ang_deg.mean():.2f}°")
    print(f"  Angular    std    : {all_ang_deg.std():.2f}°")
    print(f"  Angular    p95    : {np.percentile(all_ang_deg, 95):.2f}°")
    print(f"{'='*58}")

    result = {
        "n_embeddings":    len(all_l2),
        "l2_mean":         float(all_l2.mean()),
        "l2_std":          float(all_l2.std()),
        "l2_median":       float(np.median(all_l2)),
        "l2_p95":          float(np.percentile(all_l2, 95)),
        "l2_max":          float(all_l2.max()),
        "angular_mean_deg": float(all_ang_deg.mean()),
        "angular_std_deg":  float(all_ang_deg.std()),
        "angular_p95_deg":  float(np.percentile(all_ang_deg, 95)),
    }

    out_path = os.path.join(RESULTS_DIR, "recon_error.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    # 플롯용 샘플 저장 (최대 100k)
    rng    = np.random.default_rng(42)
    n_save = min(100_000, len(all_l2))
    idx    = rng.choice(len(all_l2), size=n_save, replace=False)
    np.save(os.path.join(RESULTS_DIR, "recon_l2_sample.npy"),  all_l2[idx])
    np.save(os.path.join(RESULTS_DIR, "recon_ang_sample.npy"), all_ang_deg[idx])

    print(f"\nSaved → {out_path}")
    print(f"Saved sample arrays → recon_l2_sample.npy, recon_ang_sample.npy")


if __name__ == "__main__":
    main()
