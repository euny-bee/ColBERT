#!/usr/bin/env python3
"""
reindex_float16.py
------------------
subset 200K를 float16 residuals로 새로 인덱싱.

multiprocessing spawn 방식 때문에 런타임 패치가 child process에 전달되지 않음.
따라서 ColBERT 소스 파일을 직접 임시 수정 → 인덱싱 → 원본 복원 방식 사용.
"""

import os
import sys
import torch

COLBERT_DIR = os.path.expanduser("~/ColBERT")
SUBSET_COLLECTION = os.path.join(COLBERT_DIR, "data/msmarco/subset/collection.tsv")
INDEX_NAME = "subset.analog"

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)

RESIDUAL_PY     = os.path.join(COLBERT_DIR, "colbert/indexing/codecs/residual.py")
RESIDUAL_EMB_PY = os.path.join(COLBERT_DIR, "colbert/indexing/codecs/residual_embeddings.py")


# ============================================================
# 소스 파일 임시 패치 / 복원
# ============================================================

def patch_sources():
    """float16 인덱싱을 위해 ColBERT 소스 2개 임시 수정."""

    # --- residual_embeddings.py: uint8 assert 제거 ---
    with open(RESIDUAL_EMB_PY, "r") as f:
        emb_orig = f.read()

    emb_patched = emb_orig.replace(
        "        assert residuals.dtype == torch.uint8\n",
        "        # assert residuals.dtype == torch.uint8  # float16 mode\n",
    )
    assert emb_patched != emb_orig, "residual_embeddings.py 패치 실패 - 타겟 문자열 없음"

    with open(RESIDUAL_EMB_PY, "w") as f:
        f.write(emb_patched)

    # --- residual.py: binarize() + decompress() 교체 ---
    with open(RESIDUAL_PY, "r") as f:
        res_orig = f.read()

    # binarize() 전체 교체
    old_binarize = (
        "    def binarize(self, residuals):\n"
        "        residuals = torch.bucketize(residuals.float(), self.bucket_cutoffs).to(dtype=torch.uint8)\n"
        "        residuals = residuals.unsqueeze(-1).expand(*residuals.size(), self.nbits)  # add a new nbits-wide dim\n"
        "        residuals = residuals >> self.arange_bits  # divide by 2^bit for each bit position\n"
        "        residuals = residuals & 1  # apply mod 2 to binarize\n"
        "\n"
        "        assert self.dim % 8 == 0\n"
        "        assert self.dim % (self.nbits * 8) == 0, (self.dim, self.nbits)\n"
        "\n"
        "        if self.use_gpu:\n"
        "            residuals_packed = ResidualCodec.packbits(residuals.contiguous().flatten())\n"
        "        else:\n"
        "            residuals_packed = np.packbits(np.asarray(residuals.contiguous().flatten()))\n"
        "        residuals_packed = torch.as_tensor(residuals_packed, dtype=torch.uint8)\n"
        "        residuals_packed = residuals_packed.reshape(residuals.size(0), self.dim // 8 * self.nbits)\n"
        "\n"
        "        return residuals_packed\n"
    )
    new_binarize = (
        "    def binarize(self, residuals):\n"
        "        # float16 mode: 양자화 없이 그대로 저장\n"
        "        return residuals.half()  # (N, 128) float16\n"
    )
    assert old_binarize in res_orig, "residual.py binarize() 패치 실패 - 타겟 문자열 없음"
    res_patched = res_orig.replace(old_binarize, new_binarize)

    # decompress() 전체 교체 (avg_residual 계산에 사용)
    old_decompress = (
        "    #@profile\n"
        "    def decompress(self, compressed_embs: Embeddings):\n"
        "        \"\"\"\n"
        "            We batch below even if the target device is CUDA to avoid large temporary buffers causing OOM.\n"
        "        \"\"\"\n"
        "\n"
        "        codes, residuals = compressed_embs.codes, compressed_embs.residuals\n"
        "\n"
        "        D = []\n"
        "        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):\n"
        "            if self.use_gpu:\n"
        "                codes_, residuals_ = codes_.cuda(), residuals_.cuda()\n"
        "                centroids_ = ResidualCodec.decompress_residuals(\n"
        "                    residuals_,\n"
        "                    self.bucket_weights,\n"
        "                    self.reversed_bit_map,\n"
        "                    self.decompression_lookup_table,\n"
        "                    codes_,\n"
        "                    self.centroids,\n"
        "                    self.dim,\n"
        "                    self.nbits,\n"
        "                ).cuda()\n"
        "            else:\n"
        "                # TODO: Remove dead code\n"
        "                centroids_ = self.lookup_centroids(codes_, out_device='cpu')\n"
        "                residuals_ = self.reversed_bit_map[residuals_.long()]\n"
        "                residuals_ = self.decompression_lookup_table[residuals_.long()]\n"
        "                residuals_ = residuals_.reshape(residuals_.shape[0], -1)\n"
        "                residuals_ = self.bucket_weights[residuals_.long()]\n"
        "                centroids_.add_(residuals_)\n"
        "\n"
        "            if self.use_gpu:\n"
        "                D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()\n"
        "            else:\n"
        "                D_ = torch.nn.functional.normalize(centroids_.to(torch.float32), p=2, dim=-1)\n"
        "            D.append(D_)\n"
        "\n"
        "        return torch.cat(D)\n"
    )
    new_decompress = (
        "    #@profile\n"
        "    def decompress(self, compressed_embs: Embeddings):\n"
        "        \"\"\"\n"
        "            float16 mode: residuals를 직접 centroid에 더함.\n"
        "        \"\"\"\n"
        "\n"
        "        codes, residuals = compressed_embs.codes, compressed_embs.residuals\n"
        "\n"
        "        D = []\n"
        "        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):\n"
        "            if self.use_gpu:\n"
        "                codes_     = codes_.cuda()\n"
        "                residuals_ = residuals_.cuda().half()\n"
        "            else:\n"
        "                residuals_ = residuals_.float()\n"
        "            centroids_ = self.lookup_centroids(codes_, out_device=codes_.device)\n"
        "            centroids_ = centroids_ + residuals_\n"
        "            if self.use_gpu:\n"
        "                D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()\n"
        "            else:\n"
        "                D_ = torch.nn.functional.normalize(centroids_.float(), p=2, dim=-1)\n"
        "            D.append(D_)\n"
        "\n"
        "        return torch.cat(D)\n"
    )
    assert old_decompress in res_patched, "residual.py decompress() 패치 실패 - 타겟 문자열 없음"
    res_patched = res_patched.replace(old_decompress, new_decompress)

    with open(RESIDUAL_PY, "w") as f:
        f.write(res_patched)

    return emb_orig, res_orig


def restore_sources(emb_orig, res_orig):
    """원본 소스 복원."""
    with open(RESIDUAL_EMB_PY, "w") as f:
        f.write(emb_orig)
    with open(RESIDUAL_PY, "w") as f:
        f.write(res_orig)
    print("Sources restored.")


# ============================================================
# main
# ============================================================

def main():
    print("Patching ColBERT sources for float16 indexing...")
    emb_orig, res_orig = patch_sources()
    print("  residual_embeddings.py: uint8 assert removed")
    print("  residual.py: binarize() → float16, decompress() → float16 add")

    try:
        from colbert import Indexer
        from colbert.infra import Run, RunConfig, ColBERTConfig

        print(f"\nIndexing subset with float16 residuals → {INDEX_NAME}")
        with Run().context(RunConfig(nranks=1, experiment="msmarco")):
            config = ColBERTConfig(
                nbits=2,
                doc_maxlen=220,
                query_maxlen=32,
                bsize=16,
                index_bsize=32,
                avoid_fork_if_possible=True,
            )
            indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
            indexer.index(
                name=INDEX_NAME,
                collection=SUBSET_COLLECTION,
                overwrite=True,
            )
    finally:
        restore_sources(emb_orig, res_orig)

    print("\nVerifying float16 residuals...")
    index_path = os.path.join(COLBERT_DIR, f"experiments/msmarco/indexes/{INDEX_NAME}")
    residuals = torch.load(os.path.join(index_path, "0.residuals.pt"), map_location="cpu")
    print(f"  0.residuals.pt: dtype={residuals.dtype}, shape={residuals.shape}")
    assert residuals.dtype == torch.float16, f"Expected float16, got {residuals.dtype}"
    assert residuals.shape[1] == 128, f"Expected 128 dims, got {residuals.shape[1]}"
    print("  OK: float16 residuals verified.")


if __name__ == "__main__":
    main()
