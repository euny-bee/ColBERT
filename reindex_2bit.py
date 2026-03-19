#!/usr/bin/env python3
"""
reindex_2bit.py
---------------
subset.quantized 인덱스를 nbits=2 (정상 2-bit 양자화)로 재인덱싱.
"""

import os
import sys
import torch

COLBERT_DIR = os.path.expanduser("~/ColBERT")
SUBSET_COLLECTION = os.path.join(COLBERT_DIR, "data/msmarco/subset/collection.tsv")
INDEX_NAME = "subset.quantized"

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)

from colbert import Indexer
from colbert.infra import Run, RunConfig, ColBERTConfig


def main():
    print("[1/2] Indexing subset with 2-bit quantized residuals...")

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

    print("[2/2] Verifying residuals dtype and shape...")
    index_path = os.path.join(COLBERT_DIR, f"experiments/msmarco/indexes/{INDEX_NAME}")
    residuals = torch.load(os.path.join(index_path, "0.residuals.pt"), map_location="cpu")
    print(f"  0.residuals.pt: dtype={residuals.dtype}, shape={residuals.shape}")

    assert residuals.dtype == torch.uint8, f"Expected uint8, got {residuals.dtype}"
    assert residuals.shape[1] == 32, f"Expected 32 bytes per embedding (2-bit x 128dim), got {residuals.shape[1]}"
    print("  OK: 2-bit quantization verified.")


if __name__ == "__main__":
    main()
