#!/usr/bin/env python3
"""
ColBERTv2 Overnight Experiment: Quantized (2-bit) vs Analog (float16) Residuals
================================================================================

This script runs the full comparison pipeline:

  Step 3:  Search the full quantized index (8.8M passages) + evaluate
           → Compare with ColBERTv2 paper Table 4 numbers
  Step 4:  Create a 500k subset collection (memory-efficient comparison)
  Step 5A: Index subset with quantized (2-bit) + search + evaluate
  Step 5B: Patch code for analog (float16) + index subset + search + evaluate
  Step 6:  Print comparison table

Why subset?
  Full analog index needs ~130 GB RAM for search (float16 residuals are 8x
  larger than 2-bit packed). The 500k subset keeps both experiments in 16 GB.
  All relevant passages from qrels are included for fair evaluation.

Hardware requirements:
  - Step 3 (full search):  ~20 GB RAM (may use swap, might OOM on 16 GB)
  - Steps 5A/5B (subset): ~10 GB RAM max
  - Disk: ~15 GB free for subset indexes + rankings

Usage (run from WSL):
  cd ~/ColBERT
  source ~/colbert-env/bin/activate
  nohup python run_overnight.py > overnight.log 2>&1 &

  # Monitor progress:
  tail -f overnight.log
"""

import os
import sys
import time
import json
import shutil
import random
import subprocess
import traceback

# ============================
# Configuration
# ============================
COLBERT_DIR = os.path.expanduser("~/ColBERT")
DATA_DIR = os.path.join(COLBERT_DIR, "data/msmarco")
RESULTS_DIR = os.path.join(COLBERT_DIR, "experiments/msmarco")

QRELS_PATH = os.path.join(DATA_DIR, "qrels.dev.small.tsv")
QUERIES_PATH = os.path.join(DATA_DIR, "queries.dev.small.tsv")
COLLECTION_PATH = os.path.join(DATA_DIR, "collection.tsv")

SUBSET_SIZE = 500_000
SUBSET_DIR = os.path.join(DATA_DIR, "subset")
SUBSET_COLLECTION = os.path.join(SUBSET_DIR, "collection.tsv")
SUBSET_QRELS = os.path.join(SUBSET_DIR, "qrels.tsv")

# 패치 대상은 실제 import되는 경로 (/mnt/c/... Windows 마운트)
_CODE_DIR = "/mnt/c/Users/dmsdu/ColBERT"
RESIDUAL_PY = os.path.join(_CODE_DIR, "colbert/indexing/codecs/residual.py")
RESIDUAL_EMB_PY = os.path.join(_CODE_DIR, "colbert/indexing/codecs/residual_embeddings.py")

RESULTS_FILE = os.path.join(RESULTS_DIR, "overnight_results.json")


# ============================
# Utilities
# ============================
def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def save_result(key, metrics):
    """Append metrics to the results JSON file."""
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    results[key] = metrics
    results[key]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"Results saved: {key} = {metrics}")


def run_eval(ranking_path, qrels_path):
    """Run MS MARCO evaluation and parse MRR@10, R@50, R@1000."""
    log(f"Evaluating: {ranking_path}")
    result = subprocess.run(
        [sys.executable, "-m", "utility.evaluate.msmarco_passages",
         "--qrels", qrels_path, "--ranking", ranking_path],
        capture_output=True, text=True, cwd=COLBERT_DIR
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(result.stderr, flush=True)

    metrics = {}
    for line in result.stdout.split('\n'):
        line = line.strip()
        if 'only for ranked' in line:
            continue
        if 'MRR@10 =' in line:
            metrics['MRR@10'] = float(line.split('=')[-1].strip())
        elif 'Recall@50 =' in line:
            metrics['R@50'] = float(line.split('=')[-1].strip())
        elif 'Recall@1000 =' in line:
            metrics['R@1k'] = float(line.split('=')[-1].strip())
    return metrics


# ============================
# Step 4: Create subset
# ============================
def create_subset(subset_size=None):
    """
    Create a passage subset that includes ALL relevant passages from qrels.
    PIDs are renumbered 0..N-1 and qrels are adjusted accordingly.
    """
    subset_size = subset_size or SUBSET_SIZE
    log(f"Creating subset of {subset_size:,} passages...")
    os.makedirs(SUBSET_DIR, exist_ok=True)

    # 1) Collect all relevant PIDs from qrels
    relevant_pids = set()
    with open(QRELS_PATH) as f:
        for line in f:
            parts = line.strip().split()
            relevant_pids.add(int(parts[2]))
    log(f"  Relevant passage IDs in qrels: {len(relevant_pids):,}")

    # 2) Count total passages
    total_passages = 0
    with open(COLLECTION_PATH) as f:
        for _ in f:
            total_passages += 1
    log(f"  Total passages in collection: {total_passages:,}")

    # 3) Select additional random PIDs
    num_random = max(0, subset_size - len(relevant_pids))
    non_relevant = list(set(range(total_passages)) - relevant_pids)
    random.seed(42)
    random_pids = set(random.sample(non_relevant, min(num_random, len(non_relevant))))
    subset_pids = relevant_pids | random_pids
    log(f"  Subset: {len(relevant_pids):,} relevant + {len(random_pids):,} random = {len(subset_pids):,}")

    # 4) Build PID mapping: original_pid -> new_pid (sorted for determinism)
    sorted_pids = sorted(subset_pids)
    pid_map = {old: new for new, old in enumerate(sorted_pids)}

    # 5) Write subset collection.tsv
    log("  Writing subset collection...")
    written = 0
    with open(COLLECTION_PATH) as fin, open(SUBSET_COLLECTION, 'w') as fout:
        for line_idx, line in enumerate(fin):
            if line_idx in subset_pids:
                parts = line.strip('\n\r ').split('\t', 1)
                passage_text = parts[1] if len(parts) > 1 else parts[0]
                fout.write(f"{pid_map[line_idx]}\t{passage_text}\n")
                written += 1
    log(f"  Wrote {written:,} passages to {SUBSET_COLLECTION}")

    # 6) Write adjusted qrels
    log("  Writing adjusted qrels...")
    with open(QRELS_PATH) as fin, open(SUBSET_QRELS, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            qid, zero, pid, label = parts[0], parts[1], int(parts[2]), parts[3]
            new_pid = pid_map[pid]
            fout.write(f"{qid}\t{zero}\t{new_pid}\t{label}\n")
    log(f"  Wrote adjusted qrels to {SUBSET_QRELS}")

    log("Subset creation complete!")


# ============================
# File patching for analog mode
# ============================
def backup_files():
    log("Backing up source files...")
    for f in [RESIDUAL_PY, RESIDUAL_EMB_PY]:
        shutil.copy2(f, f + ".bak")
    log("  Backups created (.bak files)")


def restore_files():
    log("Restoring original source files...")
    for f in [RESIDUAL_PY, RESIDUAL_EMB_PY]:
        bak = f + ".bak"
        if os.path.exists(bak):
            shutil.copy2(bak, f)
            os.remove(bak)
    log("  Original files restored")


def apply_analog_patches():
    """
    Patch residual.py and residual_embeddings.py for analog (float16) residuals.

    Changes:
      1. compress(): store residuals as float16 instead of calling binarize()
      2. decompress(): add float16 residuals to centroids (no bit unpacking)
      3. ResidualEmbeddings.__init__(): accept float16 dtype
      4. ResidualEmbeddings.load_chunks(): allocate float16 arrays with full dim
    """
    log("Applying analog patches...")

    # --- Patch residual.py ---
    with open(RESIDUAL_PY, 'r') as f:
        content = f.read()

    # Patch 1: compress() — skip binarize, store float16
    old_compress = "            residuals.append(self.binarize(residuals_).cpu())"
    new_compress = "            residuals.append(residuals_.half().cpu())  # ANALOG: float16"
    assert old_compress in content, "Cannot find compress() patch target in residual.py"
    content = content.replace(old_compress, new_compress)

    # Patch 2: decompress() — simple centroid + residual addition
    old_decompress = """    #@profile
    def decompress(self, compressed_embs: Embeddings):
        \"\"\"
            We batch below even if the target device is CUDA to avoid large temporary buffers causing OOM.
        \"\"\"

        codes, residuals = compressed_embs.codes, compressed_embs.residuals

        D = []
        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            if self.use_gpu:
                codes_, residuals_ = codes_.cuda(), residuals_.cuda()
                centroids_ = ResidualCodec.decompress_residuals(
                    residuals_,
                    self.bucket_weights,
                    self.reversed_bit_map,
                    self.decompression_lookup_table,
                    codes_,
                    self.centroids,
                    self.dim,
                    self.nbits,
                ).cuda()
            else:
                # TODO: Remove dead code
                centroids_ = self.lookup_centroids(codes_, out_device='cpu')
                residuals_ = self.reversed_bit_map[residuals_.long()]
                residuals_ = self.decompression_lookup_table[residuals_.long()]
                residuals_ = residuals_.reshape(residuals_.shape[0], -1)
                residuals_ = self.bucket_weights[residuals_.long()]
                centroids_.add_(residuals_)

            if self.use_gpu:
                D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()
            else:
                D_ = torch.nn.functional.normalize(centroids_.to(torch.float32), p=2, dim=-1)
            D.append(D_)

        return torch.cat(D)"""

    new_decompress = """    #@profile
    def decompress(self, compressed_embs: Embeddings):
        \"\"\"
            ANALOG: Add float16 residuals directly to centroids (no bit unpacking).
        \"\"\"

        codes, residuals = compressed_embs.codes, compressed_embs.residuals

        D = []
        for codes_, residuals_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            if self.use_gpu:
                codes_ = codes_.cuda()
                residuals_ = residuals_.cuda().half()
            else:
                residuals_ = residuals_.float()
            centroids_ = self.lookup_centroids(codes_, out_device=codes_.device)
            centroids_ = centroids_ + residuals_

            if self.use_gpu:
                D_ = torch.nn.functional.normalize(centroids_, p=2, dim=-1).half()
            else:
                D_ = torch.nn.functional.normalize(centroids_.to(torch.float32), p=2, dim=-1)
            D.append(D_)

        return torch.cat(D)"""

    assert old_decompress in content, "Cannot find decompress() patch target in residual.py"
    content = content.replace(old_decompress, new_decompress)

    with open(RESIDUAL_PY, 'w') as f:
        f.write(content)
    log("  Patched residual.py (compress + decompress)")

    # --- Patch residual_embeddings.py ---
    with open(RESIDUAL_EMB_PY, 'r') as f:
        content = f.read()

    # Patch 3: Remove dtype assertion
    old_assert = "        assert residuals.dtype == torch.uint8"
    new_assert = "        # ANALOG: accept float16 residuals too\n        # assert residuals.dtype == torch.uint8"
    assert old_assert in content, "Cannot find dtype assert in residual_embeddings.py"
    content = content.replace(old_assert, new_assert)

    # Patch 4: load_chunks allocation — full dim with float16
    old_alloc = "            residuals = torch.empty(num_embeddings, dim // 8 * nbits, dtype=torch.uint8)"
    new_alloc = "            residuals = torch.empty(num_embeddings, dim, dtype=torch.float16)  # ANALOG: full-dim float16"
    assert old_alloc in content, "Cannot find load_chunks allocation in residual_embeddings.py"
    content = content.replace(old_alloc, new_alloc)

    with open(RESIDUAL_EMB_PY, 'w') as f:
        f.write(content)
    log("  Patched residual_embeddings.py (init + load_chunks)")

    log("Analog patches applied successfully!")


# ============================
# Step functions (run as subprocesses for clean module imports)
# ============================
def _step3_search_full_quantized():
    """Search the full 8.8M quantized index and evaluate."""
    os.chdir(COLBERT_DIR)
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    ranking_dir = os.path.join(RESULTS_DIR, "rankings")
    os.makedirs(ranking_dir, exist_ok=True)

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32)

        log("Loading full quantized index (msmarco.nbits2)...")
        searcher = Searcher(index="msmarco.nbits2", config=config)

        log("Loading queries...")
        queries = Queries(QUERIES_PATH)

        log(f"Searching {len(queries)} queries with k=1000...")
        start = time.time()
        ranking = searcher.search_all(queries, k=1000)
        elapsed = time.time() - start
        log(f"Search done in {elapsed:.0f}s ({elapsed/len(queries)*1000:.0f} ms/query)")

        ranking_path = os.path.join(ranking_dir, "full_quantized.ranking.tsv")
        ranking.save(ranking_path)

    metrics = run_eval(ranking_path, QRELS_PATH)
    save_result("full_quantized", metrics)


def _step5a_subset_quantized():
    """Index and search the subset with quantized (2-bit) residuals."""
    os.chdir(COLBERT_DIR)
    from colbert import Indexer, Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    ranking_dir = os.path.join(RESULTS_DIR, "rankings")
    os.makedirs(ranking_dir, exist_ok=True)

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            nbits=2, doc_maxlen=220, query_maxlen=32,
            bsize=16, index_bsize=32,
            avoid_fork_if_possible=True,
        )

        log("Indexing subset with quantized (2-bit) residuals...")
        start = time.time()
        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
        indexer.index(
            name="subset.quantized",
            collection=SUBSET_COLLECTION,
            overwrite=True,
        )
        log(f"Subset quantized indexing done in {time.time()-start:.0f}s")

        log("Searching subset quantized index...")
        searcher = Searcher(index="subset.quantized", config=config)
        queries = Queries(QUERIES_PATH)

        start = time.time()
        ranking = searcher.search_all(queries, k=1000)
        log(f"Search done in {time.time()-start:.0f}s")

        ranking_path = os.path.join(ranking_dir, "subset_quantized.ranking.tsv")
        for ext in ['', '.meta']:
            p = ranking_path + ext
            if os.path.exists(p):
                os.remove(p)
        ranking.save(ranking_path)

    metrics = run_eval(ranking_path, SUBSET_QRELS)
    save_result("subset_quantized", metrics)


def _step5b_subset_analog():
    """Index the subset with analog (float16) residuals (indexing only)."""
    os.chdir(COLBERT_DIR)
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            nbits=2, doc_maxlen=220, query_maxlen=32,
            bsize=16, index_bsize=32,
            avoid_fork_if_possible=True,
        )

        log("Indexing subset with analog (float16) residuals...")
        start = time.time()
        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
        indexer.index(
            name="subset.analog",
            collection=SUBSET_COLLECTION,
            overwrite=True,
        )
        log(f"Subset analog indexing done in {time.time()-start:.0f}s")


def _step5b_search_analog(k=1000):
    """Search the analog index with given k."""
    os.chdir(COLBERT_DIR)
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    ranking_dir = os.path.join(RESULTS_DIR, "rankings")
    os.makedirs(ranking_dir, exist_ok=True)

    with Run().context(RunConfig(nranks=1, experiment="msmarco")):
        config = ColBERTConfig(
            nbits=2, doc_maxlen=220, query_maxlen=32,
        )

        log(f"Searching subset analog index with k={k}...")
        searcher = Searcher(index="subset.analog", config=config)
        queries = Queries(QUERIES_PATH)

        start = time.time()
        ranking = searcher.search_all(queries, k=k)
        log(f"Search done in {time.time()-start:.0f}s")

        ranking_path = os.path.join(ranking_dir, "subset_analog.ranking.tsv")
        for ext in ['', '.meta']:
            p = ranking_path + ext
            if os.path.exists(p):
                os.remove(p)
        ranking.save(ranking_path)

    metrics = run_eval(ranking_path, SUBSET_QRELS)
    save_result("subset_analog", metrics)


def _step6_compare():
    """Print final comparison table."""
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    print("\n" + "=" * 70)
    print("  ColBERTv2: Quantized (2-bit) vs Analog (float16) Residuals")
    print("  MS MARCO Passage Ranking — Official Dev (6,980 queries)")
    print("=" * 70)

    # Full collection results
    if "full_quantized" in results:
        m = results["full_quantized"]
        print(f"\n  [Full Collection, 8.8M passages] Quantized (2-bit):")
        print(f"    MRR@10 = {m.get('MRR@10', 'N/A')}")
        print(f"    R@50   = {m.get('R@50', 'N/A')}")
        print(f"    R@1k   = {m.get('R@1k', 'N/A')}")
        print(f"    Paper:   MRR@10=0.397, R@50=0.868, R@1k=0.984")
    else:
        print("\n  [Full Collection] Not available (Step 3 may have been skipped or failed)")

    # Subset comparison
    if "subset_quantized" in results and "subset_analog" in results:
        sq = results["subset_quantized"]
        sa = results["subset_analog"]

        print(f"\n  [Subset, {SUBSET_SIZE:,} passages] Quantized vs Analog:")
        print(f"  {'Metric':<10} {'Quantized (2-bit)':<20} {'Analog (float16)':<20} {'Delta':<10}")
        print(f"  {'-'*58}")

        for metric in ['MRR@10', 'R@50', 'R@1k']:
            vq = sq.get(metric)
            va = sa.get(metric)
            if isinstance(vq, (int, float)) and isinstance(va, (int, float)):
                delta = va - vq
                sign = "+" if delta > 0 else ""
                print(f"  {metric:<10} {vq:<20.4f} {va:<20.4f} {sign}{delta:.4f}")
            else:
                print(f"  {metric:<10} {str(vq):<20} {str(va):<20} {'N/A':<10}")

        print(f"\n  Note: Subset metrics differ from full-collection metrics.")
        print(f"  The Delta column shows the quality gain from not quantizing residuals.")
    else:
        missing = []
        if "subset_quantized" not in results:
            missing.append("subset_quantized")
        if "subset_analog" not in results:
            missing.append("subset_analog")
        print(f"\n  [Subset] Missing results: {', '.join(missing)}")

    # Storage efficiency comparison
    print(f"\n  [Storage Efficiency — Analog Device Perspective]")
    print(f"  {'':30} {'Quantized (2-bit)':<20} {'Analog':<20}")
    print(f"  {'-'*68}")

    dim = 128
    nbits = 2
    q_devices_per_vec = dim * nbits   # 256 bit-cells
    a_devices_per_vec = dim           # 128 analog devices

    print(f"  {'Devices per vector':<30} {q_devices_per_vec:<20} {a_devices_per_vec:<20}")
    print(f"  {'Precision per device':<30} {'2 levels (1 bit)':<20} {'Continuous':<20}")
    print(f"  {'Ratio':<30} {'1x':<20} {f'{a_devices_per_vec/q_devices_per_vec:.0%} ({a_devices_per_vec}/{q_devices_per_vec})':<20}")

    # Estimate total devices from index metadata
    for key, label in [("subset_quantized", "Subset"), ("full_quantized", "Full (8.8M)")]:
        index_name = "subset.quantized" if key == "subset_quantized" else "msmarco.nbits2"
        metadata_path = os.path.join(RESULTS_DIR, "indexes", index_name, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            num_embs = meta.get("num_embeddings", 0)
            q_total = num_embs * q_devices_per_vec
            a_total = num_embs * a_devices_per_vec
            print(f"\n  [{label}] Total embeddings: {num_embs:,}")
            print(f"  {'Total devices (quantized)':<30} {q_total:,.0f}")
            print(f"  {'Total devices (analog)':<30} {a_total:,.0f} ({a_devices_per_vec/q_devices_per_vec:.0%})")

    print(f"\n  Analog uses 50% fewer devices with higher precision per device.")
    print("\n" + "=" * 70 + "\n")


# ============================
# Main orchestrator
# ============================
def main():
    start_time = time.time()

    log("=" * 60)
    log("ColBERTv2 Overnight Experiment")
    log("Quantized (2-bit) vs Analog (float16) Residuals")
    log("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------ Step 3: SKIPPED (full 8.8M search needs ~20 GB RAM) ------
    log("")
    log("STEP 3: SKIPPED (full index search requires too much RAM)")

    # ------ Step 4: Create subset ------
    log("")
    log("=" * 40)
    log("STEP 4: Creating subset collection")
    log("=" * 40)
    try:
        create_subset()
    except Exception as e:
        log(f"Step 4 error: {e}")
        traceback.print_exc()
        log("Cannot continue without subset. Exiting.")
        sys.exit(1)

    # ------ Step 5A: Subset quantized ------
    log("")
    log("=" * 40)
    log("STEP 5A: Subset quantized index + search + eval")
    log("=" * 40)
    try:
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "step5a"],
            cwd=COLBERT_DIR
        ).returncode
        if rc != 0:
            log(f"Step 5A failed (rc={rc})")
    except Exception as e:
        log(f"Step 5A error: {e}")
        traceback.print_exc()

    # ------ Step 5B: Subset analog ------
    log("")
    log("=" * 40)
    log("STEP 5B: Subset analog index + search + eval")
    log("=" * 40)
    backup_files()
    try:
        apply_analog_patches()
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "step5b"],
            cwd=COLBERT_DIR
        ).returncode
        if rc != 0:
            log(f"Step 5B failed (rc={rc})")
    except Exception as e:
        log(f"Step 5B error: {e}")
        traceback.print_exc()
    finally:
        restore_files()

    # ------ Step 6: Compare ------
    log("")
    log("=" * 40)
    log("STEP 6: Final comparison")
    log("=" * 40)
    _step6_compare()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    log(f"Total elapsed time: {hours}h {minutes}m")
    log("Overnight experiment complete!")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        step = sys.argv[1]
        dispatch = {
            "step3": _step3_search_full_quantized,
            "step5a": _step5a_subset_quantized,
            "step5b": _step5b_subset_analog,
            "step5b_search": lambda: _step5b_search_analog(k=1000),
            "step5b_search100": lambda: _step5b_search_analog(k=100),
            "step6": _step6_compare,
        }
        if step in dispatch:
            try:
                dispatch[step]()
            except Exception as e:
                log(f"Step {step} failed with exception: {e}")
                traceback.print_exc()
                sys.exit(1)
        else:
            print(f"Unknown step: {step}")
            print(f"Available: {', '.join(dispatch.keys())}")
            sys.exit(1)
    else:
        main()
