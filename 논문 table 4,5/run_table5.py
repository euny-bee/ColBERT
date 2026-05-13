#!/usr/bin/env python3
"""
ColBERTv2 Table 5 재현: BEIR benchmark — Quantized (2-bit) vs Analog (float16)
===============================================================================
Usage (WSL):
  cd ~/ColBERT && source ~/colbert-env/bin/activate
  nohup python run_table5.py > table5.log 2>&1 &
"""

import os
import sys
import time
import json
import shutil
import torch
import subprocess
import traceback
import glob as glob_module

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

COLBERT_DIR = os.path.expanduser("~/ColBERT")
sys.path.insert(0, COLBERT_DIR)

# ============================
# Config
# ============================
CHECKPOINT = "colbert-ir/colbertv2.0"
NBITS = 2
DOC_MAXLEN = 220
QUERY_MAXLEN = 32
INDEX_BSIZE = 32
SEARCH_K = 100

DATA_ROOT = os.path.join(COLBERT_DIR, "data/table5")
RESULTS_DIR = os.path.join(COLBERT_DIR, "experiments/table5")
RESULTS_FILE = os.path.join(RESULTS_DIR, "table5_results.json")

# Smallest first for quick initial results
BEIR_DATASETS = [
    "nfcorpus", "scifact", "arguana", "scidocs", "fiqa",
    "trec-covid", "webis-touche2020", "quora",
    "nq", "dbpedia-entity", "climate-fever", "fever", "hotpotqa",
]

SUBSET_THRESHOLD = 200_000
# Analog residuals are 8x larger than quantized (128*2 vs 128*2/8 bytes per embedding).
# 200K docs → ~13M embeddings → analog ~3.5 GB RAM (safe for 16 GB system)
MAX_ANALOG_EMBEDDINGS = 15_000_000  # ~3.9 GB for float16 residuals

# Source files that need to be in original state for quantized indexing
RESIDUAL_PY = os.path.join(COLBERT_DIR, "colbert/indexing/codecs/residual.py")
RESIDUAL_EMB_PY = os.path.join(COLBERT_DIR, "colbert/indexing/codecs/residual_embeddings.py")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def save_result(results, key, metrics):
    results[key] = metrics
    results[key]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


# ============================
# Source file management
# ============================
def ensure_original_source_files():
    """Ensure residual.py and residual_embeddings.py are in original (quantized) state.
    If they're in patched (analog) state, restore via git checkout."""
    with open(RESIDUAL_EMB_PY) as f:
        content = f.read()

    if "assert residuals.dtype == torch.uint8" not in content:
        log("WARNING: Source files are in patched (analog) state. Restoring originals...")
        subprocess.run(
            ["git", "checkout", "--",
             "colbert/indexing/codecs/residual.py",
             "colbert/indexing/codecs/residual_embeddings.py"],
            cwd=COLBERT_DIR, check=True
        )
        # Clear __pycache__ to prevent stale bytecode
        for pycache in glob_module.glob(
            os.path.join(COLBERT_DIR, "colbert/indexing/codecs/__pycache__/*")
        ):
            os.remove(pycache)
        log("  Source files restored to original state")
    else:
        log("Source files are in original state (OK)")


def _apply_analog_monkey_patches():
    """Apply runtime monkey-patches for analog (float16) residual search.
    This is more reliable than file patching (avoids __pycache__ issues)."""
    import tqdm
    from colbert.indexing.codecs import residual_embeddings as rem_mod
    from colbert.indexing.codecs.residual import ResidualCodec
    from colbert.utils.utils import print_message

    ResidualEmbeddings = rem_mod.ResidualEmbeddings

    # Patch 1: __init__ — accept any dtype (not just uint8)
    def _patched_init(self, codes, residuals):
        assert codes.size(0) == residuals.size(0), (codes.size(), residuals.size())
        assert codes.dim() == 1 and residuals.dim() == 2, (codes.size(), residuals.size())
        self.codes = codes.to(torch.int32)
        self.residuals = residuals
    ResidualEmbeddings.__init__ = _patched_init

    # Patch 2: load_chunks — detect dtype from first chunk, allocate accordingly
    @classmethod
    def _patched_load_chunks(cls, index_path, chunk_idxs, num_embeddings, load_index_with_mmap=False):
        num_embeddings += 512  # pad for strides

        if load_index_with_mmap:
            raise ValueError("mmap not supported for analog index")

        print_message("#> Loading codes and residuals (analog)...")

        # Detect residual shape/dtype from first chunk
        first_chunk = cls.load(index_path, chunk_idxs[0])
        res_dim = first_chunk.residuals.shape[1]
        res_dtype = first_chunk.residuals.dtype

        codes = torch.empty(num_embeddings, dtype=torch.int32)
        residuals = torch.empty(num_embeddings, res_dim, dtype=res_dtype)

        codes_offset = 0
        for i, chunk_idx in enumerate(tqdm.tqdm(chunk_idxs)):
            chunk = first_chunk if i == 0 else cls.load(index_path, chunk_idx)
            codes_endpos = codes_offset + chunk.codes.size(0)
            codes[codes_offset:codes_endpos] = chunk.codes
            residuals[codes_offset:codes_endpos] = chunk.residuals
            codes_offset = codes_endpos

        return cls(codes, residuals)
    ResidualEmbeddings.load_chunks = _patched_load_chunks

    # Patch 3: decompress — simple centroid + float16 residual addition
    def _patched_decompress(self, compressed_embs):
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
        return torch.cat(D)
    ResidualCodec.decompress = _patched_decompress

    log("  Runtime analog monkey-patches applied")


# ============================
# BEIR download & convert to ColBERT TSV
# ============================
def download_and_convert_beir(dataset_name):
    """Download BEIR dataset and convert to ColBERT format. Returns paths dict."""
    dataset_dir = os.path.join(DATA_ROOT, "beir", dataset_name)
    collection_path = os.path.join(dataset_dir, "collection.tsv")
    queries_path = os.path.join(dataset_dir, "queries.tsv")
    qrels_path = os.path.join(dataset_dir, "qrels.tsv")

    if os.path.exists(collection_path) and os.path.exists(queries_path) and os.path.exists(qrels_path):
        log(f"  {dataset_name}: already converted, skipping download")
        num_docs = sum(1 for _ in open(collection_path))
        return {"collection": collection_path, "queries": queries_path,
                "qrels": qrels_path, "dir": dataset_dir, "num_docs": num_docs}

    log(f"  Downloading {dataset_name} from HuggingFace...")
    from datasets import load_dataset

    # HuggingFace BEIR dataset name mapping
    hf_name_map = {"webis-touche2020": "webis-touche2020-v2"}
    hf_name = hf_name_map.get(dataset_name, dataset_name)

    corpus_ds = load_dataset(f"BeIR/{hf_name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{hf_name}", "queries", split="queries")

    # Convert corpus/queries to dicts
    corpus = {str(row["_id"]): {"title": row.get("title", ""), "text": row.get("text", "")} for row in corpus_ds}
    queries = {str(row["_id"]): row["text"] for row in queries_ds}

    # qrels: download TSV directly from HuggingFace (tiny file, avoids datasets compat issues)
    import csv, urllib.request
    qrels_url = f"https://huggingface.co/datasets/BeIR/{hf_name}-qrels/resolve/main/test.tsv"
    qrels = {}
    try:
        with urllib.request.urlopen(qrels_url, timeout=30) as resp:
            reader = csv.DictReader(resp.read().decode('utf-8').splitlines(), delimiter='\t')
            for row in reader:
                qid = str(row["query-id"])
                did = str(row["corpus-id"])
                score = int(row["score"])
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][did] = score
    except Exception as e:
        log(f"  Warning: qrels download failed ({e}), trying BEIR server...")
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
        beir_download_dir = os.path.join(DATA_ROOT, "beir_raw")
        os.makedirs(beir_download_dir, exist_ok=True)
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = beir_util.download_and_unzip(url, beir_download_dir)
        _, _, qrels = GenericDataLoader(data_path).load(split="test")

    os.makedirs(dataset_dir, exist_ok=True)

    # Write collection.tsv (pid\ttext)
    log(f"  Writing collection ({len(corpus)} docs)...")
    # Sort by doc_id for deterministic ordering, remap to 0-based
    sorted_doc_ids = sorted(corpus.keys())
    docid_map = {did: idx for idx, did in enumerate(sorted_doc_ids)}

    with open(collection_path, 'w', encoding='utf-8') as f:
        for did in sorted_doc_ids:
            doc = corpus[did]
            title = doc.get("title", "").strip()
            text = doc.get("text", "").strip()
            combined = f"{title} {text}".strip() if title else text
            combined = combined.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            f.write(f"{docid_map[did]}\t{combined}\n")

    # Write queries.tsv (qid\ttext)
    log(f"  Writing queries ({len(queries)} queries)...")
    sorted_qids = sorted(queries.keys())
    qid_map = {qid: idx for idx, qid in enumerate(sorted_qids)}

    with open(queries_path, 'w', encoding='utf-8') as f:
        for qid in sorted_qids:
            q_text = queries[qid].replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            f.write(f"{qid_map[qid]}\t{q_text}\n")

    # Write qrels.tsv (qid\t0\tpid\tlabel)
    log(f"  Writing qrels...")
    with open(qrels_path, 'w', encoding='utf-8') as f:
        for qid, doc_rels in qrels.items():
            for did, label in doc_rels.items():
                if did in docid_map and qid in qid_map:
                    f.write(f"{qid_map[qid]}\t0\t{docid_map[did]}\t{label}\n")

    # Save ID mappings for trec_eval
    with open(os.path.join(dataset_dir, "docid_map.json"), 'w') as f:
        json.dump(docid_map, f)
    with open(os.path.join(dataset_dir, "qid_map.json"), 'w') as f:
        json.dump(qid_map, f)

    num_docs = len(corpus)
    log(f"  {dataset_name}: {num_docs} docs, {len(queries)} queries")

    return {"collection": collection_path, "queries": queries_path,
            "qrels": qrels_path, "dir": dataset_dir, "num_docs": num_docs}


# ============================
# Subset creation (for large datasets)
# ============================
def create_beir_subset(paths, subset_size=SUBSET_THRESHOLD):
    """Create subset preserving all qrels-relevant docs. Returns updated paths."""
    import random

    subset_dir = os.path.join(paths["dir"], "subset")
    subset_collection = os.path.join(subset_dir, "collection.tsv")
    subset_qrels = os.path.join(subset_dir, "qrels.tsv")
    subset_queries = os.path.join(subset_dir, "queries.tsv")

    if os.path.exists(subset_collection):
        num_docs = sum(1 for _ in open(subset_collection))
        log(f"  Subset already exists ({num_docs} docs)")
        return {"collection": subset_collection, "queries": subset_queries,
                "qrels": subset_qrels, "dir": subset_dir, "num_docs": num_docs}

    os.makedirs(subset_dir, exist_ok=True)
    log(f"  Creating {subset_size:,} subset...")

    # Collect relevant PIDs
    relevant_pids = set()
    with open(paths["qrels"]) as f:
        for line in f:
            parts = line.strip().split('\t')
            relevant_pids.add(int(parts[2]))

    total = paths["num_docs"]
    num_random = max(0, subset_size - len(relevant_pids))
    non_relevant = list(set(range(total)) - relevant_pids)
    random.seed(42)
    random_pids = set(random.sample(non_relevant, min(num_random, len(non_relevant))))
    subset_pids = relevant_pids | random_pids

    sorted_pids = sorted(subset_pids)
    pid_map = {old: new for new, old in enumerate(sorted_pids)}

    # Write subset collection
    written = 0
    with open(paths["collection"]) as fin, open(subset_collection, 'w', encoding='utf-8') as fout:
        for line_idx, line in enumerate(fin):
            if line_idx in subset_pids:
                parts = line.strip('\n\r ').split('\t', 1)
                text = parts[1] if len(parts) > 1 else parts[0]
                fout.write(f"{pid_map[line_idx]}\t{text}\n")
                written += 1

    # Write adjusted qrels
    with open(paths["qrels"]) as fin, open(subset_qrels, 'w', encoding='utf-8') as fout:
        for line in fin:
            parts = line.strip().split('\t')
            qid, zero, pid, label = parts[0], parts[1], int(parts[2]), parts[3]
            if pid in pid_map:
                fout.write(f"{qid}\t{zero}\t{pid_map[pid]}\t{label}\n")

    # Copy queries as-is
    shutil.copy2(paths["queries"], subset_queries)

    log(f"  Subset: {len(relevant_pids)} relevant + {len(random_pids)} random = {written}")
    return {"collection": subset_collection, "queries": subset_queries,
            "qrels": subset_qrels, "dir": subset_dir, "num_docs": written}


# ============================
# nDCG@10 evaluation
# ============================
def evaluate_ndcg(ranking_path, qrels_path):
    """Compute nDCG@10 using pytrec_eval."""
    import pytrec_eval

    # Load qrels
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            qid, pid, label = parts[0], parts[2], int(parts[3])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][pid] = label

    # Load ranking
    run = {}
    with open(ranking_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            qid, pid = parts[0], parts[1]
            rank = int(parts[2])
            score = float(parts[3]) if len(parts) > 3 else 1000 - rank
            if qid not in run:
                run[qid] = {}
            run[qid][pid] = score

    # Convert keys to strings for pytrec_eval
    qrels_str = {str(q): {str(d): v for d, v in docs.items()} for q, docs in qrels.items()}
    run_str = {str(q): {str(d): v for d, v in docs.items()} for q, docs in run.items()}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_str, {'ndcg_cut_10'})
    results = evaluator.evaluate(run_str)

    ndcg_scores = [v['ndcg_cut_10'] for v in results.values()]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    return {"nDCG@10": round(avg_ndcg, 4), "num_queries": len(ndcg_scores)}


# ============================
# Subprocess steps
# ============================
def _do_index(dataset_name, collection_path, index_name):
    """Index a collection (quantized). Also saves raw embeddings."""
    os.chdir(COLBERT_DIR)
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig

    raw_embs_dir = os.path.join(DATA_ROOT, "beir", dataset_name, "raw_embs")
    os.makedirs(raw_embs_dir, exist_ok=True)

    # Monkey-patch to save raw embeddings
    from colbert.indexing.index_saver import IndexSaver
    _orig_save_chunk = IndexSaver.save_chunk

    def _patched_save_chunk(self, chunk_idx, offset, embs, doclens):
        raw_path = os.path.join(raw_embs_dir, f"{chunk_idx}.pt")
        torch.save(embs.half().cpu(), raw_path)
        log(f"    Saved raw embeddings chunk {chunk_idx}: {embs.shape}")
        return _orig_save_chunk(self, chunk_idx, offset, embs, doclens)

    IndexSaver.save_chunk = _patched_save_chunk

    try:
        with Run().context(RunConfig(nranks=1, experiment="table5", avoid_fork_if_possible=True)):
            config = ColBERTConfig(
                nbits=NBITS, doc_maxlen=DOC_MAXLEN, query_maxlen=QUERY_MAXLEN,
                index_bsize=INDEX_BSIZE, avoid_fork_if_possible=True,
            )
            indexer = Indexer(checkpoint=CHECKPOINT, config=config)
            indexer.index(name=index_name, collection=collection_path, overwrite=True)
    finally:
        IndexSaver.save_chunk = _orig_save_chunk

    log(f"  Quantized index '{index_name}' built")


def _do_search(index_name, queries_path, ranking_path):
    """Search an index and save ranking."""
    os.chdir(COLBERT_DIR)
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    with Run().context(RunConfig(nranks=1, experiment="table5")):
        config = ColBERTConfig(nbits=NBITS, doc_maxlen=DOC_MAXLEN, query_maxlen=QUERY_MAXLEN)
        searcher = Searcher(index=index_name, config=config)
        queries = Queries(queries_path)

        log(f"  Searching {len(queries)} queries (k={SEARCH_K})...")
        start = time.time()
        ranking = searcher.search_all(queries, k=SEARCH_K)
        log(f"  Search done in {time.time()-start:.0f}s")

        os.makedirs(os.path.dirname(ranking_path), exist_ok=True)
        ranking.save(ranking_path)


def _do_search_analog(index_name, queries_path, ranking_path):
    """Search an analog index with runtime monkey-patches (no file modification)."""
    os.chdir(COLBERT_DIR)

    # Apply monkey-patches BEFORE creating Searcher
    _apply_analog_monkey_patches()

    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    with Run().context(RunConfig(nranks=1, experiment="table5")):
        config = ColBERTConfig(nbits=NBITS, doc_maxlen=DOC_MAXLEN, query_maxlen=QUERY_MAXLEN)
        searcher = Searcher(index=index_name, config=config)
        queries = Queries(queries_path)

        log(f"  Searching {len(queries)} queries (k={SEARCH_K}, analog)...")
        start = time.time()
        ranking = searcher.search_all(queries, k=SEARCH_K)
        log(f"  Search done in {time.time()-start:.0f}s")

        os.makedirs(os.path.dirname(ranking_path), exist_ok=True)
        ranking.save(ranking_path)


def _build_analog_index(dataset_name, quantized_index_name, analog_index_name):
    """Build analog index by copying quantized and replacing residuals with float16."""
    os.chdir(COLBERT_DIR)
    from colbert.indexing.codecs.residual import ResidualCodec

    q_index_dir = os.path.join(RESULTS_DIR, "indexes", quantized_index_name)
    a_index_dir = os.path.join(RESULTS_DIR, "indexes", analog_index_name)
    raw_embs_dir = os.path.join(DATA_ROOT, "beir", dataset_name, "raw_embs")

    # Copy entire quantized index
    if os.path.exists(a_index_dir):
        shutil.rmtree(a_index_dir)
    shutil.copytree(q_index_dir, a_index_dir)
    log(f"  Copied quantized index → analog index dir")

    # Load codec for centroid lookup
    codec = ResidualCodec.load(index_path=a_index_dir)

    # Replace residuals with float16 (process in sub-batches to avoid GPU OOM)
    chunk_files = sorted(glob_module.glob(os.path.join(raw_embs_dir, "*.pt")))
    for chunk_file in chunk_files:
        chunk_idx = int(os.path.basename(chunk_file).replace(".pt", ""))
        embs = torch.load(chunk_file, map_location='cpu', weights_only=True)

        # Process in sub-batches to avoid GPU OOM on large chunks
        all_residuals = []
        for sub_batch in embs.split(1 << 16):  # 65K at a time
            if codec.use_gpu:
                sub = sub_batch.cuda().half()
            else:
                sub = sub_batch.float()
            codes_sub = codec.compress_into_codes(sub, out_device='cpu')
            centroids_sub = codec.lookup_centroids(codes_sub, out_device='cpu')
            res_sub = (sub_batch.float() - centroids_sub.float()).half()
            all_residuals.append(res_sub)
        residuals = torch.cat(all_residuals)

        # Save: codes stay the same, replace residuals
        residuals_path = os.path.join(a_index_dir, f"{chunk_idx}.residuals.pt")
        torch.save(residuals, residuals_path)
        log(f"    Chunk {chunk_idx}: {residuals.shape} float16 residuals saved")

    log(f"  Analog index '{analog_index_name}' built")


# ============================
# Per-dataset pipeline
# ============================
def run_dataset(dataset_name, results):
    """Full pipeline for one BEIR dataset."""
    log(f"\n{'='*50}")
    log(f"DATASET: {dataset_name}")
    log(f"{'='*50}")

    t0 = time.time()

    # 1. Download & convert
    log("Step 1: Download & convert")
    paths = download_and_convert_beir(dataset_name)

    # 2. Subset if needed
    if paths["num_docs"] > SUBSET_THRESHOLD:
        log(f"Step 2: Creating subset ({paths['num_docs']:,} > {SUBSET_THRESHOLD:,})")
        paths = create_beir_subset(paths)
    else:
        log(f"Step 2: No subset needed ({paths['num_docs']:,} docs)")

    q_index = f"beir.{dataset_name}.quantized"
    a_index = f"beir.{dataset_name}.analog"
    ranking_dir = os.path.join(RESULTS_DIR, "rankings")
    os.makedirs(ranking_dir, exist_ok=True)

    # 3. Quantized indexing (subprocess for clean imports)
    log("Step 3: Quantized indexing")
    rc = subprocess.run([
        sys.executable, os.path.abspath(__file__),
        "index", dataset_name, paths["collection"], q_index
    ], cwd=COLBERT_DIR).returncode
    if rc != 0:
        log(f"  FAILED (rc={rc}), skipping dataset")
        return

    # 4. Quantized search
    log("Step 4: Quantized search")
    q_ranking = os.path.join(ranking_dir, f"{dataset_name}.quantized.tsv")
    rc = subprocess.run([
        sys.executable, os.path.abspath(__file__),
        "search", q_index, paths["queries"], q_ranking
    ], cwd=COLBERT_DIR).returncode
    if rc != 0:
        log(f"  Quantized search failed (rc={rc})")

    # 4b. Evaluate quantized
    if os.path.exists(q_ranking):
        metrics = evaluate_ndcg(q_ranking, paths["qrels"])
        log(f"  Quantized nDCG@10 = {metrics['nDCG@10']}")
        save_result(results, f"{dataset_name}.quantized", metrics)

    # 5. Check embedding count before analog (OOM guard)
    q_index_dir = os.path.join(RESULTS_DIR, "indexes", q_index)
    metadata_path = os.path.join(q_index_dir, "metadata.json")
    num_embeddings = 0
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            num_embeddings = json.load(f).get("num_embeddings", 0)
    analog_ram_gb = num_embeddings * 128 * 2 / 1e9
    log(f"  Embeddings: {num_embeddings:,} → analog RAM ~{analog_ram_gb:.1f} GB")

    if num_embeddings > MAX_ANALOG_EMBEDDINGS:
        log(f"  SKIPPING analog: {num_embeddings:,} embeddings > {MAX_ANALOG_EMBEDDINGS:,} limit (OOM risk)")
        log(f"  Quantized-only result saved for {dataset_name}")
        # Cleanup raw embeddings
        raw_embs_dir = os.path.join(DATA_ROOT, "beir", dataset_name, "raw_embs")
        if os.path.exists(raw_embs_dir):
            shutil.rmtree(raw_embs_dir)
        elapsed = time.time() - t0
        log(f"  {dataset_name} done in {elapsed/60:.1f} min (quantized only)")
        return

    # 5b. Build analog index (no re-encoding)
    log("Step 5: Build analog index")
    rc = subprocess.run([
        sys.executable, os.path.abspath(__file__),
        "build_analog", dataset_name, q_index, a_index
    ], cwd=COLBERT_DIR).returncode
    if rc != 0:
        log(f"  Analog build failed (rc={rc})")

    # 6. Analog search (runtime monkey-patching, no file modification)
    log("Step 6: Analog search")
    a_ranking = os.path.join(ranking_dir, f"{dataset_name}.analog.tsv")
    rc = subprocess.run([
        sys.executable, os.path.abspath(__file__),
        "search_analog", a_index, paths["queries"], a_ranking
    ], cwd=COLBERT_DIR).returncode
    if rc != 0:
        log(f"  Analog search failed (rc={rc})")

    # 6b. Evaluate analog
    if os.path.exists(a_ranking):
        metrics = evaluate_ndcg(a_ranking, paths["qrels"])
        log(f"  Analog nDCG@10 = {metrics['nDCG@10']}")
        save_result(results, f"{dataset_name}.analog", metrics)

    # 7. Cleanup raw embeddings
    raw_embs_dir = os.path.join(DATA_ROOT, "beir", dataset_name, "raw_embs")
    if os.path.exists(raw_embs_dir):
        shutil.rmtree(raw_embs_dir)
        log("  Raw embeddings deleted")

    elapsed = time.time() - t0
    log(f"  {dataset_name} done in {elapsed/60:.1f} min")


# ============================
# Summary
# ============================
def print_summary(results):
    print(f"\n{'='*70}")
    print("  ColBERTv2 Table 5: BEIR — Quantized (2-bit) vs Analog (float16)")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20} {'Quantized':<12} {'Analog':<12} {'Delta':<10}")
    print(f"  {'-'*54}")

    for ds in BEIR_DATASETS:
        vq = results.get(f"{ds}.quantized", {}).get("nDCG@10")
        va = results.get(f"{ds}.analog", {}).get("nDCG@10")
        if isinstance(vq, (int, float)) and isinstance(va, (int, float)):
            delta = va - vq
            sign = "+" if delta > 0 else ""
            print(f"  {ds:<20} {vq:<12.4f} {va:<12.4f} {sign}{delta:.4f}")
        elif vq is not None:
            print(f"  {ds:<20} {vq:<12.4f} {'N/A':<12} {'N/A'}")
        else:
            print(f"  {ds:<20} {'N/A':<12} {'N/A':<12} {'N/A'}")

    # Average
    q_scores = [results.get(f"{ds}.quantized", {}).get("nDCG@10") for ds in BEIR_DATASETS]
    a_scores = [results.get(f"{ds}.analog", {}).get("nDCG@10") for ds in BEIR_DATASETS]
    q_valid = [s for s in q_scores if isinstance(s, (int, float))]
    a_valid = [s for s in a_scores if isinstance(s, (int, float))]
    if q_valid and a_valid:
        q_avg = sum(q_valid) / len(q_valid)
        a_avg = sum(a_valid) / len(a_valid)
        print(f"  {'-'*54}")
        print(f"  {'AVG (completed)':<20} {q_avg:<12.4f} {a_avg:<12.4f} {'+' if a_avg-q_avg>0 else ''}{a_avg-q_avg:.4f}")

    print(f"{'='*70}\n")


# ============================
# Main
# ============================
def main():
    log("=" * 60)
    log("ColBERTv2 Table 5: BEIR — Quantized vs Analog")
    log("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Ensure source files are in original state for quantized indexing
    ensure_original_source_files()

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    for dataset_name in BEIR_DATASETS:
        # Skip already completed datasets
        q_key = f"{dataset_name}.quantized"
        a_key = f"{dataset_name}.analog"
        if q_key in results and a_key in results:
            log(f"\nSkipping {dataset_name} (already completed)")
            continue

        try:
            run_dataset(dataset_name, results)
        except Exception as e:
            log(f"ERROR on {dataset_name}: {e}")
            traceback.print_exc()
            continue

    print_summary(results)
    log("All done!")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "index":
            _, _, dataset_name, collection_path, index_name = sys.argv
            _do_index(dataset_name, collection_path, index_name)

        elif cmd == "search":
            _, _, index_name, queries_path, ranking_path = sys.argv
            _do_search(index_name, queries_path, ranking_path)

        elif cmd == "search_analog":
            _, _, index_name, queries_path, ranking_path = sys.argv
            _do_search_analog(index_name, queries_path, ranking_path)

        elif cmd == "build_analog":
            _, _, dataset_name, q_index, a_index = sys.argv
            _build_analog_index(dataset_name, q_index, a_index)

        elif cmd == "summary":
            results = {}
            if os.path.exists(RESULTS_FILE):
                with open(RESULTS_FILE) as f:
                    results = json.load(f)
            print_summary(results)

        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        main()
