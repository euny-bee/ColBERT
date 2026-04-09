#!/usr/bin/env python3
"""
run_beir.py
-----------
SciFact / NFCorpus / SCIDOCS / FiQA 4개 데이터셋에 대해
float16 vs 2-bit 인덱스 비교 실험을 자동으로 수행.

각 데이터셋마다:
  1. HuggingFace 다운로드
  2. make_fair (collection.tsv, queries.test.tsv, qrels.test.int.tsv)
  3. step1: float16 + 2-bit 인덱싱
  4. step2: 검색 + nDCG@10
  5. 인덱스 삭제 (C: 공간 회수)

결과: experiments/beir_all_results.json

실행:
  python run_beir.py
"""

import os
import sys
import gzip
import glob
import json
import math
import shutil
import subprocess
import time
import torch
import numpy as np

COLBERT_DIR = os.path.expanduser("~/ColBERT")
BEIR_DIR    = "D:/beir"
CHECKPOINT  = "colbert-ir/colbertv2.0"
NBITS       = 2
DIM         = 128
INDEX_BSIZE = 64
SEARCH_K    = 1000
NDOCS       = 4096

RESULTS_FILE = os.path.join(COLBERT_DIR, "experiments/beir_all_results.json")

# 인덱스는 D: 드라이브에 저장 (용량 확보)
INDEX_BASE = "D:/beir_indexes"

os.chdir(COLBERT_DIR)
sys.path.insert(0, COLBERT_DIR)

DATASETS = [
    {"name": "codesearchnet", "hf_corpus": "CoIR-Retrieval/CodeSearchNet","hf_qrels": "CoIR-Retrieval/CodeSearchNet", "subset": "python"},
    {"name": "bright",        "hf_corpus": "xlangai/BRIGHT",             "hf_qrels": None,                       "bright": True},
    {"name": "mrtydi_ko",     "mrtydi_ko":  True},
    {"name": "miracl_ko",     "miracl_ko":  True},
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. 다운로드 + 변환 (gz / parquet 모두 처리)
# ============================================================
def _parquet_dir_to_jsonl(parquet_dir, out_path):
    """parquet 디렉토리 → jsonl 변환 (pandas 사용)"""
    import pandas as pd
    import json as _json
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        return False
    with open(out_path, "w", encoding="utf-8") as fout:
        for pf in files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                fout.write(_json.dumps(dict(row), ensure_ascii=False) + "\n")
    return True


def download(ds):
    name = ds["name"]
    base = os.path.join(BEIR_DIR, name)
    os.makedirs(base, exist_ok=True)

    # Mr.TyDi / MIRACL: make_fair에서 HF datasets 직접 로드 → 별도 다운로드 불필요
    if ds.get("mrtydi_ko") or ds.get("miracl_ko"):
        return

    from huggingface_hub import snapshot_download

    # corpus + queries
    corpus_dir = os.path.join(base, "hf")
    if not os.path.exists(os.path.join(base, "corpus.jsonl")):
        if not os.path.exists(corpus_dir) or not os.listdir(corpus_dir):
            log(f"  Downloading {ds['hf_corpus']}...")
            snapshot_download(repo_id=ds["hf_corpus"], repo_type="dataset",
                              local_dir=corpus_dir)

        # 방법 1: .gz 파일
        gz_files = glob.glob(os.path.join(corpus_dir, "*.gz"))
        if gz_files:
            for gz in gz_files:
                fname = os.path.basename(gz[:-3])
                out = os.path.join(base, fname)
                log(f"  Decompressing {os.path.basename(gz)}...")
                with gzip.open(gz, "rb") as fi, open(out, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
        else:
            # 방법 2: parquet 디렉토리 (subset 지원)
            subset = ds.get("subset", "")
            for split in ["corpus", "queries"]:
                # subset prefix 시도 (예: python-corpus)
                pdir = os.path.join(corpus_dir, f"{subset}-{split}" if subset else split)
                if not os.path.isdir(pdir):
                    pdir = os.path.join(corpus_dir, split)
                out  = os.path.join(base, f"{split}.jsonl")
                if os.path.isdir(pdir):
                    log(f"  Converting parquet → {split}.jsonl...")
                    _parquet_dir_to_jsonl(pdir, out)
    else:
        log(f"  corpus.jsonl already exists, skip download.")

    # qrels (BRIGHT는 query 안에 내장 → 별도 다운로드 불필요)
    if ds.get("bright"):
        return

    qrels_dir = os.path.join(base, "hf-qrels")
    qrels_dst = os.path.join(base, "qrels", "test.tsv")
    if not os.path.exists(qrels_dst):
        log(f"  Downloading {ds['hf_qrels']}...")
        snapshot_download(repo_id=ds["hf_qrels"], repo_type="dataset",
                          local_dir=qrels_dir)
        os.makedirs(os.path.join(base, "qrels"), exist_ok=True)
        # test.tsv 또는 test.tsv.gz 탐색
        src = os.path.join(qrels_dir, "test.tsv")
        if not os.path.exists(src):
            # parquet → tsv 변환 시도 (subset 지원, test split만)
            import pandas as pd
            subset = ds.get("subset", "")
            # subset prefix 시도 (예: python-qrels/test-*.parquet)
            if subset:
                pfiles = glob.glob(os.path.join(qrels_dir, f"{subset}-qrels", "test*.parquet"))
                if not pfiles:
                    pfiles = glob.glob(os.path.join(qrels_dir, f"{subset}*", "test*.parquet"))
            else:
                pfiles = glob.glob(os.path.join(qrels_dir, "**/test*.parquet"), recursive=True)
            if pfiles:
                rows = []
                for pf in pfiles:
                    df = pd.read_parquet(pf)
                    rows.append(df)
                df_all = pd.concat(rows)
                os.makedirs(os.path.join(base, "qrels"), exist_ok=True)
                df_all.to_csv(qrels_dst, sep="\t", index=False)
                log(f"  qrels parquet → {qrels_dst}")
        else:
            shutil.copy(src, qrels_dst)
    else:
        log(f"  qrels/test.tsv already exists, skip download.")


# ============================================================
# 2-BRIGHT. BRIGHT 전용 make_fair
# ============================================================
def _make_fair_bright(base, out_collection, out_queries, out_qrels):
    import pandas as pd
    import json as _json

    if os.path.exists(out_collection) and os.path.exists(out_queries) and os.path.exists(out_qrels):
        log(f"  BRIGHT fair files already exist, skip.")
        with open(out_queries, encoding="utf-8") as f:
            return sum(1 for _ in f)

    hf_dir = os.path.join(base, "hf")

    # corpus: documents config (id, content 컬럼)
    log("  BRIGHT: converting corpus parquet...")
    corpus_files = sorted(glob.glob(os.path.join(hf_dir, "documents", "**/*.parquet"), recursive=True))
    if not corpus_files:
        corpus_files = sorted(glob.glob(os.path.join(hf_dir, "**/*corpus*.parquet"), recursive=True))

    orig_to_pid = {}
    pid = 0
    with open(out_collection, "w", encoding="utf-8") as fout:
        for pf in corpus_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                doc_id = str(row.get("id", row.get("_id", "")))
                content = str(row.get("content", row.get("text", ""))).replace("\t", " ").replace("\n", " ").replace("\r", " ")
                orig_to_pid[doc_id] = pid
                fout.write(f"{pid}\t{content}\n")
                pid += 1
    log(f"  BRIGHT corpus: {pid:,} passages")

    # queries + qrels: reasoning config (query, id, gold_ids 컬럼)
    log("  BRIGHT: converting queries + building qrels...")
    query_files = sorted(glob.glob(os.path.join(hf_dir, "examples", "**/*.parquet"), recursive=True))
    if not query_files:
        query_files = sorted(glob.glob(os.path.join(hf_dir, "**/*reas*.parquet"), recursive=True))

    int_qid = 0
    with open(out_queries, "w", encoding="utf-8") as fq, \
         open(out_qrels, "w", encoding="utf-8") as fr:
        for pf in query_files:
            df = pd.read_parquet(pf)
            for _, row in df.iterrows():
                qid_orig = str(row.get("id", int_qid))
                query_text = str(row.get("query", "")).replace("\t", " ").replace("\n", " ").replace("\r", " ")
                gold_ids = row.get("gold_ids", [])
                if isinstance(gold_ids, str):
                    gold_ids = _json.loads(gold_ids)

                fq.write(f"{int_qid}\t{query_text}\n")

                for gid in gold_ids:
                    new_pid = orig_to_pid.get(str(gid))
                    if new_pid is not None:
                        fr.write(f"{int_qid}\t{new_pid}\t1\n")

                int_qid += 1

    log(f"  BRIGHT queries: {int_qid}, qrels written")
    return int_qid


# ============================================================
# 2-MRTYDI. Mr.TyDi Korean 전용 make_fair
# ============================================================
def _make_fair_mrtydi_ko(base, out_collection, out_queries, out_qrels):
    import json as _json
    from huggingface_hub import snapshot_download

    if os.path.exists(out_collection) and os.path.exists(out_queries) and os.path.exists(out_qrels):
        log("  Mr.TyDi Ko fair files already exist, skip.")
        with open(out_queries, encoding="utf-8") as f:
            return sum(1 for _ in f)

    cache_dir = os.path.join(base, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def _parse_file(path):
        """jsonl / jsonl.gz / parquet 파일을 row dict 리스트로 반환"""
        import gzip as _gz
        if path.endswith(".parquet"):
            import pandas as pd
            return pd.read_parquet(path).to_dict("records")
        rows = []
        opener = _gz.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(_json.loads(line))
        return rows

    # --- Corpus ---
    corpus_hf = os.path.join(cache_dir, "mr-tydi-corpus")
    if not os.path.exists(corpus_hf) or not os.listdir(corpus_hf):
        log("  Downloading castorini/mr-tydi-corpus...")
        snapshot_download("castorini/mr-tydi-corpus", repo_type="dataset",
                          local_dir=corpus_hf)

    # 직접 경로 사용 (mrtydi-v1.1-korean/corpus.jsonl.gz)
    corpus_file = os.path.join(corpus_hf, "mrtydi-v1.1-korean", "corpus.jsonl.gz")
    kor_files = [corpus_file] if os.path.exists(corpus_file) else []
    if not kor_files:
        # 폴백: glob으로 탐색
        kor_files = sorted(glob.glob(os.path.join(corpus_hf, "**", "*korean*", "corpus*"), recursive=True))
    log(f"  Corpus files: {[os.path.basename(f) for f in kor_files]}")

    orig_to_pid = {}
    pid = 0
    with open(out_collection, "w", encoding="utf-8") as fout:
        for cf in kor_files:
            for row in _parse_file(cf):
                doc_id = str(row.get("docid", row.get("id", "")))
                title  = str(row.get("title", "")).replace("\t"," ").replace("\n"," ").replace("\r"," ")
                text   = str(row.get("text", row.get("contents",""))).replace("\t"," ").replace("\n"," ").replace("\r"," ")
                combined = f"{title} {text}".strip() if title else text
                orig_to_pid[doc_id] = pid
                fout.write(f"{pid}\t{combined}\n")
                pid += 1
    log(f"  Corpus: {pid:,} passages")

    # --- Queries + Qrels ---
    queries_hf = os.path.join(cache_dir, "mr-tydi")
    if not os.path.exists(queries_hf) or not os.listdir(queries_hf):
        log("  Downloading castorini/mr-tydi...")
        snapshot_download("castorini/mr-tydi", repo_type="dataset",
                          local_dir=queries_hf)

    # ir-format-data 사용 (TSV 포맷)
    ir_dir = os.path.join(queries_hf, "mrtydi-v1.1-korean", "ir-format-data")
    topics_file = os.path.join(ir_dir, "topics.test.txt")
    qrels_file  = os.path.join(ir_dir, "qrels.test.txt")
    log(f"  Topics: {os.path.basename(topics_file)}, Qrels: {os.path.basename(qrels_file)}")

    # topics: {qid}\t{query}
    orig_qid_to_text = {}
    with open(topics_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                orig_qid_to_text[parts[0]] = parts[1].replace("\t"," ").replace("\r"," ")

    # qrels: {qid}\tQ0\t{docid}\t{rel}
    from collections import defaultdict
    orig_qrels = defaultdict(list)
    with open(qrels_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                orig_qrels[parts[0]].append((parts[2], int(parts[3])))

    int_qid = 0
    with open(out_queries, "w", encoding="utf-8") as fq, \
         open(out_qrels,   "w", encoding="utf-8") as fr:
        for orig_qid, query_text in orig_qid_to_text.items():
            fq.write(f"{int_qid}\t{query_text}\n")
            for docid, rel in orig_qrels.get(orig_qid, []):
                new_pid = orig_to_pid.get(docid)
                if new_pid is not None and rel >= 1:
                    fr.write(f"{int_qid}\t{new_pid}\t{rel}\n")
            int_qid += 1
    log(f"  Queries: {int_qid}")
    return int_qid


# ============================================================
# 2-MIRACL. MIRACL Korean 전용 make_fair
# ============================================================
def _make_fair_miracl_ko(base, out_collection, out_queries, out_qrels):
    import json as _json
    from huggingface_hub import snapshot_download

    if os.path.exists(out_collection) and os.path.exists(out_queries) and os.path.exists(out_qrels):
        log("  MIRACL Ko fair files already exist, skip.")
        with open(out_queries, encoding="utf-8") as f:
            return sum(1 for _ in f)

    cache_dir = os.path.join(base, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def _parse_file(path):
        import gzip as _gz
        if path.endswith(".parquet"):
            import pandas as pd
            return pd.read_parquet(path).to_dict("records")
        rows = []
        opener = _gz.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(_json.loads(line))
        return rows

    # --- Corpus ---
    corpus_hf = os.path.join(cache_dir, "miracl-corpus")
    if not os.path.exists(corpus_hf) or not os.listdir(corpus_hf):
        log("  Downloading miracl/miracl-corpus (ko)...")
        snapshot_download("miracl/miracl-corpus", repo_type="dataset",
                          local_dir=corpus_hf)

    # 직접 경로 사용 (miracl-corpus-v1.0-ko/docs-*.jsonl.gz)
    ko_corpus_dir = os.path.join(corpus_hf, "miracl-corpus-v1.0-ko")
    kor_files = sorted(glob.glob(os.path.join(ko_corpus_dir, "docs-*.jsonl.gz")))
    if not kor_files:
        kor_files = sorted(glob.glob(os.path.join(corpus_hf, "**", "*ko*", "docs*.jsonl.gz"), recursive=True))
    log(f"  Corpus files: {[os.path.basename(f) for f in kor_files]}")

    orig_to_pid = {}
    pid = 0
    with open(out_collection, "w", encoding="utf-8") as fout:
        for cf in kor_files:
            for row in _parse_file(cf):
                doc_id = str(row.get("docid", row.get("id", "")))
                title  = str(row.get("title","")).replace("\t"," ").replace("\n"," ").replace("\r"," ")
                text   = str(row.get("text", row.get("contents",""))).replace("\t"," ").replace("\n"," ").replace("\r"," ")
                combined = f"{title} {text}".strip() if title else text
                orig_to_pid[doc_id] = pid
                fout.write(f"{pid}\t{combined}\n")
                pid += 1
    log(f"  Corpus: {pid:,} passages")

    # --- Queries + Qrels (TSV 포맷) ---
    # topics: {qid}\t{query}
    # qrels:  {qid}\tQ0\t{docid}\t{relevance}  (TREC format)
    queries_hf = os.path.join(cache_dir, "miracl")
    if not os.path.exists(queries_hf) or not os.listdir(queries_hf):
        log("  Downloading miracl/miracl (ko)...")
        snapshot_download("miracl/miracl", repo_type="dataset",
                          local_dir=queries_hf)

    ko_query_dir = os.path.join(queries_hf, "miracl-v1.0-ko")
    topics_file = os.path.join(ko_query_dir, "topics", "topics.miracl-v1.0-ko-dev.tsv")
    qrels_file  = os.path.join(ko_query_dir, "qrels",  "qrels.miracl-v1.0-ko-dev.tsv")
    log(f"  Topics file: {os.path.basename(topics_file)}")
    log(f"  Qrels file:  {os.path.basename(qrels_file)}")

    # topics: orig_qid -> query_text
    orig_qid_to_text = {}
    with open(topics_file, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                orig_qid_to_text[parts[0]] = parts[1].replace("\t"," ").replace("\r"," ")

    # qrels: orig_qid -> [(docid, rel), ...]
    from collections import defaultdict
    orig_qrels = defaultdict(list)
    with open(qrels_file, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _q0, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                orig_qrels[qid].append((docid, rel))

    int_qid = 0
    with open(out_queries, "w", encoding="utf-8") as fq, \
         open(out_qrels,   "w", encoding="utf-8") as fr:
        for orig_qid, query_text in orig_qid_to_text.items():
            fq.write(f"{int_qid}\t{query_text}\n")
            for docid, rel in orig_qrels.get(orig_qid, []):
                new_pid = orig_to_pid.get(docid)
                if new_pid is not None and rel >= 1:
                    fr.write(f"{int_qid}\t{new_pid}\t{rel}\n")
            int_qid += 1
    log(f"  Queries: {int_qid}")
    return int_qid


# ============================================================
# 2. make_fair
# ============================================================
def make_fair(ds):
    name = ds["name"]
    base = os.path.join(BEIR_DIR, name)

    out_collection = os.path.join(base, "collection.tsv")
    out_queries    = os.path.join(base, "queries.test.tsv")
    out_qrels      = os.path.join(base, "qrels.test.int.tsv")

    # BRIGHT 전용 처리
    if ds.get("bright"):
        return _make_fair_bright(base, out_collection, out_queries, out_qrels)

    # Mr.TyDi Korean 전용 처리
    if ds.get("mrtydi_ko"):
        return _make_fair_mrtydi_ko(base, out_collection, out_queries, out_qrels)

    # MIRACL Korean 전용 처리
    if ds.get("miracl_ko"):
        return _make_fair_miracl_ko(base, out_collection, out_queries, out_qrels)

    corpus_path  = os.path.join(base, "corpus.jsonl")
    queries_path = os.path.join(base, "queries.jsonl")
    qrels_path   = os.path.join(base, "qrels", "test.tsv")

    if os.path.exists(out_collection) and os.path.exists(out_queries) and os.path.exists(out_qrels):
        log(f"  Fair files already exist, skip make_fair.")
        # 쿼리 수 반환
        with open(out_queries, encoding="utf-8") as f:
            n_q = sum(1 for _ in f)
        return n_q

    import json as _json

    # qrels 로드 (헤더 있을 수 있음)
    qrel_rows = []
    relevant_doc_ids = set()
    test_query_ids   = set()
    with open(qrels_path, encoding="utf-8") as f:
        header_line = f.readline().strip().split("\t")
        # 헤더 컬럼 인덱스 탐색
        try:
            qid_col  = header_line.index("query-id")  if "query-id"  in header_line else None
            pid_col  = header_line.index("corpus-id") if "corpus-id" in header_line else None
            scr_col  = header_line.index("score")     if "score"     in header_line else None
        except Exception:
            qid_col = pid_col = scr_col = None

        # 헤더 아님 → 첫 줄도 데이터
        try:
            int(header_line[0])
            f.seek(0)
            qid_col = pid_col = scr_col = None  # 표준 0,1,2 순서
        except ValueError:
            pass  # 헤더였음

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                if qid_col is not None:
                    orig_qid   = parts[qid_col]
                    orig_doc_id= parts[pid_col]
                    score      = parts[scr_col]
                else:
                    orig_qid, orig_doc_id, score = parts[0], parts[1], parts[2]
                score_int = int(float(score))
            except (ValueError, IndexError):
                continue
            qrel_rows.append((orig_qid, orig_doc_id, score_int))
            test_query_ids.add(orig_qid)
            if score_int >= 1:
                relevant_doc_ids.add(orig_doc_id)

    log(f"  qrels: {len(qrel_rows)} rows, {len(test_query_ids)} queries, "
        f"{len(relevant_doc_ids)} relevant docs")

    # corpus 전체 변환 (string doc_id → int pid, pid == line_idx)
    orig_to_pid = {}
    with open(corpus_path, encoding="utf-8") as fin, \
         open(out_collection, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            doc = _json.loads(line)
            orig_id = doc["_id"]
            title = doc.get("title", "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
            text  = doc.get("text",  "").replace("\t", " ").replace("\n", " ").replace("\r", " ")
            combined = f"{title} {text}".strip() if title else text
            orig_to_pid[orig_id] = i
            fout.write(f"{i}\t{combined}\n")

    n_docs = len(orig_to_pid)
    log(f"  corpus: {n_docs:,} passages → collection.tsv")

    # query ID 리매핑 (string → int)
    orig_qid_to_int = {}
    int_counter = 0
    with open(queries_path, encoding="utf-8") as fin, \
         open(out_queries, "w", encoding="utf-8") as fout:
        for line in fin:
            q = _json.loads(line)
            if q["_id"] not in test_query_ids:
                continue
            text = q["text"].replace("\t", " ").replace("\n", " ")
            orig_qid_to_int[q["_id"]] = int_counter
            fout.write(f"{int_counter}\t{text}\n")
            int_counter += 1

    log(f"  queries: {int_counter} → queries.test.tsv")

    # qrels 리매핑
    score_offset = ds.get("score_offset", 0)
    mapped = 0
    with open(out_qrels, "w", encoding="utf-8") as fout:
        for orig_qid, orig_doc_id, score in qrel_rows:
            if orig_qid not in orig_qid_to_int or orig_doc_id not in orig_to_pid:
                continue
            fout.write(f"{orig_qid_to_int[orig_qid]}\t{orig_to_pid[orig_doc_id]}\t{score + score_offset}\n")
            mapped += 1

    log(f"  qrels: {mapped} rows → qrels.test.int.tsv")
    return int_counter


# ============================================================
# 3. 인덱싱 (subprocess로 분리)
# ============================================================
def build_indexes(ds):
    name = ds["name"]
    base        = os.path.join(BEIR_DIR, name)
    index_root  = os.path.join(INDEX_BASE, f"beir_{name}", "indexes")
    raw_dir     = os.path.join(base, "raw_embs")
    collection  = os.path.join(base, "collection.tsv")
    analog_name = f"{name}.analog"
    bit2_name   = f"{name}.2bit"

    log(f"  [A] Encoding...")
    rc = subprocess.run(
        [sys.executable, os.path.abspath(__file__),
         "encode", name],
        cwd=COLBERT_DIR
    ).returncode
    if rc != 0:
        raise RuntimeError(f"Encoding failed for {name}")

    log(f"  [B] Building float16 index...")
    _build_analog(name)

    log(f"  [C] Building 2-bit index...")
    _build_2bit(name)

    # raw embeddings 삭제
    if os.path.exists(raw_dir):
        shutil.rmtree(raw_dir)
        log(f"  raw_embs cleaned.")


def _do_encode(name):
    """subprocess 진입점: 인코딩 + raw embs 저장"""
    from colbert import Indexer
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.indexing.index_saver import IndexSaver

    base       = os.path.join(BEIR_DIR, name)
    collection = os.path.join(base, "collection.tsv")
    index_root = os.path.join(INDEX_BASE, name)
    raw_dir    = os.path.join(base, "raw_embs")
    analog_name = f"{name}.analog"

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(index_root, exist_ok=True)

    analog_path = os.path.join(index_root, analog_name)
    if os.path.exists(analog_path):
        shutil.rmtree(analog_path)

    _orig_save_chunk = IndexSaver.save_chunk

    def _patched_save_chunk(self, chunk_idx, offset, embs, doclens):
        raw_path = os.path.join(raw_dir, f"{chunk_idx}.pt")
        torch.save(embs.half().cpu(), raw_path)
        return _orig_save_chunk(self, chunk_idx, offset, embs, doclens)

    IndexSaver.save_chunk = _patched_save_chunk

    try:
        with Run().context(RunConfig(nranks=1, experiment=f"beir_{name}",
                                     root=INDEX_BASE,
                                     avoid_fork_if_possible=True)):
            config = ColBERTConfig(
                nbits=NBITS, doc_maxlen=220, query_maxlen=32,
                index_bsize=INDEX_BSIZE, avoid_fork_if_possible=True,
            )
            indexer = Indexer(checkpoint=CHECKPOINT, config=config)
            indexer.index(name=analog_name, collection=collection, overwrite=True)
    finally:
        IndexSaver.save_chunk = _orig_save_chunk


def _build_analog(name):
    from colbert.indexing.codecs.residual import ResidualCodec

    base        = os.path.join(BEIR_DIR, name)
    index_root  = os.path.join(INDEX_BASE, f"beir_{name}", "indexes")
    raw_dir     = os.path.join(base, "raw_embs")
    analog_name = f"{name}.analog"
    index_path  = os.path.join(index_root, analog_name)

    codec = ResidualCodec.load(index_path=index_path)
    chunk_files = sorted(
        glob.glob(os.path.join(raw_dir, "*.pt")),
        key=lambda p: int(os.path.basename(p).replace(".pt", ""))
    )

    for chunk_file in chunk_files:
        chunk_idx = int(os.path.basename(chunk_file).replace(".pt", ""))
        embs = torch.load(chunk_file, map_location="cpu", weights_only=True)

        all_residuals = []
        for sub in embs.split(1 << 16):
            if codec.use_gpu:
                sub_dev = sub.cuda().half()
            else:
                sub_dev = sub.float()
            codes     = codec.compress_into_codes(sub_dev, out_device="cpu")
            centroids = codec.lookup_centroids(codes, out_device="cpu")
            residuals = sub.half() - centroids.half()
            all_residuals.append(residuals.cpu())
            if codec.use_gpu:
                torch.cuda.empty_cache()

        chunk_residuals = torch.cat(all_residuals)
        out_path = os.path.join(index_path, f"{chunk_idx}.residuals.pt")
        torch.save(chunk_residuals, out_path)

    log(f"  float16 index done → {index_path}")


def _build_2bit(name):
    index_root  = os.path.join(INDEX_BASE, f"beir_{name}", "indexes")
    base        = os.path.join(BEIR_DIR, name)
    raw_dir     = os.path.join(base, "raw_embs")
    analog_name = f"{name}.analog"
    bit2_name   = f"{name}.2bit"
    analog_path = os.path.join(index_root, analog_name)
    bit2_path   = os.path.join(index_root, bit2_name)

    if os.path.exists(bit2_path):
        shutil.rmtree(bit2_path)
    shutil.copytree(analog_path, bit2_path)

    residual_files = sorted(
        glob.glob(os.path.join(analog_path, "*.residuals.pt")),
        key=lambda p: int(os.path.basename(p).split(".")[0])
    )

    # bucket_cutoffs 계산
    SAMPLE_PER_CHUNK = 300_000
    all_samples = []
    for rf in residual_files:
        r = torch.load(rf, map_location="cpu", weights_only=True)
        flat = r.flatten()
        if flat.numel() > SAMPLE_PER_CHUNK:
            idx = torch.randperm(flat.numel())[:SAMPLE_PER_CHUNK]
            flat = flat[idx]
        all_samples.append(flat)
        del r

    all_flat = torch.cat(all_samples)
    if all_flat.numel() > 5_000_000:
        idx = torch.randperm(all_flat.numel())[:5_000_000]
        all_flat = all_flat[idx]

    num_buckets = 2 ** NBITS
    cutoff_qs = torch.linspace(0, 1, num_buckets + 1)[1:-1]
    bucket_cutoffs = torch.quantile(all_flat.float(), cutoff_qs)

    bounds = torch.cat([torch.tensor([-1e4]), bucket_cutoffs, torch.tensor([1e4])])
    flat_f32 = all_flat.float()
    bucket_weights = torch.stack([
        flat_f32[(flat_f32 >= bounds[i]) & (flat_f32 < bounds[i + 1])].median()
        for i in range(num_buckets)
    ])
    torch.save((bucket_cutoffs, bucket_weights), os.path.join(bit2_path, "buckets.pt"))

    arange_bits = torch.arange(0, NBITS, dtype=torch.uint8)
    for rf in residual_files:
        chunk_idx = int(os.path.basename(rf).split(".")[0])
        res_f16 = torch.load(rf, map_location="cpu", weights_only=True)
        res = torch.bucketize(res_f16.float(), bucket_cutoffs).to(torch.uint8)
        res = res.unsqueeze(-1).expand(*res.size(), NBITS)
        res = res >> arange_bits & 1
        packed_np = np.packbits(np.asarray(res.contiguous().flatten()))
        packed    = torch.as_tensor(packed_np, dtype=torch.uint8)
        packed    = packed.reshape(res_f16.size(0), DIM // 8 * NBITS)
        out_path  = os.path.join(bit2_path, f"{chunk_idx}.residuals.pt")
        torch.save(packed, out_path)

    log(f"  2-bit index done → {bit2_path}")


# ============================================================
# 4. float16 검색용 monkey-patch
# ============================================================
def _apply_analog_patches():
    import tqdm
    from colbert.indexing.codecs import residual_embeddings as rem_mod
    from colbert.indexing.codecs.residual import ResidualCodec
    from colbert.utils.utils import print_message

    ResidualEmbeddings = rem_mod.ResidualEmbeddings

    def _init(self, codes, residuals):
        assert codes.size(0) == residuals.size(0)
        assert codes.dim() == 1 and residuals.dim() == 2
        self.codes     = codes.to(torch.int32)
        self.residuals = residuals
    ResidualEmbeddings.__init__ = _init

    @classmethod
    def _load_chunks(cls, index_path, chunk_idxs, num_embeddings,
                     load_index_with_mmap=False):
        num_embeddings += 512
        print_message("#> Loading codes and residuals (float16 mode)...")
        first = cls.load(index_path, chunk_idxs[0])
        res_dim   = first.residuals.shape[1]
        res_dtype = first.residuals.dtype
        codes     = torch.empty(num_embeddings, dtype=torch.int32)
        residuals = torch.empty(num_embeddings, res_dim, dtype=res_dtype)
        offset = 0
        for i, idx in enumerate(tqdm.tqdm(chunk_idxs)):
            chunk = first if i == 0 else cls.load(index_path, idx)
            end   = offset + chunk.codes.size(0)
            codes[offset:end]     = chunk.codes
            residuals[offset:end] = chunk.residuals
            offset = end
        return cls(codes, residuals)
    ResidualEmbeddings.load_chunks = _load_chunks

    def _decompress(self, compressed_embs):
        codes, residuals = compressed_embs.codes, compressed_embs.residuals
        D = []
        for codes_, res_ in zip(codes.split(1 << 15), residuals.split(1 << 15)):
            if self.use_gpu:
                codes_ = codes_.cuda()
                res_   = res_.cuda().half()
            else:
                res_ = res_.float()
            centroids_ = self.lookup_centroids(codes_, out_device=codes_.device)
            vec_ = centroids_ + res_
            if self.use_gpu:
                D.append(torch.nn.functional.normalize(vec_, p=2, dim=-1).half())
            else:
                D.append(torch.nn.functional.normalize(vec_.float(), p=2, dim=-1))
        return torch.cat(D)
    ResidualCodec.decompress = _decompress


# ============================================================
# 5. 검색 (subprocess 진입점)
# ============================================================
def _do_search(name, index_name, analog=False):
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    from colbert.data import Queries

    base        = os.path.join(BEIR_DIR, name)
    ranking_dir = os.path.join(COLBERT_DIR, f"experiments/beir_{name}/rankings")
    queries_path = os.path.join(base, "queries.test.tsv")
    ranking_path = os.path.join(ranking_dir, f"{index_name}.tsv")

    os.makedirs(ranking_dir, exist_ok=True)
    if os.path.exists(ranking_path):
        os.remove(ranking_path)

    if analog:
        _apply_analog_patches()

    torch.cuda.empty_cache()

    index_path = os.path.join(INDEX_BASE, f"beir_{name}", "indexes", index_name)

    with Run().context(RunConfig(nranks=1, experiment=f"beir_{name}")):
        config   = ColBERTConfig(nbits=2, doc_maxlen=220, query_maxlen=32, ndocs=NDOCS)
        searcher = Searcher(index=index_path, config=config)
        queries  = Queries(queries_path)
        ranking  = searcher.search_all(queries, k=SEARCH_K)

    with open(ranking_path, "w", encoding="utf-8") as f:
        for items in ranking.flat_ranking:
            f.write('\t'.join(str(x) for x in items) + '\n')
    log(f"  Saved → {ranking_path}")


# ============================================================
# 6. nDCG@10 계산 (graded 지원)
# ============================================================
def compute_ndcg10(ranking_path, qrels_path):
    qrels = {}
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid, score = str(parts[0]), int(parts[1]), int(parts[2])
            qrels.setdefault(qid, {})[pid] = score

    rankings = {}
    with open(ranking_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, pid, rank = str(parts[0]), int(parts[1]), int(parts[2])
            rankings.setdefault(qid, []).append((rank, pid))

    ndcg_sum = mrr_sum = r50_sum = r1k_sum = 0.0
    n = 0
    for qid, rel_dict in qrels.items():
        if qid not in rankings:
            continue
        if not any(s >= 1 for s in rel_dict.values()):
            continue
        n += 1
        ranked = [pid for _, pid in sorted(rankings[qid])]
        rel_set = {pid for pid, s in rel_dict.items() if s >= 1}

        # nDCG@10
        dcg = sum(rel_dict.get(pid, 0) / math.log2(r + 2)
                  for r, pid in enumerate(ranked[:10]))
        ideal_gains = sorted(rel_dict.values(), reverse=True)[:10]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains) if g > 0)
        ndcg_sum += dcg / idcg if idcg > 0 else 0.0

        # MRR@10
        for r, pid in enumerate(ranked[:10], 1):
            if pid in rel_set:
                mrr_sum += 1.0 / r
                break

        # R@50, R@1k
        if any(pid in rel_set for pid in ranked[:50]):
            r50_sum += 1.0
        if any(pid in rel_set for pid in ranked[:1000]):
            r1k_sum += 1.0

    return {
        "nDCG@10":    round(ndcg_sum / n, 4) if n else 0.0,
        "MRR@10":     round(mrr_sum  / n, 4) if n else 0.0,
        "R@50":       round(r50_sum  / n, 4) if n else 0.0,
        "R@1k":       round(r1k_sum  / n, 4) if n else 0.0,
        "num_queries": n,
    }


# ============================================================
# 7. 인덱스 삭제
# ============================================================
def cleanup_indexes(name):
    index_dir = os.path.join(INDEX_BASE, f"beir_{name}")
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        log(f"  Indexes deleted: {index_dir}")


# ============================================================
# main
# ============================================================
def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

    # 기존 결과 로드
    all_results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            all_results = json.load(f)

    for ds in DATASETS:
        name = ds["name"]

        # 이미 완료된 데이터셋 건너뜀 (elapsed_min 키 존재 = 완료)
        if name in all_results and "elapsed_min" in all_results[name]:
            log(f"  [{name.upper()}] already done, skipping.")
            continue

        log(f"\n{'='*60}")
        log(f"  Dataset: {name.upper()}")
        log(f"{'='*60}")

        base         = os.path.join(BEIR_DIR, name)
        ranking_dir  = os.path.join(COLBERT_DIR, f"experiments/beir_{name}/rankings")
        qrels_path   = os.path.join(base, "qrels.test.int.tsv")
        analog_name  = f"{name}.analog"
        bit2_name    = f"{name}.2bit"

        t0 = time.time()

        # 1. 다운로드
        log("[1] Downloading...")
        download(ds)

        # 2. make_fair
        log("[2] make_fair...")
        make_fair(ds)

        # 3. 인덱싱
        log("[3] Building indexes...")
        build_indexes(ds)

        # 4. float16 검색
        log("[4] Searching float16...")
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "search_analog", name],
            cwd=COLBERT_DIR
        ).returncode
        analog_ranking = os.path.join(ranking_dir, f"{analog_name}.tsv")
        f16_metrics = compute_ndcg10(analog_ranking, qrels_path) if rc == 0 else {}
        log(f"  float16  nDCG@10={f16_metrics.get('nDCG@10', 'ERR')}")

        # 5. 2-bit 검색
        log("[5] Searching 2-bit...")
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "search_2bit", name],
            cwd=COLBERT_DIR
        ).returncode
        bit2_ranking = os.path.join(ranking_dir, f"{bit2_name}.tsv")
        b2_metrics = compute_ndcg10(bit2_ranking, qrels_path) if rc == 0 else {}
        log(f"  2-bit    nDCG@10={b2_metrics.get('nDCG@10', 'ERR')}")

        # 6. 인덱스 삭제
        log("[6] Cleaning up indexes...")
        cleanup_indexes(name)

        elapsed = time.time() - t0
        all_results[name] = {
            "float16": f16_metrics,
            "2bit":    b2_metrics,
            "elapsed_min": round(elapsed / 60, 1),
        }

        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        # 완료 결과 출력
        print(f"\n{'='*62}")
        print(f"  [{name.upper()}] 완료 ({elapsed/60:.1f}min)  n={f16_metrics.get('num_queries','?')}")
        print(f"{'='*62}")
        print(f"  {'Metric':<12} {'float16':>10} {'2-bit':>10} {'Delta':>10}")
        print(f"  {'-'*50}")
        for metric in ["nDCG@10", "MRR@10", "R@50", "R@1k"]:
            f16v = f16_metrics.get(metric)
            b2v  = b2_metrics.get(metric)
            if isinstance(f16v, float) and isinstance(b2v, float):
                delta = f16v - b2v
                sign  = "+" if delta >= 0 else ""
                print(f"  {metric:<12} {f16v:>10.4f} {b2v:>10.4f} {sign}{delta:>9.4f}")
        print(f"{'='*62}\n")

    # 최종 결과 테이블
    for metric in ["nDCG@10", "MRR@10", "R@50", "R@1k"]:
        print(f"\n{'='*65}")
        print(f"  BEIR Float16 vs 2-bit  [{metric}]")
        print(f"{'='*65}")
        print(f"  {'Dataset':<16} {'float16':>10} {'2-bit':>10} {'Delta':>10}")
        print(f"  {'-'*52}")
        for ds_name, res in all_results.items():
            f16 = res.get("float16", {}).get(metric)
            b2  = res.get("2bit",    {}).get(metric)
            if isinstance(f16, float) and isinstance(b2, float):
                delta = f16 - b2
                sign  = "+" if delta >= 0 else ""
                print(f"  {ds_name:<16} {f16:>10.4f} {b2:>10.4f} {sign}{delta:>9.4f}")
        print(f"{'='*65}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd  = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else None
        if cmd == "encode" and name:
            _do_encode(name)
        elif cmd == "search_analog" and name:
            analog_name = f"{name}.analog"
            ranking_dir = os.path.join(COLBERT_DIR, f"experiments/beir_{name}/rankings")
            _do_search(name, analog_name, analog=True)
        elif cmd == "search_2bit" and name:
            bit2_name   = f"{name}.2bit"
            ranking_dir = os.path.join(COLBERT_DIR, f"experiments/beir_{name}/rankings")
            _do_search(name, bit2_name, analog=False)
        else:
            print(f"Unknown command: {cmd}")
            sys.exit(1)
    else:
        main()
