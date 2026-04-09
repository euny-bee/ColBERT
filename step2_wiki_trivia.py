"""
Step 2: Build Wikipedia corpus for TriviaQA experiment (Option C)
- All 879,481 passages from psgs_w100.tsv (partial download)
- + 66,881 gold passages from trivia-dev.json (not in file)
- Total: ~946K passages, all 8,837 queries evaluable
"""

import csv
import io
import json
import os

TRIVIA_DEV  = "D:/DPR/downloads/data/retriever/trivia-dev.json"
WIKI_TSV    = "D:/DPR/downloads/data/wikipedia_split/psgs_w100.tsv"
OUT_DIR     = "D:/beir/wiki_trivia_500k"
MAX_PID     = 879481  # last readable passage ID in our file

os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: collect gold passages from JSON ───────────────────────────────────
print("Loading TriviaQA dev...")
with open(TRIVIA_DEV, encoding="utf-8") as f:
    trivia = json.load(f)

# gold passages not in file (psg_id > MAX_PID) — keep unique
missing_gold = {}   # str(psg_id) -> (title, text)
all_gold_ids = set()
for q in trivia:
    for ctx in q.get("positive_ctxs", []):
        pid = str(ctx["psg_id"])
        all_gold_ids.add(pid)
        if int(ctx["psg_id"]) > MAX_PID:
            missing_gold[pid] = (ctx["title"], ctx["text"])

print(f"  Queries  : {len(trivia):,}")
print(f"  Gold IDs : {len(all_gold_ids):,}")
print(f"  Missing from file: {len(missing_gold):,}")

# ── Step 2: write collection.tsv ──────────────────────────────────────────────
collection_path = os.path.join(OUT_DIR, "collection.tsv")
print(f"\nWriting collection.tsv ...")

new_pid = 0
orig_to_new = {}   # original psg_id (str) -> new 0-based pid

with open(collection_path, "w", encoding="utf-8") as out:
    # (a) passages from file
    print("  Reading psgs_w100.tsv (csv-quoted) ...")
    with open(WIKI_TSV, "rb") as binf:
        text_f = io.TextIOWrapper(binf, encoding="utf-8", errors="replace")
        reader = csv.reader(text_f, delimiter="\t")
        next(reader)  # skip header: id / text / title
        try:
            for row in reader:
                if len(row) < 3:
                    continue
                pid, text, title = row[0], row[1], row[2]
                passage = f"{title}. {text}".replace("\t", " ").replace("\n", " ")
                out.write(f"{new_pid}\t{passage}\n")
                orig_to_new[pid] = new_pid
                new_pid += 1
                if new_pid % 200000 == 0:
                    print(f"    {new_pid:,} passages written")
        except (PermissionError, OSError):
            pass  # file truncated at 879,481 rows
    print(f"  File passages: {new_pid:,}")

    # (b) missing gold passages from JSON
    for pid, (title, text) in missing_gold.items():
        passage = f"{title}. {text}".replace("\t", " ").replace("\n", " ")
        out.write(f"{new_pid}\t{passage}\n")
        orig_to_new[pid] = new_pid
        new_pid += 1

print(f"  Total passages: {new_pid:,} -> {collection_path}")
print(f"  File size: {os.path.getsize(collection_path)/1e6:.0f} MB")

# ── Step 3: write queries.tsv ─────────────────────────────────────────────────
queries_path = os.path.join(OUT_DIR, "queries.tsv")
with open(queries_path, "w", encoding="utf-8") as out:
    for i, q in enumerate(trivia):
        out.write(f"{i}\t{q['question']}\n")
print(f"Queries: {len(trivia):,} -> {queries_path}")

# ── Step 4: write answers.json ────────────────────────────────────────────────
answers_path = os.path.join(OUT_DIR, "answers.json")
answers = {str(i): q["answers"] for i, q in enumerate(trivia)}
with open(answers_path, "w", encoding="utf-8") as out:
    json.dump(answers, out, ensure_ascii=False)
print(f"Answers -> {answers_path}")

# ── Step 5: write pid map (gold coverage check) ───────────────────────────────
covered = sum(1 for pid in all_gold_ids if pid in orig_to_new)
print(f"\nGold coverage: {covered:,}/{len(all_gold_ids):,} ({100*covered/len(all_gold_ids):.1f}%)")

pid_map_path = os.path.join(OUT_DIR, "pid_map.json")
with open(pid_map_path, "w", encoding="utf-8") as out:
    json.dump(orig_to_new, out)
print(f"PID map -> {pid_map_path}")

print("\nDone!")
