# ColBERTv2 Table 5 재현 — 진행상황 요약
작성일: 2026-03-07

---

## 목표
ColBERTv2 논문 Table 5 재현 — **2-bit quantized residual vs analog(float16) residual** 정확도 비교

벤치마크:
1. **BEIR** (13개 데이터셋) — nDCG@10
2. **LoTTE** (5 도메인 × 2 쿼리타입) — Success@5
3. **Wikipedia Open QA** (NQ, TriviaQA, SQuAD) — Success@5
4. **노이즈 시뮬레이션** (analog residuals에 σ=[0.01, 0.05, 0.1, 0.2] 가우시안 노이즈)

---

## 환경
- **이 컴퓨터**: WSL2, RTX 4060 8GB VRAM, 16GB RAM
- **ColBERT 체크포인트**: `colbert-ir/colbertv2.0` (HuggingFace)
- **스크립트**: `run_table5.py`
- **결과 파일**: `/home/eunybe/ColBERT/experiments/table5/table5_results.json`
- **로그**: `/mnt/c/Users/dmsdu/ColBERT/table5.log`

---

## 완료된 BEIR 결과 (nDCG@10)

| Dataset | Docs | Quantized | Analog | Δ | 완료 시각 |
|---------|------|-----------|--------|---|----------|
| NFCorpus | 3.6K | **0.332** | **0.337** | +0.005 | 18:22 |
| SciFact | 5.2K | **0.663** | **0.667** | +0.004 | 18:28 |
| ArguAna | 8.7K | **0.336** | **0.333** | -0.003 | 18:34 |
| FiQA | 57.6K | **0.352** | **0.354** | +0.002 | 19:49 |
| TREC-COVID | 171K | **0.722** | — (스킵) | — | 20:22 |
| Quora | 200K subset | **0.907** | **0.909** | +0.002 | 21:21 |

> TREC-COVID analog 스킵 이유: 23.8M embeddings > MAX_ANALOG_EMBEDDINGS(15M) 한도 초과 (OOM 방지)

---

## 미완료 항목

| Dataset | 상태 | 원인 |
|---------|------|------|
| SCIDOCS | ❌ 실패 | TSV 줄바꿈 버그 (수정 완료) |
| Touché (webis-touche2020) | ❌ 실패 | HuggingFace 이름 오류 (`BeIR/webis-touche2020-v2` 없음) |
| TREC-COVID analog | ⚠️ 스킵 | 23.8M embeddings > 15M 한도 |
| NQ | 🔄 인덱싱 완료, 결과 미저장 | 실행 중단됨 |
| DBPedia | ⬜ 미실행 | |
| Climate-FEVER | ⬜ 미실행 | |
| FEVER | ⬜ 미실행 | |
| HotpotQA | ⬜ 미실행 | |
| **LoTTE** | ❌ 미구현 | 스크립트에 없음 |
| **Wikipedia Open QA** | ❌ 미구현 | 스크립트에 없음 |
| **노이즈 시뮬레이션** | ❌ 미구현 | 스크립트에 없음 |

---

## 핵심 코드: `run_table5.py`

### 주요 설정값
```python
CHECKPOINT = "colbert-ir/colbertv2.0"
NBITS = 2
DOC_MAXLEN = 220
QUERY_MAXLEN = 32
INDEX_BSIZE = 32
SEARCH_K = 100
SUBSET_THRESHOLD = 200_000      # 초과 시 subset 생성
MAX_ANALOG_EMBEDDINGS = 15_000_000  # 초과 시 analog 스킵
```

### 핵심 함수 구조
- `download_and_convert_beir()` — HuggingFace로 다운로드, TSV 변환
- `_do_index()` — `IndexSaver.save_chunk` monkey-patch로 raw embeddings 저장하며 인덱싱
- `_build_analog_index()` — raw embeddings → float16 residuals로 analog 인덱스 생성
- `_do_search_analog()` — decompress 패치 적용 후 analog 검색
- `run_dataset()` — 데이터셋당 전체 파이프라인 실행
- `main()` — 이미 완료된 데이터셋 스킵 후 루프 실행

### 중요한 버그 수정 사항 (이미 적용됨)
1. **`avoid_fork_if_possible=True`** — RunConfig와 ColBERTConfig **양쪽 모두** 설정 필수. 한쪽만 하면 monkey-patch가 child process에 상속 안 됨
2. **TSV 줄바꿈 처리** — 문서 텍스트의 `\n`, `\r`, `\t` 제거:
   ```python
   combined = combined.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
   ```
3. **HuggingFace 다운로드** — BEIR 서버(독일 대학)가 5KB/s로 느림 → `datasets` 라이브러리로 교체
4. **qrels 로딩** — `datasets` 라이브러리 호환성 문제 → `urllib.request`로 TSV 직접 다운로드

---

## 다른 컴퓨터에서 이어서 시작하는 법

### 1. 환경 준비
```bash
git clone https://github.com/stanford-futuredata/ColBERT.git
cd ColBERT
pip install -e .
pip install beir pytrec_eval-terrier datasets
```

### 2. 파일 복사
- `run_table5.py` — 메인 스크립트
- `table5_results.json` — 완료된 결과 (자동 스킵용), 경로: `/home/{user}/ColBERT/experiments/table5/table5_results.json`

### 3. 실행
```bash
cd ~/ColBERT
source ~/colbert-env/bin/activate
python run_table5.py > table5.log 2>&1 &
tail -f table5.log
```
→ 이미 완료된 6개 데이터셋은 자동 스킵됨

### 4. 수정 필요한 버그 (이어서 할 때 먼저 수정)

**Bug 1 — Touché HuggingFace 이름 오류**
`run_table5.py`의 `hf_name_map`에서:
```python
# 현재 (오류)
hf_name_map = {"webis-touche2020": "webis-touche2020-v2"}
# 수정 필요 — 올바른 HF 이름 확인 후 변경 또는 삭제
hf_name_map = {}  # 임시: 원본 이름 그대로 사용해보기
```

**Bug 2 — TREC-COVID analog 스킵**
analog도 하려면 `MAX_ANALOG_EMBEDDINGS` 올리거나 (RAM 여유 있을 때):
```python
MAX_ANALOG_EMBEDDINGS = 30_000_000  # 16GB RAM 기준 약 7.7GB
```

**Bug 3 — SCIDOCS 재실행**
기존 손상된 파일 삭제 후 재실행하면 자동으로 재다운로드:
```bash
rm -f ~/ColBERT/data/table5/beir/scidocs/collection.tsv
```

---

## 다음 단계 (구현 필요)

### LoTTE 벤치마크
- 데이터: https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz
- 도메인: writing, recreation, science, technology, lifestyle (각 search/forum 쿼리)
- 메트릭: Success@5

### Wikipedia Open QA
- 데이터: DPR Wikipedia 덤프 사용
- NQ, TriviaQA, SQuAD — 각 21M 문서 → 200K subset
- 메트릭: Success@5

### 노이즈 시뮬레이션
analog residuals에 가우시안 노이즈 추가:
```python
for sigma in [0.01, 0.05, 0.1, 0.2]:
    noisy_residuals = residuals + torch.randn_like(residuals) * sigma
    # 검색 후 nDCG@10 평가
```
