# ColBERTv2 Quantized vs Analog Residual 실험 진행 기록

## 실험 목적
ColBERTv2의 잔차(residual) 저장 방식을 비교하여 아날로그 메모리 소자의 가치를 정량적으로 검증한다.

- **Quantized (디지털)**: 잔차를 2-bit로 양자화 → 128차원 × 2bit = **256 bit-cells/벡터**
- **Analog (아날로그 소자)**: 잔차를 연속값으로 저장 → 128차원 × 1소자 = **128 devices/벡터**
- **핵심 질문**: 소자 수 50% 절약하면서 검색 품질은 어떻게 변하는가?

## 실험 환경
- **하드웨어**: RTX 4060 8GB VRAM, 16GB RAM, Windows 11 (WSL2)
- **모델**: colbert-ir/colbertv2.0 (bert-base-uncased, 128-dim embeddings)
- **데이터셋**: MS MARCO Passage Ranking (Dev Small, 6,980 queries)
- **컬렉션**: 8.8M 중 500k 서브셋 (qrels 관련 7,433개 전체 포함 + 랜덤 492,567개)
- **Python**: 3.12, faiss-gpu-cu12, PyTorch with CUDA

## 전체 플랜 및 진행 상황

### 메인 파이프라인

| Step | 내용 | 상태 |
|------|------|------|
| Step 1 | MS MARCO 데이터 다운로드 (collection, queries, qrels) | **완료** (이전 세션) |
| Step 2 | Full 8.8M quantized (2-bit) 인덱싱 | **완료** (이전 세션) |
| Step 3 | Full 8.8M 검색 (quantized) | **SKIP** (RAM 부족, ~20GB 필요) |
| Step 4 | 500k 서브셋 생성 (qrels 관련 7,433 + 랜덤 492,567) | **완료** |
| Step 5A | 서브셋 quantized (2-bit) 인덱싱 + 검색 + 평가 | **완료** (MRR@10=0.7750) |
| **Step 5B** | **서브셋 analog (float16) 인덱싱 + 검색 + 평가** | **진행 중** ← 현재 |
| Step 5C | 아날로그 소자 노이즈 시뮬레이션 (σ=0.01~0.2) | 대기 (5B 완료 후 자동 실행) |
| Step 6 | Quantized vs Analog 비교 테이블 출력 | 대기 |
| Step 7 | `experiment_summary.md` 요약 파일 생성 | 대기 |

### Step 5B 세부 진행

| 단계 | 내용 | 상태 |
|------|------|------|
| 코드 패치 | residual.py + residual_embeddings.py 4곳 수정 | **완료** |
| 5B-1a | 샘플 인코딩 (123,936개) | **완료** |
| 5B-1b | k-means 클러스터링 (65,536 centroids, GPU) | **완료** (~11분) |
| **5B-1c** | **전체 인코딩 + float16 압축 (20 chunks)** | **진행 중 (17/20)** ← 현재 |
| 5B-1d | IVF 생성 + 메타데이터 저장 | 대기 (~2분) |
| 5B-2 | 검색 k=1000 (6,980 queries) | 대기 (~5분) |
| 코드 복원 | 원본 파일 복구 (.bak → 원본) | 대기 |

### Step 5C 노이즈 시뮬레이션 세부

| 단계 | 내용 | 상태 |
|------|------|------|
| 5C-0 | 원본 residual 백업 + 코드 패치 | 대기 |
| 5C-1 | σ=0.01 노이즈 추가 → 검색 → 평가 | 대기 (~5분) |
| 5C-2 | σ=0.05 노이즈 추가 → 검색 → 평가 | 대기 (~5분) |
| 5C-3 | σ=0.1 노이즈 추가 → 검색 → 평가 | 대기 (~5분) |
| 5C-4 | σ=0.2 노이즈 추가 → 검색 → 평가 | 대기 (~5분) |
| 5C-5 | 원본 복원 + 결과 저장 + 코드 복원 | 대기 |

### OOM 시 fallback 플랜
5B-2 검색에서 OOM 발생 시 → 200k 서브셋으로 줄여서 5A + 5B 전체 재실행

## 현재까지의 결과

### Step 5A: Quantized (2-bit) — 완료
| Metric | 값 |
|--------|-----|
| MRR@10 | 0.7750 |
| R@50 | 0.9803 |
| R@1k | 0.9952 |

*500k 서브셋이므로 논문 수치(MRR@10=0.397)보다 높음. 관련 문서가 모두 포함되어 있기 때문. Delta 비교가 핵심.*

### 인덱스 정보
- 총 임베딩 수: 33,772,753 (500k 패시지 × 평균 ~67.5 토큰)
- Quantized 소자 수: 33,772,753 × 256 = **8,645,824,768 bit-cells**
- Analog 소자 수: 33,772,753 × 128 = **4,322,912,384 devices** (50% 절약)

## 코드 패치 상세 (Analog 모드)

### 패치 대상 파일 및 변경 내용

**1. `colbert/indexing/codecs/residual.py` — compress()**
```python
# Before (quantized):
residuals.append(self.binarize(residuals_).cpu())
# After (analog):
residuals.append(residuals_.half().cpu())  # float16 저장
```

**2. `colbert/indexing/codecs/residual.py` — decompress()**
```python
# Before: CUDA bit unpacking (reversed_bit_map, decompression_lookup_table 등)
# After: 단순 centroid + residual 덧셈
centroids_ = self.lookup_centroids(codes_, out_device=codes_.device)
centroids_ = centroids_ + residuals_
```

**3. `colbert/indexing/codecs/residual_embeddings.py` — __init__()**
```python
# Before:
assert residuals.dtype == torch.uint8
# After: 제거 (float16 허용)
```

**4. `colbert/indexing/codecs/residual_embeddings.py` — load_chunks()**
```python
# Before:
residuals = torch.empty(num_embeddings, dim // 8 * nbits, dtype=torch.uint8)
# After:
residuals = torch.empty(num_embeddings, dim, dtype=torch.float16)
```

## 노이즈 시뮬레이션 방법

아날로그 소자는 완벽한 연속값을 저장하지 못하고 노이즈가 존재한다.
이를 시뮬레이션하기 위해 저장된 float16 residual에 가우시안 노이즈를 추가한다.

```
r' = r + N(0, σ)    (σ가 클수록 소자 노이즈가 심함)
```

σ = [0.01, 0.05, 0.1, 0.2] 4단계로 테스트하여 MRR@10 하락폭을 측정한다.
하락폭이 작을수록 아날로그 소자의 노이즈에 강건한 것이다.

## 파일 구조

| 파일 | 용도 |
|------|------|
| `run_overnight.py` | 전체 실험 파이프라인 (서브셋 생성, 인덱싱, 검색, 평가, 비교) |
| `run_remaining.py` | 5B + 노이즈 시뮬레이션 + Step 6 + 요약 생성 |
| `experiment_summary.md` | 최종 결과 요약 (실험 완료 후 자동 생성) |
| `experiment_progress.md` | 이 파일 — 진행 기록 |
| `experiments/msmarco/overnight_results.json` | 수치 결과 (JSON) |
| `experiments/msmarco/indexes/subset.quantized/` | Quantized 인덱스 |
| `experiments/msmarco/indexes/subset.analog/` | Analog 인덱스 |
| `experiments/msmarco/rankings/` | 검색 결과 랭킹 파일 |
| `data/msmarco/subset/` | 500k 서브셋 컬렉션 + 조정된 qrels |

## 트러블슈팅 기록

| 문제 | 원인 | 해결 |
|------|------|------|
| Step 3 OOM | Full 8.8M 검색에 ~20GB RAM 필요 | Step 3 스킵, 500k 서브셋으로 대체 |
| k-means CPU로 50분+ | faiss-cpu만 설치됨 | faiss-gpu-cu12 설치, faiss-cpu 제거 → ~11분 |
| WSL 백그라운드 실행 실패 | `wsl bash -c "... &"` 작동 안 함 | Claude의 `run_in_background` 기능 사용 |
| 로그 파일 미발견 | WSL `~/ColBERT`와 Windows 경로 불일치 | `/mnt/c/Users/dmsdu/ColBERT` 경로 통일 |

## 예상 완료 시간
- 5B 완료: ~15:10
- 노이즈 시뮬레이션 완료: ~15:35
- 최종 요약 파일 생성: ~15:35
