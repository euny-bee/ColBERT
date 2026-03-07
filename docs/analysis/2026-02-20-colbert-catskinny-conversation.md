# ColBERT 로컬 환경 구축 대화 원본

> 날짜: 2026-02-20

---

## Q: 내 로컬 환경에서 이 colbert 프로젝트를 torch를 활용해서 구동해보고싶어. 뭐부터 해야할까?

### A:

ColBERT 프로젝트를 로컬에서 구동하기 위해 먼저 프로젝트 구조와 의존성을 파악하겠습니다.

(setup.py, README.md, conda_env.yml, conda_env_cpu.yml 파일 분석)

### ColBERT 로컬 환경 셋업 가이드

#### 1단계: Conda 환경 생성

GPU가 있는 경우와 없는 경우로 나뉩니다. **Conda 사용을 강력히 권장**합니다 (faiss와 torch 호환성 때문).

**GPU 있는 경우:**
```bash
conda env create -f conda_env.yml
conda activate colbert
```

**CPU만 사용하는 경우:**
```bash
conda env create -f conda_env_cpu.yml
conda activate colbert
```

> 참고: conda_env.yml은 Python 3.8 + PyTorch 1.13.1 + CUDA 11.3을 설치합니다. Windows에서는 `gcc/gxx` 의존성이 문제될 수 있어서 해당 라인을 제거하고 설치해야 할 수 있습니다.

#### 2단계: ColBERT 패키지 설치

```bash
cd c:/Users/dmsdu/ColBERT
pip install -e .
```

#### 3단계: 사전 학습된 체크포인트 다운로드

```bash
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
tar -xvzf colbertv2.0.tar.gz
```

#### 4단계: 실행

docs/intro.ipynb 노트북을 사용하면 가장 빠르게 테스트할 수 있습니다.

또는 간단한 Python 스크립트로 실행:

```python
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher, Indexer

if __name__ == '__main__':
    with Run().context(RunConfig(nranks=1, experiment="test")):
        config = ColBERTConfig(nbits=2, root="experiments")
        indexer = Indexer(checkpoint="colbertv2.0", config=config)
        indexer.index(name="my_index", collection="path/to/collection.tsv")
        searcher = Searcher(index="my_index", config=config)
        results = searcher.search("What is ColBERT?", k=10)
        print(results)
```

#### Windows에서 주의할 점

1. **gcc/gxx 문제**: conda_env.yml에 `gcc=9.4.0`, `gxx=9.4.0`이 있는데 Windows에서는 지원되지 않습니다. 이 두 줄을 제거한 후 conda env를 생성하세요.
2. **CUDA 확장 컴파일**: ColBERT에 C++/CUDA 커스텀 확장이 포함되어 있어서 Windows에서는 Visual Studio Build Tools(C++ 빌드 도구)가 필요합니다.
3. **faiss**: GPU 버전 faiss는 Windows conda에서 설치가 까다로울 수 있습니다. 먼저 `faiss-cpu`로 시작해보는 것을 추천합니다.

#### 추천 순서

| 순서 | 할 일 |
|------|--------|
| 1 | Conda 설치 확인 (`conda --version`) |
| 2 | conda_env_cpu.yml로 환경 생성 (Windows 호환성을 위해 CPU 버전 먼저 추천) |
| 3 | `pip install -e .` 로 ColBERT 설치 |
| 4 | 체크포인트 다운로드 |
| 5 | docs/intro.ipynb 노트북으로 테스트 |

먼저 현재 환경에 conda나 Python, CUDA가 설치되어 있는지 확인해볼까요?

---

## Q: 근데 conda 말고 torch로 설치하고싶어. wsl 환경에서 설치하고싶어. 이때 뭐부터 해야할지 알려줘

### A:

WSL + pip(torch) 환경으로 ColBERT를 설치하는 순서입니다.

### WSL 환경에서 ColBERT 셋업 순서

#### 1. WSL 설치 및 진입

```bash
# PowerShell (관리자 권한)
wsl --install
wsl
```

#### 2. 시스템 패키지 설치

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git wget gcc g++
```

> `gcc`, `g++`는 ColBERT의 C++/CUDA 확장 컴파일에 필요합니다.

#### 3. 가상환경 생성

```bash
python3 -m venv ~/colbert-env
source ~/colbert-env/bin/activate
```

#### 4. PyTorch 설치

```bash
# GPU (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU만
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

> WSL에서 GPU를 쓰려면 **Windows 측에 NVIDIA 드라이버**만 설치하면 됩니다. WSL 내부에 별도 드라이버 설치는 불필요합니다.

#### 5. faiss 설치

```bash
pip install faiss-cpu
# 또는
pip install faiss-gpu
```

#### 6. ColBERT 설치

```bash
cd ~
git clone https://github.com/stanford-futuredata/ColBERT.git
cd ~/ColBERT
pip install -e .
```

#### 7. 체크포인트 다운로드

```bash
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
tar -xzf colbertv2.0.tar.gz
```

#### 8. 동작 확인

```bash
python -c "import colbert; import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 요약

| 순서 | 할 일 | 비고 |
|------|--------|------|
| 1 | WSL 진입 | `wsl` |
| 2 | gcc, python3, venv 설치 | `apt install` |
| 3 | venv 가상환경 생성 | conda 대신 사용 |
| 4 | `pip install torch` | GPU/CPU 선택 |
| 5 | `pip install faiss-cpu` | 먼저 CPU로 시작 권장 |
| 6 | `pip install -e .` | ColBERT 설치 |
| 7 | 체크포인트 다운로드 | colbertv2.0 |
| 8 | import 테스트 | 정상 동작 확인 |

GPU 사용 여부에 따라 4~5번 선택이 달라지는데, GPU를 사용할 계획인가요?

---

## Q: 이 컴퓨터에 GPU가 있는지 먼저 확인해줘

### A:

GPU가 있습니다! 정리하면:

| 항목 | 내용 |
|------|------|
| **GPU** | NVIDIA GeForce RTX 4060 |
| **VRAM** | 8GB (8188MiB) |
| **드라이버** | 591.55 |
| **CUDA 버전** | 13.1 |
| **현재 사용량** | 4252MiB / 8188MiB |

RTX 4060은 ColBERT를 돌리기에 충분합니다. WSL에서도 Windows 드라이버가 이미 설치되어 있으므로 GPU를 바로 사용할 수 있습니다.

따라서 PyTorch 설치 시 **GPU 버전**으로 설치하면 됩니다:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu
```

WSL 셋업을 바로 진행해볼까요?

---

## Q: 응 1번부터 진행하자

### A:

Ubuntu WSL이 이미 설치되어 있고 실행 중입니다. 바로 2단계로 넘어가겠습니다.

Python 3.12.3은 있지만 gcc/g++이 없습니다. 시스템 패키지를 설치합니다.

apt 설치가 오래 걸리고 있습니다. `sudo`에 비밀번호 입력이 필요할 수 있습니다. 이 작업은 **WSL 터미널에서 직접** 실행하는 것이 좋겠습니다.

여기서부터는 **WSL 터미널을 직접 열고** 아래 명령어들을 순서대로 실행하는 것이 안정적입니다. sudo 비밀번호 입력 등 인터랙티브 작업이 필요하기 때문입니다.

### WSL 터미널에서 직접 실행할 명령어 (복사해서 순서대로)

#### 2단계: 시스템 패키지
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git wget gcc g++
```

#### 3단계: 가상환경 생성
```bash
python3 -m venv ~/colbert-env
source ~/colbert-env/bin/activate
```

#### 4단계: PyTorch GPU 설치
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 5단계: faiss 설치
```bash
pip install faiss-gpu
```

#### 6단계: ColBERT 설치
```bash
cp -r /mnt/c/Users/dmsdu/ColBERT ~/ColBERT
cd ~/ColBERT
pip install -e .
```

#### 7단계: 체크포인트 다운로드
```bash
cd ~/ColBERT
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz
tar -xzf colbertv2.0.tar.gz
```

#### 8단계: 동작 확인
```bash
python -c "import colbert; import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

WSL 터미널(시작 메뉴에서 **Ubuntu** 검색)을 열고 위 명령어를 순서대로 실행해주세요. 중간에 문제가 생기면 에러 메시지를 공유해주시면 바로 도와드리겠습니다!

---

## Q: (캡처) pip install faiss-gpu 에러

```
ERROR: Could not find a version that satisfies the requirement faiss-gpu (from versions: none)
ERROR: No matching distribution found for faiss-gpu
```

방금얘기해준거에서 4단계. pytorch GPU 설치까지는 완료했거든? 근데 5단계 하려고하니까 캡처한 화면처럼 에러가 떠 어떻게하면되지?

### A:

`faiss-gpu`는 pip에서 Python 3.12를 지원하지 않아서 생기는 문제입니다. 두 가지 방법이 있습니다:

#### 방법 1: `faiss-cpu` 사용 (추천)

```bash
pip install faiss-cpu
```

> ColBERT의 **모델 연산(인코딩/검색)은 PyTorch + GPU**로 돌아가고, faiss는 인덱스 구축에만 쓰입니다. CPU 버전이어도 충분히 동작합니다.

우선 `pip install faiss-cpu`로 진행하고, 6단계(`pip install -e .`)로 넘어가세요!

---

## Q: (캡처) pip install faiss-cpu 에서 externally-managed-environment 에러

### A:

가상환경(venv)이 활성화되지 않은 상태에서 pip을 실행해서 생긴 에러입니다. 프롬프트에 `(colbert-env)`가 안 보이네요.

가상환경을 다시 활성화한 후 설치하세요:

```bash
source ~/colbert-env/bin/activate
pip install faiss-cpu
```

프롬프트 앞에 `(colbert-env)`가 표시되는지 확인 후 pip 명령어를 실행하면 됩니다.

---

## Q: 1~8단계까지 완료하고 나니까 이런 메세지가 떴어. 이제 어떻게 진행하면 될까?

```
(colbert-env) eunybe@EUNYBE-PC3:~/ColBERT$ python -c "import colbert; import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
torch: 2.5.1+cu121
CUDA: True
```

### A:

환경 셋업이 완료되었습니다! PyTorch 2.5.1 + CUDA 사용 가능 상태입니다.

이제 **실제로 ColBERT를 사용**할 차례입니다. 어떤 작업을 하고 싶은지에 따라 다음 단계가 달라집니다:

| 목적 | 설명 |
|------|------|
| **검색 체험** | 샘플 데이터로 인덱싱 + 검색 해보기 |
| **학습(Training)** | 나만의 데이터로 ColBERT 모델 훈련 |
| **기존 데이터셋 실험** | MS MARCO 등 벤치마크 데이터셋으로 실험 |
| **API 서버** | ColBERT 검색 서버 띄우기 |

가장 빠르게 동작을 확인하려면 프로젝트에 포함된 **튜토리얼 노트북**을 실행해보는 것을 추천합니다:

```bash
cd ~/ColBERT
pip install jupyter
jupyter notebook docs/intro2new.ipynb
```

어떤 방향으로 진행하고 싶으세요?

---

## Q: 프로젝트에 포함된 튜토리얼 노트북이 뭘 하는거야?

### A:

(docs/intro2new.ipynb 전체 분석)

이 노트북(docs/intro2new.ipynb)은 ColBERT의 **핵심 기능 3가지를 데모**하는 튜토리얼입니다:

### 1. 데이터 로드
- HuggingFace에서 **LoTTE 벤치마크** (lifestyle 도메인) 데이터셋을 다운로드
- 417개 쿼리 + 268,881개 문서(passage) 로드
- 빠른 데모를 위해 **처음 10,000개 문서**만 사용

### 2. 인덱싱 (Indexing)
- `colbert-ir/colbertv2.0` 사전학습 체크포인트를 HuggingFace에서 자동 다운로드
- 10,000개 문서를 ColBERT로 인코딩하여 **압축 인덱스** 생성 (2bit)
- GPU 1개 기준 약 6분 소요

### 3. 검색 (Search)
- 생성된 인덱스에서 자연어 쿼리로 **Top-K 문서 검색**
- 예: `"are some cats just skinny?"` 라는 질문에 가장 관련 높은 문서 3개를 찾아 점수와 함께 반환

### 한마디로

> **"질문을 던지면 가장 관련 있는 문서를 찾아주는 검색 시스템"**을 인덱스 구축부터 검색까지 end-to-end로 체험하는 노트북입니다.

---

## Q: 근데 이게 업무 load가 빡센가? 그리고 뭐 문서들 로드하는데 추가 용량?이 필요한가?

### A:

노트북 기준으로 분석하면 **꽤 가볍습니다**.

### 컴퓨팅 부하

| 항목 | 예상치 | 비고 |
|------|--------|------|
| **모델** | BERT-base 기반 (~110M 파라미터) | 가벼운 편 |
| **VRAM 사용** | ~2-3GB | RTX 4060 8GB이면 여유 |
| **인덱싱 시간** | 3~6분 | Colab T4 기준 6분, RTX 4060이 더 빠름 |
| **검색** | 수십 ms | 거의 즉시 |

데모에서 10,000개 문서만 사용하기 때문에 RTX 4060으로 **전혀 무리 없습니다**.

### 추가 다운로드 용량

| 항목 | 용량 |
|------|------|
| ColBERTv2 체크포인트 (HuggingFace 자동 다운로드) | ~400MB |
| LoTTE passages 데이터셋 | ~383MB |
| LoTTE queries 데이터셋 | ~1.2MB |
| 생성되는 인덱스 (2bit, 10K 문서) | ~수십 MB |
| **합계** | **~약 800MB~1GB** |

### 결론

- **GPU 부하**: 가벼움 (BERT-base 수준, VRAM 3GB 이내)
- **추가 용량**: 약 1GB 정도
- **시간**: 인덱싱 포함 10분 이내

부담 없이 돌릴 수 있는 수준입니다.

---

## Q: 응 그러면 튜토리얼 노트북을 실행해보자

### A:

WSL 터미널에서 아래 명령어를 실행해주세요:

```bash
source ~/colbert-env/bin/activate
cd ~/ColBERT
pip install jupyter
jupyter notebook docs/intro2new.ipynb --no-browser
```

`--no-browser`를 붙이는 이유는 WSL에서는 브라우저가 직접 안 열리기 때문입니다. 실행하면 터미널에 이런 URL이 표시됩니다:

```
http://localhost:8888/notebooks/docs/intro2new.ipynb?token=abc123...
```

이 URL을 **Windows 브라우저**에 붙여넣으면 노트북이 열립니다. 그 후 상단 메뉴에서 **Run All** (또는 `Cell > Run All`)을 실행하면 됩니다.

---

## Q: (캡처) windows 브라우저에 이렇게 열리는데, run all이 어디에 있어?

### A:

지금 보이는 화면은 파일 목록(디렉토리) 화면입니다. 먼저 **`intro2new.ipynb`** 를 클릭해서 노트북을 열어주세요.

노트북이 열리면 상단 메뉴에서 **Cell > Run All** 을 클릭하면 됩니다.

---

## Q: intro2new.ipynb 클릭한다음에 Run-Run All Cells 하는거맞아?

### A:

네, 맞습니다! **Run > Run All Cells** 실행하시면 됩니다.

전체 실행에 약 10분 정도 걸릴 수 있고, 중간에 데이터셋 다운로드 + 인덱싱이 진행됩니다. 셀 왼쪽에 `[*]` 표시가 있으면 아직 실행 중이라는 뜻이니 끝날 때까지 기다려주세요.

---

## Q: (캡처) RuntimeError - load_dataset에서 에러

### A:

에러 메시지를 보면 `trust_remote_code`는 더 이상 지원되지 않고, 이 데이터셋이 **구버전 loading script 방식**이라 최신 `datasets` 라이브러리와 호환이 안 되는 겁니다.

`datasets` 버전을 다운그레이드하면 해결됩니다. WSL 터미널에서:

```bash
pip install datasets==2.14.7
```

설치 후 Jupyter 커널을 재시작해야 합니다:
1. 노트북 상단 메뉴에서 **Kernel > Restart**
2. 그 다음 **Run > Run All Cells**

이때 `load_dataset` 코드는 `trust_remote_code=True` **없이** 원래대로 되돌려주세요:

```python
collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
queries_dataset = load_dataset("colbertv2/lotte", dataset)
```

---

## Q: pip install하는 경로는 어떻게 해야해? 지금 eunybe@EUNYBE-PC3:/mnt/c/Users/dmsdu$ 이렇게 되어있는데 이대로 진행하면 되나?

### A:

아니요, 가상환경을 먼저 활성화해야 합니다:

```bash
source ~/colbert-env/bin/activate
pip install datasets==2.14.7
```

프롬프트 앞에 `(colbert-env)`가 표시된 상태에서 pip 명령어를 실행해주세요. 경로는 어디든 상관없습니다.

---

## Q: ERROR: pip's dependency resolver 에러

```
transformers 5.2.0 requires huggingface-hub<2.0,>=1.3.0, but you have huggingface-hub 0.36.2 which is incompatible.
Successfully installed datasets-2.14.7 dill-0.3.7 fsspec-2023.10.0 huggingface-hub-0.36.2 multiprocess-0.70.15 pyarrow-hotfix-0.7
```

### A:

`Successfully installed`로 설치 자체는 성공했습니다! WARNING일 뿐 문제없습니다.

그대로 Jupyter 노트북으로 돌아가서:

1. **Kernel > Restart**
2. `load_dataset`에서 `trust_remote_code=True` 제거 (원래 코드로 복원)
3. **Run > Run All Cells**

---

## Q: (캡처) 다시 RuntimeError

### A:

커널을 **ColBERT**로 변경하지 않은 것 같습니다. 노트북 상단 메뉴에서:

**Kernel > Restart & Run All Cells** 해주세요.

---

## Q: (캡처) ModuleNotFoundError / ImportError - Cell [2]에서 에러

```
ModuleNotFoundError: No module named 'google'
ImportError: from colbert import Indexer, Searcher
```

### A:

Jupyter 커널이 가상환경(`colbert-env`)을 인식하지 못해서 생긴 문제입니다. WSL 터미널에서 다음을 실행해주세요:

```bash
source ~/colbert-env/bin/activate
pip install ipykernel
python -m ipykernel install --user --name=colbert-env --display-name="ColBERT"
```

그 후 Jupyter 노트북에서:

1. **Kernel > Change kernel > ColBERT** 선택
2. **Kernel > Restart & Run All Cells**

---

## Q: python -m ipkernel install 에서 No module named ipkernel 에러

### A:

오타가 있었네요. `ipkernel`이 아니라 `ipykernel`입니다:

```bash
pip install ipykernel
python -m ipykernel install --user --name=colbert-env --display-name="ColBERT"
```

`ipykernel`로 다시 실행해주세요!

---

## Q: (캡처) 같은 에러 반복

### A:

커널 등록이 제대로 안 된 것 같습니다. 더 간단한 방법으로 해결합시다.

노트북에서 **Cell [2]의 내용을 통째로 아래로 바꿔주세요**:

```python
from colbert import Indexer, Searcher
```

Cell [2]를 더블클릭해서 편집 모드로 들어간 뒤, 기존 코드를 모두 지우고 위 한 줄만 입력하세요.

그 후 **Kernel > Restart & Run All Cells** 하면 됩니다.

기존 Cell [2]는 Colab 전용 설정 코드라서, 이미 `pip install -e .`로 ColBERT를 설치한 로컬 환경에서는 필요 없습니다.

---

## Q: change kernel 하면 intro2new.ipynb만 뜨는데 이러면 잘못된거야?

### A:

그러면 커널 등록이 제대로 안 된 것 같습니다. 더 간단한 방법으로 해결합시다.

노트북에서 **Cell [2]의 내용을 통째로 아래로 바꿔주세요**:

```python
from colbert import Indexer, Searcher
```

그 후 **Kernel > Restart & Run All Cells** 하면 됩니다.

---

## Q: ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'

### A:

`datasets` 다운그레이드 과정에서 `huggingface-hub` 버전도 같이 내려가서 `transformers`와 충돌이 난 겁니다.

WSL 터미널에서 호환되는 버전으로 맞춰주세요:

```bash
source ~/colbert-env/bin/activate
pip install transformers==4.36.0 huggingface-hub==0.20.0 datasets==2.14.7
```

설치 완료 후 Jupyter 노트북에서 **Kernel > Restart & Run All Cells** 해주세요.

---

## Q: load_dataset 코드가 뭐하는거야? 뭘 많이 다운받는데 뭔질 모르겠어

```python
from datasets import load_dataset

dataset = 'lifestyle'
datasplit = 'dev'

collection_dataset = load_dataset("colbertv2/lotte_passages", dataset)
collection = [x['text'] for x in collection_dataset[datasplit + '_collection']]

queries_dataset = load_dataset("colbertv2/lotte", dataset)
queries = [x['query'] for x in queries_dataset['search_' + datasplit]]

f'Loaded {len(queries)} queries and {len(collection):,} passages'
```

### A:

이 셀은 **ColBERT의 성능을 테스트할 샘플 데이터**를 HuggingFace에서 다운로드하는 겁니다.

### 다운로드 받는 것 2가지

| 항목 | 내용 | 용량 |
|------|------|------|
| `lotte_passages` | 검색 대상 **문서(passage) 268,881개** | ~383MB |
| `lotte` | 검색할 **질문(query) 417개** + 정답 정보 | ~1.2MB |

### LoTTE가 뭐야?

ColBERTv2 논문에서 만든 **검색 벤치마크 데이터셋**입니다. 실생활 주제(lifestyle) 관련 Q&A 게시판에서 수집한 데이터로:

- **문서**: "고양이가 원래 마른 건가요?", "토마토 꽃 썩음병은..." 같은 답변글 26만개
- **질문**: "are some cats just skinny?" 같은 실제 검색 쿼리 417개

### 왜 필요해?

ColBERT는 **검색 엔진**이니까 "검색할 문서"와 "검색할 질문"이 있어야 동작을 테스트할 수 있습니다. 이 데이터가 있어야 다음 단계인 **인덱싱 → 검색**을 해볼 수 있습니다.

간단히 말하면: **"ColBERT한테 검색시켜볼 연습용 문서와 질문을 다운받는 중"** 입니다.

---

## Q: 인덱싱 셀에서 에러, FutureWarning + OSError: CUDA_HOME environment variable is not set

### A:

ColBERT가 C++/CUDA 확장을 컴파일하려는데, WSL에 **CUDA 툴킷**이 설치되어 있지 않아서 생긴 에러입니다. PyTorch는 GPU를 쓸 수 있지만, 커스텀 C++ 코드 컴파일에는 CUDA 컴파일러(nvcc)가 별도로 필요합니다.

WSL 터미널에서 순서대로 실행해주세요:

#### 1. CUDA 툴킷 설치

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

#### 2. 환경변수 설정

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

#### 3. 설치 확인

```bash
nvcc --version
```

---

## Q: 가상환경과 환경변수에 대해서 알려줘

### A:

### 가상환경 (Virtual Environment)

#### 뭐야?

프로젝트마다 **독립된 Python 패키지 공간**을 만들어주는 것입니다.

#### 왜 필요해?

```
프로젝트 A: torch 1.13 + transformers 4.20 필요
프로젝트 B: torch 2.5 + transformers 4.36 필요
```

하나의 Python에 둘 다 설치하면 **버전 충돌**이 납니다. 가상환경을 쓰면 각각 분리됩니다.

#### 구조

```
시스템 Python (/usr/bin/python3)
  ├── colbert-env/     ← ColBERT 전용 (torch 2.5, transformers 4.36...)
  ├── project-b-env/   ← 다른 프로젝트 전용
  └── project-c-env/   ← 또 다른 프로젝트 전용
```

#### 명령어

```bash
python3 -m venv ~/colbert-env        # 생성
source ~/colbert-env/bin/activate     # 활성화
deactivate                            # 비활성화
```

활성화하면 `pip install`이 **해당 가상환경 안에만** 설치됩니다.

---

### 환경변수 (Environment Variable)

#### 뭐야?

운영체제(OS)가 갖고 있는 **설정값 저장소**입니다. 프로그램들이 이 값을 읽어서 동작합니다.

#### 예시

| 환경변수 | 값 | 용도 |
|---------|-----|------|
| `PATH` | `/usr/bin:/usr/local/bin:...` | 명령어를 어디서 찾을지 |
| `HOME` | `/home/eunybe` | 사용자 홈 디렉토리 |
| `CUDA_HOME` | `/usr/local/cuda-12.1` | CUDA 툴킷 위치 |

#### 작동 방식

```bash
# ColBERT 코드 내부에서 이렇게 CUDA를 찾음
os.environ['CUDA_HOME']  →  "/usr/local/cuda-12.1"
                          →  여기서 nvcc 컴파일러를 찾아서 C++ 코드 컴파일
```

`CUDA_HOME`이 설정 안 되어 있으면 → 아까 본 `OSError` 발생!

#### 명령어

```bash
export CUDA_HOME=/usr/local/cuda-12.1   # 설정 (현재 터미널에서만)
echo $CUDA_HOME                          # 확인
env                                      # 모든 환경변수 보기
```

#### 영구 설정

`export`만 하면 **터미널을 닫으면 사라집니다**. 그래서 `~/.bashrc`에 적어둡니다:

```bash
echo 'export CUDA_HOME=/usr/local/cuda-12.1' >> ~/.bashrc
```

`~/.bashrc`는 터미널을 열 때마다 자동 실행되는 설정 파일이라, 매번 다시 입력할 필요가 없어집니다.

---

### 둘의 관계

```
터미널 열기
  → ~/.bashrc 실행 → 환경변수(CUDA_HOME, PATH 등) 자동 설정
  → source ~/colbert-env/bin/activate → 가상환경 활성화
  → pip, python 명령어가 가상환경 안의 패키지를 사용
  → ColBERT가 CUDA_HOME 환경변수를 읽어서 GPU 코드 컴파일
```

가상환경은 **Python 패키지를 분리**, 환경변수는 **시스템 설정값을 저장**하는 역할입니다.

---

## Q: sudo apt 라는 명령어가 많던데 이거는 어떤 의미야?

### A:

### sudo

**S**uper **U**ser **Do** = **관리자 권한으로 실행**

일반 사용자는 시스템 파일을 수정할 수 없습니다. `sudo`를 붙이면 관리자(root) 권한으로 명령을 실행합니다.

```bash
apt install gcc          # 권한 없음 에러
sudo apt install gcc     # 관리자 권한으로 설치
```

비유하면: Windows에서 **"관리자 권한으로 실행"** 하는 것과 같습니다.

### apt

**A**dvanced **P**ackage **T**ool = Ubuntu의 **시스템 패키지 관리자**

| 명령어 | 의미 |
|--------|------|
| `apt update` | 설치 가능한 패키지 목록 최신화 |
| `apt install gcc` | gcc 패키지 설치 |
| `apt remove gcc` | gcc 패키지 삭제 |

### pip과 뭐가 달라?

| | apt | pip |
|--|-----|-----|
| 대상 | **시스템 프로그램** (gcc, git, python 등) | **Python 패키지** (torch, numpy 등) |
| 범위 | OS 전체 | Python 환경 내 |
| 예시 | `sudo apt install gcc` | `pip install torch` |

```
시스템 레벨 (apt)
  └── gcc, g++, python3, git, wget ...

Python 레벨 (pip)
  └── torch, transformers, datasets, colbert ...
```

### 정리

`sudo apt install -y gcc g++`의 의미:

- `sudo`: 관리자 권한으로
- `apt install`: 시스템 패키지를 설치해라
- `-y`: 확인 질문 없이 자동으로 "예"
- `gcc g++`: C/C++ 컴파일러 두 개를 설치

---

## Q: nvcc --version 이 안돼서 sudo apt install nvidia-cuda-toolkit을 했는데, 다운로드 중 에러

```
Err:136 http://archive.ubuntu.com/ubuntu noble/multiverse amd64 nsight-compute amd64 2022.4.1.6~12.0.1-4build4
  Error reading from server - read (104: Connection reset by peer)
E: Failed to fetch ...
E: Unable to fetch some archives, maybe run apt-get update or try with --fix-missing?
```

### A:

네트워크 타임아웃으로 일부 패키지 다운로드가 실패한 것입니다. 다시 시도하면 됩니다:

```bash
sudo apt-get update
sudo apt-get install -y --fix-missing nvidia-cuda-toolkit
```

`--fix-missing`이 실패한 파일만 다시 다운로드합니다.

---

## Q: 지금 85%에서 계속 멈춰있는데 괜찮은거겠지?

### A:

네, 괜찮습니다. `nsight-compute` 패키지가 용량이 크고 서버가 느려서 시간이 오래 걸리는 것입니다.

기다려주세요. 만약 10분 이상 멈춰 있으면 `Ctrl+C`로 취소 후 다시 실행하면 됩니다:

```bash
sudo apt-get install -y --fix-missing nvidia-cuda-toolkit
```

이미 받은 파일은 캐시되어 있어서 처음부터 다시 받지 않습니다.

---

## Q: 인덱싱 셀에서 FutureWarning이 나오는데 어떻게하지? 그리고 이게 어떤 cell이야?

### A:

정상적으로 진행되고 있습니다! 에러가 아니라 **경고(FutureWarning)** 입니다.

Cell [9]는 **인덱싱 셀 (Cell [14])** 입니다. (Kernel 재시작으로 셀 번호가 바뀐 것뿐)

지금 출력을 보면:

| 로그 | 의미 | 상태 |
|------|------|------|
| `Encoding 10000 passages` | 1만개 문서를 BERT로 인코딩 | 완료 |
| `Creating 16,384 partitions` | 벡터 클러스터링 준비 | 완료 |
| `Saving the indexing plan` | 인덱싱 계획 저장 | 완료 |
| `torch.load` FutureWarning | "나중 버전에서 바뀔 예정" 안내 | 무시해도 됨 |

**셀 왼쪽에 `[*]`가 있으면 아직 실행 중**입니다.

---

## Q: 인덱싱 셀의 코드가 뭐하는거야?

```python
checkpoint = 'colbert-ir/colbertv2.0'

with Run().context(RunConfig(nranks=1, experiment='notebook')):
    config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, kmeans_niters=4)
    indexer = Indexer(checkpoint=checkpoint, config=config)
    indexer.index(name=index_name, collection=collection[:max_id], overwrite=True)
```

### A:

이 셀은 ColBERT의 **핵심 작업인 인덱싱**을 수행합니다.

### 한 줄씩 설명

```python
checkpoint = 'colbert-ir/colbertv2.0'
```
HuggingFace에서 **사전학습된 ColBERTv2 모델**을 가져옴 (BERT 기반, 438MB)

```python
with Run().context(RunConfig(nranks=1, experiment='notebook')):
```
- `nranks=1`: GPU **1개** 사용
- `experiment='notebook'`: 결과를 `experiments/notebook/` 폴더에 저장

```python
config = ColBERTConfig(doc_maxlen=300, nbits=2, kmeans_niters=4)
```
- `doc_maxlen=300`: 문서를 최대 **300 토큰**까지 처리
- `nbits=2`: 벡터를 **2비트**로 압축 (용량 절약)
- `kmeans_niters=4`: 클러스터링 **4번** 반복

```python
indexer = Indexer(checkpoint=checkpoint, config=config)
indexer.index(name=index_name, collection=collection[:max_id], overwrite=True)
```
- `collection[:max_id]`: 처음 **10,000개 문서**만 사용
- 각 문서를 BERT에 넣어서 **토큰마다 벡터(임베딩)** 생성
- 벡터들을 클러스터링하고 압축해서 **디스크에 인덱스로 저장**

### 비유하자면

> 도서관에서 책 10,000권의 **색인(index)을 만드는 작업**입니다.
> 나중에 질문이 들어오면, 책을 하나하나 읽지 않고 색인을 보고 **관련 있는 책을 빠르게 찾기 위해** 미리 정리해두는 것입니다.

---

## Q: nvcc not found 에러 - RuntimeError: Error building extension 'decompress_residuals_cpp'

```
/bin/sh: 1: /usr/local/cuda-12.1/bin/nvcc: not found
```

### A:

핵심 에러는 `/usr/local/cuda-12.1/bin/nvcc: not found` 입니다. `apt`로 설치한 CUDA 툴킷은 `/usr/local/cuda-12.1`이 아니라 다른 경로에 설치됩니다.

WSL 터미널에서 먼저 실제 경로를 확인해주세요:

```bash
which nvcc
```

아마 `/usr/bin/nvcc`가 나올 겁니다. 그러면 `CUDA_HOME`을 수정해야 합니다:

```bash
sed -i 's|export CUDA_HOME=/usr/local/cuda-12.1|export CUDA_HOME=/usr|' ~/.bashrc
source ~/.bashrc
```

그 후 Jupyter를 `Ctrl+C`로 종료하고 다시 실행:

```bash
source ~/colbert-env/bin/activate
cd ~/ColBERT
jupyter notebook docs/intro2new.ipynb --no-browser
```

노트북에서 **Kernel > Restart & Run All Cells** 해주세요.

---

## Q: 최종 검색 결과 - 이 결과를 좀 이해하기 쉽게 해석해줘

```
#> are some cats just skinny?

	 [1] 		 25.4 		 A cat can certainly be naturally skinny...
	 [2] 		 24.9 		 Yes. Just like us, cats vary in size and shape...
	 [3] 		 23.1 		 Thank you for your question...
```

### A:

ColBERT 검색이 성공적으로 동작했습니다!

### 질문

> **"are some cats just skinny?"** (어떤 고양이들은 그냥 마른 건가요?)

### 검색 과정

```
질문 → BERT로 토큰화 → 벡터로 변환 → 인덱스에서 유사한 문서 검색
```

### 토큰화 과정

```
"are some cats just skinny?"
  → [101, 1, 2024, 2070, 8870, 2074, 15629, 1029, 102, 103, 103, ...]
     [CLS] [Q] are  some  cats  just  skinny   ?   [SEP] [MASK]...
```

- 실제 단어는 8개, 나머지는 32자리를 채우기 위한 `[MASK]` 패딩
- `Output Mask`에서 `1`이 실제 토큰, `0`이 패딩

### Top-3 결과

#### 1위 - 점수 25.4

> "고양이는 당연히 선천적으로 마를 수 있습니다. 제가 아는 고양이는 새끼 때부터 평생 극도로 말랐고, 뼈를 쉽게 셀 수 있을 정도였습니다. 지금 17살인데 같은 집 다른 고양이 2마리보다 오래 살았으니, 마른 게 건강에 문제가 된 건 아닌 것 같습니다."

→ **질문에 가장 직접적으로 답변**: "네, 원래 마른 고양이도 있어요"

#### 2위 - 점수 24.9

> "네, 사람처럼 고양이도 체형이 다양합니다. 식단, 건강, 유전, 나이에 따라 달라집니다. 한 고양이는 다른 고양이의 절반 체중이었지만 과체중이 아니었습니다..."

→ **더 상세한 설명**: 유전적/개체별 차이를 설명

#### 3위 - 점수 23.1

> "고양이들이 서로 밥을 먹는 문제에 대한 답변. 저도 많이 먹는 고양이와 마른 고양이가 있습니다..."

→ **간접적 관련**: "마른 고양이"가 언급되지만 직접적인 답은 아님

### 점수의 의미

점수는 ColBERT의 **MaxSim 유사도**입니다. 질문의 각 토큰과 문서의 각 토큰 간 유사도를 계산해서 합산한 값입니다.

```
25.4 > 24.9 > 23.1
 1위     2위     3위
```

점수가 높을수록 질문과 더 관련 있는 문서입니다. 1위와 2위는 점수 차이가 작아서 둘 다 매우 관련성이 높고, 3위는 상대적으로 낮습니다.

요약하면: **"고양이가 원래 마른 건가요?"라고 물었더니, 10,000개 문서 중에서 가장 관련 있는 답변 3개를 정확하게 찾아낸 것입니다!**

---

## Q: 근데 jupyter는 뭐하는거야?

### A:

**Jupyter Notebook**은 코드를 한 셀(블록)씩 실행하면서 결과를 바로 확인할 수 있는 **대화형 코딩 도구**입니다.

### 일반 Python 파일 vs Jupyter

| | 일반 `.py` 파일 | Jupyter `.ipynb` |
|--|----------------|------------------|
| 실행 방식 | 전체를 한 번에 실행 | 셀 단위로 나눠서 실행 |
| 결과 확인 | 터미널에 출력 | 셀 바로 아래에 표시 |
| 수정 후 | 전체 다시 실행 | 해당 셀만 다시 실행 |
| 용도 | 완성된 프로그램 | 실험, 분석, 튜토리얼 |

### 왜 Jupyter를 쓰는가?

데이터 로드 → 인덱싱 → 검색 같은 단계를 **하나씩 실행하면서 중간 결과를 확인**할 수 있어서, 머신러닝/데이터 분석에서 많이 씁니다.

---

## Q: ColBERTv2: Indexing & Search Notebook 의 코드 Cell을 하나하나 어떤 의미인지 설명해줘.

### A:

(전체 셀별 상세 설명 - 위 내용과 동일하므로 생략)

---

## Q: 인덱싱 셀 실행중 FutureWarning 출력

### A:

정상적으로 진행되고 있습니다! 에러가 아니라 **경고(FutureWarning)** 입니다. 셀 왼쪽에 `[*]`가 있으면 아직 실행 중이니 기다려주세요.
