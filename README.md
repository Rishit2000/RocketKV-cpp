# RocketKV-cpp: Two-Stage KV Cache Compression in llama.cpp

High-performance C++ implementation of **RocketKV** integrated with [`llama.cpp`](https://github.com/ggerganov/llama.cpp) for low-latency, memory-efficient LLM inference on consumer-edge devices (Apple Silicon Metal GPU).

---

## Repository Structure

```
RocketKV-cpp/
├── rocket.h / rocket.cpp       # Core RocketKV eviction algorithms (C++17)
├── runner.cpp                  # C++ batch inference runner linked to llama.cpp/GGML
├── test_rocket.cpp             # Unit tests for RocketKV kernel operations
├── Makefile                    # Build scripts for Metal and CPU compilation
├── eval.py                     # CLI benchmark driver
├── eval_suite/                 # Benchmarking orchestration & scoring engine
│   ├── engine.py               # Metric aggregation & comparison table renderer
│   ├── runner.py               # Process batch runner client
│   ├── datasets.py             # LongBench loader & RULER procedural generator
│   ├── metrics.py              # Official scoring metrics (F1, ROUGE, EM, etc.)
│   └── config.py               # Declarative JSON configuration parser
├── configs/                    # Experiment configuration files
│   ├── longbench.json          # Standard LongBench evaluation (6 tasks, 8k context)
│   ├── ruler.json              # NVIDIA RULER synthetic benchmark (13 tasks)
│   └── default.json            # Fast sanity check config
├── scripts/
│   ├── download_model.sh       # Script to fetch Llama-3.2 GGUF weights
│   ├── download_longbench.sh   # Script to download LongBench JSONL datasets
│   └── prepare_ruler_data.py   # RULER synthetic dataset generator
└── environment.yml             # Conda environment definition
```

---

## Step-by-Step Setup & Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/Rishit2000/RocketKV-cpp.git
cd RocketKV-cpp
```

### 2. Set Up the Python Environment

Using Conda:

```bash
conda env create -f environment.yml
conda activate rocketkv
```

Or using standard `pip`:

```bash
pip install datasets transformers tiktoken numpy
```

---

### 3. Clone and Build `llama.cpp`

RocketKV links against a pinned, patched `llama.cpp`. The build script checks the revision, applies the graph integration patch, and builds the libraries and runner:

```bash
bash scripts/stage2.sh build
```

The revision and patch are stored in `patches/`. Existing checkouts at another revision or with conflicting edits are rejected, not reset. `JOBS=4` is the default build parallelism; Metal is enabled on macOS, but Stage 2 currently executes on CPU.

---

### 4. Build the RocketKV Runner

Compile the C++ runner binary using the provided `Makefile`:

```bash
make runner
```

To run unit tests verifying the RocketKV eviction kernel:

```bash
make test_rocket
./test_rocket
```

### Stage 2: build, test, and run

Stage 2 is connected to every decode layer through a GGML custom CPU node. It reads each layer's real KV cache, sorts live slots by absolute position, and feeds the sparse attention result into the original output projection. Prefill and the existing Stage 1 policy are unchanged except for capture/error-handling fixes.

```bash
# Algorithm, UBSan, and real-model integration tests.
# Downloads a pinned, SHA-256-verified 1.1 MB Llama test model if needed.
bash scripts/stage2.sh test

# Three-way CPU comparison: full KV, Stage 1, and Stage 1 + HSA.
bash scripts/stage2.sh run /path/to/llama-model.gguf

# Use your own existing batch JSON and output directory.
bash scripts/stage2.sh run /path/to/llama-model.gguf samples.json results/my-stage2-run
```

`scripts/stage2.sh` is the single Stage 2 entry point; `--help` lists its build/test/run subcommands and environment overrides. Both test and run build automatically. Results include actual per-layer HSA calls, selected/candidate counts, initialization time, and latency per decode forward. The Python evaluation suite keeps its original modes; use the runner or this script for HSA.

The runner supports `--rocket-mode off|stage1|hsa`. Legacy `--rocket 1` remains Stage-1-only; do not combine the two flags. HSA currently requires a supported Llama model, F16 KV, one sequence, and CPU execution. Unsupported combinations fail explicitly. This is a **functional CPU reference**, not an optimized Metal kernel or a multi-turn implementation; no speedup or task-quality claim is implied by the tiny-model tests.

The test script uses UBSan by default. To also run ASan, set `SANITIZERS=address,undefined` and select a compatible compiler with `CXX`. On the development machine, Apple's ASan runtime deadlocked before `main()`; that run was stopped and is not reported as passing. Sanitizer runs have a timeout rather than hanging indefinitely.

For diagnostics, `--hsa-full` selects all current candidates and `--trace-logits` records per-layer attention outputs and logits. Tests use fixed `forced_tokens` in batch JSON. The native comparator uses `--reference-f32` only during decode: ordinary CPU F16 FlashAttention rounds query/accumulator values, whereas the Rocket reference accumulates in F32. Native F32 matmul/softmax is the numerical oracle; default inference still uses the original native attention path.

---

### 5. Download Model Weights

Download the quantized `Llama-3.2-3B-Instruct-Q8_0.gguf` model using the helper script:

```bash
bash scripts/download_model.sh
```

*(Alternatively, download any compatible Llama-3.2 GGUF file from Hugging Face into the root directory.)*

---

### 6. Prepare Evaluation Datasets

Because raw dataset files are large, they are excluded from Git and should be fetched or generated locally:

#### A. LongBench Datasets (Official)
Download all official LongBench JSONL evaluation sets directly from Hugging Face into `data/longbench/`:

```bash
bash scripts/download_longbench.sh
```

#### B. NVIDIA RULER Benchmark (Synthetic)
Generate the 13 official RULER tasks (500 samples per task at 8,192 context length) into `data/ruler/`:

```bash
python3 scripts/prepare_ruler_data.py --context-len 8192 --num-samples 500
```

*(Note: If `data/ruler/` is not pre-generated, `eval.py` will also generate the required evaluation samples procedurally on-the-fly.)*

---

### 7. Run Evaluation Benchmarks

#### LongBench Evaluation
To evaluate across LongBench tasks comparing `full-kv`, `quantized-kv (q8_0)`, and `rocket-kv` (budgets 512, 1024, 2048):

```bash
python3 eval.py configs/longbench.json
```

#### NVIDIA RULER Benchmark
To run the RULER synthetic long-context evaluation suite:

```bash
python3 eval.py configs/ruler.json
```

---

## Benchmark Configuration

All benchmark parameters are defined in declarative JSON files under `configs/`:

```json
{
  "model": {
    "path": "Llama-3.2-3B-Instruct-Q8_0.gguf",
    "runner": "./runner",
    "max_tokens": 32,
    "n_ctx": 9216
  },
  "benchmark": {
    "suite": "longbench",
    "tasks": ["2wikimqa", "qasper", "multi_news", "trec"],
    "context_len": 8192,
    "num_samples": 30
  },
  "attention": {
    "attn_type": "all",
    "kv_type": "q8_0",
    "budget": [512, 1024, 2048],
    "window_size": 32,
    "kernel_size": 63
  },
  "management": {
    "output_dir": "results/longbench",
    "data_dir": "data"
  }
}
```

---

## Citation & References

- **RocketKV Paper**: *Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression* ([arXiv:2502.14051](https://arxiv.org/abs/2502.14051), ICML 2025).
- **llama.cpp**: *LLM inference in C/C++* by Georgi Gerganov and contributors ([github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)).

