# RocketKV-cpp: Two-Stage KV Cache Compression in llama.cpp

C++17 implementation of **RocketKV** prefill eviction and hybrid sparse decode, integrated with a pinned, patched [`llama.cpp`](https://github.com/ggerganov/llama.cpp).

> **Current status:** Stage 1 + Stage 2 is connected to real model inference as a **CPU reference** for supported single-sequence Llama models with F16 KV. There is no Metal HSA kernel or multi-turn support yet. The included small-model tests establish correctness, not a large-model speedup or quality claim.

---

## Repository Structure

```
RocketKV-cpp/
├── rocket.h / rocket.cpp       # Budget policy, Stage 1 eviction, Stage 2 CPU algorithm
├── rocket_adapter.h / rocket_adapter.cpp # GGML node, KV mapping, tensor packing
├── runner.cpp                  # Mode selection, inference loop, errors, metrics
├── test_rocket.cpp             # Deterministic algorithm and boundary tests
├── tests/
│   └── test_stage2_integration.py # Real-model equivalence and script-entry tests
├── patches/
│   ├── llama.cpp-revision     # Pinned upstream commit
│   └── llama-stage2.patch     # Context hooks and attention graph integration
├── Makefile                    # Incremental runner and algorithm-test builds
├── eval.py                     # Legacy Python benchmark driver (Stage 1 only for Rocket)
├── eval_suite/                 # Benchmarking orchestration & scoring engine
│   ├── engine.py               # Metric aggregation & comparison table renderer
│   ├── runner.py               # Process batch runner client
│   ├── datasets.py             # LongBench loader & RULER procedural generator
│   ├── metrics.py              # Official scoring metrics (F1, ROUGE, EM, etc.)
│   └── config.py               # Declarative JSON configuration parser
├── configs/                    # Experiment configuration files
│   ├── longbench.json          # LongBench subset (4 tasks, 30 samples/task, 8k context)
│   ├── ruler.json              # NVIDIA RULER synthetic benchmark (13 tasks)
│   └── default.json            # Fast sanity check config
├── scripts/
│   ├── stage2.sh               # Single entry point: build / test / run
│   ├── download_model.sh       # Script to fetch Llama-3.2 GGUF weights
│   ├── download_longbench.sh   # Script to download LongBench JSONL datasets
│   └── prepare_ruler_data.py   # RULER synthetic dataset generator
└── environment.yml             # Conda environment definition
```

---

## What Changed for Stage 2

| Component | Responsibility |
|---|---|
| `runner.cpp` | Select `off`, `stage1`, or `hsa`; hand over retained positions; manage request state and report actual forward/layer calls. |
| `rocket_adapter.cpp/.h` | Read each layer's real KV cache, map positions to live slots, pack tensors, and call the existing sparse attention algorithm from a GGML CPU node. |
| `patches/llama-stage2.patch` | Add context-scoped hooks and KV-position access; replace decode attention inside the graph; prevent stale graph reuse after mode changes. |
| `rocket.cpp/.h` | Retain the algorithm structure while adding input checks, short-prompt/zero-query fixes, deterministic top-r ties, and optional selection output for tests. |

```text
Native prefill -> Stage 1 prompt-KV eviction -> initialize adapter
  -> each decode forward:
       each layer: write current K/V -> HSA attention -> original Wo / residual / FFN
  -> logits and sampling
```

HSA runs **inside** `llama_decode()`, not as an unused calculation after it returns. The first generated token still comes from prefill logits. Stage 1 retains the existing last-layer, globally shared selection policy; Stage 2 selects independently for each layer and KV group without permanently deleting its unselected candidates.

The adapter files separate integration from algorithm code. The upstream patch is kept small and reproducible instead of committing the ignored `llama.cpp/` checkout.

---

## Step-by-Step Setup & Quickstart

### 1. Clone the Repository

The Stage 2 implementation is published on `xiang/dev`:

```bash
git clone --branch xiang/dev https://github.com/Rishit2000/RocketKV-cpp.git
cd RocketKV-cpp
```

Build requirements: Git, CMake, `make`, and a C++17 compiler. The test/run entry points also use Python 3; downloading the test model requires `curl`.

### 2. Set Up the Python Evaluation Environment (Optional)

This environment is for dataset preparation and `eval.py`. The C++ build and Stage 2 regression tests do not require these Python packages.

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

### 4. Incremental Builds and Algorithm Tests

Step 3 already builds the runner. After editing C++ code, rebuild with:

```bash
make runner
```

To run unit tests verifying the RocketKV eviction kernel:

```bash
make test_rocket
./test_rocket
```

### Stage 2: build, test, and run

Use the unified entry point for the complete validation suite or a three-way CPU comparison:

```bash
# Algorithm, UBSan, and real-model integration tests.
# Downloads a pinned, SHA-256-verified 1.1 MB Llama test model if needed.
bash scripts/stage2.sh test

# Three-way CPU comparison: full KV, Stage 1, and Stage 1 + HSA.
bash scripts/stage2.sh run /path/to/llama-model.gguf

# Use your own existing batch JSON and output directory.
bash scripts/stage2.sh run /path/to/llama-model.gguf samples.json results/my-stage2-run
```

`scripts/stage2.sh` is the single Stage 2 entry point; `--help` lists its build/test/run subcommands and environment overrides. Both test and run build automatically. Results include actual per-layer HSA calls, selected/candidate counts, initialization time, and latency per decode forward. KV size fields are logical estimates, not measured backend allocations. The Python evaluation suite keeps its original modes; use the runner or this script for HSA.

| Runner mode | Prefill / eviction | Decode attention |
|---|---|---|
| `--rocket-mode off` | Native prefill, no eviction | Native |
| `--rocket-mode stage1` | Native prefill + Stage 1 eviction | Native |
| `--rocket-mode hsa` | Native prefill + the same Stage 1 eviction | CPU-reference HSA |

Legacy `--rocket 0/1` still selects off/stage1; do not combine it with `--rocket-mode`. HSA requires F16 **KV**, not necessarily F16 model weights. Unsupported model/backend/type combinations fail explicitly. The comparison script forces all three modes to CPU/F16 KV.

To run **only HSA** on an existing batch, rather than all three comparison modes:

```bash
./runner --model /path/to/llama-model.gguf --batch samples.json \
  --rocket-mode hsa --kv-type f16 --gpu-layers 0 --budget 1024 \
  --n-ctx 9216 --context-len 8192 --output hsa-results.json
```

The batch input is a JSON array, for example:

```json
[{"sample_id": 1, "prompt": "Once upon a time", "max_tokens": 32}]
```

The built-in demonstration is a smoke test, not a LongBench/RULER accuracy evaluation. `test` covers algorithm boundaries, all-candidate attention/logit equivalence, sparse graph connectivity, state isolation, legacy modes, failure cases, and script dispatch. The reference currently packs candidate K/V on CPU and recomputes chunk information each step; persistent summaries and backend optimization remain future work.

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

> **Important:** these `eval.py` commands use the original Python pipeline. Its `rocket-kv` mode is **Stage 1 only**, and `attn_type: "all"` does not include HSA. Formal Stage 1 + Stage 2 evaluation still needs a minimal HSA mode connection to this pipeline, or explicit batch export, HSA execution, and result scoring.

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

Legacy Python benchmark parameters are defined in declarative JSON files under `configs/`. These are not automatically consumed by `scripts/stage2.sh`:

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

Dataset-specific generation limits can override `model.max_tokens`. For example, LongBench `multi_news` uses a limit of 512 in `eval_suite/config/dataset2maxlen.json`; do not estimate all task runtimes using the example's value of 32.

---

## Citation & References

- **RocketKV Paper**: *Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression* ([arXiv:2502.14051](https://arxiv.org/abs/2502.14051), ICML 2025).
- **llama.cpp**: *LLM inference in C/C++* by Georgi Gerganov and contributors ([github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)).

