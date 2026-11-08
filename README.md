# RocketKV-cpp: Two-Stage KV Cache Compression in llama.cpp

C++17 implementation of **RocketKV** prefill eviction and hybrid sparse decode, integrated with a pinned, patched [`llama.cpp`](https://github.com/ggerganov/llama.cpp).

> **Current status:** Stage 1 + Stage 2 supports CPU-reference and Metal HSA for single-sequence Llama models with F16 KV. The Metal backend executes cached chunk summaries, approximate scoring, deterministic Top-K selection and exact sparse attention on GPU. Stage 1 eviction and KV position bookkeeping remain on CPU. Multi-turn inference is not supported.

---

## Repository Structure

```text
RocketKV-cpp/
├── include/                   # Public C++ interfaces
├── src/
│   ├── runner.cpp             # Model loading, modes, inference loop and metrics
│   ├── rocket.cpp             # CPU reference algorithms
│   ├── rocket_adapter.cpp     # GGML graph integration and KV position mapping
│   └── metal/                 # Metal kernels, dispatch wrapper and non-macOS stub
├── tests/                     # CPU/Metal kernel tests and model integration tests
├── scripts/                   # Build/test/run entry point and download helpers
├── configs/                   # Benchmark configurations
├── eval.py / eval_suite/      # Legacy Python evaluation pipeline
├── patches/                   # Pinned llama.cpp revision and integration patch
├── Makefile
├── environment.yml
├── llama.cpp/                 # Downloaded upstream checkout (ignored)
├── build/                     # Binaries, generated shaders and objects (ignored)
├── models/                    # Downloaded GGUF models (ignored)
├── data/                      # Evaluation datasets (ignored)
└── results/                   # Experiment outputs and logs (ignored)
```

---

## What Changed for Stage 2

| Component | Responsibility |
|---|---|
| `src/runner.cpp` | Select `off`, `stage1`, or `hsa`; hand over retained positions; manage request state and report actual forward/layer calls. |
| `src/rocket_adapter.cpp`, `include/rocket_adapter.h` | Map retained positions to live KV slots and dispatch sparse attention through GGML Metal or the CPU callback. |
| `src/metal/`, `include/rocket_metal.h` | Metal HSA kernels and GGML GPU dispatch. |
| `patches/llama-stage2.patch` | Add context-scoped hooks and KV-position access; replace decode attention inside the graph; prevent stale graph reuse after mode changes. |
| `src/rocket.cpp`, `include/rocket.h` | Retain the algorithm structure while adding input checks, short-prompt/zero-query fixes, deterministic top-r ties, and optional selection output for tests. |

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

Build requirements: Git, CMake, `make`, and a C++17 compiler. Python 3 embeds the Metal shader during macOS builds and runs the test/run entry points; downloading the test model requires `curl`.

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

The revision and patch are stored in `patches/`. Existing checkouts at another revision or with conflicting edits are rejected, not reset. `JOBS=4` is the default build parallelism; Metal model offload is enabled on macOS. Stage 2 uses the CPU reference by default; select `--stage2-backend metal` for GPU HSA.

---

### 4. Incremental Builds and Algorithm Tests

Step 3 already builds the runner. After editing C++ code, rebuild with:

```bash
make runner
```

To run unit tests verifying the RocketKV eviction kernel:

```bash
make test_rocket
./build/test_rocket
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

`scripts/stage2.sh` is the single Stage 2 entry point; `--help` lists its build/test/run subcommands and environment overrides. Both test and run build automatically. Results include actual per-layer HSA calls, selected/candidate counts, initialization time, latency per decode forward, requested GPU layers, and the HSA model backend/device. KV size fields are logical estimates, not measured backend allocations. The Python evaluation suite keeps its original modes; use the runner or this script for HSA.

| Runner mode | Prefill / eviction | Decode attention |
|---|---|---|
| `--rocket-mode off` | Native prefill, no eviction | Native |
| `--rocket-mode stage1` | Native prefill + Stage 1 eviction | Native |
| `--rocket-mode hsa` | Native prefill + the same Stage 1 eviction | CPU-reference or Metal HSA |

Legacy `--rocket 0/1` still selects off/stage1; do not combine it with `--rocket-mode`. HSA requires F16 **KV**, not necessarily F16 model weights. The default remains CPU-reference HSA. Use `--stage2-backend metal` for GPU HSA, and `--gpu-layers 99` for Metal model layers. With Metal HSA, omitted `--gpu-layers` defaults to all model layers on Metal; an explicit `--gpu-layers 0` keeps model computation on CPU for kernel comparisons. Unsupported backends fail explicitly, with no silent CPU fallback.

```bash
# Three-way comparison, with model layers and Stage 2 HSA on Metal.
STAGE2_BACKEND=metal GPU_LAYERS=99 THREADS=16 BUDGET=1024 N_CTX=9216 CONTEXT_LEN=8192 \
  bash scripts/stage2.sh run /path/to/llama-model.gguf samples.json results/stage2-metal

# CPU, Metal, direct kernel comparisons, and UBSan regression tests.
ROCKET_TEST_METAL=1 bash scripts/stage2.sh test
```

The Metal implementation is isolated in `src/metal/`; `src/rocket.cpp` remains the CPU reference. Kernels are embedded into the binary at build time. With compatible shared Metal buffers, HSA encodes into GGML's existing command encoder, reads Q/K/V and writes its output directly, and checks GPU errors after decode synchronization. Scratch buffers are isolated per layer; the scheduler tracks the complete tensor dependencies. CPU inputs and unsupported buffers retain synchronous staging, with direct K/V binding where available. `scripts/stage2.sh build` applies the optional buffer and custom-op bridges from `patches/`.

GPU kernels read selected F16 K/V with F32 accumulation. Per-layer chunk min/max summaries reuse immutable KV prefixes and update the tail on append; request resets, changed slots or summary shapes rebuild them. Single-token chunks read K directly. Top-K ranks chunks and expands their token IDs, using threadgroup memory when it fits and device memory otherwise. Both paths preserve score/index tie-breaking and chronological selected-token order. Per-change measurements and code differences are recorded in [HSA_OPTIMIZATION_LOG.md](HSA_OPTIMIZATION_LOG.md).

Supported Metal shapes are head dimensions 1–256, up to 256 query heads per KV group, and 1–32,768 live candidates, with F32 Q and F16 K/V. Unsupported shapes and non-finite input/output fail explicitly. The shader uses F32 accumulation and deterministic score/index tie-breaking; floating-point reduction order can differ from the CPU reference, particularly near selection boundaries. Full-model quantized CPU/Metal math also differs. Tests compare selected indices and attention on controlled inputs, and layer outputs/logits in real-model integration tests.

`stage2_backend: "metal"` records GPU HSA execution; `model_backend` separately records model placement. CPU model + Metal HSA is supported for diagnostics. Metal model + `cpu-reference` HSA remains available as a hybrid baseline.

To run **only HSA** on an existing batch, rather than all three comparison modes:

```bash
./build/runner --model /path/to/llama-model.gguf --batch samples.json \
  --rocket-mode hsa --kv-type f16 --gpu-layers 0 --budget 1024 \
  --n-ctx 9216 --context-len 8192 --output hsa-results.json
```

The batch input is a JSON array, for example:

```json
[{"sample_id": 1, "prompt": "Once upon a time", "max_tokens": 32}]
```

The built-in demonstration is a smoke test, not a LongBench/RULER accuracy evaluation. `test` covers algorithm boundaries, all-candidate attention/logit equivalence, sparse graph connectivity, state isolation, legacy modes, failure cases, and script dispatch. The CPU reference recomputes chunk information each step; Metal caches it per layer and uses the scheduler's command encoder for compatible buffers.

The test script uses UBSan by default. To also run ASan, set `SANITIZERS=address,undefined` and select a compatible compiler with `CXX`. On the development machine, Apple's ASan runtime deadlocked before `main()`; that run was stopped and is not reported as passing. Sanitizer runs have a timeout rather than hanging indefinitely.

For diagnostics, `--hsa-full` selects all current candidates and `--trace-logits` records per-layer attention outputs and logits. Tests use fixed `forced_tokens` in batch JSON. The native comparator uses `--reference-f32` only during decode: ordinary CPU F16 FlashAttention rounds query/accumulator values, whereas the Rocket reference accumulates in F32. Native F32 matmul/softmax is the numerical oracle; default inference still uses the original native attention path.

---

### 5. Download Model Weights

Download the quantized `Llama-3.2-3B-Instruct-Q8_0.gguf` model using the helper script:

```bash
bash scripts/download_model.sh
```

*(Alternatively, download any compatible Llama-3.2 GGUF file from Hugging Face into `models/`.)*

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

> **Important:** these `eval.py` commands use the original Python pipeline. Its `rocket-kv` mode is **Stage 1 only**, and `attn_type: "all"` does not include HSA. Use `scripts/run_longbench_stage2.py` below for the scored three-mode Metal comparison.

#### Scored LongBench comparison with Metal HSA

```bash
python3 -m pip install rouge==1.0.1 fuzzywuzzy==0.18.0 python-Levenshtein==0.27.1
python3 scripts/run_longbench_stage2.py --config configs/longbench.json \
  --budget 1024 --threads 16 --output results/longbench_stage2
```

This entry point runs **off / stage1 / hsa** with F16 KV and Metal model offload, using Metal HSA in the third mode. It reads the task list, sample count, model, context limit and eviction settings from the config; `--budget` selects one budget, rather than the legacy config's attention-mode/budget sweep. The default small set is the first 30 samples of each of four tasks (120 samples, 360 evaluations). Task-specific generation limits are retained.

Every sample has a scored JSON checkpoint and a runner log. `status.json` tracks progress, `summary.json` contains per-task scores and timing, and `manifest.json` records the binary, inputs and settings. Re-running the same command resumes completed checkpoints; changed settings require a new output directory. Each sample loads the model separately for durable checkpoints, so wall time includes startup overhead. Inference time is reported separately. Scores reuse this repository's prompt templates, prediction cleanup and metrics; they are not a claim of official leaderboard equivalence.

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
    "path": "models/Llama-3.2-3B-Instruct-Q8_0.gguf",
    "runner": "./build/runner",
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

