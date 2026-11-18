# RocketKV-cpp

RocketKV reduces the KV cache used by single-sequence Llama inference in a pinned, patched `llama.cpp`. **Stage 1** evicts prompt tokens after prefill. **Stage 2** uses hybrid sparse attention (HSA) during decode: each layer and KV group selects from the retained candidates at each step. HSA requires F16 KV; model weights can be quantized.

[Repository layout](#repository-layout) · [Inference pipeline](#inference-pipeline) · [Run LongBench](#run-longbench-on-metal) · [Results and resume](#read-and-resume-a-run) · [Development](#development)

## Repository layout

```text
RocketKV-cpp/
├── include/                  Public interfaces for the algorithms, adapter, and Metal backend
├── src/
│   ├── runner.cpp            Model loading, prefill/decode loop, sampling, and metrics
│   ├── rocket.cpp            CPU reference for Stage 1 eviction and Stage 2 attention
│   ├── rocket_adapter.cpp    Maps retained positions to live KV slots; connects HSA to GGML
│   └── metal/                GPU kernels (.metal), dispatch (.mm), and non-Metal build stub
├── patches/                  Pinned llama.cpp revision, decode graph hooks, and Metal bridges
├── scripts/                  Build, test, download, and benchmark entry points
├── configs/                  Model paths, task lists, sample counts, and context limits
├── eval_suite/               Dataset loaders, prompt templates, and scoring functions
├── eval.py                   Legacy LongBench/RULER evaluator (no Stage 2 HSA)
├── tests/                    CPU/Metal kernel tests and model integration tests
├── docs/                     Pipeline notes and benchmark reports
├── Makefile                  Incremental runner and test builds
└── environment.yml           Python evaluation environment
```

Local inputs and outputs are separate from source: `llama.cpp/` is the patched upstream checkout, `build/` contains the compiled runner, `models/` and `data/` hold benchmark inputs, and `results/` holds predictions, logs, and summaries.

## Inference pipeline

| Phase | Dual-Metal execution |
|---|---|
| Prefill | The model processes the prompt with native attention. When eviction is needed, Metal captures the last layer's Q/K. Prefill logits produce the first output token. |
| Stage 1: reduce the prompt cache once | Metal scores prompt positions. The CPU makes one Top-K selection shared across layers and deletes the other KV entries. |
| Stage 2: select attention inputs at each decode step | The CPU adapter maps retained positions to cache slots. Inside `llama_decode()`, Metal scores candidates, selects Top-K per layer and KV group, and computes sparse attention. Unselected candidates remain available for the next step. |

`off` keeps the full cache and uses native decode attention. `stage1` evicts prompt KV but keeps native decode attention. `hsa` combines Stage 1 eviction with Stage 2 sparse decode.

## Run LongBench on Metal

From the repository root on a Metal-capable Mac, build `build/runner` and prepare the evaluation environment. The build requires Git, CMake, Make, Python 3, and a C++17 compiler.

```bash
conda env create -f environment.yml   # First setup only
conda activate rocketkv
bash scripts/stage2.sh build
bash scripts/download_model.sh       # Llama-3.2-3B-Instruct, Q8_0 weights
bash scripts/download_longbench.sh   # JSONL files in data/longbench/
```

**Run both stages on Metal** (4 tasks, 30 samples each):

```bash
python3 scripts/run_longbench_stage2.py --config configs/longbench.json \
  --mode hsa --stage1-backend metal --budget 1024 --output results/metal
```

The scoring script sets F16 KV, requests Metal model offload, and selects Metal HSA in `hsa` mode. Keep `--stage1-backend metal` explicit: Stage 1 scoring defaults to CPU.

| Experiment | Change to the command above |
|---|---|
| Compare full KV, Stage 1 only, and both stages | Replace `--mode hsa` with `--mode all` and use `--output results/metal-compare`. |
| Run every sample of all 16 configured tasks | Use `--config configs/longbench_full.json` and a new output directory. All 16 task files must be present. |

## Read and resume a run

- `status.json` shows progress; `summary.json` reports scores and timings per task and mode.
- Each sample has a scored checkpoint and runner log. Wall time includes model startup; `ttft_ms` and `decode_ms_per_forward` measure inference. KV size is a logical estimate.
- Re-run the same command to resume. The driver checks `manifest.json` against the inputs, settings, binary, driver hash, and Git commit. Use a new directory when these change.

For samples that evict prompt tokens, the dual-Metal run requires `stage1_capture=metal` and `stage1_capture_host_bytes=0`. For HSA it checks `stage2_layer_calls == model_layers * decode_forward_calls`. The first output token comes from prefill logits.

## Development

| Entry point | Purpose |
|---|---|
| `make runner` | Rebuild `build/runner` after C++ changes. |
| `ROCKET_TEST_METAL=1 bash scripts/stage2.sh test` | Run CPU and Metal checks, integration tests, and UBSan. |
| `bash scripts/stage2.sh run MODEL.gguf` | Compare `off`, `stage1`, and `hsa` on a demo batch (CPU by default). |
| `scripts/bench_stage1.py` | Measure Stage 1 capture and scoring separately from LongBench scoring. |
| `python3 eval.py configs/ruler.json` | Run the legacy evaluator; its `rocket-kv` mode uses Stage 1 only. |

Further reading: [RocketKV paper](https://arxiv.org/abs/2502.14051) · [HSA optimization log](HSA_OPTIMIZATION_LOG.md).
