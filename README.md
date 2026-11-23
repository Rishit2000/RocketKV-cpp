# RocketKV-cpp

RocketKV reduces the KV cache used by single-sequence Llama inference in a pinned, patched `llama.cpp`. **Stage 1** evicts prompt tokens after prefill. **Stage 2** uses hybrid sparse attention (HSA) during decode: each layer and KV group selects from the retained candidates at each step. HSA requires F16 KV; model weights can be quantized.

[Repository layout](#repository-layout) · [Inference pipeline](#inference-pipeline) · [Run LongBench and RULER](#run-longbench-and-ruler-on-metal) · [Results and resume](#read-and-resume-a-run) · [Development](#development)

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
├── scripts/                  Shell entry points only
│   ├── stage2.sh             Build, test, and demo-run the runner
│   ├── data/                 Download the model and LongBench; generate official RULER inputs
│   ├── eval/                 One script per benchmark experiment (longbench.sh, ruler.sh)
│   └── bench/                Timing benchmarks
├── configs/                  Model paths, task lists, sample counts, and context limits
├── eval_suite/               Dataset loaders, prompt protocol, scoring, and the drivers the scripts call
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

## Run LongBench and RULER on Metal

From the repository root on a Metal-capable Mac, build `build/runner` and prepare the inputs. The build requires Git, CMake, Make, Python 3, and a C++17 compiler.

```bash
conda env create -f environment.yml    # First setup only
conda activate rocketkv
bash scripts/stage2.sh build
bash scripts/data/download_model.sh     # Q8_0 GGUF weights + HF tokenizer in models/
bash scripts/data/download_longbench.sh # official JSONL files in data/longbench/
bash scripts/data/prepare_ruler.sh      # official RULER inputs in data/ruler/<len>/<task>/
```

`prepare_ruler.sh` runs the NVIDIA RULER generator vendored in [NVlabs/RocketKV](https://github.com/NVlabs/RocketKV) (`eval/ruler_utils`), cloned to `../RocketKV` unless `ROCKETKV_DIR` points elsewhere.

| Experiment | Command | Inputs |
|---|---|---|
| LongBench | `bash scripts/eval/longbench.sh results/longbench_32k` | 16 English tasks, all samples, middle-truncated to 32k tokens |
| RULER | `bash scripts/eval/ruler.sh results/ruler_5task` | `niah_single_2`, `niah_multikey_2`, `vt`, `fwe`, `qa_1`; 10 samples each at 2k/4k/8k/16k/32k |

Each command runs full KV once, then `stage1` and `hsa` for every budget in the config (512, 1024, 2048), with Metal Stage 1 scoring and Metal HSA. Add `--mode hsa`, `--budget 1024`, or `--dry-run` (load and format every input, print the plan, run nothing) after the output directory.

Prompts and scores follow the RocketKV paper code. LongBench prompts are middle-truncated with the HF tokenizer and wrapped in the llama3 chat template, except `trec`, `triviaqa`, `samsum`, `lcc`, and `repobench-p`; scoring is LongBench's `scorer()`. RULER inputs are used as generated (llama3 template and answer prefix included); every task generates up to 128 tokens and is scored with RULER's `string_match_all` (`string_match_part` for `qa_*`). As in RocketKV, chat-formatted prompts start with two BOS tokens. Differences from the paper are recorded in each config's `_comment`.

## Read and resume a run

- `status.json` shows progress; `summary.json` reports scores and timings per run (`off`, `stage1_b1024`, ...) and task (`<len>/<task>` for RULER).
- Each sample has a scored checkpoint and runner log. Wall time includes model startup; `ttft_ms` and `decode_ms_per_forward` measure inference. KV size is a logical estimate.
- Re-run the same command to resume. The driver checks `manifest.json` against the inputs, settings, binary, driver hash, and Git commit. Use a new directory when these change.

For samples that evict prompt tokens, the dual-Metal run requires `stage1_capture=metal` and `stage1_capture_host_bytes=0`. For HSA it checks `stage2_layer_calls == model_layers * decode_forward_calls`. The first output token comes from prefill logits.

## Development

| Entry point | Purpose |
|---|---|
| `make runner` | Rebuild `build/runner` after C++ changes. |
| `ROCKET_TEST_METAL=1 bash scripts/stage2.sh test` | Run CPU and Metal checks, integration tests, and UBSan. |
| `bash scripts/stage2.sh run MODEL.gguf` | Compare `off`, `stage1`, and `hsa` on a demo batch (CPU by default). |
| `bash scripts/bench/stage1.sh` | Measure Stage 1 capture and scoring separately from task scoring. |
| `python3 eval.py configs/longbench.json` | Run the legacy evaluator; its `rocket-kv` mode uses Stage 1 only. RULER configs need `data/ruler/<len>/` from `prepare_ruler.sh`. |

Further reading: [RocketKV paper](https://arxiv.org/abs/2502.14051) · [HSA optimization log](HSA_OPTIMIZATION_LOG.md).
