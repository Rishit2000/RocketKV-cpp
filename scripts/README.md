# Safetensors conversion and LongBench evaluation

This tree follows [NVlabs/RocketKV](https://github.com/NVlabs/RocketKV) as closely as possible.
The LongBench bash launcher and `pipeline/` Python entry points are the upstream scripts,
modified only so generation calls the local C++ `runner` instead of InfLLM / Hugging Face generate.

## Prepare the original model

```bash
python3.12 -m venv .venv-longbench
.venv-longbench/bin/python -m pip install -r scripts/longbench/requirements.txt
.venv-longbench/bin/hf auth login
.venv-longbench/bin/python scripts/dataset_prep/hf_to_gguf.py --check-only
.venv-longbench/bin/python -m pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
.venv-longbench/bin/python scripts/dataset_prep/hf_to_gguf.py
make runner
```

## Prepare data

Upstream layout is `dataset/longbench/<task>` (HF `save_to_disk`). The helper also writes JSONL under `data/longbench/`:

```bash
.venv-longbench/bin/python scripts/dataset_prep/download_longbench.py
# or identical to upstream:
.venv-longbench/bin/python scripts/dataset_prep/download_longbench.py --hf-hub
```

## Run LongBench (same CLI as upstream)

```bash
bash scripts/longbench/llama3.2-3b-instruct.sh rocket results 1024
```

That script loops the 16 English tasks and calls:

```bash
python pipeline/inf_stream_llm/main.py \
  --method rocket \
  --token_budget 1024 \
  --exp_desc longbench_<dataset>_Llama-3.2-3B-Instruct_rocket_1024 \
  --pipeline_config_dir config/pipeline_config/longbench/Llama-3.2-3B-Instruct.json \
  --eval_config_dir config/eval_config/longbench/<dataset>.json \
  --output_folder_dir results/longbench/rocket/1024/Llama-3.2-3B-Instruct/<dataset>/
```

Optional environment overrides:

- `PYTHON` – interpreter (default `python`)
- `ROCKETKV_MODEL` – path to the F16 GGUF
- `ROCKETKV_RUNNER` – path to the `runner` binary
- `ROCKETKV_TOKENIZER` – tokenizer directory
- `ROCKETKV_CONTEXT_LEN` – prompt truncation length override (default: pipeline `model_max_len`)

Example 8K run:

```bash
ROCKETKV_CONTEXT_LEN=8192 PYTHON=.venv-longbench/bin/python \
  bash scripts/longbench/llama3.2-3b-instruct.sh rocket results_8k 1024
```

## Result layout (unchanged from upstream)

```
results/longbench/rocket/1024/Llama-3.2-3B-Instruct/TASK/
  input_config/input_pipeline_config.json
  input_config/input_eval_config.json
  pred/rocket/TASK_llama3_token_budget_1024.jsonl
  pred/rocket/result.json
  raw_results.json
  output_config.json
  exp.log
```

`input_pipeline_config.json` keeps the same fields as the reference RocketKV configs.
After all tasks finish, the launcher writes `longbench_result_summary.json` via
`visualization/longbench_results_summary/long_bench_tasks_summary.py`.

## What changed vs NVlabs/RocketKV

| Kept exact / nearly exact | Replaced for C++ runner |
| --- | --- |
| `scripts/longbench/*.sh` (model name only) | `pipeline/runner_backend.py` |
| `pipeline/main_utils.py`, `model_utils.py` | `pipeline/inf_stream_llm/inf_llm/*` loads tokenizer + runner |
| `pipeline/inf_stream_llm/eval_longbech.py` | `infllm_utils.get_pred` still builds prompts the same way, then batches through `runner` |
| `eval/longbench_utils/*`, `config/eval_config/*` | no InfLLM / CUDA generate path |
| `visualization/longbench_results_summary/*` | |
