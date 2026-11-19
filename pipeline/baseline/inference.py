"""Baseline inference shim: same entry points as upstream, backed by the C++ runner."""
from pipeline.runner_backend import initialize_model_tokenizer, run_prompts, strip_bos


def batch_generate(input_ids, model, tokenizer, max_new_tokens):
    if hasattr(input_ids, "tolist"):
        ids = input_ids[0].tolist() if getattr(input_ids, "dim", lambda: 1)() > 1 else input_ids.tolist()
    else:
        ids = list(input_ids)
    prompt = strip_bos(tokenizer, tokenizer.decode(ids, skip_special_tokens=False))
    results = run_prompts(model, [prompt], max_new_tokens, model.pipeline_params)
    return [results[0]["generated"]]
