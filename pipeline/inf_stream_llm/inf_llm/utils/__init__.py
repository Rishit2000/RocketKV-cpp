"""Minimal stubs replacing InfLLM attention patches; generation uses the C++ runner."""


def patch_hf(model, method, **kwargs):
    # RocketKV compression is applied inside the C++ runner; keep the HF object API.
    return model


def patch_model_center(model, method, **kwargs):
    return model


class GreedySearch:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._last_output = None

    def generate(self, input_ids=None, max_length=32, chunk_size=None, extra_end_token_ids=None, prompt=None):
        from pipeline.runner_backend import run_prompts, strip_bos

        if prompt is None:
            if hasattr(input_ids, "tolist"):
                ids = input_ids.tolist()
            else:
                ids = list(input_ids)
            prompt = self.tokenizer.decode(ids, skip_special_tokens=False)
        prompt = strip_bos(self.tokenizer, prompt)
        results = run_prompts(self.model, [prompt], max_length, self.model.pipeline_params)
        self._last_output = results[0]["generated"]
        return [self._last_output]

    def clear(self):
        self._last_output = None
