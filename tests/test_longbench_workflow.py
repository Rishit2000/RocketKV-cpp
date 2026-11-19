"""Validate runner wiring through the upstream RocketKV LongBench scripts."""
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline" / "inf_stream_llm"))

from eval_suite.runner import execute_runner_batch
import pipeline.runner_backend as runner_backend
import infllm_utils


class WorkflowTests(unittest.TestCase):
    def test_both_stages_override_environment(self):
        commands = []
        environments = []

        class Process:
            def __init__(self, command, **kwargs):
                commands.append(command)
                environments.append(kwargs["env"])
                self.stderr = io.StringIO()
                self.returncode = 0
                output = Path(command[command.index("--output") + 1])
                output.write_text('[{"generated": "42"}]')

            def wait(self):
                return 0

        with patch("eval_suite.runner.subprocess.Popen", Process), patch.dict("os.environ", {"ROCKET_HSA": "0", "ROCKET_HSA_SKIP": "12"}):
            for enabled in (True, False):
                result = execute_runner_batch("runner", "model", "f16", [], enable_rocket=enabled)
                self.assertEqual(result[0]["generated"], "42")
        for command, enabled in zip(commands, (True, False)):
            self.assertEqual(command[command.index("--hsa") + 1], "1" if enabled else "0")
            self.assertEqual(command[command.index("--rocket") + 1], "1" if enabled else "0")
            self.assertEqual(command[command.index("--hsa-skip") + 1], "0")
        for environment in environments:
            self.assertNotIn("ROCKET_HSA", environment)
            self.assertNotIn("ROCKET_HSA_SKIP", environment)

    def test_get_pred_writes_upstream_fields_via_runner(self):
        class FakeTokenizer:
            bos_token = "<bos>"
            pad_token = "</s>"
            eos_token = "</s>"

            def __call__(self, prompt, truncation=False, add_special_tokens=True):
                return types.SimpleNamespace(input_ids=list(range(10)))

            def decode(self, ids, skip_special_tokens=True):
                return "decoded"

            def encode(self, text, add_special_tokens=False):
                return [1]

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "<bos>chat:" + messages[0]["content"]

        tokenizer = FakeTokenizer()
        model = runner_backend.RunnerModel(
            ROOT / "models" / "x.gguf",
            ROOT / "runner",
            {"method": "rocket", "token_budget": 1024, "model_max_len": 8192, "chat_template": "llama3", "window_size": 32, "kernel_size": 63},
        )

        data = [{"context": "C", "input": "Q?", "answers": ["42"], "all_classes": None, "length": 12}]
        eval_params = {
            "dataset": "hotpotqa",
            "instruction": "{context}\n{input}",
            "max_new_tokens": 32,
        }
        pipeline_params = {
            "method": "rocket",
            "token_budget": 1024,
            "model_max_len": 8192,
            "chat_template": "llama3",
            "truncation_mode": "middle",
        }

        with patch.object(infllm_utils, "run_prompts", return_value=[{"generated": "42", "metrics": {"prompt_tokens": 10}}]) as mocked:
            preds = infllm_utils.get_pred(model, tokenizer, data, eval_params, pipeline_params)

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(preds[0]["pred"], "42")
        self.assertEqual(preds[0]["answers"], ["42"])
        self.assertEqual(preds[0]["length"], 12)
        self.assertIn("token_length", preds[0])
        self.assertEqual(pipeline_params["kernel_size"], 63)
        self.assertEqual(pipeline_params["window_size"], 32)
        self.assertIn("prompt_budget", pipeline_params)
        self.assertIn("compression_ratio", pipeline_params)

    def test_shell_and_pipeline_entrypoints_exist(self):
        self.assertTrue((ROOT / "scripts/longbench/llama3.2-3b-instruct.sh").is_file())
        self.assertTrue((ROOT / "pipeline/inf_stream_llm/main.py").is_file())
        self.assertTrue((ROOT / "pipeline/inf_stream_llm/eval_longbech.py").is_file())
        self.assertTrue((ROOT / "config/pipeline_config/longbench/Llama-3.2-3B-Instruct.json").is_file())
        self.assertTrue((ROOT / "config/eval_config/longbench/hotpotqa.json").is_file())
        script = (ROOT / "scripts/longbench/llama3.2-3b-instruct.sh").read_text()
        self.assertIn('model="Llama-3.2-3B-Instruct"', script)
        self.assertIn("pipeline/inf_stream_llm/main.py", script)

    def test_register_writes_input_pipeline_fields(self):
        import pipeline.main_utils as main_utils

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pipeline_config = directory / "pipeline.json"
            pipeline_config.write_text(json.dumps({
                "pipeline_params": {
                    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
                    "tokenizer_name": "meta-llama/Llama-3.2-3B-Instruct",
                    "chat_template": "llama3",
                    "model_max_len": 127500,
                    "fattn": True,
                    "use_flash_attn": True,
                    "truncation_mode": "middle",
                    "batch_size": 1,
                    "out_of_max_len_allowed": True,
                    "base": 500000,
                    "rope_theta_factor": 1.0,
                }
            }))
            eval_config = directory / "eval.json"
            eval_config.write_text((ROOT / "config/eval_config/longbench/hotpotqa.json").read_text())
            output = directory / "out"
            output.mkdir()
            args = types.SimpleNamespace(
                exp_desc="longbench_hotpotqa_Llama-3.2-3B-Instruct_rocket_1024",
                pipeline_config_dir=str(pipeline_config),
                eval_config_dir=str(eval_config),
                output_folder_dir=str(output) + "/",
                job_post_via="terminal",
                method="rocket",
                token_budget=1024,
                scdq_mode=False,
                dataset=None,
                max_seq_length=4000,
            )
            main_utils.set_logger(args.output_folder_dir, args)
            config = main_utils.register_args_and_configs(args)
            saved = json.loads((output / "input_config" / "input_pipeline_config.json").read_text())
            ref = json.loads((ROOT / "reference-results/Llama-3.2-3B-Instruct/hotpotqa/input_config/input_pipeline_config.json").read_text())
            self.assertEqual(saved["pipeline_params"], ref["pipeline_params"])
            self.assertEqual(config["pipeline_params"]["method"], "rocket")


if __name__ == "__main__":
    unittest.main()
