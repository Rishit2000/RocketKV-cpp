import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("ROCKET_TEST_MODEL", ROOT / "models/stories260K.gguf")).resolve()
RUNNER = Path(os.environ.get("ROCKET_RUNNER", ROOT / "build" / "runner")).resolve()
PROMPT = (
    "Alice keeps the blue key in a wooden box. Bob has a red bicycle. "
    "The bird found a golden coin beside the river. "
) * 8 + "Where does Alice keep the key?"
FORCED = [10, 11, 12, 13, 14, 15]


class Stage2ScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def invoke(self, *args):
        return subprocess.run(
            ["bash", str(ROOT / "scripts/stage2.sh"), *args],
            cwd=self.temp.name, env=dict(os.environ, JOBS="invalid"),
            text=True, capture_output=True, timeout=20,
        )

    def test_help_never_builds(self):
        for args in [(), ("--help",), ("build", "--help"), ("test", "--help"), ("run", "--help")]:
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage: scripts/stage2.sh", result.stdout)

    def test_unknown_command_is_rejected(self):
        result = self.invoke("unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unknown command", result.stderr)

    def test_bad_arguments_are_rejected_before_build(self):
        for args in [("build", "extra"), ("test", "one", "two"), ("run",)]:
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Usage:", result.stderr)
                self.assertNotIn("JOBS", result.stderr)

    def test_missing_inputs_are_rejected_before_build(self):
        for args in [("test", "missing.gguf"), ("run", "missing.gguf"),
                     ("run", str(Path(__file__).resolve()), "missing.json")]:
            with self.subTest(args=args):
                result = self.invoke(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("not found", result.stderr)
                self.assertNotIn("JOBS", result.stderr)


class Stage2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL.is_file() or not RUNNER.is_file():
            raise RuntimeError("Run scripts/stage2.sh test to build and download the pinned test model")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.counter = 0

    def run_batch(self, samples, mode="hsa", extra=(), error=None):
        self.counter += 1
        path = Path(self.temp.name)
        source = path / f"input-{self.counter}.json"
        target = path / f"output-{self.counter}.json"
        source.write_text(json.dumps(samples), encoding="utf-8")
        command = [
            str(RUNNER), "--model", str(MODEL), "--batch", str(source),
            "--output", str(target), "--n-ctx", "512", "--gpu-layers", "0",
            "--context-len", "256",
            "--threads", "2", "--budget", "16", "--window-size", "4",
            "--kernel-size", "3", "--trace-logits",
        ]
        if mode is not None:
            command.extend(["--rocket-mode", mode])
        result = subprocess.run(command + list(extra), text=True, capture_output=True, timeout=120)
        if error is not None:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            self.assertFalse(target.exists(), "A failed batch must not publish successful results")
            return None
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = json.loads(target.read_text())
        self.assertEqual(len(outputs), len(samples))
        for sample, output in zip(samples, outputs):
            self.assertEqual(output["sample_id"], sample["sample_id"])
            self.assertEqual(output["status"], "ok")
        return outputs

    def sample(self, prompt=PROMPT, tokens=FORCED, sample_id=1):
        return {
            "sample_id": sample_id, "prompt": prompt,
            "max_tokens": len(tokens), "forced_tokens": list(tokens),
        }

    def assert_close(self, left, right, tolerance=2e-4):
        self.assertEqual(len(left), len(right))
        maximum = 0.0
        for a, b in zip(left, right):
            if isinstance(a, list):
                maximum = max(maximum, self.assert_close(a, b, tolerance))
            else:
                self.assertTrue(math.isfinite(a) and math.isfinite(b))
                maximum = max(maximum, abs(a - b))
                self.assertLessEqual(abs(a - b), tolerance + tolerance * abs(b))
        return maximum

    def assert_hsa_calls(self, result, steps, backend="cpu-reference"):
        metrics = result["metrics"]
        self.assertEqual(metrics["effective_mode"], "hsa")
        self.assertEqual(metrics["stage2_backend"], backend)
        self.assertEqual(metrics["decode_forward_calls"], steps)
        self.assertEqual(metrics["stage2_layer_calls"], metrics["model_layers"] * steps)
        self.assertEqual(metrics["stage2_calls_per_layer"], [steps] * metrics["model_layers"])
        self.assertEqual(len(result["attention"]), steps)

    def test_full_selection_matches_dense_layers_and_logits(self):
        sample = self.sample()
        native = self.run_batch([sample], "stage1", ["--reference-f32"])[0]
        full = self.run_batch([sample], extra=["--hsa-full"])[0]
        self.assertEqual(native["kept_positions"], full["kept_positions"])
        kept = full["kept_positions"]
        self.assertLess(len(kept), full["metrics"]["prompt_tokens"])
        self.assertTrue(any(b > a + 1 for a, b in zip(kept, kept[1:])))
        self.assertEqual(full["token_ids"], FORCED)
        attention_error = self.assert_close(full["attention"], native["attention"])
        logits_error = self.assert_close(full["logits"], native["logits"])
        self.assert_hsa_calls(full, len(FORCED) - 1)
        metrics = full["metrics"]
        self.assertEqual(metrics["stage2_candidate_min"], len(kept) + 1)
        self.assertEqual(metrics["stage2_candidate_max"], len(kept) + len(FORCED) - 1)
        self.assertEqual(metrics["stage2_selected_min"], metrics["stage2_candidate_min"])
        print(f"\nFull-selection max errors: attention={attention_error:.3g}, logits={logits_error:.3g}")

    def test_sparse_outputs_reach_logits(self):
        sample = self.sample()
        sparse = self.run_batch([sample])[0]
        full = self.run_batch([sample], extra=["--hsa-full"])[0]
        self.assert_hsa_calls(sparse, len(FORCED) - 1)
        metrics = sparse["metrics"]
        self.assertEqual(metrics["stage2_selected_min"], metrics["stage2_k"])
        self.assertEqual(metrics["stage2_selected_max"], metrics["stage2_k"])
        self.assertLess(metrics["stage2_selected_max"], metrics["stage2_candidate_min"])
        self.assert_close(sparse["logits"][0], full["logits"][0], 0)
        difference = max(abs(a - b) for a, b in zip(sparse["logits"][1], full["logits"][1]))
        self.assertGreater(difference, 1e-5, "Sparse output is not connected to the model")

    def test_single_token_prefill_tail_microbatch_and_zero_forwards(self):
        sample = self.sample(tokens=FORCED[:3])
        tail = self.run_batch([sample], extra=["--context-len", "9", "--ubatch", "8"])[0]
        self.assertEqual(tail["metrics"]["prompt_tokens"], 9)
        self.assert_hsa_calls(tail, 2)
        short = self.run_batch([sample], extra=["--context-len", "1", "--ubatch", "1"])[0]
        self.assertEqual(short["kept_positions"], [0])
        self.assert_hsa_calls(short, 2)
        single = self.run_batch([self.sample(tokens=FORCED[:1])],
                                extra=["--context-len", "1"])[0]
        self.assert_hsa_calls(single, 0)
        self.assertIsNone(single["metrics"]["decode_ms_per_forward"])

    def test_reset_between_requests_with_different_budgets(self):
        first = self.sample()
        second = self.sample(prompt="A little bird.", tokens=FORCED[:3], sample_id=2)
        together = self.run_batch([first, second])
        alone = self.run_batch([second])[0]
        self.assert_hsa_calls(together[0], 5)
        self.assert_hsa_calls(together[1], 2)
        self.assertEqual(together[1]["kept_positions"], alone["kept_positions"])
        self.assert_close(together[1]["logits"], alone["logits"], 0)
        self.assert_close(together[1]["attention"], alone["attention"], 0)

    def test_legacy_modes_preserve_native_paths(self):
        sample = self.sample(tokens=FORCED[:3])
        for mode, legacy in [("off", "0"), ("stage1", "1")]:
            with self.subTest(mode=mode):
                modern = self.run_batch([sample], mode)[0]
                old = self.run_batch([sample], None, ["--rocket", legacy])[0]
                self.assertEqual(old["metrics"]["stage2_layer_calls"], 0)
                self.assertEqual(old["metrics"]["effective_mode"], mode)
                self.assert_close(old["logits"], modern["logits"], 0)
                self.assert_close(old["attention"], modern["attention"], 0)

    def test_invalid_requests_fail_explicitly(self):
        sample = self.sample(tokens=FORCED[:2])
        cases = [
            (["--kv-type", "q8_0"], "requires F16"),
            (["--stage2-backend", "unknown"], "Unknown stage2 backend"),
            (["--rocket", "1"], "Do not combine"),
            (["--budget", "0"], "requires F16"),
            (["--n-ctx", "8", "--context-len", "0"], "exceeds --n-ctx"),
            (["--window-size", "0"], "Invalid budget"),
            (["--unknown-option"], "Unknown argument"),
        ]
        for options, message in cases:
            with self.subTest(options=options):
                self.run_batch([sample], extra=options, error=message)
        self.run_batch([dict(sample, max_tokens=0)], error="Invalid generation length")
        self.run_batch([dict(sample, forced_tokens=[-1, 10])], error="Invalid or EOG forced token")

    @unittest.skipUnless(os.environ.get("ROCKET_TEST_METAL") == "1", "Set ROCKET_TEST_METAL=1 on a Metal GPU build")
    def test_metal_full_selection_matches_cpu(self):
        # Keep every prompt position so small CPU/Metal prefill differences cannot
        # change the Stage 1 eviction boundary used for numerical comparison.
        sample = self.sample()
        options = ["--hsa-full", "--window-size", "512"]
        cpu = self.run_batch([sample], extra=options)[0]
        metal = self.run_batch([sample], extra=options + ["--gpu-layers", "99"])[0]
        self.assertEqual(cpu["kept_positions"], metal["kept_positions"])
        self.assertEqual(metal["metrics"]["model_backend"], "metal")
        self.assertTrue(metal["metrics"]["model_gpu_device"])
        self.assertEqual(metal["metrics"]["requested_gpu_layers"], 99)
        self.assert_hsa_calls(metal, len(FORCED) - 1)
        # Quantized model matmuls differ between CPU and Metal even without
        # Rocket. Bound hybrid numerical drift against that native baseline.
        native_cpu = self.run_batch([sample], "off")[0]
        native_metal = self.run_batch([sample], "off", ["--gpu-layers", "99"])[0]

        def flatten(values):
            for value in values:
                if isinstance(value, list):
                    yield from flatten(value)
                else:
                    yield value

        def relative_error(left, right):
            left, right = list(flatten(left)), list(flatten(right))
            self.assertEqual(len(left), len(right))
            self.assertTrue(all(math.isfinite(x) for x in left + right))
            self.assertLess(max(abs(a - b) for a, b in zip(left, right)), 0.05)
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) /
                             max(sum(b * b for b in right), 1e-12))

        for field in ("attention", "logits"):
            baseline = relative_error(native_metal[field], native_cpu[field])
            hybrid = relative_error(metal[field], cpu[field])
            self.assertLessEqual(hybrid, max(1e-4, baseline * 1.5))

    @unittest.skipUnless(os.environ.get("ROCKET_TEST_METAL") == "1", "Set ROCKET_TEST_METAL=1 on a Metal GPU build")
    def test_metal_sparse_batch_state_isolation(self):
        samples = [self.sample(), self.sample(prompt="The bird found a coin.", tokens=FORCED[:3], sample_id=2)]
        options = ["--gpu-layers", "99"]
        together = self.run_batch(samples, extra=options)
        for sample, result in zip(samples, together):
            alone = self.run_batch([sample], extra=options)[0]
            self.assert_hsa_calls(result, len(sample["forced_tokens"]) - 1)
            self.assertEqual(result["kept_positions"], alone["kept_positions"])
            self.assert_close(result["logits"], alone["logits"], 2e-4)
        m = together[0]["metrics"]
        self.assertLess(m["stage2_selected_max"], m["stage2_candidate_min"])

    @unittest.skipUnless(os.environ.get("ROCKET_TEST_METAL") == "1", "Set ROCKET_TEST_METAL=1 on macOS")
    def test_metal_kernel_matches_cpu_reference(self):
        for full in (False, True):
            with self.subTest(full=full):
                options = ["--hsa-full"] if full else []
                sample = self.sample()
                cpu = self.run_batch([sample], extra=options)[0]
                gpu = self.run_batch([sample], extra=options + ["--stage2-backend", "metal"])[0]
                self.assertEqual(cpu["kept_positions"], gpu["kept_positions"])
                self.assert_hsa_calls(gpu, len(FORCED) - 1, "metal")
                self.assert_close(gpu["attention"], cpu["attention"], 2e-4)
                self.assert_close(gpu["logits"], cpu["logits"], 2e-4)

    @unittest.skipUnless(os.environ.get("ROCKET_TEST_METAL") == "1", "Set ROCKET_TEST_METAL=1 on macOS")
    def test_metal_kernel_with_model_offload_and_reset(self):
        samples = [self.sample(), self.sample(prompt="The bird found a coin.", tokens=FORCED[:3], sample_id=2)]
        options = ["--stage2-backend", "metal", "--gpu-layers", "99"]
        together = self.run_batch(samples, extra=options)
        for sample, result in zip(samples, together):
            alone = self.run_batch([sample], extra=options)[0]
            self.assert_hsa_calls(result, len(sample["forced_tokens"]) - 1, "metal")
            self.assertEqual(result["metrics"]["model_backend"], "metal")
            self.assertEqual(result["kept_positions"], alone["kept_positions"])
            self.assert_close(result["logits"], alone["logits"], 2e-4)


if __name__ == "__main__":
    unittest.main()
