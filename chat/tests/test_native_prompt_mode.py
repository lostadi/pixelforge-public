"""Native raw/completion token policy; synthetic model, no NPU calls."""
from pathlib import Path
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

CHAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAT.parent / "tools/arm_npu"))
from create_tiny_gguf import create

RUNNER = Path(os.environ.get("PIXELFORGE_CHAT_TEST_RUNNER", CHAT / "build/bin/pixelforge-chat"))


@unittest.skipUnless(RUNNER.is_file(), "Build the native runner first")
class NativePromptModeTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory(prefix="pixelforge-native-prompt-")
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.model = self.root / "tiny.gguf"
        create(self.model)

    def environment(self, mode):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("PIXELFORGE_GGML_") and not key.startswith("PIXELFORGE_CHAT_")}
        if mode is not None:
            env["PIXELFORGE_CHAT_PROMPT_MODE"] = mode
        return env

    def run_prompt(self, prompt, mode=None, *, extra=None, done=True):
        request = dict(id="token-policy", prompt=prompt, max_tokens=2,
                       context_size=256, prefill_cpu=True)
        request.update(extra or {})
        process = subprocess.run([str(RUNNER), str(self.model), "1"],
                                 input=json.dumps(request) + "\n", text=True, capture_output=True,
                                 env=self.environment(mode), timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        frames = [json.loads(line) for line in process.stdout.splitlines()]
        self.assertEqual(frames[0]["type"], "ready")
        if not done:
            return frames
        result = next(frame for frame in frames if frame["type"] == "done")
        self.assertEqual(result["npu_weight_matmul_operations"], 0)
        return frames[0], result

    def test_raw_native_bos_matches_formatted_once_and_deduplicates_explicit_bos(self):
        ready, formatted = self.run_prompt("<s>Hello", "chat-template")
        self.assertEqual(ready["prompt_mode"], "chat-template")
        _, plain_formatted = self.run_prompt("Hello")
        ready, raw = self.run_prompt("Hello", "raw-completion")
        _, explicit = self.run_prompt("<s>Hello", "raw-completion")
        self.assertEqual(ready["prompt_mode"], "raw-completion")
        self.assertEqual(raw["prompt_mode"], "raw-completion")
        self.assertEqual(raw["prompt_tokens"], plain_formatted["prompt_tokens"] + 1)
        self.assertEqual(raw["prompt_tokens"], formatted["prompt_tokens"])
        self.assertEqual(explicit["prompt_tokens"], formatted["prompt_tokens"])
        self.assertEqual(raw["generated_tokens"], formatted["generated_tokens"])
        self.assertEqual(explicit["generated_tokens"], formatted["generated_tokens"])

    def test_raw_native_eos_policy_deduplicates_explicit_edge_tokens(self):
        key = b"tokenizer.ggml.add_eos_token"
        before = struct.pack("<Q", len(key)) + key + struct.pack("<I", 7) + b"\0"
        data = self.model.read_bytes()
        self.assertEqual(data.count(before), 1)
        self.model.write_bytes(data.replace(before, before[:-1] + b"\1", 1))
        _, formatted = self.run_prompt("<s>Hello</s>", "chat-template")
        _, raw = self.run_prompt("Hello", "raw-completion")
        _, explicit = self.run_prompt("<s>Hello</s>", "raw-completion")
        self.assertEqual(raw["prompt_tokens"], formatted["prompt_tokens"])
        self.assertEqual(explicit["prompt_tokens"], formatted["prompt_tokens"])
        self.assertEqual(raw["generated_tokens"], formatted["generated_tokens"])
        self.assertEqual(explicit["generated_tokens"], formatted["generated_tokens"])

    def test_context_limit_is_checked_after_automatic_duplicate_removal(self):
        prompt = "<s>" + "H" * 253
        _, formatted = self.run_prompt(prompt, "chat-template")
        _, raw = self.run_prompt(prompt, "raw-completion")
        self.assertEqual(formatted["prompt_tokens"], 254)
        self.assertEqual(raw["prompt_tokens"], 254)
        self.assertEqual(raw["generated_tokens"], formatted["generated_tokens"])

    def test_unknown_mode_fails_before_model_or_context_loading(self):
        missing = self.root / "missing.gguf"
        process = subprocess.run([str(RUNNER), str(missing), "1"], stdin=subprocess.DEVNULL,
                                 text=True, capture_output=True,
                                 env=self.environment("unsupported"), timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("PIXELFORGE_CHAT_PROMPT_MODE", process.stderr)
        self.assertNotIn("failed to open", process.stderr)
        self.assertFalse(process.stdout)

    def test_request_cannot_override_the_trusted_prompt_mode(self):
        frames = self.run_prompt("Hello", "chat-template",
                                 extra={"prompt_mode": "raw-completion"}, done=False)
        self.assertEqual(frames[0]["prompt_mode"], "chat-template")
        self.assertTrue(any(frame["type"] == "error" and "Unknown request field" in frame["error"] for frame in frames))
        self.assertFalse(any(frame["type"] == "done" for frame in frames))


if __name__ == "__main__":
    unittest.main()
