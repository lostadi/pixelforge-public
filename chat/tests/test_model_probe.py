"""Real pinned native metadata/tokenizer probes; no inference or NPU calls."""
from pathlib import Path
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

CHAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAT))
sys.path.insert(0, str(CHAT.parent / "tools/arm_npu"))
from model_info import probe_model, describe_model
from create_tiny_gguf import create
from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan
from test_gguf_loader import field, string

RUNNER = Path(os.environ.get("PIXELFORGE_CHAT_TEST_RUNNER", CHAT / "build/bin/pixelforge-chat"))


def encoder_fixture(path):
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "##s"]
    metadata = [("general.architecture", 8, "bert"), ("general.name", 8, "Synthetic encoder"),
                ("general.alignment", 4, 32), ("bert.context_length", 4, 256),
                ("bert.embedding_length", 4, 32), ("bert.feed_forward_length", 4, 64),
                ("bert.block_count", 4, 1), ("bert.attention.head_count", 4, 4),
                ("bert.attention.layer_norm_epsilon", 6, 1e-5),
                ("tokenizer.ggml.model", 8, "bert"), ("tokenizer.ggml.tokens", 9, (8, tokens)),
                ("tokenizer.ggml.bos_token_id", 4, 2), ("tokenizer.ggml.eos_token_id", 4, 3),
                ("tokenizer.ggml.unknown_token_id", 4, 1), ("tokenizer.ggml.padding_token_id", 4, 0),
                ("tokenizer.ggml.mask_token_id", 4, 4)]
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
    for name, kind, value in metadata:
        data += (string(name) + struct.pack("<If", kind, value)) if kind == 6 else field(name, kind, value)
    path.write_bytes(data + b"\0" * ((-len(data)) % 32))


@unittest.skipUnless(RUNNER.is_file(), "Build the native runner first")
class NativeModelProbeTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory(prefix="pixelforge-model-probe-")
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.model = self.root / "non-huihui-llama.gguf"
        create(self.model)

    def test_probe_uses_actual_llama_tokens_and_never_launches_bridge(self):
        before = hashlib.sha256(self.model.read_bytes()).hexdigest()
        marker = self.root / "broker-started"
        broker = self.root / "broker"
        broker.write_text("#!/bin/sh\ntouch " + str(marker) + "\nexit 99\n")
        broker.chmod(0o700)
        env = dict(os.environ, PIXELFORGE_GGML_BROKER=str(broker),
                   PIXELFORGE_GGML_PLAN=str(self.root / "nonexistent-plan"),
                   PIXELFORGE_GGML_NPU_TENSORS="invalid",
                   PIXELFORGE_CHAT_PREFILL_BATCH="invalid",
                   PIXELFORGE_CHAT_FLASH_ATTN="invalid",
                   PIXELFORGE_CHAT_PREFIX_CACHE_MIB="invalid",
                   PIXELFORGE_CHAT_PROMPT_MODE="invalid")
        result = subprocess.run([str(RUNNER), "--probe-model", str(self.model)],
                                env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        info = json.loads(result.stdout)
        self.assertEqual(info["schema"], "pixelforge.gguf.probe/v1")
        self.assertTrue(info["metadata_only"])
        self.assertEqual(info["prompt_modes"], ["chat-template", "raw-completion"])
        self.assertTrue(info["decoder_only"])
        self.assertEqual((info["architecture"], info["context_length"], info["vocabulary_size"]), ("llama", 256, 259))
        self.assertEqual((info["bos_token"], info["eos_token"]), ("<s>", "</s>"))
        self.assertIsNone(info["chat_template"])
        self.assertFalse(marker.exists())
        self.assertEqual(hashlib.sha256(self.model.read_bytes()).hexdigest(), before)

    def test_real_probe_and_metadata_describe_raw_completion(self):
        info = probe_model(RUNNER, self.model)
        with GGUFWeights(self.model) as weights:
            model = describe_model(weights, build_plan(weights), probe=info)
        self.assertEqual(model["architecture"], "llama")
        self.assertEqual(model["prompt_mode"], "raw-completion")
        self.assertEqual(model["template_variables"], {"bos_token": "<s>", "eos_token": "</s>"})

    def test_malformed_gguf_is_rejected_before_inference(self):
        bad = self.root / "malformed.gguf"
        bad.write_bytes(b"not a GGUF model")
        with self.assertRaisesRegex(ValueError, "eligibility probe failed"):
            probe_model(RUNNER, bad)

    def test_real_encoder_vocabulary_load_is_rejected_for_chat(self):
        encoder = self.root / "encoder.gguf"
        encoder_fixture(encoder)
        with self.assertRaisesRegex(ValueError, "decoder-only"):
            probe_model(RUNNER, encoder)


if __name__ == "__main__":
    unittest.main()
