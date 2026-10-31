"""Model-native metadata and bounded eligibility-probe contracts."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_info import describe_model, probe_model, PROBE_SCHEMA


class ModelInfoTests(unittest.TestCase):
    def weights(self, *, template=True):
        fields = {"general.architecture": "llama", "general.name": "Local plain Llama",
                  "llama.context_length": 256, "tokenizer.ggml.tokens": ["word", "<START>", "<STOP>"],
                  "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2}
        if template:
            fields["tokenizer.chat_template"] = "{{ bos_token }}{{ messages[0].content }}{{ eos_token }}"
        return SimpleNamespace(path=Path("/models/plain-model.gguf"), fields=fields)

    def result(self, *, template=None, **updates):
        value = dict(schema=PROBE_SCHEMA, architecture="llama", decoder_only=True,
                     metadata_only=True, context_length=256, vocabulary_size=3,
                     chat_template=template, bos_token="<START>", eos_token="<STOP>")
        value.update(updates)
        return value

    def test_non_huihui_metadata_uses_own_tokens_and_template(self):
        weights = self.weights()
        info = describe_model(weights, {"architecture": "llama"})
        self.assertEqual(info["name"], "Local plain Llama")
        self.assertEqual(info["prompt_mode"], "chat-template")
        self.assertEqual(info["template_variables"], {"bos_token": "<START>", "eos_token": "<STOP>"})
        del weights.fields["general.name"]
        self.assertEqual(describe_model(weights, {"architecture": "llama"})["name"], "plain-model")

    def test_absent_template_is_explicit_raw_completion(self):
        info = describe_model(self.weights(template=False), {"architecture": "llama"})
        self.assertEqual(info["prompt_mode"], "raw-completion")
        self.assertIsNone(info["chat_template"])

    def test_malformed_specified_template_is_rejected(self):
        for template in (None, 123, "", "{% if %}"):
            weights = self.weights()
            weights.fields["tokenizer.chat_template"] = template
            with self.subTest(template=template), self.assertRaisesRegex(ValueError, "template is malformed"):
                describe_model(weights, {"architecture": "llama"})

    def test_native_tokenizer_defaults_are_authoritative_when_ids_are_absent(self):
        weights = self.weights(template=False)
        del weights.fields["tokenizer.ggml.bos_token_id"]
        del weights.fields["tokenizer.ggml.eos_token_id"]
        inferred = describe_model(weights, {"architecture": "llama"})
        self.assertEqual(inferred["template_variables"], {"bos_token": "", "eos_token": ""})
        actual = describe_model(weights, {"architecture": "llama"}, probe=self.result())
        self.assertEqual(actual["template_variables"], {"bos_token": "<START>", "eos_token": "<STOP>"})

    def test_invalid_gguf_token_ids_and_inconsistent_probe_are_rejected(self):
        weights = self.weights(template=False)
        weights.fields["tokenizer.ggml.bos_token_id"] = 3
        with self.assertRaisesRegex(ValueError, "token ID"):
            describe_model(weights, {"architecture": "llama"})
        for changes in ({"architecture": "qwen35"}, {"context_length": 512}, {"vocabulary_size": 5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                describe_model(self.weights(template=False), {"architecture": "llama"}, probe=self.result(**changes))
        with self.assertRaisesRegex(ValueError, "chat template differs"):
            describe_model(self.weights(), {"architecture": "llama"}, probe=self.result())

    def test_probe_has_literal_argv_no_stdin_and_no_inherited_bridge_settings(self):
        completed = SimpleNamespace(returncode=0, stdout=json.dumps(self.result()), stderr="")
        with patch.dict(os.environ, {"PIXELFORGE_GGML_BROKER": "/unwanted/broker", "PIXELFORGE_GGML_OTHER": "x"}), \
                patch("model_info.subprocess.run", return_value=completed) as run:
            self.assertEqual(probe_model("/runner with spaces", "/model with spaces.gguf"), self.result())
        args, options = run.call_args
        self.assertEqual(args[0], ["/runner with spaces", "--probe-model", "/model with spaces.gguf"])
        self.assertFalse(any(key.startswith("PIXELFORGE_GGML_") for key in options["env"]))
        self.assertEqual(options["stdin"], subprocess.DEVNULL)
        self.assertNotIn("shell", options)

    def test_probe_rejects_non_decoder_bad_schema_and_invalid_metadata(self):
        for changes in ({"decoder_only": False}, {"schema": "other"}, {"metadata_only": False},
                        {"context_length": True}, {"vocabulary_size": 0}, {"bos_token": None},
                        {"prompt_modes": "raw-completion"}, {"prompt_modes": ["raw-completion", "raw-completion"]}):
            completed = SimpleNamespace(returncode=0, stdout=json.dumps(self.result(**changes)), stderr="")
            with self.subTest(changes=changes), patch("model_info.subprocess.run", return_value=completed), self.assertRaises(ValueError):
                probe_model("/runner", "/model")
        with patch("model_info.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="not json", stderr="")):
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                probe_model("/runner", "/model")

    def test_skipped_token_arrays_are_resolved_by_the_native_probe(self):
        weights = self.weights(template=False)
        del weights.fields["tokenizer.ggml.tokens"]
        info = describe_model(weights, {"architecture": "llama"}, probe=self.result())
        self.assertEqual(info["template_variables"], {"bos_token": "<START>", "eos_token": "<STOP>"})
        self.assertEqual(info["vocabulary_size"], 3)

    def test_model_name_and_special_tokens_reject_control_injection(self):
        for name in ("x" * 257, "model\nname", "model\0name"):
            weights = self.weights()
            weights.fields["general.name"] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "model name"):
                describe_model(weights, {"architecture": "llama"})
        weights = self.weights()
        weights.fields["tokenizer.ggml.tokens"][1] = "<START>\0"
        with self.assertRaisesRegex(ValueError, "contains NUL"):
            describe_model(weights, {"architecture": "llama"})

    def test_probe_failure_and_timeout_are_bounded_errors(self):
        with patch("model_info.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout="", stderr="bad GGUF")):
            with self.assertRaisesRegex(ValueError, "bad GGUF"):
                probe_model("/runner", "/model")
        with patch("model_info.subprocess.run", side_effect=subprocess.TimeoutExpired("probe", 1)):
            with self.assertRaisesRegex(ValueError, "probe failed"):
                probe_model("/runner", "/model")


if __name__ == "__main__":
    unittest.main()
