"""Inspect a GGUF and the pinned native loader without executing model work."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

PROBE_SCHEMA = "pixelforge.gguf.probe/v1"


def _positive(value, name, *, optional=False):
    if optional and value is None:
        return None
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _template(value, *, present):
    if not present:
        return None
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("The specified GGUF chat template is malformed")
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    try:
        ImmutableSandboxedEnvironment().parse(value)
    except Exception as error:
        raise ValueError("The specified GGUF chat template is malformed: " + str(error)[:300]) from error
    return value


def _validate_probe(value):
    if not isinstance(value, dict) or value.get("schema") != PROBE_SCHEMA:
        raise ValueError("Native model probe returned an unsupported schema")
    architecture = value.get("architecture")
    if not isinstance(architecture, str) or not architecture.strip() or "\0" in architecture:
        raise ValueError("Native model probe returned no valid architecture")
    if value.get("decoder_only") is not True:
        raise ValueError("The native engine requires a decoder-only GGUF")
    if value.get("metadata_only") is not True:
        raise ValueError("Native model probe did not attest metadata-only loading")
    _positive(value.get("context_length"), "Native context length")
    _positive(value.get("vocabulary_size"), "Native vocabulary size")
    for name in ("bos_token", "eos_token"):
        if not isinstance(value.get(name), str) or "\0" in value[name]:
            raise ValueError("Native model probe returned an invalid " + name)
    modes = value.get("prompt_modes")
    if modes is not None and (not isinstance(modes, list) or not modes or
                              any(not isinstance(mode, str) or mode not in ("chat-template", "raw-completion") for mode in modes)
                              or len(set(modes)) != len(modes)):
        raise ValueError("Native model probe returned invalid prompt mode capabilities")
    template = value.get("chat_template")
    _template(template, present=template is not None)
    return value


def probe_model(binary, model, *, timeout=60):
    """Ask the native engine to load metadata/vocabulary only, never a context.

    The original model is inspected in place. All inherited bridge settings
    are removed, and the native probe also clears those settings defensively.
    A successful probe establishes loader/tokenizer eligibility, not hardware
    execution, numerical quality, or supported NPU bank coverage.
    """
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("PIXELFORGE_GGML_")}
    try:
        result = subprocess.run([str(binary), "--probe-model", str(model)],
                                env=environment, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as error:
        raise ValueError("Native model eligibility probe failed: " + str(error)[:500]) from error
    if result.returncode:
        raise ValueError("Native model eligibility probe failed: " + result.stderr.strip()[-1000:])
    if len(result.stdout.encode()) > 4 * 1024**2:
        raise ValueError("Native model probe response exceeds its bounded metadata frame")
    try:
        value = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise ValueError("Native model probe returned invalid JSON") from error
    return _validate_probe(value)


def describe_model(weights, plan, *, probe=None):
    """Describe model-native prompt metadata; missing templates mean raw mode.

    Pass a successful native probe during real registration. Without it,
    special tokens come from explicit GGUF token IDs, with empty strings for
    IDs absent from metadata; native tokenizer defaults are never invented.
    """
    fields = weights.fields
    architecture = fields.get("general.architecture")
    if (not isinstance(architecture, str) or not architecture.strip()
            or architecture != plan.get("architecture") or "\0" in architecture):
        raise ValueError("GGUF architecture does not match its compiler plan")
    template = _template(fields.get("tokenizer.chat_template"),
                         present="tokenizer.chat_template" in fields)
    tokens = fields.get("tokenizer.ggml.tokens")
    if tokens is not None and (not isinstance(tokens, list) or
                               any(not isinstance(token, str) for token in tokens)):
        raise ValueError("GGUF tokenizer vocabulary must contain strings")

    def special(name):
        ident = fields.get("tokenizer.ggml." + name + "_token_id")
        if ident is None or ident == -1 or ident == 4294967295:
            return ""
        if type(ident) is not int or ident < 0:
            raise ValueError("GGUF " + name + " token ID is outside its vocabulary")
        if tokens is None:
            # The memory-light GGUF reader intentionally skips token arrays.
            # Real registration supplies the native vocabulary-only probe.
            return ""
        if ident >= len(tokens):
            raise ValueError("GGUF " + name + " token ID is outside its vocabulary")
        token = tokens[ident]
        if "\0" in token:
            raise ValueError("GGUF " + name + " token contains NUL")
        return token

    variables = {"bos_token": special("bos"), "eos_token": special("eos")}
    context = _positive(fields.get(architecture + ".context_length"),
                        "GGUF context length", optional=True)
    vocabulary_size = len(tokens) if tokens else None
    if probe is not None:
        probe = _validate_probe(probe)
        if probe["architecture"] != architecture:
            raise ValueError("Native model architecture differs from the GGUF compiler plan")
        if template is not None and probe.get("chat_template") != template:
            raise ValueError("Native model chat template differs from the GGUF metadata")
        if vocabulary_size is not None and probe["vocabulary_size"] != vocabulary_size:
            raise ValueError("Native vocabulary differs from the GGUF metadata")
        if context is not None and probe["context_length"] != context:
            raise ValueError("Native context length differs from the GGUF metadata")
        template = probe.get("chat_template")
        variables = {name: probe[name] for name in ("bos_token", "eos_token")}
        context = probe["context_length"]
        vocabulary_size = probe["vocabulary_size"]
    name = fields.get("general.name")
    if not isinstance(name, str) or not name.strip():
        name = Path(weights.path).stem
    if len(name) > 256 or any(ord(char) < 32 for char in name):
        raise ValueError("GGUF model name must contain at most 256 characters without control bytes")
    return {"name": name, "architecture": architecture,
            "prompt_mode": "chat-template" if template is not None else "raw-completion",
            "chat_template": template, "template_variables": variables,
            "context_length": context, "vocabulary_size": vocabulary_size}
