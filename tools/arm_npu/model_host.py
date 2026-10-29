"""Full MiniCPM5 Llama host with streamed GGUF weights and injectable NPU FC.

Every linear call uses GGUF-native row order. Q/K activations are unpermuted
before the ordinary Hugging Face half-vector RoPE. Small attention, activation,
normalization and KV-cache operations run on the host. A linear hook can execute
all dense layers on the NPU without creating a duplicate 10 GB checkpoint.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import importlib
import json
import os
import pathlib
import time

import numpy as np

from gguf_hf_loader import GGUFWeights


class MiniCPM5Host:
    def __init__(self, weights, linear=None, row_batch=256, threads=4):
        self.weights = weights if isinstance(weights, GGUFWeights) else GGUFWeights(weights)
        self.config = self.weights.llama_config()
        self.row_batch = row_batch
        self.linear_hook = linear
        self.norms = {}
        self.calls = []
        self.position = 0
        self.cache = [[] for _ in range(self.config['num_hidden_layers'])]
        self.hidden = self.config['hidden_size']
        self.heads = self.config['num_attention_heads']
        self.kv_heads = self.config['num_key_value_heads']
        self.head_dim = self.config['head_dim']
        self.eps = self.config['rms_norm_eps']
        self.inv_freq = 1 / (self.config['rope_theta'] **
                             (np.arange(0, self.head_dim, 2, dtype=np.float32) / self.head_dim))
        if self.config['hidden_size'] != 2048 or self.config['num_hidden_layers'] != 42:
            raise ValueError('This runner currently validates MiniCPM5-2B only; other Llama models need their RoPE scaling config')
        self.threads = threads
        self.torch = None

    def reset(self):
        self.position = 0
        self.cache = [[] for _ in self.cache]
        self.calls.clear()

    def norm(self, name, x):
        if name not in self.norms:
            self.norms[name] = self.weights.numpy(name)
        variance = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        return x * np.reciprocal(np.sqrt(variance + self.eps)) * self.norms[name]

    def cpu_linear(self, name, x):
        """Bound dequant memory to a row batch; use ARM PyTorch SGEMM."""
        if self.torch is None:
            import torch
            torch.set_num_threads(self.threads)
            self.torch = torch
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        descriptor = self.weights.tensors[name]
        rows, columns = descriptor.shape
        if x.shape[1] != columns:
            raise ValueError(f'Input shape mismatch for {name}: {x.shape} vs {descriptor.shape}')
        result = np.empty((len(x), rows), dtype=np.float32)
        xt = self.torch.from_numpy(x)
        for start in range(0, rows, self.row_batch):
            stop = min(rows, start + self.row_batch)
            w = self.weights.numpy_rows(name, start, stop)
            result[:, start:stop] = self.torch.mm(xt, self.torch.from_numpy(w).T).numpy()
        return result

    def linear(self, name, x):
        start = time.monotonic()
        result = self.cpu_linear(name, x) if self.linear_hook is None else self.linear_hook(name, x)
        result = np.asarray(result, dtype=np.float32)
        if result.ndim == 1:
            result = result[None, :]
        expected = (len(x), self.weights.tensors[name].shape[0])
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError(f'Invalid FC result for {name}: {result.shape}, expected {expected}')
        self.calls.append({'tensor': name, 'seconds': time.monotonic() - start,
                           'backend': 'cpu-reference' if self.linear_hook is None else 'injected'})
        return result

    @staticmethod
    def unpermute_activation(x, heads):
        # GGUF rows interleave the real/imaginary halves of each HF attention head.
        width = x.shape[-1] // heads
        return x.reshape(len(x), heads, width // 2, 2).swapaxes(-1, -2).reshape(x.shape)

    def rope(self, x):
        angles = np.float32(self.position) * self.inv_freq
        phase = np.concatenate([angles, angles])
        cosine, sine = np.cos(phase), np.sin(phase)
        half = x.shape[-1] // 2
        rotated = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
        return x * cosine + rotated * sine

    def layer(self, layer, x):
        prefix = f'blk.{layer}.'
        residual = x
        h = self.norm(prefix + 'attn_norm.weight', x)
        q = self.unpermute_activation(self.linear(prefix + 'attn_q.weight', h), self.heads).reshape(self.heads, self.head_dim)
        k = self.unpermute_activation(self.linear(prefix + 'attn_k.weight', h), self.kv_heads).reshape(self.kv_heads, self.head_dim)
        v = self.linear(prefix + 'attn_v.weight', h).reshape(self.kv_heads, self.head_dim)
        q, k = self.rope(q), self.rope(k)
        self.cache[layer].append((k.copy(), v.copy()))
        keys = np.stack([entry[0] for entry in self.cache[layer]], axis=1)
        values = np.stack([entry[1] for entry in self.cache[layer]], axis=1)
        repetitions = self.heads // self.kv_heads
        keys, values = np.repeat(keys, repetitions, axis=0), np.repeat(values, repetitions, axis=0)
        scores = np.einsum('hd,htd->ht', q, keys) / np.sqrt(np.float32(self.head_dim))
        probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        attention = np.einsum('ht,htd->hd', probabilities, values).reshape(1, self.heads * self.head_dim)
        x = residual + self.linear(prefix + 'attn_output.weight', attention)
        residual = x
        h = self.norm(prefix + 'ffn_norm.weight', x)
        gate = self.linear(prefix + 'ffn_gate.weight', h)
        up = self.linear(prefix + 'ffn_up.weight', h)
        # Clipping affects only saturated sigmoid values and avoids overflow.
        activated = gate / (1 + np.exp(-np.clip(gate, -80, 80)))
        return residual + self.linear(prefix + 'ffn_down.weight', activated * up)

    def hidden_step(self, token_id, layers=None, trace=None):
        x = self.weights.numpy_rows('token_embd.weight', int(token_id), int(token_id)+1)
        count = self.config['num_hidden_layers'] if layers is None else layers
        for layer in range(count):
            x = self.layer(layer, x)
            if trace is not None:
                trace(layer, x)
        self.position += 1
        return x

    def prefill_cpu_batch(self, token_ids):
        """CPU golden path: read each weight once for the whole short prompt.

        The NPU hook path remains sequential, because the current NPU FC ABI is
        a single activation row. This method is solely a faster CPU reference.
        """
        if self.linear_hook is not None or self.position != 0:
            raise ValueError('Batched CPU prefill requires a fresh CPU reference state')
        length = len(token_ids)
        if length > 256:
            raise ValueError('Reference prompt is bounded to256tokens')
        x = np.concatenate([self.weights.numpy_rows('token_embd.weight', int(i), int(i)+1)
                            for i in token_ids], axis=0)
        phases = np.arange(length, dtype=np.float32)[:, None] * self.inv_freq[None, :]
        phases = np.concatenate([phases, phases], axis=-1)
        cosine, sine = np.cos(phases), np.sin(phases)
        def rope(value):
            half = value.shape[-1] // 2
            rotated = np.concatenate([-value[..., half:], value[..., :half]], axis=-1)
            return value * cosine[None, :, :] + rotated * sine[None, :, :]
        causal = np.triu(np.ones((length, length), dtype=bool), k=1)
        for layer in range(self.config['num_hidden_layers']):
            prefix = f'blk.{layer}.'
            residual = x
            h = self.norm(prefix + 'attn_norm.weight', x)
            q = self.unpermute_activation(self.linear(prefix + 'attn_q.weight', h), self.heads)
            k = self.unpermute_activation(self.linear(prefix + 'attn_k.weight', h), self.kv_heads)
            v = self.linear(prefix + 'attn_v.weight', h)
            q = rope(q.reshape(length, self.heads, self.head_dim).transpose(1, 0, 2))
            k = rope(k.reshape(length, self.kv_heads, self.head_dim).transpose(1, 0, 2))
            v = v.reshape(length, self.kv_heads, self.head_dim).transpose(1, 0, 2)
            self.cache[layer] = [(k[:, position].copy(), v[:, position].copy())
                                 for position in range(length)]
            repetition = self.heads // self.kv_heads
            keys, values = np.repeat(k, repetition, axis=0), np.repeat(v, repetition, axis=0)
            scores = np.einsum('hqd,hkd->hqk', q, keys) / np.sqrt(np.float32(self.head_dim))
            scores[:, causal] = -np.inf
            probabilities = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            attention = np.einsum('hqk,hkd->hqd', probabilities, values)
            attention = attention.transpose(1, 0, 2).reshape(length, self.heads * self.head_dim)
            x = residual + self.linear(prefix + 'attn_output.weight', attention)
            residual = x
            h = self.norm(prefix + 'ffn_norm.weight', x)
            gate = self.linear(prefix + 'ffn_gate.weight', h)
            up = self.linear(prefix + 'ffn_up.weight', h)
            activated = gate / (1 + np.exp(-np.clip(gate, -80, 80)))
            x = residual + self.linear(prefix + 'ffn_down.weight', activated * up)
        self.position = length
        return x[-1:]

    def logits(self, hidden):
        normalized = self.norm('output_norm.weight', hidden)
        output_name = 'output.weight' if 'output.weight' in self.weights.tensors else 'token_embd.weight'
        return self.linear(output_name, normalized)[0]

    def step(self, token_id):
        return self.logits(self.hidden_step(token_id))

    def generate(self, prompt_ids, max_new_tokens=16, eos_ids=(1, 130073), callback=None):
        if not prompt_ids:
            raise ValueError('Empty token prompt')
        self.reset()
        start = time.monotonic()
        if self.linear_hook is None and len(prompt_ids) > 1:
            hidden = self.prefill_cpu_batch(prompt_ids)
        else:
            hidden = None
            for token_id in prompt_ids:
                hidden = self.hidden_step(token_id)
        output = []
        logits = self.logits(hidden)
        for index in range(max_new_tokens):
            token_id = int(np.argmax(logits))
            output.append(token_id)
            if callback is not None:
                callback(token_id, index, time.monotonic() - start)
            if token_id in eos_ids or index + 1 == max_new_tokens:
                break
            logits = self.step(token_id)
        return output


def tokenizer(directory):
    """Pinned Rust tokenizer + sandboxed template, with no Torch dependency."""
    return LocalTokenizer(directory)


class LocalTokenizer:
    def __init__(self, directory):
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        directory = pathlib.Path(directory)
        self.backend = Tokenizer.from_file(str(directory/'tokenizer.json'))
        config = json.loads((directory/'tokenizer_config.json').read_text())
        self.bos_token = config['bos_token']
        self.eos_token = config['eos_token']
        self.pad_token = config['pad_token']
        self.environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                        extensions=['jinja2.ext.loopcontrols'])
        def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                              separators=separators, sort_keys=sort_keys)
        def raise_exception(message):
            raise ValueError(message)
        self.environment.filters['tojson'] = tojson
        self.environment.globals['raise_exception'] = raise_exception
        self.template = self.environment.from_string((directory/'chat_template.jinja').read_text())

    def encode(self, text, add_special_tokens=True):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.backend.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=False, tools=None, **kwargs):
        rendered = self.template.render(messages=messages, tools=tools,
                                        bos_token=self.bos_token, eos_token=self.eos_token,
                                        pad_token=self.pad_token,
                                        add_generation_prompt=add_generation_prompt,
                                        enable_thinking=enable_thinking, **kwargs)
        return self.encode(rendered, add_special_tokens=False) if tokenize else rendered


def write_private_report(path, report):
    """Create one private report and refuse to overwrite an existing artifact."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/data/data/com.termux/files/home/ostadix-voice/runtime/models/Huihui-MiniCPM5-2B-abliterated.Q8_0.gguf')
    parser.add_argument('--tokenizer', required=True, help='Directory populated by fetch_tokenizer.py')
    parser.add_argument('--prompt', default='Say hello.')
    parser.add_argument('--raw-prompt', action='store_true')
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--hook', help='Python module:function; factory receives the GGUFWeights and returns linear(name,x)')
    parser.add_argument('--report')
    args = parser.parse_args()
    if not 1 <= args.tokens <= 128:
        parser.error('--tokens must be 1..128')
    weights = GGUFWeights(args.model)
    tok = tokenizer(args.tokenizer)
    if args.raw_prompt:
        prompt_ids = tok.encode(args.prompt, add_special_tokens=False)
    else:
        prompt_ids = tok.apply_chat_template([{'role':'user','content':args.prompt}],
                                            tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False)
        # Transformers5.19 may return BatchEncoding by default.
        if isinstance(prompt_ids, Mapping):
            prompt_ids = prompt_ids['input_ids']
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], list):
            if len(prompt_ids) != 1:
                raise ValueError('Expected exactly one chat prompt')
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(i) for i in prompt_ids]
    hook = None
    if args.hook:
        module, name = args.hook.split(':', 1)
        hook = getattr(importlib.import_module(module), name)(weights)
    runner = MiniCPM5Host(weights, linear=hook, threads=args.threads)
    print(json.dumps({'event':'start','prompt_tokens':len(prompt_ids),'backend':'cpu-reference' if hook is None else args.hook}), flush=True)
    started = time.monotonic()
    emitted = []
    def callback(token, index, seconds):
        emitted.append({'token':token,'seconds':seconds})
        print(json.dumps({'event':'token','id':token,'text':tok.decode([token]),'seconds':seconds}), flush=True)
    try:
        output = runner.generate(prompt_ids, max_new_tokens=args.tokens, callback=callback)
        report = {'source':'huihui-ai/Huihui-MiniCPM5-2B-abliterated', 'model':str(weights.path),
                  'backend':'cpu-reference' if hook is None else args.hook,
                  'prompt':args.prompt,'prompt_ids':prompt_ids,'output_ids':output,
                  'output':tok.decode(output, skip_special_tokens=True),
                  'seconds':time.monotonic()-started,'token_timings':emitted,'linear_calls':runner.calls}
        if hook is not None and hasattr(hook, 'stats'):
            report['npu_stats'] = hook.stats()
    finally:
        if hook is not None and hasattr(hook, 'close'):
            hook.close()
    if args.report:
        write_private_report(args.report, report)
    print(json.dumps({'event':'complete','output':report['output'],'seconds':report['seconds'],'linear_calls':len(runner.calls)}), flush=True)


if __name__ == '__main__':
    main()
