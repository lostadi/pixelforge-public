The Python host reads the user's Huihui GGUF weights directly and authors
constant FC graphs. The Android vendor compiler builds these graphs on the
phone, then the native helper runs the resulting code on the Tensor NPU. All
linear layers, including the vocabulary projection, use this helper. The host
runs embedding lookup, RMS normalization, RoPE, attention, KV cache, SiLU,
tokenization and greedy sampling. Failed NPU calls stop the run.

The current host validates **Huihui-MiniCPM5-2B-abliterated**, whose model
configuration is LlamaForCausalLM. Other architectures need their corresponding
host implementation and numerical checks. The graph author accepts ordinary
weight matrices and is not restricted to Google-provided model weights.

Only NumPy, FlatBuffers, Rust tokenizers and Jinja2 are needed for this NPU
path. The generated LiteRT 2.3.0 schema is included unchanged with its
Apache-2.0 license and hash in `SCHEMA_PROVENANCE.json`. TensorFlow, JAX,
PyTorch, Transformers, and the LiteRT Python interpreter are unnecessary.
The separate CPU reference path imports PyTorch only when explicitly selected.

After the repository's ARM runtime setup installs the native helper:

```sh
custom-models/minicpm5/venv/bin/python tools/arm_npu/model_host.py \
  --model custom-models/minicpm5/model.gguf \
  --tokenizer custom-models/minicpm5/tokenizer \
  --hook npu_linear:factory --prompt 'Hi' --tokens 8
```

The default native command enters the Android filesystem through
`/proc/1/root` and runs `/data/local/tmp/pixelforge-npu/npu_pipe`. This requires
the rooted Android host used by this bundle. The library search order starts
with `/system/lib64`, then `/vendor/lib64`, then the Termux libraries. Loading
the Termux libraries first caused a compiler stall on the tested device.

The default FC uses float32 input/output and per-row int8 constants, decoded
and requantized from GGUF weights. This second quantization introduces error.
Each input is divided by `max(maxabs(input), 1)` and the output is multiplied
by the same value. This preserves the mathematical linear map and avoids
nonfinite outputs observed for large FFN activations on the tested NPU.
Full float32 constant FCs have been validated as individual layers, but double
the compiled weight footprint. They are not the default full-model mode.
Experimental FP16 graphs are excluded from this kit.

The native helper retains compiled bytes for their full documented ABI
lifetime and unloads driver graph state after each invocation. Cached code
can be bounded without retaining the whole model at once:

```sh
PIXELFORGE_NPU_CACHE_MIB=512 custom-models/minicpm5/venv/bin/python \
  tools/arm_npu/model_host.py --model custom-models/minicpm5/model.gguf \
  --tokenizer custom-models/minicpm5/tokenizer --hook npu_linear:factory \
  --prompt 'Hi' --tokens 4
```

An LRU release uses the native `F` command; evicted matrices are compiled
again when next needed. Lower budgets therefore reduce speed. A single matrix
larger than the budget is permitted. The vocabulary projection is split into
4,096 output rows by default. The memory guard releases only this run's cached
code when kernel headroom is low; it never closes applications or services.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `PIXELFORGE_NPU_CACHE_MIB` | `0` | Compiled-code LRU budget; zero leaves the cache unlimited. |
| `PIXELFORGE_NPU_MIN_FREE_MIB` | `128` | Available kernel memory guard; zero disables it. |
| `PIXELFORGE_LM_BANK_ROWS` | `4096` | Output rows per vocabulary projection graph. |
| `PIXELFORGE_FC_WEIGHT_DTYPE` | `int8` | `int8` default; `float32` uses more memory. |
| `PIXELFORGE_NPU_STDERR` | Private state file | Native compiler/driver log. |
| `PIXELFORGE_NPU_FAILURE_DIR` | Private state directory | Numeric failure input/output dumps. |
| `PIXELFORGE_NPU_COMMAND_JSON` | Android command above | Explicit JSON argv for another native helper location. |

Reproduce the two-request hardware check with:

```sh
custom-models/minicpm5/venv/bin/python tools/arm_npu/validate_runtime.py \
  --report custom-models/minicpm5/validation/new-run.json
```

It runs proper chat prompts for a greeting and arithmetic in the same cache,
records actual NPU invocations, hardware counters, compile/decode timings,
rail telemetry and private first-logit arrays, then closes its own helper.
Report files must be new paths and are created with mode 0600. Rail power
estimates require the documented compatible kernel ABI assumption; charging
battery readings do not measure whole-device discharge power.

`test_npu_protocol.py` checks framing, activation scaling, LRU release and
bounded writes with a fake helper. Those protocol tests do not claim hardware
inference. Hardware results come from `validate_runtime.py` and the recorded
NPU counter changes.
