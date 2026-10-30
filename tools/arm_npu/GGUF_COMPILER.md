# Compile your own GGUF on the Tensor NPU

This path reads your original GGUF and compiles ordinary weight matrices on
the ARM64 phone. It does not download a Google-prepared model or require an
x86 compiler. The Android vendor compiler and runtime already installed on
this Tensor device produce and execute device-specific DarwiNN bytecode.
Ubuntu supplies the Python tools and architecture engine; the native helper
runs inside the Android filesystem.

For unchanged original Q5_K/Q6_K weights, use the separate
[original-group route](ORIGINAL_GROUPS.md). Its geometry-only programs accept
losslessly unpacked original levels and coefficients, reconstruct weights on
NPU and share immutable DMA weight buffers across device-program eviction.
It supports separate B16 prompt and B1 decode programs. The static constant
compiler documented below remains available with its own precision and
artifact format.

The pinned llama.cpp architecture engine owns the model's forward graph and
embedded GGUF tokenizer. Eligible rank-two weight matrix multiplications go
through the NPU helper. Attention, normalization, sampling and other graph
operations run on CPU. Recognition of an architecture in the engine is not
a numerical validation of that model on this hardware. An NPU failure stops
the run; eligible weight multiplications do not silently fall back to CPU.

## Prepare the tools

The public source checkout can build the conversational path without the
private offline addon. From the repository root:

```sh
python3 -m venv chat/venv
chat/venv/bin/python -m pip install -r chat/requirements-chat.txt
bash tools/arm_npu/build_native.sh /data/local/tmp/pixelforge-npu/npu_pipe_gguf
bash chat/build_native.sh --fetch
```

The helper build requires the Termux ARM64 C++ compiler on the supported
rooted phone. `--fetch` downloads the pinned architecture-engine source,
not model weights. The direct Python compiler commands in the
[original-group guide](ORIGINAL_GROUPS.md) use this virtual environment.
The [chat installer](../../chat/README.md) provides the conversational host.

The older `gguf.sh` commands below use a separately supplied, verified offline
addon. That addon is not part of this public source release.

Run these commands from your existing Ubuntu root shell, in the repository:

```sh
/usr/bin/python3 tools/arm_npu/setup_gguf.py
```

This installs the separate generic offline addon under
`custom-models/gguf-runtime`: the ARM architecture engine, dedicated native
helper, pinned NumPy/FlatBuffers wheels and their virtual environment. The
weights you choose are supplied separately. The private addon contains the
large binary and upstream source assets excluded from Git.

To rebuild both executables yourself on the phone:

```sh
bash tools/arm_npu/build_native.sh /data/local/tmp/pixelforge-npu/npu_pipe_gguf
bash tools/arm_npu/build_llama_npu.sh "$PWD/custom-models/gguf-runtime/build"
mkdir -p custom-models/gguf-runtime/bin
cp custom-models/gguf-runtime/build/build/bin/pixelforge-gguf \
  custom-models/gguf-runtime/bin/pixelforge-gguf
```

The setup command uses the private offline addon and its pinned ARM Python
wheels. The generic compiler needs NumPy and FlatBuffers; it does
not use PyTorch, TensorFlow, JAX or the LiteRT Python interpreter. If you use
another environment, set `PIXELFORGE_PYTHON` to its Python executable before
running `gguf.sh`.

The isolated engine build copies the pinned llama.cpp revision from the local
checkout named by `PIXELFORGE_LLAMA_SOURCE`. Its default is the captured
`ostadix-voice/runtime/llama-cpp` checkout. On another installation, provide a
local checkout, or the addon's pinned source archive, containing commit
`254b177307b14d055bbe4c050c9c441a1e06d8a4`:

```sh
PIXELFORGE_LLAMA_SOURCE=/path/to/llama.cpp \
  bash tools/arm_npu/build_llama_npu.sh "$PWD/custom-models/gguf-runtime/build"

# Or use the hash-verified bundled source, without the original checkout:
PIXELFORGE_LLAMA_ARCHIVE="$PWD/custom-models/gguf-runtime/source/llama-source.tar.gz" \
  bash tools/arm_npu/build_llama_npu.sh "$PWD/custom-models/gguf-runtime/build"
```

The build directory is a disposable copy owned by these tools. The build
script refuses to patch an unrelated existing source tree. The helper name
`npu_pipe_gguf` keeps this path separate from the original validated MiniCPM
runtime helper.

## Export the compiler as one self-extracting file

The compiler kit can be exported separately from the original device snapshot:

```sh
python3 tools/arm_npu/pack_gguf.py /root/pixelforge-gguf-compiler.run
sh /root/pixelforge-gguf-compiler.run /root/pixelforge-gguf-kit
python3 /root/pixelforge-gguf-kit/tools/arm_npu/setup_gguf.py
```

Choose fresh archive and destination paths. The export includes the compiler
source, architecture engine, pinned upstream rebuild source, native helper,
licenses and offline wheels. Extraction verifies its payload hash and all six
published runtime asset pins. It excludes model weights, the original private
snapshot and derived environments. Supply your chosen GGUF separately; the
supported Android vendor runtime, Ubuntu ARM64 and Python 3.12 remain required.

## Inspect and plan before compiling

Replace `/path/to/your-model.gguf` with the model you chose:

```sh
bash tools/arm_npu/gguf.sh inspect /path/to/your-model.gguf
bash tools/arm_npu/gguf.sh compile \
  --model /path/to/your-model.gguf --output /path/to/compiled-model --plan-only
```

Inspection reads header metadata and tensor descriptors without decoding or
hashing all weights. Planning hashes the complete source once, chooses bounded
row banks, identifies unsupported operations, and estimates required disk.
`--plan-only` creates no artifact directory and starts no NPU helper. It reports
`would_fit: false` when space is insufficient while still showing the plan.

The current compiler accepts supported GGUF quantizations for rank-two
`.weight` tensors, including tied embedding/output weights. Unknown
quantizations, indexed rank-three MoE expert matrices, and split GGUF shards
are reported as unsupported. Higher-rank tensors are not flattened into
invented dense layers. This is a general matrix compiler plus an architecture
engine, not a promise that every possible GGUF can generate correctly today.

## Save native compiled artifacts

Compile the supported rank-two tensor plan:

```sh
bash tools/arm_npu/gguf.sh compile \
  --model /path/to/your-model.gguf --output /path/to/compiled-model
```

Or start with one real projection when disk is limited:

```sh
bash tools/arm_npu/gguf.sh compile \
  --model /path/to/your-model.gguf --output /path/to/compiled-model \
  --tensor blk.0.attn_q.weight --rows 0:256
```

Use the names from `inspect --tensors`; architectures need not use the example
name. A selected tensor or row subset is marked partial. A row subset is a
projection test; generation reuses it only if its range matches a complete
bank in the runtime plan. To finish later, use
the same source, quantization and bank size with `--resume`:

```sh
bash tools/arm_npu/gguf.sh compile \
  --model /path/to/your-model.gguf --output /path/to/compiled-model --resume
```

Already verified banks load their saved bytecode rather than recompiling.
Bulk compilation releases each native graph after saving it, so the whole
model's compiled code does not need to remain in RAM. Compiling matrices alone
does not perform an inference or establish generation quality.

The package contains `MODEL.json`, hashed per-bank metadata and native `.dgc`
bytecode in `graphs/`. It pins the GGUF SHA-256, tensor dimensions and row
ranges, authored graph SHA-256, bytecode SHA-256, compiler/firmware fingerprint,
graph-author/decoder source hashes and D/L protocol ABI. An incompatible
fingerprint or corrupted artifact is rejected. Completion verifies every
expected bank and records the completed weight formats.

By default the package references the existing GGUF. Add `--copy-source` to
include `model.gguf` inside the package; source copying also needs disk space.
The package is then self-contained for weights, while the matching Python kit,
architecture engine and Android vendor runtime remain execution dependencies.
Bytecode is bound to the compiler/firmware fingerprint of the device where it
was built. Keep the kit version used to build it; changing decoder or graph
author source requires a new package.

## Generate and verify that saved code is used

```sh
bash tools/arm_npu/gguf.sh run \
  --model /path/to/your-model.gguf --artifacts /path/to/compiled-model \
  --read-only --prompt 'Hi' --tokens 4
```

`--read-only` refuses missing banks and never compiles a cache miss. Match the
compilation `--bank-rows` and `--weight-dtype` if you changed their defaults.
Without `--read-only`, an artifact package can acquire missing matrices as
generation requests them. Without `--artifacts`, compiled code exists only
in that process's RAM and is compiled again after restart.

The command prints a path to its private `RESULT.json`. Check `success`, the
reported NPU invocation count, hardware counters, and
`broker.npu.loaded_models`. A fully saved read-only run should have
`broker.npu.compiled_models` equal to zero. Device counters include all TPU
clients, so combine them with this run's successful NPU responses and recorded
matrix operations. Compare generated output against a CPU reference to check
model functionality. Use `--cpu` for a reference run with the same model and
prompt.

## RAM, disk and precision

There is no fixed whole-model RAM requirement. `run --cache-mib 512` limits
this run's retained native code; an individual larger graph is allowed and
evicted when other graphs need space. `--bank-rows` bounds individual matrix
graphs. The memory guard releases only graphs owned by this run and stops if
the estimated working set will not fit. It never force-closes applications.

The default FC graphs use float32 input/output and per-row int8 constants
decoded and requantized from the GGUF. This introduces additional error;
check a CPU reference for each model before relying on output quality.
`--weight-dtype float32` preserves decoded constants at a larger footprint.
Float32 mode needs its own hardware/numerical validation; the complete-model
checks recorded here use int8 FC constants. The prompt is raw text, and the
embedded GGUF tokenizer is used; this runner does not apply a chat template.
Unsupported experimental FP16 modes are not exposed.

Measured compiled int8 FC artifacts occupy roughly two bytes per weight plus
a compiler envelope per graph; float32 constants occupy roughly four. This
can exceed the original quantized GGUF's size. The default full MiniCPM5
rank-two plan includes its embedding matrix and estimates about 5.7 GiB plus
the source model. A RAM cache limit does not shrink saved artifacts. Use an
output filesystem with sufficient space, a smaller model, or selected tensor
compilation. The compiler performs a disk preflight before starting native
work or copying source weights.

On this device, Huihui LFM2.5 1.2B Q8_0 generated two raw-prompt tokens through
this path, matching the same engine's CPU choices. With a 512 MiB code cache,
the run made 429 NPU invocations and took 386 seconds, including 367 seconds
of compilation and recompilation. That establishes prefill and one recurrent
decoding step; it does not measure sustained speed or model quality. A fresh
synthetic-model process separately loaded 16 saved banks with zero compilation.
Full real-model artifact storage needs more space than is currently free on
the phone; inspect the disk plan before choosing an output filesystem.
