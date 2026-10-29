# Huihui inference compiled on the phone

The rooted Pixel 10 Pro compiled and ran the complete
`huihui-ai/Huihui-MiniCPM5-2B-abliterated` checkpoint on October 13, 2026.
The greeting generated **“Hi, I am a MiniCPM”**. Its first four tokens matched
the independently computed CPU reference exactly. All 326 constant graphs
were compiled locally by the Android ARM64 vendor compiler. The hardware
inference counter increased by exactly the 6,430 native invocations recorded
by the runner. See the [measured results](benchmarks/20261013-native-huihui/README.md).

All attention projections, feed-forward matrices and output-head banks execute
on the NPU, with no CPU linear fallback. Tokenization, embedding lookup,
RMSNorm, RoPE, attention, KV cache, SiLU and greedy sampling run on the CPU.
This is a mixed CPU/NPU pipeline. The MiniCPM-specific runner described here
validates this 42-layer checkpoint. The separate
[general GGUF compiler](tools/arm_npu/GGUF_COMPILER.md) preserves model graphs
with llama.cpp and has also run Huihui LFM2.5. Other models still need their
own hardware and numerical validation.

## Run on this device

From the existing Ubuntu root terminal:

```sh
cd /root/ostadix-device-bundle
python3 tools/arm_npu/runtime.py verify
python3 tools/arm_npu/runtime.py setup
python3 tools/arm_npu/runtime.py run --prompt 'Hi' --tokens 8
```

Setup verifies the private addon, installs its five hash-pinned ARM64 wheels
without network access, and copies its native helper into the Android root
at `/data/local/tmp/pixelforge-npu/npu_pipe`. The helper is launched through
`chroot /proc/1/root` so Binder and vendor services resolve correctly.
Requirements are Ubuntu ARM64 with CPython 3.12, venv and pip; this rooted
Tensor G5 device's compatible firmware, vendor compiler and dispatch
libraries; and Termux's C++ runtime. The included binary makes rebuilding
optional.

`run` verifies the actual GGUF, all six original tokenizer/configuration
assets and the installed native helper before inference. It uses the NPU
backend by default. Tokens stream as JSON, and the final event includes the
answer and private report path. `--report /path/to/new-report.json` selects a
path; existing reports are preserved. Failure reports retain the error and
completed NPU work. This is an isolated CLI and does not change the model used
by the existing Ostadix voice UI or production service.

The private addon is `custom-models/minicpm5/`: the actual 2,679,711,680-byte
Q8_0 GGUF, six original support files, five wheels, exact requirements, native
helper and hash manifest. These files are excluded from GitHub. A source clone
needs those assets before the offline commands can work. Model and quantized
publisher revisions and hashes are in
[`HUIHUI_ASSETS.json`](tools/arm_npu/HUIHUI_ASSETS.json). Q8_0 does not recover
the original BF16 checkpoint exactly.

## Rebuild and transport

Optional rebuild, using this phone's Termux ARM64 Clang while the CLI is idle:

```sh
/usr/sbin/chroot /proc/1/root \
  /data/data/com.termux/files/usr/bin/bash \
  /data/data/com.termux/files/home/ostadix-device-bundle/tools/arm_npu/build_native.sh
```

If rebuilding, update the addon's native file and manifest to match before
using the verified runner. [Compiler notes](tools/arm_npu/NATIVE_COMPILER.md)
identify the official headers, source pins, compiler and dispatch mechanism.

The original [`setup.sh`](setup.sh) packs the complete private directory into
one self-extracting file. Use external storage with enough free space:

```sh
python3 tools/arm_npu/runtime.py verify
sh setup.sh pack /external/storage/pixelforge.run
sh /external/storage/pixelforge.run /destination/pixelforge
cd /destination/pixelforge
python3 tools/arm_npu/runtime.py verify
python3 tools/arm_npu/runtime.py setup
python3 tools/arm_npu/runtime.py run --prompt 'Hi' --tokens 8
```

The archive retains offline model inputs, wheels and native code. It excludes
the derived venv, custom-model result/validation files and bytecode caches.
Recreate the venv after extraction. A full second archive has not been written
on this nearly full phone; extraction is tested with an independent fixture
after deleting its original directory. This is a reconstruction for compatible
rooted devices rather than a generic factory-reset installer.

## Memory, speed and capacity

The validated model stores 2,516,756,480 parameters. Its 326 compiled graphs
total 4,723,694,080 bytes. Compiled code stays alive in host RAM while cached;
the driver loads and unloads one projection's execution state per invocation.
Code buffers remain alive for the lifetime required by the dispatch API.
The runtime does not close other applications automatically. It releases its
own cache under pressure and stops if the working set still cannot fit.
During development,
unrelated background apps were closed with the user's authorization and an
idle Google-precompiled service was temporarily paused. Termux, the current
UI and essential services were preserved.

The first full run took 303.2 seconds, including 227.5 seconds of compilation.
Its first token arrived at 280.7 seconds; subsequent short decoding averaged
0.31 tokens/second. Native peak RSS was about 4.44 GiB; minimum system
MemAvailable was about 2.87 GiB. This establishes a working capacity point,
not a maximum or optimal model size. Its 130,560-token vocabulary requires
32 output-head banks per generated token.

An optional owned-code LRU trades repeated compilation for lower RAM use:

```sh
PIXELFORGE_NPU_CACHE_MIB=1024 python3 tools/arm_npu/runtime.py run --prompt 'Hi' --tokens 8
```

The default (`0`) has no fixed code-cache cap. Per-operation available-memory
checks protect compilation and invocation working sets and evict only this
runner's code. Smaller caps have not been benchmarked with the complete model.
Longer contexts, larger models and sustained thermal behavior are unmeasured.
Optimal size requires comparing quality, latency, context memory and energy
on multiple validated checkpoints.

Int8 row quantization changes the original GGUF block quantization. First-token
logits differed from CPU despite matching the first four greeting tokens:
relative L2 error was 0.1265 and cosine similarity 0.9920. This is functionality
and hardware validation, not an established general quality score.
Activation normalization before each linear operation, followed by inverse
output scaling, avoids a nonfinite intermediate observed in layer 7 without
changing the mathematical linear function. Proprietary compiler arithmetic
is not fully characterized.

The first charging run cannot establish whole-phone power draw. A later
unplugged launcher run averaged approximately 3.31 W of whole-phone net battery
discharge, including compilation, CPU inference work, NPU activity and the UI.
Its preceding samples include other work and are not a controlled idle baseline.
See the benchmark report for sampling and units. Conditional ODPM estimates
for TPU-related rails are separate and require an unverified kernel ABI
assumption. Repeated warm requests with a controlled idle baseline or external
meter are needed for steady inference power and energy per answer.
