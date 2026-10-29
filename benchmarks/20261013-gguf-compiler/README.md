# Saved GGUF compilation and architecture-engine validation

On October 13, 2026, the ARM64 Pixel 10 Pro / Tensor G5 compiled ordinary GGUF
weight matrices into persistent NPU bytecode and generated through a pinned
llama.cpp architecture engine. A fresh process loaded the saved synthetic
model's 16 banks with **zero compilation**, executed 76 native NPU calls, and
matched all four CPU-reference token choices. The separate real Huihui LFM2.5
1.2B Q8_0 test matched both CPU-reference tokens, exercising prefill and one
recurrent decoding step. These are functionality
and numerical checks; they do not establish support for every GGUF or an
optimal model size.

[VALIDATION.json](VALIDATION.json) contains the sanitized run summaries,
full synthetic logits, top real-model logits, source/executable hashes,
timings and shared hardware-counter observations. The
[saved-package inventory](TINY_COMPILED_PACKAGE.json) records every bank's
graph and bytecode hashes, source identity and compiler/firmware fingerprint.
The [compiler and offline-kit guide](../../tools/arm_npu/GGUF_COMPILER.md)
explains installation, compilation, resume and read-only generation.

## Saved-package proof

The deterministic synthetic fixture has one Llama layer, a 256-wide hidden
state, a 512-wide feed-forward layer and 259 vocabulary entries. It is not a
useful language model. Its eight rank-two weight matrices include the tied
embedding/output matrix and are split into banks of at most 192 rows.

The saved package embeds the 2,632,704-byte source GGUF and contains 16
verified int8-constant artifacts totaling **11,927,168 bytes** of native
code. Its manifest records `complete: true` and
`completed_weight_dtypes: ["int8"]`. A subsequent default CLI run used the
published final helper and the saved package in read-only mode:

| Recorded measurement | Published run |
| --- | ---: |
| Saved banks loaded | 16 |
| Graphs compiled | 0 |
| Native NPU invocations | 76 |
| Engine weight-matmul operations | 32 |
| Generated token IDs | 246, 246, 246, 246 |
| Total elapsed time | 1.403 s |
| Native invocation time summed by the client | 0.592 s |
| Native cleanup | Successful |

One engine matrix operation can contain multiple activation vectors or output
banks, so the engine operation count and native invocation count differ. The
[published matrix records](tiny_npu_published-matrices.json) identify each
materialized graph as `operation: "load"`. The final helper's SHA-256 is
`45003ec33d56b12daacd312dab5efdebebe214dc7e804faac4b27140bde0caee`;
the earlier hardware runs used a different helper hash recorded separately.

This proves that the runtime can reuse exported device code after process
restart. A verified package is required for read-only execution; a cache miss
or an incompatible source/compiler/firmware/author fingerprint stops the run.
The saved code remains specific to this device's compiler and firmware.

## CPU numerical comparison

All **1,036 logits** were compared across four generation steps, rather than
checking token IDs alone. The initial and published NPU runs have identical
logit arrays and both match all four CPU token choices:

| Comparison with the decoded-GGUF CPU reference | Result |
| --- | ---: |
| Relative L2 error | 0.0155573 (1.556%) |
| RMSE | 0.0256641 |
| Cosine similarity | 0.9998799 |
| Per-step relative L2 error | 1.903%, 1.408%, 1.438%, 1.449% |
| Zero-projection negative-control RMSE | 2.49073 |

The negative control changes the model's projection weights and produces a
much larger error. Together with the recorded NPU operations and full-logit
comparison, this supplies stronger evidence than a hardware counter or a
matching token alone. It establishes the synthetic graph's behavior, not
general language quality.

The FC graphs use float32 input/output with per-row int8 constants decoded
and requantized from the GGUF. This additional quantization explains why
numerical equivalence is not claimed. Full float32 FC mode was not validated
by this benchmark.

## Real Huihui LFM2.5 smoke test

The installed Huihui LFM2.5 1.2B Q8_0 GGUF has source SHA-256
`6f4c9e0baf72f74c486c8e44a0e64475f8f73042f28e1704f7929b77b70705e7`.
Its architecture is `lfm2`, exercising a different graph from the synthetic
Llama fixture. The final CPU and NPU runs used the same published engine
SHA-256 and both generated **521, 859**, decoding to **", I"**. This covers
prefill and one recurrent decode step through the real model graph.

The final NPU run used the published helper and a **512 MiB code-cache
budget**. It recorded 186 engine weight-matmul operations and 429 successful
native invocations, with successful engine, broker and native cleanup:

| Recorded measurement | Final real-model run |
| --- | ---: |
| Total elapsed time | 386.450 s |
| First / second generation-step time | 200.753 / 184.837 s |
| Compilation time | 366.574 s |
| Native invocation time summed by the client | 10.835 s |
| Compilation operations, including recompilation | 300 |
| Cache evictions | 269 |
| Native code retained before cleanup | 535,455,744 bytes |

The [final matrix records](huihui_lfm2_npu_published-matrices.json) retain all
300 compilation records, including repeated banks after eviction.
The bounded cache demonstrates execution without retaining the complete
compiled model in RAM. Recompilation dominates both steps, so these timings
do not measure warm decoding with a complete persistent package.

The CPU and NPU top-10 token lists share eight entries at each step. Maximum
absolute logit drift among those shared entries is 1.0773 for the first step
and 0.7678 for the second. These are top-10 observations, **not full-vocabulary
error metrics**; matching token IDs does not establish numerical equivalence.

The earlier one-token NPU run compiled 150 banks, made 279 native invocations
for 93 engine matrix operations, and closed its helper successfully. Its recorded
total time was 199.169 seconds, including 185.284 seconds of compilation;
this is not a warm decode-speed measurement. The
[matrix records](huihui_lfm2_npu-matrices.json) include the real short-convolution
input/output projections as well as other dense weights.

The architecture engine runs attention, convolution, recurrent state,
normalization, tokenization and sampling on CPU. Eligible rank-two weight
matrix multiplications use the NPU, and their failures stop generation.
Two matching raw-prompt tokens establish this bounded functionality check.
They do not establish sustained decoding, chat quality, full-logit equivalence,
or an acceptable quantization error for this real model.

## Attribution and support limits

An independent MiniCPM inference test was active during these measurements.
Device counters cover all TPU clients: the published synthetic window rose
by 77 while this helper recorded 76 calls, the initial LFM2 window rose
by 459 while its helper recorded 279, and the final LFM2 window rose by 777
while its helper recorded 429. Even some CPU-only windows show a
counter increase. The excess activity is unattributed; none of these device
deltas should be presented as an exact count belonging to this run. Client
operation records, successful responses and numerical comparisons provide
the run-specific evidence. Concurrent activity also limits latency comparisons.

The report preserves whole-phone battery endpoint observations. They are not
NPU watts, integrated inference energy, or a controlled power benchmark.
This experiment does not identify an optimal model size, sustained throughput
or energy per answer.

The engine revision is
`254b177307b14d055bbe4c050c9c441a1e06d8a4`. The generic offline kit includes
the pinned ARM engine, native helper, upstream source archive and
NumPy/FlatBuffers wheels; user-selected GGUF weights remain separate.
See the [offline asset pins](../../tools/arm_npu/GGUF_RUNTIME_ASSETS.json)
and [setup tool](../../tools/arm_npu/setup_gguf.py).

The final self-extracting compiler kit was also extracted into a fresh
directory and installed entirely from its offline wheels. Its own engine and
Python environment then loaded the 16-bank fixture, compiled nothing, made
76 NPU calls, and reproduced the published fixture's complete logits.
[STANDALONE_EXPORT.json](STANDALONE_EXPORT.json) records the 54,291,517-byte
archive's SHA-256 and successful setup/run checks. The source checks comprise
82 Python tests and separate native protocol/cleanup fixtures.

Unknown graph operations, unsupported quantizations, indexed rank-three MoE
expert multiplications and split GGUF shards remain explicit limitations.
Architecture-name recognition does not prove valid generation on hardware.
The separate [MiniCPM5 validation](../20261013-native-huihui/README.md)
documents the earlier complete custom-model chat path.

[MOCK_SQ_UNLOAD_VALIDATION.json](MOCK_SQ_UNLOAD_VALIDATION.json) records native
cleanup/error-path tests with zero hardware API calls. Those checks support
failure handling and are separate from the inference evidence above.
