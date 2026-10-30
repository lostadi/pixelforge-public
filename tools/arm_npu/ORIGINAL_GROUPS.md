# Original Q5_K/Q6_K weights on the Tensor NPU

The `original_groups` and opt-in `original_columns` routes keep the original
quantized integers and their group coefficients. The host unpacks Q5_K or
Q6_K storage; the NPU applies the original scale and, for Q5_K, subtracts
the original offset before computing matrix products. `original_groups`
computes `Y = XWᵀ` for a row bank. `original_columns` computes disjoint column
products and the host adds them in order using F32 arithmetic. Both preserve
the GGUF and introduce no new weight quantizer. Their INT8 inputs contain
the original integer levels.

Q5_K uses 32-value groups with scale and offset. Q6_K uses 16-value groups
with scale and signed levels. The reconstructed host values match the pinned
GGUF decoder bit for bit. Device arithmetic still needs a numerical reference
check; preserving the source representation does not make vendor arithmetic
identical to the CPU decoder.

On the reference NeoHorse 1 4B Q5_K model, both prompt processing and decoding
have run with all **249 main Q5_K/Q6_K projection tensors** selected for NPU.
That count belongs to this model and placement. Attention, recurrent GDN,
normalization, sampling and the ordinary FFN elementwise operations remain
on CPU. Whole grouped-FFN fusion is outside this validated runtime route.

The column family has separate programs and acceptance requirements.
Primitive Q5_K/Q6_K checks establish only the tested geometries. A later
ordinary Termux file-tool test completed with B64 prompt programs, B1
decoding and all 249 main projections selected on NPU. Its roughly
17-minute result belongs to that phone and profile; it is not a CPU speedup
or model-wide quality result. See the
[full application validation](../../docs/VALIDATION.md).

## Canonical column ranges

`original_columns` splits the original matrix along its input columns,
taking successive logical spans of 3072 columns in whole 256-value blocks.
Only a shorter tail of a split matrix is zero-padded to a physical program
width of 3072. An unsplit matrix of at most 3072 columns keeps its original
width. For the common dimensions:

| Original columns | Logical source widths | Physical program widths |
| --- | --- | --- |
| 2560 | 2560 | 2560 |
| 4096 | 3072 + 1024 | 3072 + 3072 |
| 9216 | 3072 + 3072 + 3072 | 3072 + 3072 + 3072 |

Each NPU partial receives the matching raw F32 activation slice and the
unchanged original levels and coefficients for that source range. A split
tail extends activation, levels, scale and any offset ports with exact zeros;
all values inside the original range stay unchanged. The host
performs ordered F32 additions, with no CPU weight matrix multiplication
inside the selected projection. Addition order can change rounding, so
compare the complete sum with the original decoder's reference output.

The source manifest retains original tensor names and dimensions. Canonical
aliases exist only inside the column cache; they are not placement names or
native wire inputs. The package pins the wrapper, stable group kernel,
original GGUF hash, source tensor plan and device/compiler fingerprint.
Every selected row/column/batch geometry needs its own device reference check.
B64 is an option only after that complete inventory passes; a compiled or
partially validated package is insufficient.

The padded-tail recipe and wrapper fingerprint differ from the older
balanced layout, so those caches are rejected. Recompile into a separate
package with the current source; the GGUF and original tensor plan remain
unchanged.

## Programs and shared buffers

The compiled program describes matrix dimensions, original quantization kind
and batch size. It contains no model weights. Equal geometries share code,
while the package separately pins the GGUF hash and exact tensor plan.
Changing the bound source bank replaces every weight input before invocation;
the runtime checks the binding generation before accepting a warm program.

With `--weight-buffer-cache-mib` enabled, unpacked original levels and
coefficients occupy immutable shared DMA buffers. The buffer key includes
source hash, tensor, canonical row/column ranges, port and quantization kind.
It excludes batch size, so B1 decoding and prompt processing share the same weight
buffers. A device program can be evicted and later import those buffers
without unpacking or copying the bank again. When the retained weight budget
fills, admitted buffers stay cached and later banks stream through temporary
buffers. Their parent descriptors close after import acknowledgement; the
helper owns its received descriptors. Physical memory pressure can evict
retained buffers in LRU order and then pauses new admission for that process.
The device program LRU has a separate budget. These buffers use shared system
memory; they are not a claim that the model fits in NPU SRAM.

Known compiled-code and input/output port bytes are charged to
`--native-runtime-cache-mib`, including all physical tail padding. Shared
weight admission also charges every padded byte. Vendor workspaces, CPU model
state and optional prompt snapshots consume additional RAM. A warm weight-cache hit reuses
existing immutable buffers rather than reserving a fresh cold-unpack buffer.

Each source parameter needs approximately 1.25 bytes in the unpacked group
representation, before column-tail padding, page padding and program resources.
This is larger than the original packed GGUF, and smaller than a complete F32 expansion.
The memory guard checks for at least 1 GiB of available headroom before shared
weight allocations. It evicts only this process's cache entries and stops if
the required working set will not fit.

## Compile the geometry packages

Use the matching rooted Tensor device, helper and Python dependencies from
[the compiler guide](GGUF_COMPILER.md). Replace the model and output paths:

```sh
chat/venv/bin/python tools/arm_npu/compile_grouped_fc.py \
  --model /srv/models/your-huihui.gguf --output /srv/pixelforge/grouped \
  --bank-rows 4096 --batch-vectors 16 --plan-only

for batch in 1 16; do
  chat/venv/bin/python tools/arm_npu/compile_grouped_fc.py \
    --model /srv/models/your-huihui.gguf --output /srv/pixelforge/grouped \
    --bank-rows 4096 --batch-vectors "$batch" --resume
done
```

The compiler hashes the source once per invocation and compiles each distinct
geometry once. `--tensor NAME` or `--tensor-pattern REGEX` narrows the source
selection; the pattern must match the complete tensor name. The grouped
compiler includes only Q5_K/Q6_K rank-two weights. It does not copy weights,
decode them into an artifact, or invoke an inference. `--plan-only` writes
no package and starts no helper. Keep `--bank-rows` and both batch sizes
consistent with the installation plan.

The packages are `grouped-fc-v1/b1/MODEL.json` and
`grouped-fc-v1/b16/MODEL.json` beneath the supplied output directory.
Their hashed metadata pins geometry, original quantization kind, recipe,
author source and device/compiler identity. Deployment requires the exact
source hash and all geometries needed by the explicit P/D placements.
Read-only startup verifies them before the helper starts and never compiles
a missing program.

For the column family, `compile_grouped_columns.py` accepts the same model,
output, row-bank, batch and original-tensor selection arguments. Inspect
its plan before any device compilation:

```sh
chat/venv/bin/python tools/arm_npu/compile_grouped_columns.py \
  --model /srv/models/your-huihui.gguf --output /srv/pixelforge/columns \
  --bank-rows 4096 --batch-vectors 64 --plan-only
```

Compile and numerically validate B1 and the intended maximum prompt batch
before installing that route. Its manifests live at
`grouped-columns-v1/b1/MODEL.json` and `grouped-columns-v1/bB/MODEL.json`,
where `B` is that reviewed batch size. The compiler's completion means code
was saved; numerical acceptance and the full tool conversation are separate
checks. Keep the two program families in their own directories.

Only original **Q5_K and Q6_K with whole 256-value blocks per row** are
supported by this route. The installer rejects a selected F32 or other
quantization instead of silently placing it elsewhere. On a mixed-format
model, supply reviewed `--select-tensors` and `--select-prefill-tensors`
lists containing only the intended supported projections. Compiling a
geometry does not validate every model or tensor shape that might use it.

## Install with explicit placement

From the complete repository, follow the
[installer command](https://github.com/lostadi/pixelforge-public/blob/main/chat/README.md)
and add:

```sh
--dynamic-weight-mode original_groups --weight-dtype float32 \
--npu-prefill --batch-vectors 16 --decode-batch-vectors 1 \
--native-runtime-cache-mib 128 --weight-buffer-cache-mib 4096
```

Keep the source-specific projection lists explicit. The `float32` setting
here selects activation/coefficient math and transport; it does not convert
the original group levels into a new weight quantization. The required
helper supports sticky inputs and, when shared buffers are enabled, DMA
imports and deferred weight allocation. Missing capabilities stop startup.
The memory budgets are limits, not reservations or a promise that every
phone can sustain them.

For a fully validated column package, select `original_columns` and its
reviewed maximum with `--batch-vectors B`. The scalar budget and placement
options stay the same. With NPU prompt processing, a maximum above one and
B1 decoding, a single prompt vector uses the B1 graph while retaining phase
P. B1 must therefore cover the union of selected prompt and decode weights;
the maximum-batch package must cover every selected prompt weight. Read-only
preflight expands original names into all required aliases and row banks,
checks their bytecode and pins both manifests before starting inference.

| Installer option | Default | Meaning |
| --- | --- | --- |
| `--dynamic-weight-mode` | `off` | `original_groups` uses original Q5/Q6 groups; `original_columns` adds canonical column products and host F32 sums; `float32` decodes runtime weights on CPU |
| `--weight-dtype` | `int8` | Legacy static constant precision; use `float32` for dynamic routes |
| `--npu-prefill` | absent | Prompt processing is CPU by default; enable reviewed prompt NPU placement |
| `--select-prefill-tensors` | decode selection | Independent prompt projection list |
| `--batch-vectors` | `1` | Maximum NPU prompt batch; validate every required family/shape before use |
| `--decode-batch-vectors` | `1` | B1 decoding, or the prompt batch with zero-padded tails |
| `--prefill-batch-size` | `512` | Native prompt chunk size; NPU bank batches divide that request |
| `--native-runtime-cache-mib` | `128` | Known program/port budget, 0–2048 MiB; dynamic routes require a positive value; vendor workspace is additional |
| `--weight-buffer-cache-mib` | `0` | Retained shared weight budget, 0–6144 MiB, with a streamed overflow tail; zero uses sticky pipe uploads |
| `--native-max-resident-models` | `2` | Device program count limit, 1–64, active only with shared weight caching |
| `--cache-mib` | `256` | Host-side compiled code cache budget, separate from shared weights |
| `--npu-performance-mode` | `default` | Preserve vendor preference; `balanced`, `high`, `sustained` request SDK modes 2, 3, 4 |
| `--flash-attention` | `off` | CPU attention algorithm; `auto` uses upstream support checks and `on` requires support |
| `--cpu-affinity` | unset | Optional allowed CPU mask for this service and its children; verified before they start |
| `--prefix-cache-mib` | `0` | Exact prompt-state snapshots, 0–512 MiB; 256 MiB is an explicit phone-profile opt-in |

These are trusted installation settings. An inference request cannot change
placement, precision, buffers or performance policy. Android still controls
clock and thermal behavior. CPU attention remains on CPU with every flash
attention setting. Static F32 batching, gate/up pairing and complete F32
FFN programs use separate opt-in recipes and cannot be combined with the
dynamic group route.

The optional prefix cache stores prompt-token checkpoints and the engine's
CPU attention/recurrent GDN state in RAM. It captures prompt chunk boundaries,
never generated decode tokens. Before reuse, the engine clears live state
and verifies exact token-prefix, serialized-byte and sequence-position
agreement. A failed restore discards snapshots, clears live state again and
performs fresh prompt processing. At least one remaining prompt token is
processed for logits. The cache budget bounds snapshot bytes and bookkeeping;
it writes no persistent cache and replays no host actions.

Ready and completion receipts attest the prefix budget. Completion usage
includes hit, reused tokens, retained bytes/entries and actual prompt tokens
processed; NPU counters count only work actually performed. A prefix hit
does not assign a selected weight projection to CPU. Main turns retain no
elapsed deadline, and explicit cancellation ends the owned inference process.
The optional CPU mask applies only to that process tree; startup checks the
requested subset against the current allowed mask and verifies the result.

A chosen NPU failure ends generation. The broker does not replay the weight
operation on CPU or switch models. Successful projection calls establish
the selected hardware path; they do not establish a CPU speedup, sustained
throughput or model-wide quality.
