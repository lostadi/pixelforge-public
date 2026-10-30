# Conversational PixelForge backend

This backend runs a supplied Huihui GGUF through the pinned llama.cpp architecture engine and PixelForge's compiled Tensor NPU matrix bridge. The phone host validates model JSON requests and owns tool execution. Model output never executes root commands in this service.

The reference phone used `gemma4` and `qwen35`. Prompt and decoding placements are explicit. The `original_groups` route has run the NeoHorse model's 249 main Q5_K/Q6_K projections on NPU in both phases, keeping the original integer levels and group coefficients. CPU unpacking preserves those values; NPU programs reconstruct the weights and multiply them. Attention, recurrent GDN, normalization, sampling and ordinary FFN elementwise operations remain on CPU. A chosen NPU operation that fails aborts inference.

Shared DMA weight buffers survive device-program eviction and serve both B1 decoding and the configured prompt batch. The earlier full-projection path used B16 prompt programs. Compiled code is reused by geometry, with separate source-bank and generation checks before invocation. The GGUF stays unchanged and no extra weight quantization is introduced. See [the original-group guide](../tools/arm_npu/ORIGINAL_GROUPS.md) for the supported formats, compiler commands, memory budgets and default settings.

## Build and test

Use Ubuntu ARM64 with Python 3.12, CMake, a C++17 compiler, make, git, patch and ripgrep. From the repository root:

```sh
python3 -m venv chat/venv
chat/venv/bin/python -m pip install -r chat/requirements-chat.txt
bash chat/build_native.sh --fetch
chat/venv/bin/python -m unittest discover -s chat/tests -v
```

`--fetch` downloads only llama.cpp source at commit `254b177307b14d055bbe4c050c9c441a1e06d8a4`. For offline builds, use `--source /path/to/llama-git-checkout` containing that commit, or `--archive /path/to/llama-source.tar.gz` matching the archive hash in `tools/arm_npu/LLAMA_ENGINE.json`. The script builds every architecture-engine library from source. It does not reuse a private static library or installed service. Outputs stay under ignored `chat/build/`; the runner is `chat/build/bin/pixelforge-chat`, and hashes are in `chat/build/NATIVE_BUILD.json`.

The native tests use a synthetic tiny GGUF and a simulated matrix broker. They test routing, selected-operation failure, per-request counters and shutdown; they do not prove hardware execution. Native tests skip until the binary exists. Root-owned bootstrap/cleanup fixtures skip without root.

The source cache records both the upstream revision and the applied bridge
patch hash. A stale or unrecorded patch stops the build before compilation.
Move the reported `chat/build/source` cache aside and rerun with a pinned
source input to regenerate it.

Native replies report `context_init_seconds`, `prefill_seconds` and
`decode_seconds` separately. `generation_seconds` remains prefill plus decode;
it excludes context initialization, source validation and host-tool work.
These fields describe an individual run and do not establish an overall
speedup.

## Prepare local model banks

Supply your own model, matching compiled banks and the Android helper built from `tools/arm_npu/npu_pipe.cc`. Follow [the compiler guide](../tools/arm_npu/GGUF_COMPILER.md) for the compatible rooted Tensor runtime and its helper setup. GGUF model weights, DarwiNN bytecode, proprietary vendor runtime libraries and SDK binaries are not included here.

For unchanged Q5_K/Q6_K weights, use `compile_grouped_fc.py` to prepare the separate B1 and B16 geometry packages. The installation options are `--dynamic-weight-mode original_groups --weight-dtype float32 --npu-prefill --batch-vectors 16 --decode-batch-vectors 1`, together with explicit source-specific projection lists. Every selected tensor must be a supported original Q5_K/Q6_K matrix. Root preflight checks each phase's exact source plan and hashed program inventory without starting hardware or compiling a cache miss.

For original-group NPU prompt processing with a maximum batch above one and B1 decoding, single-vector prompt work also uses the B1 NPU graph while keeping phase P. The B1 package must cover the union of selected prompt and decode weights; the maximum-batch package covers the selected prompt weights. Preflight verifies and pins both inventories. A partial older package missing a prompt-only B1 geometry fails startup until that geometry is compiled separately.

The opt-in `--dynamic-weight-mode original_columns` uses separate `grouped-columns-v1/bB` packages. It takes successive 3072-column logical ranges from a wide original Q5_K/Q6_K matrix. Only a shorter split tail is zero-padded to a physical width of 3072; unsplit matrices up to 3072 columns keep their width. Thus 4096 uses logical spans `[0:3072]` and `[3072:4096]`, both with 3072-column programs, while 2560 stays whole and 9216 uses three full spans. The tail's activation, original levels, scale and any offset ports gain exact zeros beyond the source range. The NPU reconstructs the original groups and computes each column product; the host adds the partial products in order using F32 arithmetic. The original GGUF remains unchanged and no additional quantizer is used. Select original weight names only; aliases are internal to the source-pinned cache. Preflight expands every selected weight into all required column and row ranges, including the B1 prompt/decode union, and charges physical padded port sizes. The new recipe and source fingerprint reject older balanced-column packages. Constant gate/up and full-FFN fusion recipes cannot be combined with this mode. Use B64 only after all required row/column/batch geometries have passed device reference checks. A partial geometry package or an individual successful B64 primitive does not establish whole-phone acceptance.

The earlier static-bank route uses `compile_gguf.py`. Compile only explicitly selected tensors when using hybrid placement, with the same source, output and bank size and `--resume` for subsequent tensors. Its default INT8 constants requantize decoded GGUF weights; `--weight-dtype float32` preserves decoded values in larger constant artifacts. Keep those packages separate from original-group programs. Never add `--copy-source` unless you independently want a model copy. Saved packages must match the exact GGUF hash and device/compiler fingerprint; startup verifies every selected bank with compilation disabled.

## Review and install

Run the installer inside the already configured Ubuntu root environment. An existing mounted Ubuntu session must remain available to the Android bootstrap. Supply paths as seen in that environment; `--ubuntu-root` identifies its root directory in Android's host filesystem. The Termux home and voice runtime must already exist and belong to the same application account.

```sh
chat/venv/bin/python chat/install.py \
  --ubuntu-root /path/in/android/to/ubuntu/rootfs \
  --python /absolute/ubuntu/path/to/chat/venv/bin/python \
  --model /srv/models/your-huihui.gguf \
  --artifacts /srv/pixelforge/your-compiled-banks \
  --helper /data/local/tmp/pixelforge-npu/npu_pipe_gguf \
  --termux-home /data/data/com.termux/files/home \
  --app-runtime /data/data/com.termux/files/home/ostadix-voice/runtime \
  --model-name 'Your Huihui model' \
  --provenance-repo huihui-ai/YOUR_MODEL \
  --architecture qwen35 \
  --select-tensors blk.0.ffn_down.weight,blk.0.ffn_gate.weight,blk.0.ffn_up.weight
```

This is a dry run. It prints the exact output paths, discovered Termux UID, model/source/helper hashes, bootstrap and generated profile. It neither downloads nor copies weights or compiled banks. Review that manifest, then repeat the command with `--apply`. `--replace` permits replacing an idle installation; the shared chat lock rejects replacement while the backend is running.

Defaults are `/opt/pixelforge-chat` for installed source and `/data/adb/ostadix-pixelforge` for root state. Change them with `--service-dir` and `--state-dir`. The default installed `chat.py` entrypoint matches the exported voice adapter. A custom state directory also requires configuring that adapter's launcher path. Optional arguments include `--native-binary`, `--bank-rows 4096`, `--cache-mib 256` and an explicit `--hardware-counter-path`. The installer generates hashes; do not copy the illustrative [profile template](examples/profile.example.json) into a running installation.

`--weight-buffer-cache-mib` limits retained immutable shared weight buffers. Once its admission budget fills, admitted buffers stay cached and later banks stream through temporary buffers. Physical memory pressure can evict retained entries and pause new admission. A warm cached bank reuses its existing buffers. `--native-runtime-cache-mib` separately bounds known compiled-code and port resources; vendor workspaces and the model's CPU state also consume RAM. `--native-max-resident-models` accepts 1 to 64 and defaults to 2. It limits retained device programs when the weight-buffer cache is positive; the device byte budget still applies. With weight buffer caching off, the count limit is inactive and completion usage reports `null` for it.

`--npu-performance-mode` accepts `default`, `balanced`, `high` or `sustained`. The default leaves the vendor graph preference unchanged; the other values request the documented SDK modes 2, 3 and 4 respectively. Android retains control of clock and thermal policy. These root profile settings cannot be changed by an inference request.

`--cpu-affinity 2,3,4,5,6,7` optionally restricts the service and its inference children to an explicit CPU mask. Startup requires every requested CPU to belong to the process's allowed mask, applies it before spawning children and verifies the result. Ready and completion events attest the configured mask. Omit the option to preserve the existing mask; use CPU indices appropriate to the actual device.

`--prefix-cache-mib` accepts 0 to 512 and defaults to 0. `--prefix-cache-mib 256` is an explicit phone-profile opt-in. It bounds an in-memory native cache of exact prompt-token checkpoints, including the architecture engine's attention and recurrent GDN state, captured at prompt chunk boundaries. The engine clears its live state before a restore, verifies checkpoint byte and position counts, and processes at least one remaining prompt token for logits. A failed restore clears the snapshots and live state before fresh prompt processing. The cache writes no persistent snapshots and replays no host actions. Ready events confirm the trusted budget; completion usage reports `prefix_cache_hit`, `prefix_tokens_reused`, `prefix_cache_bytes`, `prefix_cache_entries` and `prefill_tokens_processed`. NPU operation counts remain the operations actually performed for that turn. Cache limits bound RAM, not elapsed turn time.

The bootstrap reads only its root-owned local `bootstrap.json`. It verifies the configured Ubuntu root inode, interpreter and installed source hashes before entering that mount namespace. It does not import any old device-specific launcher. The service accepts inference fields only: `id`, `prompt`, `max_tokens`, `context_size` and `prefill_cpu`. IPC callers cannot select a model, filesystem output, helper command or root command. Main turns have no elapsed deadline; explicit cancellation terminates only the recorded process identities. Tool commands still use the phone host's ordinary action policy and their own limits.

After installing the exported voice source and its `c` launcher, start `c` and select `/engine pixelforge`. The service loads on a request. Switching profiles means reviewing a new installer manifest with the actual model/bank inputs and replacing an idle local installation; no private device-profile catalogue is shipped. `manage.py inspect --manifest /opt/pixelforge-chat/INSTALL_MANIFEST.json` shows the recorded output scope.

## Evidence and licenses

[Reference acceptance](REFERENCE_ACCEPTANCE.json) records one controlled tool call on the original phone: three selected FFN tensors, seven compiled banks, 214.6 seconds, 72 matrix operations and 168 native NPU invocations, with no compilation during inference. Its profile and binary hashes identify that historical deployment.

The [fresh public build hardware record](../docs/evidence/public-source-hardware-20261029.json) independently establishes a real file read and exact reply through the source-built decoder, isolated public installer and ordinary Termux host: 66 matrix operations, 154 native invocations and successful cleanup. It took 707.9 seconds under concurrent phone load. Neither run is a general quality benchmark, speedup claim or clean-phone installation result.

New host code follows the repository license. The linked architecture engine and bridge retain [ggml's MIT notice](native/LICENSE.ggml). The LiteRT-derived schema used by the compiler kit retains its separate Apache notice under `tools/arm_npu/`. User-supplied model and vendor-runtime terms remain separate.
