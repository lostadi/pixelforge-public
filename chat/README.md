# Conversational PixelForge backend

This backend runs a supplied Huihui GGUF through the pinned llama.cpp architecture engine and PixelForge's compiled Tensor NPU matrix bridge. The phone host validates model JSON requests and owns tool execution. Model output never executes root commands in this service.

The reference phone used `gemma4` and `qwen35`. CPU prompt processing is explicit. With selected placement, only the named weight projections execute on NPU; other model operations remain on CPU. Omitting the selection requests all planned dense decode weights and requires a complete bank package. A chosen NPU operation that fails aborts inference; it does not silently move to CPU.

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

Compile only explicitly selected tensors when using hybrid placement. For example, invoke `tools/arm_npu/compile_gguf.py` with the same `--model`, `--output` and `--bank-rows 4096`, once for each `--tensor blk.0.ffn_down.weight`, `blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight`; use `--resume` for subsequent tensors. Never add `--copy-source` unless you independently want a separate model copy. All-dense mode needs the full package instead. Saved banks must match the exact GGUF hash and device/compiler fingerprint; startup verifies every selected bank with compilation disabled.

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

The bootstrap reads only its root-owned local `bootstrap.json`. It verifies the configured Ubuntu root inode, interpreter and installed source hashes before entering that mount namespace. It does not import any old device-specific launcher. The service accepts inference fields only: `id`, `prompt`, `max_tokens`, `context_size` and `prefill_cpu`. IPC callers cannot select a model, filesystem output, helper command or root command. Main turns have no elapsed deadline; explicit cancellation terminates only the recorded process identities. Tool commands still use the phone host's ordinary action policy and their own limits.

After installing the exported voice source and its `c` launcher, start `c` and select `/engine pixelforge`. The service loads on a request. Switching profiles means reviewing a new installer manifest with the actual model/bank inputs and replacing an idle local installation; no private device-profile catalogue is shipped. `manage.py inspect --manifest /opt/pixelforge-chat/INSTALL_MANIFEST.json` shows the recorded output scope.

## Evidence and licenses

[Reference acceptance](REFERENCE_ACCEPTANCE.json) records one controlled tool call on the original phone: three selected FFN tensors, seven compiled banks, 214.6 seconds, 72 matrix operations and 168 native NPU invocations, with no compilation during inference. Its profile and binary hashes identify that historical deployment.

The [fresh public build hardware record](../docs/evidence/public-source-hardware-20261029.json) independently establishes a real file read and exact reply through the source-built decoder, isolated public installer and ordinary Termux host: 66 matrix operations, 154 native invocations and successful cleanup. It took 707.9 seconds under concurrent phone load. Neither run is a general quality benchmark, speedup claim or clean-phone installation result.

New host code follows the repository license. The linked architecture engine and bridge retain [ggml's MIT notice](native/LICENSE.ggml). The LiteRT-derived schema used by the compiler kit retains its separate Apache notice under `tools/arm_npu/`. User-supplied model and vendor-runtime terms remain separate.
