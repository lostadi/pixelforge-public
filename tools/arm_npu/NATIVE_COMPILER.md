The compiler runs natively on the phone's ARM64 CPU and emits code for its
Tensor G5 NPU. It uses the device's existing Android vendor compiler through
the published LiteRT `EdgeTpuCompiler*` ABI, then executes its DarwiNN DGC0
bytecode through an explicitly NPU SouthBound graph node. No x86 emulation
or Tensor SDK archive is involved.

Build in Termux, or from the Ubuntu chroot with access to Termux's toolkit:

```sh
bash tools/arm_npu/build_native.sh
```

From this Ubuntu chroot, launch the native Android helper with:

```sh
chroot /proc/1/root /system/bin/env \
  LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/data/data/com.termux/files/usr/lib \
  LD_PRELOAD=/system/lib64/libbinder_ndk.so \
  /data/local/tmp/pixelforge-npu/npu_pipe
```

The host Android filesystem supplies the working Binder device that this
chroot lacks. The explicit library order matters: a Termux-first order hung
in an isolated test. Compiler and runtime libraries come from this device's
vendor partition and are not redistributed. This launch route requires root.
The source build uses Termux clang++, falling back to clang with its C++ driver
mode. It writes a temporary executable and installs it atomically. A validated
prebuilt helper may instead be restored by the device bundle setup.

The helper redirects vendor stdout to stderr before library initialization;
send stderr to a log file. Its stdout is an ASCII-plus-binary protocol:

* Startup: `READY PIXELFORGE_NPU_PIPE 1\n`.
* Capabilities: `I CAPABILITIES\n`; success: `OK CAPABILITIES C R F I D L Q\n`.
  A client using persistent artifacts must check for `D` and `L` before compiling.
* Compile: `C NAME NINPUTS NOUTPUTS MODEL_BYTES INPUT_BYTES... OUTPUT_BYTES...\n`,
  followed by the raw TFLite model. Success: `OK COMPILED NAME BYTECODE_BYTES\n`.
* Export: `D NAME\n`; success: `OK BYTECODE BYTECODE_BYTES\n`, followed by the
  cached raw DGC0 bytecode. This does not compile or invoke the model.
* Load: `L NAME NINPUTS NOUTPUTS BYTECODE_BYTES INPUT_BYTES... OUTPUT_BYTES...\n`,
  followed by the raw DGC0 bytecode. Success: `OK LOADED NAME BYTECODE_BYTES\n`.
  The helper validates its format and device graph before caching it. Supplied
  code and tensor ports have the same bounds as compilation: 2 GiB payload,
  512 inputs/outputs, 1 GiB per tensor, and 2 GiB of tensor buffers in total.
  Header sizes are unsigned decimal digits only. The frontend must verify bytecode,
  source-model, tensor-port, and matching compiler/firmware fingerprints before
  loading a persistent artifact. Exported code is specific to that runtime.
* Infer: `R NAME\n`, followed by each raw input tensor. Success:
  `OK OUTPUT NOUTPUTS TOTAL_BYTES\n`, then each raw output tensor.
* Free a cached model: `F NAME\n`; success: `OK FREED\n`.
* Inspect cached state: `I NAME\n`; success: `OK INFO BYTECODE_BYTES INVCTX_PRESENT\n`.
* Quit: `Q\n`; success after context cleanup: `OK QUIT\n` and process exit zero.

Compilation validates the resulting graph, then unloads its device resources.
Each inference rebuilds its scheduling container, graph, invocation context,
and buffers with fresh unique IDs. It copies output and confirms resource
cleanup before reporting successful inference.
Compiled code stays valid in RAM until the cache entry is freed, as required
by the upstream API. There is no CPU fallback in this helper.
The native helper requests a kernel parent-death signal before vendor startup,
so terminating its broker also terminates this owned native process.

This keeps hardware-buffer use near the size of the currently executing
projection. It still needs RAM for all cached compiled projections. In the
recorded A/B/A test, two real 2048×2048 weight projections retained 18.1 MB
of compiled code while unique idle DMA buffers returned to 36,864 bytes after
each request. These are two vendor context message queues. Their duplicated
descriptors sum to 73,728 bytes; the validation counts exported buffer inodes
once. Releasing only the invocation context retained weight buffers across
requests, so the default unloads the entire device graph and container.

[NATIVE_VALIDATION.json](NATIVE_VALIDATION.json) records 12 interleaved runs of
different same-shape real and negated Huihui weights. Every output was finite
and agreed with its CPU reference within 0.00583. Whole-model generation and
the CPU operations surrounding these NPU projections are recorded separately.

The vendor compiler accepted constant-weight FLOAT32 and hybrid int8/FLOAT32
fully connected graphs. Tested dynamic-weight FC/BATCH_MATMUL graphs failed
with Binder DEAD_OBJECT; the following constant-weight tests recovered. An
FP16 constant followed by DEQUANTIZE compiled and ran, but its constants were
expanded into four bytes per weight in the compiled artifact. The default
hybrid path uses about two bytes per weight in the compiled artifact.

The unchanged upstream headers are Apache-2.0 licensed; see
[LICENSE.litert](LICENSE.litert) and [UPSTREAM_APIS.json](UPSTREAM_APIS.json).
