# User-selected Huihui models on Tensor G5

The complete `huihui-ai/Huihui-MiniCPM5-2B-abliterated` model now generates
answers through a mixed CPU/NPU pipeline compiled on this ARM64 phone. All
dense projections and output-head banks execute on the NPU. The first greeting
was “Hi, I am a MiniCPM,” with its first four tokens matching CPU reference
inference. Packaged two-request validation also returned the correct “4” for
2+2. See [setup and usage](ARM_NPU.md) and the
[benchmark evidence](benchmarks/20261013-native-huihui/README.md).

This path uses the user's actual Huihui Q8_0 checkpoint, pinned original
tokenizer and chat template, and locally authored constant graphs. It uses
Google's installed vendor compiler/runtime to operate the hardware. It does
not use Google-precompiled model weights or an external compilation host.
Compiled size, numerical differences and CPU operations are documented.
Support for arbitrary model architectures is not established by this one
working architecture. The separate [general GGUF path](tools/arm_npu/GGUF_COMPILER.md)
now preserves architecture semantics through llama.cpp, decodes 35 weight
formats, and persists NPU code. It has also matched both CPU token choices
from the installed Huihui LFM2.5 1.2B Q8_0 model, covering prefill and one
recurrent decode step with the same published architecture engine. The
[new evidence](benchmarks/20261013-gguf-compiler/README.md) distinguishes this
functionality test from numerical equivalence and throughput benchmarks.

## Native ARM compiler

The phone's `/vendor/lib64/libedgetpu_litert.so` exposes the compiler declared
in LiteRT's [official compiler ABI](https://github.com/google-ai-edge/LiteRT/blob/main/litert/vendors/google_tensor/edgetpu_compiler_api.h).
The [on-device adapter](https://github.com/google-ai-edge/LiteRT/blob/main/litert/vendors/google_tensor/adapter_odc.cc)
and [SouthBound dispatch API](https://github.com/google-ai-edge/LiteRT/blob/main/litert/vendors/google_tensor/dispatch/sb_api.h)
describe the interfaces. The repository includes unchanged pinned ABI headers,
their license, and the locally built ARM64 compiler/dispatch helper source.
An NPU node is requested explicitly; failed calls terminate inference.

MiniCPM5 has a standard 42-layer Llama architecture. This was the first installed
Huihui candidate with a sufficiently simple architecture to implement and
validate here. Tokenization, embedding lookup, RMSNorm, RoPE, attention, KV
cache, SiLU and sampling execute on CPU. This particular runner validates
MiniCPM5 only; the generic GGUF architecture engine is a separate runner.

## Packaged SDK route

[LiteRT's Hugging Face exporter](https://developers.google.com/edge/litert/conversion/pytorch/genai)
is a separate route accepting original local safetensors. The checked exporter
revision is `91dd077a9584f28bc82f135016358e9433bdf260`. Family recognition does
not establish that all operations compile on G5.

The [documented SDK workstation](https://developers.google.com/edge/litert/next/tensor-sdk)
uses Linux x86_64. That is a packaged SDK restriction; native Android ARM
compilation works here. The checked [SDK installer](https://github.com/google-ai-edge/LiteRT/blob/main/ci/tools/python/vendor_sdk/google_tensor/setup.py)
downloads SDK 6.1.0 when `GOOGLE_TENSOR_SDK_BETA` is unset. A separately supplied
beta archive is optional. Earlier claims that x86 or that archive were
mandatory for all NPU compilation are superseded.

`tools/plan_huihui.py` checks metadata for the packaged route. It never
downloads weights, compiles a model, or changes deployment:

```sh
python tools/plan_huihui.py \
  --repo huihui-ai/YOUR_CHOSEN_MODEL \
  --revision YOUR_EXACT_40_CHARACTER_COMMIT \
  --checkpoint /build-host/original-huihui-checkpoint \
  --output-dir /build-host/compiled-huihui-g5 \
  --save new-conversion-plan.json
```

Saved plans are uncompiled and unvalidated. Older examples preserve earlier
investigation results; the native route supersedes their ARM blocker.

## Other models and deployment

The installed `huihui-ai/Huihui-gemma-4-E2B-it-abliterated-v2` checkpoint is
pinned at `2f7b088470ce94da58fc1e5319b8ed88aed685a6`. Its Q6_K GGUF is
3,845,340,768 bytes. No original safetensors or compiled Huihui `.litertlm`
were found in the inspected locations. Gemma 4 remains a conversion target.
The generic engine recognizes its architecture, but no fresh Gemma 4 NPU
generation was performed in the GGUF compiler validation. MiniCPM5 and LFM2.5
success do not prove Gemma 4 numerical correctness.

The new CLI is separate from the existing Ostadix voice service. Its production
adapter can accept non-`litert-community` provenance after profile and hash
checks, but this runner's native protocol is not a `.litertlm` artifact.
Connecting it to the voice UI requires a backend implementation. The packaged
CLI already provides direct custom-model inference on the NPU.
