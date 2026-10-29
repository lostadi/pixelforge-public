# Inference paths

The model, execution backend and host tool dispatcher are separate parts of
the system. Their names identify configuration. Successful model and device
operations establish what actually ran.

## Huihui through PixelForge

The public [chat service](chat/service.py) loads a supplied Huihui GGUF through
the pinned [native architecture engine](chat/native/chat_runner.cc). The host
renders the model's own template. CPU prompt processing is explicit; decoding
uses either the selected NPU tensor list or all planned dense projections.

The [matrix bridge](chat/native/ggml_npu_bridge.cc) distinguishes NPU-selected
weights, explicitly CPU-placed weights, CPU prefill and dynamic operations.
Selected NPU failures stop the model. They do not become a CPU fallback.
Compiled banks are checked against the source weights, graph recipe and device
fingerprint before they are loaded.

The [Termux client](source/ostadix-voice/pixelforge_chat.py) accepts completed
responses only with positive per-turn NPU operation counts and matching model
and placement metadata. It passes model requests to the existing ordinary-user
[tool coordinator](source/ostadix-voice/local_chat.py). Root inference receives
bounded model data; the host owns tool validation, saved action policy and
receipts. Explicit cancellation closes the owned model process without replay.

The tested NeoHorse 4B profile selects three first-block feed-forward tensors.
It leaves the remaining model operations on CPU. [Live evidence](docs/VALIDATION.md)
records a real file-tool loop and its measured scope.

## Other conversation engines

The source also retains a llama.cpp CPU client, an optional Gemini network
client and a Codex adapter. Engine choice is explicit and saved by the voice
interface. API credentials and active model manifests are supplied locally;
they are absent from the public source archive.

The independent [LiteRT NPU adapter](source/ostadix-voice/npu_runtime.py) and
[native host](source/ostadix-voice/runtime/npu/resident.c) support a separately
configured device-specific `.litertlm` pipeline. That is a distinct model path
from the converted Huihui weights used by PixelForge. An NPU backend flag alone
does not establish the model's provenance or successful hardware execution.

## Historical device integration

The [AICore extension source](source/OSTADIX/apps/aicore-ostadix-extension/README.md)
documents an earlier Gemini Nano integration through installed Google apps.
It requires its own compatible system assets, activation and framework setup.
Its presence here is source provenance, not a fresh Nano inference result or
a generic installation route.

Earlier capture and restoration utilities are separated from public packaging.
The original phone inventory, activation settings, private payload and live
configuration are excluded from the public source distribution.
