# Third-party sources and dependencies

PixelForge is authored and maintained by Lee Daghlar Ostadi. It uses existing
language runtimes, model architectures and platform interfaces. Their original
attribution and licenses remain attached to their code.

| Component | Use | License and provenance |
| --- | --- | --- |
| llama.cpp / ggml | Model architectures, tokenization and CPU operations | MIT; [upstream](https://github.com/ggml-org/llama.cpp), revision recorded in `tools/arm_npu/LLAMA_ENGINE.json`; license in `tools/arm_npu/LICENSE.ggml` |
| LiteRT | Graph schema, C API headers and Tensor interfaces | Apache-2.0; notices in the source and `tools/arm_npu/LICENSE.litert` |
| Ostadix | Optional O program and command integration | LGPL-2.1; `source/OSTADIX/LICENSE` and `source/OSTADIX/NOTICE` |
| NumPy | Host-side tensor data | External Python dependency; BSD-3-Clause |
| FlatBuffers | Graph serialization | External Python dependency; Apache-2.0 |
| Jinja2 / MarkupSafe | Sandboxed model chat-template rendering | External Python dependencies; BSD-3-Clause |
| Huihui model checkpoints | User-supplied inference weights | Model-specific terms and pinned provenance; weights are not included |

Google Tensor vendor libraries and firmware are supplied by the supported
device. PixelForge does not grant permission to redistribute them. This public
source release includes neither those binaries nor a private phone backup.

The generated graph schema and imported GGUF decoder files retain their
upstream notices. Build scripts fetch or consume pinned upstream source;
their revisions and hashes are part of the recorded build inputs.

The original Q5_K/Q6_K bit layouts and group unpacking in
`tools/arm_npu/author_grouped_quant.py` follow the pinned llama.cpp/ggml
decoder, including its scale/minimum and signed-level layout. This adaptation
retains **Copyright (c) 2023-2026 The ggml authors**, under the MIT terms in
[LICENSE.ggml](tools/arm_npu/LICENSE.ggml). The source revision and reference
file hashes remain in
[GGUF_QUANTIZATION_PROVENANCE.json](tools/arm_npu/GGUF_QUANTIZATION_PROVENANCE.json).
