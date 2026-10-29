# GGUF inspection and decoding

`gguf_inspect.py MODEL.gguf` reports the architecture fields, tensor-format counts, and decoding availability. Add `--tensors` for every tensor's shape, offset, and storage size. Inspection reads the header and descriptors, skips large tokenizer arrays, and maps the opened file; it does not hash or decode model weights or run inference.

The reader accepts little-endian GGUF v2/v3. It handles any architecture's metadata and retains unknown tensor kinds for inspection. An unknown kind has `size_known=false`; its reported byte count is the available storage extent, including possible padding. Decoding an unsupported kind raises `UnsupportedQuantizationError`. Recognizing a container or quantization format does not establish full-model NPU architecture support.

The NumPy-only decoder includes F32/F16/BF16, integer scalar formats, F64, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K, IQ2_XXS/IQ2_XS/IQ3_XXS/IQ1_S/IQ4_NL/IQ3_S/IQ2_S/IQ4_XS/IQ1_M, TQ1_0/TQ2_0, MXFP4/NVFP4, and Q1_0/Q2_0. `weights.decode_supported_types` reports the implemented IDs. `weights.TYPES` describes known block layouts; callers should use the former to decide whether a tensor can be decoded.

`numpy_rows(name, start, stop)` decodes a contiguous matrix row range, with bounded intermediate batches. It preserves GGUF row order. Architecture-specific transformations such as the Llama Q/K permutation belong in the execution adapter.

The reader opens a regular file once and retains it. Parsing, NumPy mapping, and `sha256()` use that same descriptor. `source_identity` is `(device, inode, size, mtime_ns, ctime_ns)`. `check_source_unchanged(check_path=True)` checks both the opened file metadata and whether its path still identifies that source. Hashing checks before and after the read; later digest reuse also checks the source. Keep the model unchanged while it is in use. Use the reader as a context manager or call `close()` to release its file handle and mapping reference.

The quantizer implementation comes from [the pinned llama.cpp source](https://github.com/ggml-org/llama.cpp/tree/254b177307b14d055bbe4c050c9c441a1e06d8a4). [GGUF_QUANTIZATION_PROVENANCE.json](GGUF_QUANTIZATION_PROVENANCE.json) records source hashes and adaptations. The bundled `gguf_vendor` package contains only constants, lazy NumPy support, and quantizers, avoiding upstream reader/writer/downloader imports. It needs NumPy and the standard library. The upstream MIT license is in [LICENSE.ggml](LICENSE.ggml).

Run the synthetic parser, same-file hashing, and multiple-row primary-reference checks with:

```sh
python -m unittest discover -s tools/arm_npu -p test_gguf_loader.py -v
```
