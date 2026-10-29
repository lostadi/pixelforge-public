# Huihui MiniCPM5 on the Tensor G5 NPU

On October 13, 2026, the phone compiled the selected **Huihui-MiniCPM5-2B-abliterated** weights locally on Ubuntu aarch64 and completed generation through its native Tensor G5 NPU. The fixed chat prompt `Hi` produced **`Hi, I am a MiniCPM`**. The model has **2,516,756,480 parameters**. The source is the ordinary Huihui model, quantized to GGUF; this run used no Google-precompiled language model.

All seven dense matrices in every one of the 42 decoder layers, and all 130,560 output-vocabulary rows, ran through the native NPU hook with no CPU dense-matrix fallback. Tokenization, embedding lookup, RMS normalization, RoPE, attention, the KV cache, SiLU, and greedy sampling ran on the CPU. This is a working custom-model inference path with NPU matrix acceleration. It does not establish that every transformer operation or arbitrary architecture runs on the NPU.

The complete [run summary](npu-chat-Hi.json), [326-graph inventory](npu-graph-manifest.json), [CPU comparison](logits-comparison.json), and [hardware measurements](hardware-summary.json) contain the supporting numbers. Each graph entry identifies its original GGUF tensor, exact row range, shape, SHA-256, compiled size, and invocation count. A later [packaged two-request run](packaged-two-request-v2.json) also completed `Hi` and a fixed arithmetic prompt through the offline runtime. These files publish only those fixed benchmark prompts and numeric evidence. Model weights, activation arrays, private runtime paths, and process identities remain outside this public report.

## Functionality and coverage

The 294 decoder graphs each ran 21 times: 14 prompt tokens and seven additional decode steps. The 32 output-head banks each ran eight times. Both the native invocation log and the device inference-counter increase were **6,430**:

```text
294 × (14 + 8 − 1) + 32 × 8 = 6,430
```

The graph inventory covers **2,249,195,520 dense weight elements**. All recorded NPU results were finite. The firmware-crash and watchdog counters stayed at zero. The device counter is shared with other clients; the matching increase supports the native invocation evidence rather than identifying a client by itself.

The CPU reference emitted `[19301, 33, 354, 1017]`, or `Hi, I am`. The NPU run emitted `[19301, 33, 354, 1017, 280, 44524, 7739, 66]`, or `Hi, I am a MiniCPM`. Their first four greedy tokens agree. Their first next-token logits differ: maximum absolute difference **1.5982**, mean absolute difference **0.2725**, and relative L2 difference **0.12652**.

The [dequantization checks](gguf-dequant-validation.json) matched the GGUF reference implementation exactly on sampled Q8_0 and Q6_K tensors. The [two-layer causal/cache test](host-hf-two-layer-validation.json) and [42-layer CPU reference check](cpu-raw-Hello-golden.json) validated the host's transformer math against Hugging Face. The [tokenizer check](tokenizer-validation.json) validated the exact pinned chat formatting. The NPU graphs then re-quantized the source's Q8_0 blocks into one INT8 scale per output row, and scaled activations to prevent floating-point overflow. This changes the computation. A matching short prefix does not establish equivalent quality, perplexity, task accuracy, or long-context behavior.

## Timing and memory

This was one cold run with lazy compilation and sequential one-token prefill. It used the Pixel 10 Pro / Tensor G5 and the device's Android vendor compiler and dispatch runtime. No x86 machine was involved.

| Measurement | Observed value |
| --- | ---: |
| Prompt / generated tokens | 14 / 8 |
| Cold run to completion | 303.3 s |
| Time to first generated token | 280.7 s |
| Graph compilation, accumulated | 227.5 s |
| Throughput across the seven later-token gaps | 0.311 tokens/s |
| Median later-token gap | 3.33 s |
| Native invoke-call time, accumulated | 69.7 s |
| Source GGUF | 2,679,711,680 bytes |
| Compiled code cache | 4,723,694,080 bytes / 4.40 GiB |
| Native process RSS / high-water RSS | 4.41 / 4.44 GiB |
| Lowest sampled system memory available | 3,008,080 KiB / 2.87 GiB |

The invocation timer includes dispatch loading, invocation, unloading, and IPC; it is not a pure accelerator kernel timer. The later-token gaps also include CPU transformer operations and runtime loading. The original run instrumented its first token with local tensor dumps, so these results describe that prototype run rather than a production throughput ceiling.

The native process retained all compiled code until runtime unload, following the compiler ABI's ownership requirement. This accounts for much of its RSS. Runtime graph resources were loaded and unloaded per request; **326 cached code objects does not mean 326 models resident in accelerator SRAM**. Sampled available memory recovered to 7,710,280 KiB after cleanup. The memory numbers describe this 14-token prompt and eight-token continuation, not a long-context capacity test.

**Working capacity established here: one 2.52B-parameter MiniCPM5 model. Maximum capacity and optimal model size remain unmeasured.** Comparing other sizes, architectures, quantization formats, and longer contexts is needed before calling any size optimal. Compilation and runtime-loading overhead are substantial in this prototype.

## Packaged runtime: two requests and cache reuse

The [packaged two-request validation](packaged-two-request-v2.json) used the offline Python environment, packaged GGUF/tokenizer assets, and the native helper. Both requests shared one compiled code cache, with the transformer KV state reset between prompts. `Hi` emitted exactly the four CPU-reference tokens, **`Hi, I am`**. `Answer with one digit: what is 2+2?` emitted **`4`**, followed by EOS.

| Measurement | Cold `Hi` request | Warm arithmetic request |
| --- | ---: | ---: |
| Prompt / emitted tokens | 14 / 4 | 25 / 2, including EOS |
| Complete request | 359.4 s | 90.9 s |
| Time to first token | 348.0 s | 86.9 s |
| New compiled graphs | 326 | 0 |
| Compilation time | 291.5 s | 0 s |
| Native NPU invocations | 5,126 | 7,708 |
| Later-token throughput | 0.264 tokens/s, three gaps | 0.249 tokens/s, one gap to EOS |

All **326 graph hashes, tensor shapes, and row ranges matched the earlier inventory**. The native invocation total and device inference-counter increase both equaled **12,834**. The arithmetic request reused the existing compiled code; its 25-token sequential prefill and per-invocation runtime loading still took substantial time. A single gap ending at EOS is not a sustained-generation benchmark.

The packaged Python process had none of `torch`, `transformers`, `ai_edge_litert`, `tensorflow`, or `jax` in its loaded modules at the end of the two requests. The lowest sampled available system memory was **2,310,044 KiB**. The native code cache remained **4,723,694,080 bytes**. Artifact/source provenance was appended after the run: the recorded helper and four Python-file digests were collected immediately after that owned run, before replacing its executable or changing guard/ID logic. The tested helper SHA-256 is `b0284c400e66d1d4a74f11dc14015ccf581ede8e550502bbd2186ae1881e3e7c`; the report lists the Python-file digests and pinned model/tokenizer sources. **Later guard and native-ID patches are outside this completed run.** The final native binary has separate bounded validation. [PUBLIC_PACKAGED_VALIDATION_MANIFEST.json](PUBLIC_PACKAGED_VALIDATION_MANIFEST.json) hashes the sanitized packaged report and this README.

## Final launcher and native helper

The [final published CLI run](final-published-cli.json) completed `Hi` with one generated token, **`19301` / `Hi`**, using native helper SHA-256 `fea5924b2130a1bbdb2819e3b13181d60126c9dd2750c08ea3da55ef1cab2eb0`. Model, tokenizer/support, and installed-helper hashes were verified by the launcher. Six source-file hashes were captured while that launch was running and are recorded in the public evidence.

The final run compiled all **326 graphs**, whose hashes and tensor row ranges exactly matched the earlier inventory. Its **4,148 native invocations** equaled the device inference-counter increase: `294 × 14 + 32 = 4,148`. Total time was **316.536 s**, including hash verification and cleanup; time to first token was **310.042 s**, and accumulated compilation was **251.767 s**. This confirms the final launcher/helper through all 42 layers and the vocabulary head. One emitted token provides no subsequent-token throughput measurement. Earlier multi-token and arithmetic runs retain their separate artifact/version records.

The [final native smoke test](final-native-smoke.json) made four NPU calls using real Huihui query rows and a negative-weight variant. Positive/negative/positive reuse and explicit free/recompile passed, with maximum recorded CPU absolute error **0.000235081**. A truncated header at EOF returned `ERR HEADER_SIZE` and exit code `2`; all owned test helpers closed. These are bounded FC/protocol checks. [PUBLIC_FINAL_VALIDATION_MANIFEST.json](PUBLIC_FINAL_VALIDATION_MANIFEST.json) hashes both final reports and this README.

## Power measurement limits

Every hardware sample in the first eight-token run showed **USB connected and the battery charging**. Battery current therefore cannot establish whole-device power draw for that run. No isolated NPU wattage is claimed.

Two TPU rail counters increased during the cold run. Assuming the installed CPM driver uses the microjoule energy counters and millisecond timestamps documented in [this pinned Google ODPM source](https://android.googlesource.com/kernel/google-modules/soc/gs/+/204078758737776959ecc85c527a00fd2cabb62f/drivers/iio/power/odpm-whi.c#1315), their combined delta would be **9.895632 J over 302.332 s**, or **0.032731 W**. The exact installed CPM revision and units were not verified. This conditional device-wide rail average includes a window dominated by CPU compilation, excludes other phone rails, and can include other NPU clients. It does not measure steady-state inference power or total phone consumption. The hardware report preserves the raw rail deltas alongside these qualifications.

The packaged run started while charging, then recorded 21 unplugged/discharging samples near the end of arithmetic, spanning **21.1 s**. Current times voltage gave a whole-device net battery-discharge mean of **3.04 W**, median **2.67 W**, and range **1.23–7.95 W** in that short window. This includes the CPU, display, radios, background activity, and accelerator. No idle baseline was subtracted or stable sustained window established. The packaged report labels this separately from its conditional TPU rail estimates; it is not NPU-only power.

The [whole-phone observer for the final CLI run](whole-phone-power-observer.json) collected **88 unplugged/discharging samples**. Its 62 samples while the CLI worker ran had mean **3.313 W**, median **3.149 W**, and range **1.260–10.730 W**. The 22 preceding samples averaged **2.929 W**, but included other work and were not a controlled idle baseline. Power was calculated from battery current in microamperes and voltage in microvolts, following the [Linux power-supply units](https://docs.kernel.org/power/power_supply_class.html#units). These are whole-phone current-times-voltage estimates; they establish neither NPU-only draw nor incremental inference power.

## Model and reproducibility

The original model is [huihui-ai/Huihui-MiniCPM5-2B-abliterated](https://huggingface.co/huihui-ai/Huihui-MiniCPM5-2B-abliterated/tree/aea962bb5558e659624eeefc8c1b9f418a3e6ffb), revision `aea962bb5558e659624eeefc8c1b9f418a3e6ffb`. The local Q8_0 GGUF came from [the pinned mradermacher conversion](https://huggingface.co/mradermacher/Huihui-MiniCPM5-2B-abliterated-GGUF/tree/fc1e8318e85635d142e95a0f9977487717c40b58), revision `fc1e8318e85635d142e95a0f9977487717c40b58`; its SHA-256 is `ffc8f95be9e7fab3c8d8a50c6634828433b31ca80a803349f47dc3b35aaaeeab`. [HUIHUI_ASSETS.json](HUIHUI_ASSETS.json) pins the six model configuration/tokenizer assets separately.

The source implementation is in [tools/arm_npu](../../tools/arm_npu/). The [native compiler instructions](../../tools/arm_npu/NATIVE_COMPILER.md) describe the matching Android runtime prerequisite and ARM build. Once that setup and the tokenizer assets are ready, run the host with `--hook npu_linear:factory --prompt Hi --tokens 8`. Its report is created privately and refuses to overwrite an existing file. [PUBLIC_NPU_VALIDATION_MANIFEST.json](PUBLIC_NPU_VALIDATION_MANIFEST.json) hashes the public NPU evidence; [PUBLIC_CPU_VALIDATION_MANIFEST.json](PUBLIC_CPU_VALIDATION_MANIFEST.json) separately hashes the earlier CPU checks.
