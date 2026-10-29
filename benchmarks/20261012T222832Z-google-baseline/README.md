# Tensor G5 hardware baseline, October 12, 2026

This run used the already-installed **Google-precompiled Gemma 4 E2B** artifact.
It is a hardware/service baseline, not evidence that the user's chosen
huihui-ai weights run on the NPU. No deployment profile was changed.

Nine requests completed through the `litert-npu` service, with one warm-up,
three correctness checks and five identical short generation repetitions.
Engine PID 11676 stayed constant and the service finished ready and idle.

| Measurement | Observed value |
| --- | --- |
| Completed NPU requests | 9 / 9 |
| Strict correctness checks | 2 / 3 |
| Repeated generation output | 47 completion tokens per request |
| Median client latency, five repetitions | 3.55 s |
| Sample p95 client latency | 3.94 s |
| Median native time to first token | 0.163 s |
| Median native decode throughput | 13.93 tokens/s |
| Median effective completion throughput, including prefill | 13.37 tokens/s |
| Kernel-visible physical RAM | 15.21 GiB |
| Available RAM during run | 1.78–2.13 GiB |
| Referenced DMA buffers in separate capacity observation | 305 unique buffers, 3.18 GiB |
| Whole-phone sampled battery discharge before workload | 1.90 W mean |
| Whole-phone sampled battery discharge during workload | 5.34 W mean |
| Whole-phone sampled battery discharge after workload | 3.46 W mean |
| Combined TPU rail draw during sampled workload | approximately 0.55 W mean, 11.09 J |

Arithmetic returned `5`; the literal check returned `READY`. Extraction
returned the correct color/count wrapped in Markdown fences, so the strict
bare-JSON check failed. The suite therefore reports `functional_pass: false`.
Generation repetitions measured completion/timing, not factual correctness:
the response incorrectly described wireless transmission as sound waves.
This is useful evidence against treating successful generation as quality.

The native host explicitly selects NPU, with no CPU fallback. Its runtime and
GoogleTensor dispatch libraries were mapped before/after. The device-wide
`inference_count` changed from `0 0 2766` to `0 0 3040`; these are driver counts,
not the number of requests or tokens. No competing service queue work was
observed, but other Android TPU clients remain possible.

Ordinary Termux could access the service and its own new timing metadata, but
SELinux denied battery, TPU sysfs and ODPM reads. A separate root collector
sampled these read-only at the same time. It did not execute the model as root
or alter Android permissions. The child used the existing Termux UID and app
context; no inherited private environment was saved.

TPU power is an **ABI-based estimate**, combining CH6 `S7M_VDD_TPU` and CH8
`S9M_VDD_TPU_M`. The energy format matches Google's documented ODPM
microjoule/millisecond ABI; the exact installed CPM driver revision is
unverified. Decimal instantaneous power was preserved but not converted.
Counters had zero TPU energy delta in the nearby idle baseline; that means
zero reported accumulated energy, not proof of literally zero leakage.
Rail totals exclude host CPU/shared-memory work and can include other clients.
Whole-phone power includes radios, display, diagnostic collection and other
work. The changing after-baseline precludes a clean causal subtraction.

Samples cover 20.108 seconds inside the workload window. Sampling does not
cover every request boundary exactly, so no per-token/per-answer energy is
claimed. These five warm repetitions do not establish sustained thermal
behavior, cold-start latency or an optimal model size.

The separate Ostadix HTTPS bridge check returned `1 + 1 = 2` in 0.845 s with
native execution evidence. No fresh Gemini Nano request was performed.

Evidence: [analysis](analysis.json), [request results](requests.json),
[capacity observation](capacity.json),
[Ostadix check](assistant-check.json), and mapped-library evidence
[before](process-before.json)/[after](process-after.json).
The raw root telemetry stream is excluded from the public source archive.

The script copied for this run preceded later thermal-unit and native-metadata
attribution hardening. Its SHA-256 is recorded in `analysis.json`. Native token
counts and elapsed times were rechecked against the returned job usage.
Ignore interpreted thermal temperatures for types other than `battery` and
`soc_therm`; the raw readings remain available. The current harness preserves
unverified thermal units rather than inventing a conversion.
