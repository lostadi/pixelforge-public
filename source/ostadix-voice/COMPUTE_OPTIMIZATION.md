> Historical implementation notes from the earlier device integration.
> These do not describe the current default engine or establish current
> performance. Associated local runtime reports are not distributed here;
> use [the current source guide](README.md) for setup and supported entrypoints.

# Compute optimization results — 2026-10-06

Implemented:
- Agent instructions request compact JSON. This is a prompt hint, not enforced schema decoding.
- Bounded exact rational arithmetic in exact_compute.py, exposed as a calculate evidence tool.
- Tasks prefixed calculate: execute with zero model calls, even if the model is stopped. Cancellation takes precedence. Natural-language requests are not automatically classified as arithmetic.
- Removed one rendered-prompt allocation and copy per native request.
- Added host preflight, generation and prompt-token timing to native stderr.
- Rebuilt and installed the resident; updated its pinned SHA256. Baseline source, binary and manifest are retained in runtime/npu/optimization.

Measurements:
- 1,000 arithmetic calls: median 0.081328 milliseconds for (123456789*987654321)/9+0.1+0.2, exact result 135480701236261413/10. Excludes interface and task startup.
- Unconstrained JSON arithmetic: 1.208288 seconds, 16 output tokens.
- Schema-constrained same question: 8.124824 seconds, 6 output tokens. Preflight 7.621019 seconds, generation 0.503778 seconds.
- Two repeated constrained requests: 7.344118 and 7.344924 seconds. Constraint setup remained expensive, so constraints are disabled in production.
- Conversation cloning failed with UNIMPLEMENTED: GetRuntimeConfig not implemented for backend: LiteRT NPU Compiled Model. Cross-request prefix caching was not installed.
These are small diagnostic trials, not a controlled throughput benchmark. No end-to-end speedup is claimed for the copy removal or compact-JSON hint.

Backend limits:
- Context preflight still renders and tokenizes before send_message renders internally. Removing this duplication requires an appropriate API that preserves chat semantics and context validation.
- GetProfileSummary is unimplemented for this backend. Host timing does not provide per-operator NPU timing.
- The model is precompiled for Tensor G5. No supported graph compiler/profile workflow was found in available capabilities. Fusion, tensor layout, quantization and precision were not modified.
- CPU auxiliary operations exist, but no measured evidence identifies a particular auxiliary operation as the bottleneck.
- Reusing a mutable conversation across independent tasks risks leaking context; it does not replace verified prefix-state cloning.

Validation:
156 tests passed across arithmetic, local agent/actions, terminal/conversation handling, NPU runtime and native protocol.
Native compilation with -O2 -Wall -Wextra succeeded without diagnostics.
Experimental execution exercised the prompt-buffer and timing changes against the actual NPU library.
Production resident was verified ready after installation; no additional manual generation was submitted after installation.

Usage:
In the terminal: /agent ask calculate: 0.1+0.2
From the app root: python local_agent.py ask --workspace ./example-workspace 'calculate: 0.1+0.2'

Evidence:
runtime/npu/optimization contains experiment logs, JSON results, arithmetic benchmark and deployment record.
benchmark.py and benchmark_constraints.py temporarily stop and restore the resident and submit four and two requests respectively. Run deliberately while idle.


## Follow-up implementation and live validation

- Removed double JSON encoding of request history. A representative two-request history shrank from 133 to 106 characters with identical decoded requests. This is a character-count comparison, not a measured token or latency reduction.
- Compacted evidence JSON serialization while preserving string content, the complete host journal and duplicate-request checks. Existing evidence-preview truncation remains a limitation.
- Changed the runtime instruction from "Return concise plain text" to "Follow the requested response format; be concise" to avoid contradicting agent JSON requests.
- Restarted the idle resident to load the adapter change and verified readiness.
- 160 relevant regression tests passed.
- Live task 74e4a7f532854a8a9967c9378af83faf completed two NPU calls: requested optimization-evidence.json (1.082565 seconds, 14 tokens), then correctly identified generation as dominant and reported measurement limitations (2.330662 seconds, 36 tokens). Host performed the read; the model did not directly access files.
- A separate local design review returned a generic persistent-context suggestion. It was not adopted as evidence that unsupported cache APIs work.

Production measurement:
48 successful requests in the then-current resident log had median host preflight 0.0383555 seconds and median generation 1.2949515 seconds. Host preflight represented 2.5128% of aggregate measured phase time. Generation includes prefill and decode. Mixed workloads and device conditions prevent causal before/after conclusions. Raw samples are preserved in runtime/npu/optimization/production_timings.json.

Scope and next steps:
The resident model, bounded evidence workflow, exact edit/test proposals, deterministic arithmetic and separately enabled optimizer are implemented. Model-proposed actions remain host mediated. Ordinary spoken input still routes to the main assistant; these changes do not establish autonomous local voice handling or arbitrary NPU-executed tools.
Conversation cloning and detailed profiling remain backend blockers. Graph fusion, precision changes and KV-state caching require a supported model/runtime/compiler route and validation; none is claimed implemented.
A useful next experiment is a matched workload benchmark of shorter task/evidence formats, measuring correctness, output tokens and total latency together. Prompt reductions alone are not proof of speedup.
The removed four-minute local task deadline was separate from the reported ten-minute assistant interruption. No ten-minute platform limit has been removed.
