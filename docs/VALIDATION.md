# What the checks establish

PixelForge's evidence follows the execution boundary. A model name identifies
weights. A compilation result identifies an artifact. A successful invocation
identifies an operation that ran. The application result then establishes the
particular task that completed.

## v0.2.0: ordinary Termux tool acceptance

I checked the full conversation path on the rooted Pixel 10 Pro with Tensor
G5. The client ran as ordinary Termux UID/GID 10402 in the
`untrusted_app_27` SELinux application domain; its exact app context was
verified. The fixed root-owned inference host remained a separate boundary.

The model chose a `workspace` read request. The host validated and executed
that request, returned the real file result, and the second model turn
returned the requested numeric value from the file. Both turns completed.
The host used no intent router or synthesized answer for this demonstration.
Private file paths, prompts, session identifiers and the probe value are
omitted here.

The run took **1007.335 seconds, about 17 minutes**. This is a slow functional
acceptance test, not a CPU performance comparison. It used the
`original_columns` route with all **249 main Q5_K/Q6_K projection tensors**
selected for NPU in both phases, B64 prompt programs and B1 decoding.
CPU-placed weight and CPU-prefill weight counters were zero on both turns.
Attention, recurrent GDN, normalization, sampling and ordinary FFN elementwise
operations remained on CPU; column partial products were added on CPU in
ordered F32 arithmetic.

| Model turn | Prompt tokens | Reused prefix | Prompt tokens processed | NPU prompt weight operations | NPU decode weight operations |
| --- | --- | --- | --- | --- | --- |
| Tool request | 718 | 0 | 718 | 497 | 3984 |
| Answer after the tool result | 888 | 512 | 376 | 249 | 4482 |

Those are **9212 logical NPU weight operations** across the two turns. The
runtime separately reported **28,404 native invocations**. The driver counter
advanced by **28,478**, leaving a difference of **74** that has not been
resolved. The two counts are not asserted to be equal, and a driver counter
alone does not establish ownership of every operation.

The source GGUF stayed unchanged, with SHA-256
`cc4a3cc480fe0e0142333244aaa3f2c0d4feda6c78d296976e8513d02aabbbcd`.
The source was Huihui NeoHorse 1 4B. Original integer levels and group
coefficients were preserved, including exact zero extension of a split
column tail. This introduces no additional weight quantization. It does
not imply bit-identical CPU and NPU arithmetic: specific primitive reference
checks measured relative L2 errors of roughly **0.0003 to 0.00084**. Those
measurements are not a full-model error bound or general quality result.

The acceptance profile used the following settings. Later residency tuning
is separate from this recorded run.

| Setting | Tested value |
| --- | --- |
| Native resident-program limit | 2 |
| Known native program/port budget | 128 MiB |
| Retained shared-weight budget | 5632 MiB |
| Prefix snapshot budget | 256 MiB |
| CPU affinity | 2, 3, 4, 5, 6, 7 |
| Vendor performance preference | `default` |

The second turn reused **512 prompt tokens** from one retained snapshot
charged at **69,483,380 bytes**. It still processed 376 prompt tokens and
reported only the NPU work actually performed. Snapshot reuse establishes
that this repeated prefix was restored; it does not establish a speedup
against a fresh prompt or CPU baseline. Main turns retain no elapsed deadline,
and explicit cancellation remains available.

This test establishes the selected model's ordinary-app tool path on this
phone and profile. It does not establish broad model quality, faster-than-CPU
execution, battery savings, sustained thermal behavior, another phone's
compatibility or reboot recovery. Read
[the original-group mechanism and limits](../tools/arm_npu/ORIGINAL_GROUPS.md).

### Residency tuning and selected phone profile

I separately compared resident-program limits of two and eight in one matched
short pair. Each case ran a first request and a second request, using 17 prompt
tokens and matching two-token outputs. All 249 main projections were selected
for NPU in both phases, with zero CPU-placed weight operations on both turns.
Each case had a 512 MiB parent-side pressure allocation. That allocation was
recreated between cases, rather than held continuously across the pair.

| Resident-program limit | Request | Prompt processing, s | Decoding, s | Generation total, s |
| --- | --- | --- | --- | --- |
| 2 | First | 63.2438 | 23.9248 | 87.1686 |
| 2 | Second | 30.1819 | 23.2406 | 53.4225 |
| 8 | First | 60.2656 | 18.5461 | 78.8117 |
| 8 | Second | 26.2794 | 19.3429 | 45.6223 |

The second-request generation time was **14.6% lower** with a limit of eight
in this pair. This is one short comparison, not a repeated performance study,
a CPU comparison or a repeat of the complete file-tool task. The table measures
generation only; the 1007.335-second acceptance record above remains unchanged.

The owned helper counters provide separate mechanism evidence for that pair:

| Counter or retained state | Limit 2 | Limit 8 |
| --- | --- | --- |
| Runtime program loads | 742 | 31 |
| Runtime program evictions | 737 | 0 |
| Original-group unpacks | 814 | 712 |
| Retained DMA weight bytes, approximate | 4.459 GB | 4.815 GB |

The eight-program case retained eight programs with **72,043,968 known
charged bytes**, below its 128 MiB native budget. The 1 GiB memory-headroom
guard remained enabled. Source weights stayed unchanged, no programs were
compiled during these requests, and broker/native cleanup passed. Driver
counter deltas are not used as exact owned invocation counts for this tuning.

The selected phone profile explicitly sets `native_max_resident_models=8`.
The generic installer default remains two. This selection is based on the
matched pair, with the existing bounded cache and memory-headroom checks
retained.

### Installed application check

After installing the selected profile, I checked the actual application's
default metadata and fixed launcher from ordinary Termux UID 10402 in its
verified SELinux app domain. Installation verification matched all 76
recorded source pins. The short request completed cleanly in **74.676 seconds**;
readiness was observed at **6.445 seconds**. It used 17 prompt tokens and
returned the requested two-token response.

The installed profile used a program limit of eight, B64 prompt programs and
B1 decoding. Both 249-weight phase selections matched the configuration.
The response recorded **249 NPU prompt weight operations and 249 NPU decode
weight operations**, with zero CPU-placed and CPU-prefill weight operations.
Prompt processing took **52.886 seconds** and decoding **14.018 seconds**.
The model hash matched the unchanged GGUF above, and the process exited with
status zero.

This establishes the installed default application's short inference route.
The complete model-chosen file-tool acceptance remains the separate
1007.335-second cap-two run; the installed cap-eight smoke is not a repeat
of that task.

## Historical v0.1.0 hybrid phone checks, 2026-10-29

Device: Pixel 10 Pro, Tensor G5, approximately 16 GB physical RAM. Requests
entered from ordinary Termux through the fixed root-owned inference host.
No cloud model or microphone was used in these tests.

| Check | Observed result | Scope |
| --- | --- | --- |
| Huihui Gemma 4 E2B arithmetic | `5`; 277 NPU weight operations; hardware counter advanced | CPU prefill, all planned dense decoding projections on NPU; 62.3 s generation call |
| Huihui NeoHorse 1 4B arithmetic | `5`; three NPU weight operations; hardware counter advanced | Three first-block feed-forward projections on NPU, remaining operations on CPU; 7.1 s generation call |
| NeoHorse file-tool loop | Real file read, host result and exact final answer | Two model turns; 214.6 s including startup and orchestration |
| Owned-host cancellation | Host exited; job remained cancelled | Separate generation; approximately 0.42 s cancellation |
| Installed `c` startup | Model displayed; text interface exited cleanly | Startup/status/shutdown; no physical audio test |

CPU prefill also produces the logits for the first output token. The NPU
counts establish subsequent decoding operations; the short arithmetic answers
alone do not attribute their visible digit exclusively to the NPU.

The file-tool loop made 72 NPU weight operations and 168 native invocations.
Seven saved banks were loaded, zero banks were compiled during inference, and
native cleanup succeeded. Driver activity was observed alongside those owned
successful calls. Device counters can include other TPU clients, so the counter
alone is not the attribution.

These are narrow functional checks. They do not establish a general quality
ranking, CPU speedup, battery savings, sustained thermal behavior, installation
on another phone or reboot recovery. Those NeoHorse checks used hybrid placement.

The public [reference record](../chat/REFERENCE_ACCEPTANCE.json) contains the
historical model identity, placement, counts and timing. Private workspaces, conversations,
process IDs and installation paths are excluded.

## v0.2.0 source and fixture checks

A focused integration pass completed **122 tests**: 61 service/preflight,
19 installer and 42 voice-transport checks. These include source-pinned
column aliases, physical zero-padding charges, the B1 prompt/decode union,
artifact tampering, trusted CPU masks, Termux home SELinux label inheritance,
strict prefix-cache receipts and
legacy profiles with caching off. They use synthetic models or mocked
process boundaries and do not invoke the phone NPU.

The compiler/runtime fixtures also cover lossless original groups, shared
buffer ownership and streaming admission, source binding, selected-operation
failure, phase routing and ordered column sums. Native sequence-cache
fixtures compare fresh and restored prompt state, including recurrent state.
Fixture results and a live task establish different parts of the path.

## Historical v0.1.0 public build and fixture checks

The fresh public native build and isolated public installer also passed the
file-tool loop on this phone. The ordinary Termux client read a real file and
returned its exact contents across two model turns: **66 NPU matrix operations,
154 native invocations**, seven saved banks loaded, zero compiled during
inference and successful native cleanup. The record pins the tested inference
sources; one later trailing-blank-line cleanup preserves the identical Python
AST. The dependency versions match the candidate's Python 3.12 requirements.

That run took **707.9 seconds** under concurrent phone load and thermal
throttling. It is a functional check, not a performance comparison. It used
existing model banks, the device helper and Ubuntu runtime. See the
[public build hardware record](evidence/public-source-hardware-20261029.json)
for hashes and limits.

The release adds source-boundary validation, deterministic archives,
installation-plan tests and native protocol fixtures. Chat fixtures cover
persistent requests, explicit placement, unknown tensors, failed selected NPU
requests, root-process ownership and cleanup. Simulated brokers test protocol
behavior; they are not hardware inference evidence.

The current offline compiler fixtures also cover fixed-batch padding,
geometry reuse, source-specific weight binding, lossless original groups,
cross-phase buffer keys and partial-upload failure. The actual helper is
compiled against mock vendor APIs to check sticky inputs, DMA descriptor
ownership, generation invalidation and cache eviction without Android or
NPU access. Optional complete F32 FFN fixtures establish graph and protocol
behavior; they do not enable whole original-group FFN fusion.

The voice source includes the conversation coordinator, PixelForge transport,
tool validation, cancellation and terminal tests. Offline tests use temporary
fixtures. Live `integration_*.py` scripts are separate and require an explicit
phone run. CI does not download model weights or invoke a cloud API.

The complete application suite ran **1,110 tests: 1,100 passed and 10 skipped**.
The Android transport fixtures passed 10 tests and the native C protocol stub
passed six. The native/chat suite passed 19 tests, including isolated root
ownership and cleanup checks. Python 3.11 syntax parsing passed for every
exported Python source file. These fixtures do not establish physical audio
capture or playback.

Earlier compiler checks remain under `benchmarks/`, with their original dates
and limitations. The old [source snapshot report](evidence/source-snapshot-20261012.json)
is historical evidence, not validation of the current release.
