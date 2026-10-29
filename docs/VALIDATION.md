# What the checks establish

PixelForge's evidence follows the execution boundary. A model name identifies
weights. A compilation result identifies an artifact. A successful invocation
identifies an operation that ran. The application result then establishes the
particular task that completed.

## Live phone checks, 2026-10-29

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
on another phone or reboot recovery. The selected NeoHorse path is hybrid.

The public [reference record](../chat/REFERENCE_ACCEPTANCE.json) contains the
model identity, placement, counts and timing. Private workspaces, conversations,
process IDs and installation paths are excluded.

## Source and fixture checks

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
