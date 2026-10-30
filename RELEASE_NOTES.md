# PixelForge release notes

## v0.2.0: original weights and the full phone tool path

I extended the NPU path to use the original GGUF integer levels and group
coefficients as weight inputs. The column route takes successive 3072-column
spans and zero-pads only a shorter split tail to that program width. NPU
programs reconstruct the weights and compute each partial product; the host
adds the partials in order using F32 arithmetic. The source GGUF stays
unchanged and no additional weight quantizer is introduced. Vendor float
execution and the ordered sums still have numerical error: the measured
primitive comparisons were roughly 0.0003 to 0.00084 relative L2, not global
bit identity with CPU execution.

I checked this path through ordinary Termux in its normal SELinux app domain.
The model selected a real workspace file read, received the host result and
returned the requested value on a second model turn. Both turns selected all
249 main Q5_K/Q6_K projections for NPU prompt processing and decoding, with
zero CPU-placed weight operations. Attention, recurrent GDN, normalization,
sampling and ordinary FFN elementwise operations remain on CPU.

The test took **1007.335 seconds, about 17 minutes**, with a resident-program
limit of two. It is still slow. It establishes this model, hardware path and
tool loop together; I am not claiming that it is faster than CPU inference.
The runtime reported 28,404 native invocations while the driver counter
advanced by 28,478. The unresolved difference is 74; the counts are not
treated as identical.

I separately compared program limits of two and eight with matching short
requests: 17 prompt tokens, two output tokens and all 249 projections selected
on NPU in both phases. Each case had 512 MiB of parent-side pressure, recreated
between cases. Second-request generation took 53.4225 seconds with a limit of
two and 45.6223 seconds with eight, a 14.6% reduction in this one pair. I selected
eight for the phone profile; the generic default remains two. This does not
replace the 1007.335-second full tool record or establish a general speedup.
Owned helper loads fell from 742 to 31 and evictions from 737 to zero. The
eight-program case charged 72,043,968 known bytes against its 128 MiB native
budget, retained the 1 GiB headroom guard and completed cleanup without
compiling programs during the requests.

I then checked the installed application's default route from ordinary
Termux. Its short 17-token prompt returned the requested two-token response
in 74.676 seconds, with 249 NPU prompt and 249 NPU decode weight operations,
zero CPU-placed weights and a clean exit. All 76 installed source pins matched.
This checks the actual installed cap-eight route. The full model-chosen
file-tool result above remains the cap-two run.

The optional prompt-state cache retained CPU attention and recurrent state
for an exact token prefix. The second turn reused 512 tokens from a
69,483,380-byte snapshot within its 256 MiB opt-in budget. Caching defaults
to off. Failed restores clear state and start fresh prompt processing.
Shared weight admission retains accepted buffers and streams overflow;
device program, weight-buffer and prefix budgets stay separate. An optional
CPU mask applies only to the inference service and its children. Main turns
retain no elapsed deadline, and explicit cancellation remains available.

The focused service, installer and voice pass completed 122 tests. The
[validation record](docs/VALIDATION.md) separates the new ordinary-app test,
its measured limits and the historical v0.1.0 checks. Model weights, compiled
programs and vendor binaries remain separate inputs. Upstream notices and
the LGPL-2.1-only license for original PixelForge code are retained.

Lee Daghlar Ostadi

## v0.1.0: public source release, 2026-10-29

I have brought the working model compiler and the phone conversation path into
one reviewable source release.

The new chat runner keeps the model and broker resident, applies the model's
chat template, supports explicit CPU/NPU placement and carries per-turn
operation counts back to the host. The `c` adapter uses the existing host tool
dispatcher. Model output is validated before it becomes an action, and an NPU
failure does not trigger an unannounced model substitution.

The selected phone profile is Huihui NeoHorse 1 4B. Three feed-forward
projections run on the Tensor G5 NPU; the rest of the model runs on CPU. A live
file-read task completed through two model turns with 72 NPU weight operations
and 168 native invocations. It took 214.6 seconds. This establishes that the
model, device path and tool loop work together; overall quality and speed
remain separate questions.

I also checked the fresh public source build through an isolated installation.
It completed a real file read and exact reply with 66 NPU matrix operations,
154 native invocations and successful cleanup. The run took 707.9 seconds under
concurrent phone load. The complete application suite ran 1,110 tests:
1,100 passed and 10 skipped.

This release includes native source, a configurable installation plan, voice
adapter, tests, provenance and source-only packaging. It builds the decoder
from pinned llama.cpp source. Model weights, compiled banks and vendor runtime
binaries remain separate inputs.

Private device-restoration files are outside the public source boundary.
Original PixelForge code and documentation are LGPL-2.1-only, with upstream
notices retained.

Lee Daghlar Ostadi
