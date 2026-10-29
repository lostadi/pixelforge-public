# PixelForge v0.1.0: public source release, 2026-10-29

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
