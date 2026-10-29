# Compiler and single-file launcher stress validation

These checks exercise the custom GGUF compiler, its saved NPU code and the
single-file Ostadix launcher on the original ARM64 Tensor G5 phone. They use
synthetic inputs and the existing small compiled Llama fixture. The user-selected
Huihui models' separate inference evidence remains in the earlier benchmark
directories.

The ten fresh hardware runs made **2,156 attributed NPU invocations**, loaded
764 saved graphs, and exercised 645 LRU evictions without compiling a graph.
All 104 generated token choices matched the same CPU engine. The 26,936
full-vocabulary logit values had relative L2 errors between 0.01447 and 0.01690.
Repeated runs and cache-budget changes produced identical NPU logits. Every
observed native helper exited, its TPU client disappeared, and firmware crash
and watchdog counters had zero increases. The original Gemma service remained
enabled and ready with zero restarts.

| Check | Scope | Result |
| --- | --- | --- |
| GGUF reader | 1,500 container cases and 700 randomized decode cases across 35 formats | Zero unexpected errors |
| Artifact cache | 120 races, tamper checks, fault/retry, relocation and CPU-stub cases | 120 passed |
| Native protocol | 120,120 deterministic cases, executed normally and under UBSan | 240,240 checks passed |
| Native cleanup | Seven mocked real-Model cleanup scenarios, eight repetitions each | 56 passed |
| Tiny NPU execution | Ten fresh runs, CPU threads 1/2/4, cache budgets 0/1/4 MiB, two prompts | 2,156 owned calls; cleanup passed |
| Cache initialization after the fix | Fresh two-token hardware run with existing bytecode | 44 owned calls, 16 loads, zero compiles |
| Final single-file launcher hardware check | Four-token run through the rebuilt `.O` | 76 owned calls; all 1,036 logits matched the prior NPU reference exactly |
| Launcher extraction locking | Eight cases with six concurrent dummy-kit creators | One extraction per case; tampering rejected |

The final single-file launcher checks and exact artifact hashes are recorded
in [VALIDATION.json](VALIDATION.json). Detailed hardware results are in
[RUNTIME.json](RUNTIME.json); reader, artifact and native reports are separate
JSON files alongside it.

The ten-run stress, cache-fix smoke check and final launcher hardware check
made **2,276 attributed NPU calls** in total. The last two checks add 120 calls
to the ten-run window; their counts are kept separate in the JSON reports.

Two defects found during stress testing were fixed. A failed first `MODEL.json`
publication previously left an output directory that prevented a clean retry;
initial identity now commits before auxiliary directories under the directory
lock. Long Unicode arguments previously exceeded Linux's limit for a single
environment string; the launcher now uses a private JSON request file. ASCII
JSON preserves Unix filenames containing non-UTF-8 bytes through both the
request and Ostadix's result transport. Regression tests cover these cases.

AddressSanitizer could not start on this host. Its GCC runtime failed its shadow
allocator mapping before `main`, and the alternative Clang installation lacked
the ASan runtime libraries. UBSan ran successfully. This is partial sanitizer
coverage; the failed zero-work probes are retained in [NATIVE.json](NATIVE.json).

The hardware fixture is a 2.6 MB synthetic model, not a useful language model.
Its native helper's sampled peak RSS was 25,376 KiB; RSS excludes allocations
owned by other processes or the driver. The largest observed descriptor count
was 40. Fresh helper exit and TPU-client removal establish owned-process cleanup,
not the absence of every possible driver or long-term memory leak.

Whole-device TPU counters increased by 2,180 during the ten-run window; only
2,156 calls belong to these jobs. The extra 24 calls are unattributed. Battery
endpoint readings were 5.58 W and 3.97 W for the entire phone, at 37.9–38.1 C.
They do not establish NPU-only power, energy per token, sustained throughput,
maximum model size or an optimal model size. Attention and other architecture
operations still execute on the CPU. Parser/dequantizer coverage does not
establish support for every GGUF architecture or full-model NPU execution.

From a prepared PixelForge checkout, repeat the CPU stress suites with fresh
report paths:

```sh
PYTHON=custom-models/gguf-runtime/venv/bin/python
"$PYTHON" tools/arm_npu/stress_gguf.py --report /tmp/reader-stress.json
"$PYTHON" tools/arm_npu/stress_artifacts.py --report /tmp/artifact-stress.json
python3 tools/arm_npu/stress_native_protocol.py --report /tmp/native-stress.json
"$PYTHON" -m unittest discover -s tools/arm_npu -p 'test_*.py'
python3 -m unittest test_capture.py test_bundle.py test_benchmark.py
```

Hardware stress requires an existing complete tiny AOT package, compatible
Android runtime and a fresh result directory. It refuses models above 16 MiB
and executes serially without restarting services:

```sh
"$PYTHON" tools/arm_npu/stress_runtime.py \
  --artifacts /path/to/existing-tiny-aot --bank-rows 192 \
  --output /tmp/fresh-runtime-stress
```

The single-file integration and its repeatable launcher stress driver are
documented in [the Ostadix integration guide](../../integrations/ostadix/README.md).
