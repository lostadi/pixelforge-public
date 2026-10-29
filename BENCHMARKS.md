# Functionality, capacity and power

The [compiler and launcher stress report](benchmarks/20261013-stress/README.md)
records repeated saved-code NPU execution, cache evictions, process cleanup,
malformed containers, cache races and protocol fault cases. These bounded
checks do not measure maximum model capacity or an optimal model size.

The intended deployment uses user-selected **huihui-ai abliterated weights**.
Google-supplied precompiled models are hardware baselines only. The existing
Gemma benchmark does not prove that an arbitrary checkpoint can be compiled
or that arbitrary Huihui models run on the NPU. The complete MiniCPM5 Huihui
checkpoint now has [fresh native NPU validation](benchmarks/20261013-native-huihui/README.md)
and a [separate offline runner](ARM_NPU.md). See the [custom model notes](CUSTOM_MODELS.md)
for the conversion prerequisites and evidence needed before making that claim.

The [October 13 GGUF compiler validation](benchmarks/20261013-gguf-compiler/README.md)
adds persistent native-code reuse: a fresh read-only process loaded 16 saved
banks with zero compilation, made 76 NPU calls, and matched four synthetic
CPU token choices with all 1,036 logits compared. It also records both matching
CPU token choices from the real Huihui LFM2.5 model, covering prefill and
one recurrent decode step. Shared device counters, concurrent load and int8
requantization limit the interpretation; this is
functionality evidence rather than a model-quality or power benchmark.

The [October 12 hardware baseline](benchmarks/20261012T222832Z-google-baseline/README.md)
completed nine NPU requests and captured simultaneous power readings. Its
strict format suite passed two of three checks. This fresh experiment is
separate from the earlier static snapshot validation.

`benchmark.py` checks the existing Gemma LiteRT NPU service. It never enables,
restarts or replaces it. The default command only inspects the profile and
hardware; inference requires `--run`. Run it from ordinary Termux with the
deployed `ostadix-voice` source available:

```sh
python benchmark.py
python benchmark.py --run --output "$HOME/pixelforge-benchmark.json"
python -m unittest test_benchmark.py
```

The offline bundle directory is root-owned and private. To run from Termux,
copy just `benchmark.py` to a directory owned by that app, or use the GitHub
checkout. Do not change the offline payload's permissions.

The suite sends its own fixed prompts through `LocalClient` under a unique
owner, performs a warm-up, checks arithmetic, exact output and structured
extraction, then repeats a short generation prompt. It records correctness,
queue/service/client latency, actual completion tokens and resource samples.
It waits for an idle service and stops if a request fails or remains pending.
The initial suite measures warm inference; it does not measure cold loading,
long-context quality, maximum concurrency or sustained thermal performance.

Android may deny power/sysfs reads from ordinary Termux. In a separate
authorized Android root terminal, collect read-only samples while the ordinary
Termux benchmark runs:

```sh
python benchmark.py --collect-seconds 120 --pid ENGINE_PID --output new-power-samples.json
```

Use the engine PID returned by the service's status, not an old PID from a
report. This collector never connects to or changes the service. Its monotonic
timestamps can be aligned with the request report. The `energy_delta` helper
converts ODPM pairs only with explicit `abi_assumed=True`; retain that
qualification when reporting rail watts or joules. Missing access stays
missing in the report; no Android permissions are changed.

Native decode speed and time to first token come from newly appended benchmark
metadata in `engine.log`. They are attributed only when a single matching
request block is observed without competing queue activity or PID changes.
Tokens divided by total request time measure effective throughput and include
prefill and other work. They are not pure decode speed. The short correctness
checks are a smoke test, not a general model quality evaluation.

## Capacity and model choice

The installed profile is Gemma 4 E2B IT compiled for Tensor G5, with one active
request, a 1,280-token context and at most 512 output tokens. The native host
requires rendered prompt tokens + reserved output + 8 control tokens to fit
within that context. The model family's advertised context is not this
deployment's context limit.

Model file size, process RSS and device memory are different measurements.
Inspect DMA buffer allocations as well: the NPU's driver buffers can be much
larger than RSS. Deduplicate DMA buffer descriptors by inode; duplicated file
descriptors refer to the same allocation. Device-wide buffer totals already
include the process buffers and must not be added to them. Zram stores
compressed pages in physical RAM; swap capacity is not additional physical RAM.

There is no measured optimum until compatible models are compared on the
same task set, context, output budget, device temperature and background load.
Keep the existing E2B result as a hardware baseline. A smaller vendor comparison is the
publisher's [Gemma 3 1B Tensor G5 artifact](https://huggingface.co/litert-community/Gemma3-1B-IT/tree/a6306a4e292016480083b73b8dc6f3f939ae04c3),
1,678,542,365 bytes. [Google's NPU guide](https://developers.google.com/edge/litert/next/litert_lm_npu)
also lists the Tensor G5 1B model. Lower memory and energy use are hypotheses
to test; parameter count alone does not establish throughput or task accuracy.
These Google-published artifacts are not the user's requested deployment models.

The installed E2B model's [pinned publisher revision](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1)
contains its exact G5 artifact. The inspected [E4B publisher revision](https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm/tree/2eee7ac325f20eb8c9ac1d0e972f7c84663062da)
has generic/GPU/web artifacts and no Tensor G5 NPU artifact. E4B is therefore
not an established drop-in NPU upgrade for this bundle. Arbitrary larger models
cannot be judged compatible from RAM size alone.

Choose the smallest compatible model that meets a representative correctness
target, then compare median/p95 latency, memory headroom, joules per successful
answer and sustained performance. These measurements can identify an optimum
for an interactive assistant or a battery budget; neither goal has a universal
best parameter count.

## Power interpretation

Battery `current_now` is in microamps and `voltage_now` in microvolts under the
[Linux power supply ABI](https://www.kernel.org/doc/html/latest/power/power_supply_class.html).
Their product divided by 10^12 gives net battery power in watts. Report charging
state and sign alongside it. Battery discharge measures the whole phone,
including screen, radios, CPU and this diagnostic session. Charging data cannot
be interpreted as inference consumption.

This device also exposes `cpm:ODPM` rail telemetry at
`/sys/bus/iio/devices/iio:device0`. Its enabled rail map names
`CH6[S7M_VDD_TPU]` and `CH8[S9M_VDD_TPU_M]` as TPU rails. Raw `energy_value`
uses the same timestamp/channel format documented by Google's
[ODPM driver](https://android.googlesource.com/kernel/google-modules/soc/gs/+/204078758737776959ecc85c527a00fd2cabb62f/drivers/iio/power/odpm-whi.c):
milliseconds and cumulative microwatt-seconds (microjoules). Converting this
device's counters uses that matching ABI; the exact installed CPM driver
revision has not been independently verified. Preserve raw values and this
qualification. Its decimal `inst_power` units are not verified and should not
be silently converted into watts.

For two valid cumulative samples, energy is the sum of the two rail deltas
divided by 10^6, and mean power is that energy divided by elapsed seconds.
Reject a counter reset, changed rail mapping or measurement duration reset.
Report both the absolute rail draw and any subtraction of a nearby idle
baseline. Rail energy includes all TPU clients and excludes auxiliary CPU and
shared-memory work. Device-wide inference counters are supporting evidence;
correlate them with successful NPU-only responses, stable engine identity and
mapped runtime/dispatch libraries. Some exposed counters remain zero.

## Other routes

The Ostadix pinned HTTPS host's `ostadix_assistant_service.py check` executes
`1 + 1` and tests the native program bridge. It does not test Gemini Nano.
Nano/AICore requires a separate explicitly armed Android probe and Google's
running process. Its quality, latency and power should be reported separately.
The current chroot cannot open Android Binder, so no Nano inference is claimed
by the Gemma benchmark.
