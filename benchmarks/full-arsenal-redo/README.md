# Column-completion reproduction

This source-only fixture compares the existing
`ColumnProgramCache.mark_complete` method with the compiler's
`mark_columns_complete` helper. Both routes use identical cache and graph
implementation files. The materializer rejects differences between the chosen
source and reference trees before staging the dependency closure.

`dimensions.json` records 281 observed tensor dimensions and five compiled-program
byte sizes. The workload generates a marker source file and deterministic
DGC0-shaped byte arrays. These arrays are verification test data and cannot
establish native compilation, NPU numerical behavior or model execution.

Each route constructs the same fixture and performs three fresh completion
passes. Process measurements include imports and fixture construction. Functional
stdout must match; operation counters go to separate evidence files. This is a
completion-workload comparison, not an end-to-end compiler or inference benchmark.

## Materialize and compare

Use Node.js and an explicitly selected Python with NumPy and FlatBuffers installed.
Provide a reference checkout containing the unchanged cache/graph implementation
and a source checkout containing the new compiler helper. Both can be extracted
source trees. Choose an output directory outside both trees; it must not exist.

```sh
node prepare_project.js --source-root /absolute/source-tree \
  --reference-root /absolute/reference-tree \
  --python /absolute/python --output /absolute/fresh-comparison
o optimize /absolute/fresh-comparison --route completion-forward \
  --receipt /absolute/forward-receipt.json --json
o optimize /absolute/fresh-comparison --route completion-reverse \
  --receipt /absolute/reverse-receipt.json --json
```

Repeat fresh optimize invocations in alternating route order when measuring.
Do not use a cached selection receipt as a new measurement. Keep other substantial
local work idle. Inspect completed receipts and exact output agreement before
interpreting the timings.

`PINNED_INPUTS.json` records source hashes, the selected Python binary hash and
NumPy/FlatBuffers versions. The generated O manifest declares every command input.
No model, vendor library, SDK or device helper is accessed. The original private
comparison staged 72 files; this materializer stages only the same eight imported
cache/graph files. That changes staging work, so it is not a bit-identical replay
of the original O project bundle.

## Independent Common Lisp coverage oracle

`coverage.lisp` derives canonical row/column geometry from `dense-plan.sexp` without
importing the Python planner. Run from this directory, or the generated project:

```sh
sbcl --script coverage.lisp
```

Through Ostadix, execute a Common Lisp block containing `(load "coverage.lisp")`
with its working directory set here. The oracle writes `lisp-coverage.json` and
prints its result: 465 logical tiles and five distinct geometries. The input
contains dimensions and quantization type IDs only. It contains no weights,
prompts, conversations or deployment state.

## Measured comparison

[RESULTS.json](RESULTS.json) records five fresh native O comparisons, exact output
agreement, source hashes, real-package operation counts, allocation replay
results and the separately checked Voice outcome. Timings apply to the complete
three-pass completion fixture process, including imports and setup. They are
not whole-compiler, native-kernel or generation measurements.

Invocation/eviction details now form a bounded recent tail; use exact counters
rather than the number of retained diagnostic rows. Compile/load history keeps
its existing full-history semantics.

## Telemetry allocation replay

Use the existing Python environment to account for the actual Python runtime
objects. Supply your own private, complete broker report; the tool emits only
aggregate counts, equivalence checks and memory measurements. It rejects
already-truncated invocation histories and refuses to overwrite an output.

```sh
python telemetry_replay.py --source-root /absolute/source-tree \
  --report /path/to/private/broker-report.json \
  --output /absolute/fresh-aggregate.json
```

The default cases replay the original history and 200,000 events. The allocation
boundary includes new containers and diagnostic dictionaries; shared field
objects that were preloaded from the report are excluded in both alternatives.
This is not process RSS or a latency benchmark.

For read-only inspection, Ostadix's SQL backend can query a report directly.
For example, replace the generic path below with a private local report:

```sql
SELECT
  json_extract(readfile('/path/to/private/broker-report.json'),
               '$.npu.npu_invocations') AS total_native_calls,
  json_array_length(readfile('/path/to/private/broker-report.json'),
                    '$.npu.invocations') AS retained_details,
  json_extract(readfile('/path/to/private/broker-report.json'),
               '$.npu.invocation_trace.dropped') AS dropped_details;
```

Legacy complete-history reports have no dropped-details field. Current reports
state their retention policy explicitly. Logical weight operations and physical
native invocations are separate counters.
