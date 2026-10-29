# PixelForge through Ostadix

`pixelforge-npu.O` is an executable O program that invokes the installed
ostadix-BM evaluator and its registered Python backend. It embeds the GGUF
compiler kit, architecture engine, Android helper, offline Python wheels, and
the integration controller and worker. The kit is verified and materialized
into a persistent cache when needed. Supply your own GGUF model separately.

The O program orchestrates the existing native tools. Eligible weight matrix
multiplications execute on the Tensor NPU; the architecture engine performs
attention, normalization, sampling, and other operations on CPU. This does not
establish support for every GGUF architecture or lower these implementations
through O-core. See the [GGUF compiler guide](../../tools/arm_npu/GGUF_COMPILER.md)
for supported operations, precision, storage requirements, and validation limits.

## Run the published artifact

Keep `pixelforge-npu.O`, `pixelforge-npu.O.sha256`, and `pixelforge-npu.O.json`
together. From this directory in your Ubuntu root terminal:

```sh
sha256sum -c pixelforge-npu.O.sha256
chmod +x pixelforge-npu.O
./pixelforge-npu.O help
./pixelforge-npu.O verify
./pixelforge-npu.O setup
./pixelforge-npu.O doctor
```

Execution requires the installed ostadix-BM O evaluator and Python backend,
Ubuntu ARM64 CPython 3.12, and a compatible rooted Android Tensor vendor
compiler/runtime and Binder environment. The O evaluator and Android vendor
libraries remain external dependencies. Rebuild and native-test commands also
require their local build toolchains. Legacy MiniCPM tool aliases require their
separate model/tokenizer addon and Python environment.

Set `O_LANG_ROOT` to your ostadix-BM checkout when it is outside the standard
installation paths. `OSTADIX_NPU_O` can select its native O executable. The
launcher forwards the actual artifact path and arguments to that evaluator.

## Compile and run your model

```sh
./pixelforge-npu.O inspect /path/to/model.gguf
./pixelforge-npu.O plan --model /path/to/model.gguf --output /path/to/compiled
./pixelforge-npu.O compile --model /path/to/model.gguf --output /path/to/compiled
./pixelforge-npu.O run --model /path/to/model.gguf \
  --artifacts /path/to/compiled --read-only --prompt Hi --tokens 2
```

Planning estimates disk requirements without starting the NPU. Resume an
interrupted compilation with the same source, bank layout, and weight format
using `compile --resume`. Read-only generation requires every requested bank
to exist and refuses compilation on a cache miss. Compiled code is bound to
the matching source, graph author, compiler, and firmware fingerprint.

Use `cpu` for an explicit CPU reference, `compare` for fresh CPU/NPU runs, or
`compare-reports CPU_RESULT.json NPU_RESULT.json` for existing reports.
Matching tokens provide a bounded functionality check; they do not establish
full-logit equivalence or language quality.

Other actions include `extract`, `build-native`, `build-engine`, `package`,
`test`, `test-native`, `fixture`, `graph`, `collect`, and `tool`. `package`
exports the generic compiler kit as a fresh `.run` archive. `test` and
`test-native` run software fixtures without hardware calls. `collect` reads
hardware observations. Use `ACTION --help` for options; global `--kit-dir`,
`--python`, and `--deadline` options precede the action. Relative paths use
the caller's current directory. Ctrl-C requests cleanup of the command's
owned worker processes.

## Sources and rebuilding

The integration sources, builder, worker, tests, and full usage instructions
are in the local [tools directory](tools/). These commands assume your current
directory is `integrations/ostadix`. The external ostadix-BM checkout supplies
the O evaluator and backends; the PixelForge adapter sources remain here.
The underlying compiler sources are in [tools/arm_npu](../../tools/arm_npu).

To rebuild, select a compiler archive matching the builder's verified release
pin and choose fresh output and receipt paths:

```sh
export O_LANG_ROOT=/path/to/your/ostadix-BM
python3 ./tools/build_integrated.py \
  --archive /path/to/pixelforge-gguf-compiler.run \
  --output ./pixelforge-npu-new.O
```

The builder verifies the compiler archive, embeds its exact file inventory,
and writes checksum and provenance companions. A compiler release update
requires updating the builder's archive pin and rebuilding the O artifact.

The source checkout also includes `stress_integrated.py`. With an existing
verified kit, numeric Python environment, and synthetic GGUF smaller than
4 MiB, repeat its read-only launcher and process cleanup checks:

```sh
python3 ./tools/stress_integrated.py \
  --artifact ./pixelforge-npu.O --kit-dir /path/to/existing/kit \
  --numeric-python /path/to/existing/venv/bin/python \
  --model /path/to/tiny-synthetic.gguf --report /tmp/fresh-launcher-stress.json
```

The harness performs no installation or NPU inference. The committed
[validation report](../../benchmarks/20261013-stress/VALIDATION.json) separates
these software checks from the measured hardware runs.

Run the integration source tests on Linux with Python and pidfd support:

```sh
python3 -m unittest discover -s ./tools -p 'test_*.py'
```

These tests use software fixtures and owned subprocesses without NPU calls.

## License and provenance

[COPYING](COPYING) is a verbatim copy of the ostadix-BM project license.
The embedded compiler kit retains its separate third-party license files,
upstream source archive, and wheel licenses.

The adjacent checksum covers the complete O file. Its JSON receipt records
the artifact size and SHA-256, embedded compiler archive SHA-256, file count,
execution mechanism, and prerequisites. The compiler kit carries published
runtime asset pins and upstream provenance. Consult the release's accompanying
validation evidence for the commands actually tested; source recognition and
packaging integrity alone do not establish numerical correctness on hardware.

[The original validation record](tools/VALIDATION.json) retains the paths and
hashes from the tested build. [Relocation validation](tools/RELOCATION_VALIDATION.json)
records the source move into PixelForge and checks the relocated tools.
