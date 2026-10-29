# Ostadix Voice source

A terminal and optional browser conversation interface with validated local tools,
continuous Android voice, and selectable PixelForge, local CPU, Gemini, and Codex
engines. This directory contains source and offline tests. Models, compiled NPU
banks, runtime libraries, credentials, signing keys, conversations, and device
configuration are supplied locally and are not included.

## Install the source shortcut

The Python application uses Python 3.11 or newer and the standard library on
Linux or Android Termux. Termux is required for the Android voice and phone
controls. Use the ordinary Termux account for the interface; the root inference
launcher is installed separately.

From a fresh repository checkout:

```sh
python3 source/ostadix-voice/install_shortcut.py --shell both
```

This creates `~/.local/bin/c` and appends a managed alias to `.bashrc` and `.zshrc`.
Open a new shell or use the alias printed by the installer. Existing commands,
aliases, functions, and symlinked shell configuration are preserved; a conflicting
installation stops with an explanation. `--shell none` installs only the command,
and `--bin-dir` selects another destination. Installing the shortcut does not
start an engine, install Android packages, download models, or record audio.

Choose an existing project directory on first launch:

```sh
c --no-audio /absolute/path/to/project
```

`--no-audio` provides a text-only interface. Ordinary `c` requests continuous
voice, which additionally requires the Android helper described below. State is
created under `~/.local/state/ostadix-voice`; it is separate from the file workspace.
A normal launch verifies the OSTADIX runtime/MCP pair and may fetch/build a
missing pair. `--offline` requires an already verified pair and avoids that setup;
`c --update-ostadix` explicitly updates it. OSTADIX setup needs Git, Rust/Cargo,
a compiler toolchain, and the upstream repository's build dependencies.

## Configure an inference engine

- **PixelForge:** follow the repository's [chat setup](../../chat/README.md).
  Its installer consumes a local model profile and generates
  `runtime/pixelforge-chat.json`; no active phone profile is distributed here.
  Select `/engine pixelforge` while idle after installation. The interface
  validates a completed response's model identity and positive NPU operation
  count before allowing its tool request to proceed. Explicit hybrid placement
  means selected dense projections run on the NPU while other dense operations,
  attention, normalization, sampling, and prompt processing can remain on CPU.
  Failures of selected NPU projections are errors, not automatic CPU fallback.
- **Local CPU:** `python3 source/ostadix-voice/setup_local_model.py --help` lists
  pinned model profiles. Running the installer can download weights and build
  llama.cpp; it is an explicit, separate step. The resulting local manifest stays
  under the ignored `runtime/` directory. Select `/engine local` after setup.
- **Gemini:** requires Node.js and an existing compatible Gemini CLI credential
  store. This source does not include credentials or create an account. Select
  `/engine gemini` only when that cloud engine is wanted.
- **Codex:** requires the locally installed Codex CLI and the user's own sign-in.
  Select `/engine codex` when that engine is wanted.

`/controls` reports the selected engine, microphone, speaker, and action policy.
`/cancel` cancels the current task, `/stop` controls audio, and `/quit` saves and
closes the interface. Main conversation generation has no application elapsed
limit; helper jobs, startup, and individual tools retain their own limits. Tools
execute through the ordinary host account under the saved action policy. The inference host does not execute model-requested tools; the ordinary host
controls any separately approved root commands.
The interface can remember automatic approval; review `/approvals` before using
it with a workspace containing important files.

## Optional Android voice

The helper source is in [android_voice](android_voice/README.md). Building it needs
a JDK, local Android SDK platform jar, `d8`, `aapt2`, and `apksigner`. Set
`ANDROID_JAR`, or point `ANDROID_HOME`/`ANDROID_SDK_ROOT` at a local SDK. The build
never downloads the SDK. Signing material is generated locally and must remain
private. One-shot dictation and playback use the optional Termux:API app and
corresponding Termux command package. A compatible on-device recognizer and an
offline TTS voice are required for continuous voice; availability varies by device.

The optional root optimizer guardian is separate from normal conversation setup.
It must be installed deliberately with this installation's app root and actual
Termux SELinux context (`OSTADIX_GUARD_APP_ROOT` and
`OSTADIX_TERMUX_APP_CONTEXT`). It refuses to resume workers without a valid local
ordinary-app identity. It is not installed by `install_shortcut.py`.

## Offline verification

These commands use fixtures, temporary workspaces, and loopback services. They
do not download a model, contact an inference provider, or open audio:

```sh
cd source/ostadix-voice
python3 -m unittest discover -s tests -p 'test_*.py' -q
python3 -m unittest android_voice.test_bridge -q
python3 runtime/npu/test_resident_protocol.py -q
```

The native protocol test compiles a deterministic C API stub using `CC`, `cc`,
or `clang` and skips when no native compiler is available. Platform-specific
fixtures may skip where the host lacks their required facilities. Scripts named
`integration_*.py` and `probe_codex.py` are separate, opt-in live checks: read them
before invoking them with `--live`. They are not included in test discovery.
`runtime/pixelforge.O` and `pixelforge_runner.py` preserve the earlier bounded
probe interface; normal conversation uses the separately installed chat host.

See the repository [license](../../LICENSE) and preserved third-party notices.
The included LiteRT C API headers retain their Apache-2.0 notices and license.
