"""Optional, explicit compiler-to-Voice registration after saved-bank success."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

BOOTSTRAP = Path("/data/adb/ostadix-pixelforge/bootstrap.json")


def _model_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ValueError("Voice model ID must contain 1 to 64 lowercase letters, digits, underscores, or hyphens")
    return value


def add_voice_arguments(parser):
    parser.add_argument("--voice-model-id", help="Register the successfully compiled package in Ostadix Voice")
    parser.add_argument("--select-voice", action="store_true", help="Select the newly registered Voice model; requires --voice-model-id")
    parser.add_argument("--replace-voice", action="store_true", help="Replace this Voice registry entry; requires --voice-model-id")


def validate_voice_arguments(parser, args):
    ident = getattr(args, "voice_model_id", None)
    if not ident and (getattr(args, "select_voice", False) or getattr(args, "replace_voice", False)):
        parser.error("--select-voice and --replace-voice require --voice-model-id")
    if ident is not None:
        try:
            _model_id(ident)
        except ValueError as error:
            parser.error(str(error))
        if args.plan_only:
            parser.error("--voice-model-id requires actual compilation, not --plan-only")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _trusted_bootstrap(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if info.st_uid != 0 or info.st_mode & 0o022 or not stat.S_ISREG(info.st_mode):
            raise ValueError("Voice handoff bootstrap must be a root-owned non-writable regular file")
        with os.fdopen(fd, "r", closefd=False) as stream:
            return json.load(stream)
    finally:
        os.close(fd)


def _installer():
    # Source checkouts use the companion installer from this same tree.
    local = Path(__file__).resolve().parents[2] / "chat/install.py"
    if local.is_file():
        return str(Path(sys.executable).absolute()), local
    # Installed compiler kits instead use the already pinned service, never an
    # environment-supplied executable or a guessed model-specific directory.
    bootstrap = _trusted_bootstrap(BOOTSTRAP)
    if not isinstance(bootstrap, dict):
        raise ValueError("Voice handoff bootstrap must contain an object")
    service = Path(bootstrap["service"])
    python = Path(bootstrap["python"])
    if not service.is_absolute() or not python.is_absolute() or ".." in service.parts or ".." in python.parts:
        raise ValueError("Voice handoff bootstrap contains invalid absolute paths")
    installer = service.parent / "install.py"
    pins = bootstrap.get("ubuntu_pins", {})
    if not isinstance(pins, dict):
        raise ValueError("Voice handoff bootstrap must contain installer/interpreter pins")
    for path in (installer, python):
        expected = pins.get(str(path))
        if ((path == installer and path.is_symlink()) or not path.is_file() or not isinstance(expected, str)
                or _sha256(path) != expected):
            raise ValueError("Voice handoff requires a pinned installed interpreter and installer")
    return str(python), installer


def register_compiled(args, *, package, model, selected_tensors=None,
                      weight_dtype, dynamic_weight_mode="off", batch_vectors=1,
                      fused_gate_up=False, fused_bank_rows=None,
                      fused_gate_up_tensors=None):
    """Register only an explicitly requested, already compiled package.

    Return 3 on handoff failure: the compiler has succeeded and its saved
    artifacts remain valid, while registration/selection has not succeeded.
    No installer is discovered or invoked when no Voice model ID was supplied.
    """
    ident = getattr(args, "voice_model_id", None)
    if ident is None:
        return 0
    event = {"event": "voice_registration", "model_id": ident,
             "compile_complete": True, "artifacts": str(Path(package).resolve())}
    try:
        _model_id(ident)
        if getattr(args, "plan_only", False):
            raise ValueError("Voice handoff cannot run during --plan-only")
        if selected_tensors is not None:
            if not selected_tensors or len(set(selected_tensors)) != len(selected_tensors):
                raise ValueError("Voice handoff requires distinct actual selected tensor names")
        python, installer = _installer()
        command = [python, str(installer), "--from-compiled", str(Path(package).resolve()),
                   "--model", str(Path(model).resolve()), "--model-id", ident, "--apply",
                   "--weight-dtype", weight_dtype, "--batch-vectors", str(batch_vectors),
                   "--dynamic-weight-mode", dynamic_weight_mode]
        if selected_tensors is not None:
            command += ["--select-tensors", ",".join(selected_tensors)]
        if fused_gate_up:
            command += ["--fused-gate-up"]
            if fused_bank_rows is not None:
                command += ["--fused-bank-rows", str(fused_bank_rows)]
            if fused_gate_up_tensors:
                command += ["--fused-gate-up-tensors", ",".join(fused_gate_up_tensors)]
        if getattr(args, "select_voice", False):
            command.append("--select")
        if getattr(args, "replace_voice", False):
            command.append("--replace")
        result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip() or "Installer exited " + str(result.returncode)
            raise ValueError(detail[-1000:])
        event.update(success=True, selected=bool(getattr(args, "select_voice", False)))
        print(json.dumps(event), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        event.update(success=False, error=str(error)[-1000:])
        print(json.dumps(event), flush=True)
        return 3
