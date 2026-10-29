#!/usr/bin/env python3
"""Install compiled Ostadix tools without a runtime shell dispatcher.

All required inputs are checked before changing installed files. Each file is
replaced atomically so an already running process keeps its original executable.
The evaluator has a stable name even where O and o share a directory entry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile


RUST_TOOLS = (
    "O", "o-cli", "olangc", "ocorec", "o-link", "o-unlink", "ogit",
    "o-live-host", "o-node", "octl", "o-registry", "o-info", "ostadix-device",
    "ocore-kernel-world-record",
)
NATIVE_MAGICS = {b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
                 b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
                 b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca"}


def validate_native(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"compiled executable missing: {path}; build the selected tools first")
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic not in NATIVE_MAGICS and not magic.startswith(b"MZ"):
        raise ValueError(f"expected a compiled executable, found another file format: {path}")


def atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o755)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def case_insensitive_directory(directory: Path) -> bool:
    descriptor, temporary = tempfile.mkstemp(prefix=".ostadix-case-", dir=directory)
    os.close(descriptor)
    probe = Path(temporary)
    try:
        return probe.with_name(probe.name.upper()).exists()
    finally:
        probe.unlink()


def install(repo: Path, build: Path, destination: Path, *, front_door_only: bool = False,
            include_c: bool = False, include_notebook: bool = False, c_build_dir: Path | None = None,
            dry_run: bool = False) -> list[str]:
    repo, build = repo.resolve(), build.resolve()
    destination = destination.absolute()
    names = ("O", "o-cli") if front_door_only else RUST_TOOLS
    copies = [(build / name, destination / name) for name in names]
    if not front_door_only and include_notebook:
        copies.append((build / "o-notebook", destination / "o-notebook"))
    if include_c:
        c_build = c_build_dir.resolve() if c_build_dir else repo / "c_cpp"
        copies.extend((c_build / name, destination / alias)
                      for name, alias in (("O", "o-c"), ("olangc", "olangc-c")))
    # Copy the evaluator before installing o; on case-insensitive filesystems
    # the last replacement also becomes uppercase O, without losing raw O.
    copies.extend(((build / "O", destination / "ostadix-evaluator"),
                   (build / "o-cli", destination / "o")))
    plan = [f"replace {target} from {source}" for source, target in copies]
    metadata = destination / "ostadix-install.json"
    plan.append(f"write {metadata} for {repo}")
    if dry_run:
        return plan
    for source, target in copies:
        validate_native(source)
        if target.is_dir() and not target.is_symlink():
            raise ValueError(f"installation target is a directory: {target}")
    if metadata.is_dir():
        raise ValueError(f"installation metadata target is a directory: {metadata}")
    destination.mkdir(parents=True, exist_ok=True)
    shared_front_door = case_insensitive_directory(destination)
    if shared_front_door:
        plan = [entry for entry in plan if entry != f"replace {destination / 'O'} from {build / 'O'}"]
        plan.append(f"O/o share the compiled front door in {destination}; raw evaluation uses ostadix-evaluator")
    for source, target in copies:
        # Keep the previous front door usable until its final replacement.
        # Installing raw O here would temporarily replace o on macOS.
        if shared_front_door and target.name == "O":
            continue
        atomic_copy(source, target)
    descriptor, temporary = tempfile.mkstemp(prefix=".ostadix-install.", dir=destination)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"schema": 1, "repo_root": str(repo), "backends_dir": str(repo / "backends")}, stream, indent=2)
            stream.write("\n")
        os.chmod(temporary, 0o644)
        os.replace(temporary, metadata)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--front-door-only", action="store_true")
    parser.add_argument("--include-c", action="store_true")
    parser.add_argument("--c-build-dir", type=Path, help="C Make/CMake build directory (default: REPO/c_cpp)")
    parser.add_argument("--include-notebook", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        for line in install(args.repo_root, args.build_dir or args.repo_root / "target/release",
                            args.bin_dir, front_door_only=args.front_door_only,
                            include_c=args.include_c, include_notebook=args.include_notebook,
                            c_build_dir=args.c_build_dir, dry_run=args.dry_run):
            print(("[DRY] " if args.dry_run else "") + line)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ostadix installation failed: {error}\nExisting processes keep their executable; fix the input and rerun installation.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
