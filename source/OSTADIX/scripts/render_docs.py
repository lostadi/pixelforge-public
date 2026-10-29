#!/usr/bin/env python3
"""Render or verify the complete tracked Markdown/HTML export set."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

if __package__:
    from .check_language_sources import SourceError, below_root, tracked_files
else:
    from check_language_sources import SourceError, below_root, tracked_files

ROOT = Path(__file__).resolve().parents[1]


def load_exports(root: Path) -> dict:
    data = json.loads((root / "docs/html-exports.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schema", "renderer", "version", "sources"} or data["schema"] != "ostadix.html-exports/v1" or data["renderer"] != "marked" or not isinstance(data["version"], str) or not data["version"].strip():
        raise SourceError("invalid HTML export manifest")
    sources = data["sources"]
    if not isinstance(sources, list) or not sources or not all(isinstance(source, str) for source in sources) or sources != sorted(set(sources)):
        raise SourceError("HTML sources must be nonempty, sorted, unique")
    tracked = tracked_files(root)
    pairs = {p for p in tracked if p.endswith(".md") and str(Path(p).with_suffix(".html")) in tracked}
    if set(sources) != pairs:
        raise SourceError(f"HTML export coverage mismatch; missing={sorted(pairs - set(sources))}; stale={sorted(set(sources) - pairs)}")
    for source in sources:
        below_root(root, source)
        below_root(root, str(Path(source).with_suffix(".html")))
    return data


def render(root: Path, source: str, marked: str, write: bool) -> tuple[str, bool, int]:
    with tempfile.TemporaryDirectory(prefix="ostadix-doc-render-") as temporary:
        output = Path(temporary) / "rendered.html"
        # A regular output file avoids Node CLI stdout/pipe truncation on large
        # documents. Never regard bounded tool-output capture as the artifact.
        subprocess.run([marked, str(root / source), "--output", str(output)], cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
        expected = output.read_bytes().rstrip(b"\n") + b"\n"
    target = (root / source).with_suffix(".html")
    matches = target.read_bytes() == expected
    if write and not matches:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as replacement:
                temporary_path = Path(replacement.name)
                replacement.write(expected)
                replacement.flush()
                os.fsync(replacement.fileno())
            temporary_path.chmod(stat.S_IMODE(target.stat().st_mode))
            os.replace(temporary_path, target)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return source, matches, len(expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "write", "validate"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--marked", default="marked")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if args.jobs < 1:
            raise SourceError("jobs must be positive")
        root = args.root.resolve()
        data = load_exports(root)
        if args.action == "validate":
            print(f"html-exports: {len(data['sources'])} tracked pairs classified")
            return 0
        version = subprocess.run([args.marked, "--version"], check=True, capture_output=True, text=True, timeout=15).stdout.strip()
        if version != data["version"]:
            raise SourceError(f"expected marked {data['version']}, found {version!r}; use the pinned renderer")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda source: render(root, source, args.marked, args.action == "write"), data["sources"]))
        changed = [path for path, matches, _ in results if not matches]
        for path in changed:
            print(f"{'updated' if args.action == 'write' else 'drift'}: {Path(path).with_suffix('.html')}")
        print(f"html-exports: {len(results)} complete documents; {sum(size for _, _, size in results)} bytes; {len(changed)} {'updated' if args.action == 'write' else 'drifted'}")
        return int(bool(changed) and args.action == "check")
    except (SourceError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"html-exports: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
