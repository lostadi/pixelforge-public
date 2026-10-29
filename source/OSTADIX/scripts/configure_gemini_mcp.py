#!/usr/bin/env python3
"""Idempotently register the installed Ostadix MCP server for Gemini CLI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def configure(settings_path: Path, command: Path) -> bool:
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"refusing to modify invalid Gemini settings: {error}")
        if not isinstance(settings, dict):
            raise SystemExit("refusing to modify Gemini settings that are not a JSON object")
    else:
        settings = {}

    servers = settings.setdefault("mcpServers", {})
    if not isinstance(servers, dict):
        raise SystemExit("refusing to replace non-object Gemini mcpServers settings")

    desired = {"command": os.fspath(command.resolve()), "args": []}
    if servers.get("ostadix") == desired:
        return False
    servers["ostadix"] = desired

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{settings_path.name}.", dir=settings_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(settings, output, indent=2, sort_keys=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, settings_path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--command", type=Path, required=True)
    arguments = parser.parse_args()
    changed = configure(arguments.settings, arguments.command)
    print(f"Gemini MCP registration: {'updated' if changed else 'already current'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
