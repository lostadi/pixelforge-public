"""Inspect installations and manage root-owned registered model bundles."""
import argparse
import json
from pathlib import Path
import sys

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if argv and argv[0] == "register":
        import install
        return install.main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--manifest", type=Path, required=True)
    for name in ("list", "select"):
        command = commands.add_parser(name)
        command.add_argument("--state-dir", type=Path, default=Path("/data/adb/ostadix-pixelforge"))
        if name == "select":
            command.add_argument("model_id")
    args = parser.parse_args(argv)
    if args.action == "inspect":
        manifest = json.loads(args.manifest.read_text())
        if manifest.get("schema") != "pixelforge.chat.install.v1":
            raise ValueError("Unknown installation manifest")
        result = {key: manifest[key] for key in ("schema", "termux_uid", "model_copied", "compiled_banks_copied", "writes")}
    else:
        import registry
        result = registry.list_models(args.state_dir) if args.action == "list" else registry.select_model(args.state_dir, args.model_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
