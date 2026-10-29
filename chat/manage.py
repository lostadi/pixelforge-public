"""Inspect a generated installation or stage a different explicit local profile.

Use install.py with a new model/artifact pair to produce a fresh manifest. This
command deliberately does not contain a catalogue of private device profiles.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect',))
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get('schema') != 'pixelforge.chat.install.v1':
        raise ValueError('Unknown installation manifest')
    print(json.dumps({key: manifest[key] for key in ('schema', 'termux_uid', 'model_copied', 'compiled_banks_copied', 'writes')}, indent=2))


if __name__ == '__main__':
    main()
