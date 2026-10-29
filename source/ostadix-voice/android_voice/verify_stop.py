"""Explicit, ordinary-Termux microphone teardown probe; no audio is saved."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from android_voice.bridge import NativeVoiceBridge


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Explicitly open the microphone for this probe')
    parser.add_argument('--output', type=Path, help='Fresh output path for the local report')
    args = parser.parse_args(argv)
    if not args.live:
        parser.exit(0, 'No recording started. A hardware teardown probe requires --live --output NEW_PATH.\n')
    if args.output is None:
        parser.error('--live requires --output NEW_PATH')
    if os.getuid() == 0:
        parser.error('Run the probe in ordinary Termux, outside root')
    if args.output.exists() or args.output.is_symlink():
        parser.error('The output path already exists')
    bridge = NativeVoiceBridge(lambda event: None)
    report = {'transcripts_saved': False, 'pcm_saved': False, 'playback_used': False,
              'uid': os.getuid(), 'passed': False}
    try:
        before = time.monotonic()
        bridge.start(timeout=15)
        report['ready_seconds'] = round(time.monotonic() - before, 3)
    finally:
        before = time.monotonic()
        bridge.close()
        report['close_seconds'] = round(time.monotonic() - before, 3)
        report['native_stop_ack'] = bridge._native_stopped.is_set()
        report['after_close'] = {key: bridge.capabilities.get(key) for key in
                                 ('mic_active', 'capture_continuous', 'speaking', 'capture_state')}
        report['passed'] = report['native_stop_ack'] and report['after_close']['mic_active'] is False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print(json.dumps(report))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
