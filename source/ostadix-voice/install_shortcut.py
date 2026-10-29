#!/usr/bin/env python3
"""Install the source launcher as c; never overwrite another command or alias."""
import argparse
from pathlib import Path
import os
import re
import shlex
import shutil
import sys


MARKER = '# PixelForge Ostadix Voice source shortcut'


def install(source, bin_dir, rc_paths=(), python=None):
    source = Path(source).resolve(strict=True)
    if not (source / 'launch.py').is_file():
        raise ValueError('The selected source directory does not contain launch.py')
    launcher = Path(bin_dir).expanduser().absolute() / 'c'
    shell = shutil.which('sh') or '/bin/sh'
    body = ('#!' + shell + '\n' + MARKER + '\nexec ' +
            shlex.join([str(Path(python or sys.executable).resolve()), str(source / 'launch.py')]) + ' "$@"\n')
    # The alias value itself must retain quoting when its path contains spaces.
    alias = MARKER + '\nalias c=' + shlex.quote(shlex.quote(str(launcher))) + '\n'
    updates = []
    for raw in rc_paths:
        path = Path(raw).expanduser().absolute()
        if path.is_symlink():
            raise ValueError('Shell configuration is a symlink; add the printed alias manually: ' + str(path))
        previous = path.read_text() if path.exists() else ''
        if alias in previous:
            continue
        if re.search(r'(?m)^\s*(?:alias\s+c=|(?:function\s+)?c\s*\(|function\s+c\s*\{)', previous):
            raise ValueError('An existing c alias/function is preserved in ' + str(path))
        updates.append((path, previous))
    if launcher.is_symlink() or launcher.exists() and launcher.read_text() != body:
        raise ValueError('An existing c command is preserved: ' + str(launcher))
    launcher.parent.mkdir(parents=True, exist_ok=True)
    if not launcher.exists():
        descriptor = os.open(launcher, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o755)
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(body)
    for path, previous in updates:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(('\n' if previous and not previous.endswith('\n') else '') + '\n' + alias)
    return launcher, alias.rstrip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, default=Path.home() / '.local/bin')
    parser.add_argument('--shell', choices=('none', 'bash', 'zsh', 'both'), default='none',
                        help='Append a managed c alias to the chosen shell rc; existing aliases are preserved')
    args = parser.parse_args(argv)
    paths = [Path.home() / name for name in ('.bashrc', '.zshrc')
             if args.shell == 'both' or args.shell == ('bash' if name == '.bashrc' else 'zsh')]
    try:
        launcher, alias = install(Path(__file__).parent, args.bin_dir, paths)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    print('Installed: ' + str(launcher))
    print('Open a new shell, or use:\n' + alias)
    print('No model, Android service, shell command or inference was started.')


if __name__ == '__main__':
    main()
