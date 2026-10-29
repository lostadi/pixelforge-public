import os
from pathlib import Path
import subprocess
import shlex
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from install_shortcut import install


class ShortcutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source with spaces'
        self.source.mkdir()
        (self.source / 'launch.py').write_text('import sys\nprint(repr(sys.argv[1:]))\n')
        self.bin_dir = self.root / 'my bin'
        self.rc = self.root / '.zshrc'

    def test_quoted_wrapper_forwards_arguments_and_idempotent_alias_preserves_config(self):
        self.rc.write_text('export EXISTING=preserved\n')
        launcher, alias = install(self.source, self.bin_dir, [self.rc])
        result = subprocess.run([str(launcher), 'a b', 'literal$(id)'], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "['a b', 'literal$(id)']")
        content = self.rc.read_bytes()
        self.assertTrue(content.startswith(b'export EXISTING=preserved\n'))
        self.assertIn(alias, content.decode())
        invoked = subprocess.run([shutil.which('sh'), '-c', '. ' + shlex.quote(str(self.rc)) + '\nc "a b"\n'],
                                 capture_output=True, text=True, check=True)
        self.assertEqual(invoked.stdout.strip(), "['a b']")
        install(self.source, self.bin_dir, [self.rc])
        self.assertEqual(self.rc.read_bytes(), content)
        self.assertTrue(os.access(launcher, os.X_OK))

    def test_existing_command_is_never_replaced_and_rc_is_not_modified(self):
        self.bin_dir.mkdir()
        (self.bin_dir / 'c').write_text('user command')
        self.rc.write_text('user config\n')
        with self.assertRaisesRegex(ValueError, 'preserved'):
            install(self.source, self.bin_dir, [self.rc])
        self.assertEqual((self.bin_dir / 'c').read_text(), 'user command')
        self.assertEqual(self.rc.read_text(), 'user config\n')

    def test_existing_alias_or_symlink_is_preserved_before_install(self):
        self.rc.write_text('alias c="my-existing-command"\n')
        with self.assertRaisesRegex(ValueError, 'preserved'):
            install(self.source, self.bin_dir, [self.rc])
        self.assertFalse((self.bin_dir / 'c').exists())
        self.rc.unlink()
        target = self.root / 'managed-config'; target.write_text('preserved')
        self.rc.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            install(self.source, self.bin_dir, [self.rc])
        self.assertEqual(target.read_text(), 'preserved')

    def test_zsh_function_syntax_is_preserved(self):
        self.rc.write_text('function c { printf "existing"; }\n')
        with self.assertRaisesRegex(ValueError, 'preserved'):
            install(self.source, self.bin_dir, [self.rc])
        self.assertFalse((self.bin_dir / 'c').exists())


if __name__ == '__main__':
    unittest.main()
