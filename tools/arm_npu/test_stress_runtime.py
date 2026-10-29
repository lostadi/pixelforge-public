"""CPU-only regression for the stress driver's forced-cleanup deadline."""
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from stress_runtime import stop_owned_tree


class CleanupTests(unittest.TestCase):
    def test_forces_only_owned_term_ignoring_tree_and_preserves_sentinel(self):
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            source = ("import os,signal,subprocess,sys,time;"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                      "p=subprocess.Popen([sys.executable,'-c','import signal,time;"
                      "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'],start_new_session=True);"
                      "open(sys.argv[1],'w').write(str(p.pid));time.sleep(30)")
            sentinel = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
            process = subprocess.Popen([sys.executable, "-c", source, str(ready)], start_new_session=True)
            child_fd = None
            try:
                deadline = time.monotonic() + 5
                while not ready.is_file() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(ready.is_file())
                child_fd = os.pidfd_open(int(ready.read_text()))
                began = time.monotonic()
                self.assertTrue(stop_owned_tree(process, grace=.1))
                self.assertLess(time.monotonic() - began, 6)
                self.assertEqual(process.returncode, -signal.SIGKILL)
                self.assertTrue(select.select([child_fd], [], [], 0)[0])
                self.assertIsNone(sentinel.poll())
            finally:
                if child_fd is not None:
                    if not select.select([child_fd], [], [], 0)[0]:
                        signal.pidfd_send_signal(child_fd, signal.SIGKILL)
                    os.close(child_fd)
                for owned in (process, sentinel):
                    if owned.poll() is None:
                        owned.kill()
                    owned.wait(timeout=5)

    def test_graceful_exit_requires_no_forced_cleanup(self):
        process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
        try:
            self.assertFalse(stop_owned_tree(process, grace=1))
            self.assertEqual(process.returncode, -signal.SIGTERM)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
