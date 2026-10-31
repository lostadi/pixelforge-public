"""Offline model registration, source pins, idle selection and publication rollback."""
import contextlib
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import android_launcher
import registry

@unittest.skipUnless(os.getuid() == 0, "Root-owned registry fixtures")
class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix="pixelforge-registry-")
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.state = self.root / "state"
        self.state.mkdir(mode=0o700)
        self.home = self.root / "app"
        self.home.mkdir(mode=0o700)
        self.runtime = self.home / "runtime"
        self.runtime.mkdir(mode=0o700)
        self.uid = 12666
        os.chown(self.home, self.uid, self.uid)
        os.chown(self.runtime, self.uid, self.uid)
        self.python = self.root / "python"
        self.python.write_bytes(b"fixture interpreter")
        (self.state / "chat.py").write_text("# fixed fixture launcher\n")
        self.legacy_bootstrap, self.legacy_client = self.bundle("Legacy", "gemma4")
        (self.state / "bootstrap.json").write_text(json.dumps(self.legacy_bootstrap))
        self.write_client({"backend": registry.BACKEND, "model": {"model_name": "Untrusted edit"}})

    def write_client(self, data):
        path = self.runtime / "pixelforge-chat.json"
        path.write_text(json.dumps(data))
        path.chmod(0o600)
        os.chown(path, self.uid, self.uid)

    def bundle(self, name, architecture="granite"):
        directory = self.root / (name.lower() + "-service")
        directory.mkdir(mode=0o700)
        service = directory / "service.py"
        service.write_text("# source " + name + "\n")
        model = {"model_name": name, "model_sha256": hashlib.sha256(name.encode()).hexdigest(),
                 "architecture": architecture, "provenance_repo": "local/fixture",
                 "context_per_worker": 4096, "max_output_tokens": 512,
                 "execution": "Fixture selected NPU projection"}
        profile = directory / "profile.json"
        profile.write_text(json.dumps({"backend": registry.BACKEND, "model": model,
                                      "architecture": architecture}))
        bootstrap = {"ubuntu_root": "/", "python": str(self.python), "service": str(service),
                     "ubuntu_pins": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (self.python, service, profile)}}
        metadata = {"backend": registry.BACKEND, "experimental": True, "model": model,
                    "launcher": str(self.state / "chat.py")}
        return bootstrap, metadata

    def verify(self, bootstrap):
        registry._bootstrap(bootstrap)
        for path, expected in bootstrap["ubuntu_pins"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("source pin changed")
        return os.open("/", os.O_RDONLY | os.O_DIRECTORY), os.open("/proc/self/ns/mnt", os.O_RDONLY)

    def register(self, ident="granite", name="Granite", **kwargs):
        bootstrap, metadata = self.bundle(name)
        return registry.register_model(self.state, ident, bootstrap, metadata, self.runtime,
                                       self.uid, label_source=self.home, verifier=self.verify, **kwargs)

    def snapshot(self):
        return {str(p): p.read_bytes() for p in [self.state / "models.json", self.state / "bootstrap.json",
            self.runtime / "pixelforge-chat.json", self.runtime / "pixelforge-models.json"] if p.exists()}

    def test_registration_preserves_pinned_legacy_default_and_publishes_passive_catalog(self):
        catalog = self.register()
        self.assertEqual(catalog["selected_model_id"], "default")
        self.assertEqual(set(catalog["models"]), {"default", "granite"})
        self.assertEqual(catalog["models"]["default"]["model"]["model_name"], "Legacy")
        self.assertNotIn("bootstrap", json.dumps(catalog))
        self.assertEqual(json.loads((self.state / "bootstrap.json").read_text()), self.legacy_bootstrap)
        self.assertEqual(registry.list_models(self.state), catalog)
        self.assertEqual((self.runtime / "pixelforge-models.json").stat().st_uid, self.uid)

    def test_selection_switches_only_to_registered_pinned_bundle(self):
        self.register()
        catalog = registry.select_model(self.state, "granite", verifier=self.verify)
        self.assertEqual(catalog["selected_model_id"], "granite")
        active = json.loads((self.runtime / "pixelforge-chat.json").read_text())
        self.assertEqual(active["model_id"], "granite")
        self.assertEqual(active["model"]["model_name"], "Granite")
        self.assertEqual(active["launcher"], str(self.state / "chat.py"))
        self.assertEqual(Path(json.loads((self.state / "bootstrap.json").read_text())["service"]).parent.name, "granite-service")
        self.assertEqual((self.state / "models.json").stat().st_mode & 0o777, 0o600)

    def test_default_pin_verifier_uses_mounted_root_without_running_model_code(self):
        bootstrap, metadata = self.bundle("Verified")
        before = len(list(Path("/proc/self/fd").iterdir()))
        catalog = registry.register_model(self.state, "verified", bootstrap, metadata, self.runtime,
                                          self.uid, label_source=self.home, select=True)
        self.assertEqual(catalog["selected_model_id"], "verified")
        self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)

    def test_installer_callback_runs_inside_global_idle_lock(self):
        calls = []
        def installer():
            with (self.state / "chat.lock").open("a") as lock:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            calls.append("inside-lock")
        catalog = self.register(installer=installer)
        self.assertEqual(calls, ["inside-lock"])
        self.assertIn("granite", catalog["models"])

    def test_duplicate_and_active_replacement_flags_reject_before_installer(self):
        self.register(select=True)
        bootstrap, metadata = self.bundle("Replacement")
        params = (self.state, "granite", bootstrap, metadata, self.runtime, self.uid)
        installer = Mock()
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "replacement"):
            registry.register_model(*params, verifier=self.verify, installer=installer)
        with self.assertRaisesRegex(ValueError, "active model"):
            registry.register_model(*params, replace=True, verifier=self.verify, installer=installer)
        installer.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_failed_installer_callback_preserves_registry_and_active_publication(self):
        self.register(select=True)
        bootstrap, metadata = self.bundle("NewVersion")
        before = self.snapshot()
        orphan = self.root / "new-version-orphan"
        def installer():
            orphan.mkdir()
            (orphan / "partial.py").write_text("# incomplete new version")
            raise OSError("injected bundle copy failure")
        with self.assertRaisesRegex(OSError, "bundle copy"):
            registry.register_model(self.state, "newversion", bootstrap, metadata, self.runtime,
                self.uid, verifier=self.verify, installer=installer, select=True)
        self.assertTrue(orphan.exists())
        self.assertEqual(self.snapshot(), before)
        self.assertNotIn("newversion", registry.list_models(self.state)["models"])

    def test_failed_source_verification_after_callback_preserves_active_publication(self):
        self.register(select=True)
        bootstrap, metadata = self.bundle("BadVersion")
        service = Path(bootstrap["service"])
        before = self.snapshot()
        calls = []
        def installer():
            calls.append("copied")
            service.write_text("# bad new version")
        with self.assertRaisesRegex(ValueError, "source pin"):
            registry.register_model(self.state, "badversion", bootstrap, metadata, self.runtime,
                self.uid, verifier=self.verify, installer=installer, select=True)
        self.assertEqual(calls, ["copied"])
        self.assertEqual(self.snapshot(), before)

    def test_new_bundle_is_verified_only_after_installer_callback_creates_it(self):
        self.register()
        bootstrap, metadata = self.bundle("Deferred")
        files = {Path(name): Path(name).read_bytes() for name in bootstrap["ubuntu_pins"]
                 if name != str(self.python)}
        for path in files:
            path.unlink()
        before = self.snapshot()
        events = []
        def installer():
            self.assertEqual(self.snapshot(), before)
            self.assertTrue(all(not p.exists() for p in files))
            for path, data in files.items():
                path.write_bytes(data)
            events.append("installed")
        def verify(value):
            if value["service"] == bootstrap["service"]:
                self.assertEqual(events, ["installed"])
                events.append("verified")
            return self.verify(value)
        catalog = registry.register_model(self.state, "deferred", bootstrap, metadata, self.runtime,
            self.uid, verifier=verify, installer=installer, select=True)
        self.assertEqual(events, ["installed", "verified"])
        self.assertEqual(catalog["selected_model_id"], "deferred")
        self.assertEqual(json.loads((self.state / "bootstrap.json").read_text()), bootstrap)

    def test_native_xattr_bindings_preserve_exact_filesystem_bytes(self):
        path = self.root / "native-xattr"
        path.write_bytes(b"fixture")
        value = b"exact-label-bytes\0with-tail"
        with path.open("rb") as stream:
            try:
                registry._native_setxattr(stream.fileno(), "user.pixelforge_fixture", value)
            except OSError as error:
                if error.errno in (errno.ENOTSUP, errno.EPERM):
                    self.skipTest("Filesystem does not permit user xattr fixtures")
                raise
        self.assertEqual(registry._native_getxattr(path, "user.pixelforge_fixture"), value)

    def test_missing_os_xattr_wrappers_copy_real_selinux_label(self):
        path = self.root / "native-selinux"
        path.write_bytes(b"fixture")
        try:
            expected = registry._native_getxattr(path, "security.selinux")
        except OSError as error:
            if error.errno in (errno.ENODATA, errno.ENOTSUP):
                self.skipTest("Filesystem has no SELinux labels")
            raise
        with patch.object(registry.os, "getxattr", None, create=True), patch.object(registry.os, "setxattr", None, create=True):
            self.assertEqual(registry._read_selinux_label(path), expected)
            with path.open("rb") as stream:
                registry._set_selinux_label(stream.fileno(), expected)
            self.assertEqual(registry._read_selinux_label(path), expected)

    def test_missing_os_xattr_wrappers_support_normal_registration_and_selection(self):
        with patch.object(registry.os, "getxattr", None, create=True), patch.object(registry.os, "setxattr", None, create=True):
            self.register()
            catalog = registry.select_model(self.state, "granite", verifier=self.verify)
        self.assertEqual(catalog["selected_model_id"], "granite")
        self.assertEqual(json.loads((self.runtime / "pixelforge-chat.json").read_text())["model_id"], "granite")

    def test_native_xattr_errno_is_preserved(self):
        def denied(*args):
            ctypes.set_errno(errno.EACCES)
            return -1
        with patch.object(registry, "_native_xattr_function", return_value=denied):
            with self.assertRaises(OSError) as read:
                registry._native_getxattr(self.home, "security.selinux")
            self.assertEqual(read.exception.errno, errno.EACCES)
            with self.assertRaises(OSError) as write:
                registry._native_setxattr(0, "security.selinux", b"label")
            self.assertEqual(write.exception.errno, errno.EACCES)

    def test_only_missing_or_unsupported_labels_are_optional(self):
        with patch.object(registry.os, "getxattr", None, create=True):
            for number in (errno.ENODATA, errno.ENOTSUP):
                with self.subTest(number=number), patch.object(registry, "_native_getxattr", side_effect=OSError(number, "fixture")):
                    self.assertIsNone(registry._read_selinux_label(self.home))
            with patch.object(registry, "_native_getxattr", side_effect=OSError(errno.EACCES, "denied")):
                with self.assertRaises(OSError):
                    registry._read_selinux_label(self.home)

    def test_label_setter_unavailable_fails_before_any_selected_file_write(self):
        self.register()
        before = self.snapshot()
        with patch.object(registry.os, "getxattr", None, create=True), patch.object(registry.os, "setxattr", None, create=True), \
                patch.object(registry, "_native_getxattr", return_value=b"u:object_r:app_data_file:s0"), \
                patch.object(registry, "_native_xattr_function", side_effect=RuntimeError("missing fsetxattr")), \
                patch.object(registry, "_atomic_bytes") as atomic:
            with self.assertRaisesRegex(RuntimeError, "fsetxattr"):
                registry.select_model(self.state, "granite", verifier=self.verify)
        atomic.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_rollback_continues_to_restore_bootstrap_after_client_restore_failure(self):
        self.register()
        before = self.snapshot()
        atomic = registry._atomic_bytes
        def denied_client(path, *args, **kwargs):
            if path.name == "pixelforge-chat.json":
                raise OSError(errno.EACCES, "persistent client label denial")
            return atomic(path, *args, **kwargs)
        with patch.object(registry, "_atomic_bytes", side_effect=denied_client):
            with self.assertRaisesRegex(RuntimeError, "rollback incomplete") as failure:
                registry.select_model(self.state, "granite", verifier=self.verify)
        self.assertIsInstance(failure.exception.__cause__, OSError)
        self.assertIn("pixelforge-chat.json", str(failure.exception))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(json.loads((self.state / "bootstrap.json").read_text()), self.legacy_bootstrap)

    def test_global_inference_lock_prevents_selection_without_publication(self):
        self.register()
        before = self.snapshot()
        with (self.state / "chat.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                registry.select_model(self.state, "granite", verifier=self.verify)
        self.assertEqual(self.snapshot(), before)

    def test_changed_registered_source_pin_rejects_selection(self):
        self.register()
        (self.root / "granite-service/service.py").write_text("# tampered\n")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "source pin"):
            registry.select_model(self.state, "granite", verifier=self.verify)
        self.assertEqual(self.snapshot(), before)

    def test_unknown_or_invalid_ids_never_publish(self):
        self.register()
        before = self.snapshot()
        for value in ("missing", "../granite", "/tmp/owned", "x;id", "UPPER", "", "x" * 65, True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                registry.select_model(self.state, value, verifier=self.verify)
        self.assertEqual(self.snapshot(), before)

    def test_atomic_publication_failure_restores_selected_files(self):
        self.register()
        before = self.snapshot()
        atomic = registry._atomic_bytes
        failed = False
        def injected(path, *args, **kwargs):
            nonlocal failed
            if path.name == "pixelforge-models.json" and not failed:
                failed = True
                raise OSError("injected publication failure")
            return atomic(path, *args, **kwargs)
        with patch.object(registry, "_atomic_bytes", side_effect=injected):
            with self.assertRaisesRegex(OSError, "injected"):
                registry.select_model(self.state, "granite", verifier=self.verify)
        self.assertTrue(failed)
        self.assertEqual(self.snapshot(), before)

    def test_root_registry_symlinks_and_writable_inputs_are_rejected(self):
        self.register()
        path = self.state / "models.json"
        path.chmod(0o666)
        with self.assertRaises(ValueError):
            registry.list_models(self.state)
        path.chmod(0o600)
        target = self.state / "other.json"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(OSError):
            registry.list_models(self.state)

    def test_replacing_selected_model_requires_explicit_replace_and_select(self):
        self.register(select=True)
        bootstrap, metadata = self.bundle("GraniteNew")
        params = (self.state, "granite", bootstrap, metadata, self.runtime, self.uid)
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "replacement"):
            registry.register_model(*params, verifier=self.verify)
        with self.assertRaisesRegex(ValueError, "active model"):
            registry.register_model(*params, replace=True, verifier=self.verify)
        self.assertEqual(self.snapshot(), before)
        catalog = registry.register_model(*params, replace=True, select=True, verifier=self.verify)
        self.assertEqual(catalog["selected_model_id"], "granite")
        self.assertEqual(json.loads((self.runtime / "pixelforge-chat.json").read_text())["model"]["model_name"], "GraniteNew")

    def test_launcher_management_flags_have_no_free_command_interface(self):
        fake = self.state / "chat.py"
        module = Mock()
        module.list_models.return_value = {"schema": registry.CATALOG_SCHEMA, "models": {}}
        module.select_model.return_value = {"schema": registry.CATALOG_SCHEMA, "selected_model_id": "granite", "models": {}}
        with patch.object(android_launcher, "__file__", str(fake)), patch.object(android_launcher, "registry_module", return_value=module):
            with patch.object(sys, "argv", [str(fake), "--list-models"]), contextlib.redirect_stdout(io.StringIO()):
                android_launcher.main()
            module.list_models.assert_called_once_with(self.state)
            with patch.object(sys, "argv", [str(fake), "--select-model", "granite"]), contextlib.redirect_stdout(io.StringIO()):
                android_launcher.main()
            module.select_model.assert_called_once_with(self.state, "granite", verifier=android_launcher.ubuntu_handles)
            for args in (["--model", "/tmp/model"], ["--select-model", "x", "extra"], ["--shell"]):
                with self.subTest(args=args), patch.object(sys, "argv", [str(fake), *args]), self.assertRaises(ValueError):
                    android_launcher.main()

    def test_launcher_checks_exact_helper_bytes_and_rejects_symlink(self):
        helper = self.state / "registry.py"
        helper.write_text("fixture_value = 23\n")
        self.assertEqual(android_launcher.registry_module(self.state / "chat.py").fixture_value, 23)
        helper.chmod(0o666)
        with self.assertRaises(ValueError):
            android_launcher.registry_module(self.state / "chat.py")
        helper.chmod(0o600)
        moved = self.state / "other.py"
        helper.rename(moved)
        helper.symlink_to(moved)
        with self.assertRaises(OSError):
            android_launcher.registry_module(self.state / "chat.py")

if __name__ == "__main__":
    unittest.main()
