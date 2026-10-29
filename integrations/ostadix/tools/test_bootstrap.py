"""Check integrity and self-contained archive behavior without hardware use."""
import base64
import hashlib
import importlib.util
import io
import json
import os
import stat
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap = load("bootstrap")
builder = load("build_integrated")


def compiler_archive(path, names=None):
    contents = io.BytesIO()
    with tarfile.open(fileobj=contents, mode="w:gz") as archive:
        for name, content in (names or [("README.md", b"example"), ("tools/tool.py", b"pass\n")]):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    payload = contents.getvalue()
    header = 'OFFSET = int("{:020d}")\nPAYLOAD_SHA256 = "{}"\n'.format(0, hashlib.sha256(payload).hexdigest())
    header = 'OFFSET = int("{:020d}")\nPAYLOAD_SHA256 = "{}"\n'.format(len(header.encode()), hashlib.sha256(payload).hexdigest())
    path.write_bytes(header.encode() + payload)
    return path


class IntegrityTests(unittest.TestCase):
    def test_long_unicode_argv_uses_private_request_and_preserves_every_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake-O"
            observed = root / "observed.json"
            fake.write_text('#!/usr/bin/python3\nimport json,os,stat\n'
                            'path=os.environ["OSTADIX_NPU_ARGV_FILE"]\n'
                            'value=json.load(open(path))\n'
                            'json.dump({"argv":value,"path":path,"mode":stat.S_IMODE(os.stat(path).st_mode),'
                            '"parent_mode":stat.S_IMODE(os.stat(os.path.dirname(path)).st_mode),'
                            '"old_environment": "OSTADIX_NPU_ARGV" in os.environ},'
                            'open(os.environ["TEST_OBSERVED"],"w"))\n'
                            'json.dump({"exit_code":0},open(os.environ["OSTADIX_NPU_RESULT_FILE"],"w"))\n')
            fake.chmod(0o755)
            arguments = ["run", "--prompt", "🙂" * 12000, "$(literal); 'quote'\\n", *(["界" * 1000] * 30)]
            expected = [str(root / "tool.O")] + arguments
            self.assertGreater(len(json.dumps(expected).encode()), 131072)
            with mock.patch.object(bootstrap, "_find_ostadix", return_value=(str(fake), root)), \
                    mock.patch.object(bootstrap.sys, "argv", ["-c"] + expected), \
                    mock.patch.dict(os.environ, {"TEST_OBSERVED": str(observed), "OSTADIX_NPU_ARGV": "stale"}):
                with self.assertRaises(SystemExit) as result:
                    bootstrap._bootstrap()
                self.assertEqual(result.exception.code, 0)
            captured = json.loads(observed.read_text())
            self.assertEqual(captured["argv"], expected)
            self.assertEqual(captured["mode"], 0o600)
            self.assertEqual(captured["parent_mode"], 0o700)
            self.assertFalse(captured["old_environment"])
            self.assertFalse(Path(captured["path"]).exists())

    def test_request_file_preserves_surrogateescaped_filename_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake-O"
            observed = root / "observed.json"
            filename_bytes = os.fsencode(root) + b"/invalid-utf8-\xff.gguf"
            with open(filename_bytes, "wb") as fixture:
                fixture.write(b"filename round trip")
            filename = os.fsdecode(filename_bytes)
            fake.write_text('#!/usr/bin/python3\nimport json,os\n'
                            'path=os.environ["OSTADIX_NPU_ARGV_FILE"]\n'
                            'value=json.load(open(path))\n'
                            'target=value[-1]\n'
                            'json.dump({"argv":value,"path":path,"filename_hex":os.fsencode(target).hex(),'
                            '"contents_hex":open(target,"rb").read().hex()},'
                            'open(os.environ["TEST_OBSERVED"],"w"))\n'
                            'json.dump({"exit_code":0},open(os.environ["OSTADIX_NPU_RESULT_FILE"],"w"))\n')
            fake.chmod(0o755)
            expected = [str(root / "tool.O"), "inspect", filename]
            with mock.patch.object(bootstrap, "_find_ostadix", return_value=(str(fake), root)), \
                    mock.patch.object(bootstrap.sys, "argv", ["-c"] + expected), \
                    mock.patch.dict(os.environ, {"TEST_OBSERVED": str(observed)}):
                with self.assertRaises(SystemExit) as result:
                    bootstrap._bootstrap()
                self.assertEqual(result.exception.code, 0)
            captured = json.loads(observed.read_text())
            self.assertEqual(captured["argv"], expected)
            self.assertEqual(captured["filename_hex"], filename_bytes.hex())
            self.assertEqual(bytes.fromhex(captured["contents_hex"]), b"filename round trip")
            self.assertFalse(Path(captured["path"]).exists())

    def _render_result_with_surrogateescaped_path(self):
        filename_bytes = b"filename-\xff.gguf"
        result = {"exit_code": 0, "path": os.fsdecode(filename_bytes), "note": "界🙂"}
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "result.json"
            with mock.patch.object(bootstrap, "_request", return_value=(Path("/tmp/tool.O"), ["inspect"])), \
                    mock.patch.object(bootstrap, "integration_main", return_value=result, create=True), \
                    mock.patch.dict(os.environ, {"OSTADIX_NPU_RESULT_FILE": str(receipt)}):
                rendered = bootstrap._execute_integrated()
            self.assertEqual(json.loads(receipt.read_text()), result)
        return result, rendered, filename_bytes

    def test_result_json_preserves_surrogateescaped_paths_without_runtime(self):
        result, rendered, filename_bytes = self._render_result_with_surrogateescaped_path()
        restored = json.loads(rendered)
        self.assertEqual(restored, result)
        rendered.encode("utf8")
        self.assertEqual(os.fsencode(restored["path"]), filename_bytes)

    def test_result_json_crosses_real_cbor_with_surrogateescaped_paths(self):
        try:
            _executable, ostadix_root = bootstrap._find_ostadix()
        except FileNotFoundError as error:
            self.skipTest("Installed OSTADIX backend unavailable: " + str(error))
        codec_path = ostadix_root / "backends/o_shim_common.py"
        if not codec_path.is_file():
            self.skipTest("Installed OSTADIX CBOR codec unavailable: " + str(codec_path))
        spec = importlib.util.spec_from_file_location("pixelforge_test_o_shim_common", codec_path)
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        result, rendered, filename_bytes = self._render_result_with_surrogateescaped_path()
        # Match python_shim.send_ok's actual OValue string envelope.
        envelope = {"status": "ok", "value": {"t": "str", "v": rendered}}
        wire = io.BytesIO()
        codec.write_wire_message(envelope, wire)
        wire.seek(0)
        recovered = codec.read_wire_message(wire)
        self.assertEqual(recovered, envelope)
        restored = json.loads(recovered["value"]["v"])
        self.assertEqual(os.fsencode(restored["path"]), filename_bytes)
        self.assertEqual(restored["note"], result["note"])

    def test_request_file_takes_precedence_and_rejects_symlink_or_invalid_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = root / "request.json"
            request.write_text(json.dumps(["/tmp/tool.O", "inspect", "a $(literal); 界.gguf"]))
            with mock.patch.dict(os.environ, {"OSTADIX_NPU_ARGV_FILE": str(request), "OSTADIX_NPU_ARGV": "invalid"}):
                source, args = bootstrap._request()
                self.assertEqual(source, Path("/tmp/tool.O"))
                self.assertEqual(args, ["inspect", "a $(literal); 界.gguf"])
                request.write_text('["source", 3]')
                with self.assertRaisesRegex(ValueError, "array of strings"):
                    bootstrap._request()
                request.unlink()
                request.symlink_to(root / "outside")
                with self.assertRaises(OSError):
                    bootstrap._request()

    def test_request_file_has_a_bounded_read(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "large.json"
            limit = bootstrap._request_byte_limit()
            with request.open("wb") as output:
                output.truncate(limit + 1)
            with mock.patch.dict(os.environ, {"OSTADIX_NPU_ARGV_FILE": str(request)}):
                with self.assertRaisesRegex(ValueError, str(limit) + " bytes"):
                    bootstrap._request()

    def test_request_file_accepts_valid_argv_escaping_beyond_eight_mib(self):
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / "escaped.json"
            expected = ["/tmp/tool.O"] + ["\x01" * 60000] * 26
            original_bytes = sum(len(os.fsencode(arg)) + 1 for arg in expected)
            if original_bytes + 65536 > os.sysconf("SC_ARG_MAX"):
                self.skipTest("Host argv limit is too small for this >8 MiB escaped request fixture")
            with request.open("w", encoding="utf8") as output:
                json.dump(expected, output, ensure_ascii=True)
            self.assertGreater(request.stat().st_size, 8 * 1024**2)
            with mock.patch.dict(os.environ, {"OSTADIX_NPU_ARGV_FILE": str(request)}):
                source, args = bootstrap._request()
            self.assertEqual(source, Path(expected[0]))
            self.assertEqual(args, expected[1:])

    def test_request_bound_follows_host_argv_limit_with_fallback(self):
        with mock.patch.object(bootstrap.os, "sysconf", return_value=4 * 1024**2):
            self.assertGreaterEqual(bootstrap._request_byte_limit(), 6 * 4 * 1024**2)
        with mock.patch.object(bootstrap.os, "sysconf", side_effect=ValueError("unavailable")):
            self.assertEqual(bootstrap._request_byte_limit(), 16 * 1024**2)
        with mock.patch.object(bootstrap.os, "sysconf", return_value=-1):
            self.assertEqual(bootstrap._request_byte_limit(), 16 * 1024**2)

    def test_success_receipt_requires_successful_ostadix_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake-O"
            fake.write_text('#!/usr/bin/python3\nimport json,os\n'
                            'open(os.environ["OSTADIX_NPU_RESULT_FILE"],"w").write(json.dumps({"exit_code":0}))\n'
                            'raise SystemExit(17)\n')
            fake.chmod(0o755)
            with mock.patch.object(bootstrap, "_find_ostadix", return_value=(str(fake), root)), \
                    mock.patch.object(bootstrap.sys, "argv", ["-c", str(root / "tool.O"), "help"]):
                with self.assertRaises(SystemExit) as failure:
                    bootstrap._bootstrap()
                self.assertEqual(failure.exception.code, 17)

    def test_child_failure_code_survives_ostadix_error_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "fake-O"
            fake.write_text('#!/usr/bin/python3\nimport json,os\n'
                            'open(os.environ["OSTADIX_NPU_RESULT_FILE"],"w").write(json.dumps({"exit_code":7}))\n'
                            'raise SystemExit(1)\n')
            fake.chmod(0o755)
            with mock.patch.object(bootstrap, "_find_ostadix", return_value=(str(fake), root)), \
                    mock.patch.object(bootstrap.sys, "argv", ["-c", str(root / "tool.O"), "help"]):
                with self.assertRaises(SystemExit) as failure:
                    bootstrap._bootstrap()
                self.assertEqual(failure.exception.code, 7)

    def test_payload_decode_checks_all_bytes(self):
        data = b"compiler data\x00" * 600
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "tool.O"
            source.write_bytes(b"O^(text^(example)_text)_O\n# OSTADIX_NPU_PAYLOAD_V1 "
                               + digest.encode() + b"\n# " + base64.b64encode(data)
                               + b"\n# END_OSTADIX_NPU_PAYLOAD_V1\n")
            with mock.patch.multiple(bootstrap, ARCHIVE_SHA256=digest,
                                     ARCHIVE_BYTES=len(data), create=True):
                bootstrap._decode_archive(source, root / "good.run")
                self.assertEqual((root / "good.run").read_bytes(), data)
                source.write_bytes(source.read_bytes().replace(base64.b64encode(data), base64.b64encode(data[:-1] + b"X")))
                with self.assertRaisesRegex(ValueError, "hash or size"):
                    bootstrap._decode_archive(source, root / "bad.run")

    def test_cached_source_tampering_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "tool.py"
            code.write_bytes(b"original")
            pins = [{"path": "tool.py", "bytes": 8,
                     "sha256": hashlib.sha256(b"original").hexdigest()}]
            with mock.patch.object(bootstrap, "ARCHIVE_FILES", pins, create=True):
                self.assertEqual(bootstrap._verify_kit(root), root)
                code.write_bytes(b"modified")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    bootstrap._verify_kit(root)
                code.unlink()
                code.symlink_to(root / "outside.py")
                with self.assertRaisesRegex(ValueError, "Redirected"):
                    bootstrap._verify_kit(root)

    def test_json_request_keeps_shell_metacharacters_literal(self):
        argv = ["/tmp/script.O", "inspect", "/tmp/a $(touch bad); 'model'.gguf"]
        with mock.patch.dict(os.environ, {"OSTADIX_NPU_ARGV": json.dumps(argv)}):
            source, args = bootstrap._request()
            self.assertEqual(str(source), argv[0])
            self.assertEqual(args, argv[1:])
        with mock.patch.dict(os.environ, {"OSTADIX_NPU_ARGV": '["source", 3]'}):
            with self.assertRaisesRegex(ValueError, "array of strings"):
                bootstrap._request()


class BuilderTests(unittest.TestCase):
    def test_archive_members_hashed_and_payload_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            packed = compiler_archive(Path(directory) / "kit.run")
            entries, _identity = builder.inspect_archive(packed)
            self.assertEqual([entry["path"] for entry in entries], ["README.md", "tools/tool.py"])
            self.assertEqual(entries[0]["sha256"], hashlib.sha256(b"example").hexdigest())
            data = packed.read_bytes()
            packed.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "payload hash"):
                builder.inspect_archive(packed)

    def test_unsafe_archive_path_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            packed = compiler_archive(Path(directory) / "kit.run", [("../escape", b"x")])
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                builder.inspect_archive(packed)

    def test_build_refuses_existing_output_and_publishes_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = compiler_archive(root / "kit.run")
            (root / "bootstrap.py").write_text("def _execute_integrated(): return 'ok'\ndef _bootstrap(): pass\n")
            (root / "controller.py").write_text("def integration_main(argv, materialize): return {'exit_code': 0}\n")
            (root / "worker.py").write_text("# test fixture worker source\n")
            output = root / "kit.O"
            with mock.patch.object(builder, "HERE", root), \
                    mock.patch.object(builder, "EXPECTED_ARCHIVE_SHA256", builder.sha256_file(packed)):
                receipt = builder.build(packed, output)
                self.assertEqual(receipt["embedded_files"], 2)
                self.assertIn(b"# OSTADIX_NPU_PAYLOAD_V1", output.read_bytes())
                self.assertTrue(os.access(output, os.X_OK))
                with self.assertRaises(FileExistsError):
                    builder.build(packed, output)
            self.assertLess(len(output.read_bytes().splitlines()[0]), 255)

    def test_unverified_archive_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = compiler_archive(root / "kit.run", [("credentials.json", b"private")])
            with self.assertRaisesRegex(ValueError, "release pin"):
                builder.build(packed, root / "kit.O")
            self.assertFalse((root / "kit.O").exists())

    def test_output_replacement_is_not_reported_or_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packed = compiler_archive(root / "kit.run")
            (root / "bootstrap.py").write_text("def _execute_integrated(): return 'ok'\ndef _bootstrap(): pass\n")
            (root / "controller.py").write_text("def integration_main(argv, materialize): return {'exit_code': 0}\n")
            (root / "worker.py").write_text("# test fixture worker source\n")
            output = root / "kit.O"
            real_link = os.link
            def replace(source, destination):
                real_link(source, destination)
                Path(destination).unlink()
                Path(destination).write_text("user replacement")
            with mock.patch.object(builder, "HERE", root), \
                    mock.patch.object(builder, "EXPECTED_ARCHIVE_SHA256", builder.sha256_file(packed)), \
                    mock.patch.object(builder.os, "link", side_effect=replace):
                with self.assertRaisesRegex(ValueError, "replaced"):
                    builder.build(packed, output)
            self.assertEqual(output.read_text(), "user replacement")
            self.assertFalse(output.with_suffix(".O.json").exists())

    def test_build_preserves_preexisting_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "kit.O"
            sidecar = output.with_suffix(".O.json")
            sidecar.write_text("user data")
            packed = compiler_archive(root / "kit.run")
            with self.assertRaises(FileExistsError):
                builder.build(packed, output)
            self.assertEqual(sidecar.read_text(), "user data")
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
