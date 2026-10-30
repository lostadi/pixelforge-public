"""Atomic, hash-verified persistent DarwiNN artifacts for custom GGUF weights."""
from __future__ import annotations

import fcntl
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
import time

FORMAT = "pixelforge.darwinn.fc.v1"
RECIPE = "fc-v1-f32-ports-zero-bias-per-row-int8-or-f32-constants"
BATCH_RECIPE = "fc-batch-v1-f32-ports-f32-constants"
FUSED_RECIPE = "fc-fused-gate-up-v1-f32-ports-f32-constants"
FULL_FFN_FORMAT = "pixelforge.darwinn.full-ffn.v1"
FULL_FFN_RECIPE = "swiglu-ffn-v1-raw-f32-ports-f32-constants"
DYNAMIC_FC_FORMAT = "pixelforge.darwinn.dynamic-fc.v1"
DYNAMIC_FC_RECIPE = "dynamic-bmm-v1-f32-weight-input"
GROUPED_FC_FORMAT = "pixelforge.darwinn.grouped-fc.v1"
GROUPED_FC_RECIPE = "original-groups-bmm-v1-unpack-no-requant"
GROUPED_COLUMNS_FORMAT = "pixelforge.darwinn.grouped-columns.v1"
GROUPED_COLUMNS_RECIPE = "original-groups-columns-v1-greedy3072-zero-padded-split-tail-f32-partial-sum"
MAX_CODE_BYTES = 2 * 1024 ** 3


def batch_directory(directory, batch_vectors):
    """Keep new batch recipes separate from an existing single-vector package."""
    if type(batch_vectors) is not int or not 1 <= batch_vectors <= 512:
        raise ValueError("batch_vectors must be an integer from 1 to 512")
    directory = Path(directory)
    return directory if batch_vectors == 1 else directory / "batch-fc-v1" / f"b{batch_vectors}"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshot(path, hash_bytes=True):
    with Path(path).open("rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Source GGUF must be a regular file")
        digest = hashlib.sha256()
        if hash_bytes:
            for block in iter(lambda: source.read(4 * 1024 ** 2), b""):
                digest.update(block)
        after = os.fstat(source.fileno())
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if not stat.S_ISREG(before.st_mode) or identity(before) != identity(after) or identity(before) != identity(Path(path).stat()):
        raise ValueError("Source GGUF changed while collecting its identity")
    return digest.hexdigest() if hash_bytes else None, identity(before)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def device_fingerprint():
    """Fingerprint the compiler and its platform without changing the device."""
    library = Path("/vendor/lib64/libedgetpu_litert.so")
    firmware = Path("/sys/devices/platform/sswrp_tpu@36000000/36500000.buenos/firmware_version")
    kit = Path(__file__).parent
    author_files = ("schema_py_generated.py", "build_constant_fc.py", "gguf_hf_loader.py",
                    "gguf_quantizers.py", "GGUF_QUANTIZATION_PROVENANCE.json",
                    "gguf_vendor/quants.py", "gguf_vendor/constants.py", "gguf_vendor/lazy.py")
    result = {"compiler_library_sha256": file_sha256(library),
              "machine": os.uname().machine, "kernel_release": os.uname().release,
              "firmware_version": firmware.read_text().strip(),
              "native_protocol_abi": "pixelforge-pipe-DL-v1",
              "author_decoder_sha256": {name: file_sha256(kit / name) for name in author_files}}
    properties = {}
    for path in (Path("/system/build.prop"), Path("/vendor/build.prop")):
        if path.is_file():
            for line in path.read_text(errors="replace").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    if key.endswith("build.fingerprint") or key in ("ro.product.board", "ro.board.platform"):
                        properties[key] = value
    result["android_properties"] = properties
    return result


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _embedded_path(directory, source):
    relative = PurePosixPath(source["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Embedded GGUF must have a contained relative path")
    path = directory.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory):
        raise ValueError("Embedded GGUF must be a contained regular file")
    return path


def _atomic(path, data):
    if path.is_symlink():
        raise ValueError(f"Artifact path is a symlink: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as destination:
            os.fchmod(destination.fileno(), 0o600)
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextlib.contextmanager
def _package_root(directory, read_only, recipe=RECIPE, artifact_format=FORMAT):
    if read_only:
        yield
        return
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        entries = list(directory.iterdir())
        manifest = directory / "MODEL.json"
        if entries:
            if manifest.is_symlink() or not manifest.is_file():
                raise ValueError("Output directory is nonempty and is not a compiled-model package")
            try:
                value = json.loads(manifest.read_text())
            except (ValueError, OSError) as error:
                raise ValueError("Output directory has no valid compiled-model manifest") from error
            if value.get("format") != artifact_format or value.get("recipe") != recipe:
                raise ValueError("Output directory is not a recognized compiled-model package")
        directory.chmod(0o700)
        yield
    finally:
        os.close(fd)


class ArtifactCache:
    """Store one device/weight identity; read-only mode never compiles misses."""
    def __init__(self, directory, source_gguf=None, *, source_sha256=None,
                 fingerprint=None, read_only=False, plan=None, verify_source=True,
                 source_identity=None, batch_vectors=1, fused_gate_up=False, _program_kind=None):
        if type(batch_vectors) is not int or not 1 <= batch_vectors <= 512:
            raise ValueError("batch_vectors must be an integer from 1 to 512")
        self.batch_vectors = batch_vectors
        self.fused_gate_up = fused_gate_up
        programs = {'full-ffn-v1': (FULL_FFN_FORMAT,FULL_FFN_RECIPE),
                    'dynamic-fc-v1': (DYNAMIC_FC_FORMAT,DYNAMIC_FC_RECIPE),
                    'grouped-fc-v1': (GROUPED_FC_FORMAT,GROUPED_FC_RECIPE),
                    'grouped-columns-v1': (GROUPED_COLUMNS_FORMAT,GROUPED_COLUMNS_RECIPE)}
        if _program_kind not in (None,*programs) or (_program_kind and fused_gate_up):
            raise ValueError("Unsupported artifact program kind")
        self.program_kind = _program_kind
        if _program_kind:
            self.format,self.recipe=programs[_program_kind]
        else:
            self.format=FORMAT
            self.recipe=FUSED_RECIPE if fused_gate_up else (RECIPE if batch_vectors==1 else BATCH_RECIPE)
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("Artifact directory must not be a symlink")
        self.directory = self.directory.resolve()
        self.read_only = read_only
        self.expected_plan = plan
        self.fingerprint = fingerprint if fingerprint is not None else device_fingerprint()
        if batch_vectors > 1 and not _program_kind:
            self.fingerprint = dict(self.fingerprint, batch_graph_author_sha256=
                file_sha256(Path(__file__).with_name("author_batched_fc.py")))
        if fused_gate_up:
            self.fingerprint = dict(self.fingerprint, fused_gate_up_author_sha256=
                file_sha256(Path(__file__).with_name("fused_ffn.py")))
        if _program_kind == 'full-ffn-v1':
            self.fingerprint = dict(self.fingerprint, full_ffn_author_sha256=
                file_sha256(Path(__file__).with_name("author_full_ffn.py")))
        if _program_kind == 'dynamic-fc-v1':
            self.fingerprint = dict(self.fingerprint, dynamic_fc_author_sha256=
                file_sha256(Path(__file__).with_name('author_dynamic_fc.py')))
        if _program_kind in ('grouped-fc-v1','grouped-columns-v1'):
            self.fingerprint=dict(self.fingerprint,grouped_fc_author_sha256=
                file_sha256(Path(__file__).with_name('author_grouped_quant.py')))
        if _program_kind == 'grouped-columns-v1':
            self.fingerprint=dict(self.fingerprint,grouped_columns_author_sha256=
                file_sha256(Path(__file__).with_name('grouped_columns.py')))
        self.manifest_path = self.directory / "MODEL.json"
        self.graphs = self.directory / "graphs"
        with _package_root(self.directory, read_only, self.recipe, self.format):
            self._initialize(source_gguf, source_sha256, plan, verify_source, source_identity)

    def _initialize(self, source_gguf, source_sha256, plan, verify_source, source_identity):
        if self.graphs.is_symlink():
            raise ValueError("Artifact graph directory must not be a symlink")
        existing = self._manifest() if self.manifest_path.exists() else None
        if existing and existing.get("source_gguf", {}).get("embedded"):
            _embedded_path(self.directory, existing["source_gguf"])
        if source_gguf is None:
            if existing is None:
                raise ValueError("Source GGUF is required for a new artifact package")
            source = existing["source_gguf"]
            if source["embedded"]:
                source_gguf = _embedded_path(self.directory, source)
            else:
                source_gguf = Path(source["path"])
        self.source_gguf = Path(source_gguf).resolve()
        actual, identity = _source_snapshot(self.source_gguf, hash_bytes=verify_source or source_sha256 is None)
        actual = actual if actual is not None else source_sha256
        if source_identity is not None and identity != tuple(source_identity):
            raise ValueError("Artifact source inode differs from the GGUF reader mapping")
        if not _digest(actual) or (source_sha256 is not None and actual != source_sha256):
            raise ValueError("Source GGUF SHA256 does not match the requested identity")
        self.source_sha256 = actual
        self.source_bytes = self.source_gguf.stat().st_size
        self.source_identity = identity
        if plan is not None and plan.get("model_sha256") != self.source_sha256:
            raise ValueError("Architecture plan belongs to different GGUF bytes")
        if existing is not None:
            self._validate_manifest(existing)
        elif self.read_only:
            raise FileNotFoundError("Persistent compiled-model manifest is missing")
        else:
            value = {"format": self.format, "recipe": self.recipe, "created_unix_seconds": time.time(),
                     "source_gguf": {"path": str(self.source_gguf), "embedded": False,
                                     "bytes": self.source_bytes, "sha256": actual},
                     "device_fingerprint": self.fingerprint, "plan": plan,
                     "complete": False, "completed_weight_dtypes": [], "artifacts": {}}
            if self.batch_vectors > 1:
                value["batch_vectors"] = self.batch_vectors
            # The enclosing directory flock serializes all initializers.
            # Commit identity before auxiliaries so a failed first rename
            # leaves an empty, retryable output rather than an unrecognized
            # directory containing only graphs/ and .lock.
            _atomic(self.manifest_path, canonical(value) + b"\n")
        if not self.read_only:
            self.graphs.mkdir(mode=0o700, parents=True, exist_ok=True)
            self.graphs.chmod(0o700)

    def _manifest(self):
        if self.manifest_path.is_symlink():
            raise ValueError("Compiled-model manifest must not be a symlink")
        return json.loads(self.manifest_path.read_text())

    def _validate_manifest(self, value):
        stat = self.source_gguf.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != self.source_identity:
            raise ValueError("Source GGUF changed during this artifact session")
        if (value.get("format") != self.format or value.get("recipe") != self.recipe
                or value.get("batch_vectors", 1) != self.batch_vectors):
            raise ValueError("Unsupported compiled artifact format/FC recipe")
        source = value.get("source_gguf", {})
        if source.get("sha256") != self.source_sha256 or source.get("bytes") != self.source_bytes:
            raise ValueError("Compiled artifacts belong to different GGUF weights")
        if value.get("device_fingerprint") != self.fingerprint:
            raise ValueError("Compiled artifacts belong to a different compiler/firmware fingerprint")
        if self.expected_plan is not None and canonical(value.get("plan")) != canonical(self.expected_plan):
            raise ValueError("Compiled artifact architecture plan differs; use its original bank layout or a new package")

    def _update(self, change):
        if self.read_only:
            raise PermissionError("Compiled artifact package is read-only")
        fd = os.open(self.directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            value = self._manifest()
            self._validate_manifest(value)
            change(value)
            _atomic(self.manifest_path, canonical(value) + b"\n")
        finally:
            os.close(fd)

    def identity(self, name, start, stop, dtype, columns):
        if dtype not in ("int8", "float32") or not isinstance(name, str) or not name:
            raise ValueError("Invalid FC artifact key")
        if any(type(value) is not int for value in (start, stop, columns)) or not (0 <= start < stop and columns > 0):
            raise ValueError("Invalid FC artifact dimensions")
        fields = {"tensor": name, "row_start": start, "row_stop": stop,
                  "rows": stop - start, "columns": columns, "weight_dtype": dtype,
                  "source_gguf_sha256": self.source_sha256, "recipe": self.recipe,
                  "device_fingerprint_sha256": hashlib.sha256(canonical(self.fingerprint)).hexdigest()}
        if self.batch_vectors > 1 or self.fused_gate_up:
            if dtype != "float32":
                raise ValueError("Batched/fused artifacts require float32 constants")
        if self.batch_vectors > 1:
            fields["batch_vectors"] = self.batch_vectors
        return hashlib.sha256(canonical(fields)).hexdigest(), fields

    def bank_rows(self, name):
        plan = self._manifest().get("plan") or {}
        return next((item["bank_rows"] for item in plan.get("dense_tensors", []) if item["name"] == name), None)

    def _output_bytes(self, name, start, stop, columns):
        return self.batch_vectors * (stop-start) * 4

    def get(self, name, start, stop, dtype, columns, *, reserve=None, load_code=True):
        key, expected = self.identity(name, start, stop, dtype, columns)
        manifest = self._manifest()
        self._validate_manifest(manifest)
        entry = manifest["artifacts"].get(key)
        if entry is None:
            if self.read_only:
                raise FileNotFoundError(f"No compiled artifact for {name}[{start}:{stop}]")
            return None
        if not _digest(entry.get("metadata_sha256")) or entry.get("metadata") != f"graphs/{key}-{entry['metadata_sha256']}.json":
            raise ValueError("Invalid compiled artifact metadata reference")
        path = self.directory / entry["metadata"]
        if path.is_symlink() or file_sha256(path) != entry["metadata_sha256"]:
            raise ValueError("Compiled artifact metadata hash mismatch")
        record = json.loads(path.read_text())
        if any(record.get(field) != value for field, value in expected.items()):
            raise ValueError("Compiled artifact matrix identity mismatch")
        if record.get("device_fingerprint") != self.fingerprint:
            raise ValueError("Compiled artifact fingerprint mismatch")
        digest = record.get("bytecode_sha256")
        if not _digest(digest) or record.get("bytecode_path") != f"graphs/{digest}.dgc":
            raise ValueError("Invalid compiled bytecode path/hash")
        if not _digest(record.get("graph_sha256")) or record.get("input_bytes") != self.batch_vectors * columns * 4 or record.get("output_bytes") != self._output_bytes(name, start, stop, columns):
            raise ValueError("Invalid compiled graph hash/ports")
        code_path = self.directory / record["bytecode_path"]
        if code_path.is_symlink() or not 8 <= record.get("compiled_bytes", 0) <= MAX_CODE_BYTES:
            raise ValueError("Invalid compiled bytecode size/path")
        if code_path.stat().st_size != record["compiled_bytes"]:
            raise ValueError("Compiled bytecode size mismatch")
        if not load_code:
            with code_path.open("rb") as source:
                header = source.read(8)
            if header[4:8] != b"DGC0" or file_sha256(code_path) != digest:
                raise ValueError("Compiled bytecode hash/identifier mismatch")
            return record, None
        if reserve is not None:
            reserve(record["compiled_bytes"])
        code = code_path.read_bytes()
        if hashlib.sha256(code).hexdigest() != digest or code[4:8] != b"DGC0":
            raise ValueError("Compiled bytecode hash/identifier mismatch")
        return record, code

    def store(self, name, start, stop, dtype, columns, record, code):
        if self.read_only:
            raise PermissionError("Compiled artifact package is read-only")
        key, fields = self.identity(name, start, stop, dtype, columns)
        if not 8 <= len(code) <= MAX_CODE_BYTES or code[4:8] != b"DGC0":
            raise ValueError("Native export is not a bounded DGC0 bytecode artifact")
        if not _digest(record.get("graph_sha256")) or record.get("compiled_bytes") != len(code):
            raise ValueError("Native export metadata does not match its bytecode")
        digest = hashlib.sha256(code).hexdigest()
        fields.update(graph_sha256=record["graph_sha256"], graph_bytes=record["graph_bytes"],
                      input_bytes=self.batch_vectors * columns * 4,
                      output_bytes=self._output_bytes(name, start, stop, columns),
                      compiled_bytes=len(code), bytecode_sha256=digest,
                      bytecode_path=f"graphs/{digest}.dgc", device_fingerprint=self.fingerprint)
        code_path = self.directory / fields["bytecode_path"]
        if code_path.exists():
            if code_path.is_symlink() or file_sha256(code_path) != digest:
                raise ValueError("Existing content-addressed bytecode is corrupted")
        else:
            _atomic(code_path, code)
        data = canonical(fields) + b"\n"
        metadata_sha256 = hashlib.sha256(data).hexdigest()
        metadata_path = self.graphs / f"{key}-{metadata_sha256}.json"
        _atomic(metadata_path, data)
        self._update(lambda value: value["artifacts"].update({key: {
            "metadata": f"graphs/{key}-{metadata_sha256}.json", "metadata_sha256": metadata_sha256}}))
        return fields

    def mark_complete(self, weight_dtype="int8"):
        plan = self._manifest().get("plan") or {}
        if not plan.get("dense_tensors") or plan.get("unsupported_tensor_operations"):
            raise ValueError("Cannot mark an empty or unsupported tensor plan complete")
        for tensor in plan["dense_tensors"]:
            for start in range(0, tensor["rows"], tensor["bank_rows"]):
                found = self.get(tensor["name"], start, min(start+tensor["bank_rows"], tensor["rows"]),
                                 weight_dtype, tensor["columns"], load_code=False)
                if found is None:
                    raise ValueError("Cannot mark incomplete tensor artifacts complete")
        def complete(value):
            dtypes = set(value.get("completed_weight_dtypes", []))
            dtypes.add(weight_dtype)
            value.update(complete=True, completed_weight_dtypes=sorted(dtypes))
        self._update(complete)

    def embed_source(self):
        """Copy source weights atomically; callers preflight disk first."""
        target = self.directory / "model.gguf"
        if target.is_symlink():
            raise ValueError("Embedded model destination must not be a symlink")
        if target.resolve() == self.source_gguf:
            pass
        elif target.exists():
            if target.is_symlink() or file_sha256(target) != self.source_sha256:
                raise ValueError("Embedded model.gguf does not match source weights")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".model-", dir=self.directory)
            try:
                with os.fdopen(fd, "wb") as destination, self.source_gguf.open("rb") as source:
                    os.fchmod(destination.fileno(), 0o600)
                    for block in iter(lambda: source.read(4 * 1024 ** 2), b""):
                        destination.write(block)
                    destination.flush()
                    os.fsync(destination.fileno())
                if file_sha256(temporary) != self.source_sha256:
                    raise ValueError("Source GGUF changed while copying")
                os.replace(temporary, target)
            finally:
                Path(temporary).unlink(missing_ok=True)
        self._update(lambda value: value["source_gguf"].update(path="model.gguf", embedded=True))
