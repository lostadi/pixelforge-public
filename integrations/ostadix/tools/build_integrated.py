#!/usr/bin/env python3
"""Build one executable, parse-valid .O containing the complete offline GGUF kit."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
# Compiler export includes the stress-verified interrupted-build recovery fix.
# A new compiler release deliberately requires updating this provenance pin.
EXPECTED_ARCHIVE_SHA256 = "62fd651ff57242ca5f71a5ce861f90ac320cd536e09bf4d8b8e751485c83bc16"
SHEBANG = ("#!/usr/bin/env -S python3 -c 'import sys;exec(open(sys.argv[1]).read(262144)"
           ".split(\"\\n\",1)[1].split(\"python^(\",1)[1].split(\")_python\",1)[0])'\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def published_snapshot(path, owned):
    expected = (owned.st_dev, owned.st_ino)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as source:
        before = os.fstat(source.fileno())
        if (before.st_dev, before.st_ino) != expected:
            raise ValueError("Published output was replaced")
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(source.fileno())
    current = path.lstat()
    if (file_identity(before) != file_identity(after)
            or (current.st_dev, current.st_ino) != expected):
        raise ValueError("Published output changed while hashing")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def inspect_archive(path):
    before = path.stat()
    with path.open("rb") as source:
        header = source.read(16384)
        offset_match = re.search(rb'OFFSET = int\("(\d{20})"\)', header)
        payload_match = re.search(rb'PAYLOAD_SHA256 = "([0-9a-f]{64})"', header)
        if not offset_match or not payload_match:
            raise ValueError("Not a PixelForge self-extracting compiler archive")
        offset = int(offset_match.group(1))
        if not 0 < offset < before.st_size:
            raise ValueError("Invalid compiler archive payload offset")
        source.seek(offset)
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest().encode() != payload_match.group(1):
            raise ValueError("Compiler archive payload hash mismatch")
        source.seek(offset)
        entries = []
        names = set()
        with tarfile.open(fileobj=source, mode="r|gz") as packed:
            for member in packed:
                name = PurePosixPath(member.name)
                if (not member.isfile() or name.is_absolute() or ".." in name.parts
                        or str(name) != member.name or member.name in names):
                    raise ValueError("Unsafe or duplicate compiler archive member")
                names.add(member.name)
                digest = hashlib.sha256()
                size = 0
                with packed.extractfile(member) as current:
                    for chunk in iter(lambda: current.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                if size != member.size:
                    raise ValueError("Truncated compiler archive member")
                entries.append({"path": member.name, "bytes": size,
                                "sha256": digest.hexdigest()})
    after = path.stat()
    if file_identity(before) != file_identity(after):
        raise ValueError("Compiler archive changed during inspection")
    return entries, file_identity(before)


def build(archive, output):
    archive = Path(archive).resolve()
    output = Path(output).expanduser().absolute()
    for candidate in (output, output.with_suffix(".O.sha256"), output.with_suffix(".O.json")):
        if candidate.exists() or candidate.is_symlink():
            raise FileExistsError("Choose fresh output and receipt paths: " + str(candidate))
    entries, identity = inspect_archive(archive)
    digest = sha256_file(archive)
    if digest != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Compiler archive does not match the verified release pin")
    if file_identity(archive.stat()) != identity:
        raise ValueError("Compiler archive changed after inspection")
    metadata = ("ARCHIVE_SHA256 = " + repr(digest) + "\nARCHIVE_BYTES = "
                + str(archive.stat().st_size) + "\nARCHIVE_FILES = " + repr(entries) + "\n")
    worker = HERE / "worker.py"
    if not worker.is_file():
        raise FileNotFoundError("Worker lifetime guard source is required: " + str(worker))
    metadata += "WORKER_SOURCE = " + repr(worker.read_text()) + "\n"
    source = metadata + (HERE / "bootstrap.py").read_text() + "\n" + (HERE / "controller.py").read_text()
    source += '\nif "O" in globals():\n    __oval_result__ = _execute_integrated()\nelse:\n    _bootstrap()\n'
    if ")_python" in source or "python^(" in source:
        raise ValueError("Embedded controller contains an O delimiter")
    if len(source.encode()) + len(SHEBANG.encode()) > 250000:
        raise ValueError("Bootstrap prefix exceeds the shebang reader span")
    compile(source, "<integrated-O-controller>", "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pixelforge-O-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(SHEBANG.encode())
            target.write(b"O^(\npython^(\n")
            target.write(source.encode())
            target.write(b")_python\n)_O\n")
            target.write(b"# OSTADIX_NPU_PAYLOAD_V1 " + digest.encode() + b"\n")
            embedded_digest = hashlib.sha256()
            with archive.open("rb") as packed:
                for chunk in iter(lambda: packed.read(3072), b""):
                    embedded_digest.update(chunk)
                    target.write(b"# " + base64.b64encode(chunk) + b"\n")
            if (embedded_digest.hexdigest() != digest
                    or file_identity(archive.stat()) != identity):
                raise ValueError("Compiler archive changed while embedding")
            target.write(b"# END_OSTADIX_NPU_PAYLOAD_V1\n")
            target.flush()
            os.fsync(target.fileno())
        os.chmod(temporary, 0o755)
        owned_output = Path(temporary).lstat()
        # Exclusive publication refuses a concurrently created user output.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    created = []
    try:
        receipt = {"format": "ostadix.pixelforge-single-file.v1", "output": str(output),
                   **published_snapshot(output, owned_output),
                   "compiler_archive_sha256": digest, "embedded_files": len(entries),
                   "execution": "ostadix-BM Python backend orchestration",
                   "prerequisites": ["ostadix-BM O evaluator and Python backend", "Ubuntu ARM64 Python 3.12",
                                     "compatible rooted Android Tensor vendor runtime", "user-supplied GGUF"]}
        receipts = [(output.with_suffix(".O.sha256"), receipt["sha256"] + "  " + output.name + "\n"),
                    (output.with_suffix(".O.json"), json.dumps(receipt, indent=2) + "\n")]
        for path, content in receipts:
            with path.open("x", encoding="utf8") as target:
                created.append((path, os.fstat(target.fileno())))
                target.write(content)
        if published_snapshot(output, owned_output) != {key: receipt[key] for key in ("bytes", "sha256")}:
            raise ValueError("Published output changed before receipt completion")
    except BaseException:
        for path, owned in created + [(output, owned_output)]:
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("/root/pixelforge-gguf-compiler.run"))
    parser.add_argument("--output", type=Path, default=Path("/root/pixelforge-npu.O"))
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.output), indent=2))


if __name__ == "__main__":
    main()
