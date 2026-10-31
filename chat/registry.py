"""Root-owned model registration and idle selection; inference requests stay fixed."""
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid

REGISTRY_SCHEMA = "pixelforge.models.registry.v1"
CATALOG_SCHEMA = "pixelforge.models.v1"
BACKEND = "pixelforge-npu-dense+ggml-cpu"
MAX_MODELS = 64
MODEL_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
SUMMARY_FIELDS = ("model_name", "architecture", "model_sha256", "provenance_repo",
                  "context_per_worker", "max_output_tokens", "execution")

def validate_model_id(value):
    if not isinstance(value, str) or MODEL_ID.fullmatch(value) is None:
        raise ValueError("Model ID must contain 1 to 64 lowercase letters, digits, underscores or hyphens")
    return value

def _absolute(value):
    if not isinstance(value, (str, Path)):
        raise ValueError("Registry paths must be absolute")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Registry paths must be absolute without parent traversal")
    return path

def _directory(path, owner=0):
    path = _absolute(path)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != owner or info.st_mode & 0o022:
        raise ValueError("Registry directories must have their expected owner and no group/world write access")
    return path

def _read_bytes(path, owner=0, limit=2 * 1024**2):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != owner or info.st_mode & 0o022:
            raise ValueError("Registry input must be a protected regular file with the expected owner")
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Registry input exceeds its size limit")
    return data

def _json(path, owner=0):
    return json.loads(_read_bytes(path, owner))

def _bootstrap(value):
    if not isinstance(value, dict) or set(value) != {"ubuntu_root", "python", "service", "ubuntu_pins"}:
        raise ValueError("Invalid registered bootstrap")
    for key in ("ubuntu_root", "python", "service"):
        _absolute(value[key])
    pins = value["ubuntu_pins"]
    if not isinstance(pins, dict) or not pins or len(pins) > 512:
        raise ValueError("Registered bootstrap must pin its runtime")
    for path, expected in pins.items():
        _absolute(path)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError("Invalid registered source pin")
    if value["python"] not in pins or value["service"] not in pins:
        raise ValueError("Registered bootstrap must pin interpreter and service")
    return value

@contextmanager
def _verified_root(bootstrap, verifier=None):
    """Verify the same mounted-root/source pins as the fixed inference launcher."""
    _bootstrap(bootstrap)
    if verifier is None:
        from android_launcher import ubuntu_handles
        verifier = ubuntu_handles
    root_fd, namespace_fd = verifier(bootstrap)
    try:
        yield Path("/proc/self/fd") / str(root_fd)
    finally:
        os.close(root_fd)
        os.close(namespace_fd)

def _metadata(value, state, model_id):
    if not isinstance(value, dict) or value.get("backend") != BACKEND:
        raise ValueError("Unsupported registered inference backend")
    model = value.get("model")
    if (not isinstance(model, dict) or not isinstance(model.get("model_name"), str)
            or not model["model_name"].strip() or len(model["model_name"]) > 512
            or not isinstance(model.get("model_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", model["model_sha256"]) is None):
        raise ValueError("Registered model metadata must name and pin its source")
    launcher = str(state / "chat.py")
    if value.get("launcher", launcher) != launcher:
        raise ValueError("Registered models must use the fixed inference launcher")
    return dict(value, launcher=launcher, model_id=model_id)

def _entry(state, model_id, bootstrap, metadata, app_runtime, uid, label_source=None):
    if type(uid) is not int or uid < 1:
        raise ValueError("Client metadata must belong to an ordinary application UID")
    runtime = _directory(app_runtime, uid)
    label = _directory(label_source if label_source is not None else runtime, uid)
    return {"bootstrap": _bootstrap(bootstrap),
            "client_metadata": _metadata(metadata, state, model_id),
            "app_runtime": str(runtime), "termux_uid": uid, "label_source": str(label)}

def _load(state):
    path = state / "models.json"
    if not path.exists() and not path.is_symlink():
        return None
    data = _json(path)
    if (not isinstance(data, dict) or set(data) != {"schema", "selected_model_id", "models"}
            or data["schema"] != REGISTRY_SCHEMA or not isinstance(data["models"], dict)
            or len(data["models"]) > MAX_MODELS):
        raise ValueError("Invalid root-owned model registry")
    for ident, entry in data["models"].items():
        validate_model_id(ident)
        if not isinstance(entry, dict) or set(entry) != {"bootstrap", "client_metadata", "app_runtime", "termux_uid", "label_source"}:
            raise ValueError("Invalid registered model entry")
        validated = _entry(state, ident, entry["bootstrap"], entry["client_metadata"],
                           entry["app_runtime"], entry["termux_uid"], entry["label_source"])
        if validated != entry:
            raise ValueError("Registered model entry is not canonical")
    selected = data["selected_model_id"]
    if selected is not None and selected not in data["models"]:
        raise ValueError("Selected model is missing from the registry")
    return data

def _catalog(data):
    result = {}
    for ident, entry in sorted(data["models"].items()):
        metadata = entry["client_metadata"]
        result[ident] = {"backend": metadata["backend"], "experimental": bool(metadata.get("experimental", True)),
                         "model": {key: metadata["model"][key] for key in SUMMARY_FIELDS if key in metadata["model"]}}
    return {"schema": CATALOG_SCHEMA, "selected_model_id": data["selected_model_id"], "models": result}

@contextmanager
def _locked(state):
    if os.getuid() != 0:
        raise ValueError("Model management requires the existing root environment")
    _directory(state)
    path = state / "chat.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    with os.fdopen(fd, "a") as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("Invalid global inference lock")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield

def _native_xattr_function(name, arguments, result):
    """Android CPython may omit os xattr wrappers although bionic provides them."""
    library = ctypes.CDLL(None, use_errno=True)
    try:
        function = getattr(library, name)
    except AttributeError as error:
        raise RuntimeError("Runtime cannot preserve SELinux file labels: missing " + name) from error
    function.argtypes, function.restype = arguments, result
    return function


def _native_getxattr(path, name):
    function = _native_xattr_function("getxattr",
        [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_ssize_t)
    path, name = os.fsencode(path), os.fsencode(name)
    for _ in range(3):
        size = function(path, name, None, 0)
        if size < 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        if size > 4096:
            raise ValueError("Filesystem label exceeds its size limit")
        value = ctypes.create_string_buffer(max(size, 1))
        length = function(path, name, value, size)
        if length >= 0:
            return value.raw[:length]
        number = ctypes.get_errno()
        if number != errno.ERANGE:
            raise OSError(number, os.strerror(number))
    raise OSError(errno.ERANGE, "Filesystem label changed during bounded reads")


def _native_setxattr(descriptor, name, value):
    function = _native_xattr_function("fsetxattr",
        [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int)
    data = ctypes.create_string_buffer(value)
    if function(descriptor, os.fsencode(name), data, len(value), 0) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))


def _read_selinux_label(source):
    if source is None:
        return None
    try:
        getter = getattr(os, "getxattr", None)
        value = getter(source, "security.selinux") if callable(getter) else _native_getxattr(source, "security.selinux")
    except OSError as error:
        if error.errno in (errno.ENODATA, errno.ENOTSUP):
            return None
        raise
    if not value or len(value) > 4096:
        raise ValueError("Invalid filesystem SELinux label")
    # Resolve the setter before any selected/bootstrap metadata is published.
    if not callable(getattr(os, "setxattr", None)):
        _native_xattr_function("fsetxattr",
            [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int], ctypes.c_int)
    return value


def _set_selinux_label(descriptor, value):
    try:
        setter = getattr(os, "setxattr", None)
        if callable(setter):
            setter(descriptor, "security.selinux", value)
        else:
            _native_setxattr(descriptor, "security.selinux", value)
    except OSError as error:
        if error.errno not in (errno.ENODATA, errno.ENOTSUP):
            raise


def _atomic_bytes(path, data, uid=0, label_source=None, *, label=None, label_checked=False):
    directory = _directory(path.parent, uid)
    if not label_checked:
        label = _read_selinux_label(label_source)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    temporary = ".pixelforge-model-" + uuid.uuid4().hex
    descriptor = None
    try:
        info = os.fstat(fd)
        if info.st_uid != uid or info.st_mode & 0o022:
            raise ValueError("Metadata directory ownership changed")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=fd)
        os.fchown(descriptor, uid, uid)
        if label is not None:
            _set_selinux_label(descriptor, label)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path.name, src_dir_fd=fd, dst_dir_fd=fd)
        os.fsync(fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass
        os.close(fd)

def _transaction(writes):
    """Publish while the global lock is held; restore previous files on failure."""
    saved, labels = [], {}
    for path, value, uid, label_source in writes:
        before = None
        if path.exists() or path.is_symlink():
            before = _read_bytes(path, uid)
        encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
        if label_source not in labels:
            labels[label_source] = _read_selinux_label(label_source)
        saved.append((path, encoded, before, uid, label_source, labels[label_source]))
    changed = []
    try:
        for path, encoded, before, uid, label_source, label in saved:
            changed.append((path, before, uid, label_source, label))
            _atomic_bytes(path, encoded, uid, label_source, label=label, label_checked=True)
    except BaseException as original:
        errors = []
        for path, before, uid, label_source, label in reversed(changed):
            try:
                if before is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_bytes(path, before, uid, label_source, label=label, label_checked=True)
            except BaseException as failure:
                errors.append(str(path) + ": " + type(failure).__name__ + ": " + str(failure))
        if errors:
            raise RuntimeError("Model selection failed; rollback incomplete: " + "; ".join(errors)) from original
        raise

def _publications(state, data, *, active):
    catalog = _catalog(data)
    writes = []
    clients = {}
    for entry in data["models"].values():
        runtime = entry["app_runtime"]
        identity = (entry["termux_uid"], entry["label_source"])
        if runtime in clients and clients[runtime][0] != identity[0]:
            raise ValueError("Conflicting registered client metadata owners")
        clients.setdefault(runtime, identity)
    if active:
        entry = data["models"][data["selected_model_id"]]
        writes.append((state / "bootstrap.json", entry["bootstrap"], 0, None))
    for runtime, (uid, label) in clients.items():
        directory = Path(runtime)
        if active:
            writes.append((directory / "pixelforge-chat.json", entry["client_metadata"], uid, label))
        writes.append((directory / "pixelforge-models.json", catalog, uid, label))
    writes.append((state / "models.json", data, 0, None))
    return writes

def _legacy_entry(state, app_runtime, uid, label_source, verifier):
    path = state / "bootstrap.json"
    if not path.exists() and not path.is_symlink():
        return None
    bootstrap = _bootstrap(_json(path))
    profile_path = str(Path(bootstrap["service"]).with_name("profile.json"))
    if profile_path not in bootstrap["ubuntu_pins"]:
        raise ValueError("Legacy bootstrap does not pin its profile")
    with _verified_root(bootstrap, verifier) as root:
        profile = _json(root / profile_path.lstrip("/"))
    model = dict(profile["model"])
    if "architecture" in profile:
        model.setdefault("architecture", profile["architecture"])
    metadata = {"backend": profile["backend"], "experimental": True,
                "model": model, "launcher": str(state / "chat.py")}
    return _entry(state, "default", bootstrap, metadata, app_runtime, uid, label_source)

def register_model(state_dir, model_id, bootstrap, client_metadata, app_runtime, termux_uid,
                   *, label_source=None, replace=False, select=False, verifier=None, installer=None):
    """Register an independently installed/pinned bundle; optionally select it."""
    ident = validate_model_id(model_id)
    state = _absolute(state_dir)
    with _locked(state):
        data = _load(state)
        if data is None:
            data = {"schema": REGISTRY_SCHEMA, "selected_model_id": None, "models": {}}
            legacy = _legacy_entry(state, app_runtime, termux_uid, label_source, verifier)
            if legacy is not None:
                data["models"]["default"] = legacy
                data["selected_model_id"] = "default"
        entry = _entry(state, ident, bootstrap, client_metadata, app_runtime, termux_uid, label_source)
        existing = data["models"].get(ident)
        if existing is not None and existing != entry:
            if not replace:
                raise ValueError("Model ID already registered; explicit replacement required")
            if data["selected_model_id"] == ident and not select:
                raise ValueError("Replacing the active model requires explicit selection")
        if existing is None and len(data["models"]) >= MAX_MODELS:
            raise ValueError("Model registry is full")
        if installer is not None:
            if not callable(installer):
                raise ValueError("Installer callback must be callable")
            installer()
        with _verified_root(entry["bootstrap"], verifier):
            pass
        data["models"][ident] = entry
        if select:
            data["selected_model_id"] = ident
        _transaction(_publications(state, data, active=select))
        return _catalog(data)

def select_model(state_dir, model_id, *, verifier=None):
    """Select a registered ID only when inference is idle and its sources verify."""
    ident = validate_model_id(model_id)
    state = _absolute(state_dir)
    with _locked(state):
        data = _load(state)
        if data is None or ident not in data["models"]:
            raise ValueError("Unknown registered model ID")
        with _verified_root(data["models"][ident]["bootstrap"], verifier):
            pass
        data["selected_model_id"] = ident
        _transaction(_publications(state, data, active=True))
        return _catalog(data)

def list_models(state_dir):
    """Return passive metadata; listing is not a runtime-health or NPU proof."""
    if os.getuid() != 0:
        raise ValueError("Model management requires the existing root environment")
    state = _directory(state_dir)
    data = _load(state)
    if data is None:
        return {"schema": CATALOG_SCHEMA, "selected_model_id": None, "models": {}}
    return _catalog(data)
