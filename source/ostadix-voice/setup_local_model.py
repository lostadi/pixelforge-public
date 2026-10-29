#!/usr/bin/env python3
"""Reproduce the pinned, local Huihui model runtime without a cloud API.

Run as the normal Termux user. Downloads are atomic and SHA-256 verified.
GGUFs come from Huihui or pinned community conversions of Huihui weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
SOURCE_COMMIT = "254b177307b14d055bbe4c050c9c441a1e06d8a4"
MODEL_REPO = "mradermacher/Huihui-LFM2.5-1.2B-Instruct-abliterated-GGUF"
MODEL_REVISION = "7f2f76e3de27f8cf9ec3ad2efe4dc68dd6c05265"
PROVENANCE_REPO = "huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated"
PROVENANCE_REVISION = "f76bd329399236de76a1b524e0ea20c82d57c534"
MODEL_NAME = "Huihui-LFM2.5-1.2B-Instruct-abliterated"
VARIANTS = {
    "Q8_0": {
        "filename": MODEL_NAME + ".Q8_0.gguf",
        "bytes": 1246254720,
        "sha256": "6f4c9e0baf72f74c486c8e44a0e64475f8f73042f28e1704f7929b77b70705e7",
    },
    "F16": {
        "filename": MODEL_NAME + ".f16.gguf",
        "bytes": 2343327360,
        "sha256": "dc692c3a639f4c862ce6c1e71c80e431c7a15a89ec06e8429cdbeadc1e87fce5",
    },
}

# Candidate pins were reviewed in runtime/model-candidates.json and
# runtime/model-survey.json. Keep the existing LFM constants and positional
# APIs stable for installed workflows.
PROFILES = {
    "lfm2": {
        "model_name": MODEL_NAME,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "provenance_repo": PROVENANCE_REPO,
        "provenance_revision": PROVENANCE_REVISION,
        "variants": VARIANTS,
    },
    "minicpm5": {
        "model_name": "Huihui-MiniCPM5-2B-abliterated",
        "model_repo": "mradermacher/Huihui-MiniCPM5-2B-abliterated-GGUF",
        "model_revision": "fc1e8318e85635d142e95a0f9977487717c40b58",
        "provenance_repo": "huihui-ai/Huihui-MiniCPM5-2B-abliterated",
        "provenance_revision": "aea962bb5558e659624eeefc8c1b9f418a3e6ffb",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0},
        "chat_template_kwargs": {"enable_thinking": False},
        "variants": {
            "Q8_0": {
                "filename": "Huihui-MiniCPM5-2B-abliterated.Q8_0.gguf",
                "bytes": 2679711680,
                "sha256": "ffc8f95be9e7fab3c8d8a50c6634828433b31ca80a803349f47dc3b35aaaeeab",
            },
        },
    },
    "gemma4": {
        "model_name": "Huihui-gemma-4-E2B-it-abliterated-v2",
        "model_repo": "mradermacher/Huihui-gemma-4-E2B-it-abliterated-v2-GGUF",
        "model_revision": "d6d7c4cd25430f96319715690b4839d9b8990dd0",
        "provenance_repo": "huihui-ai/Huihui-gemma-4-E2B-it-abliterated-v2",
        "provenance_revision": "2f7b088470ce94da58fc1e5319b8ed88aed685a6",
        "default_quantization": "Q6_K",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0},
        "chat_template_kwargs": {"enable_thinking": False},
        "variants": {
            "Q6_K": {
                "filename": "Huihui-gemma-4-E2B-it-abliterated-v2.Q6_K.gguf",
                "bytes": 3845340768,
                "sha256": "d58c9604501caf56d272f5abc848b18d05970b3ccad3b019eb32df072f1871c0",
            },
        },
    },
    "qwen35": {
        "model_name": "Huihui-Qwen3.5-2B-abliterated",
        "model_repo": "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF",
        "model_revision": "f36848fead3fdda244cf60195c46993d23183d4c",
        "provenance_repo": "huihui-ai/Huihui-Qwen3.5-2B-abliterated",
        "provenance_revision": "b2e291a65f29a9b148981fa5299caea5d35bd4c8",
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 20, "min_p": 0.0,
                     "presence_penalty": 2.0, "repeat_penalty": 1.0},
        "chat_template_kwargs": {"enable_thinking": False},
        "variants": {
            "Q8_0": {
                "filename": "Huihui-Qwen3.5-2B-abliterated.Q8_0.gguf",
                "bytes": 2012012992,
                "sha256": "fb724dfeb9c268e4752b944d35213e959b525cab39db021cc2bdf439901d4deb",
            },
        },
    },
    "granite41": {
        "model_name": "Huihui-granite-4.1-3b-abliterated",
        "model_repo": "mradermacher/Huihui-granite-4.1-3b-abliterated-GGUF",
        "model_revision": "c1344c80b4d9565d56c65acfce7725356d461c81",
        "provenance_repo": "huihui-ai/Huihui-granite-4.1-3b-abliterated",
        "provenance_revision": "338ec9f827d47e19fa854c2e3467bfbfd4fa486d",
        "default_quantization": "Q6_K",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
        "variants": {
            "Q6_K": {
                "filename": "Huihui-granite-4.1-3b-abliterated.Q6_K.gguf",
                "bytes": 2795617824,
                "sha256": "e80eec4cf0c8cc0b894117535707139f8e86ebc01d9ac7b11a11dad33508e7f9",
            },
        },
    },
    "llama32": {
        "model_name": "Llama-3.2-3B-Instruct-abliterated",
        "model_repo": "mradermacher/Llama-3.2-3B-Instruct-abliterated-GGUF",
        "model_revision": "0b0fc4afe030af5b9bf106fe61b64967e3de67be",
        "provenance_repo": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "provenance_revision": "ba0be3c4683117ffe70be5cc767723e0210e437e",
        "default_quantization": "Q6_K",
        "sampling": {"temperature": 0.6, "top_p": 0.9, "top_k": 50, "min_p": 0.0},
        "variants": {
            "Q6_K": {
                "filename": "Llama-3.2-3B-Instruct-abliterated.Q6_K.gguf",
                "bytes": 2967059424,
                "sha256": "7eba702b4d42fb0c911b51c1c707d984f991786dbbb4cf6831bb4adb286eb19c",
            },
        },
    },
    "neohorse": {
        "model_name": "Huihui-NeoHorse-1-4B-abliterated",
        "model_repo": "huihui-ai/Huihui-NeoHorse-1-4B-abliterated-GGUF",
        "model_revision": "fb32df77e9e1852d558d1013501e19666a3eb0bb",
        "provenance_repo": "huihui-ai/Huihui-NeoHorse-1-4B-abliterated",
        "provenance_revision": "ed87d7cdfb88e4815e95741e3e003a8942b1b270",
        "default_quantization": "Q5_K",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                     "presence_penalty": 1.5, "repeat_penalty": 1.0},
        "chat_template_kwargs": {"enable_thinking": False},
        "variants": {
            "Q5_K": {
                "filename": "Huihui-NeoHorse-1-4B-abliterated-Q5_K.gguf",
                "bytes": 3161425344,
                "sha256": "cc4a3cc480fe0e0142333244aaa3f2c0d4feda6c78d296976e8513d02aabbbcd",
            },
        },
    },
    "spark": {
        "model_name": "Huihui-Spark-X2.5-4B-abliterated",
        "model_repo": "okenk/Huihui-Spark-X2.5-4B-abliterated-GGUF",
        "model_revision": "f811e62278851cfcd9cd4d5303eca803fa872254",
        "provenance_repo": "huihui-ai/Huihui-Spark-X2.5-4B-abliterated",
        "provenance_revision": "2e7ab2831394c582f15d629e3c40d63e7d04bc75",
        "default_quantization": "Q4_K_M",
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0,
                     "presence_penalty": 0.0, "repeat_penalty": 1.0},
        "chat_template_kwargs": {"enable_thinking": False},
        "variants": {
            "Q4_K_M": {
                "filename": "Huihui-Spark-X2.5-4B-abliterated-Q4_K_M.gguf",
                "bytes": 2600224000,
                "sha256": "cf1d8e0bce2ab0b9bb6c1cd7f2092dd2552a840104c6609507fa24d6fc07ae54",
            },
        },
    },
}

# Use the already pinned Spark weights with a single 32k conversation slot.
PROFILES["huihui-spark-32k"] = {
    **PROFILES["spark"],
    "model_name": "huihui-spark-32k",
    "threads": 4,
    "threads_batch": 4,
    "parallel": 1,
    "context_per_worker": 32768,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
}

# One conversation slot leaves room for dialogue, OSTADIX guidance and tool
# results while retaining the pinned, already installed Huihui Gemma weights.
PROFILES["huihui-gemma4-8k"] = {
    **PROFILES["gemma4"],
    "threads": 4,
    "threads_batch": 4,
    "parallel": 1,
    "context_per_worker": 8192,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0",
}


def profile_variant(model: str, quantization: str) -> tuple[dict, dict]:
    if model not in PROFILES:
        raise SystemExit("Unknown model profile: " + str(model))
    profile = PROFILES[model]
    if quantization not in profile["variants"]:
        raise SystemExit(f"{model} supports only: {', '.join(profile['variants'])}")
    return profile, profile["variants"][quantization]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(*command: str) -> None:
    subprocess.run(command, check=True)


def build() -> Path:
    for tool in ("git", "cmake", "ninja", "clang"):
        if not shutil.which(tool):
            raise SystemExit("Missing " + tool + "; install Termux git cmake ninja clang first.")
    source = RUNTIME / "llama-cpp"
    if not source.exists():
        source.mkdir(parents=True)
        run("git", "init", str(source))
        run("git", "-C", str(source), "remote", "add", "origin", "https://github.com/ggml-org/llama.cpp.git")
        run("git", "-C", str(source), "fetch", "--depth", "1", "origin", SOURCE_COMMIT)
        run("git", "-C", str(source), "checkout", "--detach", SOURCE_COMMIT)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != SOURCE_COMMIT:
        raise SystemExit("Existing llama-cpp checkout is a different revision; refusing to replace it.")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise SystemExit("Existing llama-cpp checkout has changes; refusing to build modified sources.")
    build_dir = source / "build"
    run("cmake", "-S", str(source), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_OPENSSL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF", "-DGGML_NATIVE=ON",
        "-DGGML_LLAMAFILE=OFF", "-DGGML_OPENMP=OFF")
    run("cmake", "--build", str(build_dir), "--target", "llama-server", "llama-bench", "--parallel", "2")
    return build_dir / "bin" / "llama-server"


def download(quantization: str, model: str = "lfm2") -> Path:
    profile, variant = profile_variant(model, quantization)
    destination = RUNTIME / "models" / variant["filename"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == variant["bytes"] and digest(destination) == variant["sha256"]:
            print("Verified existing", destination.name, flush=True)
            return destination
        raise SystemExit("Existing model checksum differs; refusing to overwrite it: " + str(destination))
    if shutil.disk_usage(destination.parent).free < variant["bytes"] + 512 * 1024**2:
        raise SystemExit("Not enough free disk space for the model and 512 MiB reserve.")
    url = f"https://huggingface.co/{profile['model_repo']}/resolve/{profile['model_revision']}/{variant['filename']}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
        received = 0
        previous = time.monotonic()
        while chunk := response.read(4 * 1024**2):
            output.write(chunk)
            received += len(chunk)
            if received > variant["bytes"]:
                raise SystemExit("Download exceeds the pinned model size; refusing it.")
            if time.monotonic() - previous >= 3:
                print(f"Downloaded {received / 1024**2:.0f} / {variant['bytes'] / 1024**2:.0f} MiB", flush=True)
                previous = time.monotonic()
    if temporary.stat().st_size != variant["bytes"] or digest(temporary) != variant["sha256"]:
        raise SystemExit("Download failed size/SHA-256 verification; the untrusted .part file was not activated.")
    temporary.replace(destination)
    print("Verified", destination.name, flush=True)
    return destination


def model_manifest(quantization: str, model_path: Path, server_path: Path, model: str = "lfm2") -> dict:
    profile, variant = profile_variant(model, quantization)
    manifest = {
        "schema_version": 1,
        "server_path": str(server_path),
        "model_path": str(model_path),
        "model_sha256": variant["sha256"],
        "model_bytes": variant["bytes"],
        "model_repo": profile["model_repo"],
        "model_revision": profile["model_revision"],
        "model_name": profile["model_name"],
        "quantization": quantization,
        "threads": 2,
        "threads_batch": 4,
        "parallel": 2,
        "context_per_worker": 2048,
        "source_commit": SOURCE_COMMIT,
        "provenance_repo": profile["provenance_repo"],
        "provenance_revision": profile["provenance_revision"],
    }
    for key in ("threads", "threads_batch", "parallel", "context_per_worker",
                "cache_type_k", "cache_type_v"):
        if key in profile:
            manifest[key] = profile[key]
    for key in ("sampling", "chat_template_kwargs"):
        if key in profile:
            manifest[key] = profile[key].copy()
    return manifest


def activate(quantization: str, model_path: Path, server_path: Path, model: str = "lfm2") -> None:
    if not server_path.is_file() or not os.access(server_path, os.X_OK):
        raise SystemExit("Build llama-server before activating a model.")
    manifest = model_manifest(quantization, model_path, server_path, model)
    path = RUNTIME / "local-model.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    print("Activated", quantization, "in", path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=PROFILES, default="lfm2")
    parser.add_argument("--quantization", choices=sorted({q for p in PROFILES.values() for q in p["variants"]}))
    parser.add_argument("--download-only", action="store_true", help="Download and verify; do not change the selected model")
    arguments = parser.parse_args()
    if arguments.quantization is None:
        arguments.quantization = PROFILES[arguments.model].get("default_quantization", "Q8_0")
    profile_variant(arguments.model, arguments.quantization)
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise SystemExit("Run this setup from the normal Termux shell, not as root.")
    RUNTIME.mkdir(exist_ok=True)
    server_path = RUNTIME / "llama-cpp" / "build" / "bin" / "llama-server"
    if not arguments.download_only:
        server_path = build()
    model_path = download(arguments.quantization, arguments.model)
    if not arguments.download_only:
        activate(arguments.quantization, model_path, server_path, arguments.model)


if __name__ == "__main__":
    main()
