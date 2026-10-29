"""Inspect a GGUF container without hashing or decoding model weights."""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
from pathlib import Path

from gguf_hf_loader import GGUFWeights, quantization_name


def inspect_weights(weights, include_tensors=False):
    """Return source metadata; architecture execution support is a separate step."""
    formats = collections.defaultdict(lambda: {'tensors': 0, 'elements': 0, 'storage_bytes': 0})
    unsupported = set()
    unknown_sizes = []
    tensor_rows = []
    supported = weights.decode_supported_types
    for tensor in weights.tensors.values():
        name = quantization_name(tensor.type_id)
        summary = formats[(tensor.type_id, name)]
        summary['tensors'] += 1
        summary['elements'] += math.prod(tensor.shape)
        summary['storage_bytes'] += tensor.nbytes
        if tensor.type_id not in supported:
            unsupported.add(tensor.type_id)
        if not tensor.size_known:
            unknown_sizes.append(tensor.name)
        if include_tensors:
            tensor_rows.append({**dataclasses.asdict(tensor), 'type_name': name,
                                'decode_supported': tensor.type_id in supported})
    report = {
        'schema_version': 1,
        'filename': weights.path.name,
        'file_bytes': weights.file_size,
        'gguf_version': weights.version,
        'alignment_bytes': weights.alignment,
        **weights.architecture_metadata(),
        'formats': [{'type_id': kind, 'name': name, **value,
                     'decode_supported': kind in supported}
                    for (kind, name), value in sorted(formats.items())],
        'all_tensor_types_decodable': not unsupported,
        'unsupported_tensor_types': [{'type_id': kind, 'name': quantization_name(kind)}
                                     for kind in sorted(unsupported)],
        'unknown_size_tensors': unknown_sizes,
        'unknown_size_interpretation': 'Unknown-kind storage bytes describe the extent to the next tensor offset, including possible padding; element/block sizes are not inferred.',
        'tokenizer_ids': {key: value for key, value in weights.fields.items()
                          if key in ('tokenizer.ggml.bos_token_id', 'tokenizer.ggml.eos_token_id',
                                     'tokenizer.ggml.padding_token_id', 'tokenizer.ggml.add_bos_token',
                                     'tokenizer.ggml.add_eos_token')},
        'inspection_scope': 'Header, architecture fields and tensor descriptors only. No model weight hashing, tensor decoding, or inference.',
        'architecture_execution_support': 'Decoding a tensor format does not establish a compatible full-model graph or NPU execution.',
    }
    if include_tensors:
        report['tensors'] = tensor_rows
    return report


def inspect_gguf(path, include_tensors=False):
    with GGUFWeights(path) as weights:
        return inspect_weights(weights, include_tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--tensors', action='store_true', help='Include each tensor shape, offset and storage size')
    args = parser.parse_args()
    try:
        report = inspect_gguf(args.model, args.tensors)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'GGUF inspection failed: {exc}\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
