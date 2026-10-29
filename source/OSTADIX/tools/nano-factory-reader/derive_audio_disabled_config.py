#!/usr/bin/env python3
"""Derive a config using the recovered runtime's optional NoAudioEncoder branch.

The pinned source and all model weights stay unchanged. Native serializer
0x2df5490 maps GemConfigV2Plus field 6 to object+0x60 / presence bit 6;
0x2dfda88 selects NoAudioEncoder when that field is absent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

EXPECTED = '2579a3a81a1c22403cf4f930581d85e743b9b33e7340ac917508b6419f1ef54e'


def read_varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError('truncated protobuf')
        byte = data[offset]
        offset += 1
        if shift == 63 and byte > 1:
            raise ValueError('oversized varint')
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError('oversized varint')


def encode_varint(value):
    output = bytearray()
    while value > 127:
        output.append((value & 127) | 128)
        value >>= 7
    output.append(value)
    return bytes(output)


def fields(data):
    offset = 0
    while offset < len(data):
        start = offset
        tag, offset = read_varint(data, offset)
        number, wire = tag >> 3, tag & 7
        if number == 0:
            raise ValueError('zero field')
        payload = None
        if wire == 0:
            _, offset = read_varint(data, offset)
        elif wire in (1, 5):
            offset += 8 if wire == 1 else 4
        elif wire == 2:
            length, offset = read_varint(data, offset)
            payload = data[offset:offset + length]
            offset += length
        else:
            raise ValueError('unsupported wire type')
        if offset > len(data):
            raise ValueError('truncated field')
        yield number, wire, data[start:offset], payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED:
        raise ValueError('Source differs from the verified original factory config')
    outer = list(fields(source))
    if len(outer) != 1 or outer[0][:2] != (3, 2):
        raise ValueError('Expected only top-level variant 3')
    inner = list(fields(outer[0][3]))
    removed = [item for item in inner if item[0] == 6]
    if len(removed) != 1 or removed[0][1] != 2:
        raise ValueError('Expected one optional audio submessage')
    retained = b''.join(item[2] for item in inner if item[0] != 6)
    output = encode_varint(3 << 3 | 2) + encode_varint(len(retained)) + retained
    with args.output.open('xb') as stream:
        stream.write(output)
        stream.flush()
        os.fsync(stream.fileno())
    args.output.chmod(0o400)
    print(json.dumps(dict(source=str(args.source), output=str(args.output),
        original_sha256=digest, derived_sha256=hashlib.sha256(output).hexdigest(),
        original_bytes=len(source), derived_bytes=len(output), removed_field='3/6',
        removed_field_bytes=len(removed[0][2]),
        removed_field_sha256=hashlib.sha256(removed[0][2]).hexdigest(),
        all_other_nested_field_bytes_preserved=True, model_weights_modified=False,
        inference_executed=False), indent=2))


if __name__ == '__main__':
    main()
