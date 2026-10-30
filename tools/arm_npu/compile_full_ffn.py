"""Compile/resume selected full SwiGLU programs; never invoke or copy GGUFs."""
import argparse
import json
from pathlib import Path
import re
import shutil

from compile_gguf import DEFAULT_NATIVE_COMMAND
from full_ffn import FullFFNProgramCache, build_full_ffn_plan, compile_full_ffn_chunk
from gguf_hf_loader import GGUFWeights
from npu_linear import NPULinear


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Base directory; writes full-ffn-v1/bB sidecar')
    parser.add_argument('--batch-vectors', type=int, default=16)
    parser.add_argument('--hidden-chunk', type=int, default=2048)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--tensor', help='Canonical blk.N.ffn_gate.weight name')
    selection.add_argument('--tensor-pattern', help='Full-match regex over canonical gate names')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--native-command-json')
    args = parser.parse_args(argv)
    if not 1 <= args.batch_vectors <= 512 or args.hidden_chunk < 1:
        parser.error('Batch must be 1..512 and hidden chunk positive')
    try:
        pattern = re.compile(args.tensor_pattern) if args.tensor_pattern is not None else None
    except re.error:
        parser.error('Invalid tensor regular expression')
    output = args.output/'full-ffn-v1'/f'b{args.batch_vectors}'
    with GGUFWeights(args.model) as weights:
        digest = weights.sha256()
        plan = build_full_ffn_plan(weights, digest, args.hidden_chunk)
        layers = [x for x in plan['layers'] if (args.tensor is None or x['gate'] == args.tensor)
                  and (pattern is None or pattern.fullmatch(x['gate']))]
        if not layers:
            raise ValueError('No full FFN layers match the requested selection')
        options = dict(source_sha256=digest, source_identity=weights.source_identity,
                       verify_source=False, plan=plan, batch_vectors=args.batch_vectors)
        cache = None
        if (output/'MODEL.json').exists():
            if not args.resume:
                raise FileExistsError('Full FFN package exists; use --resume')
            cache = FullFFNProgramCache(output, args.model, read_only=args.plan_only, **options)
        chunks = []
        required = 128 * 1024**2
        for layer in layers:
            for start in range(0, layer['intermediate'], args.hidden_chunk):
                stop = min(start+args.hidden_chunk, layer['intermediate'])
                if cache is not None:
                    key, _ = cache.identity(layer['gate'], start, stop, 'float32', layer['hidden'])
                    if key in cache._manifest()['artifacts']:
                        cache.get(layer['gate'], start, stop, 'float32', layer['hidden'], load_code=False)
                        continue
                chunks.append((layer, start, stop))
                required += 3*layer['hidden']*(stop-start)*4 + 2*1024**2
        location = output.resolve()
        while not location.exists():
            location = location.parent
        available = shutil.disk_usage(location).free
        print(json.dumps(dict(event='plan', output=str(output), selected_layers=len(layers),
            missing_chunks=len(chunks), batch_vectors=args.batch_vectors, hidden_chunk=args.hidden_chunk,
            source_gguf_sha256=digest, required_bytes_estimate=required, available_bytes=available,
            would_fit=available >= required, precision='original decoded float32 constants; vendor arithmetic must be validated')),
            flush=True)
        if args.plan_only:
            return 0
        if available < required:
            raise OSError('Insufficient storage for the selected full FFN programs')
        cache = cache or FullFFNProgramCache(output, args.model, **options)
        command = json.loads(args.native_command_json) if args.native_command_json else DEFAULT_NATIVE_COMMAND
        native = NPULinear(weights, command=command, weight_dtype='float32',
                           batch_vectors=args.batch_vectors, cache_mib=1, native_runtime_cache_mib=0)
        try:
            for layer, start, stop in chunks:
                record = compile_full_ffn_chunk(native, weights, cache, layer['gate'], start, stop)
                print(json.dumps(dict(event='full_ffn_ready', tensor=layer['gate'], hidden_start=start,
                    hidden_stop=stop, compiled_bytes=record['compiled_bytes'],
                    bytecode_sha256=record['bytecode_sha256'])), flush=True)
        finally:
            native.close()
        print(json.dumps(dict(event='complete', selected_layers=len(layers), compiled_chunks=len(chunks),
                              npu_invocations=native.invocations, output=str(output))), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
