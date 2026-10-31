"""Compile stable original-group BMM geometry for canonical column tiles."""
import argparse
import json
from pathlib import Path
import re
import shutil

from compile_gguf import DEFAULT_NATIVE_COMMAND, selected_banks
from gguf_hf_loader import GGUFWeights
from grouped_columns import ColumnLayout, ColumnProgramCache
from grouped_fc import compile_grouped_geometry
from model_architecture import build_plan
from npu_linear import NPULinear
from voice_handoff import add_voice_arguments, validate_voice_arguments, register_compiled


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bank-rows',type=int,required=True)
    parser.add_argument('--batch-vectors',type=int,default=64)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--tensor')
    selection.add_argument('--tensor-pattern')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--native-command-json')
    add_voice_arguments(parser)
    args=parser.parse_args(argv)
    validate_voice_arguments(parser,args)
    if args.bank_rows<1 or not 1<=args.batch_vectors<=512:
        parser.error('Positive bank rows and batch1..512 required')
    try:
        if args.tensor_pattern is not None:re.compile(args.tensor_pattern)
    except re.error:parser.error('Invalid tensor pattern')
    directory=args.output/'grouped-columns-v1'/f'b{args.batch_vectors}'
    with GGUFWeights(args.model) as weights:
        digest=weights.sha256() # Exactly one original GGUF hash for this compile session.
        plan=build_plan(weights,model_sha256=digest,bank_rows=args.bank_rows)
        layout=ColumnLayout(plan)
        original_banks=[bank for bank in selected_banks(weights,plan,args.tensor,tensor_pattern=args.tensor_pattern)
                        if weights.tensors[bank[0]].type_id in (13,14)]
        if not original_banks:raise ValueError('No original Q5_K/Q6_K matrices match selection')
        banks=[(alias,start,stop,layout.layer(alias)['columns']) for name,start,stop,_ in original_banks
               for alias in layout.aliases_for(name)]
        options=dict(source_sha256=digest,source_identity=weights.source_identity,verify_source=False,
                     plan=plan,batch_vectors=args.batch_vectors)
        cache=None
        if (directory/'MODEL.json').exists():
            if not args.resume:raise FileExistsError('Column package exists; use --resume')
            cache=ColumnProgramCache(directory,args.model,read_only=args.plan_only,**options)
        unique={}
        for alias,start,stop,columns in banks:
            unique.setdefault((layout.layer(alias)['type_id'],stop-start,columns),(alias,start,stop,columns))
        pending=[]
        for alias,start,stop,columns in unique.values():
            if cache is not None:
                key,_=cache.identity(alias,start,stop,cache.request_dtype,columns)
                if key in cache._manifest()['artifacts']:
                    cache.get(alias,start,stop,cache.request_dtype,columns,load_code=False)
                    continue
            pending.append((alias,start,stop,columns))
        location=directory.resolve()
        while not location.exists():location=location.parent
        required=128*1024**2+len(pending)*4*1024**2
        available=shutil.disk_usage(location).free
        print(json.dumps(dict(event='plan',source_gguf_sha256=digest,dynamic_weight_mode='original_columns',
            logical_original_banks=len(original_banks),column_partial_banks=len(banks),
            geometry_programs=len(unique),missing_geometries=len(pending),column_bank_maximum=3072,
            extra_weight_quantization=False,output=str(directory),required_bytes_estimate=required,
            available_bytes=available,would_fit=available>=required,
            limitation='Every row/column/batch geometry requires separate device numerical validation')),flush=True)
        if args.plan_only:return 0
        if available<required:raise OSError('Insufficient space for column geometry package')
        cache=cache or ColumnProgramCache(directory,args.model,**options)
        command=json.loads(args.native_command_json) if args.native_command_json else DEFAULT_NATIVE_COMMAND
        invocations=0
        if pending:
            native=NPULinear(weights,command=command,weight_dtype='float32',batch_vectors=args.batch_vectors,
                             min_free_mib=128,cache_mib=1,native_runtime_cache_mib=0)
            try:
                for alias,start,stop,columns in pending:
                    record=compile_grouped_geometry(native,cache,alias,start,stop)
                    print(json.dumps(dict(event='geometry_ready',rows=stop-start,columns=columns,
                        compiled_bytes=record['compiled_bytes'],bytecode_sha256=record['bytecode_sha256'])),flush=True)
                invocations=native.invocations
            finally:native.close()
        if args.tensor is None and args.tensor_pattern is None:cache.mark_complete()
        print(json.dumps(dict(event='complete',compiled_geometries=len(pending),npu_invocations=invocations)),flush=True)
    return register_compiled(args, package=directory, model=args.model,
                             selected_tensors=sorted({name for name, *_ in original_banks}),
                             weight_dtype="float32", batch_vectors=args.batch_vectors,
                             dynamic_weight_mode="original_columns")


if __name__=='__main__':raise SystemExit(main())
