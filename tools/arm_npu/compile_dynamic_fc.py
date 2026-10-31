"""Compile unique geometry-only BMM programs; no weight decoding or inference."""
import argparse
import json
from pathlib import Path
import re
import shutil

from compile_gguf import DEFAULT_NATIVE_COMMAND, selected_banks
from dynamic_fc import DynamicProgramCache, compile_dynamic_geometry
from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan
from npu_linear import NPULinear


def main(argv=None,*,original_groups=False):
    Cache,compile_program=DynamicProgramCache,compile_dynamic_geometry
    directory,dtype='dynamic-fc-v1','float32'
    if original_groups:
        from grouped_fc import GroupedProgramCache,compile_grouped_geometry
        Cache,compile_program=GroupedProgramCache,compile_grouped_geometry
        directory,dtype='grouped-fc-v1','original_groups'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bank-rows',type=int,required=True,help='Use a device-validated geometry limit')
    p.add_argument('--batch-vectors',type=int,default=16)
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--tensor')
    selection.add_argument('--tensor-pattern')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--native-command-json')
    a=p.parse_args(argv)
    if a.bank_rows<1 or not 1<=a.batch_vectors<=512:p.error('Positive bank rows and batch1..512 required')
    try:
        if a.tensor_pattern is not None:re.compile(a.tensor_pattern)
    except re.error:p.error('Invalid tensor pattern')
    output=a.output/directory/f'b{a.batch_vectors}'
    with GGUFWeights(a.model) as weights:
        digest=weights.sha256()
        plan=build_plan(weights,model_sha256=digest,bank_rows=a.bank_rows)
        banks=selected_banks(weights,plan,a.tensor,tensor_pattern=a.tensor_pattern)
        if original_groups:
            banks=[bank for bank in banks if weights.tensors[bank[0]].type_id in (13,14)]
            if not banks:raise ValueError('No original Q5_K/Q6_K weights match selection')
        options=dict(source_sha256=digest,source_identity=weights.source_identity,
                     verify_source=False,plan=plan,batch_vectors=a.batch_vectors)
        cache=None
        if (output/'MODEL.json').exists():
            if not a.resume:raise FileExistsError('Dynamic package exists; use --resume')
            cache=Cache(output,a.model,read_only=a.plan_only,**options)
        unique={}
        for name,start,stop,columns in banks:
            geometry=(weights.tensors[name].type_id if original_groups else None,stop-start,columns)
            unique.setdefault(geometry,(name,start,stop))
        pending=[]
        for name,start,stop in unique.values():
            if cache is not None:
                key,_=cache.identity(name,start,stop,dtype,cache.layer(name)['columns'])
                if key in cache._manifest()['artifacts']:
                    cache.get(name,start,stop,dtype,cache.layer(name)['columns'],load_code=False)
                    continue
            pending.append((name,start,stop))
        location=output.resolve()
        while not location.exists():location=location.parent
        required=128*1024**2+len(pending)*4*1024**2
        available=shutil.disk_usage(location).free
        print(json.dumps(dict(event='plan',source_gguf_sha256=digest,logical_banks=len(banks),
            dynamic_weight_mode=dtype,extra_weight_quantization=False,
            geometry_programs=len(unique),missing_geometries=len(pending),output=str(output),
            required_bytes_estimate=required,available_bytes=available,would_fit=available>=required,
            limitation='Every geometry needs device validation; small-BMM success does not validate larger banks')),
            flush=True)
        if a.plan_only:return 0
        if available<required:raise OSError('Insufficient space for dynamic geometry programs')
        cache=cache or Cache(output,a.model,**options)
        command=json.loads(a.native_command_json) if a.native_command_json else DEFAULT_NATIVE_COMMAND
        invocations=0
        if pending:
            native=NPULinear(weights,command=command,weight_dtype='float32',batch_vectors=a.batch_vectors,
                             min_free_mib=128,cache_mib=1,native_runtime_cache_mib=0)
            try:
                for name,start,stop in pending:
                    r=compile_program(native,cache,name,start,stop)
                    print(json.dumps(dict(event='geometry_ready',rows=stop-start,
                        columns=cache.layer(name)['columns'],compiled_bytes=r['compiled_bytes'],
                        bytecode_sha256=r['bytecode_sha256'])),flush=True)
                invocations=native.invocations
            finally:native.close()
        print(json.dumps(dict(event='complete',compiled_geometries=len(pending),npu_invocations=invocations)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
