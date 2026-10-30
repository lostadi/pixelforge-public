"""Local installation integrity helpers; profiles contain metadata, never credentials."""
import hashlib
import json
import os
from pathlib import Path
import stat
import re


def tensor_selection(value):
    if value is None:
        return None
    if (not isinstance(value, list) or not 1 <= len(value) <= 256
            or any(not isinstance(name, str) or len(name) > 63 or
                   not re.fullmatch(r'[A-Za-z0-9_.-]+\.weight', name) for name in value)
            or len(set(value)) != len(value)):
        raise ValueError('NPU selection must contain distinct bounded weight names')
    return list(value)


def cpu_affinity_selection(value):
    if value is None:
        return None
    if (not isinstance(value, list) or not value or len(value) > 256
            or any(type(cpu) is not int or not 0 <= cpu <= 255 for cpu in value)
            or len(set(value)) != len(value)):
        raise ValueError('cpu_affinity must be a nonempty list of distinct CPU indices from 0 to 255')
    return sorted(value)


def runtime_options(profile):
    """Normalize only trusted root configuration; requests cannot change it."""
    version = profile.get('phase_evidence_version', 0)
    if type(version) is not int or version not in (0, 1):
        raise ValueError('Unsupported phase evidence version')
    options = dict(weight_dtype=profile.get('weight_dtype', 'int8'),
                   cpu_prefill=profile.get('cpu_prefill', True),
                   batch_vectors=profile.get('batch_vectors', 1),
                   decode_batch_vectors=profile.get('decode_batch_vectors', 1),
                   prefill_batch_size=profile.get('prefill_batch_size', 512),
                   native_runtime_cache_mib=profile.get('native_runtime_cache_mib', 128),
                   fused_gate_up=profile.get('fused_gate_up', False),
                   fused_bank_rows=profile.get('fused_bank_rows', profile.get('bank_rows', 4096)),
                   fused_prefill_only=profile.get('fused_prefill_only', False),
                   fused_result_cache_mib=profile.get('fused_result_cache_mib', 64),
                   full_ffn=profile.get('full_ffn', False), full_ffn_phases=profile.get('full_ffn_phases', 'P'),
                   full_ffn_hidden_chunk=profile.get('full_ffn_hidden_chunk', 2048),
                   dynamic_weight_mode=profile.get('dynamic_weight_mode', 'off'),
                   flash_attention=profile.get('flash_attention', 'off'),
                   weight_buffer_cache_mib=profile.get('weight_buffer_cache_mib', 0),
                   native_max_resident_models=profile.get('native_max_resident_models', 2),
                   npu_performance_mode=profile.get('npu_performance_mode', 'default'),
                   cpu_affinity=cpu_affinity_selection(profile.get('cpu_affinity')),
                   prefix_cache_mib=profile.get('prefix_cache_mib', 0))
    if options['npu_performance_mode'] not in ('default', 'balanced', 'high', 'sustained'):
        raise ValueError('npu_performance_mode must be default, balanced, high, or sustained')
    if options['flash_attention'] not in ('off','auto','on'):
        raise ValueError('flash_attention must be off, auto, or on')
    if options['weight_dtype'] not in ('int8', 'float32'):
        raise ValueError('weight_dtype must be int8 or float32')
    for key in ('cpu_prefill', 'fused_gate_up', 'fused_prefill_only', 'full_ffn'):
        if type(options[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    for key, maximum, minimum in (('batch_vectors', 512, 1), ('prefill_batch_size', 512, 1),
                                  ('native_runtime_cache_mib', 2048, 0),
                                  ('weight_buffer_cache_mib', 6144, 0),
                                  ('native_max_resident_models', 64, 1),
                                  ('prefix_cache_mib', 512, 0),
                                  ('fused_bank_rows', 1048576, 1), ('fused_result_cache_mib', 256, 0),
                                  ('full_ffn_hidden_chunk', 1048576, 1)):
        if type(options[key]) is not int or not minimum <= options[key] <= maximum:
            raise ValueError(key + ' is outside its supported bound')
    if (type(options['decode_batch_vectors']) is not int or
            options['decode_batch_vectors'] not in (1, options['batch_vectors'])):
        raise ValueError('decode_batch_vectors must be 1 or batch_vectors')
    if (options['batch_vectors'] > 1 or options['fused_gate_up'] or options['full_ffn']) and options['weight_dtype'] != 'float32':
        raise ValueError('Batched/fused NPU banks require float32 constants')
    options['selected_npu_weight_tensors'] = tensor_selection(profile.get('selected_npu_weight_tensors'))
    prefill = profile.get('selected_prefill_npu_weight_tensors')
    options['selected_prefill_npu_weight_tensors'] = tensor_selection(
        prefill if prefill is not None else options['selected_npu_weight_tensors'])
    options['fused_gate_up_tensors'] = tensor_selection(profile.get('fused_gate_up_tensors'))
    full_names = profile.get('full_ffn_tensors')
    options['full_ffn_tensors'] = [] if full_names is None or full_names == [] else tensor_selection(full_names)
    if options['full_ffn_phases'] not in ('P', 'D', 'PD'):
        raise ValueError('full_ffn_phases must be P, D, or PD')
    if options['full_ffn']:
        if options['fused_gate_up'] or not options['full_ffn_tensors'] or version != 1:
            raise ValueError('Full FFN requires explicit gate weights and excludes gate/up-only fusion')
        if any(not name.endswith('.ffn_gate.weight') for name in options['full_ffn_tensors']):
            raise ValueError('Full FFN must name canonical gate weights')
        if 'P' in options['full_ffn_phases'] and options['cpu_prefill']:
            raise ValueError('Full FFN prompt placement requires NPU prefill')
        for phase, key in (('P', 'selected_prefill_npu_weight_tensors'), ('D', 'selected_npu_weight_tensors')):
            if phase not in options['full_ffn_phases']:
                continue
            selected = options[key]
            for gate in options['full_ffn_tensors']:
                prefix = gate[:-len('ffn_gate.weight')]
                if selected is None or any(prefix + suffix not in selected for suffix in ('ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight')):
                    raise ValueError('Full FFN requires explicit gate, up and down NPU placements for each enabled phase')
    elif options['full_ffn_tensors']:
        raise ValueError('Full FFN gate weights require full_ffn=true')
    if options['dynamic_weight_mode'] not in ('off','float32','original_groups','original_columns'):
        raise ValueError('dynamic_weight_mode must be off, float32, original_groups, or original_columns; packed is unvalidated')
    if options['dynamic_weight_mode'] != 'off':
        if (version != 1 or options['weight_dtype'] != 'float32' or options['fused_gate_up'] or options['full_ffn']
                or options['dynamic_weight_mode'] == 'float32' and options['decode_batch_vectors'] != options['batch_vectors']
                or options['native_runtime_cache_mib'] == 0):
            raise ValueError('Dynamic weights require phase evidence, float32 math, one shared batch size, positive device cache and no constant fusion recipe')
    elif options['weight_buffer_cache_mib']:
        raise ValueError('Weight buffer caching requires a dynamic weight input mode')
    return options


def execution_description(options):
    def placement(selection, phase):
        text = 'all planned dense-weight projections on NPU' if selection is None else 'selected dense-weight projections on NPU; remaining dense projections on CPU'
        if options['full_ffn'] and phase in options['full_ffn_phases']:
            text += '; selected feed-forward compute on NPU, partial sums on CPU'
        return text
    prompt = 'CPU prompt processing' if options['cpu_prefill'] else 'Prompt processing: ' + placement(options['selected_prefill_npu_weight_tensors'], 'P')
    text = prompt + '; decoding: ' + placement(options['selected_npu_weight_tensors'], 'D') + '; attention, norms and sampling on CPU'
    if options['dynamic_weight_mode'] == 'float32':
        text += '; original GGUF weight decoding on CPU'
    elif options['dynamic_weight_mode'] == 'original_groups':
        text += '; original GGUF groups unpacked losslessly on CPU, weight reconstruction on NPU'
    elif options['dynamic_weight_mode'] == 'original_columns':
        text += '; original GGUF groups unpacked losslessly on CPU, weight reconstruction and column products on NPU; column partial sums on CPU; original GGUF unchanged; no additional weight quantization'
    return text


def preflight_artifacts(profile, weights, plan, *, cache_factory=None, program_factory=None, dynamic_factory=None, grouped_factory=None, column_factory=None):
    """Verify every routed P/D bank, without compilation or hardware calls.

    Return hashes for all required sidecar manifests. Their content-addressed
    metadata and DGC blobs are verified by ArtifactCache.get(load_code=False).
    """
    from compiled_model import ArtifactCache, batch_directory
    options = runtime_options(profile)
    factory = ArtifactCache if cache_factory is None else cache_factory
    planned = {item['name']: item for item in plan['dense_tensors']}
    for key in ('selected_npu_weight_tensors', 'selected_prefill_npu_weight_tensors'):
        selection = options[key]
        if selection is not None and any(name not in planned for name in selection):
            raise ValueError('NPU placement names a weight outside the architecture plan')
    base = Path(profile['artifacts'])
    if base.is_symlink():
        raise ValueError('Artifact directory must not be a symlink')
    base = base.resolve()
    pins = {}
    if options['dynamic_weight_mode'] != 'off':
        columns = options['dynamic_weight_mode'] == 'original_columns'
        grouped = options['dynamic_weight_mode'] in ('original_groups','original_columns')
        if columns:
            from grouped_columns import ColumnProgramCache
            factory_dynamic = ColumnProgramCache if column_factory is None else column_factory
        elif grouped:
            from grouped_fc import GroupedProgramCache
            factory_dynamic = GroupedProgramCache if grouped_factory is None else grouped_factory
        else:
            from dynamic_fc import DynamicProgramCache
            factory_dynamic = DynamicProgramCache if dynamic_factory is None else dynamic_factory
        required = {}
        phases = [(options['selected_npu_weight_tensors'],options['decode_batch_vectors'])]
        if not options['cpu_prefill']:
            phases.append((options['selected_prefill_npu_weight_tensors'],options['batch_vectors']))
            if grouped and options['batch_vectors'] > 1 and options['decode_batch_vectors'] == 1:
                # A single prompt vector uses the B1 NPU recipe while retaining
                # phase P; every prompt-only weight must be available there too.
                phases.append((options['selected_prefill_npu_weight_tensors'],1))
        for selection,vectors in phases:
            required.setdefault(vectors,set()).update(planned if selection is None else selection)
        for vectors,names in sorted(required.items()):
            family = 'grouped-columns-v1' if columns else ('grouped-fc-v1' if grouped else 'dynamic-fc-v1')
            directory = base / family / ('b' + str(vectors))
            cache = factory_dynamic(directory, profile['model_path'], source_sha256=profile['model']['model_sha256'],
                source_identity=weights.source_identity, verify_source=False, read_only=True,
                plan=plan, batch_vectors=vectors)
            for name in sorted(names):
                item = planned[name]
                if grouped and (item.get('type_id') not in (13,14) or item['columns'] % 256):
                    raise ValueError('Original-group NPU placements require whole original Q5_K/Q6_K blocks: ' + name)
                aliases = cache.aliases_for(name) if columns else [name]
                if not isinstance(aliases,list) or not aliases or any(not isinstance(alias,str) for alias in aliases) or len(set(aliases)) != len(aliases):
                    raise ValueError('Column geometry aliases do not cover the selected original weight')
                for alias in aliases:
                    layer = cache.layer(alias) if columns else item
                    ranges = (cache.ranges(alias) if columns else
                              ((start,min(start+item['bank_rows'],item['rows'])) for start in range(0,item['rows'],item['bank_rows'])))
                    for start,stop in ranges:
                        missing = 'Required dynamic-weight NPU geometry is missing (b' + str(vectors) + '): ' + alias
                        try:
                            dtype = cache.request_dtype if columns else ('original_groups' if grouped else 'float32')
                            found = cache.get(alias,start,stop,dtype,layer['columns'],load_code=False)
                        except FileNotFoundError as error:
                            raise FileNotFoundError(missing) from error
                        if found is None:
                            raise FileNotFoundError(missing)
                        record = found[0]
                        if type(record.get('compiled_bytes')) is not int or record['compiled_bytes'] < 1:
                            raise ValueError('Dynamic geometry lacks its bounded compiled size')
                        sizes = (list(cache.input_sizes(alias,start,stop)) if grouped else
                                 [vectors*layer['columns']*4,(stop-start)*layer['columns']*4])
                        if not sizes or any(type(size) is not int or not 1 <= size <= 1073741824 for size in sizes):
                            raise ValueError('Invalid bounded dynamic weight input port sizes')
                        sizes.append(vectors*(stop-start)*4)
                        charged = record['compiled_bytes'] + sum((size+4095)//4096*4096 for size in sizes)
                        if charged > options['native_runtime_cache_mib']*1024**2:
                            raise ValueError('Dynamic weight bank exceeds the configured device cache budget')
            path = directory / 'MODEL.json'
            if not path.resolve().is_relative_to(base):
                raise ValueError('Dynamic-weight sidecar escapes the supplied package')
            pins[str(path.relative_to(base))]=sha256(path)
        return pins
    full_members = set()
    if options['full_ffn']:
        if plan.get('architecture') != 'qwen35':
            raise ValueError('Full SwiGLU NPU placement currently requires the verified qwen35 architecture')
        from full_ffn import FullFFNProgramCache, build_full_ffn_plan
        full_plan = build_full_ffn_plan(weights, profile['model']['model_sha256'], options['full_ffn_hidden_chunk'])
        layers = {item['gate']: item for item in full_plan['layers']}
        if any(gate not in layers for gate in options['full_ffn_tensors']):
            raise ValueError('Full FFN selection does not match compatible gate/up/down source shapes')
        directory = base / 'full-ffn-v1' / ('b' + str(options['batch_vectors']))
        factory_full = FullFFNProgramCache if program_factory is None else program_factory
        cache = factory_full(directory, profile['model_path'], source_sha256=profile['model']['model_sha256'],
            source_identity=weights.source_identity, verify_source=False, read_only=True,
            plan=full_plan, batch_vectors=options['batch_vectors'])
        for gate in options['full_ffn_tensors']:
            layer = cache.layer(gate)
            full_members.update((layer['gate'], layer['up'], layer['down']))
            for start, stop in cache.ranges(gate):
                if cache.get(gate, start, stop, 'float32', layer['hidden'], load_code=False) is None:
                    raise FileNotFoundError('Required full FFN NPU program is missing: ' + gate)
        path = directory / 'MODEL.json'
        if not path.resolve().is_relative_to(base):
            raise ValueError('Full FFN sidecar escapes the supplied package')
        pins[str(path.relative_to(base))] = sha256(path)
    fused_weights = None
    fused_members = {}
    if options['fused_gate_up']:
        from fused_ffn import FusedFFNWeights
        fused_weights = FusedFFNWeights(weights, plan, bank_rows=options['fused_bank_rows'],
                                        selected=options['fused_gate_up_tensors'])
        fused_members = {member: virtual for virtual, members in fused_weights.fused_sources.items()
                         if virtual in fused_weights.runtime_fused_names for member in members}
    packages = {}
    phases = [('D', options['selected_npu_weight_tensors'], options['decode_batch_vectors'])]
    if not options['cpu_prefill']:
        phases.append(('P', options['selected_prefill_npu_weight_tensors'], options['batch_vectors']))
    for phase, selection, vectors in phases:
        for name in planned if selection is None else selection:
            if options['full_ffn'] and phase in options['full_ffn_phases'] and name in full_members:
                continue
            use_fused = fused_weights is not None and (phase == 'P' or not options['fused_prefill_only'])
            routed = fused_members.get(name, name) if use_fused else name
            use_fused = use_fused and routed in fused_weights.runtime_fused_names
            package_plan = fused_weights.plan if use_fused else plan
            directory = base / 'fused-ffn-v1' if use_fused else base
            directory = batch_directory(directory, vectors)
            key = (str(directory), vectors, bool(use_fused))
            package = packages.setdefault(key, [package_plan, set(), False])
            package[1].add(routed)
            package[2] |= selection is None
    for (directory, vectors, fused), (package_plan, names, all_selected) in sorted(packages.items()):
        cache = factory(directory, profile['model_path'], source_sha256=profile['model']['model_sha256'],
                        source_identity=weights.source_identity, verify_source=False, read_only=True,
                        plan=package_plan, batch_vectors=vectors, fused_gate_up=fused)
        by_name = {item['name']: item for item in package_plan['dense_tensors']}
        for name in sorted(names):
            item = by_name[name]
            for begin in range(0, item['rows'], item['bank_rows']):
                record = cache.get(name, begin, min(begin + item['bank_rows'], item['rows']),
                                   options['weight_dtype'], item['columns'], load_code=False)
                if record is None:
                    raise FileNotFoundError('Required compiled NPU bank is missing: ' + name)
        if all_selected and names == set(by_name) and not cache._manifest().get('complete'):
            raise ValueError('All-dense placement requires a complete compiled artifact set')
        path = Path(directory) / 'MODEL.json'
        if not path.resolve().is_relative_to(base):
            raise ValueError('Artifact sidecar escapes the supplied package')
        pins[str(path.relative_to(base))] = sha256(path)
    return pins


def validate_flash_evidence(frame, options, *, ready=False):
    mode = options['flash_attention']
    if mode == 'off' and not {'flash_attention','cpu_flash_attention_enabled','attention_backend'} & frame.keys():
        return # Compatible with pinned older runners that always disabled it.
    if frame.get('flash_attention') != mode or frame.get('attention_backend') != 'CPU':
        raise ValueError('Native response changed the configured CPU attention setting')
    enabled = frame.get('cpu_flash_attention_enabled')
    if ready:
        if enabled is not None:
            raise ValueError('Native readiness claimed resolved attention before context creation')
    elif (type(enabled) is not bool or mode == 'off' and enabled or mode == 'on' and not enabled):
        raise ValueError('Native response lacks matching resolved CPU flash-attention evidence')


def validate_prefix_evidence(frame, profile, *, ready=False):
    """Verify bounded exact-prefix state accounting, including disabled mode."""
    budget = runtime_options(profile)['prefix_cache_mib']
    metrics = ('prefix_cache_hit','prefix_tokens_reused','prefix_cache_bytes',
               'prefix_cache_entries','prefill_tokens_processed')
    strict = budget > 0 or profile.get('phase_evidence_version') == 1
    if not strict and not ({'prefix_cache_mib',*metrics} & frame.keys()):
        return {}
    if type(frame.get('prefix_cache_mib')) is not int or frame['prefix_cache_mib'] != budget:
        raise ValueError('Prefix cache budget does not match the trusted profile')
    if ready:
        return {'prefix_cache_mib':budget}
    prompt = frame.get('prompt_tokens')
    hit = frame.get('prefix_cache_hit')
    if (type(prompt) is not int or prompt < 1 or type(hit) is not bool
            or any(type(frame.get(key)) is not int or frame[key] < 0 for key in metrics[1:])):
        raise ValueError('Invalid prefix cache execution metrics')
    reused = frame['prefix_tokens_reused']
    if (reused >= prompt or hit != (reused > 0) or frame['prefix_cache_bytes'] > budget*1024**2
            or frame['prefill_tokens_processed'] != prompt-reused
            or budget == 0 and (hit or reused or frame['prefix_cache_bytes'] or frame['prefix_cache_entries'])):
        raise ValueError('Prefix cache metrics contradict the trusted budget or prompt')
    return {'prefix_cache_mib':budget,**{key:frame[key] for key in metrics}}


def validate_phase_evidence(frame, profile):
    """Reject completed responses that contradict the configured placement."""
    options = runtime_options(profile)
    validate_prefix_evidence(frame, profile)
    if cpu_affinity_selection(frame.get('cpu_affinity')) != options['cpu_affinity']:
        raise ValueError('Completion changed the verified CPU affinity')
    validate_flash_evidence(frame, options)
    if (options['dynamic_weight_mode'] != 'off' or 'dynamic_weight_mode' in frame) and frame.get('dynamic_weight_mode') != options['dynamic_weight_mode']:
        raise ValueError('Native completion changed the configured weight input mode')
    if frame.get('prefill_cpu') is not options['cpu_prefill']:
        raise ValueError('Native completion changed the configured prefill mode')
    if tensor_selection(frame.get('selected_npu_weight_tensors')) != options['selected_npu_weight_tensors']:
        raise ValueError('Native completion changed the decode NPU placement')
    strict = profile.get('phase_evidence_version') == 1 or not options['cpu_prefill']
    prefill_key = 'selected_prefill_npu_weight_tensors'
    if strict or prefill_key in frame:
        if prefill_key not in frame or tensor_selection(frame[prefill_key]) != options[prefill_key]:
            raise ValueError('Native completion changed the prefill NPU placement')
    total = frame.get('npu_weight_matmul_operations')
    if type(total) is not int or total <= 0:
        raise ValueError('No NPU weight operations were verified; response rejected')
    keys = ('npu_prefill_weight_matmul_operations', 'npu_decode_weight_matmul_operations')
    if strict or any(key in frame for key in keys):
        prefill, decode = (frame.get(key) for key in keys)
        if (type(prefill) is not int or type(decode) is not int or min(prefill, decode) < 0
                or prefill + decode != total or (prefill != 0 if options['cpu_prefill'] else prefill <= 0)):
            raise ValueError('Native completion lacks matching prefill/decode NPU counters')
        if isinstance(frame.get('generated_tokens'), list) and len(frame['generated_tokens']) > 1 and decode == 0:
            raise ValueError('Native completion lacks NPU decode operations')
        if frame.get('prefill_batch_size') != options['prefill_batch_size']:
            raise ValueError('Native completion changed the configured prefill batch size')
    if 'cpu_prefill_weight_matmul_operations' in frame:
        value = frame['cpu_prefill_weight_matmul_operations']
        if type(value) is not int or value < 0 or not options['cpu_prefill'] and value != 0:
            raise ValueError('Native completion reported contradictory CPU-prefill operations')
    if 'cpu_placed_weight_matmul_operations' in frame:
        value = frame['cpu_placed_weight_matmul_operations']
        all_npu = options['selected_npu_weight_tensors'] is None and (options['cpu_prefill'] or options['selected_prefill_npu_weight_tensors'] is None)
        if type(value) is not int or value < 0 or all_npu and value != 0:
            raise ValueError('Native completion reported contradictory CPU weight placement')
    if options['full_ffn'] or 'npu_fused_ffn_operations' in frame:
        fields = ('npu_fused_ffn_operations', 'npu_prefill_fused_ffn_operations', 'npu_decode_fused_ffn_operations')
        values = [frame.get(key) for key in fields]
        if (any(type(value) is not int or value < 0 for value in values)
                or sum(values[1:]) != values[0] or values[0] * 3 > total
                or values[1] * 3 > frame.get('npu_prefill_weight_matmul_operations', 0)
                or values[2] * 3 > frame.get('npu_decode_weight_matmul_operations', 0)
                or frame.get('full_ffn_tensors') != options['full_ffn_tensors']
                or frame.get('full_ffn_phases') != options['full_ffn_phases']):
            raise ValueError('Native completion lacks matching full FFN execution evidence')
        if options['full_ffn']:
            if values[0] <= 0:
                raise ValueError('Configured NPU feed-forward hook was not executed')
            for phase, count in zip(('P','D'), values[1:]):
                required = phase in options['full_ffn_phases'] and (phase == 'P' or len(frame.get('generated_tokens', [])) > 1)
                if (count <= 0 if required else count != 0):
                    raise ValueError('Native completion changed full FFN execution phases')
        elif values[0] != 0:
            raise ValueError('Unexpected full FFN execution outside the configured profile')


def sha256(path):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError('Pinned input must be a regular file')
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def root_json(path, *, owner=0, limit=262144):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != owner or info.st_mode & 0o022:
            raise ValueError('Configuration must be an owned, non-writable regular file')
        content = source.read(limit + 1)
    if len(content) > limit:
        raise ValueError('Configuration exceeds its size limit')
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError('Configuration must be a JSON object')
    return result
