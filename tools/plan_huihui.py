#!/usr/bin/env python3
"""Build an offline plan for the packaged Google Tensor SDK export route.

Reads config and regular-file metadata only. Never imports model code, downloads
weights, calls a compiler, creates an output directory, or changes deployment.
Its SDK package checks do not describe the separate native ARM compiler path.
"""
import argparse
import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import sys

DOC = 'https://developers.google.com/edge/litert/conversion/pytorch/genai'
EXPORTER_REVISION = '91dd077a9584f28bc82f135016358e9433bdf260'
EXPORTER_SOURCE = ('https://github.com/google-ai-edge/litert-torch/blob/' + EXPORTER_REVISION
                   + '/litert_torch/generative/export_hf/export.py')
SDK_DOC = 'https://developers.google.com/edge/litert/next/tensor-sdk'
MODEL_TYPES = {'gemma3', 'gemma3_text', 'gemma3n', 'gemma4', 'llama', 'mistral',
               'qwen2', 'qwen3', 'smollm3'}
ARCHITECTURES = {'Gemma3ForCausalLM', 'Gemma3ForConditionalGeneration',
    'Gemma3nForCausalLM', 'Gemma3nForConditionalGeneration', 'Gemma4ForCausalLM',
    'Gemma4ForConditionalGeneration', 'LlamaForCausalLM', 'MistralForCausalLM',
    'Qwen2ForCausalLM', 'Qwen3ForCausalLM', 'SmolLM3ForCausalLM'}
PACKAGES = ('litert-torch', 'litert-torch-nightly', 'ai-edge-litert', 'torch', 'transformers', 'safetensors',
            'ai-edge-quantizer', 'ai-edge-litert-nightly', 'ai-edge-quantizer-nightly',
            'ai-edge-litert-sdk-google-tensor-nightly')


def validate_repo(value):
    if not re.fullmatch(r'huihui-ai/[A-Za-z0-9][A-Za-z0-9._-]{0,95}', value) or '..' in value or value.endswith('.'):
        raise argparse.ArgumentTypeError('Use the exact huihui-ai/NAME namespace, not a URL or another publisher')
    return value


def validate_revision(value):
    if not re.fullmatch(r'[a-fA-F0-9]{40}', value):
        raise argparse.ArgumentTypeError('Revision must be an exact 40-character commit SHA')
    return value.lower()


def local_path(value):
    if not value or '\x00' in value or '://' in value or value.startswith('~'):
        raise argparse.ArgumentTypeError('Supply a literal local filesystem path; expand ~ yourself')
    return Path(os.path.abspath(value))


def json_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError('metadata is not a regular file')
        data = source.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024:
        raise ValueError('metadata exceeds 1 MiB')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('metadata must be an object')
    return value


def inspect_checkpoint(path):
    result = {'path': str(path) if path else None, 'weights': [], 'blockers': [],
              'weight_checksums_verified': False, 'repository_revision_verified_against_local_files': False}
    if path is None:
        result['blockers'].append('checkpoint_not_supplied')
        return result
    if path.is_symlink() or not path.is_dir():
        result['blockers'].append('checkpoint_missing_or_not_a_regular_directory')
        return result
    try:
        config = json_regular(path / 'config.json')
    except (OSError, ValueError, json.JSONDecodeError):
        result['blockers'].append('config_json_missing_invalid_or_symlink')
        config = {}
    model_type = config.get('model_type') if isinstance(config.get('model_type'), str) else None
    architectures = config.get('architectures', [])
    result['model_type'] = model_type
    result['architectures'] = architectures if isinstance(architectures, list) else []
    if model_type not in MODEL_TYPES:
        result['blockers'].append('architecture_not_in_documented_exporter_model_types')
    if not isinstance(architectures, list) or any(not isinstance(name, str) or name not in ARCHITECTURES for name in architectures):
        result['blockers'].append('architecture_class_unknown_or_requires_separate_exporter_validation')
    result['remote_code_metadata_present'] = 'auto_map' in config
    if result['remote_code_metadata_present']:
        result['blockers'].append('auto_map_present_remote_model_code_will_not_be_loaded')
    try:
        with os.scandir(path) as entries:
            for count, entry in enumerate(entries):
                if count >= 4096:
                    result['blockers'].append('checkpoint_directory_inventory_limit_exceeded')
                    break
                suffix = Path(entry.name).suffix.lower()
                if suffix not in ('.safetensors', '.bin', '.gguf'):
                    continue
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    result['blockers'].append('weight_is_not_a_regular_file:' + entry.name)
                    continue
                result['weights'].append({'name': entry.name, 'bytes': info.st_size,
                    'format': suffix.lstrip('.'), 'sha256': None})
    except OSError:
        result['blockers'].append('checkpoint_weight_inventory_unreadable')
    result['weights'].sort(key=lambda item: item['name'])
    safetensors = [item for item in result['weights'] if item['format'] == 'safetensors' and item['bytes'] > 0]
    if not safetensors:
        result['blockers'].append('local_hf_safetensors_required_GGUF_is_not_exporter_input')
    result['source_weight_bytes'] = sum(item['bytes'] for item in safetensors)
    index_path = path / 'model.safetensors.index.json'
    if index_path.exists() or index_path.is_symlink():
        try:
            mapping = json_regular(index_path).get('weight_map')
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError('weight_map missing')
            known = {item['name'] for item in safetensors}
            if any(not isinstance(name, str) or name not in known for name in mapping.values()):
                result['blockers'].append('safetensors_index_references_missing_or_nonregular_shards')
        except (OSError, ValueError):
            result['blockers'].append('safetensors_index_invalid_or_symlink')
    result['tokenizer_present'] = any((path / name).is_file() and not (path / name).is_symlink()
                                    for name in ('tokenizer.json', 'tokenizer.model'))
    if not result['tokenizer_present']:
        result['blockers'].append('local_tokenizer_missing_no_network_fallback_allowed')
    result['exporter_architecture_recognized'] = model_type in MODEL_TYPES
    result['npu_compilation_support_established'] = False
    return result


def inspect_host(sdk_archive=None):
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    memory = None
    try:
        memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        pass
    system, machine = platform.system(), platform.machine()
    blockers = []
    if system != 'Linux' or machine.lower() not in ('x86_64', 'amd64'):
        blockers.append('packaged_google_tensor_SDK_requires_Linux_x86_64')
    if memory is None or memory < 16_000_000_000:
        blockers.append('minimum_16GB_host_RAM_missing_or_unverified')
    if (packages['litert-torch'] is None and packages['litert-torch-nightly'] is None) or (packages['ai-edge-litert'] is None and packages['ai-edge-litert-nightly'] is None):
        blockers.append('exporter_or_npu_sdk_python_package_missing')
    environment_present = bool(os.environ.get('GOOGLE_TENSOR_SDK_BETA'))
    archive = sdk_archive or (Path(os.environ['GOOGLE_TENSOR_SDK_BETA']) if environment_present else None)
    archive_regular = bool(archive is not None and archive.is_file() and not archive.is_symlink())
    archive_name_expected = bool(archive is not None and archive.name == 'litert_plugin_compiler.tar.gz')
    if archive is not None and (not archive_regular or not archive_name_expected):
        blockers.append('explicit_Google_Tensor_beta_compiler_archive_missing_or_unverified')
    if packages['ai-edge-litert-sdk-google-tensor-nightly'] is None:
        blockers.append('Google_Tensor_SDK_wrapper_package_missing')
    return {'system': system, 'machine': machine, 'python': platform.python_version(),
            'physical_memory_bytes': memory, 'python_packages': packages, 'blockers': blockers,
            'scope': 'Packaged SDK export route only. Native Android ARM compilation is a separate working path; see ARM_NPU.md.',
            'compiler_archive': {'environment_variable_name': 'GOOGLE_TENSOR_SDK_BETA',
                'environment_variable_present': environment_present,
                'explicit_path_supplied': sdk_archive is not None,
                'is_regular_file': archive_regular, 'has_expected_filename': archive_name_expected,
                'contents_checksums_verified': False, 'sdk_documentation': SDK_DOC,
                'interpretation': 'Metadata only; archive not opened, downloaded or included. Environment variable value is not recorded. Current upstream installer downloads SDK 6.1.0 when no beta archive is supplied.'},
            'sdk_binary_and_installed_cli_verified': False}


def build_plan(repo, revision, checkpoint, output_dir, sdk_archive=None):
    source, host = inspect_checkpoint(checkpoint), inspect_host(sdk_archive)
    blockers = list(source['blockers']) + list(host['blockers'])
    if revision is None:
        blockers.append('exact_source_repository_revision_required')
    output = str(output_dir) if output_dir else '<PLANNED_OUTPUT_DIRECTORY>'
    if output_dir is None:
        blockers.append('planned_output_directory_required')
    if checkpoint and output_dir and (checkpoint == output_dir or checkpoint in output_dir.parents):
        blockers.append('planned_output_must_not_be_inside_source_checkpoint')
    argv = ['litert-torch', 'export_hf', '--model=' + (str(checkpoint) if checkpoint else '<LOCAL_HF_SAFETENSORS_CHECKPOINT>'),
            '--output_dir=' + output, '--split_cache', '--externalize_embedder', '--prefill_lengths=128,',
            '--cache_length=1280', '--quantization_recipe=weight_only_wi8_afp32',
            '--aot_backend=GOOGLE', '--aot_soc_model=Tensor_G5', '--use_litert_lm_compiler=False',
            '--aot_compilation_config_dict=' + repr({'google_tensor_enable_large_model_support': True})]
    return {'schema': 'pixelforge.huihui-offline-conversion-plan/v1',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'uncompiled/unvalidated', 'plan_only': True, 'would_execute': False,
        'source_provenance': {'repository': repo, 'revision': revision,
            'revision_metadata_pinned': revision is not None,
            'source_url': 'https://huggingface.co/' + repo + ('/tree/' + revision if revision else ''),
            'weight_checksums_verified': False, 'local_files_match_pinned_revision': 'unverified'},
        'checkpoint': source, 'host': host,
        'requirements': {'compile_host': 'Linux x86_64 for the packaged SDK route; native Android ARM compilation also works', 'minimum_host_RAM_bytes': 16_000_000_000,
            'recommended_large_model_host_RAM_bytes': 64_000_000_000,
            'checkpoint_format': 'local Hugging Face config/tokenizer and safetensors, not GGUF',
            'software': ['litert-torch', 'ai-edge-litert[npu-sdk]', 'ai-edge-litert-sdk-google-tensor-nightly',
                'Google Tensor SDK compiler (current installer downloads SDK 6.1.0; beta archive override is optional)'],
            'target': 'Tensor_G5', 'source_revision': 'exact 40-character commit SHA',
            'acceptance': 'Successful compilation, artifact checksums, tokenizer/template validation and fresh device inference benchmark still required.'},
        'conversion_candidate': {'argv_display_only': argv, 'command_runnable': False,
            'cli_verification': 'export_hf spelling and flags verified against pinned upstream source; installed dependencies and runtime compatibility remain unverified.',
            'exporter_reference_revision': EXPORTER_REVISION, 'exporter_reference_source': EXPORTER_SOURCE,
            'compiler_path': 'Legacy GOOGLE AOT path, deprecated upstream; experimental and unexecuted.',
            'template_validation': 'Preserve the Huihui tokenizer/chat template and verify it. Gemma 4 Jinja compatibility may require explicit model-specific adaptation; no template substitution is applied here.',
            'documentation': DOC, 'trust_remote_code': False, 'offline_environment': {
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'GOOGLE_TENSOR_BACKEND_ENABLED': '1'},
            'architecture_scope': 'Documented exporter family support does not guarantee Tensor G5 NPU compilation or deployment.'},
        'blockers': blockers + ['installed_compiler_CLI_and_Tensor_G5_support_not_validated',
                                'source_weights_not_verified_against_pinned_revision']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=validate_repo)
    parser.add_argument('--revision', type=validate_revision)
    parser.add_argument('--checkpoint', type=local_path)
    parser.add_argument('--output-dir', type=local_path, help='Planned compiler destination; never created')
    parser.add_argument('--sdk-archive', type=local_path, help='Optional beta archive override; metadata check only, never read or copied')
    parser.add_argument('--save', type=local_path, help='Exclusive-create private JSON plan; defaults to stdout')
    args = parser.parse_args(argv)
    plan = build_plan(args.repo, args.revision, args.checkpoint, args.output_dir, args.sdk_archive)
    if args.save:
        fd = os.open(args.save, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(plan, output, indent=2, allow_nan=False)
            output.write('\n')
    else:
        print(json.dumps(plan, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
