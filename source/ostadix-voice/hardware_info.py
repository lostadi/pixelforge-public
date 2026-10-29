"""Read-only CPU facts and formatting, independent of any AI/model runtime."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re


CPU_ROOT = Path('/sys/devices/system/cpu')
CPUINFO = Path('/proc/cpuinfo')
PROPERTIES = {
    'device_model': 'ro.product.model',
    'soc.manufacturer': 'ro.soc.manufacturer',
    'soc.model': 'ro.soc.model',
    'board_platform': 'ro.board.platform',
    'abi': 'ro.product.cpu.abilist',
}


def _read_text(path):
    with Path(path).open('r', encoding='utf-8', errors='strict') as handle:
        text = handle.read(65537)
    if len(text) > 65536:
        raise ValueError('hardware metadata exceeds the read limit')
    return text


def _policies():
    return sorted((path for path in (CPU_ROOT / 'cpufreq').glob('policy*')
                   if re.fullmatch(r'policy\d{1,4}', path.name)), key=lambda path: int(path.name[6:]))


def _cpus(text):
    values = set()
    for item in re.split(r'[,\s]+', text.strip()):
        match = re.fullmatch(r'(\d{1,4})(?:-(\d{1,4}))?', item)
        if not match:
            raise ValueError('invalid kernel CPU list')
        first, last = int(match[1]), int(match[2] or match[1])
        if first > last or last >= 4096:
            raise ValueError('kernel CPU list is outside the bounded metadata range')
        values.update(range(first, last + 1))
    return sorted(values)


def _cpu_records(text):
    records, seen = [], set()
    for block in re.split(r'\n\s*\n', text.strip()):
        fields = {key.strip().lower(): value.strip() for line in block.splitlines()
                  if ':' in line for key, value in [line.split(':', 1)]}
        if 'processor' not in fields:
            continue
        if not re.fullmatch(r'\d{1,4}', fields['processor']):
            raise ValueError('invalid processor ID in cpuinfo')
        cpu = int(fields['processor'])
        if cpu in seen or cpu >= 4096:
            raise ValueError('duplicate or out-of-range processor ID in cpuinfo')
        seen.add(cpu)
        entry = {'cpu': cpu}
        for key in ('implementer', 'part'):
            raw = fields.get('cpu ' + key)
            if raw is not None and not re.fullmatch(r'0x[0-9a-fA-F]{1,8}', raw):
                raise ValueError('invalid CPU ' + key + ' identifier')
            entry[key] = hex(int(raw, 16)) if raw else None
        entry['architecture'] = fields.get('cpu architecture')
        records.append(entry)
    if not records:
        raise ValueError('cpuinfo did not report logical processor records')
    return records


def _id_groups(records):
    groups = defaultdict(list)
    for record in records:
        if record['implementer'] is not None or record['part'] is not None:
            groups[(record['implementer'], record['part'], record['architecture'])].append(record['cpu'])
    return [{'implementer': key[0], 'part': key[1], 'architecture': key[2],
             'cpus': sorted(cpus), 'count': len(cpus)} for key, cpus in groups.items()]


def collect_cpu_info(get_property, *, read_file=None, policy_paths=None, check=None):
    """Return observed properties, CPU IDs and cpufreq policy groups with provenance.

    ``get_property`` may query only the fixed keys in PROPERTIES. Missing reads
    stay explicitly unavailable; model labels and guessed ARM marketing names
    never supply hardware facts. No root, shell, settings or permission changes.
    """
    read_file = read_file or _read_text
    policy_paths = _policies if policy_paths is None else policy_paths
    check = check or (lambda: None)
    result = {'phone_hardware': True, 'read_only': True, 'device_model': None,
              'soc': {'manufacturer': None, 'model': None}, 'board_platform': None,
              'abi': [], 'core_count': None, 'present_cpus': None, 'online_cpus': None,
              'cpu_identifiers': [], 'clusters': [], 'sources': {}, 'unavailable': {}}

    def unavailable(field, error):
        result['unavailable'][field] = str(error)[:240]

    for field, key in PROPERTIES.items():
        check()
        result['sources'][field] = {'property': key, 'argv': ['/system/bin/getprop', key]}
        try:
            value = get_property(key).strip()
            if (not value or len(value) > 256 or any(ord(char) < 32 or ord(char) == 127 for char in value)):
                raise ValueError('property is empty or not a bounded text value')
            if field == 'abi':
                values = value.split(',')
                if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', item) for item in values):
                    raise ValueError('invalid Android ABI list')
                result[field] = values
            elif field.startswith('soc.'):
                result['soc'][field.split('.')[1]] = value
            else:
                result[field] = value
        except (OSError, ValueError, UnicodeError) as error:
            unavailable(field, error)

    def read(field, path, parse):
        check()
        result['sources'][field] = str(path)
        try:
            return parse(read_file(path))
        except (OSError, ValueError, UnicodeError) as error:
            unavailable(field, error)
            return None

    result['present_cpus'] = read('present_cpus', CPU_ROOT / 'present', _cpus)
    result['online_cpus'] = read('online_cpus', CPU_ROOT / 'online', _cpus)
    records = read('cpu_identifiers', CPUINFO, _cpu_records) or []
    result['cpu_identifiers'] = _id_groups(records)
    if result['present_cpus'] is not None:
        result['core_count'] = len(result['present_cpus'])
        result['sources']['core_count'] = str(CPU_ROOT / 'present')
    elif records:
        result['core_count'] = len(records)
        result['sources']['core_count'] = str(CPUINFO)
    else:
        unavailable('core_count', 'neither the kernel present list nor cpuinfo could be read')

    def frequency(text):
        text = text.strip()
        if not re.fullmatch(r'\d{1,12}', text) or int(text) < 1:
            raise ValueError('frequency is not a positive integer in kHz')
        return int(text)

    check()
    try:
        policies = list(policy_paths())
    except OSError as error:
        unavailable('clusters', error)
        policies = []
    if not policies:
        unavailable('clusters', 'no readable kernel cpufreq policies were found')
    for policy in policies:
        policy = Path(policy)
        if policy.parent != CPU_ROOT / 'cpufreq' or not re.fullmatch(r'policy\d{1,4}', policy.name):
            unavailable('clusters', 'unexpected cpufreq policy path was ignored')
            continue
        name = 'clusters.' + policy.name
        cpus = read(name + '.cpus', policy / 'related_cpus', _cpus)
        if cpus is None:
            cpus = read(name + '.affected_cpus', policy / 'affected_cpus', _cpus)
        maximum = read(name + '.max_frequency_khz', policy / 'cpuinfo_max_freq', frequency)
        ceiling = read(name + '.scaling_max_frequency_khz', policy / 'scaling_max_freq', frequency)
        result['clusters'].append({'policy': policy.name, 'cpus': cpus,
                                   'max_frequency_khz': maximum, 'scaling_max_frequency_khz': ceiling})
    check()
    result['status'] = 'partial' if result['unavailable'] else 'observed'
    result['note'] = ('CPU/SoC properties identify this device, independently of any selected AI model. '
                      'CPU part IDs are reported without guessing microarchitecture names. '
                      'cpufreq maximums are kernel-reported limits, not measured current speed or a benchmark.')
    return result


def format_cpu_info(result):
    """Compact deterministic terminal answer, using observed fields only."""
    soc = result.get('soc') or {}
    name, maker = soc.get('model'), soc.get('manufacturer')
    chip = ((maker + ' ' if maker and not name.lower().startswith(maker.lower()) else '') + name
            if name else 'unavailable')
    lines = ['CPU / SoC: ' + chip, 'Device: ' + (result.get('device_model') or 'unavailable')]
    count = result.get('core_count')
    lines.append('CPU cores: ' + (str(count) if type(count) is int else 'unavailable') +
                 ' · ABI: ' + (', '.join(result.get('abi') or []) or 'unavailable'))
    frequencies = []
    for cluster in result.get('clusters') or []:
        maximum, cpus = cluster.get('max_frequency_khz'), cluster.get('cpus')
        if type(maximum) is int and isinstance(cpus, list):
            ghz = f'{maximum / 1000000:.3f}'.rstrip('0').rstrip('.')
            frequencies.append(str(len(cpus)) + '× up to ' + ghz + ' GHz')
    if frequencies:
        lines.append('Kernel CPU frequency groups: ' + '; '.join(frequencies))
    if result.get('unavailable'):
        lines.append('Some hardware fields were unavailable; missing values were not inferred.')
    return '\n'.join(lines)
