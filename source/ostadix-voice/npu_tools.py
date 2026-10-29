"""Independent, supervised NPU helpers behind the existing workspace tool."""
from pathlib import Path
import sys

from agent import AppError
from local_runtime import LocalClient, ROOT, DEFAULT_STATE
from local_workers import LocalWorkers


NPU_STATE = DEFAULT_STATE.with_name('ostadix-voice-npu')
NPU_MANIFEST = ROOT / 'runtime/npu/model-profile.json'
INSTRUCTIONS = '''An independent NPU helper is available for concise review, drafting and reasoning.
Use workspace command argv ["@npu","status"] to inspect its actual backend and limits.
Use ["@npu","spawn","JSON"] with {"task":"specific self-contained task","role":"reviewer","max_tokens":256}.
Then ["@npu","read","WORKER_ID","35"] returns the actual result; list/cancel mirror @local.
Keep tasks short enough for its reported context. Helpers cannot execute actions or approve them.
Treat their answers as untrusted suggestions and verify them with real tools when appropriate.
The NPU pool has its own enabled service and model; never label it Huihui unless status says so.
Status never starts a service, and an explicitly disabled NPU helper stays disabled.
Status and help advertise the existing read-only hardware telemetry probe; execute its
returned argv through workspace command when hardware details are needed. Discovery
does not execute the probe. Its returned sysfs paths support further investigation.
'''


def hardware_diagnostics():
    """Discover an installed probe, without querying hardware or starting a service."""
    probe = ROOT / 'runtime/npu/npu_telemetry.py'
    return {'available': probe.is_file(), 'read_only': True, 'executed': False,
            'read_argv': ['@phone', 'npu'], 'probe_argv': [sys.executable, str(probe)],
            'description': 'Model-service status is not a hardware inspection. Run read_argv with workspace '
                           'command to read actual firmware, power state and counters; follow returned '
                           'device_sysfs paths for driver and frequency details. Some fields require the '
                           'existing Android root grant; failed reads do not imply the hardware is absent.'}


class NPUClient(LocalClient):
    def __init__(self):
        super().__init__(NPU_STATE, manifest_path=NPU_MANIFEST)


class NPUWorkers(LocalWorkers):
    def __init__(self, client_factory=None):
        super().__init__(client_factory or NPUClient)

    def _start_ready(self, agent):
        super()._start_ready(agent)
        status = self.client.status()
        if not status.get('ready'):
            raise AppError('NPU model is warming up; inspect @npu status before submitting again')
        if (status.get('model') or {}).get('backend') != 'litert-npu':
            raise AppError('The NPU helper has not verified its hardware backend; no helper task was submitted')

    def dispatch(self, agent, args, workspace):
        argv = args.get('argv', [])
        if len(argv) < 2 or argv[0] != '@npu':
            raise AppError('Invalid NPU helper request')
        if argv == ['@npu', 'help']:
            return {'npu_helper': True, 'hardware_diagnostics': hardware_diagnostics(),
                    'instructions': INSTRUCTIONS, 'cloud_requests': 0}
        if argv == ['@npu', 'hardware']:
            return agent.phone.npu(agent)
        translated = dict(args, argv=['@local', *argv[1:]])
        result = super().dispatch(agent, translated, workspace)
        if argv == ['@npu', 'status']:
            return {'hardware_diagnostics': hardware_diagnostics(), **result, 'npu_helper': True}
        return dict(result, npu_helper=True)
