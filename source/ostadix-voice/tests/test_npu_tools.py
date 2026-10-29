import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_runtime import conversation_instruction, selected_engine, LlamaEngine
from npu_tools import NPUClient, NPU_STATE, NPU_MANIFEST, NPUWorkers
from local_chat import tool_feedback


class NPUIntegrationTests(unittest.TestCase):
    def test_npu_client_has_independent_state_and_explicit_profile(self):
        client = NPUClient()
        self.assertEqual(client.state, NPU_STATE)
        self.assertEqual(client.manifest_path, NPU_MANIFEST)
        self.assertEqual(client.state.name, 'ostadix-voice-npu')

    def test_profile_selection_does_not_replace_default_cpu_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpu, npu = root/'cpu.json', root/'npu.json'
            cpu.write_text('{}'); npu.write_text('{"backend":"litert-npu"}')
            with patch('local_runtime.MANIFEST', cpu), patch('npu_runtime.LiteRTEngine') as factory:
                selected_engine(root, manifest_path=npu)
                factory.assert_called_once_with(root, manifest_path=npu)
                self.assertIsInstance(selected_engine(root), LlamaEngine)
                self.assertEqual(json.loads(cpu.read_text()), {})

    def test_main_role_can_request_host_tools_but_helpers_remain_text_only(self):
        main = conversation_instruction('phone conversation assistant')
        self.assertIn('The host executes', main)
        self.assertNotIn('You have no tools', main)
        self.assertIn('You have no tools', conversation_instruction('reviewer'))

    def test_npu_route_translates_only_protocol_name(self):
        pool = NPUWorkers(client_factory=lambda: Mock())
        agent, workspace = Mock(), Mock()
        args = {'operation':'command', 'argv':['@npu','status']}
        with patch('local_workers.LocalWorkers.dispatch', return_value={'ready':True}) as dispatch:
            result = pool.dispatch(agent, args, workspace)
        self.assertTrue(result['npu_helper'])
        self.assertEqual(dispatch.call_args.args[1]['argv'], ['@local','status'])
        self.assertEqual(args['argv'], ['@npu','status'])

    def test_status_discovers_hardware_probe_without_executing_it(self):
        pool = NPUWorkers(client_factory=lambda: Mock())
        with patch('local_workers.LocalWorkers.dispatch', return_value={'ready': False}), \
                patch('subprocess.Popen', side_effect=AssertionError('Discovery must not run a probe')):
            result = pool.dispatch(Mock(), {'operation': 'command', 'argv': ['@npu', 'status']}, Mock())
        probe = result['hardware_diagnostics']
        self.assertTrue(probe['available'])
        self.assertTrue(probe['read_only'])
        self.assertFalse(probe['executed'])
        self.assertEqual(probe['read_argv'], ['@phone', 'npu'])
        self.assertEqual(probe['probe_argv'], [sys.executable, str(NPU_MANIFEST.parent / 'npu_telemetry.py')])
        self.assertNotIn('firmware_version', probe)

    def test_large_snapshot_keeps_hardware_facts_and_full_result_handle_visible(self):
        raw = {'clients': 'large counter evidence ' * 1000, 'firmware_version': 'fixture firmware\n',
               'power/runtime_status': 'suspended\n'}
        snapshot = {'schema': 'ostadix.pixel-tpu-telemetry/v1', 'phone_npu': True, 'read_only': True,
                    'driver': {'name': 'fixture-driver'}, 'raw': raw, 'errors': {}, 'devfreq': []}
        feedback = json.loads(tool_feedback(snapshot, True, event=7, arguments={
            'operation': 'command', 'argv': ['@phone', 'npu']}))
        self.assertEqual(feedback['result']['driver']['name'], 'fixture-driver')
        self.assertEqual(feedback['result']['firmware_version'], 'fixture firmware')
        self.assertEqual(feedback['result']['power_state'], 'suspended')
        self.assertEqual(feedback['saved_result']['read_argv'], ['@results', 'read', '7', '0', '4000'])
        self.assertEqual(snapshot['raw'], raw)
        self.assertNotIn('preview_truncated', feedback)


if __name__ == '__main__':
    unittest.main()
