"""Live signed-in Codex, real phone volume read, simulated microphone/speaker.

Run explicitly from ordinary Termux. Uses two Codex requests; never records audio,
changes volume, opens apps or approves actions. State is isolated and temporary.
"""
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal import TerminalSession, TerminalUI
from agent import AppError

ALLOWED = (['@voice', 'on'], ['@voice', 'status'], ['@phone', 'status'],
           ['@phone', 'help'], ['@phone', 'volume'])


class AudioFixture:
    def __init__(self):
        self.events = queue.Queue()
        self.mode = None
        self.stopping = False
        self.calls = []

    def start(self, mode, text=None):
        assert self.mode is None
        self.mode = mode
        self.calls.append(mode)

    def stop(self):
        self.mode = None


def main():
    if os.geteuid() == 0:
        raise SystemExit('Run in ordinary Termux, outside root.')
    with tempfile.TemporaryDirectory(prefix='ostadix-phone-integration-') as folder:
        root = Path(folder) / 'workspace'
        root.mkdir()
        session = TerminalSession(root, Path(folder) / 'state')
        execute = session.agent.execute

        def probe_execute(arguments):
            if arguments.get('operation') != 'command' or arguments.get('argv') not in ALLOWED:
                raise AppError('This integration probe permits only voice activation/status and phone volume/status reads.')
            return execute(arguments)

        session.agent.execute = probe_execute
        audio = AudioFixture()
        ui = TerminalUI(session, speech=audio, output=lambda text: None)
        ui.local = None
        ui.restore()
        report = {'audio': 'simulated; no microphone or speaker activated', 'turns': []}
        try:
            for request, required in (
                ('Please turn on hands-free voice conversation so I can speak again after each reply. '
                 'Use the built-in voice control now and briefly confirm its actual state.', '@voice'),
                ('What is my phone\'s current media volume and maximum? Read it with your phone '
                 'controls and tell me. Do not change any setting or open any app.', '@phone'),
            ):
                previous = session.store.events(session.sid)
                after = previous[-1]['seq'] if previous else 0
                session.submit(request)
                deadline = time.monotonic() + 90
                while session.agent.active and time.monotonic() < deadline:
                    ui.pump()
                    time.sleep(.03)
                assert not session.agent.active, 'Live model request timed out'
                ui.pump()
                events = session.store.events(session.sid, after)
                calls = [event['data']['arguments'] for event in events if event['kind'] == 'tool']
                results = [event['data'] for event in events if event['kind'] == 'tool_result']
                assert all(call.get('operation') == 'command' and call.get('argv') in ALLOWED
                           for call in calls), calls
                assert any(call.get('argv', [None])[0] == required for call in calls), calls
                assert results and all(result['success'] for result in results), results
                finished = [event['data'] for event in events if event['kind'] == 'finished']
                assert finished and finished[-1]['status'] in ('idle', 'completed'), finished
                if required == '@voice':
                    assert ui.voice_enabled, 'Model did not enable the terminal voice loop'
                    assert 'speaking' in audio.calls, 'Final reply was not queued for speech'
                else:
                    assert any(result['result'].get('stream') == 'music' for result in results), results
                report['turns'].append({'tools': calls, 'results': results,
                                        'answer': finished[-1]['text']})
                ui.set_voice(False, announce=False)
            print(json.dumps(report, indent=2))
        finally:
            ui.session.agent.interface.detach()
            audio.stop()
            session.close()


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
