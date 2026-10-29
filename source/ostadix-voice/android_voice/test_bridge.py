"""Transport tests use fake peers; never open the phone microphone."""
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from android_voice.bridge import NativeVoiceBridge, NativeVoiceError

class TransportTests(unittest.TestCase):
    def launch(self, bridge, *, probe=False, bad_first=False, event=None):
        commands=[]
        threads=[]
        def run(args, **kwargs):
            port=int(args[args.index('port')+1])
            token=args[args.index('token')+1]
            def peer():
                if bad_first:
                    with socket.create_connection(('127.0.0.1',port)) as rejected:
                        rejected.sendall((json.dumps({'type':'hello','protocol':1,'token':'wrong'})+'\n').encode())
                        self.assertEqual(rejected.recv(1),b'')
                with socket.create_connection(('127.0.0.1',port)) as connection:
                    def emit(data): connection.sendall((json.dumps(data)+'\n').encode())
                    emit({'type':'hello','protocol':1,'token':token})
                    emit({'type':'capabilities','mic_active':False,'on_device_available':True})
                    if probe:return
                    reader=connection.makefile('rb')
                    for line in reader:
                        command=json.loads(line)
                        commands.append(command)
                        if command['cmd']=='start':
                            muted=command.get('microphone_muted') is True
                            emit({'type':'ready','seq':1,'mic_active':not muted,'microphone_muted':muted,'recognizer':'android_on_device'})
                            if event:emit(event)
                        if command['cmd']=='microphone_muted':
                            emit({'type':'microphone_muted','muted':command['muted'],
                                  'mic_active':not command['muted'],'audio_mode_owned':not command['muted']})
                        if command['cmd']=='stop':
                            emit({'type':'stopped','mic_active':False})
                            return
            worker=threading.Thread(target=peer,daemon=True)
            threads.append(worker);worker.start()
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        with patch('android_voice.bridge.subprocess.run',side_effect=run), patch('companion_notify._api_environment',return_value={}):
            data=bridge.probe() if probe else bridge.start(timeout=2)
        return data,commands,threads

    def test_probe_never_sends_start(self):
        bridge=NativeVoiceBridge(lambda event:None)
        caps,commands,threads=self.launch(bridge,probe=True)
        self.assertFalse(caps['mic_active']);self.assertTrue(caps['on_device_available'])
        self.assertEqual(commands,[])
        for worker in threads:worker.join(1)

    def test_authentication_rejects_wrong_token_before_accepting_session(self):
        bridge=NativeVoiceBridge(lambda event:None)
        caps,commands,threads=self.launch(bridge,bad_first=True)
        self.assertTrue(caps['mic_active'])
        bridge.close()
        for worker in threads:worker.join(1)
        self.assertEqual(commands[0],{'cmd':'start'})
        self.assertTrue(bridge._native_stopped.is_set())
        self.assertFalse(bridge.capabilities['mic_active'])
        self.assertEqual(bridge.capabilities['capture_state'],'stopped')

    def test_stop_speaking_preserves_live_connection(self):
        bridge=NativeVoiceBridge(lambda event:None)
        caps,commands,threads=self.launch(bridge)
        bridge.speak('Hello.','one');bridge.stop_speaking();bridge.speak('Still here.','two')
        bridge.close();bridge.close()
        for worker in threads:worker.join(1)
        self.assertEqual([x['cmd'] for x in commands],['start','speak','stop_speaking','speak','stop'])

    def test_partial_event_is_delivered_without_waiting_for_final(self):
        received=[];ready=threading.Event()
        def callback(event):
            received.append(event)
            if event.get('type')=='partial':ready.set()
        bridge=NativeVoiceBridge(callback)
        _,_,threads=self.launch(bridge,event={'type':'partial','seq':2,'text':'héllo world','during_playback':True})
        self.assertTrue(ready.wait(1))
        self.assertEqual(received[-1]['text'],'héllo world')
        self.assertTrue(received[-1]['during_playback'])
        bridge.close()
        for worker in threads:worker.join(1)

    def test_microphone_mute_preserves_speech_and_reports_physical_capture_state(self):
        muted=threading.Event()
        bridge=NativeVoiceBridge(lambda event:muted.set() if event.get('type')=='microphone_muted' else None)
        _,commands,threads=self.launch(bridge)
        bridge.speak('The answer continues.','answer')
        bridge.set_microphone_muted(True)
        self.assertTrue(muted.wait(1))
        self.assertTrue(bridge.capabilities['microphone_muted'])
        self.assertFalse(bridge.capabilities['mic_active'])
        self.assertFalse(bridge.capabilities['audio_mode_owned'])
        bridge.set_microphone_muted(False)
        bridge.close()
        for worker in threads:worker.join(1)
        self.assertEqual([command['cmd'] for command in commands],
                         ['start','speak','microphone_muted','microphone_muted','stop'])

    def test_muted_startup_sends_no_unmuted_start(self):
        bridge=NativeVoiceBridge(lambda event:None)
        bridge.set_microphone_muted(True)
        caps,commands,threads=self.launch(bridge)
        self.assertFalse(caps['mic_active'])
        self.assertTrue(caps['microphone_muted'])
        bridge.close()
        for worker in threads:worker.join(1)
        self.assertEqual(commands[0],{'cmd':'start','microphone_muted':True})

    def test_passive_pong_updates_metrics_after_sequence_validation(self):
        seen=threading.Event()
        bridge=NativeVoiceBridge(lambda event:seen.set() if event.get('type')=='pong' else None)
        _,commands,threads=self.launch(bridge,event={'type':'pong','seq':2,'mic_active':True,
                                                   'frames':32000,'dropped_frames':640})
        self.assertTrue(seen.wait(1))
        self.assertEqual(bridge.capabilities['frames'],32000)
        self.assertEqual(bridge.capabilities['dropped_frames'],640)
        self.assertTrue(bridge.capabilities['status_fresh'])
        self.assertEqual(commands,[{'cmd':'start'}])
        bridge.close()
        for worker in threads:worker.join(1)

    def test_launch_failure_closes_listener(self):
        bridge=NativeVoiceBridge(lambda event:None)
        with patch('android_voice.bridge.subprocess.run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='Error: unavailable')), patch('companion_notify._api_environment',return_value={}):
            with self.assertRaises(NativeVoiceError):bridge.start(timeout=.1)
        self.assertTrue(bridge._closed.is_set());self.assertIsNone(bridge._listener)

    def test_command_bound(self):
        bridge=NativeVoiceBridge(lambda event:None)
        _,commands,threads=self.launch(bridge)
        with self.assertRaises(NativeVoiceError):bridge.speak('a'*40000)
        bridge.close()
        for worker in threads:worker.join(1)
        self.assertNotIn('speak',[c['cmd'] for c in commands])

    def test_sending_after_close_fails(self):
        bridge=NativeVoiceBridge(lambda event:None)
        bridge.close()
        with self.assertRaises(NativeVoiceError):bridge.speak('No capture')

if __name__=='__main__':unittest.main()
