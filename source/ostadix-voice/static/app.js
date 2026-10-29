'use strict';
const $ = id => document.getElementById(id);
const recognitionClass = window.SpeechRecognition || window.webkitSpeechRecognition;
let session = null, cursor = 0, connected = false, taskState = 'idle', active = false;
let listening = false, speaking = false, voiceEnabled = false, recognition = null;
let approval = null, latest = '', polling = false, bootstrapping = true, submitting = false;
let pendingRequest = null, speechGeneration = 0, connectionError = false;
const draftKey = () => `ostadix-voice-draft-${session || 'new'}`;
const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* Server history still persists. */ } }
};

function error(text) { $('error').textContent = text; $('error').hidden = !text; }
function displayState() {
  const value = !connected ? 'disconnected' : listening ? 'listening' : speaking ? 'speaking' : taskState;
  $('state').textContent = value;
  $('state').dataset.state = value;
  $('talk').disabled = !connected || !recognitionClass || (active && !listening);
  $('talk').textContent = listening ? 'Finish & send' : 'Tap to talk';
  $('send').disabled = !connected || active || submitting;
  $('cancel').disabled = !connected || !active;
  $('sessions').disabled = active;
}
async function api(path, data) {
  const options = {credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.timeout(15000)};
  if (data !== undefined) Object.assign(options, {method: 'POST', headers: {'Content-Type':'application/json', 'X-Ostadix-Voice':'1'}, body:JSON.stringify(data)});
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) {
    if (response.status === 401) $('pair-help').hidden = false;
    throw new Error(body.error || `Request failed (${response.status})`);
  }
  return body;
}
function stopSpeech() {
  speechGeneration++;
  window.speechSynthesis?.cancel();
  speaking = false;
  displayState();
}
function spokenSummary(text) {
  return text.replace(/```[\s\S]*?```/g, ' Code is shown in the transcript. ')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1').replace(/[`#*_]/g, '')
    .split(/\n\s*\n/)[0].slice(0, 650);
}
function speak(text, forced = false) {
  if (!window.speechSynthesis || !window.SpeechSynthesisUtterance) {
    error('Speech playback is unavailable in this browser. The complete answer is saved below.'); return;
  }
  if (!forced && (!voiceEnabled || !$('auto-speak').checked || document.hidden)) return;
  stopSpeech();
  const generation = speechGeneration;
  const utterance = new SpeechSynthesisUtterance(spokenSummary(text));
  utterance.lang = $('language').value || navigator.language;
  utterance.onstart = () => { if (generation === speechGeneration) { speaking = true; displayState(); } };
  utterance.onend = () => { if (generation === speechGeneration) { speaking = false; displayState(); } };
  utterance.onerror = event => {
    if (generation !== speechGeneration) return;
    speaking = false; displayState();
    if (!['canceled','interrupted'].includes(event.error)) error('Audio playback failed. Tap Hear latest or Test speaker to allow playback.');
  };
  window.speechSynthesis.speak(utterance);
}
function addEvent(event, live) {
  const {kind, data} = event;
  if (kind === 'approval' || kind === 'approval_decision' || kind === 'tool') return;
  $('transcript').querySelector('.empty')?.remove();
  const div = document.createElement('article'); div.className = `entry ${kind}`;
  const role = document.createElement('span'); role.className = 'role';
  role.textContent = ({user:'You', assistant:'Codex', log:'Command output', tool_result:'Actual tool result', notice:'Status', finished:'Task outcome'})[kind] || kind;
  div.append(role);
  if (kind === 'tool_result' || kind === 'log') {
    const details = document.createElement('details'), summary = document.createElement('summary'), pre = document.createElement('pre');
    summary.textContent = kind === 'log' ? 'Show output' : data.success ? 'Show result' : 'Show error';
    pre.textContent = kind === 'log' ? data.text : JSON.stringify(data.result, null, 2);
    details.append(summary, pre); div.append(details);
  } else {
    const p = document.createElement('p'); p.textContent = data.text || ''; div.append(p);
  }
  $('transcript').append(div);
  if (kind === 'finished') {
    latest = data.text || '';
    if (live) speak(latest);
  }
}
function renderApprovals(approvals) {
  approval = approvals[0] || null;
  $('approval').hidden = !approval;
  if (approval) {
    $('approval-warning').textContent = approval.warning;
    $('approval-content').textContent = JSON.stringify(approval.arguments, null, 2);
  }
}
async function poll() {
  if (polling) return;
  polling = true;
  try {
    const data = await api(`/api/state?after=${cursor}${session ? '&session='+encodeURIComponent(session) : ''}`);
    connected = true;
    $('auto-approve').checked = data.action_policy === 'auto';
    $('auto-approve').disabled = false;
    if (connectionError) { error(''); connectionError = false; }
    if (!session) { session = data.session.id; $('prompt').value = storage.get(draftKey()) || ''; restorePending(); }
    taskState = data.session.status;
    active = data.agent_active;
    $('workspace').textContent = data.session.workspace;
    $('storage').textContent = `Saved on device: ${data.state_directory}`;
    const signature = JSON.stringify(data.sessions.map(s => [s.id, s.workspace]));
    if ($('sessions').dataset.signature !== signature) {
      $('sessions').replaceChildren(...data.sessions.map(s => {
        const option = document.createElement('option'); option.value = s.id;
        option.textContent = `${s.workspace.split('/').pop()} · ${new Date(s.created*1000).toLocaleString()}`;
        return option;
      }));
      $('sessions').dataset.signature = signature;
    }
    $('sessions').value = session;
    for (const event of data.events) { addEvent(event, !bootstrapping); cursor = event.seq; }
    if (data.events.length < 300) bootstrapping = false;
    renderApprovals(data.approvals);
    $('pair-help').hidden = true;
  } catch (e) {
    connected = false;
    connectionError = true;
    error(`${e.message} Your saved history remains on this device. If Android stopped the server, restart it in Termux.`);
  } finally { polling = false; displayState(); }
}
async function submit() {
  const text = $('prompt').value.trim();
  if (!text || submitting) return;
  storage.set(draftKey(), text);
  if (!pendingRequest || pendingRequest.text !== text || pendingRequest.session !== session) pendingRequest = {text, session, request_id:crypto.randomUUID()};
  storage.set(draftKey()+'-request', JSON.stringify(pendingRequest));
  submitting = true; displayState(); error('');
  try {
    await api('/api/task', pendingRequest);
    pendingRequest = null; $('prompt').value = ''; storage.set(draftKey(), ''); storage.set(draftKey()+'-request', '');
    await poll();
  } catch (e) { error(`${e.message} Your draft is retained. Check the transcript before retrying; the request may already have reached the server.`); }
  finally { submitting = false; displayState(); }
}
function listen() {
  voiceEnabled = true;
  if (listening) { recognition.stop(); return; }
  if (!recognitionClass) { error('Browser speech recognition is unavailable. Use your keyboard microphone or type.'); return; }
  stopSpeech(); error('');
  recognition = new recognitionClass();
  recognition.lang = $('language').value || navigator.language;
  recognition.continuous = false; recognition.interimResults = true;
  let finalText = '', hadError = false;
  recognition.onstart = () => { listening = true; displayState(); };
  recognition.onresult = event => {
    let interim = '';
    for (let i=event.resultIndex; i<event.results.length; i++) {
      if (event.results[i].isFinal) finalText += event.results[i][0].transcript + ' ';
      else interim += event.results[i][0].transcript;
    }
    $('prompt').value = (finalText+interim).trim(); storage.set(draftKey(), $('prompt').value);
  };
  recognition.onerror = event => {
    hadError = true;
    const help = {'not-allowed':'Allow microphone access for Chrome in Android Settings and for this localhost site.', 'audio-capture':'No microphone is available; check Android permissions and other recording apps.', 'network':'The browser speech service is offline. Try your keyboard microphone or type.', 'no-speech':'No speech detected. Tap to talk and try again.'};
    error(help[event.error] || `Speech recognition stopped: ${event.error}. Your draft is retained.`);
  };
  recognition.onend = () => {
    listening = false; displayState();
    if (!hadError && finalText.trim()) { $('prompt').value = finalText.trim(); submit(); }
  };
  try { recognition.start(); } catch (e) { listening = false; displayState(); error(e.message); }
}
function restorePending() {
  try { pendingRequest = JSON.parse(storage.get(draftKey()+'-request') || 'null'); } catch { pendingRequest = null; }
}
function resetSession(id) {
  stopSpeech(); session = id; cursor = 0; latest = ''; bootstrapping = true;
  $('transcript').replaceChildren(); $('prompt').value = storage.get(draftKey()) || '';
  restorePending();
}
$('composer').addEventListener('submit', event => { event.preventDefault(); voiceEnabled = true; submit(); });
$('prompt').addEventListener('input', () => storage.set(draftKey(), $('prompt').value));
$('auto-approve').addEventListener('change', async () => {
  const requested = $('auto-approve').checked;
  $('auto-approve').disabled = true;
  try { await api('/api/action-policy', {session, mode: requested ? 'auto' : 'ask'}); await poll(); }
  catch (e) { $('auto-approve').checked = !requested; error(e.message); }
  finally { $('auto-approve').disabled = !connected; }
});
$('talk').addEventListener('click', listen);
$('stop-speech').addEventListener('click', stopSpeech);
$('enable-audio').addEventListener('click', () => { voiceEnabled = true; speak('Voice is ready. Tap to talk and ask about your project.', true); });
$('replay').addEventListener('click', () => { voiceEnabled = true; if (latest) speak(latest, true); });
$('cancel').addEventListener('click', async () => {
  try { await api('/api/cancel', {session}); await poll(); } catch (e) { error(e.message); }
});
for (const [id, allow] of [['allow',true],['deny',false]]) $(id).addEventListener('click', async () => {
  if (!approval) return;
  $('allow').disabled = $('deny').disabled = true;
  try { await api('/api/approve', {session, id:approval.id, allow}); await poll(); } catch (e) { error(e.message); }
  finally { $('allow').disabled = $('deny').disabled = false; }
});
$('sessions').addEventListener('change', async () => {
  try { const data = await api('/api/select', {session:$('sessions').value}); resetSession(data.session); await poll(); } catch(e) { error(e.message); }
});
$('new-session').addEventListener('submit', async event => {
  event.preventDefault();
  try { const data = await api('/api/new', {session, path:$('project-path').value}); resetSession(data.session); await poll(); } catch(e) { error(e.message); }
});
$('inspect').addEventListener('submit', async event => {
  event.preventDefault();
  try { await api('/api/inspect', {session, path:$('file-path').value}); await poll(); } catch(e) { error(e.message); }
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) { stopSpeech(); recognition?.abort(); } else poll();
});
window.addEventListener('pagehide', () => { stopSpeech(); recognition?.abort(); });
(async () => {
  const token = location.hash.slice(1);
  if (token) {
    history.replaceState(null, '', location.pathname);
    try { await api('/api/pair', {token}); } catch(e) { error(e.message); }
  }
  if (!recognitionClass) $('voice-help').textContent = 'This browser has no speech recognition. Use the microphone on your Android keyboard to dictate into Your request, then tap Send.';
  await poll(); setInterval(poll, 1000);
})();
