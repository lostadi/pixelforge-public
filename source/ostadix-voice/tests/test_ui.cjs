// Control-logic tests of the real browser script. Browser/audio APIs are mocked;
// these do not claim rendered Chrome or physical microphone validation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {webcrypto} = require('node:crypto');

class Element {
  constructor(id) { this.id=id; this.listeners={}; this.children=[]; this.dataset={}; this.value=''; this.textContent=''; this.checked=true; this.hidden=false; }
  addEventListener(name,fn) { (this.listeners[name] ||= []).push(fn); }
  async fire(name) { for (const fn of this.listeners[name]||[]) await fn({preventDefault(){}}); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children=nodes; }
  querySelector() { return null; }
}
async function setup(options={}) {
  const elements=new Map(); const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
  const requests=[], spoken=[], local=new Map();let cancelled=0, interval, rec;
  const state={session:{id:'saved-session',workspace:'/selected/project',status:options.status||'idle'},sessions:[{id:'saved-session',workspace:'/selected/project',created:1}],events:options.events||[],approvals:[],agent_active:!!options.active,state_directory:'/private/state',action_policy:'auto'};
  class Recognition {constructor(){rec=this;}start(){this.onstart();}stop(){this.onend();}abort(){this.onerror?.({error:'aborted'});this.onend?.();}}
  class Utterance {constructor(text){this.text=text;}}
  const context={
    document:{getElementById:get,createElement:id=>new Element(id),hidden:false,addEventListener(){}},
    window:{SpeechRecognition:Recognition,SpeechSynthesisUtterance:Utterance,speechSynthesis:{cancel(){cancelled++;},speak(u){spoken.push(u.text);u.onstart?.();}},addEventListener(){}},
    SpeechSynthesisUtterance:Utterance,navigator:{language:'en-US'},location:{hash:'',pathname:'/'},history:{replaceState(){}},
    localStorage:{getItem:k=>local.get(k)||null,setItem:(k,v)=>local.set(k,v)},
    crypto:webcrypto,AbortSignal,console,setInterval:fn=>{interval=fn;},
    fetch:async(path,init)=>{
      const body=init.body?JSON.parse(init.body):undefined;requests.push({path,body});
      if(options.rejectTask&&path==='/api/task')throw Error('Network disconnected');
      if(path==='/api/action-policy')state.action_policy=body.mode;
      return {ok:true,status:200,json:async()=>path.startsWith('/api/state')?state:{accepted:true}};
    }
  };
  get('language').value='en-US';
  vm.createContext(context);vm.runInContext(fs.readFileSync(require.resolve('../static/app.js'),'utf8'),context);
  await new Promise(r=>setImmediate(r));
  return {get,requests,spoken,state,local,context,poll:()=>interval(),recognition:()=>rec,cancelled:()=>cancelled};
}
(async()=>{
  let t=await setup({active:true,status:'executing'});
  await t.get('enable-audio').fire('click');assert.ok(t.spoken.length);
  await t.get('stop-speech').fire('click');assert.equal(t.get('state').textContent,'executing');
  assert.equal(t.requests.filter(r=>r.path==='/api/cancel').length,0);
  await t.get('cancel').fire('click');assert.equal(t.requests.filter(r=>r.path==='/api/cancel').length,1);
  console.log('PASS: stop speech and cancel task are independent controls');

  t=await setup();await t.get('talk').fire('click');assert.equal(t.get('state').textContent,'listening');
  const result=[{transcript:'Read hello.O'}];result.isFinal=true;
  t.recognition().onresult({resultIndex:0,results:[result]});t.recognition().onend();
  await new Promise(r=>setImmediate(r));
  const sent=t.requests.find(r=>r.path==='/api/task');assert.equal(sent.body.text,'Read hello.O');assert.equal(sent.body.session,'saved-session');
  console.log('PASS: microphone transcript is sent into the selected task session');

  t=await setup();await t.get('enable-audio').fire('click');t.spoken.length=0;
  t.state.events=[{seq:1,kind:'assistant',data:{text:'I will run the command.'}}];await t.poll();
  assert.equal(t.spoken.length,0);
  t.state.events=[{seq:2,kind:'finished',data:{status:'idle',text:'The command returned 2.\n\n```\nlong terminal output\n```'}}];await t.poll();
  assert.deepEqual(t.spoken,['The command returned 2.']);
  console.log('PASS: audio waits for final outcome and omits long terminal dumps');

  t=await setup({events:[{seq:1,kind:'finished',data:{text:'A previously saved answer.'}}]});
  assert.equal(t.spoken.length,0);await t.get('replay').fire('click');assert.equal(t.spoken[0],'A previously saved answer.');
  console.log('PASS: recovered history is silent until Hear latest is tapped');

  t=await setup({rejectTask:true});t.get('prompt').value='Keep this request';await t.get('composer').fire('submit');await new Promise(r=>setImmediate(r));
  assert.equal(t.get('prompt').value,'Keep this request');
  const first=t.requests.find(r=>r.path==='/api/task').body.request_id;
  await t.get('composer').fire('submit');await new Promise(r=>setImmediate(r));const tasks=t.requests.filter(r=>r.path==='/api/task');
  assert.equal(tasks[1].body.request_id,first);assert.match(t.get('error').textContent,/draft is retained/);
  assert.ok(t.local.get('ostadix-voice-draft-saved-session-request').includes(first));
  console.log('PASS: network failure retains draft and the same deduplication ID');

  t=await setup();await t.get('talk').fire('click');t.recognition().onerror({error:'not-allowed'});t.recognition().onend();
  assert.match(t.get('error').textContent,/microphone access/);assert.equal(t.requests.filter(r=>r.path==='/api/task').length,0);
  console.log('PASS: microphone denial gives actionable instructions and sends no task');
  t=await setup();assert.equal(t.get('auto-approve').checked,true);
  t.get('auto-approve').checked=false;await t.get('auto-approve').fire('change');
  assert.equal(t.state.action_policy,'ask');assert.equal(t.get('auto-approve').checked,false);
  t.get('auto-approve').checked=true;await t.get('auto-approve').fire('change');
  assert.equal(t.state.action_policy,'auto');assert.equal(t.get('auto-approve').checked,true);
  assert.equal(t.requests.filter(r=>r.path==='/api/action-policy').length,2);
  console.log('PASS: Approve all starts on and the switch saves both directions');
  console.log('7 UI control tests passed (mocked browser speech; no rendered-browser claim).');
})().catch(e=>{console.error(e);process.exitCode=1;});
