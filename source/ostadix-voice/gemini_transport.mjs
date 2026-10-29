// Fixed-host Gemini HTTPS transport. Credentials never leave this process except
// in the authorization header to Google's API; stdin/stdout contain no credential.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import https from 'node:https';

const emit = value => process.stdout.write(JSON.stringify(value) + '\n');
let finished = false;
let request;
const fail = (code, retrySeconds, quota) => {
  if (finished) return;
  finished = true;
  const extra=typeof retrySeconds==='number'&&Number.isFinite(retrySeconds)&&retrySeconds>=0&&retrySeconds<=60
    ? {retry_seconds:retrySeconds} : {};
  if(quota&&['minute','day'].includes(quota.scope)){
    extra.quota_scope=quota.scope;
    if(Number.isSafeInteger(quota.value)&&quota.value>=0&&quota.value<=1000000000000)extra.quota_value=quota.value;
  }
  emit({type:'error', code, ...extra});
  request?.destroy();
  process.exitCode = 1;
};
function loadKey() {
  let derived, plain;
  try {
    const parts = fs.readFileSync(path.join(os.homedir(), '.gemini/gemini-credentials.json'), 'utf8').split(':');
    if (parts.length !== 3 || !parts.every(p => /^[0-9a-fA-F]+$/.test(p))) throw Error();
    const iv = Buffer.from(parts[0], 'hex'), tag = Buffer.from(parts[1], 'hex');
    if (![12,16].includes(iv.length) || tag.length !== 16) throw Error();
    derived = crypto.scryptSync('gemini-cli-oauth', `${os.hostname()}-${os.userInfo().username}-gemini-cli`, 32);
    const decipher = crypto.createDecipheriv('aes-256-gcm', derived, iv, {authTagLength:16});
    decipher.setAuthTag(tag);
    plain = Buffer.concat([decipher.update(Buffer.from(parts[2], 'hex')), decipher.final()]);
    const saved = JSON.parse(plain.toString('utf8'))?.['gemini-cli-api-key']?.['default-api-key'];
    const key = JSON.parse(saved)?.token?.accessToken;
    if (typeof key !== 'string' || !key.trim() || /[\r\n]/.test(key)) throw Error();
    return key;
  } finally {
    derived?.fill(0);
    plain?.fill(0);
  }
}
async function main() {
  let input = '';
  for await (const chunk of process.stdin) {
    input += chunk;
    if (input.length > 262144) { fail('request'); return; }
  }
  let payload;
  try { payload = JSON.parse(input); } catch { fail('request'); return; }
  const models=new Set(['gemini-3.8-flash','gemini-3.7-flash','gemini-4-argon','gemini-2.5-flash']);
  if (!models.has(payload?.model) || payload.stream !== true || !Array.isArray(payload.messages)) {
    fail('request'); return;
  }
  let key;
  try { key = loadKey(); } catch { fail('credentials'); return; }
  const requestedTimeout=payload._ostadix_timeout_ms;
  delete payload._ostadix_timeout_ms;
  const timeoutMillis=Number.isInteger(requestedTimeout)&&requestedTimeout>=0&&requestedTimeout<=120000
    ? requestedTimeout : 120000;
  const body = JSON.stringify(payload);
  const deadline = timeoutMillis === 0 ? null : setTimeout(() => fail('timeout'), timeoutMillis);
  request = https.request({hostname:'generativelanguage.googleapis.com',
    path:'/v1beta/openai/chat/completions', method:'POST', timeout:Math.min(30000,timeoutMillis),
    headers:{'Authorization':`Bearer ${key}`, 'Content-Type':'application/json',
      'Accept':'text/event-stream', 'Content-Length':Buffer.byteLength(body)}}, response => {
    key = null;
    if (response.statusCode === 429) {
      // Only the retry duration escapes this capped body; never log its text,
      // quota identifiers, account information or request headers.
      let retrySeconds, quota;
      const header=response.headers?.['retry-after'];
      if (typeof header==='string'&&/^\d{1,6}(?:\.\d{1,9})?$/.test(header.trim())) {
        retrySeconds=Math.min(60,Number(header.trim()));
      } else if (typeof header==='string'&&header.length<=128) {
        const stamp=Date.parse(header);
        if(Number.isFinite(stamp))retrySeconds=Math.min(60,Math.max(0,Math.ceil((stamp-Date.now())/1000)));
      }
      response.setEncoding('utf8');
      let errorBody='';
      const finishQuota=()=>{clearTimeout(deadline);fail('http_429',retrySeconds,quota);};
      response.on('data',chunk=>{
        if(finished)return;
        if(Buffer.byteLength(errorBody)+Buffer.byteLength(chunk)>32768){errorBody='';finishQuota();return;}
        errorBody+=chunk;
      });
      response.on('end',()=>{
        if(finished)return;
        try{
            const details=JSON.parse(errorBody)?.error?.details;
            if(Array.isArray(details)){
              for(const detail of details){
                if(detail?.['@type']==='type.googleapis.com/google.rpc.RetryInfo'&&retrySeconds===undefined){
                  const delay=detail.retryDelay;
                  if(typeof delay==='string'&&/^\d{1,6}(?:\.\d{1,9})?s$/.test(delay))
                    retrySeconds=Math.min(60,Number(delay.slice(0,-1)));
                }
                if(detail?.['@type']==='type.googleapis.com/google.rpc.QuotaFailure'&&Array.isArray(detail.violations)){
                  for(const violation of detail.violations){
                    const id=typeof violation?.quotaId==='string'?violation.quotaId:'';
                    const scope=/per[_ -]?day|daily/i.test(id)?'day':/per[_ -]?minute/i.test(id)?'minute':null;
                    if(!scope||quota?.scope==='day')continue;
                    const raw=violation.quotaValue;
                    const value=typeof raw==='number'?raw:typeof raw==='string'&&/^\d{1,13}$/.test(raw)?Number(raw):NaN;
                    quota={scope,value};
                  }
                }
              }
            }
        }catch{}
        errorBody='';finishQuota();
      });
      response.on('error',finishQuota);
      return;
    }
    if (response.statusCode !== 200) {
      response.resume(); clearTimeout(deadline);
      fail(`http_${response.statusCode}`); return;
    }
    response.setEncoding('utf8');
    let pending = '', total = 0, sawDone = false;
    response.on('data', chunk => {
      if (finished) return;
      total += chunk.length;
      pending = (pending + chunk).replace(/\r\n/g, '\n');
      if (total > 1048576 || pending.length > 131072) { fail('response'); return; }
      let boundary;
      while ((boundary = pending.indexOf('\n\n')) >= 0) {
        const frame = pending.slice(0,boundary); pending = pending.slice(boundary+2);
        const text = frame.split('\n').filter(line=>line.startsWith('data:')).map(line=>line.slice(5).trimStart()).join('\n');
        if (!text) continue;
        if (text === '[DONE]') { sawDone=true; continue; }
        try {
          const chunk = JSON.parse(text);
          // Never relay API error bodies or headers, which can contain auth data.
          if (chunk.error || !Array.isArray(chunk.choices)) { fail('response'); return; }
          emit({type:'chunk', chunk:{choices:chunk.choices, ...(chunk.usage ? {usage:chunk.usage} : {})}});
        } catch { fail('response'); return; }
      }
    });
    response.on('end', () => {
      clearTimeout(deadline);
      if (finished) return;
      if (!sawDone || pending.trim()) { fail('response'); return; }
      finished=true; emit({type:'done'});
    });
    response.on('error', () => {clearTimeout(deadline);fail('network');});
  });
  request.on('timeout', () => {clearTimeout(deadline);fail('timeout');});
  request.on('error', () => {clearTimeout(deadline);fail('network');});
  request.end(body);
}
main().catch(() => fail('request'));
