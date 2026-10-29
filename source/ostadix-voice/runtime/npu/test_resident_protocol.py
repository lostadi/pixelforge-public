"""Exercise the real resident-host protocol against a deterministic C API stub.

These tests deliberately do not claim or attempt hardware inference.
"""
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
STUB = r'''
#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "conversation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct LiteRtLmEngineSettings { int unused; };
struct LiteRtLmEngine { int number; };
struct LiteRtLmSessionConfig { int max_output; };
struct LiteRtLmSession { int calls; int max_output; };
struct LiteRtLmInputData { char *text; };
struct LiteRtLmResponses { char *text; };
struct LiteRtLmTokenizeResult { int count; };
LiteRtLmTokenizeResult *litert_lm_engine_tokenize(LiteRtLmEngine*e,const char*p) {
  LiteRtLmTokenizeResult*r=calloc(1,sizeof(*r));r->count=strstr(p,"too many") != NULL?1280:1;return r;
}
size_t litert_lm_tokenize_result_get_num_tokens(const LiteRtLmTokenizeResult*r) { return r->count; }
void litert_lm_tokenize_result_delete(LiteRtLmTokenizeResult*r) { free(r); }
LiteRtLmEngineSettings *litert_lm_engine_settings_create(const char*a,const char*b,const char*c,const char*d) {
  return !strcmp(b,"npu") ? calloc(1,sizeof(LiteRtLmEngineSettings)) : NULL;
}
void litert_lm_engine_settings_set_litert_dispatch_lib_dir(LiteRtLmEngineSettings*s,const char*p) {}
void litert_lm_engine_settings_set_max_num_tokens(LiteRtLmEngineSettings*s,int n) {}
void litert_lm_engine_settings_enable_benchmark(LiteRtLmEngineSettings*s) {}
void litert_lm_engine_settings_delete(LiteRtLmEngineSettings*s) { free(s); }
LiteRtLmEngine *litert_lm_engine_create(const LiteRtLmEngineSettings*s) {
  static int created; LiteRtLmEngine*e=calloc(1,sizeof(*e)); e->number=++created;
  printf("Native library stdout noise\n"); return e;
}
void litert_lm_engine_delete(LiteRtLmEngine*e) { fprintf(stderr,"engine_deleted=%d\n",e->number); free(e); }
LiteRtLmSessionConfig *litert_lm_session_config_create(void) { return calloc(1,sizeof(LiteRtLmSessionConfig)); }
void litert_lm_session_config_set_max_output_tokens(LiteRtLmSessionConfig*c,int n) { c->max_output=n; }
void litert_lm_session_config_set_apply_prompt_template(LiteRtLmSessionConfig*c,bool b) {}
void litert_lm_session_config_delete(LiteRtLmSessionConfig*c) { free(c); }
LiteRtLmSession *litert_lm_engine_create_session(LiteRtLmEngine*e,LiteRtLmSessionConfig*c) {
  if(e->number!=1) return NULL; LiteRtLmSession*s=calloc(1,sizeof(*s)); s->max_output=c->max_output; return s;
}
void litert_lm_session_delete(LiteRtLmSession*s) { free(s); }
LiteRtLmInputData *litert_lm_input_data_create(LiteRtLmInputDataType t,const void*p,size_t n) {
  LiteRtLmInputData*d=calloc(1,sizeof(*d)); d->text=strndup(p,n); return d;
}
void litert_lm_input_data_delete(LiteRtLmInputData*d) { free(d->text);free(d); }
LiteRtLmResponses *litert_lm_session_generate_content(LiteRtLmSession*s,const LiteRtLmInputData*const*p,size_t n) {
  ++s->calls; LiteRtLmResponses*r=calloc(1,sizeof(*r)); size_t len=strlen(p[0]->text)+128; r->text=malloc(len);
  snprintf(r->text,len,"engine=1 session_calls=%d max=%d prompt=%s",s->calls,s->max_output,p[0]->text); return r;
}
const char *litert_lm_responses_get_response_text_at(const LiteRtLmResponses*r,int i) { return r->text; }
void litert_lm_responses_delete(LiteRtLmResponses*r) { free(r->text);free(r); }
LiteRtLmBenchmarkInfo *litert_lm_session_get_benchmark_info(LiteRtLmSession*s) { return NULL; }
double litert_lm_benchmark_info_get_time_to_first_token(const LiteRtLmBenchmarkInfo*i) { return 0; }
int litert_lm_benchmark_info_get_num_decode_turns(const LiteRtLmBenchmarkInfo*i) { return 0; }
int litert_lm_benchmark_info_get_decode_token_count_at(const LiteRtLmBenchmarkInfo*i,int n) { return 0; }
double litert_lm_benchmark_info_get_decode_tokens_per_sec_at(const LiteRtLmBenchmarkInfo*i,int n) { return 0; }
void litert_lm_benchmark_info_delete(LiteRtLmBenchmarkInfo*i) {}

struct LiteRtLmConversationConfig { int max_output; bool thinking; };
struct LiteRtLmThinkingConfig { bool enabled; };
struct LiteRtLmConversationOptionalArgs { int max_output; };
struct LiteRtLmConversation { int calls; int max_output; };
struct LiteRtLmJsonResponse { char *text; };
LiteRtLmConversationConfig *litert_lm_conversation_config_create(void) { return calloc(1,sizeof(LiteRtLmConversationConfig)); }
void litert_lm_conversation_config_delete(LiteRtLmConversationConfig*c) { free(c); }
void litert_lm_conversation_config_set_session_config(LiteRtLmConversationConfig*c,const LiteRtLmSessionConfig*s) { c->max_output=s->max_output; }
LiteRtLmThinkingConfig *litert_lm_thinking_config_create(void) { LiteRtLmThinkingConfig*t=calloc(1,sizeof(*t));t->enabled=true;return t; }
void litert_lm_thinking_config_delete(LiteRtLmThinkingConfig*t) { free(t); }
void litert_lm_thinking_config_set_enable_thinking(LiteRtLmThinkingConfig*t,bool b) { t->enabled=b; }
void litert_lm_conversation_config_set_thinking_config(LiteRtLmConversationConfig*c,const LiteRtLmThinkingConfig*t) { c->thinking=t->enabled; }
void litert_lm_conversation_config_set_enable_constrained_decoding(LiteRtLmConversationConfig*c,bool b) {}
LiteRtLmConversationOptionalArgs *litert_lm_conversation_optional_args_create(void) { return calloc(1,sizeof(LiteRtLmConversationOptionalArgs)); }
void litert_lm_conversation_optional_args_delete(LiteRtLmConversationOptionalArgs*a) { free(a); }
void litert_lm_conversation_optional_args_set_max_output_tokens(LiteRtLmConversationOptionalArgs*a,int n) { a->max_output=n; }
LiteRtLmConversation *litert_lm_conversation_create(LiteRtLmEngine*e,LiteRtLmConversationConfig*c) {
  if(e->number!=1 || c->thinking) return NULL; LiteRtLmConversation*v=calloc(1,sizeof(*v));v->max_output=c->max_output;return v;
}
void litert_lm_conversation_delete(LiteRtLmConversation*c) { free(c); }
const char *litert_lm_conversation_render_preface_to_string(LiteRtLmConversation*c) { abort(); }
const char *litert_lm_conversation_render_message_to_string(LiteRtLmConversation*c,const char*j) { return j; }
LiteRtLmJsonResponse *litert_lm_conversation_send_message(LiteRtLmConversation*c,const char*j,const char*e,const LiteRtLmConversationOptionalArgs*a) {
  if(a->max_output!=c->max_output) return NULL; ++c->calls;
  LiteRtLmJsonResponse*r=calloc(1,sizeof(*r));size_t n=strlen(j)+150;r->text=malloc(n);
  snprintf(r->text,n,"{\"role\":\"assistant\",\"engine\":1,\"session_calls\":%d,\"max\":%d,\"echo\":%s}",c->calls,c->max_output,j);return r;
}
const char *litert_lm_json_response_get_string(const LiteRtLmJsonResponse*r) { return r->text; }
void litert_lm_json_response_delete(LiteRtLmJsonResponse*r) { free(r->text);free(r); }
LiteRtLmBenchmarkInfo *litert_lm_conversation_get_benchmark_info(LiteRtLmConversation*c) { return NULL; }
'''


class ResidentProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get('CC') or shutil.which('cc') or shutil.which('clang')
        if not compiler:
            raise unittest.SkipTest('A native C compiler is required for the offline protocol stub')
        cls.temp = tempfile.TemporaryDirectory(prefix="ostadix-npu-protocol-")
        cls.binary = Path(cls.temp.name) / "resident-stub"
        stub = Path(cls.temp.name) / "stub.c"
        stub.write_text(STUB)
        subprocess.run([
            compiler, "-O1", "-I",
            str(ROOT / "c_api-0.1.0/include"), str(ROOT / "resident.c"),
            str(stub), "-o", str(cls.binary),
        ], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_requests(self, data):
        return subprocess.run([str(self.binary), "/dev/null", "."], input=data,
                              text=True, capture_output=True, timeout=5)

    def test_reuses_engine_but_isolates_conversations(self):
        requests = "".join(f"{n}\t{base64.b64encode(s.encode()).decode()}\n"
                           for n, s in [(1, "hello"), (512, "second\nturn λ")])
        p = self.run_requests(requests)
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[0].startswith("READY\t"))
        for line, (limit, expected) in zip(lines[1:], [(1, "hello"), (512, "second\nturn λ")]):
            tag, elapsed, tokens, encoded = line.split("\t")
            self.assertEqual(tag, "JSONRESULT")
            self.assertEqual(tokens, "-1")
            self.assertGreaterEqual(float(elapsed), 0)
            message = json.loads(base64.b64decode(encoded, validate=True).decode())
            self.assertEqual(message["engine"], 1)
            self.assertEqual(message["session_calls"], 1)
            self.assertEqual(message["max"], limit)
            self.assertEqual(message["echo"]["content"][0]["text"], expected)
        self.assertIn("Native library stdout noise", p.stderr)
        self.assertIn("engine_deleted=1", p.stderr)

    def test_rejects_malformed_or_oversized_then_recovers(self):
        invalid = ["0\tdA==", "513\tdA==", "1\tdB==", "1\tAA==", "1\t", "bad", "1\t" + "A" * 40000]
        p = self.run_requests("\n".join(invalid) + "\n1\tb2s=\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = p.stdout.splitlines()
        self.assertEqual(sum(x.startswith("ERROR\t") for x in lines), len(invalid))
        self.assertTrue(lines[-1].startswith("JSONRESULT\t"))

    def test_prompt_size_boundary(self):
        encoded = base64.b64encode(b"x" * 24000).decode()
        p = self.run_requests("1\t" + encoded + "\n")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertTrue(p.stdout.splitlines()[-1].startswith("JSONRESULT\t"))
        too_long = base64.b64encode(b"x" * 24001).decode()
        p = self.run_requests("1\t" + too_long + "\n")
        self.assertTrue(p.stdout.splitlines()[-1].startswith("ERROR\t"))

    def test_context_checked_before_generation(self):
        p = self.run_requests("1\t" + base64.b64encode(b"too many").decode() + "\n")
        last = p.stdout.splitlines()[-1]
        self.assertTrue(last.startswith("ERROR\t"))
        self.assertIn("context", base64.b64decode(last.split("\t")[1]).decode())

    def test_message_json_escapes_quotes_backslashes_and_controls(self):
        prompt = 'quoted "word" \\ path\nnext\tcolumn\rreturn\x01control λ'
        p = self.run_requests("16\t" + base64.b64encode(prompt.encode()).decode() + "\n")
        last = p.stdout.splitlines()[-1]
        self.assertTrue(last.startswith("JSONRESULT\t"), p.stderr)
        message = json.loads(base64.b64decode(last.split("\t")[3]))
        self.assertEqual(message['echo']['content'][0]['text'], prompt)

    def test_parent_death_terminates_only_host(self):
        code = """
import subprocess,sys,os
p=subprocess.Popen([sys.argv[1],'/dev/null','.'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
assert p.stdout.readline().startswith('READY\\t')
print(p.pid,flush=True)
os._exit(0)
"""
        p = subprocess.run([sys.executable, "-c", code, str(self.binary)],
                           capture_output=True, text=True, timeout=5, check=True)
        pid = int(p.stdout.strip())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
            except FileNotFoundError:
                return
            if stat.rsplit(") ", 1)[1].split()[0] == "Z":
                return
            time.sleep(.02)
        self.fail("resident host survived parent death")


if __name__ == "__main__":
    unittest.main()
