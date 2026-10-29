// Persistent process host for the official LiteRT-LM C API.
// Requests: MAX_OUTPUT_TOKENS<TAB>BASE64_UTF8_PROMPT<LF>
// Replies: READY<TAB>PID<TAB>INIT_SECONDS,
//          JSONRESULT<TAB>SECONDS<TAB>COMPLETION_TOKENS<TAB>BASE64_JSON,
//          or ERROR<TAB>BASE64_TEXT. Native runtime diagnostics go to stderr.
// Backend=npu is requested. CPU auxiliary operations may still be used.
#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "conversation.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

enum { MAX_PROMPT = 24000, MAX_LINE = 32100, MAX_RESPONSE = 262144 };
static FILE *protocol;
static const char alphabet[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

static int parse_number(const char *s, long low, long high, int *out) {
  char *end;
  errno = 0;
  long n = strtol(s, &end, 10);
  if (!*s || *end || errno || n < low || n > high) return 0;
  *out = (int)n;
  return 1;
}

static void print_b64(const char *s, size_t n) {
  for (size_t i = 0; i < n; i += 3) {
    unsigned a = (unsigned char)s[i];
    unsigned b = i + 1 < n ? (unsigned char)s[i + 1] : 0;
    unsigned c = i + 2 < n ? (unsigned char)s[i + 2] : 0;
    fputc(alphabet[a >> 2], protocol);
    fputc(alphabet[((a & 3) << 4) | (b >> 4)], protocol);
    fputc(i + 1 < n ? alphabet[((b & 15) << 2) | (c >> 6)] : '=', protocol);
    fputc(i + 2 < n ? alphabet[c & 63] : '=', protocol);
  }
}

static int digit(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}

static int decode_b64(const char *s, char *out, size_t *out_size) {
  size_t n = strlen(s), used = 0;
  if (!n || n % 4 || n > ((MAX_PROMPT + 2) / 3) * 4) return 0;
  for (size_t i = 0; i < n; i += 4) {
    int a = digit(s[i]), b = digit(s[i + 1]);
    int c = s[i + 2] == '=' ? 0 : digit(s[i + 2]);
    int d = s[i + 3] == '=' ? 0 : digit(s[i + 3]);
    int padding = (s[i + 2] == '=') + (s[i + 3] == '=');
    if (a < 0 || b < 0 || c < 0 || d < 0 ||
        (padding && i + 4 != n) ||
        (s[i + 2] == '=' && s[i + 3] != '=') ||
        (padding == 2 && (b & 15)) || (padding == 1 && (c & 3))) return 0;
    unsigned value = ((unsigned)a << 18) | ((unsigned)b << 12) |
                     ((unsigned)c << 6) | (unsigned)d;
    if (used + 3 - padding > MAX_PROMPT) return 0;
    out[used++] = value >> 16;
    if (padding < 2) out[used++] = value >> 8;
    if (padding < 1) out[used++] = value;
  }
  if (!used || memchr(out, 0, used)) return 0;
  out[used] = 0;
  *out_size = used;
  return 1;
}

static void error_reply(const char *message) {
  fputs("ERROR\t", protocol);
  print_b64(message, strlen(message));
  fputc('\n', protocol);
  fflush(protocol);
}

static char *message_json(const char *text, size_t size) {
  const char prefix[] = "{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"";
  const char suffix[] = "\"}]}";
  char *json = malloc(sizeof(prefix) + size * 6 + sizeof(suffix));
  if (!json) return NULL;
  char *out = json;
  memcpy(out, prefix, sizeof(prefix) - 1); out += sizeof(prefix) - 1;
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0; i < size; ++i) {
    unsigned char c = text[i];
    if (c == '"' || c == '\\') { *out++ = '\\'; *out++ = c; }
    else if (c < 32) {
      memcpy(out, "\\u00", 4); out += 4;
      *out++ = hex[c >> 4]; *out++ = hex[c & 15];
    } else *out++ = c;
  }
  memcpy(out, suffix, sizeof(suffix));
  return json;
}

static void generate(LiteRtLmEngine *engine, int context, int max_output,
                     const char *prompt, size_t prompt_size) {
  double start = now(), preflight_done = 0, generation_done = 0;
  LiteRtLmSessionConfig *session_config = NULL;
  LiteRtLmConversationConfig *config = NULL;
  LiteRtLmThinkingConfig *thinking = NULL;
  LiteRtLmConversation *conversation = NULL;
  LiteRtLmConversationOptionalArgs *args = NULL;
  LiteRtLmJsonResponse *response = NULL;
  LiteRtLmBenchmarkInfo *info = NULL;
  LiteRtLmTokenizeResult *tokens = NULL;
  char *json = message_json(prompt, prompt_size);
  const char *failure = NULL;
  int completion_tokens = -1;
  if (!json) { failure = "Message allocation failed"; goto done; }
  session_config = litert_lm_session_config_create();
  config = litert_lm_conversation_config_create();
  thinking = litert_lm_thinking_config_create();
  args = litert_lm_conversation_optional_args_create();
  if (!session_config || !config || !thinking || !args) {
    failure = "Conversation configuration failed"; goto done;
  }
  // Conversation owns model-specific templating; legacy Session templating
  // cannot represent Gemma 4's channels correctly.
  litert_lm_session_config_set_max_output_tokens(session_config, max_output);
  litert_lm_session_config_set_apply_prompt_template(session_config, false);
  litert_lm_conversation_config_set_session_config(config, session_config);
  litert_lm_thinking_config_set_enable_thinking(thinking, false);
  litert_lm_conversation_config_set_thinking_config(config, thinking);
  litert_lm_conversation_config_set_enable_constrained_decoding(config, false);
  litert_lm_conversation_optional_args_set_max_output_tokens(args, max_output);
  conversation = litert_lm_conversation_create(engine, config);
  if (!conversation) { failure = "Conversation creation failed"; goto done; }
  // A fresh conversation's message renderer already includes its preface.
  // Rendering the empty preface separately breaks templates that read
  // messages[0], and would double-count preface tokens when present.
  const char *message = litert_lm_conversation_render_message_to_string(conversation, json);
  if (!message) { failure = "Model prompt rendering failed"; goto done; }
  size_t message_size = strnlen(message, MAX_RESPONSE + 1);
  if (message_size > MAX_RESPONSE) {
    failure = "Rendered prompt exceeds size limit"; goto done;
  }
  // Renderer-owned text stays valid until the next render or conversation delete.
  tokens = litert_lm_engine_tokenize(engine, message);
  if (!tokens) { failure = "Prompt tokenization failed"; goto done; }
  size_t count = litert_lm_tokenize_result_get_num_tokens(tokens);
  // Count the model-rendered template and reserve a small margin for any
  // additional BOS/EOS/control tokens inserted by the native input pipeline.
  if (count > (size_t)context || count + 8 + max_output > (size_t)context) {
    failure = "Task exceeds NPU context including rendered template, output and 8-token control allowance; shorten the task or reduce max_tokens";
    goto done;
  }
  preflight_done = now();
  response = litert_lm_conversation_send_message(conversation, json, NULL, args);
  generation_done = now();
  if (!response) { failure = "Generation failed; see native runtime log"; goto done; }
  info = litert_lm_conversation_get_benchmark_info(conversation);
  if (info) {
    int turns = litert_lm_benchmark_info_get_num_decode_turns(info);
    if (turns > 0) {
      completion_tokens = 0;
      for (int i = 0; i < turns; ++i) {
        int decoded = litert_lm_benchmark_info_get_decode_token_count_at(info, i);
        if (decoded < 0 || decoded > 32768) { completion_tokens = -1; break; }
        completion_tokens += decoded;
      }
    }
  }
  const char *result = litert_lm_json_response_get_string(response);
  if (!result) { failure = "Generation returned no response candidate"; goto done; }
  size_t result_size = strnlen(result, MAX_RESPONSE + 1);
  if (result_size > MAX_RESPONSE) { failure = "Generated response exceeds size limit"; goto done; }
  fprintf(protocol, "JSONRESULT\t%.6f\t%d\t", now() - start, completion_tokens);
  print_b64(result, result_size);
  fputc('\n', protocol); fflush(protocol);
  fprintf(stderr, "host_preflight_seconds=%.6f generation_seconds=%.6f prompt_tokens=%zu\n",
          preflight_done - start, generation_done - preflight_done, count);
  if (info) {
    fprintf(stderr, "request_pid=%d time_to_first_token_seconds=%.6f completion_tokens=%d\n",
            getpid(), litert_lm_benchmark_info_get_time_to_first_token(info), completion_tokens);
    int turns = litert_lm_benchmark_info_get_num_decode_turns(info);
    for (int i = 0; i < turns; ++i)
      fprintf(stderr, "decode_turn=%d tokens=%d tokens_per_second=%.3f\n", i,
              litert_lm_benchmark_info_get_decode_token_count_at(info, i),
              litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, i));
  }
done:
  if (failure) error_reply(failure);
  if (info) litert_lm_benchmark_info_delete(info);
  if (tokens) litert_lm_tokenize_result_delete(tokens);
  if (response) litert_lm_json_response_delete(response);
  if (conversation) litert_lm_conversation_delete(conversation);
  if (args) litert_lm_conversation_optional_args_delete(args);
  if (thinking) litert_lm_thinking_config_delete(thinking);
  if (config) litert_lm_conversation_config_delete(config);
  if (session_config) litert_lm_session_config_delete(session_config);
  free(json);
}

int main(int argc, char **argv) {
  protocol = stdout;
  if (argc == 2 && !strcmp(argv[1], "--protocol-test")) {
    char out[MAX_PROMPT + 1]; size_t n;
    if (!decode_b64("dGVzdA==", out, &n) || n != 4 || strcmp(out, "test") ||
        decode_b64("dGVzdB==", out, &n) || decode_b64("AA==", out, &n) ||
        decode_b64("", out, &n)) return 1;
    print_b64("test", 4); fputc('\n', protocol); return 0;
  }
  if (argc < 3 || argc > 4) {
    fprintf(stderr, "Usage: %s MODEL.litertlm DISPATCH_DIR [MAX_CONTEXT_TOKENS]\n",
            argv[0]); return 2;
  }
  int context = 1280;
  if (argc == 4 && !parse_number(argv[3], 128, 32768, &context)) {
    fprintf(stderr, "MAX_CONTEXT_TOKENS must be 128..32768\n"); return 2;
  }
  if (access(argv[1], R_OK)) { perror("model"); return 3; }
  pid_t parent = getppid();
  if (parent == 1 || prctl(PR_SET_PDEATHSIG, SIGKILL) || getppid() != parent) {
    fprintf(stderr, "Cannot establish parent death cleanup\n"); return 4;
  }
  // Keep protocol separate from native libraries that print directly to stdout.
  int protocol_fd = dup(STDOUT_FILENO);
  if (protocol_fd < 0 || !(protocol = fdopen(protocol_fd, "w")) ||
      dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    fprintf(stderr, "Cannot establish protocol output\n"); return 4;
  }
  setvbuf(protocol, NULL, _IOLBF, 0);
  double start = now();
  LiteRtLmEngineSettings *settings =
      litert_lm_engine_settings_create(argv[1], "npu", NULL, NULL);
  if (!settings) { error_reply("NPU engine settings failed"); return 5; }
  litert_lm_engine_settings_set_litert_dispatch_lib_dir(settings, argv[2]);
  litert_lm_engine_settings_set_max_num_tokens(settings, context);
  litert_lm_engine_settings_enable_benchmark(settings);
  LiteRtLmEngine *engine = litert_lm_engine_create(settings);
  litert_lm_engine_settings_delete(settings);
  if (!engine) { error_reply("NPU engine creation failed"); return 6; }
  fprintf(protocol, "READY\t%d\t%.6f\n", getpid(), now() - start);
  char line[MAX_LINE], prompt[MAX_PROMPT + 1];
  while (fgets(line, sizeof(line), stdin)) {
    size_t len = strlen(line);
    if (!len || line[len - 1] != '\n') {
      int c;
      while ((c = getchar()) != '\n' && c != EOF) {}
      error_reply("Request line exceeds limit or is incomplete");
      continue;
    }
    line[--len] = 0;
    if (len && line[len - 1] == '\r') line[--len] = 0;
    char *tab = strchr(line, '\t');
    int max_output; size_t prompt_size;
    if (!tab) { error_reply("Expected token limit and base64 prompt"); continue; }
    *tab++ = 0;
    if (!parse_number(line, 1, 512, &max_output) ||
        !decode_b64(tab, prompt, &prompt_size)) {
      error_reply("Invalid token limit or base64 prompt"); continue;
    }
    generate(engine, context, max_output, prompt, prompt_size);
  }
  litert_lm_engine_delete(engine);
  return ferror(stdin) ? 7 : 0;
}
