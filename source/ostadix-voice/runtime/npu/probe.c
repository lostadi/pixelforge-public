// Minimal host for Google's published LiteRT-LM C API. This is a diagnostic,
// not evidence of NPU inference unless a real model successfully generates.
#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "--probe")) {
    LiteRtLmEngineSettings *s = litert_lm_engine_settings_create(
        "/nonexistent/probe.litertlm", "npu", NULL, NULL);
    printf("runtime_loaded=true\nnpu_settings_accepted=%s\ninference_run=false\n",
           s ? "true" : "false");
    if (s) litert_lm_engine_settings_delete(s);
    return s ? 0 : 1;
  }
  if (argc < 4 || argc > 5) {
    fprintf(stderr, "Usage: %s --probe\n"
                    "       %s MODEL.litertlm DISPATCH_DIR PROMPT [HOLD_SECONDS]\n"
                    "Requests the npu backend; runtime may use CPU auxiliary operations.\n",
            argv[0], argv[0]);
    return 2;
  }
  if (access(argv[1], R_OK)) { perror("model"); return 3; }
  int hold = argc == 5 ? atoi(argv[4]) : 0;
  if (hold < 0 || hold > 300) {
    fprintf(stderr, "HOLD_SECONDS must be between 0 and 300.\n"); return 2;
  }
  double start = now();
  LiteRtLmEngineSettings *settings =
      litert_lm_engine_settings_create(argv[1], "npu", NULL, NULL);
  if (!settings) { fprintf(stderr, "NPU settings failed\n"); return 4; }
  litert_lm_engine_settings_set_litert_dispatch_lib_dir(settings, argv[2]);
  litert_lm_engine_settings_set_max_num_tokens(settings, 1280);
  litert_lm_engine_settings_enable_benchmark(settings);
  LiteRtLmEngine *engine = litert_lm_engine_create(settings);
  litert_lm_engine_settings_delete(settings);
  if (!engine) { fprintf(stderr, "NPU engine creation failed\n"); return 5; }
  fprintf(stderr, "engine_created=true backend_requested=npu pid=%d init_seconds=%.3f\n",
          getpid(), now() - start);
  LiteRtLmSessionConfig *config = litert_lm_session_config_create();
  litert_lm_session_config_set_max_output_tokens(config, 64);
  litert_lm_session_config_set_apply_prompt_template(config, true);
  LiteRtLmSession *session = litert_lm_engine_create_session(engine, config);
  litert_lm_session_config_delete(config);
  if (!session) {
    fprintf(stderr, "Session creation failed\n");
    litert_lm_engine_delete(engine); return 6;
  }
  LiteRtLmInputData *input = litert_lm_input_data_create(
      kLiteRtLmInputDataTypeText, argv[3], strlen(argv[3]));
  const LiteRtLmInputData *inputs[] = {input};
  start = now();
  LiteRtLmResponses *response =
      litert_lm_session_generate_content(session, inputs, 1);
  litert_lm_input_data_delete(input);
  int rc = 0;
  if (!response) { fprintf(stderr, "Generation failed\n"); rc = 7; }
  else {
    const char *text = litert_lm_responses_get_response_text_at(response, 0);
    if (!text) { fprintf(stderr, "No response candidate\n"); rc = 8; }
    else { puts(text); fflush(stdout); }
    fprintf(stderr, "generation_seconds=%.3f\n", now() - start);
    litert_lm_responses_delete(response);
    LiteRtLmBenchmarkInfo *info = litert_lm_session_get_benchmark_info(session);
    if (info) {
      fprintf(stderr, "time_to_first_token_seconds=%.6f\n",
              litert_lm_benchmark_info_get_time_to_first_token(info));
      int turns = litert_lm_benchmark_info_get_num_decode_turns(info);
      for (int i = 0; i < turns; ++i)
        fprintf(stderr, "decode_turn=%d tokens=%d tokens_per_second=%.3f\n", i,
                litert_lm_benchmark_info_get_decode_token_count_at(info, i),
                litert_lm_benchmark_info_get_decode_tokens_per_sec_at(info, i));
      litert_lm_benchmark_info_delete(info);
    }
  }
  if (!rc && hold) {
    fprintf(stderr, "Holding engine resident for %d seconds for inspection.\n", hold);
    sleep(hold);
  }
  litert_lm_session_delete(session);
  litert_lm_engine_delete(engine);
  return rc;
}
