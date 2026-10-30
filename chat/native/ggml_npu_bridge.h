#pragma once
#define PIXELFORGE_GGML_FULL_FFN_HOOK 1
#include <stdbool.h>
#include <stdint.h>

struct ggml_compute_params;
struct ggml_tensor;
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif
bool pixelforge_ggml_npu_enabled(void);
void pixelforge_ggml_npu_set_cpu_prefill(bool enabled);
void pixelforge_ggml_npu_set_prefill_phase(bool enabled);
uint64_t pixelforge_ggml_npu_cpu_prefill_matmuls(void);
uint64_t pixelforge_ggml_npu_cpu_placed_matmuls(void);
const char * pixelforge_ggml_npu_selected_tensors_csv(void);
const char * pixelforge_ggml_npu_prefill_selected_tensors_csv(void);
bool pixelforge_ggml_npu_compute(const struct ggml_compute_params * params,
                               struct ggml_tensor * dst);
// Exact dense SwiGLU FFN topology only; configuration/placement checked by fuse.
bool pixelforge_ggml_npu_ffn_pattern(const struct ggml_cgraph * graph, int node);
int pixelforge_ggml_npu_fuse(const struct ggml_compute_params * params,
                            const struct ggml_cgraph * graph, int node);
bool pixelforge_ggml_npu_finish(void);
uint64_t pixelforge_ggml_npu_requests(void);
uint64_t pixelforge_ggml_npu_cpu_matmuls(void);
uint64_t pixelforge_ggml_npu_vectors(void);
uint64_t pixelforge_ggml_npu_nanoseconds(void);
uint64_t pixelforge_ggml_npu_fused_ffns(void);
const char * pixelforge_ggml_npu_full_ffn_tensors_csv(void);
const char * pixelforge_ggml_npu_full_ffn_phases(void);
#ifdef __cplusplus
}
#endif
