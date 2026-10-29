#pragma once
#include <stdbool.h>
#include <stdint.h>

struct ggml_compute_params;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif
bool pixelforge_ggml_npu_enabled(void);
void pixelforge_ggml_npu_set_cpu_prefill(bool enabled);
uint64_t pixelforge_ggml_npu_cpu_prefill_matmuls(void);
uint64_t pixelforge_ggml_npu_cpu_placed_matmuls(void);
const char * pixelforge_ggml_npu_selected_tensors_csv(void);
bool pixelforge_ggml_npu_compute(const struct ggml_compute_params * params,
                               struct ggml_tensor * dst);
bool pixelforge_ggml_npu_finish(void);
uint64_t pixelforge_ggml_npu_requests(void);
uint64_t pixelforge_ggml_npu_cpu_matmuls(void);
#ifdef __cplusplus
}
#endif
