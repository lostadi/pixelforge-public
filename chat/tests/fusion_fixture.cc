// Exercise the pinned ggml RMS_NORM + MUL graph fusion without NPU hardware.
// A fused graph writes the final tensor directly and leaves the private RMS
// intermediate untouched. This distinguishes real fusion from equal outputs
// produced by two separate kernels.
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

int main() {
    ggml_cpu_init();
    ggml_init_params params = {8 * 1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) return 1;
    constexpr int columns = 16, rows = 4;
    constexpr float untouched = 123456.0f;
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, columns, rows);
    auto * weight = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, columns);
    auto * norm = ggml_rms_norm(ctx.get(), input, 1e-6f);
    auto * output = ggml_mul(ctx.get(), norm, weight);
    for (int i = 0; i < columns * rows; ++i) {
        ggml_set_f32_1d(input, i, float((i * 7) % 29 - 14) / 13.0f);
    }
    for (int i = 0; i < columns; ++i) {
        ggml_set_f32_1d(weight, i, float(i + 1) / 11.0f);
    }
    ggml_set_f32(norm, untouched);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, 2) != GGML_STATUS_SUCCESS) return 2;
    bool fused = true;
    for (int i = 0; i < columns * rows; ++i) {
        fused = fused && ggml_get_f32_1d(norm, i) == untouched;
    }
    std::cout << "{\"fused\":" << (fused ? "true" : "false") << ",\"output\":[";
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (int i = 0; i < columns * rows; ++i) {
        if (i) std::cout << ',';
        std::cout << ggml_get_f32_1d(output, i);
    }
    std::cout << "]}\n";
    return std::cout ? 0 : 3;
}
