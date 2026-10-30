// Synthetic exact FFN graph and adversarial consumer/topology variants.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml_npu_bridge.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    const std::string variant(argv[1]);
    const bool prefill = std::string(argv[2]) == "P";
    ggml_cpu_init();
    ggml_init_params params{4 * 1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) return 3;
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 2);
    auto * gate_weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    auto * up_weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 8);
    auto * down_weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 4);
    ggml_set_name(gate_weight, "blk.0.ffn_gate.weight");
    ggml_set_name(up_weight, "blk.0.ffn_up.weight");
    ggml_set_name(down_weight, "blk.0.ffn_down.weight");
    auto * gate = ggml_mul_mat(ctx.get(), gate_weight, input);
    if (variant == "hint") gate->op_params[1] = 1;
    auto * up = ggml_mul_mat(ctx.get(), up_weight, input);
    auto * gate_input = variant == "view" ?
        ggml_view_2d(ctx.get(), gate, gate->ne[0], gate->ne[1], gate->nb[1], 0) : gate;
    auto * glu = variant == "gelu" ? ggml_geglu_split(ctx.get(), gate_input, up) :
                                     ggml_swiglu_split(ctx.get(), gate_input, up);
    if (variant == "swapped") glu->op_params[1] = 1;
    if (variant == "output") ggml_set_output(gate);
    auto * output = ggml_mul_mat(ctx.get(), down_weight, glu);
    if (variant == "consumer") {
        auto * extra = ggml_repeat(ctx.get(), ggml_sum(ctx.get(), gate), output);
        output = ggml_add(ctx.get(), output, extra);
    }
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()),
        ggml_backend_buffer_free);
    if (!buffer) return 4;
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const float inputs[] = {3.0f, -2.0f, 1.5f, -.5f, -.25f, 2.1f, -3.5f, 4.0f};
    for (int i = 0; i < 8; ++i) ggml_set_f32_1d(input, i, inputs[i]);
    for (int i = 0; i < 32; ++i) {
        ggml_set_f32_1d(gate_weight, i, float(i % 13 - 6) / 17.0f);
        ggml_set_f32_1d(up_weight, i, float(i % 11 - 5) / 19.0f);
        ggml_set_f32_1d(down_weight, i, float(i % 7 - 3) / 23.0f);
    }
    constexpr float untouched = 123456.0f;
    for (auto * tensor : {gate, up, glu}) ggml_set_f32(tensor, untouched);
    bool matches = false;
    for (int i = 0; i < graph->n_nodes; ++i)
        if (graph->nodes[i] == gate) matches = pixelforge_ggml_npu_ffn_pattern(graph, i);
    if (variant == "hint") {
        // This rectangular toy weight is not a Hadamard matrix. Verify the
        // dispatch guard without attempting an invalid reference FWHT.
        std::cout << nlohmann::json({{"pattern", matches}}).dump() << '\n';
        return 0;
    }
    pixelforge_ggml_npu_set_prefill_phase(prefill);
    if (ggml_graph_compute_with_ctx(ctx.get(), graph, 2) != GGML_STATUS_SUCCESS) return 5;
    bool intermediates_untouched = true;
    for (auto * tensor : {gate, up, glu})
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i)
            intermediates_untouched &= ggml_get_f32_1d(tensor, i) == untouched;
    if (!pixelforge_ggml_npu_finish()) return 6;
    nlohmann::json values = nlohmann::json::array();
    for (int64_t i = 0; i < ggml_nelements(output); ++i) values.push_back(ggml_get_f32_1d(output, i));
    std::cout << nlohmann::json({{"pattern", matches}, {"intermediates_untouched", intermediates_untouched},
        {"output", values}, {"npu_weight_operations", pixelforge_ggml_npu_requests()},
        {"npu_fused_ffn_operations", pixelforge_ggml_npu_fused_ffns()},
        {"npu_weight_vectors", pixelforge_ggml_npu_vectors()}}).dump() << '\n';
}
