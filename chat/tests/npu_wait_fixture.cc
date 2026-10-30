// Real ggml threadpools and graphs; the Python broker delays deterministic math.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml_npu_bridge.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cmath>
#include <ctime>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

class Toy {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer;
    ggml_tensor * input, * gate, * up, * glu, * output;
    ggml_cgraph * graph;
    bool full;
public:
    explicit Toy(bool fused):ctx(nullptr, ggml_free),buffer(nullptr, ggml_backend_buffer_free),full(fused) {
        ggml_init_params params{1024 * 1024, nullptr, true};
        ctx.reset(ggml_init(params));
        if (!ctx) throw std::runtime_error("fixture context");
        input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 1);
        auto * gw = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        auto * uw = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        auto * dw = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 4);
        ggml_set_name(gw, "blk.0.ffn_gate.weight");
        ggml_set_name(uw, "blk.0.ffn_up.weight");
        ggml_set_name(dw, "blk.0.ffn_down.weight");
        gate = ggml_mul_mat(ctx.get(), gw, input);
        up = ggml_mul_mat(ctx.get(), uw, input);
        glu = ggml_swiglu_split(ctx.get(), gate, up);
        output = full ? ggml_mul_mat(ctx.get(), dw, glu) : gate;
        graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_cpu_buffer_type()));
        if (!buffer) throw std::runtime_error("fixture buffers");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto * weight : {gw, uw, dw})
            for (int i = 0; i < 16; ++i) ggml_set_f32_1d(weight, i, i / 4 == i % 4 ? 1.f : 0.f);
    }
    void run(int nth, int rounds) {
        auto params = ggml_threadpool_params_default(nth);
        params.poll = 0; // This fixture measures active-hook waiting, not idle-pool polling.
        std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(
            ggml_threadpool_new(&params), ggml_threadpool_free);
        if (!pool) throw std::runtime_error("fixture pool");
        auto plan = ggml_graph_plan(graph, nth, pool.get());
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.data();
        for (int round = 0; round < rounds; ++round) {
            const float x = .5f + round * .125f;
            ggml_set_f32(input, x);
            if (full) for (auto * intermediate : {gate, up, glu}) ggml_set_f32(intermediate, 123456.f);
            if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) throw std::runtime_error("fixture graph");
            const float expected = full ? x * x / (1.f + std::exp(-x)) : x;
            for (int i = 0; i < 4; ++i)
                if (std::abs(ggml_get_f32_1d(output, i) - expected) > 1e-5f)
                    throw std::runtime_error("rendezvous returned before output publication");
            if (full) for (auto * intermediate : {gate, up, glu})
                if (ggml_get_f32_1d(intermediate, 0) != 123456.f)
                    throw std::runtime_error("full FFN executed CPU intermediates");
        }
    }
    void direct() {
        ggml_compute_params params{};
        params.nth = 1; // Direct callers may have no threadpool at all.
        for (int round = 0; round < 4; ++round) {
            const float value = .25f + round * .125f;
            ggml_set_f32(input, value);
            if (!pixelforge_ggml_npu_compute(&params, gate)) throw std::runtime_error("direct hook declined");
            for (int i = 0; i < 4; ++i)
                if (ggml_get_f32_1d(gate, i) != value) throw std::runtime_error("direct output publication");
        }
    }
};

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    ggml_cpu_init();
    const std::string mode(argv[1]);
    const auto began = std::chrono::steady_clock::now();
    const auto cpu = std::clock();
    if (mode == "single") {
        Toy first(false), second(false);
        std::thread a([&] { first.direct(); });
        std::thread b([&] { second.direct(); });
        a.join();b.join();
    } else if (mode == "pools") {
        Toy matrix(false), fused(true);
        std::thread first([&] { matrix.run(4, 4); });
        std::thread second([&] { fused.run(2, 4); });
        first.join();second.join();
    } else if (mode == "matrix" || mode == "full") {
        Toy toy(mode == "full");
        toy.run(4, 5);
        // A new pool can reuse the previous pool's address with another size.
        toy.run(2, 3);
    } else return 3;
    const double cpu_seconds = double(std::clock() - cpu) / CLOCKS_PER_SEC;
    const double wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
    if (!pixelforge_ggml_npu_finish()) return 4;
    std::cout << nlohmann::json({{"cpu_seconds", cpu_seconds}, {"wall_seconds", wall_seconds},
        {"npu_weight_operations", pixelforge_ggml_npu_requests()},
        {"npu_fused_ffn_operations", pixelforge_ggml_npu_fused_ffns()}}).dump() << '\n';
}
