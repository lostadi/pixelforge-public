// Minimal deterministic architecture-engine runner; GGML dense ops use the
// optional PixelForge bridge. MIT: see LICENSE.ggml.
#include "llama.h"
#include "ggml_npu_bridge.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

int main(int argc, char **argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: pixelforge-gguf MODEL PROMPT TOKENS THREADS");
        const std::string prompt = argv[2];
        const int count = std::stoi(argv[3]), threads = std::stoi(argv[4]);
        if (count < 1 || count > 128 || threads < 1 || threads > 32 || prompt.size() > 65536)
            throw std::runtime_error("Invalid tokens/threads/prompt length");
        const auto began = std::chrono::steady_clock::now();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_extra_bufts = false;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("GGUF model loading failed");
        if (!llama_model_has_decoder(model.get()) || llama_model_has_encoder(model.get()))
            throw std::runtime_error("This runner currently supports decoder-only text generation");
        const auto *vocab = llama_model_get_vocab(model.get());
        int n = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        if (n < 1 || n + count > 1024) throw std::runtime_error("Prompt is empty or exceeds context limit");
        std::vector<llama_token> input(n);
        if (llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), n, true, true) != n)
            throw std::runtime_error("GGUF tokenization failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max(256, n + count);
        cp.n_batch = std::max(64, n);
        cp.n_ubatch = cp.n_batch;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = cp.op_offload = false;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("GGUF context initialization failed");
        json steps = json::array(), output_ids = json::array();
        std::string text;
        const auto started = std::chrono::steady_clock::now();
        for (int step = 0; step < count; ++step) {
            const auto before = std::chrono::steady_clock::now();
            if (llama_decode(ctx.get(), llama_batch_get_one(input.data(), input.size())) != 0)
                throw std::runtime_error("GGUF decode failed");
            float *logits = llama_get_logits_ith(ctx.get(), -1);
            const int size = llama_vocab_n_tokens(vocab);
            if (!logits || size < 1) throw std::runtime_error("Missing model logits");
            for (int i = 0; i < size; ++i)
                if (!std::isfinite(logits[i])) throw std::runtime_error("Nonfinite model logits");
            std::vector<int> order(size);
            std::iota(order.begin(), order.end(), 0);
            const int top = std::min(10, size);
            std::partial_sort(order.begin(), order.begin() + top, order.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            const llama_token token = order[0];
            json values = json::array();
            for (int i = 0; i < top; ++i)
                values.push_back({{"token", order[i]}, {"logit", logits[order[i]]}});
            output_ids.push_back(token);
            const bool eos = llama_vocab_is_eog(vocab, token);
            std::vector<char> piece(256);
            int length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
            if (length < 0) {
                piece.resize(-length);
                length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
            }
            if (length < 0) throw std::runtime_error("Token decoding failed");
            if (!eos) text.append(piece.data(), length);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();
            json step_report = {{"step", step}, {"token", token}, {"eos", eos},
                                {"seconds", elapsed}, {"top_logits", values}};
            if (size <= 4096) step_report["logits"] = std::vector<float>(logits, logits + size);
            steps.push_back(step_report);
            std::cerr << "PIXELFORGE_TOKEN " << token << " seconds=" << elapsed << std::endl;
            if (eos) break;
            input.assign(1, token);
        }
        if (!pixelforge_ggml_npu_finish())
            throw std::runtime_error("NPU broker completion or cleanup failed");
        json report = {{"success", true}, {"prompt_tokens", n}, {"generated_tokens", output_ids},
                       {"text", text}, {"steps", steps},
                       {"engine", "llama.cpp"}, {"upstream_commit", "254b177307b14d055bbe4c050c9c441a1e06d8a4"},
                       {"dense_npu_requested", std::getenv("PIXELFORGE_GGML_BROKER") != nullptr},
                       {"npu_weight_matmul_operations", pixelforge_ggml_npu_requests()},
                       {"cpu_dynamic_matmul_operations", pixelforge_ggml_npu_cpu_matmuls()},
                       {"generation_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()},
                       {"total_seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count()}};
        std::cout << report.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "PixelForge: " << error.what() << std::endl;
        return 1;
    }
}
