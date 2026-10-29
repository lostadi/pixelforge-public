// Resident architecture/tokenizer owner. The host supplies the model's rendered
// template and owns tool execution; this process only performs inference.
#include "llama.h"
#include "ggml_npu_bridge.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr size_t MAX_FRAME = 262144;

static void emit(const json & event) {
    std::cout << event.dump(-1, ' ', false, json::error_handler_t::replace) << std::endl;
    if (!std::cout) throw std::runtime_error("Conversation output disconnected");
}

static bool frame(std::string & value) {
    value.clear();
    char byte;
    while (std::cin.get(byte)) {
        if (byte == '\n') return true;
        if (value.size() == MAX_FRAME) throw std::runtime_error("Request frame exceeds limit");
        value += byte;
    }
    if (!value.empty()) throw std::runtime_error("Incomplete request frame");
    return false;
}

// Token boundaries can split a UTF-8 character. Hold the incomplete suffix
// until the following token rather than replacing each fragment separately.
static size_t utf8_prefix(const std::string & value) {
    size_t at = 0;
    while (at < value.size()) {
        const auto first = static_cast<unsigned char>(value[at]);
        size_t length = first < 0x80 ? 1 : first >= 0xC2 && first <= 0xDF ? 2 :
                        first >= 0xE0 && first <= 0xEF ? 3 : first >= 0xF0 && first <= 0xF4 ? 4 : 1;
        if (at + length > value.size()) break;
        at += length;
    }
    return at;
}

static int integer(const json & request, const char * key, int fallback, int minimum, int maximum) {
    if (!request.contains(key)) return fallback;
    if (!request[key].is_number_integer()) throw std::runtime_error(std::string(key) + " must be an integer");
    const auto value = request[key].get<int64_t>();
    if (value < minimum || value > maximum) throw std::runtime_error(std::string(key) + " exceeds supported bounds");
    return static_cast<int>(value);
}

static json selected_tensors() {
    const std::string csv = pixelforge_ggml_npu_selected_tensors_csv();
    if (csv.empty()) return nullptr; // Null denotes all planned weights when NPU enabled.
    json names = json::array();
    std::istringstream source(csv);
    std::string name;
    while (std::getline(source, name, ',')) names.push_back(name);
    return names;
}

static void generate(llama_model * model, int threads, const json & request) {
    if (!request.is_object()) throw std::runtime_error("Request must be an object");
    for (const auto & item : request.items()) {
        if (item.key() != "id" && item.key() != "prompt" && item.key() != "max_tokens" &&
            item.key() != "context_size" && item.key() != "prefill_cpu")
            throw std::runtime_error("Unknown request field");
    }
    if (!request.contains("id") || !request["id"].is_string() ||
        request["id"].get_ref<const std::string &>().empty() || request["id"].get_ref<const std::string &>().size() > 128)
        throw std::runtime_error("Request id must contain 1 to 128 bytes");
    if (!request.contains("prompt") || !request["prompt"].is_string()) throw std::runtime_error("Rendered prompt is required");
    const auto & prompt = request["prompt"].get_ref<const std::string &>();
    if (prompt.empty() || prompt.size() > 131072 || prompt.find('\0') != std::string::npos)
        throw std::runtime_error("Rendered prompt exceeds supported bounds");
    const int tokens = integer(request, "max_tokens", 512, 1, 512);
    const int context = integer(request, "context_size", 8192, 256, 32768);
    if (request.contains("prefill_cpu") && !request["prefill_cpu"].is_boolean())
        throw std::runtime_error("prefill_cpu must be boolean");
    const bool prefill_cpu = request.value("prefill_cpu", false);
    const std::string id = request["id"];
    const auto * vocab = llama_model_get_vocab(model);
    // The supplied Jinja output already includes its own BOS/control tokens.
    int n = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, false, true);
    if (n < 1 || n + tokens > context) throw std::runtime_error("Prompt plus output exceeds context_size");
    std::vector<llama_token> input(n);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), n, false, true) != n)
        throw std::runtime_error("Rendered prompt tokenization failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = cp.op_offload = false;
    const auto context_began = Clock::now();
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, cp), llama_free);
    if (!ctx) throw std::runtime_error("Conversation context initialization failed");
    const double context_init_seconds = std::chrono::duration<double>(Clock::now() - context_began).count();
    const uint64_t npu_before = pixelforge_ggml_npu_requests();
    const uint64_t cpu_before = pixelforge_ggml_npu_cpu_matmuls();
    const uint64_t prefill_before = pixelforge_ggml_npu_cpu_prefill_matmuls();
    const uint64_t placed_before = pixelforge_ggml_npu_cpu_placed_matmuls();
    const auto began = Clock::now();
    pixelforge_ggml_npu_set_cpu_prefill(prefill_cpu);
    for (int offset = 0; offset < n; offset += 512) {
        if (llama_decode(ctx.get(), llama_batch_get_one(input.data() + offset, std::min(512, n - offset))) != 0)
            throw std::runtime_error("Conversation prefill failed");
    }
    pixelforge_ggml_npu_set_cpu_prefill(false);
    const auto prefilled = Clock::now();
    std::string text, pending;
    json output_ids = json::array();
    bool ended = false;
    for (int step = 0; step < tokens; ++step) {
        float * logits = llama_get_logits_ith(ctx.get(), -1);
        const int size = llama_vocab_n_tokens(vocab);
        if (!logits || size < 1) throw std::runtime_error("Missing conversation logits");
        int best = 0;
        for (int index = 0; index < size; ++index) {
            if (!std::isfinite(logits[index])) throw std::runtime_error("Nonfinite conversation logits");
            if (logits[index] > logits[best]) best = index;
        }
        llama_token token = best;
        output_ids.push_back(token);
        if (llama_vocab_is_eog(vocab, token)) { ended = true; break; }
        std::vector<char> piece(256);
        int length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        if (length < 0) {
            piece.resize(-length);
            length = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        }
        if (length < 0) throw std::runtime_error("Conversation token decoding failed");
        text.append(piece.data(), length);
        pending.append(piece.data(), length);
        const size_t complete = utf8_prefix(pending);
        if (complete) {
            emit({{"type", "chunk"}, {"id", id}, {"text", pending.substr(0, complete)}});
            pending.erase(0, complete);
        }
        if (step + 1 < tokens && llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) != 0)
            throw std::runtime_error("Conversation decode failed");
    }
    if (!pending.empty()) emit({{"type", "chunk"}, {"id", id}, {"text", pending}});
    const auto decoded = Clock::now();
    emit({{"type", "done"}, {"id", id}, {"text", text}, {"generated_tokens", output_ids},
          {"prompt_tokens", n}, {"npu_weight_matmul_operations", pixelforge_ggml_npu_requests() - npu_before},
          {"cpu_dynamic_matmul_operations", pixelforge_ggml_npu_cpu_matmuls() - cpu_before},
          {"cpu_prefill_weight_matmul_operations", pixelforge_ggml_npu_cpu_prefill_matmuls() - prefill_before},
          {"cpu_placed_weight_matmul_operations", pixelforge_ggml_npu_cpu_placed_matmuls() - placed_before},
          {"selected_npu_weight_tensors", selected_tensors()},
          {"prefill_cpu", prefill_cpu}, {"token_limit_reached", !ended},
          {"context_init_seconds", context_init_seconds},
          {"prefill_seconds", std::chrono::duration<double>(prefilled - began).count()},
          {"decode_seconds", std::chrono::duration<double>(decoded - prefilled).count()},
          {"generation_seconds", std::chrono::duration<double>(decoded - began).count()}});
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: pixelforge-chat MODEL THREADS");
        const int threads = std::stoi(argv[2]);
        if (threads < 1 || threads > 32) throw std::runtime_error("Threads must be 1 to 32");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_extra_bufts = false;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model || !llama_model_has_decoder(model.get()) || llama_model_has_encoder(model.get()))
            throw std::runtime_error("A supported decoder-only GGUF is required");
        emit({{"type", "ready"}, {"context_limit", 32768}, {"output_limit", 512},
              {"dense_npu_requested", pixelforge_ggml_npu_enabled()},
              {"selected_npu_weight_tensors", selected_tensors()},
              {"prompt_format", "host-rendered-model-template"}});
        std::string line;
        while (frame(line)) {
            std::string id;
            try {
                const auto request = json::parse(line);
                if (request.is_object() && request.contains("id") && request["id"].is_string())
                    id = request["id"].get<std::string>().substr(0, 128);
                generate(model.get(), threads, request);
            } catch (const std::exception & error) {
                pixelforge_ggml_npu_set_cpu_prefill(false);
                emit({{"type", "error"}, {"id", id}, {"error", error.what()}});
                // Never continue an exchange whose engine or broker state is uncertain.
                if (pixelforge_ggml_npu_requests() > 0) throw;
            }
        }
        if (!pixelforge_ggml_npu_finish()) throw std::runtime_error("NPU broker shutdown failed");
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "PixelForge chat: " << error.what() << std::endl;
        return 1;
    }
}
