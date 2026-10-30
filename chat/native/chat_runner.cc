// Resident architecture/tokenizer owner. The host supplies the model's rendered
// template and owns tool execution; this process only performs inference.
#include "llama.h"
#include "llama-context.h" // Pinned source accessor for resolved CPU flash-attention state.
#include "ggml_npu_bridge.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <list>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
constexpr size_t MAX_FRAME = 262144;
constexpr size_t MIB = 1024 * 1024;

struct SequenceStateAPI {
    decltype(&llama_state_seq_get_size) size = llama_state_seq_get_size;
    decltype(&llama_state_seq_get_data) get = llama_state_seq_get_data;
    decltype(&llama_state_seq_set_data) set = llama_state_seq_set_data;
};

static void clear_memory(llama_context * ctx) {
    llama_synchronize(ctx);
    const auto memory = llama_get_memory(ctx);
    if (!memory) throw std::runtime_error("Conversation context has no resettable memory");
    llama_memory_clear(memory, true);
    llama_synchronize(ctx);
    if (llama_memory_seq_pos_min(memory, 0) != -1 || llama_memory_seq_pos_max(memory, 0) != -1)
        throw std::runtime_error("Conversation memory reset left sequence state");
}

class PrefixCache {
    struct Snapshot {
        std::vector<llama_token> tokens;
        std::vector<uint8_t> state;
        bool cpu_prefill;
        size_t bytes() const {
            return state.capacity() + tokens.capacity() * sizeof(llama_token) + sizeof(Snapshot) + 2 * sizeof(void *);
        }
    };
    size_t limit;
    SequenceStateAPI api;
    std::list<Snapshot> snapshots; // MRU first; only prompt states, never generated tokens.
public:
    explicit PrefixCache(size_t budget = 0, SequenceStateAPI functions = {}):limit(budget),api(functions) {}
    void clear() { snapshots.clear(); }
    size_t budget() const { return limit; }
    size_t entries() const { return snapshots.size(); }
    size_t bytes() const {
        size_t result = 0;
        for (const auto & item : snapshots) result += item.bytes();
        return result;
    }
    int restore(llama_context * ctx, const std::vector<llama_token> & input, bool cpu) {
        auto best = snapshots.end();
        for (auto it = snapshots.begin(); it != snapshots.end(); ++it)
            if (it->cpu_prefill == cpu && it->tokens.size() < input.size() &&
                (best == snapshots.end() || it->tokens.size() > best->tokens.size()) &&
                std::equal(it->tokens.begin(), it->tokens.end(), input.begin())) best = it;
        if (best == snapshots.end()) return 0;
        try {
            const int count = static_cast<int>(best->tokens.size());
            if (api.set(ctx, best->state.data(), best->state.size(), 0) != best->state.size() ||
                llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != count - 1)
                throw std::runtime_error("Prefix sequence state restore did not match its checkpoint");
            snapshots.splice(snapshots.begin(), snapshots, best);
            return count;
        } catch (const std::exception &) {
            clear();
            clear_memory(ctx); // No model operation occurs before a clean fresh prefill.
            return 0;
        }
    }
    void save(llama_context * ctx, const std::vector<llama_token> & input, int count, bool cpu) {
        if (!limit || count < 1 || size_t(count) >= input.size()) return;
        for (auto it = snapshots.begin(); it != snapshots.end(); ++it)
            if (it->cpu_prefill == cpu && it->tokens.size() == size_t(count) &&
                std::equal(it->tokens.begin(), it->tokens.end(), input.begin())) {
                snapshots.splice(snapshots.begin(), snapshots, it);
                return;
            }
        try {
            llama_synchronize(ctx);
            const size_t state_bytes = api.size(ctx, 0);
            const size_t overhead = size_t(count) * sizeof(llama_token) + sizeof(Snapshot) + 2 * sizeof(void *);
            if (!state_bytes || overhead > limit || state_bytes > limit - overhead) return;
            // Evict BEFORE allocation so retained states plus this checkpoint
            // stay within the explicit budget, including its token metadata.
            while (!snapshots.empty() && bytes() > limit - state_bytes - overhead) snapshots.pop_back();
            Snapshot item{std::vector<llama_token>(input.begin(),input.begin()+count),
                          std::vector<uint8_t>(state_bytes),cpu};
            if (item.bytes() > limit) return;
            while (!snapshots.empty() && bytes() > limit - item.bytes()) snapshots.pop_back();
            if (api.get(ctx,item.state.data(),item.state.size(),0) != item.state.size()) {
                clear();
                return; // A failed read-only snapshot never replays model work.
            }
            snapshots.push_front(std::move(item));
        } catch (const std::exception &) {
            clear(); // Cache allocation/serialization is optional; live state is untouched.
        }
    }
};

struct ConversationContext {
    std::unique_ptr<llama_context, decltype(&llama_free)> value{nullptr, llama_free};
    int size = 0;
    int batch_size = 0;
    const llama_model * owner = nullptr;
    uint64_t initializations = 0;
    PrefixCache prefix;
    explicit ConversationContext(size_t prefix_bytes = 0, SequenceStateAPI api = {}):prefix(prefix_bytes,api) {}
    void discard() { prefix.clear(); value.reset(); size = 0; batch_size = 0; owner = nullptr; }
};

static int prefix_cache_mib() {
    const char * setting = std::getenv("PIXELFORGE_CHAT_PREFIX_CACHE_MIB");
    if (!setting) return 0;
    const std::string text(setting);
    if (text.empty() || text.size() > 3 || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("PIXELFORGE_CHAT_PREFIX_CACHE_MIB must be an integer from 0 to 512");
    const int value = std::stoi(text);
    if (value > 512) throw std::runtime_error("PIXELFORGE_CHAT_PREFIX_CACHE_MIB must be an integer from 0 to 512");
    return value;
}

static int prefill_batch_size() {
    const char * setting = std::getenv("PIXELFORGE_CHAT_PREFILL_BATCH");
    if (!setting) return 512;
    const std::string text(setting);
    if (text.empty() || text.size() > 3 ||
        text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("PIXELFORGE_CHAT_PREFILL_BATCH must be an integer from 1 to 512");
    const int value = std::stoi(text);
    if (value < 1 || value > 512)
        throw std::runtime_error("PIXELFORGE_CHAT_PREFILL_BATCH must be an integer from 1 to 512");
    return value;
}

struct FlashAttention {
    std::string setting = "off";
    llama_flash_attn_type type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
};

static FlashAttention flash_attention() {
    const char * value = std::getenv("PIXELFORGE_CHAT_FLASH_ATTN");
    FlashAttention result;
    if (value) result.setting = value;
    if (result.setting == "off") result.type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    else if (result.setting == "auto") result.type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    else if (result.setting == "on") result.type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    else throw std::runtime_error("PIXELFORGE_CHAT_FLASH_ATTN must be off, auto, or on");
    return result;
}

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

static json tensor_names(const std::string & csv) {
    if (csv.empty()) return nullptr; // Null denotes all planned weights when NPU enabled.
    json names = json::array();
    std::istringstream source(csv);
    std::string name;
    while (std::getline(source, name, ',')) names.push_back(name);
    return names;
}

static json selected_tensors(bool prefill = false) {
    return tensor_names(prefill ? pixelforge_ggml_npu_prefill_selected_tensors_csv() :
                                 pixelforge_ggml_npu_selected_tensors_csv());
}

static json full_ffn_tensors() {
    const std::string csv = pixelforge_ggml_npu_full_ffn_tensors_csv();
    return csv.empty() ? json::array() : tensor_names(csv);
}

static void generate(llama_model * model, ConversationContext & resident, int threads,
                     int prefill_batch, const FlashAttention & flash, const json & request) {
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
    const bool context_reused = resident.value && resident.size == context &&
                                resident.batch_size == prefill_batch && resident.owner == model;
    int prefix_tokens_reused = 0;
    double context_init_seconds = 0.0, context_reset_seconds = 0.0;
    if (context_reused) {
        const auto reset_began = Clock::now();
        clear_memory(resident.value.get());
        prefix_tokens_reused = resident.prefix.restore(resident.value.get(), input, prefill_cpu);
        llama_perf_context_reset(resident.value.get());
        context_reset_seconds = std::chrono::duration<double>(Clock::now() - reset_began).count();
    } else {
        resident.discard();
        auto cp = llama_context_default_params();
        cp.n_ctx = context;
        cp.n_batch = cp.n_ubatch = prefill_batch;
        cp.n_threads = cp.n_threads_batch = threads;
        cp.flash_attn_type = flash.type;
        cp.offload_kqv = cp.op_offload = false;
        const auto context_began = Clock::now();
        resident.value.reset(llama_init_from_model(model, cp));
        if (!resident.value) throw std::runtime_error("Conversation context initialization failed with CPU flash attention " + flash.setting);
        resident.size = context;
        resident.batch_size = prefill_batch;
        resident.owner = model;
        ++resident.initializations;
        context_init_seconds = std::chrono::duration<double>(Clock::now() - context_began).count();
    }
    auto * ctx = resident.value.get();
    const uint64_t npu_before = pixelforge_ggml_npu_requests();
    const uint64_t cpu_before = pixelforge_ggml_npu_cpu_matmuls();
    const uint64_t prefill_before = pixelforge_ggml_npu_cpu_prefill_matmuls();
    const uint64_t placed_before = pixelforge_ggml_npu_cpu_placed_matmuls();
    const uint64_t vectors_before = pixelforge_ggml_npu_vectors();
    const uint64_t bridge_before = pixelforge_ggml_npu_nanoseconds();
    const uint64_t fused_before = pixelforge_ggml_npu_fused_ffns();
    const auto began = Clock::now();
    pixelforge_ggml_npu_set_prefill_phase(true);
    pixelforge_ggml_npu_set_cpu_prefill(prefill_cpu);
    std::vector<int8_t> output_flags(prefill_batch, 0);
    int prefill_chunks = 0;
    const int early_checkpoint = ((1024 + prefill_batch - 1) / prefill_batch) * prefill_batch;
    const int late_checkpoint = n > 128 ? ((n - 128) / prefill_batch) * prefill_batch : 0;
    for (int offset = prefix_tokens_reused; offset < n; offset += prefill_batch) {
        const int count = std::min(prefill_batch, n - offset);
        // Earlier chunks only advance attention/recurrent memory. Their LM
        // head outputs are discarded; request logits for the final token only.
        if (offset + count == n) output_flags[count - 1] = 1;
        auto batch = llama_batch_get_one(input.data() + offset, count);
        batch.logits = output_flags.data();
        if (llama_decode(ctx, batch) != 0)
            throw std::runtime_error("Conversation prefill failed");
        ++prefill_chunks;
        const int processed = offset + count;
        if (processed < n && (processed == early_checkpoint || processed == late_checkpoint))
            resident.prefix.save(ctx,input,processed,prefill_cpu);
    }
    pixelforge_ggml_npu_set_cpu_prefill(false);
    pixelforge_ggml_npu_set_prefill_phase(false);
    const auto prefilled = Clock::now();
    const uint64_t npu_prefill = pixelforge_ggml_npu_requests() - npu_before;
    const uint64_t fused_prefill = pixelforge_ggml_npu_fused_ffns() - fused_before;
    std::string text, pending;
    json output_ids = json::array();
    bool ended = false;
    for (int step = 0; step < tokens; ++step) {
        float * logits = llama_get_logits_ith(ctx, -1);
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
        if (step + 1 < tokens && llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0)
            throw std::runtime_error("Conversation decode failed");
    }
    if (!pending.empty()) emit({{"type", "chunk"}, {"id", id}, {"text", pending}});
    const auto decoded = Clock::now();
    emit({{"type", "done"}, {"id", id}, {"text", text}, {"generated_tokens", output_ids},
          {"prompt_tokens", n}, {"npu_weight_matmul_operations", pixelforge_ggml_npu_requests() - npu_before},
          {"npu_prefill_weight_matmul_operations", npu_prefill},
          {"npu_decode_weight_matmul_operations", pixelforge_ggml_npu_requests() - npu_before - npu_prefill},
          {"cpu_dynamic_matmul_operations", pixelforge_ggml_npu_cpu_matmuls() - cpu_before},
          {"cpu_prefill_weight_matmul_operations", pixelforge_ggml_npu_cpu_prefill_matmuls() - prefill_before},
          {"cpu_placed_weight_matmul_operations", pixelforge_ggml_npu_cpu_placed_matmuls() - placed_before},
          {"npu_weight_matmul_vectors", pixelforge_ggml_npu_vectors() - vectors_before},
          {"npu_bridge_seconds", double(pixelforge_ggml_npu_nanoseconds() - bridge_before) * 1e-9},
          {"npu_fused_ffn_operations", pixelforge_ggml_npu_fused_ffns() - fused_before},
          {"npu_prefill_fused_ffn_operations", fused_prefill},
          {"npu_decode_fused_ffn_operations", pixelforge_ggml_npu_fused_ffns() - fused_before - fused_prefill},
          {"full_ffn_tensors", full_ffn_tensors()},
          {"full_ffn_phases", pixelforge_ggml_npu_full_ffn_phases()},
          {"selected_npu_weight_tensors", selected_tensors()},
          {"selected_prefill_npu_weight_tensors", selected_tensors(true)},
          {"prefill_cpu", prefill_cpu}, {"token_limit_reached", !ended},
          {"flash_attention", flash.setting}, {"attention_backend", "CPU"},
          {"cpu_flash_attention_enabled", ctx->get_cparams().flash_attn},
          {"context_init_seconds", context_init_seconds},
          {"context_reused", context_reused}, {"context_initializations", resident.initializations},
          {"context_reset_seconds", context_reset_seconds}, {"prefill_batch_size", prefill_batch},
          {"prefix_cache_mib", resident.prefix.budget() / MIB},
          {"prefix_cache_hit", prefix_tokens_reused > 0}, {"prefix_tokens_reused", prefix_tokens_reused},
          {"prefix_cache_bytes", resident.prefix.bytes()}, {"prefix_cache_entries", resident.prefix.entries()},
          {"prefill_tokens_processed", n - prefix_tokens_reused},
          {"prefill_chunks", prefill_chunks}, {"prefill_logits_chunks_skipped", prefill_chunks - 1},
          {"prefill_seconds", std::chrono::duration<double>(prefilled - began).count()},
          {"decode_seconds", std::chrono::duration<double>(decoded - prefilled).count()},
          {"generation_seconds", std::chrono::duration<double>(decoded - began).count()}});
}

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: pixelforge-chat MODEL THREADS");
        const int threads = std::stoi(argv[2]);
        if (threads < 1 || threads > 32) throw std::runtime_error("Threads must be 1 to 32");
        const int prefill_batch = prefill_batch_size();
        const int prefix_mib = prefix_cache_mib();
        const FlashAttention flash = flash_attention();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_extra_bufts = false;
        mp.load_mode = LLAMA_LOAD_MODE_MMAP;
        const auto load_began = Clock::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model || !llama_model_has_decoder(model.get()) || llama_model_has_encoder(model.get()))
            throw std::runtime_error("A supported decoder-only GGUF is required");
        emit({{"type", "ready"}, {"context_limit", 32768}, {"output_limit", 512},
              {"model_load_seconds", std::chrono::duration<double>(Clock::now() - load_began).count()},
              {"prefill_batch_size", prefill_batch}, {"context_reuse", "full-memory-reset"},
              {"prefix_cache_mib", prefix_mib},
              {"flash_attention", flash.setting}, {"attention_backend", "CPU"},
              {"cpu_flash_attention_enabled", nullptr}, // Resolved only after lazy context creation.
              {"dense_npu_requested", pixelforge_ggml_npu_enabled()},
              {"selected_npu_weight_tensors", selected_tensors()},
              {"selected_prefill_npu_weight_tensors", selected_tensors(true)},
              {"full_ffn_tensors", full_ffn_tensors()},
              {"full_ffn_phases", pixelforge_ggml_npu_full_ffn_phases()},
              {"prompt_format", "host-rendered-model-template"}});
        std::string line;
        ConversationContext resident(size_t(prefix_mib) * MIB);
        while (frame(line)) {
            std::string id;
            try {
                const auto request = json::parse(line);
                if (request.is_object() && request.contains("id") && request["id"].is_string())
                    id = request["id"].get<std::string>().substr(0, 128);
                generate(model.get(), resident, threads, prefill_batch, flash, request);
            } catch (const std::exception & error) {
                pixelforge_ggml_npu_set_cpu_prefill(false);
                pixelforge_ggml_npu_set_prefill_phase(false);
                resident.discard();
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
