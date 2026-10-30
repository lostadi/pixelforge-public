// Keep llama.cpp's architecture and tokenizer; send planned dense projections
// to a persistent PixelForge broker. Disabled unless explicitly configured.
#include "ggml_npu_bridge.h"
#include "ggml-cpu-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sstream>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern char ** environ;
namespace {
using Clock = std::chrono::steady_clock;
constexpr size_t kMaxTransfer = size_t{2} << 30;
std::atomic<uint64_t> handled_requests{0}, cpu_matmuls{0}, cpu_prefill_matmuls{0}, cpu_placed_matmuls{0};
std::atomic<uint64_t> handled_vectors{0}, bridge_nanoseconds{0};
std::atomic<uint64_t> fused_ffns{0};
std::atomic<bool> cpu_prefill{false};
std::atomic<bool> prefill_phase{false};

size_t Decimal(const std::string & token, size_t maximum) {
    if (token.empty()) throw std::runtime_error("missing decimal field");
    size_t value = 0;
    for (unsigned char byte : token) {
        if (byte < '0' || byte > '9' || value > (maximum - (byte - '0')) / 10)
            throw std::runtime_error("invalid or overflowing decimal field");
        value = value * 10 + byte - '0';
    }
    if (!value) throw std::runtime_error("zero decimal field");
    return value;
}

size_t Product(size_t left, size_t right, size_t maximum = kMaxTransfer) {
    if (right && left > maximum / right) throw std::runtime_error("transfer size exceeds bound");
    return left * right;
}

bool WeightName(const char * name) {
    const std::string text(name);
    return text.size() >= 7 && text.compare(text.size() - 7, 7, ".weight") == 0;
}

bool FFNPattern(const ggml_cgraph * graph, int node) {
    if (!graph || node < 0 || node > graph->n_nodes - 4) return false;
    const auto * gate = graph->nodes[node];
    const auto * up = graph->nodes[node + 1];
    const auto * glu = graph->nodes[node + 2];
    const auto * down = graph->nodes[node + 3];
    // The final layer can contain zero-row FFN nodes when a prefill chunk
    // requests no logits. Preserve ggml's empty-op skip instead of invoking F.
    if (ggml_is_empty(gate) || ggml_is_empty(up) || ggml_is_empty(glu) || ggml_is_empty(down)) return false;
    for (const auto * tensor : {gate, up, glu, down}) {
        if (tensor->view_src) return false;
        for (int source = 2; source < GGML_MAX_SRC; ++source)
            if (tensor->src[source]) return false;
    }
    // MUL_MAT may carry a Hadamard/FWHT hint, whose CPU dispatch differs from FC.
    if (gate->op_params[1] || up->op_params[1] || down->op_params[1]) return false;
    const ggml_op operations[] = {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU, GGML_OP_MUL_MAT};
    const int outputs[] = {node + 3};
    if (!ggml_can_fuse_subgraph(graph, node, 4, operations, outputs, 1) ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->op_params[1] != 0 ||
        glu->src[0] != gate || glu->src[1] != up || down->src[1] != glu ||
        !gate->src[0] || !up->src[0] || !down->src[0] || !gate->src[1] ||
        gate->src[1] != up->src[1] || gate->type != GGML_TYPE_F32 ||
        up->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 || down->type != GGML_TYPE_F32 ||
        gate->src[1]->type != GGML_TYPE_F32 || !ggml_are_same_shape(gate, up) ||
        !ggml_are_same_shape(gate, glu)) return false;
    const std::string name(gate->src[0]->name), suffix(".ffn_gate.weight");
    if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return false;
    const auto prefix = name.substr(0, name.size() - suffix.size());
    if (prefix.size() <= 4 || prefix.compare(0, 4, "blk.") != 0 ||
        prefix.find_first_not_of("0123456789", 4) != std::string::npos ||
        std::string(up->src[0]->name) != prefix + ".ffn_up.weight" ||
        std::string(down->src[0]->name) != prefix + ".ffn_down.weight") return false;
    if (gate->src[0]->ne[0] != gate->src[1]->ne[0] || up->src[0]->ne[0] != gate->src[1]->ne[0] ||
        gate->src[0]->ne[1] != gate->ne[0] || up->src[0]->ne[1] != up->ne[0] ||
        down->src[0]->ne[0] != glu->ne[0] || down->src[0]->ne[1] != down->ne[0] ||
        down->ne[0] != gate->src[1]->ne[0]) return false;
    for (int dimension = 1; dimension < 4; ++dimension)
        if (down->ne[dimension] != gate->src[1]->ne[dimension] ||
            gate->ne[dimension] != gate->src[1]->ne[dimension]) return false;
    return true;
}

class Broker {
public:
    Broker() {
        const char * plan = std::getenv("PIXELFORGE_GGML_PLAN");
        const char * launcher = std::getenv("PIXELFORGE_GGML_BROKER");
        if (!plan || plan[0] != '/' || !launcher || launcher[0] != '/')
            throw std::runtime_error("bridge requires absolute PLAN and BROKER paths");
        executable = launcher;
        std::ifstream input(plan);
        if (!input) throw std::runtime_error("cannot read GGML dense projection plan");
        std::string line;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            std::istringstream fields(line);
            std::string columns, rows, type, name, extra;
            if (!(fields >> columns >> rows >> type >> name) || fields >> extra || name.size() >= GGML_MAX_NAME)
                throw std::runtime_error("invalid GGML dense projection plan line");
            // GGML_TYPE_F32 is zero; type identifiers alone permit zero.
            const size_t type_id = type == "0" ? 0 : Decimal(type, GGML_TYPE_COUNT - 1);
            Planned dimensions{Decimal(columns, kMaxTransfer / 4),
                               Decimal(rows, kMaxTransfer / 4), type_id};
            if (!matrices.emplace(name, dimensions).second)
                throw std::runtime_error("duplicate planned GGML weight name");
        }
        if (input.bad() || matrices.empty()) throw std::runtime_error("empty or unreadable GGML plan");
        const auto parse_selection = [&](const char * variable, std::string & csv, std::set<std::string> & names) {
            const char * choice = std::getenv(variable);
            if (!choice) return;
            csv = choice;
            if (csv.empty() || csv.size() > 131072)
                throw std::runtime_error("NPU tensor selection must not be empty or oversized");
            size_t offset = 0;
            for (;;) {
                const size_t end = csv.find(',', offset);
                const std::string name = csv.substr(offset, end == std::string::npos ? end : end - offset);
                if (name.empty() || matrices.find(name) == matrices.end() || !names.insert(name).second)
                    throw std::runtime_error("NPU tensor selection contains an empty, unknown, or duplicate weight");
                if (end == std::string::npos) break;
                offset = end + 1;
            }
        };
        parse_selection("PIXELFORGE_GGML_NPU_TENSORS", selection_csv, selected);
        parse_selection("PIXELFORGE_GGML_PREFILL_NPU_TENSORS", prefill_selection_csv, prefill_selected);
        parse_selection("PIXELFORGE_GGML_FULL_FFN_TENSORS", full_ffn_csv, full_ffn);
        for (const auto & name : full_ffn) {
            const std::string suffix(".ffn_gate.weight");
            if (name.size() <= suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
                throw std::runtime_error("Full FFN selection must name canonical gate weights");
            const auto prefix = name.substr(0, name.size() - suffix.size());
            if (!matrices.count(prefix + ".ffn_up.weight") || !matrices.count(prefix + ".ffn_down.weight"))
                throw std::runtime_error("Full FFN selection requires planned up and down weights");
        }
        if (const char * phases = std::getenv("PIXELFORGE_GGML_FULL_FFN_PHASES")) {
            full_ffn_phases = phases;
            if (full_ffn.empty() || (full_ffn_phases != "P" && full_ffn_phases != "D" && full_ffn_phases != "PD"))
                throw std::runtime_error("Full FFN phases require configured tensors and P, D, or PD");
        }
        if (const char * seconds = std::getenv("PIXELFORGE_GGML_TIMEOUT_SECONDS"))
            timeout = Decimal(seconds, 86400);
    }
    ~Broker() { Close(); }

    bool Eligible(const ggml_tensor * weight) const {
        auto found = matrices.find(weight->name);
        const ggml_tensor * origin = weight;
        for (int depth = 0; origin->view_src; ++depth) {
            if (depth >= 64) throw std::runtime_error("cyclic or excessive weight views");
            origin = origin->view_src;
        }
        if (found == matrices.end()) {
            if ((weight->op == GGML_OP_NONE && WeightName(weight->name)) ||
                (origin->op == GGML_OP_NONE && WeightName(origin->name)))
                throw std::runtime_error(std::string("unplanned dense weight: ") + weight->name);
            return false; // activation/cache attention matmuls keep ggml semantics.
        }
        if (weight->op != GGML_OP_NONE || weight->view_src || weight->ne[2] != 1 || weight->ne[3] != 1 ||
            weight->ne[0] <= 0 || weight->ne[1] <= 0 ||
            size_t(weight->ne[0]) != found->second.columns || size_t(weight->ne[1]) != found->second.rows ||
            size_t(weight->type) != found->second.type || !weight->buffer ||
            ggml_backend_buffer_get_usage(weight->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_backend_buffer_is_host(weight->buffer) ||
            weight->nb[0] != ggml_type_size(weight->type) ||
            weight->nb[1] != ggml_row_size(weight->type, weight->ne[0]))
            throw std::runtime_error(std::string("unsupported planned weight layout: ") + weight->name);
        return true;
    }

    bool Selected(const ggml_tensor * weight) const {
        const bool prefill = prefill_phase.load(std::memory_order_acquire) && !prefill_selection_csv.empty();
        const auto & csv = prefill ? prefill_selection_csv : selection_csv;
        const auto & names = prefill ? prefill_selected : selected;
        return csv.empty() || names.count(weight->name) != 0;
    }

    const std::string & SelectionCSV(bool prefill = false) const {
        return prefill && !prefill_selection_csv.empty() ? prefill_selection_csv : selection_csv;
    }
    const std::string & FullFFNCSV() const { return full_ffn_csv; }
    const std::string & FullFFNPhases() const { return full_ffn_phases; }
    bool FullFFNSelected(const ggml_tensor * gate) const {
        const char phase = prefill_phase.load(std::memory_order_acquire) ? 'P' : 'D';
        return full_ffn.count(gate->name) && full_ffn_phases.find(phase) != std::string::npos;
    }

    void Validate(const ggml_tensor * dst) const {
        const auto * weight = dst->src[0];
        const auto * input = dst->src[1];
        if (!input || input->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            input->ne[0] != weight->ne[0] || dst->ne[0] != weight->ne[1] ||
            input->ne[1] != dst->ne[1] || input->ne[2] != dst->ne[2] || input->ne[3] != dst->ne[3] ||
            !input->data || !dst->data)
            throw std::runtime_error("unsupported dense activation/output type or dimensions");
        for (int d = 0; d < 4; ++d)
            if (input->ne[d] <= 0 || dst->ne[d] <= 0)
                throw std::runtime_error("empty dense activation/output");
        size_t vectors = Product(size_t(input->ne[1]), size_t(input->ne[2]));
        vectors = Product(vectors, size_t(input->ne[3]));
        Product(Product(vectors, size_t(input->ne[0])), sizeof(float));
        Product(Product(vectors, size_t(dst->ne[0])), sizeof(float));
        // Check the full strided span before copying; nb[0] need not be 4.
        for (const auto * tensor : {input, dst}) {
            size_t span = sizeof(float);
            for (int d = 0; d < 4; ++d) {
                const size_t extent = Product(size_t(tensor->ne[d] - 1), tensor->nb[d],
                                              std::numeric_limits<size_t>::max());
                if (span > std::numeric_limits<size_t>::max() - extent)
                    throw std::runtime_error("overflowing tensor stride");
                span += extent;
            }
            if (span > ggml_nbytes(tensor)) throw std::runtime_error("tensor stride exceeds data span");
        }
    }

    void Compute(ggml_tensor * dst) {
        Exchange(dst, dst->src[1], dst->src[0]->name, 'M', 1);
    }

    void ComputeFFN(ggml_tensor * dst, const ggml_tensor * input, const char * gate) {
        // Full nonlinear FFN graphs consume original F32 X. Activation-max
        // normalization cannot be moved through SiLU and must not be applied.
        Exchange(dst, input, gate, 'F', 3);
        fused_ffns.fetch_add(1, std::memory_order_relaxed);
    }

    template<class Compute>
    void Together(const ggml_compute_params * params, Compute compute) {
        if (params->nth < 1 || params->ith < 0 || params->ith >= params->nth ||
            (params->nth > 1 && !params->threadpool))
            throw std::runtime_error("invalid NPU threadpool rendezvous");
        if (params->nth == 1) {
            compute();
            return;
        }
        Rendezvous * state;
        {
            std::lock_guard<std::mutex> lock(rendezvous_mutex);
            auto & owned = rendezvous[params->threadpool];
            if (!owned) owned = std::make_unique<Rendezvous>();
            state = owned.get();
        }
        std::unique_lock<std::mutex> lock(state->mutex);
        const auto generation = state->generation;
        if (!state->arrived) state->members = params->nth;
        if (state->members != params->nth || ++state->arrived > params->nth)
            throw std::runtime_error("inconsistent NPU threadpool rendezvous");
        if (params->ith != 0) {
            state->changed.notify_all();
            state->changed.wait(lock, [&] { return state->generation != generation; });
            return;
        }
        state->changed.wait(lock, [&] { return state->arrived == params->nth; });
        lock.unlock();
        compute(); // Only the leader performs the blocking RPC; peers sleep.
        lock.lock();
        state->arrived = 0;
        ++state->generation;
        // Publish all output writes through this mutex before any peer leaves.
        state->changed.notify_all();
    }

    void Exchange(ggml_tensor * dst, const ggml_tensor * input, const std::string & name,
                  char operation, uint64_t weight_operations) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto began = Clock::now();
        Start();
        deadline = Clock::now() + std::chrono::seconds(timeout);
        const size_t columns = size_t(input->ne[0]), rows = size_t(dst->ne[0]);
        const size_t vectors = size_t(input->ne[1]) * size_t(input->ne[2]) * size_t(input->ne[3]);
        const std::string header = std::string(1, operation) + " " + std::to_string(name.size()) + " " +
                                  std::to_string(columns) + " " + std::to_string(rows) + " " +
                                  std::to_string(vectors) + (prefill_phase.load(std::memory_order_acquire) ? " P\n" : " D\n");
        std::vector<float> activation(Product(vectors, columns, kMaxTransfer / sizeof(float)));
        if (ggml_is_contiguous(input)) {
            std::memcpy(activation.data(), input->data, activation.size() * sizeof(float));
        } else {
            for (size_t v = 0; v < vectors; ++v) {
                const size_t offset = Offset(input, v);
                for (size_t c = 0; c < columns; ++c) {
                    std::memcpy(&activation[v * columns + c],
                                static_cast<const char *>(input->data) + offset + c * input->nb[0], sizeof(float));
                }
            }
        }
        for (float value : activation)
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite dense input");
        const std::string metadata = header + name;
        Write(metadata.data(), metadata.size());
        Write(activation.data(), activation.size() * sizeof(float));
        std::istringstream response(Line());
        std::string status, count, extra;
        if (!(response >> status >> count) || status != "OK" || response >> extra ||
            Decimal(count, kMaxTransfer / 4) != rows * vectors)
            throw std::runtime_error("broker rejected projection or returned invalid output header");
        std::vector<float> output(Product(vectors, rows, kMaxTransfer / sizeof(float)));
        Read(output.data(), output.size() * sizeof(float));
        for (float value : output)
            if (!std::isfinite(value)) throw std::runtime_error("nonfinite dense result");
        if (ggml_is_contiguous(dst)) {
            std::memcpy(dst->data, output.data(), output.size() * sizeof(float));
        } else {
            for (size_t v = 0; v < vectors; ++v) {
                const size_t offset = Offset(dst, v);
                for (size_t r = 0; r < rows; ++r) {
                    std::memcpy(static_cast<char *>(dst->data) + offset + r * dst->nb[0],
                                &output[v * rows + r], sizeof(float));
                }
            }
        }
        handled_requests.fetch_add(weight_operations, std::memory_order_relaxed);
        handled_vectors.fetch_add(vectors * weight_operations, std::memory_order_relaxed);
        bridge_nanoseconds.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - began).count(),
                                     std::memory_order_relaxed);
    }

    void Finish() {
        std::lock_guard<std::mutex> lock(mutex);
        if (finished) return;
        if (pid <= 0 && handled_requests.load(std::memory_order_relaxed) == 0) {
            finished = true;
            return; // Model loaded but no NPU decode was submitted.
        }
        if (pid <= 0) throw std::runtime_error("enabled GGML bridge lost its broker");
        deadline = Clock::now() + std::chrono::seconds(25);
        Write("Q\n", 2);
        close(write_fd); write_fd = -1;
        for (;;) {
            int status = 0;
            pid_t result = waitpid(pid, &status, WNOHANG);
            if (result == pid) {
                pid = -1;
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                    throw std::runtime_error("GGML broker failed while finishing");
                close(read_fd); read_fd = -1;
                finished = true;
                return;
            }
            if (result < 0 && errno != EINTR) throw std::runtime_error("cannot reap GGML broker");
            if (Clock::now() >= deadline) throw std::runtime_error("GGML broker finish timeout");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void Close() noexcept {
        if (write_fd >= 0) { close(write_fd); write_fd = -1; }
        if (read_fd >= 0) { close(read_fd); read_fd = -1; }
        if (pid > 0) {
            int status = 0;
            for (int attempt = 0; attempt < 360; ++attempt) {
                const pid_t waited = waitpid(pid, &status, WNOHANG);
                if (waited == pid || (waited < 0 && errno == ECHILD)) { pid = -1; break; }
                if (attempt == 250) kill(-pid, SIGTERM);
                if (attempt == 300) kill(-pid, SIGKILL);
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
private:
    struct Rendezvous {
        std::mutex mutex;
        std::condition_variable changed;
        uint64_t generation = 0;
        int arrived = 0, members = 0;
    };
    struct Planned { size_t columns, rows, type; };
    std::map<std::string, Planned> matrices;
    std::set<std::string> selected;
    std::set<std::string> prefill_selected;
    std::string selection_csv;
    std::string prefill_selection_csv;
    std::set<std::string> full_ffn;
    std::string full_ffn_csv, full_ffn_phases = "P";
    std::string executable;
    std::mutex mutex;
    std::mutex rendezvous_mutex;
    std::map<ggml_threadpool *, std::unique_ptr<Rendezvous>> rendezvous;
    pid_t pid = -1;
    int write_fd = -1, read_fd = -1;
    size_t timeout = 180;
    bool finished = false;
    Clock::time_point deadline;

    static size_t Offset(const ggml_tensor * tensor, size_t vector) {
        const size_t i1 = vector % size_t(tensor->ne[1]);
        vector /= size_t(tensor->ne[1]);
        const size_t i2 = vector % size_t(tensor->ne[2]);
        const size_t i3 = vector / size_t(tensor->ne[2]);
        return i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
    }
    void Start() {
        if (finished) throw std::runtime_error("GGML broker already finished");
        if (pid > 0) return;
        int input[2], output[2];
        if (pipe2(input, O_CLOEXEC) != 0) throw std::runtime_error("broker input pipe failed");
        if (pipe2(output, O_CLOEXEC) != 0) { close(input[0]); close(input[1]); throw std::runtime_error("broker output pipe failed"); }
        posix_spawn_file_actions_t actions;
        posix_spawnattr_t attributes;
        posix_spawn_file_actions_init(&actions);
        posix_spawnattr_init(&attributes);
        posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO);
        posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO);
        for (int fd : {input[0], input[1], output[0], output[1]}) posix_spawn_file_actions_addclose(&actions, fd);
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attributes, 0);
        char * argv[] = {const_cast<char *>(executable.c_str()), nullptr};
        const int status = posix_spawn(&pid, executable.c_str(), &actions, &attributes, argv, environ);
        posix_spawnattr_destroy(&attributes);
        posix_spawn_file_actions_destroy(&actions);
        close(input[0]); close(output[1]);
        if (status != 0) { close(input[1]); close(output[0]); pid = -1; throw std::runtime_error("cannot launch GGML broker"); }
        write_fd = input[1]; read_fd = output[0];
        if (fcntl(write_fd, F_SETFL, O_NONBLOCK) < 0 || fcntl(read_fd, F_SETFL, O_NONBLOCK) < 0)
            throw std::runtime_error("cannot configure GGML broker pipes");
        signal(SIGPIPE, SIG_IGN);
    }
    void Wait(int fd, short event) {
        for (;;) {
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
            if (remaining <= 0) throw std::runtime_error("GGML broker timeout");
            pollfd item{fd, event, 0};
            int status = poll(&item, 1, int(std::min<int64_t>(remaining, 1000)));
            if (status < 0 && errno == EINTR) continue;
            if (status < 0) throw std::runtime_error("GGML broker poll failed");
            if (status && (item.revents & event)) return;
            if (status && (item.revents & (POLLERR | POLLHUP | POLLNVAL)))
                throw std::runtime_error("GGML broker closed its pipe");
        }
    }
    void Write(const void * data, size_t bytes) {
        const char * cursor = static_cast<const char *>(data);
        while (bytes) {
            Wait(write_fd, POLLOUT);
            ssize_t written = write(write_fd, cursor, bytes);
            if (written < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (written <= 0) throw std::runtime_error("GGML broker write failed");
            cursor += written; bytes -= size_t(written);
        }
    }
    void Read(void * data, size_t bytes) {
        char * cursor = static_cast<char *>(data);
        while (bytes) {
            Wait(read_fd, POLLIN);
            ssize_t received = read(read_fd, cursor, bytes);
            if (received < 0 && (errno == EAGAIN || errno == EINTR)) continue;
            if (received <= 0) throw std::runtime_error("GGML broker read failed");
            cursor += received; bytes -= size_t(received);
        }
    }
    std::string Line() {
        std::string line;
        char byte;
        while (line.size() < 65535) {
            Read(&byte, 1);
            if (byte == '\n') return line;
            line += byte;
        }
        throw std::runtime_error("GGML broker output header exceeds bound");
    }
};

Broker & Instance() { static Broker broker; return broker; }
} // namespace

extern "C" bool pixelforge_ggml_npu_enabled(void) {
    static const bool enabled = [] {
        const char * value = std::getenv("PIXELFORGE_GGML_BROKER");
        return value && *value;
    }();
    return enabled;
}

extern "C" bool pixelforge_ggml_npu_compute(const ggml_compute_params * params, ggml_tensor * dst) {
    if (!pixelforge_ggml_npu_enabled()) return false;
    try {
        Broker & broker = Instance();
        if (!broker.Eligible(dst->src[0])) {
            if (params->ith == 0) cpu_matmuls.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        broker.Validate(dst);
        // Explicit request stage, never a fallback after an NPU error.
        if (cpu_prefill.load(std::memory_order_acquire)) {
            if (params->ith == 0) cpu_prefill_matmuls.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (!broker.Selected(dst->src[0])) {
            if (params->ith == 0) cpu_placed_matmuls.fetch_add(1, std::memory_order_relaxed);
            return false; // Explicit placement only; chosen NPU errors still abort.
        }
        broker.Together(params, [&] { broker.Compute(dst); });
        return true;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "PixelForge NPU bridge failed: %s\n", error.what());
        // Exit with no eligible-matmul CPU fallback. Broker stdin closes on exit;
        // its owner must also handle EOF and close its native NPU helper.
        std::abort();
    }
}

extern "C" bool pixelforge_ggml_npu_ffn_pattern(const ggml_cgraph * graph, int node) {
    return FFNPattern(graph, node);
}

extern "C" int pixelforge_ggml_npu_fuse(const ggml_compute_params * params, const ggml_cgraph * graph, int node) {
    static const bool configured = std::getenv("PIXELFORGE_GGML_FULL_FFN_TENSORS") != nullptr;
    if (!pixelforge_ggml_npu_enabled() || !configured ||
        params->use_ref || cpu_prefill.load(std::memory_order_acquire) || !FFNPattern(graph, node)) return 0;
    try {
        Broker & broker = Instance();
        const auto * gate = graph->nodes[node];
        const auto * up = graph->nodes[node + 1];
        auto * down = graph->nodes[node + 3];
        if (!broker.FullFFNSelected(gate->src[0]) || !broker.Selected(gate->src[0]) ||
            !broker.Selected(up->src[0]) || !broker.Selected(down->src[0])) return 0;
        for (const auto * tensor : {gate, up, static_cast<const ggml_tensor *>(down)}) {
            if (!broker.Eligible(tensor->src[0])) throw std::runtime_error("Full FFN weight is not planned");
            broker.Validate(tensor);
        }
        broker.Together(params, [&] { broker.ComputeFFN(down, gate->src[1], gate->src[0]->name); });
        return 3;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "PixelForge full FFN bridge failed: %s\n", error.what());
        std::abort(); // Includes an F failure before the first successful NPU operation.
    }
}

extern "C" bool pixelforge_ggml_npu_finish(void) {
    if (!pixelforge_ggml_npu_enabled()) return true;
    try {
        Instance().Finish();
        return true;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "PixelForge NPU bridge finish failed: %s\n", error.what());
        try { Instance().Close(); } catch (...) {}
        return false;
    }
}

extern "C" uint64_t pixelforge_ggml_npu_requests(void) { return handled_requests.load(std::memory_order_relaxed); }
extern "C" uint64_t pixelforge_ggml_npu_cpu_matmuls(void) { return cpu_matmuls.load(std::memory_order_relaxed); }
extern "C" uint64_t pixelforge_ggml_npu_vectors(void) { return handled_vectors.load(std::memory_order_relaxed); }
extern "C" uint64_t pixelforge_ggml_npu_nanoseconds(void) { return bridge_nanoseconds.load(std::memory_order_relaxed); }
extern "C" uint64_t pixelforge_ggml_npu_fused_ffns(void) { return fused_ffns.load(std::memory_order_relaxed); }
extern "C" const char * pixelforge_ggml_npu_full_ffn_tensors_csv(void) { return pixelforge_ggml_npu_enabled() ? Instance().FullFFNCSV().c_str() : ""; }
extern "C" const char * pixelforge_ggml_npu_full_ffn_phases(void) { return pixelforge_ggml_npu_enabled() ? Instance().FullFFNPhases().c_str() : ""; }

extern "C" void pixelforge_ggml_npu_set_cpu_prefill(bool enabled) { cpu_prefill.store(enabled, std::memory_order_release); }
extern "C" void pixelforge_ggml_npu_set_prefill_phase(bool enabled) { prefill_phase.store(enabled, std::memory_order_release); }
extern "C" uint64_t pixelforge_ggml_npu_cpu_prefill_matmuls(void) { return cpu_prefill_matmuls.load(std::memory_order_relaxed); }

extern "C" uint64_t pixelforge_ggml_npu_cpu_placed_matmuls(void) { return cpu_placed_matmuls.load(std::memory_order_relaxed); }
extern "C" const char * pixelforge_ggml_npu_selected_tensors_csv(void) { return pixelforge_ggml_npu_enabled() ? Instance().SelectionCSV().c_str() : ""; }
extern "C" const char * pixelforge_ggml_npu_prefill_selected_tensors_csv(void) { return pixelforge_ggml_npu_enabled() ? Instance().SelectionCSV(true).c_str() : ""; }
