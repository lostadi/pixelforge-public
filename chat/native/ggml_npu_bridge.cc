// Keep llama.cpp's architecture and tokenizer; send planned dense projections
// to a persistent PixelForge broker. Disabled unless explicitly configured.
#include "ggml_npu_bridge.h"
#include "ggml-cpu-impl.h"
#include "ggml-backend.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
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
std::atomic<bool> cpu_prefill{false};

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
        if (const char * choice = std::getenv("PIXELFORGE_GGML_NPU_TENSORS")) {
            selection_csv = choice;
            if (selection_csv.empty() || selection_csv.size() > 131072)
                throw std::runtime_error("NPU tensor selection must not be empty or oversized");
            size_t offset = 0;
            for (;;) {
                const size_t end = selection_csv.find(',', offset);
                const std::string name = selection_csv.substr(offset, end == std::string::npos ? end : end - offset);
                if (name.empty() || matrices.find(name) == matrices.end() || !selected.insert(name).second)
                    throw std::runtime_error("NPU tensor selection contains an empty, unknown, or duplicate weight");
                if (end == std::string::npos) break;
                offset = end + 1;
            }
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
        return selection_csv.empty() || selected.count(weight->name) != 0;
    }

    const std::string & SelectionCSV() const { return selection_csv; }

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
        std::lock_guard<std::mutex> lock(mutex);
        Start();
        deadline = Clock::now() + std::chrono::seconds(timeout);
        const auto * input = dst->src[1];
        const size_t columns = size_t(input->ne[0]), rows = size_t(dst->ne[0]);
        const size_t vectors = size_t(input->ne[1]) * size_t(input->ne[2]) * size_t(input->ne[3]);
        const std::string name(dst->src[0]->name);
        const std::string header = "M " + std::to_string(name.size()) + " " +
                                  std::to_string(columns) + " " + std::to_string(rows) + " " +
                                  std::to_string(vectors) + "\n";
        Write(header.data(), header.size());
        Write(name.data(), name.size());
        std::vector<float> vector(columns);
        for (size_t v = 0; v < vectors; ++v) {
            const size_t offset = Offset(input, v);
            for (size_t c = 0; c < columns; ++c) {
                std::memcpy(&vector[c], static_cast<const char *>(input->data) + offset + c * input->nb[0], 4);
                if (!std::isfinite(vector[c])) throw std::runtime_error("nonfinite dense input");
            }
            Write(vector.data(), columns * 4);
        }
        std::istringstream response(Line());
        std::string status, count, extra;
        if (!(response >> status >> count) || status != "OK" || response >> extra ||
            Decimal(count, kMaxTransfer / 4) != rows * vectors)
            throw std::runtime_error("broker rejected projection or returned invalid output header");
        vector.resize(rows);
        for (size_t v = 0; v < vectors; ++v) {
            Read(vector.data(), rows * 4);
            const size_t offset = Offset(dst, v);
            for (size_t r = 0; r < rows; ++r) {
                if (!std::isfinite(vector[r])) throw std::runtime_error("nonfinite dense result");
                std::memcpy(static_cast<char *>(dst->data) + offset + r * dst->nb[0], &vector[r], 4);
            }
        }
        handled_requests.fetch_add(1, std::memory_order_relaxed);
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
    struct Planned { size_t columns, rows, type; };
    std::map<std::string, Planned> matrices;
    std::set<std::string> selected;
    std::string selection_csv;
    std::string executable;
    std::mutex mutex;
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
        ggml_barrier(params->threadpool);
        if (params->ith == 0) broker.Compute(dst);
        ggml_barrier(params->threadpool);
        return true;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "PixelForge NPU bridge failed: %s\n", error.what());
        // Exit with no eligible-matmul CPU fallback. Broker stdin closes on exit;
        // its owner must also handle EOF and close its native NPU helper.
        std::abort();
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

extern "C" void pixelforge_ggml_npu_set_cpu_prefill(bool enabled) { cpu_prefill.store(enabled, std::memory_order_release); }
extern "C" uint64_t pixelforge_ggml_npu_cpu_prefill_matmuls(void) { return cpu_prefill_matmuls.load(std::memory_order_relaxed); }

extern "C" uint64_t pixelforge_ggml_npu_cpu_placed_matmuls(void) { return cpu_placed_matmuls.load(std::memory_order_relaxed); }
extern "C" const char * pixelforge_ggml_npu_selected_tensors_csv(void) { return pixelforge_ggml_npu_enabled() ? Instance().SelectionCSV().c_str() : ""; }
