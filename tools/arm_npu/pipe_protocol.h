// Bounds and format checks shared by the native binary transport and its tests.
#ifndef PIXELFORGE_ARM_NPU_PIPE_PROTOCOL_H_
#define PIXELFORGE_ARM_NPU_PIPE_PROTOCOL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace pixelforge {
constexpr size_t kMaxPorts = 512;
constexpr size_t kMaxTensorBytes = size_t{1} << 30;
constexpr size_t kMaxPayloadBytes = size_t{2} << 30;
constexpr size_t kDefaultRuntimeCacheBytes = size_t{128} << 20;
constexpr size_t kMaxRuntimeCacheBytes = size_t{2} << 30;
constexpr size_t kBufferAlignment = 4096;

inline size_t RoundedBufferBytes(size_t bytes) {
  return (bytes + kBufferAlignment - 1) & ~(kBufferAlignment - 1);
}

struct ModelHeader {
  size_t payload_bytes = 0;
  std::vector<size_t> input_bytes;
  std::vector<size_t> output_bytes;
};

enum class HeaderStatus { kOk, kModel, kBuffer };

template<typename Unsigned>
inline bool ReadUnsignedDecimal(std::istream& source, Unsigned maximum, Unsigned& value) {
  static_assert(std::is_unsigned<Unsigned>::value, "Unsigned protocol field required");
  std::string token;
  if (!(source >> token) || token.empty()) return false;
  value = 0;
  for (unsigned char digit : token) {
    if (digit < '0' || digit > '9') return false;
    const Unsigned part = digit - '0';
    if (part > maximum || value > (maximum - part) / 10) return false;
    value = value * 10 + part;
  }
  return true;
}

inline bool ReadPositiveDecimal(std::istream& source, size_t maximum, size_t& value) {
  return ReadUnsignedDecimal(source, maximum, value) && value != 0;
}

inline bool HeaderEnded(std::istream& source) {
  source >> std::ws;
  return source.peek() == std::char_traits<char>::eof();
}

inline bool ReadStickyUploadHeader(std::istream& source, size_t& index) {
  return ReadPositiveDecimal(source, kMaxPorts - 1, index) && HeaderEnded(source);
}

inline bool ReadStickyInvokeHeader(std::istream& source, uint64_t& generation) {
  return ReadUnsignedDecimal(source, std::numeric_limits<uint64_t>::max(), generation) &&
         generation != 0 && HeaderEnded(source);
}

inline bool ReadDmaImportHeader(std::istream& source, size_t& index, size_t& bytes) {
  return ReadPositiveDecimal(source, kMaxPorts - 1, index) &&
         ReadPositiveDecimal(source, kMaxTensorBytes, bytes) && HeaderEnded(source);
}

// The command and model name have already been consumed by the caller.
inline HeaderStatus ReadModelHeader(std::istream& source, ModelHeader& header) {
  size_t inputs = 0, outputs = 0, bytes = 0;
  if (!ReadPositiveDecimal(source, kMaxPorts, inputs) ||
      !ReadPositiveDecimal(source, kMaxPorts, outputs) ||
      !ReadPositiveDecimal(source, kMaxPayloadBytes, bytes)) {
    return HeaderStatus::kModel;
  }
  header.payload_bytes = bytes;
  header.input_bytes.resize(inputs);
  header.output_bytes.resize(outputs);
  size_t total_tensor_bytes = 0;
  for (auto& value : header.input_bytes) {
    if (!ReadPositiveDecimal(source, kMaxTensorBytes, value) ||
        value > kMaxPayloadBytes - total_tensor_bytes) {
      return HeaderStatus::kBuffer;
    }
    total_tensor_bytes += value;
  }
  for (auto& value : header.output_bytes) {
    if (!ReadPositiveDecimal(source, kMaxTensorBytes, value) ||
        value > kMaxPayloadBytes - total_tensor_bytes) {
      return HeaderStatus::kBuffer;
    }
    total_tensor_bytes += value;
  }
  return HeaderEnded(source) ? HeaderStatus::kOk : HeaderStatus::kBuffer;
}

// FlatBuffers put the file identifier at bytes 4..7, after a LE root offset.
// This is a format gate, not a substitute for the vendor loader's validation.
inline bool IsDgc0(const std::vector<char>& code) {
  if (code.size() < 12 || code.size() > kMaxPayloadBytes ||
      std::memcmp(code.data() + 4, "DGC0", 4) != 0) {
    return false;
  }
  const auto* bytes = reinterpret_cast<const unsigned char*>(code.data());
  uint32_t root = uint32_t{bytes[0]} | (uint32_t{bytes[1]} << 8) |
                  (uint32_t{bytes[2]} << 16) | (uint32_t{bytes[3]} << 24);
  return root >= 8 && root % 4 == 0 && root <= code.size() - 4;
}
}  // namespace pixelforge
#endif  // PIXELFORGE_ARM_NPU_PIPE_PROTOCOL_H_
