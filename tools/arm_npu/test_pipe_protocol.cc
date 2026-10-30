// Runs on an ordinary C++ toolchain; it never initializes Android/vendor APIs.
#include "pipe_protocol.h"
#include <cassert>
#include <sstream>
#include <string>

using pixelforge::HeaderStatus;

static HeaderStatus Parse(const std::string& text) {
  std::istringstream source(text);
  pixelforge::ModelHeader header;
  return pixelforge::ReadModelHeader(source, header);
}

int main() {
  assert(Parse("1 1 1717056 8192 1024\n") == HeaderStatus::kOk);
  assert(Parse("2 2 12 8 16 24 32\n") == HeaderStatus::kOk);
  assert(Parse("1 1 2147483648 1073741824 1073741824\n") == HeaderStatus::kOk);
  assert(Parse("0 1 12 8 8\n") == HeaderStatus::kModel);
  assert(Parse("513 1 12\n") == HeaderStatus::kModel);
  assert(Parse("1 1 2147483649 8 8\n") == HeaderStatus::kModel);
  assert(Parse("1 1 -1 8 8\n") == HeaderStatus::kModel);
  assert(Parse("1 1 -18446744073709551615 8 8\n") == HeaderStatus::kModel);
  assert(Parse("1 1 +12 8 8\n") == HeaderStatus::kModel);
  assert(Parse("1 1 18446744073709551616 8 8\n") == HeaderStatus::kModel);
  assert(Parse("1 1 12 8\n") == HeaderStatus::kBuffer);
  assert(Parse("1 1 12 0 8\n") == HeaderStatus::kBuffer);
  assert(Parse("1 1 12 1073741825 8\n") == HeaderStatus::kBuffer);
  assert(Parse("2 1 12 1073741824 1073741824 4\n") == HeaderStatus::kBuffer);
  assert(Parse("1 1 12 8 8 unexpected\n") == HeaderStatus::kBuffer);
  std::istringstream ended("  \n"), trailing(" model\n");
  assert(pixelforge::HeaderEnded(ended));
  assert(!pixelforge::HeaderEnded(trailing));
  for (const auto& item : std::vector<std::pair<std::string, bool>>{
           {"0", true}, {"134217728", true}, {"2147483648", true},
           {"2147483649", false}, {"-1", false}, {"+1", false},
           {"18446744073709551616", false}, {"", false}, {"1x", false}}) {
    std::istringstream source(item.first);
    size_t value = 73;
    assert(pixelforge::ReadUnsignedDecimal(source, pixelforge::kMaxRuntimeCacheBytes,
                                         value) == item.second);
  }
  assert(pixelforge::RoundedBufferBytes(1) == 4096);
  assert(pixelforge::RoundedBufferBytes(4096) == 4096);
  assert(pixelforge::RoundedBufferBytes(4097) == 8192);
  for (const auto& item : std::vector<std::pair<std::string, bool>>{
           {"1", true}, {"511", true}, {"0", false}, {"512", false},
           {"-1", false}, {"+1", false}, {"1 extra", false}, {"", false}}) {
    std::istringstream source(item.first);size_t index=0;
    assert(pixelforge::ReadStickyUploadHeader(source,index)==item.second);
  }
  for (const auto& item : std::vector<std::pair<std::string, bool>>{
           {"1", true}, {"18446744073709551615", true}, {"0", false},
           {"18446744073709551616", false}, {"-1", false}, {"+1", false},
           {"1 extra", false}, {"", false}}) {
    std::istringstream source(item.first);uint64_t generation=0;
    assert(pixelforge::ReadStickyInvokeHeader(source,generation)==item.second);
  }

  std::vector<char> code(12, 0);
  code[0] = 8;
  std::memcpy(code.data() + 4, "DGC0", 4);
  assert(pixelforge::IsDgc0(code));
  auto malformed = code;
  malformed[4] = 'X';
  assert(!pixelforge::IsDgc0(malformed));
  malformed = code;
  malformed[0] = 9;
  assert(!pixelforge::IsDgc0(malformed));
  malformed[0] = 12;
  assert(!pixelforge::IsDgc0(malformed));
  malformed.resize(7);
  assert(!pixelforge::IsDgc0(malformed));
  assert(!pixelforge::IsDgc0({}));
  return 0;
}
