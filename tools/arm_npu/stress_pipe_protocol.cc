// Deterministic API-free stress runner; never loads a vendor/runtime library.
#include "pipe_protocol.h"
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <locale>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using pixelforge::HeaderStatus;
static bool Space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f';
}
static std::vector<std::string> Tokens(const std::string& text) {
  std::vector<std::string> result;
  for (size_t at = 0; at < text.size();) {
    while (at < text.size() && Space(text[at])) ++at;
    const size_t start = at;
    while (at < text.size() && !Space(text[at])) ++at;
    if (at != start) result.push_back(text.substr(start, at - start));
  }
  return result;
}
// Independent reference: trim leading zeros, compare decimal text against
// the maximum lexicographically, then convert only a proven in-range token.
static bool Decimal(const std::string& token, size_t maximum, size_t& result) {
  if (token.empty()) return false;
  for (unsigned char c : token) if (c < '0' || c > '9') return false;
  const size_t first = token.find_first_not_of('0');
  if (first == std::string::npos) return false;
  const std::string significant = token.substr(first), limit = std::to_string(maximum);
  if (significant.size() > limit.size() ||
      (significant.size() == limit.size() && significant > limit)) return false;
  result = static_cast<size_t>(std::stoull(significant));
  return true;
}
struct ReferenceHeader {
  HeaderStatus status = HeaderStatus::kModel;
  size_t payload = 0;
  std::vector<size_t> inputs, outputs;
};
static ReferenceHeader Reference(const std::string& text) {
  auto parts = Tokens(text);
  ReferenceHeader result;
  size_t inputs = 0, outputs = 0;
  if (parts.size() < 3 || !Decimal(parts[0], pixelforge::kMaxPorts, inputs) ||
      !Decimal(parts[1], pixelforge::kMaxPorts, outputs) ||
      !Decimal(parts[2], pixelforge::kMaxPayloadBytes, result.payload)) return result;
  result.status = HeaderStatus::kBuffer;
  if (parts.size() != 3 + inputs + outputs) return result;
  uint64_t total = 0;
  for (size_t index = 3; index < parts.size(); ++index) {
    size_t bytes = 0;
    if (!Decimal(parts[index], pixelforge::kMaxTensorBytes, bytes)) return result;
    total += bytes;
    if (total > pixelforge::kMaxPayloadBytes) return result;
    (index < 3 + inputs ? result.inputs : result.outputs).push_back(bytes);
  }
  result.status = HeaderStatus::kOk;
  return result;
}
static std::string Hex(const std::string& text) {
  std::ostringstream stream;
  for (unsigned char c : text.substr(0, 512))
    stream << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
  return stream.str();
}
struct Stress {
  uint64_t seed, cases = 0, failures = 0;
  std::array<uint64_t, 6> families{};
  std::vector<std::string> diagnostics;
  void Fail(const std::string& label, const std::string& input) {
    ++failures;
    if (diagnostics.size() < 20)
      diagnostics.push_back(label + ":case=" + std::to_string(cases) +
                            ":input_hex=" + Hex(input) + ":bytes=" + std::to_string(input.size()));
  }
  void Model(const std::string& text) {
    std::istringstream input(text);
    pixelforge::ModelHeader actual;
    auto status = pixelforge::ReadModelHeader(input, actual);
    auto expected = Reference(text);
    if (status != expected.status || (status == HeaderStatus::kOk &&
        (actual.payload_bytes != expected.payload || actual.input_bytes != expected.inputs ||
         actual.output_bytes != expected.outputs))) Fail("model_header", text);
  }
  void Number(const std::string& text, size_t maximum) {
    auto parts = Tokens(text);
    size_t expected = 0, actual = 0;
    bool reference = !parts.empty() && Decimal(parts.front(), maximum, expected);
    std::istringstream input(text);
    bool parsed = pixelforge::ReadPositiveDecimal(input, maximum, actual);
    if (parsed != reference || (parsed && actual != expected))
      Fail("decimal_max=" + std::to_string(maximum), text);
  }
  void End(const std::string& text) {
    bool reference = true;
    for (unsigned char c : text) if (!Space(c)) reference = false;
    std::istringstream input(text);
    if (pixelforge::HeaderEnded(input) != reference) Fail("header_end", text);
  }
  void Code(const std::vector<char>& bytes) {
    // The runner targets the documented little-endian ARM/x86 transport.
    uint32_t root = 0;
    if (bytes.size() >= 4) std::memcpy(&root, bytes.data(), sizeof(root));
    bool reference = bytes.size() >= 12 && bytes.size() <= pixelforge::kMaxPayloadBytes &&
        bytes[4] == 'D' && bytes[5] == 'G' && bytes[6] == 'C' && bytes[7] == '0' &&
        root >= 8 && root % 4 == 0 && root <= bytes.size() - 4;
    if (pixelforge::IsDgc0(bytes) != reference)
      Fail("dgc0", std::string(bytes.begin(), bytes.end()));
  }
};
static std::string Join(const std::vector<std::string>& tokens) {
  std::string text;
  for (const auto& token : tokens) text += token + " ";
  return text + "\n";
}
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::locale::global(std::locale::classic());
  const uint64_t seed = std::stoull(argv[1]), iterations = std::stoull(argv[2]);
  if (iterations < 100000 || iterations > 10000000) return 2;
  const auto started = std::chrono::steady_clock::now();
  Stress stress{seed, 0, 0, {}, {}};
  std::mt19937_64 random(seed);
  const std::array<size_t, 7> maxima = {0, 1, 9, 512, pixelforge::kMaxTensorBytes,
                                      pixelforge::kMaxPayloadBytes, SIZE_MAX};
  const std::vector<std::string> bad = {"", "0", "+1", "-1", "-18446744073709551615",
      "18446744073709551616", "9999999999999999999999999999999999999999", "1x", "0x10",
      "1.0", std::string("1\0", 2), "\xc2\xa0", "\xff"};
  for (size_t maximum : maxima) {
    for (const auto& token : bad) { stress.Number(token, maximum); ++stress.cases; }
    stress.Number(std::string(1024, '0') + "1", maximum); ++stress.cases;
    stress.Number(std::to_string(maximum), maximum); ++stress.cases;
  }
  const std::vector<std::string> edges = {
    "1 1 12 1 1", "512 512 12" + std::string(" "),
    "1 1 2147483648 1073741824 1073741824", "1 1 2147483649 1 1",
    "1 1 -18446744073709551615 1 1", "1 1 +12 1 1", "0 1 12 1 1",
    "513 1 12 1 1", "1 1 12 1", "1 1 12 1 1 EXTRA",
    "2 1 12 1073741824 1073741824 1", "1 1 12 1073741825 1",
    "1 1 12 0 1", "1 1 12 00001 00001", std::string("1 1 12 1 1") + std::string("\0", 1)};
  for (const auto& text : edges) { stress.Model(text); ++stress.cases; }
  const uint64_t fixed = stress.cases;
  for (uint64_t index = 0; index < iterations; ++index) {
    const size_t family = random() % stress.families.size();
    ++stress.families[family]; ++stress.cases;
    if (family == 0) {
      size_t maximum = maxima[random() % maxima.size()];
      std::string token;
      if (random() % 3 == 0) token = bad[random() % bad.size()];
      else if (random() % 2 == 0) {
        const uint64_t value = maximum == SIZE_MAX ? random() : random() % (maximum + 1);
        token = std::string(random() % 12, '0') + std::to_string(value);
      } else {
        const std::string alphabet = "0123456789+-x._e";
        for (size_t length = random() % 90; length; --length) token += alphabet[random() % alphabet.size()];
      }
      stress.Number(token, maximum);
    } else if (family == 1 || family == 2) {
      const std::array<size_t, 5> counts = {1, 2, 3, 8, 512};
      const size_t inputs = counts[random() % counts.size()], outputs = counts[random() % counts.size()];
      std::vector<std::string> tokens = {std::to_string(inputs), std::to_string(outputs),
          std::to_string(1 + random() % pixelforge::kMaxPayloadBytes)};
      const size_t cap = std::min(pixelforge::kMaxTensorBytes, pixelforge::kMaxPayloadBytes / (inputs + outputs));
      for (size_t port = 0; port < inputs + outputs; ++port)
        tokens.push_back(std::to_string(1 + random() % cap));
      if (family == 2) {
        switch (random() % 6) {
          case 0: tokens[random() % 3] = bad[random() % bad.size()]; break;
          case 1: tokens[3 + random() % (inputs + outputs)] = bad[random() % bad.size()]; break;
          case 2: tokens[3 + random() % (inputs + outputs)] = "1073741825"; break;
          case 3: tokens.resize(random() % tokens.size()); break;
          case 4: tokens.push_back("EXTRA"); break;
          default: tokens[0] = "513"; break;
        }
      }
      stress.Model(Join(tokens));
    } else if (family == 3) {
      const std::array<uint64_t, 8> dimensions = {1, 16, 256, 2048, 4096, 6144, 1ULL << 28, 1ULL << 30};
      uint64_t columns = dimensions[random() % dimensions.size()], rows = dimensions[random() % dimensions.size()];
      uint64_t vectors = dimensions[random() % 6];
      stress.Model("1 1 12 " + std::to_string(columns * vectors * 4) + " " +
                   std::to_string(rows * vectors * 4));
    } else if (family == 4) {
      std::vector<char> code(12 + 4 * (random() % 1024), 0);
      uint32_t root = 8 + 4 * (random() % ((code.size() - 8) / 4));
      std::memcpy(code.data(), &root, sizeof(root));
      std::memcpy(code.data() + 4, "DGC0", 4);
      switch (random() % 5) {
        case 0: code[4 + random() % 4] ^= 1; break;
        case 1: ++code[0]; break;
        case 2: root = uint32_t(code.size()); std::memcpy(code.data(), &root, sizeof(root)); break;
        case 3: code.resize(random() % 12); break;
        default: break;
      }
      stress.Code(code);
    } else {
      const std::string spaces = " \t\n\r\v\f";
      std::string text;
      for (size_t length = random() % 128; length; --length) text += spaces[random() % spaces.size()];
      if (random() % 2) text += random() % 2 ? "EXTRA" : std::string("\0", 1);
      stress.End(text);
    }
  }
  const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  std::cout << "{\"seed\":" << seed << ",\"random_cases\":" << iterations
            << ",\"fixed_edge_cases\":" << fixed << ",\"cases\":" << stress.cases
            << ",\"elapsed_seconds\":" << elapsed << ",\"failures\":" << stress.failures
            << ",\"family_counts\":{\"decimal\":" << stress.families[0]
            << ",\"valid_model_headers\":" << stress.families[1]
            << ",\"mutated_model_headers\":" << stress.families[2]
            << ",\"matrix_buffer_bounds\":" << stress.families[3]
            << ",\"dgc0\":" << stress.families[4]
            << ",\"command_tail\":" << stress.families[5] << "},\"failure_examples\":[";
  for (size_t index = 0; index < stress.diagnostics.size(); ++index)
    std::cout << (index ? "," : "") << "\"" << stress.diagnostics[index] << "\"";
  std::cout << "]}\n";
  return stress.failures ? 1 : 0;
}
