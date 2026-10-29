// The real Model implementation runs with API stubs, never dlopen/vendor init.
#define main pixelforge_native_entry_not_called
#include "npu_pipe.cc"
#undef main
#undef printf
#undef puts
static Model* owner = nullptr;
static bool fail_unload = true;
static ThrSqContainerHandle expected_handle = 73;
static unsigned unload_calls = 0, graph_deletes = 0, invocation_deletes = 0;
static bool expect_invocation_cleanup = false;
static void normal_exit() { std::fputs("MOCK ATEXIT\n", stderr); }
struct Sentinel { ~Sentinel() { std::fputs("MOCK DESTRUCTOR\n", stderr); } };
static bool OwnerValid() {
  return owner && owner->sq_loaded && owner->code.size() == 8 &&
         std::memcmp(owner->code.data(), "DGC0LIVE", 8) == 0;
}
static ThrStatus Load(ThrContext*, ThrSqContainerType, const void* code,
                      size_t bytes, ThrSqContainerHandle* output) {
  if (!owner || code != owner->code.data() || bytes != 8) std::_Exit(90);
  *output = expected_handle;
  return static_cast<ThrStatus>(0);
}
static ThrGraph* Graph(ThrContext*) { return nullptr; }
static ThrStatus Unload(ThrContext*, ThrSqContainerHandle handle) {
  if (!OwnerValid() || handle != expected_handle || ++unload_calls != 1) std::_Exit(91);
  if (expect_invocation_cleanup && (graph_deletes != 1 || invocation_deletes != 1)) std::_Exit(98);
  std::fputs("MOCK OWNER_VALID\n", stderr);
  return static_cast<ThrStatus>(fail_unload ? 1 : 0);
}
static ThrStatus DeleteInvocation(ThrGraph*, ThrInvocationContext*) {
  if (!OwnerValid()) std::_Exit(92);
  ++invocation_deletes;
  return static_cast<ThrStatus>(0);
}
static ThrStatus DeleteGraph(ThrGraph*) {
  if (!OwnerValid()) std::_Exit(93);
  ++graph_deletes;
  return static_cast<ThrStatus>(0);
}
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  protocol = stdout;
  api.thrLoadSqContainer_fn = &Load;
  api.thrGraphCreate_fn = &Graph;
  api.thrUnloadSqContainer_fn = &Unload;
  api.thrInvocationContextDelete_fn = &DeleteInvocation;
  api.thrGraphDelete_fn = &DeleteGraph;
  std::atexit(normal_exit);
  Sentinel sentinel;
  auto model = std::make_unique<Model>();
  owner = model.get();
  model->code.assign({'D','G','C','0','L','I','V','E'});
  std::string scenario(argv[1]);
  if (scenario == "make-failure") {
    if (model->make({4}, {4})) return 94;
    std::fputs("MOCK MAKE_FAILED\n", stderr);
    return 0; // Destructor must stop before bytecode ownership is released.
  }
  if (scenario == "zero-handle-failure") expected_handle = 0;
  model->sq = expected_handle;
  model->sq_loaded = true;
  if (scenario == "invocation-cleanup-failure") {
    expect_invocation_cleanup = true;
    model->graph = reinterpret_cast<ThrGraph*>(uintptr_t{1});
    model->inv = reinterpret_cast<ThrInvocationContext*>(uintptr_t{2});
  }
  if (scenario == "replacement-failure") {
    model.reset();
    std::fputs("MOCK RETURNED_AFTER_ERROR\n", stderr);
    return 95;
  }
  fail_unload = scenario != "success" && scenario != "double-success";
  if (!model->drop_runtime() || model->sq_loaded || model->sq != 0 ||
      model->code.size() != 8 || unload_calls != 1) return 96;
  if (scenario == "double-success" && (!model->drop_runtime() || unload_calls != 1)) return 97;
  std::puts("MOCK RETURNED_SUCCESS");
  return 0;
}
