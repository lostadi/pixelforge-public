// PixelForge native ARM64 compiler/inference pipe, based on official LiteRT APIs.
// Protocol: C name nin nout tflite_bytes input_sizes... output_sizes...\n + model
//           L name nin nout bytecode_bytes input_sizes... output_sizes...\n + code
//           D name\n -> bytecode header + raw code
//           R name\n + raw inputs; F name\n; Q\n
// Responses are ASCII lines; successful R is followed by output byte buffers.
#include "edgetpu_compiler_api.h"
#include "sb_api.h"
#include "pipe_protocol.h"
// Exact declaration from AOSP libs/binder/ndk/include_platform/android/binder_process.h.
extern "C" void ABinderProcess_startThreadPool(void);
#include <dlfcn.h>
#include <signal.h>
#include <sys/prctl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#define API_LIST(X) \
 X(EdgeTpuCompilerCreate) X(EdgeTpuCompilerCompileFlatbuffer) X(EdgeTpuCompilerFreeCompiledCode) X(EdgeTpuCompilerFreeErrorMessage) X(EdgeTpuCompilerDestroy) \
 X(thrInitialize) X(thrContextCreate) X(thrContextDelete) X(thrVendorSetSystemAttributeStr) X(thrLoadSqContainer) X(thrUnloadSqContainer) \
 X(thrGraphCreate) X(thrGraphDelete) X(thrGraphAddSqNode) X(thrGraphAssignSq) X(thrGraphAddEdge) X(thrGraphConnectNodeInput) X(thrGraphConnectNodeOutput) X(thrGraphSetInputEdge) X(thrGraphSetOutputEdge) \
 X(thrInvocationContextGet) X(thrInvocationContextDelete) X(thrRegisterBuffer) X(thrUnregisterBuffer) X(thrInvocationContextAttachBuffer) X(thrInvocationContextDetachBuffer) \
 X(thrInvocationContextPrepareForInvoke) X(thrInvocationContextInvokeOnce) X(thrInvocationContextWait) \
 X(AHardwareBuffer_allocate) X(AHardwareBuffer_release) X(AHardwareBuffer_lock) X(AHardwareBuffer_unlock)
struct Api {
#define FIELD(f) decltype(&f) f##_fn=nullptr;
 API_LIST(FIELD)
#undef FIELD
 bool load(){
  void*lib=dlopen("/vendor/lib64/libedgetpu_litert.so",RTLD_NOW|RTLD_LOCAL|RTLD_NODELETE);
  if(!lib){fprintf(stderr,"dlopen: %s\n",dlerror());return false;}
#define LOAD(f) f##_fn=reinterpret_cast<decltype(&f)>(dlsym(lib,#f));if(!f##_fn){fprintf(stderr,"missing %s\n",#f);return false;}
 API_LIST(LOAD)
#undef LOAD
  return true;
 }
};
Api api;
FILE*protocol=nullptr;
#define printf(...) fprintf(protocol,__VA_ARGS__)
#define puts(text) fprintf(protocol,"%s\n",text)
ThrContext*context=nullptr;
struct Buffer {
 AHardwareBuffer*ahwb=nullptr;
 ThrBufferHandle handle=0;
 bool registered=false,attached=false;
 size_t bytes=0;
 std::string edge;
 Buffer(size_t n,std::string e):bytes(n),edge(std::move(e)){}
 Buffer(const Buffer&)=delete;
 ~Buffer(){release();}
 bool release(){
  bool ok=true;
  if(registered){if(api.thrUnregisterBuffer_fn(context,handle)!=0)ok=false;registered=false;}
  if(ahwb){api.AHardwareBuffer_release_fn(ahwb);ahwb=nullptr;}
  return ok;
 }
 bool make(){
  AHardwareBuffer_Desc d={};d.width=(uint32_t)((bytes+4095)&~(size_t)4095);d.height=1;d.layers=1;d.format=AHARDWAREBUFFER_FORMAT_BLOB;
  d.usage=AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN|AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
  if(api.AHardwareBuffer_allocate_fn(&d,&ahwb)!=0)return false;
  registered=api.thrRegisterBuffer_fn(context,kThrBufferTypeAHardwareBuffer,ahwb,0,&handle)==0;
  return registered;
 }
 bool read_input(){
  void*p=nullptr;if(api.AHardwareBuffer_lock_fn(ahwb,AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,-1,nullptr,&p)!=0)return false;
  size_t read=fread(p,1,bytes,stdin);
  int status=api.AHardwareBuffer_unlock_fn(ahwb,nullptr);
  return read==bytes && status==0;
 }
 bool copy_output(std::vector<char>&output){
  output.resize(bytes);
  void*p=nullptr;if(api.AHardwareBuffer_lock_fn(ahwb,AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,-1,nullptr,&p)!=0)return false;
  memcpy(output.data(),p,bytes);
  int status=api.AHardwareBuffer_unlock_fn(ahwb,nullptr);
  return status==0;
 }
};
struct Model {
 static size_t next_id;
 std::string prefix,node;
 std::vector<size_t>input_sizes,output_sizes;
 std::vector<char>code;
 ThrSqContainerHandle sq=0;
 bool sq_loaded=false;
 ThrGraph*graph=nullptr;
 ThrInvocationContext*inv=nullptr;
 std::vector<std::unique_ptr<Buffer>>inputs,outputs;
 ~Model(){
  drop_runtime();
 }
 bool compile(std::vector<char>&flatbuffer){
  EdgeTpuCompilerContext*cc=nullptr;
  int status=api.EdgeTpuCompilerCreate_fn(&cc);if(status!=0||!cc){if(cc)api.EdgeTpuCompilerDestroy_fn(cc);return false;}
  char**codes=nullptr;size_t*sizes=nullptr;size_t count=0;char*error=nullptr;const char opts[1]={0};
  status=api.EdgeTpuCompilerCompileFlatbuffer_fn(cc,flatbuffer.data(),flatbuffer.size(),opts,0,&codes,&sizes,&count,&error);
  if(error){fprintf(stderr,"compile error (%d): %s\n",status,error);api.EdgeTpuCompilerFreeErrorMessage_fn(cc,error);}
  bool ok=status==0 && count==1 && codes && codes[0] && sizes && sizes[0];
  if(ok)code.assign(codes[0],codes[0]+sizes[0]);
  if(codes||sizes)api.EdgeTpuCompilerFreeCompiledCode_fn(cc,codes,sizes,count);
  api.EdgeTpuCompilerDestroy_fn(cc);
  return ok;
 }
 bool make(const std::vector<size_t>&ins,const std::vector<size_t>&outs){
  input_sizes=ins;output_sizes=outs;
  return load_runtime(false) && drop_runtime();
 }
 bool load_runtime(bool create_inv=true){
  if(inv)return true;
  prefix="pixel_model_"+std::to_string(++next_id);node=prefix+"_node";
  if(api.thrLoadSqContainer_fn(context,kThrSqContainerTypeMlModel,code.data(),code.size(),&sq)!=0)return false;
  sq_loaded=true;
  graph=api.thrGraphCreate_fn(context);if(!graph)return false;
  if(api.thrGraphAddSqNode_fn(graph,node.c_str(),kThrNodeTypeNpu)!=0 || api.thrGraphAssignSq_fn(graph,node.c_str(),sq,nullptr)!=0)return false;
  for(size_t k=0;k<input_sizes.size();k++){
   inputs.emplace_back(std::make_unique<Buffer>(input_sizes[k],prefix+"_input_"+std::to_string(k)));
   const char*e=inputs.back()->edge.c_str();
   if(api.thrGraphAddEdge_fn(graph,e,kThrEdgeTypeTensor)!=0 || api.thrGraphConnectNodeInput_fn(graph,node.c_str(),e)!=0 || api.thrGraphSetInputEdge_fn(graph,e)!=0)return false;
  }
  for(size_t k=0;k<output_sizes.size();k++){
   outputs.emplace_back(std::make_unique<Buffer>(output_sizes[k],prefix+"_output_"+std::to_string(k)));
   const char*e=outputs.back()->edge.c_str();
   if(api.thrGraphAddEdge_fn(graph,e,kThrEdgeTypeTensor)!=0 || api.thrGraphConnectNodeOutput_fn(graph,node.c_str(),e)!=0 || api.thrGraphSetOutputEdge_fn(graph,e)!=0)return false;
  }
  for(auto&b:inputs)if(!b->make())return false;
  for(auto&b:outputs)if(!b->make())return false;
  return !create_inv || ensure_inv();
 }
 bool ensure_inv(){
  if(inv)return true;
  inv=api.thrInvocationContextGet_fn(graph,context);if(!inv)return false;
  for(auto&b:inputs){if(api.thrInvocationContextAttachBuffer_fn(inv,context,b->edge.c_str(),b->handle)!=0)return false;b->attached=true;}
  for(auto&b:outputs){if(api.thrInvocationContextAttachBuffer_fn(inv,context,b->edge.c_str(),b->handle)!=0)return false;b->attached=true;}
  return true;
 }
 bool drop_inv(){
  if(!inv)return true;
  bool ok=true;
  for(auto&b:inputs)if(b->attached){if(api.thrInvocationContextDetachBuffer_fn(inv,context,b->edge.c_str(),b->handle)!=0)ok=false;b->attached=false;}
  for(auto&b:outputs)if(b->attached){if(api.thrInvocationContextDetachBuffer_fn(inv,context,b->edge.c_str(),b->handle)!=0)ok=false;b->attached=false;}
  if(api.thrInvocationContextDelete_fn(graph,inv)!=0)ok=false;
  inv=nullptr;return ok;
 }
 bool drop_runtime(){
  bool ok=drop_inv();
  for(auto&b:inputs)if(!b->release())ok=false;
  for(auto&b:outputs)if(!b->release())ok=false;
  if(graph){if(api.thrGraphDelete_fn(graph)!=0)ok=false;graph=nullptr;}
  // Keep edge-name strings alive through graph deletion; the vendor may retain them.
  inputs.clear();outputs.clear();
  if(sq_loaded){
   if(api.thrUnloadSqContainer_fn(context,sq)!=0){
    // The vendor still owns a reference to code. Do not destroy its owner,
    // reuse this context, or run destructors after an unsuccessful unload.
    puts("ERR SQ_UNLOAD");fflush(protocol);std::_Exit(2);
   }
   sq_loaded=false;sq=0;
  }
  return ok;
 }
 bool invoke(){return ensure_inv() && api.thrInvocationContextPrepareForInvoke_fn(inv,false)==0 && api.thrInvocationContextInvokeOnce_fn(inv)==0 && api.thrInvocationContextWait_fn(inv)==0;}
};
size_t Model::next_id=0;
int main(){
 // A killed broker must not leave its separate native inference process alive.
 const pid_t parent_pid=getppid();
 if(parent_pid==1 || prctl(PR_SET_PDEATHSIG,SIGKILL)!=0 || getppid()!=parent_pid)return 3;
 int output_fd=dup(STDOUT_FILENO);if(output_fd<0)return 3;
 if(dup2(STDERR_FILENO,STDOUT_FILENO)<0)return 3;
 protocol=fdopen(output_fd,"wb");if(!protocol)return 3;
 setvbuf(protocol,nullptr,_IONBF,0);
 if(!api.load())return 3;
 auto start_pool=reinterpret_cast<decltype(&ABinderProcess_startThreadPool)>(dlsym(RTLD_DEFAULT,"ABinderProcess_startThreadPool"));
 if(start_pool)start_pool();
 if(api.thrInitialize_fn()!=0)return 3;
 context=api.thrContextCreate_fn();if(!context)return 3;
 if(api.thrVendorSetSystemAttributeStr_fn(context,"edgetpu_use_tpu_tachyon","1")!=0 || api.thrVendorSetSystemAttributeStr_fn(context,"sq_container_load_method","dmabuf")!=0)return 3;
 printf("READY PIXELFORGE_NPU_PIPE 1\n");
 std::map<std::string,std::unique_ptr<Model>>models;
 int exit_status=0;
 bool quit_requested=false;
 char line[65536];
 while(fgets(line,sizeof(line),stdin)){
  if(!strchr(line,'\n')){puts("ERR HEADER_SIZE");exit_status=2;break;}
  std::istringstream s(line);char op=0;s>>op;
  if(op=='Q'){
   if(!pixelforge::HeaderEnded(s)){puts("ERR COMMAND_HEADER");exit_status=2;break;}
   quit_requested=true;break;
  }
  std::string name;s>>name;if(name.empty()||name.size()>128){puts("ERR NAME");exit_status=2;break;}
  if((op=='R'||op=='D'||op=='F'||op=='I')&&!pixelforge::HeaderEnded(s)){puts("ERR COMMAND_HEADER");exit_status=2;break;}
  if(op=='F'){models.erase(name);puts("OK FREED");continue;}
  if(op=='I'){
   if(name=="CAPABILITIES")puts("OK CAPABILITIES C R F I D L Q");
   else{auto it=models.find(name);if(it==models.end())puts("ERR UNKNOWN_MODEL");else printf("OK INFO %zu %d\n",it->second->code.size(),it->second->inv?1:0);}
   continue;
  }
  if(op=='D'){
   auto it=models.find(name);if(it==models.end()){puts("ERR UNKNOWN_MODEL");exit_status=2;break;}
   const auto&code=it->second->code;
   printf("OK BYTECODE %zu\n",code.size());
   if(fwrite(code.data(),1,code.size(),protocol)!=code.size()){exit_status=2;break;}
   continue;
  }
  if(op=='C'||op=='L'){
   pixelforge::ModelHeader header;
   auto status=pixelforge::ReadModelHeader(s,header);
   if(status==pixelforge::HeaderStatus::kModel){puts(op=='C'?"ERR COMPILE_HEADER":"ERR LOAD_HEADER");exit_status=2;break;}
   if(status!=pixelforge::HeaderStatus::kOk){puts("ERR BUFFER_HEADER");exit_status=2;break;}
   std::vector<char>payload(header.payload_bytes);
   if(fread(payload.data(),1,payload.size(),stdin)!=payload.size()){puts(op=='C'?"ERR MODEL_INPUT":"ERR BYTECODE_INPUT");exit_status=2;break;}
   models.erase(name);
   auto m=std::make_unique<Model>();
   if(op=='C'){
    if(!m->compile(payload)){puts("ERR COMPILE");continue;}
    payload.clear();payload.shrink_to_fit();
   }else{
    if(!pixelforge::IsDgc0(payload)){puts("ERR BYTECODE_FORMAT");continue;}
    m->code=std::move(payload);
   }
   if(!m->make(header.input_bytes,header.output_bytes)){puts("ERR MODEL_LOAD");continue;}
   size_t bytes=m->code.size();
   models[name]=std::move(m);printf("OK %s %s %zu\n",op=='C'?"COMPILED":"LOADED",name.c_str(),bytes);
  }else if(op=='R'){
   auto it=models.find(name);if(it==models.end()){puts("ERR UNKNOWN_MODEL");exit_status=2;break;}
   Model&m=*it->second;if(!m.load_runtime()){puts("ERR MODEL_RELOAD");exit_status=2;break;}
   bool ok=true;for(auto&b:m.inputs)if(!b->read_input()){ok=false;break;}
   if(!ok){puts("ERR TENSOR_INPUT");exit_status=2;break;}
   if(!m.invoke()){puts("ERR INVOKE");exit_status=2;break;}
   size_t total=0;
   std::vector<std::vector<char>>result;
   for(auto&b:m.outputs){total+=b->bytes;result.emplace_back();if(!b->copy_output(result.back())){ok=false;break;}}
   if(!ok){puts("ERR OUTPUT_BUFFER");exit_status=2;break;}
   if(!m.drop_runtime()){puts("ERR RUNTIME_CLEANUP");exit_status=2;break;}
   printf("OK OUTPUT %zu %zu\n",result.size(),total);
   for(const auto&bytes:result)if(fwrite(bytes.data(),1,bytes.size(),protocol)!=bytes.size())ok=false;
   if(!ok){exit_status=2;break;}
  }else {puts("ERR COMMAND");exit_status=2;break;}
 }
 if(ferror(stdin))exit_status=2;
 for(auto&model:models)if(!model.second->drop_runtime())exit_status=2;
 models.clear();if(api.thrContextDelete_fn(context)!=0)exit_status=2;
 if(quit_requested)puts(exit_status==0?"OK QUIT":"ERR SHUTDOWN_CLEANUP");
 return exit_status;
}
