// PixelForge native ARM64 compiler/inference pipe, based on official LiteRT APIs.
// Protocol: C name nin nout tflite_bytes input_sizes... output_sizes...\n + model
//           L name nin nout bytecode_bytes input_sizes... output_sizes...\n + code
//           D name\n -> bytecode header + raw code
//           R name\n + raw inputs; F name\n; B runtime_cache_bytes\n; Q\n
//           U name weight_port\n + raw port; V name generation\n + raw input0
//           I MODEL name\n -> generation, loaded, port count, sticky-ready bitmap
//           Z name weight_port logical_bytes\n + identical SCM_RIGHTS socket packet
// Responses are ASCII lines; successful R is followed by output byte buffers.
#include "edgetpu_compiler_api.h"
#include "sb_api.h"
#include "pipe_protocol.h"
// Exact declaration from AOSP libs/binder/ndk/include_platform/android/binder_process.h.
extern "C" void ABinderProcess_startThreadPool(void);
#include <dlfcn.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include <fcntl.h>
#include <climits>
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
 decltype(&thrGraphAnnotateGraph) thrGraphAnnotateGraph_fn=nullptr;
 decltype(&thrGetVendorApiVersion) thrGetVendorApiVersion_fn=nullptr;
 bool load(){
  void*lib=dlopen("/vendor/lib64/libedgetpu_litert.so",RTLD_NOW|RTLD_LOCAL|RTLD_NODELETE);
  if(!lib){fprintf(stderr,"dlopen: %s\n",dlerror());return false;}
#define LOAD(f) f##_fn=reinterpret_cast<decltype(&f)>(dlsym(lib,#f));if(!f##_fn){fprintf(stderr,"missing %s\n",#f);return false;}
 API_LIST(LOAD)
#undef LOAD
  thrGraphAnnotateGraph_fn=reinterpret_cast<decltype(&thrGraphAnnotateGraph)>(dlsym(lib,"thrGraphAnnotateGraph"));
  thrGetVendorApiVersion_fn=reinterpret_cast<decltype(&thrGetVendorApiVersion)>(dlsym(lib,"thrGetVendorApiVersion"));
  return true;
 }
};
Api api;
FILE*protocol=nullptr;
#define printf(...) fprintf(protocol,__VA_ARGS__)
#define puts(text) fprintf(protocol,"%s\n",text)
ThrContext*context=nullptr;
[[noreturn]] void RuntimeFatal(const char*reason){
 // A failing vendor release may still retain references to dependent objects.
 // Exit without destructors instead of freeing their owners or reusing context.
 printf("ERR %s\n",reason);fflush(protocol);std::_Exit(2);
}
struct PerformancePolicy {
 bool initialized=false,supported=false,valid=false;
 std::string name="default";
 int mode=0;
 static bool version_supported(const char*raw){
  if(!raw)return false;
  const std::string version(raw);
  if(version.empty() || version.size()>256)return false;
  const size_t suffix=version.find_first_of("-+");
  const std::string core=version.substr(0,suffix);
  size_t numbers[3]{},begin=0;
  for(size_t k=0;k<3;++k){
   const size_t end=k==2?core.size():core.find('.',begin);
   if(end==std::string::npos || end<=begin)return false;
   const std::string token=core.substr(begin,end-begin);
   if(token.find_first_not_of("0123456789")!=std::string::npos)return false;
   std::istringstream source(token);
   if(!pixelforge::ReadUnsignedDecimal(source,size_t{UINT_MAX},numbers[k]))return false;
   begin=end+1;
  }
  // LiteRT's kPerformanceModeAnnotation feature requires southbound >=0.18.
  return numbers[0]>0 || numbers[1]>=18;
 }
 bool configure(){
  if(initialized)return valid;
  initialized=true;
  supported=api.thrGraphAnnotateGraph_fn && api.thrGetVendorApiVersion_fn &&
   version_supported(api.thrGetVendorApiVersion_fn());
  if(const char*setting=std::getenv("PIXELFORGE_NPU_PERFORMANCE_MODE")){
   name=setting;
   // Values match LiteRT's LiteRtGoogleTensorOptionsPerformanceMode enum.
   if(name=="balanced")mode=2;
   else if(name=="high")mode=3;
   else if(name=="sustained")mode=4;
   else return false;
   if(!supported)return false;
  }
  valid=true;return true;
 }
 void annotate(ThrGraph*graph)const{
  if(mode && api.thrGraphAnnotateGraph_fn(graph,"edgetpu_performance_mode",std::to_string(mode).c_str())!=0)
   RuntimeFatal("PERFORMANCE_MODE");
 }
};
PerformancePolicy performance_policy;
static int(*close_fd_fn)(int)=::close;
class UniqueFd {
 int value=-1;
public:
 explicit UniqueFd(int fd=-1):value(fd){}
 ~UniqueFd(){reset();}
 UniqueFd(const UniqueFd&)=delete;
 UniqueFd&operator=(const UniqueFd&)=delete;
 UniqueFd(UniqueFd&&other)noexcept:value(other.release()){}
 UniqueFd&operator=(UniqueFd&&other)noexcept{reset(other.release());return *this;}
 int get()const{return value;}
 int release(){int fd=value;value=-1;return fd;}
 void reset(int fd=-1){if(value>=0)close_fd_fn(value);value=fd;}
};
struct DmaChannel {
 UniqueFd socket;
 bool initialized=false,valid=false,defer_weight_allocation=false;
 bool configure(){
  if(initialized)return valid;
  initialized=true;
  const char*deferred=std::getenv("PIXELFORGE_NPU_DEFER_WEIGHT_ALLOCATION");
  if(deferred){
   if(std::strcmp(deferred,"0")!=0 && std::strcmp(deferred,"1")!=0)return false;
   defer_weight_allocation=std::strcmp(deferred,"1")==0;
  }
  const char*setting=std::getenv("PIXELFORGE_NPU_FD_SOCKET");
  if(!setting){valid=!defer_weight_allocation;return valid;}
  const std::string text(setting);
  if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos)return false;
  std::istringstream source(text);size_t number=0;
  if(!pixelforge::ReadUnsignedDecimal(source,size_t{INT_MAX},number) || number<3)return false;
  const int fd=static_cast<int>(number);
  int type=0;socklen_t size=sizeof(type);
  if(getsockopt(fd,SOL_SOCKET,SO_TYPE,&type,&size)!=0 || type!=SOCK_SEQPACKET)return false;
  sockaddr_un local{},peer{};socklen_t local_size=sizeof(local),peer_size=sizeof(peer);
  if(getsockname(fd,reinterpret_cast<sockaddr*>(&local),&local_size)!=0 ||
     getpeername(fd,reinterpret_cast<sockaddr*>(&peer),&peer_size)!=0 ||
     local.sun_family!=AF_UNIX || peer.sun_family!=AF_UNIX ||
     local_size!=offsetof(sockaddr_un,sun_path) || peer_size!=offsetof(sockaddr_un,sun_path))return false;
  struct {pid_t pid;uid_t uid;gid_t gid;} credentials{};size=sizeof(credentials);
  if(getsockopt(fd,SOL_SOCKET,SO_PEERCRED,&credentials,&size)!=0 || size!=sizeof(credentials) ||
     credentials.uid!=geteuid() || credentials.pid!=getppid())return false;
  const int flags=fcntl(fd,F_GETFD);
  if(flags<0 || fcntl(fd,F_SETFD,flags|FD_CLOEXEC)!=0)return false;
  socket.reset(fd);valid=true;return true;
 }
 bool receive(const std::string&expected,UniqueFd&result){
  if(socket.get()<0)return false;
  char payload[512];alignas(cmsghdr) char control[CMSG_SPACE(sizeof(int)*16)]{};
  iovec data{payload,sizeof(payload)};msghdr message{};
  message.msg_iov=&data;message.msg_iovlen=1;message.msg_control=control;message.msg_controllen=sizeof(control);
  const ssize_t count=recvmsg(socket.get(),&message,MSG_DONTWAIT|MSG_CMSG_CLOEXEC);
  if(count<0)return false;
  std::vector<UniqueFd>descriptors;bool valid_control=true;
  for(cmsghdr*part=CMSG_FIRSTHDR(&message);part;part=CMSG_NXTHDR(&message,part)){
   if(part->cmsg_level!=SOL_SOCKET || part->cmsg_type!=SCM_RIGHTS || part->cmsg_len<CMSG_LEN(0)){
    valid_control=false;continue;
   }
   const size_t bytes=part->cmsg_len-CMSG_LEN(0);
   if(bytes%sizeof(int))valid_control=false;
   const int*fds=reinterpret_cast<const int*>(CMSG_DATA(part));
   for(size_t i=0;i<bytes/sizeof(int);++i)descriptors.emplace_back(fds[i]);
  }
  if(!valid_control || (message.msg_flags&(MSG_TRUNC|MSG_CTRUNC)) || descriptors.size()!=1 ||
     count!=static_cast<ssize_t>(expected.size()) || std::memcmp(payload,expected.data(),expected.size())!=0)return false;
  const int flags=fcntl(descriptors[0].get(),F_GETFD);
  if(flags<0 || fcntl(descriptors[0].get(),F_SETFD,flags|FD_CLOEXEC)!=0)return false;
  result=std::move(descriptors[0]);return true;
 }
};
DmaChannel dma_channel;
static int RealDmaSync(int fd,uint64_t flags){
 dma_buf_sync sync{flags};return ioctl(fd,DMA_BUF_IOCTL_SYNC,&sync);
}
// Mock tests replace only this syscall adapter; production uses kernel DMA sync.
static int(*dma_sync_fn)(int,uint64_t)=RealDmaSync;
static bool ValidateDmaFd(int fd,size_t bytes){
 const size_t allocation=pixelforge::RoundedBufferBytes(bytes);struct stat info{};
 if(fstat(fd,&info)!=0 || info.st_size<0 || static_cast<uint64_t>(info.st_size)!=allocation ||
    lseek(fd,0,SEEK_END)!=static_cast<off_t>(allocation) || lseek(fd,0,SEEK_SET)!=0)return false;
 if(dma_sync_fn(fd,DMA_BUF_SYNC_START|DMA_BUF_SYNC_READ)!=0)return false;
 return dma_sync_fn(fd,DMA_BUF_SYNC_END|DMA_BUF_SYNC_READ)==0;
}
struct Buffer {
 AHardwareBuffer*ahwb=nullptr;
 UniqueFd dma_fd;
 ThrBufferHandle handle=0;
 bool registered=false,attached=false;
 size_t bytes=0;
 std::string edge;
 Buffer(size_t n,std::string e):bytes(n),edge(std::move(e)){}
 Buffer(const Buffer&)=delete;
 ~Buffer(){release();}
 bool release(){
  if(attached)RuntimeFatal("BUFFER_STILL_ATTACHED");
  if(registered){if(api.thrUnregisterBuffer_fn(context,handle)!=0)RuntimeFatal("BUFFER_UNREGISTER");registered=false;}
  if(ahwb){api.AHardwareBuffer_release_fn(ahwb);ahwb=nullptr;}
  dma_fd.reset();
  return true;
 }
 bool import_dma(UniqueFd descriptor){
  if(registered || ahwb || dma_fd.get()>=0 || descriptor.get()<0)return false;
  dma_fd=std::move(descriptor);int fd=dma_fd.get();
  registered=api.thrRegisterBuffer_fn(context,kThrBufferTypeDmaBuf,&fd,bytes,&handle)==0;
  return registered;
 }
 bool make(){
  AHardwareBuffer_Desc d={};d.width=(uint32_t)pixelforge::RoundedBufferBytes(bytes);d.height=1;d.layers=1;d.format=AHARDWAREBUFFER_FORMAT_BLOB;
  d.usage=AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN|AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
  if(api.AHardwareBuffer_allocate_fn(&d,&ahwb)!=0)return false;
  registered=api.thrRegisterBuffer_fn(context,kThrBufferTypeAHardwareBuffer,ahwb,0,&handle)==0;
  return registered;
 }
 bool read_input(){
  if(dma_fd.get()>=0 || !ahwb)return false;
  void*p=nullptr;if(api.AHardwareBuffer_lock_fn(ahwb,AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN,-1,nullptr,&p)!=0)return false;
  size_t read=fread(p,1,bytes,stdin);
  int status=api.AHardwareBuffer_unlock_fn(ahwb,nullptr);
  if(status!=0)RuntimeFatal("BUFFER_UNLOCK");
  return read==bytes;
 }
 bool copy_output(std::vector<char>&output){
  if(dma_fd.get()>=0 || !ahwb)return false;
  output.resize(bytes);
  void*p=nullptr;if(api.AHardwareBuffer_lock_fn(ahwb,AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN,-1,nullptr,&p)!=0)return false;
  memcpy(output.data(),p,bytes);
  int status=api.AHardwareBuffer_unlock_fn(ahwb,nullptr);
  if(status!=0)RuntimeFatal("BUFFER_UNLOCK");
  return true;
 }
};
struct Model {
 static size_t next_id;
 static uint64_t next_generation;
 std::string prefix,node;
 std::vector<size_t>input_sizes,output_sizes;
 std::vector<char>code;
 ThrSqContainerHandle sq=0;
 bool sq_loaded=false;
 ThrGraph*graph=nullptr;
 ThrInvocationContext*inv=nullptr;
 bool runtime_ready=false;
 size_t last_use=0;
 uint64_t generation=0;
 std::vector<bool>input_ready;
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
 size_t runtime_bytes()const{
  size_t total=code.size();
  for(size_t bytes:input_sizes)total+=pixelforge::RoundedBufferBytes(bytes);
  for(size_t bytes:output_sizes)total+=pixelforge::RoundedBufferBytes(bytes);
  return total;
 }
 void advance_generation(){
  if(next_generation==std::numeric_limits<uint64_t>::max())RuntimeFatal("GENERATION_OVERFLOW");
  generation=++next_generation;
 }
 void invalidate_inputs(){
  input_ready.assign(input_sizes.size(),false);advance_generation();
 }
 std::string ready_bits()const{
  std::string ready(input_sizes.size(),'0');
  if(runtime_ready && input_ready.size()==input_sizes.size())
   for(size_t k=1;k<input_ready.size();++k)if(input_ready[k])ready[k]='1';
  return ready;
 }
 bool sticky_ready(uint64_t expected)const{
  return runtime_ready && generation==expected && input_sizes.size()>1 &&
   input_ready.size()==input_sizes.size() &&
   std::all_of(input_ready.begin()+1,input_ready.end(),[](bool ready){return ready;});
 }
 bool upload_weight(size_t index){
  if(!runtime_ready || index==0 || index>=inputs.size())return false;
  input_ready[index]=false;advance_generation();
  if(!inputs[index]->read_input())return false;
  input_ready[index]=true;return true;
 }
 bool has_imported_weights()const{
  for(size_t index=1;index<inputs.size();++index)if(inputs[index]->dma_fd.get()>=0)return true;
  return false;
 }
 bool deferred_weights()const{
  return dma_channel.defer_weight_allocation && input_sizes.size()>1;
 }
 bool import_weight(size_t index,UniqueFd descriptor){
  if(!runtime_ready || index==0 || index>=inputs.size())return false;
  Buffer&buffer=*inputs[index];input_ready[index]=false;advance_generation();
  if(buffer.attached){
   if(!inv || api.thrInvocationContextDetachBuffer_fn(inv,context,buffer.edge.c_str(),buffer.handle)!=0)
    RuntimeFatal("BUFFER_DETACH");
   buffer.attached=false;
  }
  buffer.release();
  if(!buffer.import_dma(std::move(descriptor)))return false;
  if(inv){
   if(api.thrInvocationContextAttachBuffer_fn(inv,context,buffer.edge.c_str(),buffer.handle)!=0)return false;
   buffer.attached=true;
  }
  input_ready[index]=true;return true;
 }
 bool load_runtime(bool create_inv=true){
  if(runtime_ready)return !create_inv || ensure_inv();
  // Also retire partially-created resources before any retry.
  if(sq_loaded || graph || !inputs.empty() || !outputs.empty())drop_runtime();
  if(!build_runtime() || (create_inv && !ensure_inv())){drop_runtime();return false;}
  return true;
 }
 bool build_runtime(){
  prefix="pixel_model_"+std::to_string(++next_id);node=prefix+"_node";
  if(api.thrLoadSqContainer_fn(context,kThrSqContainerTypeMlModel,code.data(),code.size(),&sq)!=0)return false;
  sq_loaded=true;
  graph=api.thrGraphCreate_fn(context);if(!graph)return false;
  performance_policy.annotate(graph);
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
  // Imported weight ports retain their logical charge but never allocate a
  // temporary AHWB. Invocation creation waits until Z registers every port.
  for(size_t k=0;k<inputs.size();++k)if((k==0 || !deferred_weights()) && !inputs[k]->make())return false;
  for(auto&b:outputs)if(!b->make())return false;
  runtime_ready=true;invalidate_inputs();return true;
 }
 bool ensure_inv(){
  if(inv)return true;
  for(const auto&b:inputs)if(!b->registered)return false;
  for(const auto&b:outputs)if(!b->registered)return false;
  inv=api.thrInvocationContextGet_fn(graph,context);if(!inv)return false;
  for(auto&b:inputs){if(api.thrInvocationContextAttachBuffer_fn(inv,context,b->edge.c_str(),b->handle)!=0){drop_inv();return false;}b->attached=true;}
  for(auto&b:outputs){if(api.thrInvocationContextAttachBuffer_fn(inv,context,b->edge.c_str(),b->handle)!=0){drop_inv();return false;}b->attached=true;}
  return true;
 }
 bool drop_inv(){
  if(!inv)return true;
  for(auto&b:inputs)if(b->attached){if(api.thrInvocationContextDetachBuffer_fn(inv,context,b->edge.c_str(),b->handle)!=0)RuntimeFatal("BUFFER_DETACH");b->attached=false;}
  for(auto&b:outputs)if(b->attached){if(api.thrInvocationContextDetachBuffer_fn(inv,context,b->edge.c_str(),b->handle)!=0)RuntimeFatal("BUFFER_DETACH");b->attached=false;}
  if(api.thrInvocationContextDelete_fn(graph,inv)!=0)RuntimeFatal("INVOCATION_DELETE");
  inv=nullptr;return true;
 }
 bool drop_runtime(){
  if(runtime_ready || sq_loaded || graph || inv || !inputs.empty() || !outputs.empty())invalidate_inputs();
  bool ok=drop_inv();
  for(auto&b:inputs)if(!b->release())ok=false;
  for(auto&b:outputs)if(!b->release())ok=false;
  if(graph){if(api.thrGraphDelete_fn(graph)!=0)RuntimeFatal("GRAPH_DELETE");graph=nullptr;}
  // Keep edge-name strings alive through graph deletion; the vendor may retain them.
  inputs.clear();outputs.clear();
  if(sq_loaded){
   if(api.thrUnloadSqContainer_fn(context,sq)!=0){
    // The vendor still owns a reference to code. Do not destroy its owner,
    // reuse this context, or run destructors after an unsuccessful unload.
    RuntimeFatal("SQ_UNLOAD");
   }
   sq_loaded=false;sq=0;
  }
  runtime_ready=false;return ok;
 }
 bool invoke(){
  if(!ensure_inv() || api.thrInvocationContextPrepareForInvoke_fn(inv,false)!=0)return false;
  // On invoke/wait failure it is unknown whether the device still references
  // its buffers. Do not attempt normal destruction or reuse in that state.
  if(api.thrInvocationContextInvokeOnce_fn(inv)!=0 || api.thrInvocationContextWait_fn(inv)!=0)RuntimeFatal("INVOKE");
  return true;
 }
};
size_t Model::next_id=0;
uint64_t Model::next_generation=0;
using Models=std::map<std::string,std::unique_ptr<Model>>;
struct RuntimeCache {
 Models&models;
 size_t budget=pixelforge::kDefaultRuntimeCacheBytes;
 size_t clock=0,loads=0,hits=0,evictions=0;
 size_t max_models=0;
 explicit RuntimeCache(Models&owned):models(owned){}
 bool configure_model_limit(){
  const char*setting=std::getenv("PIXELFORGE_NPU_MAX_RESIDENT_MODELS");
  if(!setting)return true;
  const std::string text(setting);std::istringstream source(text);
  if(text.empty() || text.find_first_not_of("0123456789")!=std::string::npos ||
     !pixelforge::ReadPositiveDecimal(source,64,max_models) || !pixelforge::HeaderEnded(source))return false;
  return true;
 }
 size_t resident_bytes()const{
  size_t bytes=0;for(const auto&m:models)if(m.second->runtime_ready)bytes+=m.second->runtime_bytes();return bytes;
 }
 size_t resident_count()const{
  size_t count=0;for(const auto&m:models)if(m.second->runtime_ready)++count;return count;
 }
 void trim(size_t incoming=0,size_t incoming_models=0){
  // If one model exceeds the budget, all retained runtimes are evicted first;
  // that model runs transiently and finish() releases it before replying.
  while(resident_count() && (incoming>budget || resident_bytes()>budget-incoming ||
        (max_models && resident_count()+incoming_models>max_models))){
   Model*oldest=nullptr;
   for(auto&m:models)if(m.second->runtime_ready && (!oldest || m.second->last_use<oldest->last_use))oldest=m.second.get();
   oldest->drop_runtime();++evictions;
  }
 }
 bool acquire(Model&model,bool create_inv=true){
  if(model.runtime_ready){
   if(create_inv && !model.ensure_inv())return false;
   ++hits;model.last_use=++clock;return true;
  }
  trim(model.runtime_bytes(),1);
  if(!model.load_runtime(create_inv))return false;
  ++loads;model.last_use=++clock;return true;
 }
 void finish(Model&model){if(model.runtime_bytes()>budget)model.drop_runtime();}
 void configure(size_t bytes){budget=bytes;trim();}
};
int RunProtocol(){
 if(!dma_channel.configure()){puts("ERR DMA_SOCKET");return 2;}
 if(!performance_policy.configure()){puts("ERR PERFORMANCE_MODE");return 2;}
 Models models;
 RuntimeCache cache(models);
 int exit_status=0;
 if(!cache.configure_model_limit()){puts("ERR CACHE_MODEL_LIMIT");exit_status=2;}
 bool quit_requested=false;
 char line[65536];
 while(exit_status==0 && fgets(line,sizeof(line),stdin)){
  if(!strchr(line,'\n')){puts("ERR HEADER_SIZE");exit_status=2;break;}
  std::istringstream s(line);char op=0;s>>op;
  if(op=='Q'){
   if(!pixelforge::HeaderEnded(s)){puts("ERR COMMAND_HEADER");exit_status=2;break;}
   quit_requested=true;break;
  }
  if(op=='B'){
   size_t bytes=0;
   if(!pixelforge::ReadUnsignedDecimal(s,pixelforge::kMaxRuntimeCacheBytes,bytes) || !pixelforge::HeaderEnded(s)){puts("ERR CACHE_HEADER");exit_status=2;break;}
   cache.configure(bytes);printf("OK CACHE %zu\n",cache.budget);continue;
  }
  std::string name;s>>name;if(name.empty()||name.size()>128){puts("ERR NAME");exit_status=2;break;}
  if(op=='I' && name=="CACHE"){
   std::string target;
   if(s>>target){
    if(target!="LIMIT" || !pixelforge::HeaderEnded(s)){puts("ERR COMMAND_HEADER");exit_status=2;break;}
    printf("OK CACHE_LIMIT MODELS %zu\n",cache.max_models);continue;
   }
  }
  if(op=='I' && name=="MODEL"){
   std::string target;
   if(s>>target){
    if(target.size()>128 || !pixelforge::HeaderEnded(s)){puts("ERR COMMAND_HEADER");exit_status=2;break;}
    auto it=models.find(target);if(it==models.end()){puts("ERR UNKNOWN_MODEL");continue;}
    const Model&m=*it->second;
    printf("OK MODEL %s %llu %d %zu %s\n",target.c_str(),static_cast<unsigned long long>(m.generation),
     m.runtime_ready?1:0,m.input_sizes.size(),m.ready_bits().c_str());
    continue;
   }
  }
  size_t upload_index=0,import_bytes=0;uint64_t expected_generation=0;
  if(op=='U' && !pixelforge::ReadStickyUploadHeader(s,upload_index)){puts("ERR STICKY_UPLOAD_HEADER");exit_status=2;break;}
  if(op=='Z' && !pixelforge::ReadDmaImportHeader(s,upload_index,import_bytes)){puts("ERR DMA_IMPORT_HEADER");exit_status=2;break;}
  if(op=='V' && !pixelforge::ReadStickyInvokeHeader(s,expected_generation)){puts("ERR STICKY_INVOKE_HEADER");exit_status=2;break;}
  if((op=='R'||op=='D'||op=='F'||op=='I')&&!pixelforge::HeaderEnded(s)){puts("ERR COMMAND_HEADER");exit_status=2;break;}
  if(op=='F'){models.erase(name);puts("OK FREED");continue;}
  if(op=='I'){
   if(name=="CAPABILITIES")printf("OK CAPABILITIES C R F I D L Q B PERSISTENT_RUNTIME U V STICKY_INPUTS_V1 RESIDENT_MODEL_LIMIT_V1%s%s%s\n",
    dma_channel.socket.get()>=0?" Z DMA_BUF_IMPORT_V1":"",
    dma_channel.defer_weight_allocation?" DEFERRED_DMA_WEIGHTS_V1":"",
    performance_policy.supported?" PERFORMANCE_MODE_V1":"");
   else if(name=="PERFORMANCE_MODE")printf("OK PERFORMANCE_MODE %s %d\n",performance_policy.name.c_str(),performance_policy.mode);
   else if(name=="CACHE")printf("OK CACHE %zu %zu %zu %zu %zu %zu\n",cache.budget,cache.resident_bytes(),cache.resident_count(),cache.loads,cache.hits,cache.evictions);
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
   if(name=="CACHE" || name=="CAPABILITIES" || name=="PERFORMANCE_MODE"){puts("ERR NAME");exit_status=2;break;}
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
   m->input_sizes=header.input_bytes;m->output_sizes=header.output_bytes;
   cache.trim(m->runtime_bytes(),1);
   if(!m->make(header.input_bytes,header.output_bytes)){puts("ERR MODEL_LOAD");continue;}
   size_t bytes=m->code.size();
   models[name]=std::move(m);printf("OK %s %s %zu\n",op=='C'?"COMPILED":"LOADED",name.c_str(),bytes);
  }else if(op=='Z'){
   if(dma_channel.socket.get()<0){puts("ERR DMA_CHANNEL");exit_status=2;break;}
   auto it=models.find(name);if(it==models.end()){puts("ERR UNKNOWN_MODEL");exit_status=2;break;}
   Model&m=*it->second;
   if(upload_index>=m.input_sizes.size() || import_bytes!=m.input_sizes[upload_index]){puts("ERR DMA_PORT_SIZE");exit_status=2;break;}
   if(m.runtime_bytes()>cache.budget){puts("ERR STICKY_CACHE_BUDGET");exit_status=2;break;}
   UniqueFd descriptor;
   if(!dma_channel.receive(line,descriptor)){puts("ERR DMA_FRAME");exit_status=2;break;}
   if(!ValidateDmaFd(descriptor.get(),import_bytes)){puts("ERR DMA_BUFFER");exit_status=2;break;}
   if(!cache.acquire(m,!m.deferred_weights())){puts("ERR MODEL_RELOAD");exit_status=2;break;}
   if(!m.import_weight(upload_index,std::move(descriptor))){puts("ERR DMA_IMPORT");exit_status=2;break;}
   printf("OK IMPORTED %s %zu %llu\n",name.c_str(),upload_index,static_cast<unsigned long long>(m.generation));
  }else if(op=='U'){
   auto it=models.find(name);if(it==models.end()){puts("ERR UNKNOWN_MODEL");exit_status=2;break;}
   Model&m=*it->second;
   if(upload_index>=m.input_sizes.size()){puts("ERR STICKY_INPUT_INDEX");exit_status=2;break;}
   if(m.deferred_weights()){puts("ERR DMA_LEGACY_INPUT");exit_status=2;break;}
   if(m.runtime_ready && m.inputs[upload_index]->dma_fd.get()>=0){puts("ERR DMA_LEGACY_INPUT");exit_status=2;break;}
   // Sticky ports must survive the ACK. Never accept an upload which finish()
   // would silently discard; legacy R retains its transient oversize behavior.
   if(m.runtime_bytes()>cache.budget){puts("ERR STICKY_CACHE_BUDGET");exit_status=2;break;}
   if(!cache.acquire(m)){puts("ERR MODEL_RELOAD");exit_status=2;break;}
   if(!m.upload_weight(upload_index)){puts("ERR TENSOR_INPUT");exit_status=2;break;}
   printf("OK UPLOADED %s %zu %llu\n",name.c_str(),upload_index,static_cast<unsigned long long>(m.generation));
  }else if(op=='R'||op=='V'){
   auto it=models.find(name);if(it==models.end()){puts("ERR UNKNOWN_MODEL");exit_status=2;break;}
   Model&m=*it->second;
   if(op=='R' && (m.deferred_weights() || m.has_imported_weights())){puts("ERR DMA_LEGACY_INPUT");exit_status=2;break;}
   if(op=='V' && !m.sticky_ready(expected_generation)){puts("ERR STICKY_STATE");exit_status=2;break;}
   if(!cache.acquire(m)){puts("ERR MODEL_RELOAD");exit_status=2;break;}
   bool ok=true;
   if(op=='R'){
    m.invalidate_inputs(); // Legacy input writes invalidate any old weight token.
    for(auto&b:m.inputs)if(!b->read_input()){ok=false;break;}
   }else ok=m.inputs[0]->read_input();
   if(!ok){puts("ERR TENSOR_INPUT");exit_status=2;break;}
   if(!m.invoke()){puts("ERR INVOKE");exit_status=2;break;}
   size_t total=0;
   std::vector<std::vector<char>>result;
   for(auto&b:m.outputs){total+=b->bytes;result.emplace_back();if(!b->copy_output(result.back())){ok=false;break;}}
   if(!ok){puts("ERR OUTPUT_BUFFER");exit_status=2;break;}
   cache.finish(m);
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

int main(){
 // A killed broker must not leave its separate native inference process alive.
 const pid_t parent_pid=getppid();
 if(parent_pid==1 || prctl(PR_SET_PDEATHSIG,SIGKILL)!=0 || getppid()!=parent_pid)return 3;
 int output_fd=dup(STDOUT_FILENO);if(output_fd<0)return 3;
 if(dup2(STDERR_FILENO,STDOUT_FILENO)<0)return 3;
 protocol=fdopen(output_fd,"wb");if(!protocol)return 3;
 setvbuf(protocol,nullptr,_IONBF,0);
 if(!dma_channel.configure()){puts("ERR DMA_SOCKET");return 3;}
 if(!api.load())return 3;
 if(!performance_policy.configure()){puts("ERR PERFORMANCE_MODE");return 3;}
 auto start_pool=reinterpret_cast<decltype(&ABinderProcess_startThreadPool)>(dlsym(RTLD_DEFAULT,"ABinderProcess_startThreadPool"));
 if(start_pool)start_pool();
 if(api.thrInitialize_fn()!=0)return 3;
 context=api.thrContextCreate_fn();if(!context)return 3;
 if(api.thrVendorSetSystemAttributeStr_fn(context,"edgetpu_use_tpu_tachyon","1")!=0 || api.thrVendorSetSystemAttributeStr_fn(context,"sq_container_load_method","dmabuf")!=0)return 3;
 printf("READY PIXELFORGE_NPU_PIPE 1\n");
 return RunProtocol();
}
