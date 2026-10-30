// Tests the actual helper Model/RuntimeCache with a deterministic mock C API.
// It never calls the native entry point, dlopen, a compiler API, or real NPU.
#define main pixelforge_native_entry_not_called
#include "npu_pipe.cc"
#undef main
#undef printf
#undef puts
#include <cassert>
#include <set>
#include <sys/mman.h>

struct AHardwareBuffer { std::vector<char> data; bool locked=false; };
struct ThrContext {};
struct ThrGraph { std::vector<std::string> edges; };
struct ThrInvocationContext { ThrGraph* graph; std::map<std::string, ThrBufferHandle> buffers; };
struct CodeOwner { const char* data; size_t size; char first; };
static std::map<ThrSqContainerHandle, CodeOwner> sqs;
static std::set<ThrGraph*> graphs;
static std::set<ThrInvocationContext*> invocations;
static std::set<AHardwareBuffer*> ahbs;
static std::map<ThrBufferHandle, AHardwareBuffer*> registered;
struct DmaRegistration {int fd;void*data;size_t bytes;};
static std::map<ThrBufferHandle,DmaRegistration> dma_registered;
static size_t dma_registers=0,dma_unregisters=0,descriptor_closes=0;
static size_t next_handle=1, load_calls=0, graph_calls=0, inv_calls=0;
static size_t allocations=0, releases=0, attach_calls=0, invokes=0, waits=0;
static size_t allocated_bytes=0, peak_bytes=0;
static size_t performance_annotations=0;
static std::string failing;
static bool failed=false;
static bool Fail(const char*name){
  if(failing==name && !failed){
    failed=true;
    std::fprintf(stderr,"FAIL %s sq=%zu graph=%zu inv=%zu ahb=%zu reg=%zu dma=%zu\n",name,sqs.size(),graphs.size(),invocations.size(),ahbs.size(),registered.size()+dma_registered.size(),dma_registered.size());
    return true;
  }
  return false;
}
static ThrStatus Status(bool ok){return static_cast<ThrStatus>(ok?0:-1);}
static ThrStatus Load(ThrContext*, ThrSqContainerType, const void*data, size_t bytes, ThrSqContainerHandle*out){
  ++load_calls;if(Fail("load"))return Status(false);
  assert(data && bytes);
  *out=next_handle++;sqs[*out]={static_cast<const char*>(data),bytes,*static_cast<const char*>(data)};
  return Status(true);
}
static ThrStatus Unload(ThrContext*, ThrSqContainerHandle handle){
  auto it=sqs.find(handle);assert(it!=sqs.end());
  assert(it->second.data[0]==it->second.first); // Code owner must remain alive.
  if(Fail("unload"))return Status(false);
  sqs.erase(it);return Status(true);
}
static ThrGraph* Graph(ThrContext*){
  ++graph_calls;if(Fail("graph"))return nullptr;
  auto*g=new ThrGraph;graphs.insert(g);return g;
}
static const char* VendorVersion(){
  const char*version=std::getenv("PIXELFORGE_TEST_VENDOR_VERSION");
  return version?version:"0.20.0";
}
static ThrStatus AnnotateGraph(ThrGraph*g,const char*key,const char*value){
  assert(graphs.count(g) && g->edges.empty());
  assert(std::strcmp(key,"edgetpu_performance_mode")==0);
  assert(std::string(value)=="2" || std::string(value)=="3" || std::string(value)=="4");
  ++performance_annotations;return Status(!Fail("annotation"));
}
static ThrStatus DeleteGraph(ThrGraph*g){
  assert(graphs.count(g));for(auto*i:invocations)assert(i->graph!=g);
  if(Fail("delete-graph"))return Status(false);
  graphs.erase(g);delete g;return Status(true);
}
static ThrStatus Node(ThrGraph*g,ThrNodeId,ThrNodeType){assert(graphs.count(g));return Status(!Fail("node"));}
static ThrStatus Assign(ThrGraph*g,ThrNodeId,ThrSqContainerHandle handle,const char*){
  assert(graphs.count(g) && sqs.count(handle));return Status(!Fail("assign"));
}
static ThrStatus Edge(ThrGraph*g,ThrEdgeId e,ThrEdgeType){g->edges.emplace_back(e);return Status(!Fail("edge"));}
static ThrStatus Connect(ThrGraph*,ThrNodeId,ThrEdgeId){return Status(!Fail("connect"));}
static ThrStatus SetEdge(ThrGraph*,ThrEdgeId){return Status(!Fail("set-edge"));}
static int Allocate(const AHardwareBuffer_Desc*d,AHardwareBuffer**out){
  if(Fail("allocate"))return -1;
  assert(d->height==1 && d->layers==1 && d->width%4096==0);
  *out=new AHardwareBuffer;(*out)->data.resize(d->width);ahbs.insert(*out);
  ++allocations;allocated_bytes+=d->width;peak_bytes=std::max(peak_bytes,allocated_bytes);return 0;
}
static void Release(AHardwareBuffer*b){
  assert(ahbs.count(b) && !b->locked);for(auto&r:registered)assert(r.second!=b);
  allocated_bytes-=b->data.size();++releases;ahbs.erase(b);delete b;
}
static int Lock(AHardwareBuffer*b,uint64_t,int32_t,const void*,void**out){
  assert(ahbs.count(b) && !b->locked);if(Fail("lock"))return -1;
  b->locked=true;*out=b->data.data();return 0;
}
static int Unlock(AHardwareBuffer*b,int32_t*){
  assert(b->locked);b->locked=false;return Fail("unlock")?-1:0;
}
static ThrStatus Register(ThrContext*,ThrBufferType type,void*data,size_t bytes,ThrBufferHandle*out){
  if(type==kThrBufferTypeDmaBuf){
    const int fd=*static_cast<int*>(data);
    assert(fcntl(fd,F_GETFD)&FD_CLOEXEC);
    if(Fail("dma-register"))return Status(false);
    void*mapping=mmap(nullptr,bytes,PROT_READ,MAP_SHARED,fd,0);
    assert(mapping!=MAP_FAILED);*out=next_handle++;
    dma_registered[*out]={fd,mapping,bytes};++dma_registers;return Status(true);
  }
  assert(ahbs.count(static_cast<AHardwareBuffer*>(data)));
  if(Fail("register"))return Status(false);
  *out=next_handle++;registered[*out]=static_cast<AHardwareBuffer*>(data);return Status(true);
}
static ThrStatus Unregister(ThrContext*,ThrBufferHandle handle){
  assert(registered.count(handle)||dma_registered.count(handle));for(auto*i:invocations)for(auto&b:i->buffers)assert(b.second!=handle);
  if(dma_registered.count(handle)){
    auto item=dma_registered.at(handle);assert(fcntl(item.fd,F_GETFD)>=0);
    if(Fail("dma-unregister"))return Status(false);
    assert(munmap(item.data,item.bytes)==0);dma_registered.erase(handle);++dma_unregisters;return Status(true);
  }
  if(Fail("unregister"))return Status(false);
  registered.erase(handle);return Status(true);
}
static ThrInvocationContext* Invocation(ThrGraph*g,ThrContext*){
  ++inv_calls;if(Fail("invocation"))return nullptr;
  auto*i=new ThrInvocationContext{g,{}};invocations.insert(i);return i;
}
static ThrStatus Attach(ThrInvocationContext*i,ThrContext*,ThrEdgeId e,ThrBufferHandle handle){
  assert(invocations.count(i) && (registered.count(handle)||dma_registered.count(handle)));++attach_calls;
  // Fail the second attachment so partial attachment cleanup is exercised.
  if(failing=="attach" && attach_calls%2==0 && Fail("attach"))return Status(false);
  i->buffers[e]=handle;return Status(true);
}
static ThrStatus Detach(ThrInvocationContext*i,ThrContext*,ThrEdgeId e,ThrBufferHandle handle){
  assert(i->buffers.at(e)==handle);if(Fail("detach"))return Status(false);
  i->buffers.erase(e);return Status(true);
}
static ThrStatus DeleteInvocation(ThrGraph*g,ThrInvocationContext*i){
  assert(invocations.count(i) && i->graph==g && i->buffers.empty());
  if(Fail("delete-invocation"))return Status(false);
  invocations.erase(i);delete i;return Status(true);
}
static ThrStatus Prepare(ThrInvocationContext*i,bool){
  assert(i->buffers.size()==i->graph->edges.size());return Status(!Fail("prepare"));
}
static std::pair<char*,size_t> Data(ThrBufferHandle handle){
  if(registered.count(handle)){auto&bytes=registered.at(handle)->data;return {bytes.data(),bytes.size()};}
  const auto&item=dma_registered.at(handle);return {static_cast<char*>(item.data),item.bytes};
}
static ThrStatus Invoke(ThrInvocationContext*i){
  ++invokes;if(Fail("invoke"))return Status(false);
  const auto first=Data(i->buffers.at(i->graph->edges.front()));
  const auto last=Data(i->buffers.at(i->graph->edges.back()));
  std::copy_n(first.first,std::min(first.second,last.second),last.first);
  // Multi-port protocol fixtures depend on every uploaded weight port, so a
  // missing/replaced sticky upload cannot pass as activation-only echo.
  for(const auto&edge:i->graph->edges){
    if(edge.find("_input_")==std::string::npos || edge==i->graph->edges.front())continue;
    const auto weight=Data(i->buffers.at(edge));
    for(size_t k=0;k<last.second;++k)
      last.first[k]=static_cast<char>(static_cast<unsigned char>(last.first[k])+static_cast<unsigned char>(weight.first[k%weight.second]));
  }
  return Status(true);
}
static ThrStatus Wait(ThrInvocationContext*){++waits;return Status(!Fail("wait"));}
static ThrStatus DeleteContext(ThrContext*){
  assert(sqs.empty() && graphs.empty() && invocations.empty() && ahbs.empty() && registered.empty() && dma_registered.empty());
  return Status(!Fail("delete-context"));
}
static int MockClose(int fd){
  assert(fcntl(fd,F_GETFD)>=0);++descriptor_closes;return ::close(fd);
}
static int MockDmaSync(int fd,uint64_t flags){
  char name[256];const auto path="/proc/self/fd/"+std::to_string(fd);
  const auto count=readlink(path.c_str(),name,sizeof(name));
  if(count>0 && std::string(name,size_t(count)).find("pixelforge-dma-test")!=std::string::npos){
    if(Fail("dma-sync") || ((flags&DMA_BUF_SYNC_END) && Fail("dma-sync-end")))return -1;
    return 0;
  }
  return RealDmaSync(fd,flags);
}
static void Install(){
  protocol=stdout;static ThrContext mock_context;context=&mock_context;
  setvbuf(protocol,nullptr,_IONBF,0); // Match the native helper's interactive pipe.
  close_fd_fn=&MockClose;dma_sync_fn=&MockDmaSync;
  api.thrLoadSqContainer_fn=&Load;api.thrUnloadSqContainer_fn=&Unload;
  api.thrGraphCreate_fn=&Graph;api.thrGraphDelete_fn=&DeleteGraph;
  api.thrGraphAddSqNode_fn=&Node;api.thrGraphAssignSq_fn=&Assign;
  api.thrGraphAddEdge_fn=&Edge;api.thrGraphConnectNodeInput_fn=&Connect;
  api.thrGraphConnectNodeOutput_fn=&Connect;api.thrGraphSetInputEdge_fn=&SetEdge;api.thrGraphSetOutputEdge_fn=&SetEdge;
  api.AHardwareBuffer_allocate_fn=&Allocate;api.AHardwareBuffer_release_fn=&Release;
  api.AHardwareBuffer_lock_fn=&Lock;api.AHardwareBuffer_unlock_fn=&Unlock;
  api.thrRegisterBuffer_fn=&Register;api.thrUnregisterBuffer_fn=&Unregister;
  api.thrInvocationContextGet_fn=&Invocation;api.thrInvocationContextAttachBuffer_fn=&Attach;
  api.thrInvocationContextDetachBuffer_fn=&Detach;api.thrInvocationContextDelete_fn=&DeleteInvocation;
  api.thrInvocationContextPrepareForInvoke_fn=&Prepare;api.thrInvocationContextInvokeOnce_fn=&Invoke;api.thrInvocationContextWait_fn=&Wait;
  api.thrContextDelete_fn=&DeleteContext;
  if(const char*support=std::getenv("PIXELFORGE_TEST_PERFORMANCE_API")){
    if(std::string(support)!="missing-annotation")api.thrGraphAnnotateGraph_fn=&AnnotateGraph;
    if(std::string(support)!="missing-version")api.thrGetVendorApiVersion_fn=&VendorVersion;
  }
}
static void Empty(){assert(sqs.empty() && graphs.empty() && invocations.empty() && ahbs.empty() && registered.empty() && dma_registered.empty());assert(allocated_bytes==0);}
static Model& Add(Models&models,const std::string&name,size_t in=16,size_t out=16){
  auto model=std::make_unique<Model>();model->code.assign(12,'d');
  model->input_sizes={in};model->output_sizes={out};
  auto*result=model.get();models[name]=std::move(model);return *result;
}
static void Run(Model&m,unsigned char value){
  void*p=nullptr;assert(Lock(m.inputs[0]->ahwb,0,-1,nullptr,&p)==0);
  std::memset(p,value,m.inputs[0]->bytes);assert(Unlock(m.inputs[0]->ahwb,nullptr)==0);
  assert(m.invoke());std::vector<char>out;assert(m.outputs[0]->copy_output(out));
  assert(static_cast<unsigned char>(out.front())==value);
}
int main(int argc,char**argv){
  assert(argc==2);Install();const std::string scenario(argv[1]);
  if(scenario.rfind("protocol",0)==0){
    if(scenario.size()>9)failing=scenario.substr(9);
    const int result=RunProtocol();Empty();assert(allocations==releases);
    std::fprintf(stderr,"MOCK COUNTERS load=%zu graph=%zu inv=%zu alloc=%zu invoke=%zu dma_register=%zu dma_unregister=%zu fd_close=%zu performance_annotations=%zu\n",load_calls,graph_calls,inv_calls,allocations,invokes,dma_registers,dma_unregisters,descriptor_closes,performance_annotations);
    return result;
  }
  Models models;RuntimeCache cache(models);
  Model&a=Add(models,"a");
  if(scenario=="reuse"){
    const auto original=a.code;
    assert(cache.acquire(a));Run(a,7);cache.finish(a);
    assert(cache.acquire(a));Run(a,19);cache.finish(a);
    assert(cache.loads==1 && cache.hits==1 && load_calls==1 && graph_calls==1 && inv_calls==1);
    assert(allocations==2 && attach_calls==2 && invokes==2 && waits==2 && a.code==original);
    assert(cache.resident_count()==1 && cache.resident_bytes()==8204);
  }else if(scenario=="lru"){
    auto&b=Add(models,"b");auto&c=Add(models,"c");cache.configure(a.runtime_bytes()*2);
    assert(cache.acquire(a));cache.finish(a);assert(cache.acquire(b));cache.finish(b);
    assert(cache.acquire(a));cache.finish(a);assert(cache.acquire(c));cache.finish(c);
    assert(a.runtime_ready && !b.runtime_ready && c.runtime_ready && cache.evictions==1);
    assert(cache.resident_bytes()<=cache.budget && peak_bytes<=16384);
    assert(cache.acquire(b));cache.finish(b);
    assert(!a.runtime_ready && b.runtime_ready && cache.loads==4 && cache.hits==1);
  }else if(scenario=="batch-accounting"){
    auto&b=Add(models,"batch",3*4097,3*2049);cache.configure(b.runtime_bytes());
    assert(b.runtime_bytes()==12+16384+8192);assert(cache.acquire(b));cache.finish(b);
    assert(allocated_bytes==24576 && cache.resident_bytes()==24588);
    assert(cache.acquire(a));cache.finish(a);assert(!b.runtime_ready);
  }else if(scenario=="zero" || scenario=="oversize"){
    cache.configure(scenario=="zero"?0:8192);
    for(int n=0;n<3;++n){assert(cache.acquire(a));Run(a,n);cache.finish(a);assert(cache.resident_count()==0);Empty();}
    assert(cache.loads==3 && cache.hits==0 && allocations==6 && releases==6);
  }else if(scenario=="shrink-free-replace"){
    auto&b=Add(models,"b");assert(cache.acquire(a));cache.finish(a);assert(cache.acquire(b));cache.finish(b);
    cache.configure(a.runtime_bytes());assert(!a.runtime_ready && b.runtime_ready);
    models.erase("b");Empty();assert(cache.resident_bytes()==0);
    auto&replacement=Add(models,"a");assert(cache.acquire(replacement));cache.finish(replacement);
    cache.configure(0);Empty();assert(cache.resident_count()==0);
  }else if(scenario=="validate"){
    assert(a.make({16},{16}));Empty();assert(cache.loads==0);
    assert(cache.acquire(a));Run(a,23);cache.finish(a);assert(load_calls==2);
  }else if(scenario.rfind("partial-",0)==0){
    failing=scenario.substr(8);assert(!cache.acquire(a));assert(failed);Empty();
    assert(cache.loads==0 && cache.resident_count()==0);
    assert(cache.acquire(a));Run(a,29);cache.finish(a);
  }else if(scenario.rfind("fatal-",0)==0){
    assert(cache.acquire(a));failing=scenario.substr(6);
    if(failing=="unlock"){std::vector<char>out;a.outputs[0]->copy_output(out);}
    else if(failing=="invoke" || failing=="wait")a.invoke();else a.drop_runtime();
    return 87; // Fatal cleanup must never return or run model destructors.
  }else if(scenario=="prepare-error"){
    assert(cache.acquire(a));failing="prepare";assert(!a.invoke());assert(invokes==0);
  }else{assert(false);}
  models.clear();Empty();assert(allocations==releases);
  std::puts("PASS NO_VENDOR_INITIALIZATION NO_NPU_INVOCATIONS");return 0;
}
