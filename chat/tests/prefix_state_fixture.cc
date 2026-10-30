// Test-only API fault injection and small byte-budget eviction with real models.
#define main pixelforge_chat_entry_not_called
#include "../native/chat_runner.cc"
#undef main

static int faults = 0;
static size_t short_set(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq) {
    const size_t result = llama_state_seq_set_data(ctx,src,size,seq);
    ++faults;
    return result ? result - 1 : 0; // Mutated state must be fully cleared before fresh prefill.
}
static size_t wrong_position(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq) {
    const size_t result = llama_state_seq_set_data(ctx,src,size,seq);
    llama_memory_clear(llama_get_memory(ctx),true);
    ++faults;
    return result;
}
static size_t short_get(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq) {
    const size_t result = llama_state_seq_get_data(ctx,dst,size,seq);
    ++faults;
    return result ? result - 1 : 0;
}
static std::vector<llama_token> tokenize(llama_model * model, const std::string & text) {
    const auto * vocab = llama_model_get_vocab(model);
    const int count = -llama_tokenize(vocab,text.data(),text.size(),nullptr,0,false,true);
    std::vector<llama_token> result(count);
    if (llama_tokenize(vocab,text.data(),text.size(),result.data(),count,false,true) != count)
        throw std::runtime_error("fixture tokenization");
    return result;
}
static void prefix_decode(llama_context * ctx, const std::vector<llama_token> & input, int count) {
    std::vector<int8_t> flags(16,0);
    for (int offset=0;offset<count;offset+=16) {
        auto batch=llama_batch_get_one(const_cast<llama_token *>(input.data()+offset),std::min(16,count-offset));
        batch.logits=flags.data();
        if (llama_decode(ctx,batch)!=0)throw std::runtime_error("fixture prefix decode");
    }
    llama_synchronize(ctx);
}
int main(int argc,char ** argv) {
    if (argc!=3)return 2;
    llama_backend_init();
    auto mp=llama_model_default_params();
    mp.n_gpu_layers=0;mp.use_extra_bufts=false;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[2],mp),llama_model_free);
    if (!model)return 3;
    const std::string mode(argv[1]);
    if (mode=="eviction") {
        auto cp=llama_context_default_params();
        cp.n_ctx=256;cp.n_batch=cp.n_ubatch=16;cp.n_threads=cp.n_threads_batch=1;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;cp.offload_kqv=cp.op_offload=false;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        if (!ctx)return 4;
        const auto a=tokenize(model.get(),std::string(64,'a')+std::string(140,'x'));
        const auto b=tokenize(model.get(),std::string(64,'b')+std::string(140,'y'));
        prefix_decode(ctx.get(),a,64);
        PrefixCache measured(MIB);
        measured.save(ctx.get(),a,64,true);
        const size_t budget=measured.bytes();
        if (!budget || measured.entries()!=1)return 5;
        measured.clear();
        PrefixCache bounded(budget);
        bounded.save(ctx.get(),a,64,true);
        clear_memory(ctx.get());prefix_decode(ctx.get(),b,64);
        bounded.save(ctx.get(),b,64,true);
        if (bounded.entries()!=1 || bounded.bytes()>budget)return 6;
        clear_memory(ctx.get());
        if (bounded.restore(ctx.get(),a,true)!=0)return 7;
        if (bounded.restore(ctx.get(),b,true)!=64)return 8;
        bounded.clear();
        PrefixCache undersized(budget-1);
        undersized.save(ctx.get(),b,64,true);
        if (undersized.entries() || undersized.bytes())return 9;
        std::cout<<json({{"eviction",true},{"bounded_bytes",budget},{"undersized_entries",0}}).dump()<<'\n';
        return 0;
    }
    SequenceStateAPI api;
    if (mode=="short-set")api.set=short_set;
    else if (mode=="wrong-position")api.set=wrong_position;
    else if (mode=="short-get")api.get=short_get;
    else return 10;
    ConversationContext resident(MIB,api);
    std::string line;
    while(frame(line))generate(model.get(),resident,1,16,FlashAttention{},json::parse(line));
    std::cerr<<"PREFIX_FIXTURE_FAULTS "<<faults<<'\n';
}
