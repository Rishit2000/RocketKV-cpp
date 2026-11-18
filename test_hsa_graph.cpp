#include "llama-hsa.h"
#include "rocket.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-metal.h"
#include "llama-context.h"
#include "llama-kv-cache.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
#include <cstring>

static void test_context_cache(const char * model_path) {
    llama_log_set([](ggml_log_level level,const char * text,void *) {
        if(level==GGML_LOG_LEVEL_ERROR) std::fputs(text,stderr);
    },nullptr);
    llama_backend_init();
    auto * model=llama_model_load_from_file(model_path,llama_model_default_params());
    assert(model);
    auto params=llama_context_default_params();
    params.n_ctx=1024;params.n_batch=512;params.n_ubatch=512;
    params.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.hsa_topk=1;
    auto * cached=llama_init_from_model(model,params);
    auto * raw=llama_init_from_model(model,params);
    assert(cached && raw);
    // Full token selection isolates cache lifetime from approximate-score ties.
    for(auto * ctx:{cached,raw}) llama_set_hsa_params(ctx,0,2,70,192);
    const int vocab_size=llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto decode=[&](llama_context * ctx,int pos,int count) {
        auto batch=llama_batch_init(count,0,1);
        batch.n_tokens=count;
        for(int i=0;i<count;++i) {
            batch.token[i]=100+(pos+i)%64;batch.pos[i]=pos+i;
            batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=i==count-1;
        }
        assert(llama_decode(ctx,batch)==0);
        llama_batch_free(batch);
    };
    auto compare=[&](int pos,int count=1) {
        // Compare one forward from identical keys rather than accumulating rounding across steps.
        cached->memory_update(false);
        std::vector<uint8_t> state(llama_state_get_size(cached));
        assert(llama_state_get_data(cached,state.data(),state.size())>0);
        assert(llama_state_set_data(raw,state.data(),state.size())>0);
        decode(cached,pos,count);decode(raw,pos,count);
        const float * a=llama_get_logits_ith(cached,-1),*b=llama_get_logits_ith(raw,-1);
        float error=0.f;
        for(int i=0;i<vocab_size;++i) { assert(std::isfinite(a[i]) && std::isfinite(b[i]));error=std::max(error,std::fabs(a[i]-b[i])); }
        if(error>5e-3f) std::fprintf(stderr,"Context parity error at %d: %g\n",pos,error);
        assert(error<5e-3f);
    };
    compare(0,128);
    assert(llama_hsa_kt_rebuild(cached));
    for(int pos=128;pos<134;++pos) { compare(pos);assert(cached->get_cparams().hsa_kt_ready); }
    // Replacing bounds after decode must discard graphs holding the previous tensors.
    assert(llama_hsa_kt_rebuild(cached));compare(134);
    for(auto * ctx:{cached,raw}) assert(llama_memory_seq_rm(llama_get_memory(ctx),0,0,4));
    compare(135);assert(!cached->get_cparams().hsa_kt_ready);
    assert(llama_hsa_kt_rebuild(cached));compare(136,3);
    assert(!cached->get_cparams().hsa_kt_ready);
    assert(llama_hsa_kt_rebuild(cached));
    for(auto * ctx:{cached,raw}) llama_memory_seq_add(llama_get_memory(ctx),0,-1,-1,4);
    compare(143);assert(!cached->get_cparams().hsa_kt_ready);
    assert(llama_hsa_kt_rebuild(cached));
    std::vector<uint8_t> state(llama_state_get_size(cached));
    assert(llama_state_get_data(cached,state.data(),state.size())>0);
    assert(llama_state_set_data(cached,state.data(),state.size())>0);
    assert(!cached->get_cparams().hsa_kt_ready);compare(144);
    assert(llama_hsa_kt_rebuild(cached));
    // Force capacity growth without hundreds of model forward passes.
    cached->get_cparams().hsa_k_max[0]->ne[1]=1;
    cached->get_cparams().hsa_k_min[0]->ne[1]=1;
    compare(145);
    assert(cached->get_cparams().hsa_kt_ready && cached->get_cparams().hsa_k_max[0]->ne[1]>1);
    llama_set_hsa_params(cached,0,0,0,0);
    assert(!cached->get_cparams().hsa_kt_ready);
    llama_set_hsa_params(cached,0,2,70,512);
    assert(llama_hsa_kt_rebuild(cached));decode(cached,146,1);
    assert(!cached->get_cparams().hsa_kt_ready);
    auto full=llama_get_memory(cached)->init_full();
    const auto * cache=dynamic_cast<const llama_kv_cache_context *>(full.get());
    assert(cache);
    llama_seq_id seq=0,*seqs=&seq;
    llama_pos pos=0;
    llama_ubatch ubatch{};ubatch.n_tokens=1;ubatch.seq_id=&seqs;ubatch.pos=&pos;
    for(int p:{-1,0,17,9999}) for(bool causal:{false,true}) for(uint32_t extent:{64,256,1024}) {
        pos=p;
        assert(cache->get_hsa_order_size(ubatch,causal,extent)==cache->get_hsa_order(ubatch,causal,extent).size());
    }
    full.reset();
    llama_free(cached);llama_free(raw);llama_model_free(model);llama_backend_free();
    std::puts("HSA context append, rebuild, eviction, continuation, shift, and restore: PASS");
}

static void test_bounds_updates(ggml_backend_t backend) {
    const int Hkv=2,H=3,N=256,L=192;
    for(int D:{63,128}) for(int chunk:{2,3}) {
        auto * ctx=ggml_init({8*1024*1024,nullptr,true});
        auto * q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,H*Hkv,1);
        auto * k=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,Hkv,N);
        auto * order=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,L);
        auto * mask=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,N);
        auto * hi=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,(N+chunk-1)/chunk,Hkv);
        auto * lo=ggml_dup_tensor(ctx,hi);
        auto * meta=ggml_hsa_query(ctx,q,Hkv,std::min(70,D));
        auto * cached=ggml_hsa_decode_logits(ctx,meta,hi,lo,k,order,mask,chunk);
        auto * raw=ggml_hsa_logits(ctx,meta,k,order,mask,chunk);
        auto * init=ggml_new_graph(ctx);
        ggml_build_forward_expand(init,ggml_hsa_bounds(ctx,hi,lo,k,order,mask,chunk,false));
        auto * graph=ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,cached);ggml_build_forward_expand(graph,raw);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        std::mt19937 gen(483);
        std::normal_distribution<float> normal;
        std::vector<float> qv(D*H*Hkv),mv(N,-INFINITY),a(ggml_nelements(raw)),b(a.size());
        std::vector<ggml_fp16_t> keys(D*Hkv*N);
        for(auto & value:keys) value=ggml_fp32_to_fp16(normal(gen));
        ggml_backend_tensor_set(k,keys.data(),0,ggml_nbytes(k));
        std::vector<int32_t> rows(L,0);
        for(int live=128;live<=137;++live) {
            for(auto & value:qv) value=normal(gen);
            for(int t=0;t<live;++t) { rows[t]=N-1-t;mv[rows[t]]=0.f; }
            ggml_backend_tensor_set(q,qv.data(),0,ggml_nbytes(q));
            ggml_backend_tensor_set(order,rows.data(),0,ggml_nbytes(order));
            ggml_backend_tensor_set(mask,mv.data(),0,ggml_nbytes(mask));
            if(live==128) assert(ggml_backend_graph_compute(backend,init)==GGML_STATUS_SUCCESS);
            assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
            ggml_backend_tensor_get(cached,a.data(),0,ggml_nbytes(cached));
            ggml_backend_tensor_get(raw,b.data(),0,ggml_nbytes(raw));
            for(size_t i=0;i<a.size();++i) {
                if(a[i]!=b[i] && std::fabs(a[i]-b[i])>1e-4f) std::fprintf(stderr,"Bounds update mismatch chunk=%d live=%d index=%zu: %g vs %g\n",chunk,live,i,a[i],b[i]);
                assert(a[i]==b[i] || std::fabs(a[i]-b[i])<1e-4f);
            }
        }
        ggml_backend_buffer_free(buffer);ggml_free(ctx);
    }
}

static void test_row_attention_padding(ggml_backend_t backend) {
    const int D=128,Hkv=2,H=3,N=256,live=129,keep=192;
    auto * ctx=ggml_init({8*1024*1024,nullptr,true});
    auto * q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,H*Hkv,1);
    auto * k=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,Hkv,N);
    auto * v=ggml_dup_tensor(ctx,k);
    auto * m=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,N);
    auto * out=build_hsa_attention(ctx,q,k,v,m,3,70,keep,1/std::sqrt(float(D)));
    auto * graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    std::mt19937 gen(918);
    std::normal_distribution<float> normal;
    std::vector<float> qv(D*H*Hkv),mask(N,-INFINITY),kc(D*Hkv*live),vc(kc.size());
    std::vector<ggml_fp16_t> keys(D*Hkv*N,ggml_fp32_to_fp16(NAN)),values(keys);
    for(auto & x:qv) x=normal(gen);
    int i=0;
    for(int t=0;t<N;++t) if((t+1)*live/N!=t*live/N) {
        mask[t]=0.f;
        for(int h=0;h<Hkv;++h) for(int d=0;d<D;++d) {
            auto key=ggml_fp32_to_fp16(normal(gen)),value=ggml_fp32_to_fp16(normal(gen));
            keys[(t*Hkv+h)*D+d]=key;values[(t*Hkv+h)*D+d]=value;
            kc[(h*live+i)*D+d]=ggml_fp16_to_fp32(key);vc[(h*live+i)*D+d]=ggml_fp16_to_fp32(value);
        }
        ++i;
    }
    RocketConfig cfg{};
    cfg.dim_head=D;cfg.num_heads=H*Hkv;cfg.num_heads_kv=Hkv;cfg.chunk_size=3;cfg.r_dim=70;cfg.topk_decode=keep;
    auto expected=compute_hybrid_sparse_decode(qv.data(),kc.data(),vc.data(),live,cfg);
    ggml_backend_tensor_set(q,qv.data(),0,ggml_nbytes(q));
    ggml_backend_tensor_set(k,keys.data(),0,ggml_nbytes(k));
    ggml_backend_tensor_set(v,values.data(),0,ggml_nbytes(v));
    ggml_backend_tensor_set(m,mask.data(),0,ggml_nbytes(m));
    assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
    std::vector<float> actual(expected.size());ggml_backend_tensor_get(out,actual.data(),0,ggml_nbytes(out));
    for(size_t j=0;j<actual.size();++j) assert(std::isfinite(actual[j]) && std::fabs(actual[j]-expected[j])<2e-3f);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}

static void bench_hsa(ggml_backend_t backend) {
    const int D=128,Hkv=8,Hq=24,N=8192,live=3533;
    ggml_init_params params{32*1024*1024,nullptr,true};
    auto * ctx=ggml_init(params);
    auto * q=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,Hq,1);
    auto * k=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,Hkv,N);
    auto * v=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,D,Hkv,N);
    auto * m=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,N);
    auto * order=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,((live+63)/64)*64);
    auto * out=build_hsa_attention(ctx,q,k,v,m,2,70,512,1/std::sqrt(float(D)),nullptr,order);
    auto * full=ggml_new_graph_custom(ctx,2048,false);
    ggml_build_forward_expand(full,out);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
    std::mt19937 gen(42);
    std::normal_distribution<float> normal(0,1);
    std::vector<float> qv(D*Hq),mv(N,-INFINITY);
    std::vector<ggml_fp16_t> kv(D*Hkv*N),vv(kv.size());
    for(auto & x:qv) x=normal(gen);
    for(auto & x:kv) x=ggml_fp32_to_fp16(normal(gen));
    for(auto & x:vv) x=ggml_fp32_to_fp16(normal(gen));
    for(int t=0;t<N;++t) if((t+1)*live/N!=t*live/N) mv[t]=0;
    ggml_backend_tensor_set(q,qv.data(),0,ggml_nbytes(q));
    ggml_backend_tensor_set(k,kv.data(),0,ggml_nbytes(k));
    ggml_backend_tensor_set(v,vv.data(),0,ggml_nbytes(v));
    ggml_backend_tensor_set(m,mv.data(),0,ggml_nbytes(m));
    std::vector<int32_t> ordered;
    for(int t=0;t<N;++t) if(std::isfinite(mv[t])) ordered.push_back(t);
    ordered.resize(order->ne[0],0);
    ggml_backend_tensor_set(order,ordered.data(),0,ggml_nbytes(order));
    // Prefix timings expose which stage dominates without changing the decode path.
    for(int i=0;i<ggml_graph_n_nodes(full);++i) {
        auto * node=ggml_graph_node(full,i);
        if(node->op!=GGML_OP_HSA && node->op!=GGML_OP_TOP_K && node->op!=GGML_OP_FLASH_ATTN_EXT) continue;
        auto * graph=ggml_new_graph_custom(ctx,2048,false);
        ggml_build_forward_expand(graph,node);
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_synchronize(backend);
        auto start=std::chrono::steady_clock::now();
        for(int repeat=0;repeat<10;++repeat) { assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS); ggml_backend_synchronize(backend); }
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/10;
        std::printf("HSA prefix %s mode%d: %.1f us (%d nodes)\n",ggml_op_name(node->op),node->op_params[0],us,ggml_graph_n_nodes(graph));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    if(argc==3 && std::strcmp(argv[1],"--context")==0) { test_context_cache(argv[2]);return 0; }
    ggml_backend_t backend = argc > 1 ? ggml_backend_metal_init() : ggml_backend_cpu_init();
    assert(backend);
    std::mt19937 gen(483);
    std::normal_distribution<float> normal(0,1);
    for (int iteration = 0; iteration < 46; ++iteration) {
        const int trial=iteration%23;
        const bool cached=iteration>=23;
        const int D=trial>=20 ? 128 : 64, Hkv=2, G=trial>=20 ? 3 : 2, Hq=G*Hkv;
        const int N=trial==20 ? 1031 : trial>20 ? 4096 : 13, live=trial==20 ? 777 : trial>20 ? 3685 : 8+trial%3;
        const int chunk=1+trial%4;
        const int r=trial==19 ? D : trial>=20 ? 70 : 4, keep=trial==18 ? live : trial==20 ? 64 : trial>20 ? 512 : 3;
        RocketConfig cfg{};
        cfg.dim_head=D; cfg.num_heads=Hq; cfg.num_heads_kv=Hkv;
        cfg.chunk_size=chunk; cfg.r_dim=r; cfg.topk_decode=keep;
        std::vector<float> q(D*Hq),k(D*Hkv*N),v(k.size()),mask(N,-INFINITY);
        std::vector<float> kc(D*Hkv*live),vc(kc.size());
        for (auto & x:q) x=normal(gen);
        for (auto & x:k) x=normal(gen);
        for (auto & x:v) x=normal(gen);
        if(trial==19) for(int kv=0;kv<Hkv;++kv) for(int d=0;d<D;++d) q[(kv*G+1)*D+d]=-q[kv*G*D+d];
        const bool half_cache=trial%2==0;
        std::vector<ggml_fp16_t> kh(k.size()),vh(v.size());
        if(half_cache) for(size_t i=0;i<k.size();++i) {
            kh[i]=ggml_fp32_to_fp16(k[i]);vh[i]=ggml_fp32_to_fp16(v[i]);
            k[i]=ggml_fp16_to_fp32(kh[i]);v[i]=ggml_fp16_to_fp32(vh[i]);
        }
        int li=0;
        for (int t=0;t<N;++t) {
            if ((t+1)*live/N == t*live/N) continue;
            mask[t]=0;
            for(int h=0;h<Hkv;++h) for(int d=0;d<D;++d) {
                kc[(h*live+li)*D+d]=k[(t*Hkv+h)*D+d];
                vc[(h*live+li)*D+d]=v[(t*Hkv+h)*D+d];
            }
            ++li;
        }
        const bool chronological=trial%2==1 || trial==20;
        std::vector<int32_t> chronology;
        if(chronological) {
            for(int t=0;t<N;++t) if(std::isfinite(mask[t])) chronology.push_back(t);
            std::reverse(chronology.begin(),chronology.end());
            for(int i=0;i<live;++i) for(int h=0;h<Hkv;++h) for(int d=0;d<D;++d) {
                kc[(h*live+i)*D+d]=k[(chronology[i]*Hkv+h)*D+d];
                vc[(h*live+i)*D+d]=v[(chronology[i]*Hkv+h)*D+d];
            }
            int pad=0; while(std::isfinite(mask[pad])) ++pad;
            chronology.push_back(pad);
        }
        auto expected=compute_hybrid_sparse_decode(q.data(),kc.data(),vc.data(),live,cfg);
        ggml_init_params params{16*1024*1024,nullptr,true};
        ggml_context * ctx=ggml_init(params);
        auto * qt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,Hq,1);
        auto * kt=ggml_new_tensor_3d(ctx,half_cache ? GGML_TYPE_F16 : GGML_TYPE_F32,D,Hkv,N);
        auto * vt=ggml_new_tensor_3d(ctx,half_cache ? GGML_TYPE_F16 : GGML_TYPE_F32,D,Hkv,N);
        auto * mt=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,N);
        auto * chronological_input=chronological ? ggml_new_tensor_1d(ctx,GGML_TYPE_I32,chronology.size()) : nullptr;
        auto * bounds_max=cached ? ggml_new_tensor_3d(ctx,kt->type,D,(N+chunk-1)/chunk,Hkv) : nullptr;
        auto * bounds_min=cached ? ggml_dup_tensor(ctx,bounds_max) : nullptr;
        ggml_tensor * rows = nullptr;
        auto * out=build_hsa_attention(ctx,qt,kt,vt,mt,chunk,r,keep,1/std::sqrt(float(D)), &rows,chronological_input,bounds_max,bounds_min);
        ggml_cgraph * init_graph=nullptr;
        if(cached) {
            auto * order=chronological_input;
            if(!order) {
                char name[64];
                std::snprintf(name,sizeof(name),"hsa_order_%p_%lld",(void *)mt,(long long)N);
                order=ggml_get_tensor(ctx,name);
            }
            init_graph=ggml_new_graph_custom(ctx,2048,false);
            ggml_build_forward_expand(init_graph,ggml_hsa_bounds(ctx,bounds_max,bounds_min,kt,order,mt,chunk,false));
        }
        auto * graph=ggml_new_graph_custom(ctx,2048,false);
        ggml_build_forward_expand(graph,out);
        ggml_build_forward_expand(graph,rows);
        if(trial==0) {
            auto * shared=build_hsa_attention(ctx,qt,kt,vt,mt,chunk,r,keep,1/std::sqrt(float(D)));
            ggml_build_forward_expand(graph,shared);
            int compactions=0;
            for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
                auto * node=ggml_graph_node(graph,i);
                if(node->op==GGML_OP_HSA && node->op_params[0]==0) ++compactions;
            }
            assert(compactions==1);
        }
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        ggml_backend_tensor_set(qt,q.data(),0,q.size()*4);
        ggml_backend_tensor_set(kt,half_cache ? (const void *)kh.data() : (const void *)k.data(),0,ggml_nbytes(kt));
        ggml_backend_tensor_set(vt,half_cache ? (const void *)vh.data() : (const void *)v.data(),0,ggml_nbytes(vt));
        ggml_backend_tensor_set(mt,mask.data(),0,mask.size()*4);
        if(chronological) ggml_backend_tensor_set(chronological_input,chronology.data(),0,chronology.size()*sizeof(int32_t));
        if(cached) assert(ggml_backend_graph_compute(backend,init_graph)==GGML_STATUS_SUCCESS);
        assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        std::vector<float> actual(expected.size());
        ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
        std::vector<int32_t> selected(keep*Hkv);
        for(int kv=0;kv<Hkv;++kv) ggml_backend_tensor_get(rows,selected.data()+kv*keep,kv*rows->nb[1],keep*sizeof(int32_t));
        // Chunk expansion creates exact ties. Accept any valid top-k tie choice,
        // as torch.topk also does not guarantee stable indices for ties.
        for (int hkv=0;hkv<Hkv;++hkv) {
            std::vector<int> dims(D);
            for(int d=0;d<D;++d) dims[d]=d;
            auto imp = [&](int d) { float x=0; for(int h=0;h<G;++h) x+=std::fabs(q[(hkv*G+h)*D+d]); return x; };
            std::sort(dims.begin(),dims.end(),[&](int a,int b){return imp(a)>imp(b);});
            std::vector<float> scores(live,0.f);
            for(int h=0;h<G;++h) {
                float all=0,kept=0;
                const float * qh=q.data()+(hkv*G+h)*D;
                for(int d=0;d<D;++d) all+=std::fabs(qh[d]);
                for(int ri=0;ri<r;++ri) kept+=std::fabs(qh[dims[ri]]);
                std::vector<float> logits(live,0.f);
                for(int t=0;t<live;++t) for(int ri=0;ri<r;++ri) {
                    int d=dims[ri];
                    float sumq=0; for(int h2=0;h2<G;++h2) sumq+=q[(hkv*G+h2)*D+d];
                    float sign=sumq>0 ? 1.f : -1.f;
                    float mx=-INFINITY;
                    for(int j=t/chunk*chunk;j<std::min(live,t/chunk*chunk+chunk);++j) mx=std::max(mx,sign*kc[(hkv*live+j)*D+d]);
                    logits[t]+=qh[d]*sign*mx/std::sqrt(D*kept/all);
                }
                float mx=*std::max_element(logits.begin(),logits.end()),sum=0;
                for(auto & x:logits){x=std::exp(x-mx);sum+=x;}
                for(int t=0;t<live;++t) scores[t]+=logits[t]/sum;
            }
            auto ranked=scores;
            std::sort(ranked.begin(),ranked.end(),std::greater<float>());
            for(int i=0;i<keep;++i) {
                int t=selected[hkv*keep+i];
                assert(t>=0 && t<live);
                assert(scores[t]+1e-6f>=ranked[keep-1]);
                for(int j=0;j<i;++j) assert(t!=selected[hkv*keep+j]);
            }
            for(int h=0;h<G;++h) {
                std::vector<float> logits(keep,0.f);
                for(int i=0;i<keep;++i) for(int d=0;d<D;++d) logits[i]+=q[(hkv*G+h)*D+d]*kc[(hkv*live+selected[hkv*keep+i])*D+d]/std::sqrt(float(D));
                float mx=*std::max_element(logits.begin(),logits.end()),sum=0;
                for(auto & x:logits){x=std::exp(x-mx);sum+=x;}
                for(int d=0;d<D;++d){
                    float x=0;
                    for(int i=0;i<keep;++i) x+=logits[i]/sum*vc[(hkv*live+selected[hkv*keep+i])*D+d];
                    expected[(hkv*G+h)*D+d]=x;
                }
            }
        }
        for(size_t i=0;i<actual.size();++i) {
            if(std::fabs(actual[i]-expected[i])>2e-3f || !std::isfinite(actual[i])) {
                std::fprintf(stderr,"trial %d element %zu: got %f expected %f\n",trial,i,actual[i],expected[i]);
                return 1;
            }
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    // Reuse one graph while the query, keys and live cache length change.
    {
        const int D=128,Hkv=2,Hq=6,N=65;
        ggml_init_params params{16*1024*1024,nullptr,true};
        auto * ctx=ggml_init(params);
        auto * qt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,Hq,1);
        auto * kt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,Hkv,N);
        auto * vt=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,D,Hkv,N);
        auto * mt=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,N);
        auto * out=build_hsa_attention(ctx,qt,kt,vt,mt,3,D,N,1/std::sqrt(float(D)));
        auto * graph=ggml_new_graph_custom(ctx,2048,false);
        ggml_build_forward_expand(graph,out);
        auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);
        for(int step=0;step<3;++step) {
            int live=43+step*7;
            std::vector<float> q(D*Hq),k(D*Hkv*N),v(k.size()),m(N,-INFINITY),kc(D*Hkv*live),vc(kc.size());
            for(auto & x:q) x=normal(gen);
            for(auto & x:k) x=normal(gen);
            for(auto & x:v) x=normal(gen);
            int li=0;
            for(int t=0;t<N;++t) {
                if((t+1)*live/N==t*live/N) continue;
                m[t]=0;
                for(int h=0;h<Hkv;++h) for(int d=0;d<D;++d) {
                    kc[(h*live+li)*D+d]=k[(t*Hkv+h)*D+d];
                    vc[(h*live+li)*D+d]=v[(t*Hkv+h)*D+d];
                }
                ++li;
            }
            for(int t=0;t<N;++t) if(!std::isfinite(m[t])) {
                for(int h=0;h<Hkv;++h) for(int d=0;d<D;++d) {
                    k[(t*Hkv+h)*D+d]=NAN;
                    v[(t*Hkv+h)*D+d]=NAN;
                }
            }
            RocketConfig cfg{};
            cfg.num_heads=Hq;cfg.num_heads_kv=Hkv;cfg.dim_head=D;cfg.chunk_size=3;cfg.r_dim=D;cfg.topk_decode=live;
            auto expected=compute_hybrid_sparse_decode(q.data(),kc.data(),vc.data(),live,cfg);
            ggml_backend_tensor_set(qt,q.data(),0,q.size()*4);
            ggml_backend_tensor_set(kt,k.data(),0,k.size()*4);
            ggml_backend_tensor_set(vt,v.data(),0,v.size()*4);
            ggml_backend_tensor_set(mt,m.data(),0,m.size()*4);
            assert(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
            std::vector<float> actual(expected.size());
            ggml_backend_tensor_get(out,actual.data(),0,actual.size()*4);
            for(size_t i=0;i<actual.size();++i) assert(std::isfinite(actual[i]) && std::fabs(actual[i]-expected[i])<2e-3f);
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    test_bounds_updates(backend);
    test_row_attention_padding(backend);
    if(argc>2) bench_hsa(backend);
    ggml_backend_free(backend);
    std::puts("HSA graph matches compact-cache CPU reference: PASS");
}
