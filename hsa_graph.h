#pragma once

#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

// Single-token HSA. q [D,Hq,1], k/v [D,Hkv,N], mask [N,...].
inline ggml_tensor * build_hsa_attention_reference(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * mask, int chunk, int r, int topk, float scale, ggml_tensor ** selected_rows = nullptr, ggml_tensor * chronological_order = nullptr) {
    const int64_t D = q->ne[0], Hkv = k->ne[1], H = q->ne[1] / Hkv, N = chronological_order ? chronological_order->ne[0] : k->ne[2];
    const int64_t C = (N + chunk - 1) / chunk, P = C * chunk, keep = std::min<int64_t>(topk, N);
    r = std::min<int64_t>(r, D);
    auto f32 = [&](ggml_tensor * t) { return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32); };
    auto perm = [&](ggml_tensor * t, int a, int b, int c, int d) { return ggml_cont(ctx, ggml_permute(ctx, t, a,b,c,d)); };
    ggml_tensor * Q = ggml_reshape_3d(ctx, ggml_cont(ctx, f32(q)), D,H,Hkv);
    ggml_tensor * m = f32(ggml_view_1d(ctx, mask, k->ne[2], 0));
    // Sort valid physical rows chronologically; masked cache holes move to the end.
    ggml_tensor * order = chronological_order ? chronological_order : ggml_argsort(ctx, ggml_sub(ctx, ggml_arange(ctx,0,N,1), m), GGML_SORT_ORDER_ASC);
    ggml_tensor * group_order = ggml_repeat(ctx, ggml_reshape_2d(ctx,order,N,1), ggml_new_tensor_2d(ctx,GGML_TYPE_I32,N,Hkv));
    ggml_tensor * K = ggml_get_rows(ctx, perm(f32(k),0,2,1,3),group_order);
    ggml_tensor * compact_mask = ggml_get_rows(ctx,ggml_reshape_2d(ctx,m,1,k->ne[2]),order);
    compact_mask = ggml_reshape_1d(ctx,compact_mask,N);
    ggml_tensor * one = ggml_fill(ctx,ggml_new_tensor_1d(ctx,GGML_TYPE_F32,1),1.f);
    ggml_tensor * consensus = perm(ggml_sum_rows(ctx,perm(Q,1,0,2,3)),1,0,2,3);
    ggml_tensor * sign = ggml_sub(ctx,ggml_scale(ctx,ggml_step(ctx,consensus),2.f),one);
    ggml_tensor * abs_q = ggml_abs(ctx,Q);
    ggml_tensor * dim_scores = perm(ggml_mean(ctx,perm(abs_q,1,0,2,3)),1,0,2,3);
    ggml_tensor * dims = ggml_top_k(ctx,ggml_reshape_2d(ctx,dim_scores,D,Hkv),r);
    ggml_tensor * q_hat = perm(ggml_get_rows(ctx,perm(ggml_mul(ctx,Q,sign),1,0,2,3),dims),1,0,2,3);
    ggml_tensor * signed_k = perm(ggml_mul(ctx,K,sign),1,0,2,3); // [N,D,Hkv]
    signed_k = ggml_add(ctx,signed_k,compact_mask);
    if (P > N) {
        ggml_tensor * padding = ggml_fill(ctx,ggml_new_tensor_3d(ctx,GGML_TYPE_F32,P-N,D,Hkv),-INFINITY);
        signed_k = ggml_concat(ctx,signed_k,padding,0);
    }
    ggml_tensor * chunk_k = ggml_pool_1d(ctx,signed_k,GGML_OP_POOL_MAX,chunk,chunk,0);
    // Entirely masked chunks must not produce infinity times zero in the dot product.
    chunk_k = ggml_clamp(ctx,chunk_k,-1e30f,1e30f);
    // get_rows selects dimensions: source [C,D,Hkv], result [C,r,Hkv].
    ggml_tensor * k_hat = perm(ggml_get_rows(ctx,chunk_k,dims),1,0,2,3);
    ggml_tensor * logits = ggml_mul_mat(ctx,k_hat,q_hat); // [C,H,Hkv]
    logits = ggml_repeat(ctx,ggml_reshape_4d(ctx,logits,1,C,H,Hkv),ggml_new_tensor_4d(ctx,GGML_TYPE_F32,chunk,C,H,Hkv));
    logits = ggml_reshape_3d(ctx,ggml_cont(ctx,logits),P,H,Hkv);
    logits = ggml_cont(ctx,ggml_view_3d(ctx,logits,N,H,Hkv,logits->nb[1],logits->nb[2],0));
    ggml_tensor * custom_scale = ggml_sqrt(ctx,ggml_scale(ctx,ggml_div(ctx,ggml_sum_rows(ctx,ggml_abs(ctx,q_hat)),ggml_sum_rows(ctx,abs_q)),D));
    logits = ggml_div(ctx,logits,custom_scale);
    ggml_tensor * probs = ggml_soft_max(ctx,ggml_add(ctx,logits,compact_mask));
    ggml_tensor * scores = perm(ggml_mean(ctx,perm(probs,1,0,2,3)),1,0,2,3);
    ggml_tensor * rows = ggml_top_k(ctx,ggml_reshape_2d(ctx,ggml_add(ctx,scores,compact_mask),N,Hkv),keep);
    if (selected_rows) *selected_rows = rows;
    ggml_tensor * selected_k = ggml_get_rows(ctx,K,rows);
    ggml_tensor * physical = ggml_get_rows(ctx,ggml_reshape_3d(ctx,group_order,1,N,Hkv),rows);
    physical = ggml_reshape_2d(ctx,physical,keep,Hkv);
    ggml_tensor * selected_v = ggml_get_rows(ctx,ggml_permute(ctx,v,0,2,1,3),physical);
    ggml_tensor * group_mask = ggml_repeat(ctx,ggml_reshape_3d(ctx,compact_mask,1,N,1),ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,N,Hkv));
    ggml_tensor * selected_mask = perm(ggml_get_rows(ctx,group_mask,rows),1,0,2,3);
    ggml_tensor * q_fa = ggml_reshape_4d(ctx,Q,D,1,H,Hkv);
    ggml_tensor * k_fa = ggml_reshape_4d(ctx,ggml_cast(ctx,selected_k,GGML_TYPE_F16),D,keep,1,Hkv);
    ggml_tensor * v_fa = ggml_reshape_4d(ctx,ggml_cast(ctx,selected_v,GGML_TYPE_F16),D,keep,1,Hkv);
    ggml_tensor * m_fa = ggml_reshape_4d(ctx,ggml_cast(ctx,selected_mask,GGML_TYPE_F16),keep,1,1,Hkv);
    ggml_tensor * out = ggml_flash_attn_ext(ctx,q_fa,k_fa,v_fa,m_fa,scale,0.f,0.f);
    ggml_prec_set_acc(out,GGML_PREC_F32);
    return ggml_reshape_2d(ctx,ggml_cont(ctx,out),D*H*Hkv,1);
}

// Fused selection avoids materializing signed keys and full-cache values.
inline ggml_tensor * build_hsa_attention(
        ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
        ggml_tensor * mask, int chunk, int r, int topk, float scale, ggml_tensor ** selected_rows = nullptr, ggml_tensor * chronological_order = nullptr, ggml_tensor * maximum = nullptr, ggml_tensor * minimum = nullptr) {
    if (q->ne[0] > 256 || q->ne[1]/k->ne[1] > 32) return build_hsa_attention_reference(ctx,q,k,v,mask,chunk,r,topk,scale,selected_rows,chronological_order);
    const int64_t D = q->ne[0], Hkv = k->ne[1], H = q->ne[1]/Hkv, N = chronological_order ? chronological_order->ne[0] : k->ne[2], keep = std::min<int64_t>(topk,N);
    if (q->type != GGML_TYPE_F32) q = ggml_cast(ctx,q,GGML_TYPE_F32);
    if (k->type != GGML_TYPE_F16 && k->type != GGML_TYPE_F32) k = ggml_cast(ctx,k,GGML_TYPE_F32);
    if (v->type != GGML_TYPE_F16 && v->type != GGML_TYPE_F32) v = ggml_cast(ctx,v,GGML_TYPE_F32);
    if (!ggml_is_contiguous(q)) q = ggml_cont(ctx,q);
    // Layers sharing the same mask can share compaction within this graph.
    char mask_name[64], order_name[64];
    std::snprintf(mask_name,sizeof(mask_name),"hsa_mask_%p_%lld",(void *)mask,(long long)k->ne[2]);
    std::snprintf(order_name,sizeof(order_name),"hsa_order_%p_%lld",(void *)mask,(long long)k->ne[2]);
    ggml_tensor * m = ggml_get_tensor(ctx,mask_name);
    ggml_tensor * order = chronological_order ? chronological_order : ggml_get_tensor(ctx,order_name);
    if (!m) {
        m = ggml_view_1d(ctx,mask,k->ne[2],0);
        if (m->type != GGML_TYPE_F32) m = ggml_cast(ctx,m,GGML_TYPE_F32);
        ggml_set_name(m,mask_name);
    }
    if (!order) {
        order = ggml_hsa_compact(ctx,m);
        ggml_set_name(order,order_name);
    }
    ggml_tensor * meta = ggml_hsa_query(ctx,q,Hkv,std::min<int64_t>(r,D));
    ggml_tensor * logits;
    if(maximum && minimum && maximum->ne[1] >= (N+chunk-1)/chunk) {
        logits=ggml_hsa_decode_logits(ctx,meta,maximum,minimum,k,order,m,chunk);
    } else {
        logits=ggml_hsa_logits(ctx,meta,k,order,m,chunk);
    }
    ggml_tensor * selection = ggml_hsa_select(ctx,logits,keep);
    ggml_tensor * rows = ggml_view_2d(ctx,selection,keep,Hkv,selection->nb[1],logits->ne[0]*sizeof(int32_t));
    if (selected_rows) *selected_rows = rows;
    ggml_tensor * packed = ggml_hsa_gather(ctx,k,v,order,rows,m);
    const size_t block_bytes = D*keep*Hkv*sizeof(ggml_fp16_t);
    ggml_tensor * q_fa = ggml_reshape_4d(ctx,q,D,1,H,Hkv);
    ggml_tensor * k_fa = ggml_view_4d(ctx,packed,D,keep,1,Hkv,D*2,D*keep*2,D*keep*2,0);
    ggml_tensor * v_fa = ggml_view_4d(ctx,packed,D,keep,1,Hkv,D*2,D*keep*2,D*keep*2,block_bytes);
    ggml_tensor * m_fa = ggml_view_4d(ctx,packed,keep,1,1,Hkv,keep*2,keep*2,keep*2,2*block_bytes);
    ggml_tensor * out = ggml_flash_attn_ext(ctx,q_fa,k_fa,v_fa,m_fa,scale,0.f,0.f);
    ggml_prec_set_acc(out,GGML_PREC_F32);
    return ggml_reshape_2d(ctx,ggml_cont(ctx,out),D*H*Hkv,1);
}
