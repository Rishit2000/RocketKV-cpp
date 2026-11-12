#include "hsa_graph.h"

static struct ggml_tensor * perm_cont(struct ggml_context * ctx, struct ggml_tensor * t, 
                                      int a0, int a1, int a2, int a3) {
    return ggml_cont(ctx, ggml_permute(ctx, t, a0, a1, a2, a3));
}

void run_hsa_head_dim_graph(
    ggml_backend_t backend,
    struct ggml_tensor * q,
    struct ggml_tensor * k,
    const RocketConfig & config) {
        if (!q || !k || !backend || config.r_dim <= 0) return;
        if (config.num_heads_kv <= 0 || config.dim_head <= 0) return;

        const int H = config.num_heads / config.num_heads_kv;
        const int Hkv = config.num_heads_kv;
        const int D = config.dim_head;
        const int r = config.r_dim;
        
        // num elements for Q should be (D, num_heads, seq_len)
        if (q->ne[0] != D || q->ne[1] != (int64_t) H * Hkv || q->ne[2] != 1) return;
        if (k->ne[0] != D || k->ne[1] != Hkv || k->ne[2] != 1) return;

        ggml_init_params gp = {256 * 1024, nullptr, true};
        struct ggml_context * gctx = ggml_init(gp);
        if (!gctx) return;

        // grouped-query attention reshape: index by group ID        
        struct ggml_tensor * Q = ggml_reshape_4d(gctx, q, D, H, Hkv, 1);
        struct ggml_tensor * K = perm_cont(gctx, k, 0, 2, 1, 3); // make K broadcastable
        if (Q->type != GGML_TYPE_F32) Q = ggml_cast(gctx, Q, GGML_TYPE_F32);
        if (K->type != GGML_TYPE_F32) K = ggml_cast(gctx, K, GGML_TYPE_F32);

        // compute the sum of dimensions by kv head over all groups
        struct ggml_tensor * group_sum = ggml_sum_rows(gctx, perm_cont(gctx, Q, 1, 0, 2, 3));

        // stepped contains 0-1 indicators, 2 * stepped - 1 gives sign
        struct ggml_tensor * stepped = ggml_step(gctx, group_sum);
        struct ggml_tensor * two = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
        struct ggml_tensor * one = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);
        struct ggml_tensor * sign_hd = ggml_sub(gctx, 
                ggml_mul(gctx, stepped, ggml_repeat(gctx, two, stepped)),
                ggml_repeat(gctx, one, stepped));
        struct ggml_tensor * sign = perm_cont(gctx, sign_hd, 1, 0, 2, 3);

        // Qs is a (D, H, Hkv, 1) tensor
        struct ggml_tensor * Qs = ggml_mul(gctx, Q, sign);
        struct ggml_tensor * Ks = ggml_mul(gctx, K, sign);

        // reduce mean of absolute values of Q over group dimension
        struct ggml_tensor * abs_q = ggml_abs(gctx, Q);
        // scores is dim (D, 1, Hkv, 1) tensor now
        struct ggml_tensor * scores = perm_cont(gctx, 
            ggml_mean(gcctx, perm_cont(gctx, abs_q, 1, 0, 2, 3)), 1, 0, 2, 3);
        struct ggml_tensor * idx = ggml_top_k(gctx, scores, r);

        // pick out the top-r entries of q by mean absolute value
        struct ggml_tensor * q_hat = perm_cont(gctx, 
            ggml_get_rows(gctx, perm_cont(gctx, Qs, 1, 0, 2, 3), idx), 1, 0, 2, 3);
        struct ggml_tensor * k_hat = perm_cont(gctx,
            ggml_get_rows(gctx, perm_cont(gctx, Ks, 1, 0, 2, 3), idx), 1, 0, 2, 3);
        struct ggml_tensor * sub_abs_all = ggml_sum_rows(gctx, abs_q);
        struct ggml_tensor * sum_abs_kept = ggml_sum_rows(gctx, ggml_abs(gctx, q_hat));        
        
        struct ggml_cgraph * gf = ggml_new_graph(gctx);
        ggml_build_forward_expand(gf, q_hat);
        ggml_build_forward_expand(gf, k_hat);
        ggml_build_forward_expand(gf, idx);
        ggml_build_forward_expand(gf, sum_abs_kept);
        ggml_build_forward_expand(gf, sum_abs_all);
        
        ggml_backend_buffer_t scratch = ggml_backend_alloc_ctx_tensors(gctx, backend);
        const float two_v = 2.0f;
        const float one_v = 1.0f;
        ggml_backend_tensor_set(two, &two_v, 0, sizeof(two_v));
        ggml_backend_tensor_set(one, &one_v, 0, sizeof(one_v));
        ggml_backend_graph_compute(backend, gf);
        if (scratch) ggml_backend_buffer_free(scratch);
        ggml_free(gctx);
}