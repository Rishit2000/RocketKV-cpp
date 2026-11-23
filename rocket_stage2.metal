#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// RocketKV Stage 2: Hybrid Sparse Attention (HSA) Metal Compute Kernel
// Target: Apple Silicon Unified Memory Architecture (Apple M4 GPU)
// -----------------------------------------------------------------------------

#define MAX_CHUNKS 4096
#define MAX_TOP_CHUNKS 2048

struct RocketMetalParams {
    int num_heads;       // Total query heads (e.g. 24)
    int num_heads_kv;    // Total KV heads (e.g. 8)
    int heads_per_kv;    // Query heads per KV head (e.g. 3)
    int dim_head;        // Head dimension d (e.g. 128)
    int len_k;           // Current KV cache sequence length
    int chunk_size;      // Chunk grouping size c (e.g. 2, 4)
    int num_chunks;      // Number of chunks = ceil(len_k / chunk_size)
    int r_dim;           // Truncated query dimension r (e.g. 64)
    int topk;            // Top-k tokens to attend to (e.g. 256, 512)
    float inv_sqrt_d;    // 1.0 / sqrt(dim_head)
};

// -----------------------------------------------------------------------------
// Baseline Exact Dense Attention Kernel (for ground-truth GPU verification)
// Dispatched with: Grid = (num_heads, 1, 1), Threadgroup = (32, 1, 1)
// -----------------------------------------------------------------------------
kernel void kernel_exact_dense_decode(
    device const float * Q         [[buffer(0)]], // [num_heads, dim_head]
    device const float * K         [[buffer(1)]], // [num_heads_kv, len_k, dim_head]
    device const float * V         [[buffer(2)]], // [num_heads_kv, len_k, dim_head]
    device float       * Out       [[buffer(3)]], // [num_heads, dim_head]
    constant RocketMetalParams & p [[buffer(4)]],
    uint   h_q   [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort simds [[threads_per_threadgroup]])
{
    int h_kv = h_q / p.heads_per_kv;
    device const float * q_ptr = Q + h_q * p.dim_head;
    device const float * k_base = K + h_kv * p.len_k * p.dim_head;
    device const float * v_base = V + h_kv * p.len_k * p.dim_head;
    device float * out_ptr = Out + h_q * p.dim_head;

    // Running online softmax variables (FlashAttention-style)
    float m_prev = -1e30f;
    float l_prev = 0.0f;

    // Local accumulator for output dimension elements handled by this thread
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int d0 = tid * 4;

    for (int t = 0; t < p.len_k; ++t) {
        device const float * k_t = k_base + t * p.dim_head;
        device const float * v_t = v_base + t * p.dim_head;

        // Compute dot product Q * K[t] across dim_head
        float dot = 0.0f;
        for (int d = tid; d < p.dim_head; d += simds) {
            dot += q_ptr[d] * k_t[d];
        }
        dot = simd_sum(dot);

        float score = dot * p.inv_sqrt_d;

        // Online softmax update:
        float m_curr = max(m_prev, score);
        float exp_score = exp(score - m_curr);
        float alpha = (m_prev == -1e30f) ? 0.0f : exp(m_prev - m_curr);
        l_prev = l_prev * alpha + exp_score;
        m_prev = m_curr;

        // Update output accumulator
        if (d0 < p.dim_head) {
            for (int i = 0; i < 4 && (d0 + i) < p.dim_head; ++i) {
                acc[i] = acc[i] * alpha + exp_score * v_t[d0 + i];
            }
        }
    }

    // Final normalize
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    if (d0 < p.dim_head) {
        for (int i = 0; i < 4 && (d0 + i) < p.dim_head; ++i) {
            out_ptr[d0 + i] = acc[i] * inv_l;
        }
    }
}

// -----------------------------------------------------------------------------
// RocketKV Stage 2: Fused Hybrid Sparse Decode Kernel
// Grid = (num_heads_kv, 1, 1), Threadgroup = (128, 1, 1)
// -----------------------------------------------------------------------------
kernel void kernel_rocket_sparse_decode(
    device const float * Q         [[buffer(0)]], // [num_heads, dim_head]
    device const float * K         [[buffer(1)]], // [num_heads_kv, len_k, dim_head]
    device const float * V         [[buffer(2)]], // [num_heads_kv, len_k, dim_head]
    device float       * Out       [[buffer(3)]], // [num_heads, dim_head]
    device float       * Scratch   [[buffer(4)]], // Scratch buffer [num_heads, num_chunks]
    constant RocketMetalParams & p [[buffer(5)]],
    uint   h_kv  [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort simds [[threads_per_threadgroup]])
{
    // Shared memory allocations
    threadgroup float shmem_q_sum_abs[128];
    threadgroup float shmem_q_sign[128];
    threadgroup int   shmem_top_r[128];
    threadgroup float shmem_custom_scale[8];
    threadgroup float shmem_sum_abs_all[8];
    threadgroup float shmem_chunk_scores[MAX_CHUNKS];
    threadgroup bool  shmem_selected[MAX_CHUNKS];
    threadgroup int   shmem_top_chunks[MAX_TOP_CHUNKS];
    threadgroup int   shmem_num_selected;
    threadgroup float sg_sums[4];
    threadgroup int   sg_int_sums[4];

    ushort sg_id = tid / 32;
    ushort lane_id = tid % 32;

    int heads_per_kv = p.heads_per_kv;
    device const float * q_group = Q + h_kv * heads_per_kv * p.dim_head;
    device const float * k_base = K + h_kv * p.len_k * p.dim_head;
    device const float * v_base = V + h_kv * p.len_k * p.dim_head;
    device float * out_group = Out + h_kv * heads_per_kv * p.dim_head;

    int scratch_offset = h_kv * heads_per_kv * p.num_chunks;

    // -------------------------------------------------------------------------
    // Phase 1: Query Dimension Consensus & Top-r Energy Dimension Selection
    // -------------------------------------------------------------------------
    if (tid < p.dim_head) {
        float sum_abs_d = 0.0f;
        float sum_val_d = 0.0f;
        for (int h = 0; h < heads_per_kv; ++h) {
            float q_val = q_group[h * p.dim_head + tid];
            sum_abs_d += abs(q_val);
            sum_val_d += q_val;
        }
        shmem_q_sum_abs[tid] = sum_abs_d;
        shmem_q_sign[tid] = (sum_val_d >= 0.0f) ? 1.0f : -1.0f;
    }

    if (tid < heads_per_kv) {
        float sum_all = 0.0f;
        for (int d = 0; d < p.dim_head; ++d) {
            sum_all += abs(q_group[tid * p.dim_head + d]);
        }
        shmem_sum_abs_all[tid] = sum_all;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 sorts dimensions by energy sum_abs_d to find top-r dimensions
    int r_bound = min(p.r_dim, p.dim_head);
    if (tid == 0) {
        for (int i = 0; i < p.dim_head; ++i) {
            shmem_top_r[i] = i;
        }
        for (int i = 0; i < r_bound; ++i) {
            int max_idx = i;
            float max_val = shmem_q_sum_abs[shmem_top_r[i]];
            for (int j = i + 1; j < p.dim_head; ++j) {
                float val = shmem_q_sum_abs[shmem_top_r[j]];
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            int tmp = shmem_top_r[i];
            shmem_top_r[i] = shmem_top_r[max_idx];
            shmem_top_r[max_idx] = tmp;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute custom variance scale for each query head in the GQA group
    if (tid < heads_per_kv) {
        float sum_kept = 0.0f;
        for (int ri = 0; ri < r_bound; ++ri) {
            int d = shmem_top_r[ri];
            sum_kept += abs(q_group[tid * p.dim_head + d]);
        }
        shmem_custom_scale[tid] = sqrt((float)p.dim_head * (sum_kept / (shmem_sum_abs_all[tid] + 1e-9f)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 2: Chunk-Max Key Approximation
    // -------------------------------------------------------------------------
    for (int c = tid; c < p.num_chunks; c += simds) {
        int start_tok = c * p.chunk_size;
        int end_tok = min(start_tok + p.chunk_size, p.len_k);

        // Precompute representative chunk key P_chunk for top-r dimensions
        for (int h = 0; h < heads_per_kv; ++h) {
            device const float * q_h = q_group + h * p.dim_head;
            float approx_dot = 0.0f;

            for (int ri = 0; ri < r_bound; ++ri) {
                int d = shmem_top_r[ri];
                float sign_d = shmem_q_sign[d];
                float p_val = (sign_d >= 0.0f) ? -1e30f : 1e30f;

                for (int t = start_tok; t < end_tok; ++t) {
                    float k_val = k_base[t * p.dim_head + d];
                    if (sign_d >= 0.0f) {
                        p_val = max(p_val, k_val);
                    } else {
                        p_val = min(p_val, k_val);
                    }
                }
                approx_dot += q_h[d] * p_val;
            }

            float logit = approx_dot / shmem_custom_scale[h];
            Scratch[scratch_offset + h * p.num_chunks + c] = logit;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 2b & 2c: Scaled Softmax across Chunks & Group Consensus Scoring
    // -------------------------------------------------------------------------
    for (int h = 0; h < heads_per_kv; ++h) {
        // Step 1: Find max logit across chunks for head h
        float t_max = -1e30f;
        for (int c = tid; c < p.num_chunks; c += simds) {
            t_max = max(t_max, Scratch[scratch_offset + h * p.num_chunks + c]);
        }
        float s_max = simd_max(t_max);
        if (lane_id == 0) sg_sums[sg_id] = s_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float max_l = max(max(sg_sums[0], sg_sums[1]), max(sg_sums[2], sg_sums[3]));

        // Step 2: Compute exp and sum
        float t_sum = 0.0f;
        for (int c = tid; c < p.num_chunks; c += simds) {
            float e = exp(Scratch[scratch_offset + h * p.num_chunks + c] - max_l);
            Scratch[scratch_offset + h * p.num_chunks + c] = e;
            t_sum += e;
        }
        float s_sum = simd_sum(t_sum);
        if (lane_id == 0) sg_sums[sg_id] = s_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum_exp = sg_sums[0] + sg_sums[1] + sg_sums[2] + sg_sums[3];
        float inv_sum = 1.0f / (sum_exp + 1e-9f);

        // Step 3: Normalize to get probabilities
        for (int c = tid; c < p.num_chunks; c += simds) {
            Scratch[scratch_offset + h * p.num_chunks + c] *= inv_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Step 4: Sum chunk probabilities across query heads to form group consensus
    for (int c = tid; c < p.num_chunks; c += simds) {
        float grp_prob = 0.0f;
        for (int h = 0; h < heads_per_kv; ++h) {
            grp_prob += Scratch[scratch_offset + h * p.num_chunks + c];
        }
        shmem_chunk_scores[c] = grp_prob;
        shmem_selected[c] = false;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 3: Parallel Top-K Chunk Selection with Threshold Search
    // -------------------------------------------------------------------------
    int chunks_to_select = min(p.num_chunks, max(1, (p.topk + p.chunk_size - 1) / p.chunk_size));

    // Parallel binary search across threadgroup to locate score cutoff threshold
    float t_low = 0.0f;
    float t_high = (float)heads_per_kv;

    for (int iter = 0; iter < 16; ++iter) {
        float t_mid = 0.5f * (t_low + t_high);
        int local_cnt = 0;
        for (int c = tid; c < p.num_chunks; c += simds) {
            if (shmem_chunk_scores[c] >= t_mid) local_cnt++;
        }
        int s_cnt = simd_sum(local_cnt);
        if (lane_id == 0) sg_int_sums[sg_id] = s_cnt;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        int total_cnt = sg_int_sums[0] + sg_int_sums[1] + sg_int_sums[2] + sg_int_sums[3];
        if (total_cnt >= chunks_to_select) {
            t_low = t_mid;
        } else {
            t_high = t_mid;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 flags selected chunks and gathers them in chronological order
    if (tid == 0) {
        int cnt = 0;
        // Pass 1: Select chunks strictly above cutoff
        for (int c = 0; c < p.num_chunks && cnt < chunks_to_select; ++c) {
            if (shmem_chunk_scores[c] > t_low) {
                shmem_selected[c] = true;
                cnt++;
            }
        }
        // Pass 2: Tie-breaking for boundary scores (smaller chunk index priority)
        for (int c = 0; c < p.num_chunks && cnt < chunks_to_select; ++c) {
            if (!shmem_selected[c] && shmem_chunk_scores[c] >= t_low - 1e-6f) {
                shmem_selected[c] = true;
                cnt++;
            }
        }
        // Pass 3: Fallback if any remain
        while (cnt < chunks_to_select) {
            int best_c = -1;
            float best_s = -1.0f;
            for (int c = 0; c < p.num_chunks; ++c) {
                if (!shmem_selected[c] && shmem_chunk_scores[c] > best_s) {
                    best_s = shmem_chunk_scores[c];
                    best_c = c;
                }
            }
            if (best_c >= 0) {
                shmem_selected[best_c] = true;
                cnt++;
            } else break;
        }

        // Gather indices in strict chronological order
        int idx = 0;
        for (int c = 0; c < p.num_chunks && idx < chunks_to_select; ++c) {
            if (shmem_selected[c]) {
                shmem_top_chunks[idx++] = c;
            }
        }
        shmem_num_selected = idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -------------------------------------------------------------------------
    // Phase 4: Exact Sparse Attention on Selected Chunks (128-thread reduction)
    // -------------------------------------------------------------------------
    for (int h = 0; h < heads_per_kv; ++h) {
        device const float * q_h = q_group + h * p.dim_head;
        device float * out_h = out_group + h * p.dim_head;

        float m_prev = -1e30f;
        float l_prev = 0.0f;
        float acc = 0.0f;

        int gathered_tokens = 0;

        for (int sc = 0; sc < shmem_num_selected && gathered_tokens < p.topk; ++sc) {
            int c_idx = shmem_top_chunks[sc];
            int start_tok = c_idx * p.chunk_size;
            int end_tok = min(start_tok + p.chunk_size, p.len_k);

            for (int t = start_tok; t < end_tok && gathered_tokens < p.topk; ++t) {
                device const float * k_t = k_base + t * p.dim_head;
                device const float * v_t = v_base + t * p.dim_head;

                // Dot product across 128 threads (each thread holds q_h[tid] * k_t[tid])
                float prod = (tid < p.dim_head) ? (q_h[tid] * k_t[tid]) : 0.0f;
                float s = simd_sum(prod);
                if (lane_id == 0) sg_sums[sg_id] = s;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float dot = sg_sums[0] + sg_sums[1] + sg_sums[2] + sg_sums[3];

                float score = dot * p.inv_sqrt_d;

                // Online softmax update
                float m_curr = max(m_prev, score);
                float exp_score = exp(score - m_curr);
                float alpha = (m_prev == -1e30f) ? 0.0f : exp(m_prev - m_curr);
                l_prev = l_prev * alpha + exp_score;
                m_prev = m_curr;

                if (tid < p.dim_head) {
                    acc = acc * alpha + exp_score * v_t[tid];
                }
                gathered_tokens++;
            }
        }

        // Final normalize and write output
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        if (tid < p.dim_head) {
            out_h[tid] = acc * inv_l;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
