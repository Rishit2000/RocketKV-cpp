#include "rocket.h"
#include <cmath>
#include <algorithm>
#include <cfloat>
#ifdef _OPENMP
#include <omp.h>
#endif

struct TokenScore {
    float score;
    int memory_idx;
};

RocketBudgetParams compute_rocket_params(
    int token_budget, 
    int sequence_length, 
    int max_new_tokens, 
    int head_dim, 
    int window_size) 
{
    RocketBudgetParams p;
    p.sequence_length = sequence_length;
    p.max_new_tokens = max_new_tokens;
    p.token_budget = std::min(sequence_length, token_budget);
    
    // Initial compression ratio
    float cr_initial = std::max(1.0f, (float)sequence_length / (float)p.token_budget);
    
    // Adaptive scaling exponent: alpha = min(0.2 + log2(CR) * 0.06, 0.8)
    float alpha = std::min(0.2f + std::log2(cr_initial) * 0.06f, 0.8f);
    
    // Total capacity budget allowed across prompt + generation
    int capacity_budget = (int)std::floor((float)sequence_length / std::pow(cr_initial, alpha));
    capacity_budget = std::max(capacity_budget, std::min(2 * max_new_tokens, sequence_length));
    p.capacity_budget = capacity_budget;
    
    // Prompt budget for Stage 1 prefill eviction
    int prompt_budget = std::min(sequence_length - max_new_tokens, capacity_budget - max_new_tokens);
    prompt_budget = std::max(window_size, prompt_budget);
    p.prompt_budget = prompt_budget;
    
    // Stage 2 decode top-k budget
    p.k = std::min((int)std::round((float)p.token_budget / 2.0f), capacity_budget);
    
    // Effective Stage 2 compression ratio
    float cr_stage2 = std::max(1.0f, (float)capacity_budget / (float)p.token_budget);
    p.compression_ratio = cr_stage2;
    
    // Chunk size
    int chunk_size = std::min((int)std::floor(cr_stage2), (int)std::ceil(std::sqrt(cr_stage2)));
    chunk_size = std::max(1, chunk_size);
    p.chunk_size = chunk_size;
    
    // Query dimension truncation
    int r = std::min(head_dim, std::max(1, (int)std::round((float)head_dim * (float)chunk_size / cr_stage2)));
    p.r = r;
    
    return p;
}

std::vector<int> compute_prefill_eviction(
    const float* Q, const float* K, 
    int len_q, int len_k, 
    const RocketConfig& cfg) 
{
    // If prompt is within budget, keep all tokens immediately
    if (len_k <= cfg.prompt_budget) {
        std::vector<int> kept_indices(len_k);
        for (int i = 0; i < len_k; ++i) kept_indices[i] = i;
        return kept_indices;
    }

    int obs_w = std::min(cfg.window_size, len_q);
    int non_obs_k = len_k - obs_w;
    int heads_per_kv = cfg.num_heads / cfg.num_heads_kv;
    float inv_sqrt_d = 1.0f / std::sqrt((float)cfg.dim_head);

    std::vector<float> pooled_scores(cfg.num_heads_kv * non_obs_k, 0.0f);
    int pad = cfg.kernel_size / 2;

    // Parallel Attention Scoring & 1D Max Pooling
    #pragma omp parallel for
    for (int h_kv = 0; h_kv < cfg.num_heads_kv; ++h_kv) {
        std::vector<float> head_scores(non_obs_k, 0.0f);

        for (int h_rel = 0; h_rel < heads_per_kv; ++h_rel) {
            int h = h_kv * heads_per_kv + h_rel;

            for (int w = 0; w < obs_w; ++w) {
                int q_pos = len_q - obs_w + w;
                const float* q_ptr = Q + (h * len_q + q_pos) * cfg.dim_head;

                float max_logit = -FLT_MAX;
                std::vector<float> logits(non_obs_k, -FLT_MAX);

                // Causal dot product up to the non-observation boundary
                for (int k_idx = 0; k_idx < non_obs_k; ++k_idx) {
                    float dot = 0.0f;
                    const float* k_ptr = K + (h_kv * len_k + k_idx) * cfg.dim_head;
                    for (int d = 0; d < cfg.dim_head; ++d) {
                        dot += q_ptr[d] * k_ptr[d];
                    }
                    logits[k_idx] = dot * inv_sqrt_d;
                    if (logits[k_idx] > max_logit) max_logit = logits[k_idx];
                }

                // Softmax accumulation
                float sum_exp = 0.0f;
                for (int k_idx = 0; k_idx < non_obs_k; ++k_idx) {
                    logits[k_idx] = std::exp(logits[k_idx] - max_logit);
                    sum_exp += logits[k_idx];
                }
                float inv_sum = 1.0f / (sum_exp + 1e-9f);
                for (int k_idx = 0; k_idx < non_obs_k; ++k_idx) {
                    head_scores[k_idx] += logits[k_idx] * inv_sum;
                }
            }
        }

        // 1D Max Pooling across the sequence
        for (int i = 0; i < non_obs_k; ++i) {
            float max_val = -FLT_MAX;
            int start_idx = std::max(0, i - pad);
            int end_idx = std::min(non_obs_k - 1, i + pad);
            for (int j = start_idx; j <= end_idx; ++j) {
                if (head_scores[j] > max_val) max_val = head_scores[j];
            }
            pooled_scores[h_kv * non_obs_k + i] = max_val;
        }
    }

    // Fast O(N) Selection of Top-K Survivors
    int target_keep = std::max(0, cfg.prompt_budget - obs_w);
    if (target_keep >= non_obs_k) {
        std::vector<int> kept_indices(len_k);
        for (int i = 0; i < len_k; ++i) kept_indices[i] = i;
        return kept_indices;
    }

    std::vector<TokenScore> token_importance(non_obs_k);

    for (int i = 0; i < non_obs_k; ++i) {
        float total_score = 0.0f;
        for (int h_kv = 0; h_kv < cfg.num_heads_kv; ++h_kv) {
            total_score += pooled_scores[h_kv * non_obs_k + i];
        }
        token_importance[i] = {total_score, i};
    }

    std::nth_element(
        token_importance.begin(), 
        token_importance.begin() + target_keep, 
        token_importance.end(),
        [](const TokenScore& a, const TokenScore& b) { return a.score > b.score; }
    );

    // Output chronologically sorted memory addresses
    std::vector<int> kept_indices;
    kept_indices.reserve(cfg.prompt_budget);
    for (int i = 0; i < target_keep; ++i) {
        kept_indices.push_back(token_importance[i].memory_idx);
    }
    for (int i = non_obs_k; i < len_k; ++i) {
        kept_indices.push_back(i); // Unconditionally keep observation window
    }
    std::sort(kept_indices.begin(), kept_indices.end());

    return kept_indices;
}

std::vector<float> compute_hybrid_sparse_decode(
    const float* Q, const float* K, const float* V,
    int len_k, const RocketConfig& cfg) 
{
    int heads_per_kv = std::max(1, cfg.num_heads / cfg.num_heads_kv);
    
    int chunk_size = (cfg.chunk_size > 0) ? cfg.chunk_size : 
        std::max(1, std::min((int)std::floor(cfg.compression_ratio), (int)std::ceil(std::sqrt(cfg.compression_ratio))));
    
    int num_chunks = (len_k + chunk_size - 1) / chunk_size;
    int r = (cfg.r_dim > 0) ? cfg.r_dim : 
        std::min(cfg.dim_head, std::max(1, (int)std::round((float)cfg.dim_head * (float)chunk_size / cfg.compression_ratio)));
    r = std::min(r, cfg.dim_head);
    int final_k = std::min(cfg.topk_decode, len_k);
    std::vector<float> final_output(cfg.num_heads * cfg.dim_head, 0.0f);
    if (len_k <= 0 || final_k <= 0) return final_output;

    float inv_sqrt_d = 1.0f / std::sqrt((float)cfg.dim_head);

    #pragma omp parallel for
    for (int h_kv = 0; h_kv < cfg.num_heads_kv; ++h_kv) {
        const float* k_base = K + (h_kv * len_k * cfg.dim_head);
        const float* v_base = V + (h_kv * len_k * cfg.dim_head);
        const float* q_group = Q + (h_kv * heads_per_kv * cfg.dim_head);
        float* out_group = final_output.data() + (h_kv * heads_per_kv * cfg.dim_head);

        // Query Truncation Agreement across all Query Heads in this GQA Group
        std::vector<TokenScore> abs_q_group(cfg.dim_head);
        std::vector<float> group_q_sum(cfg.dim_head, 0.0f);
        std::vector<float> sum_abs_all(heads_per_kv, 0.0f);

        // for each query head in the group, compute the sum over all of its entries
        for (int d = 0; d < cfg.dim_head; ++d) {
            float sum_abs_d = 0.0f;
            float sum_val_d = 0.0f;
            for (int h = 0; h < heads_per_kv; ++h) {
                float q_val = q_group[h * cfg.dim_head + d];
                float abs_val = std::abs(q_val);
                sum_abs_d += abs_val;
                sum_val_d += q_val;
                sum_abs_all[h] += abs_val;
            }
            abs_q_group[d] = {sum_abs_d, d};
            group_q_sum[d] = sum_val_d;
        }

        // Determine group consensus sign for key fetching
        std::vector<float> sign(cfg.dim_head);
        for (int d = 0; d < cfg.dim_head; ++d) {
            //note(keshav): to be consistent with the pytorch impl: set sign[d] = -1 if sum over query heads = 0
            sign[d] = (group_q_sum[d] > 0.0f) ? 1.0f : -1.0f;
        }

        // Top-r dimension selection agreed upon by all query heads in the group
        std::vector<int> top_r_dims(r);
        std::vector<float> sum_abs_kept(heads_per_kv, 0.0f);

        if (r >= cfg.dim_head) {
            for (int d = 0; d < cfg.dim_head; ++d) {
                top_r_dims[d] = d;
            }
            sum_abs_kept = sum_abs_all;
        } else {
            std::nth_element(
                abs_q_group.begin(), abs_q_group.begin() + r, abs_q_group.end(),
                [](const TokenScore& a, const TokenScore& b) { return a.score > b.score; }
            );

            for (int ri = 0; ri < r; ri++) {
                int d = abs_q_group[ri].memory_idx;
                top_r_dims[ri] = d;
                for (int h = 0; h < heads_per_kv; ++h) {
                    sum_abs_kept[h] += std::abs(q_group[h * cfg.dim_head + d]);
                }
            }
        }

        // Variance scaling per query head in the group
        std::vector<float> custom_scale(heads_per_kv);
        for (int h = 0; h < heads_per_kv; ++h) {
            custom_scale[h] = std::sqrt((float)cfg.dim_head * (sum_abs_kept[h] / (sum_abs_all[h] + 1e-9f)));
        }

        // Chunk-Max Key Approximation & Per-Head Softmax Probability
        std::vector<float> chunk_logits(heads_per_kv * num_chunks, 0.0f);

        for (int c = 0; c < num_chunks; ++c) {
            int start_tok = c * chunk_size;
            int end_tok = std::min(start_tok + chunk_size, len_k);

            // Compute representative chunk key P = K_max if sign[d] >= 0 else K_min
            std::vector<float> P_chunk(r);
            for (int ri = 0; ri < r; ++ri) {
                int d = top_r_dims[ri];
                if (sign[d] >= 0.0f) {
                    float max_val = -FLT_MAX;
                    for (int t = start_tok; t < end_tok; ++t) {
                        float k_val = k_base[t * cfg.dim_head + d];
                        if (k_val > max_val) max_val = k_val;
                    }
                    P_chunk[ri] = max_val;
                } else {
                    float min_val = FLT_MAX;
                    for (int t = start_tok; t < end_tok; ++t) {
                        float k_val = k_base[t * cfg.dim_head + d];
                        if (k_val < min_val) min_val = k_val;
                    }
                    P_chunk[ri] = min_val;
                }
            }

            for (int h = 0; h < heads_per_kv; ++h) {
                float head_approx = 0.0f;
                const float* q_h = q_group + h * cfg.dim_head;
                for (int ri = 0; ri < r; ++ri) {
                    int d = top_r_dims[ri];
                    head_approx += q_h[d] * P_chunk[ri];
                }
                chunk_logits[h * num_chunks + c] = head_approx / custom_scale[h];
            }
        }

        // Scaled Softmax across chunks independently per query head
        std::vector<float> chunk_probs(heads_per_kv * num_chunks, 0.0f);
        for (int h = 0; h < heads_per_kv; ++h) {
            float max_l = -FLT_MAX;
            for (int c = 0; c < num_chunks; ++c) {
                if (chunk_logits[h * num_chunks + c] > max_l) {
                    max_l = chunk_logits[h * num_chunks + c];
                }
            }
            float sum_exp = 0.0f;
            for (int c = 0; c < num_chunks; ++c) {
                chunk_probs[h * num_chunks + c] = std::exp(chunk_logits[h * num_chunks + c] - max_l);
                sum_exp += chunk_probs[h * num_chunks + c];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);
            for (int c = 0; c < num_chunks; ++c) {
                chunk_probs[h * num_chunks + c] *= inv_sum;
            }
        }

        // Group consensus token scoring (summing approximate attention probabilities across the GQA group)
        std::vector<TokenScore> group_token_scores(len_k);
        for (int c = 0; c < num_chunks; ++c) {
            int start_tok = c * chunk_size;
            int end_tok = std::min(start_tok + chunk_size, len_k);
            float group_prob = 0.0f;
            for (int h = 0; h < heads_per_kv; ++h) {
                group_prob += chunk_probs[h * num_chunks + c];
            }
            for (int t = start_tok; t < end_tok; ++t) {
                group_token_scores[t] = {group_prob, t};
            }
        }

        // Top-K Token Selection Agreement (Shared by all heads in group)
        if (final_k < len_k) {
            std::nth_element(
                group_token_scores.begin(), group_token_scores.begin() + final_k, group_token_scores.end(),
                [](const TokenScore& a, const TokenScore& b) { 
                    if (a.score != b.score) return a.score > b.score;
                    return a.memory_idx < b.memory_idx;
                }
            );
        }

        std::vector<int> kept_token_indices(final_k);
        for (int i = 0; i < final_k; ++i) {
            kept_token_indices[i] = group_token_scores[i].memory_idx;
        }
        std::sort(kept_token_indices.begin(), kept_token_indices.end());

        // Exact Sparse Attention on the Selected K Tokens for Each Head
        std::vector<float> exact_scores(final_k);

        for (int h = 0; h < heads_per_kv; ++h) {
            const float* q_h = q_group + h * cfg.dim_head;
            float* out_h = out_group + h * cfg.dim_head;
            std::fill(out_h, out_h + cfg.dim_head, 0.0f);

            float max_exact_score = -FLT_MAX;

            for (int i = 0; i < final_k; ++i) {
                int t_idx = kept_token_indices[i];
                const float* k_t = k_base + t_idx * cfg.dim_head;
                float exact_dot = 0.0f;
                for (int d = 0; d < cfg.dim_head; ++d) {
                    exact_dot += q_h[d] * k_t[d];
                }
                exact_scores[i] = exact_dot * inv_sqrt_d;
                if (exact_scores[i] > max_exact_score) max_exact_score = exact_scores[i];
            }

            float sum_exp = 0.0f;
            for (int i = 0; i < final_k; ++i) {
                exact_scores[i] = std::exp(exact_scores[i] - max_exact_score);
                sum_exp += exact_scores[i];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-9f);

            for (int i = 0; i < final_k; ++i) {
                float prob = exact_scores[i] * inv_sum;
                int t_idx = kept_token_indices[i];
                const float* v_t = v_base + t_idx * cfg.dim_head;
                for (int d = 0; d < cfg.dim_head; ++d) {
                    out_h[d] += prob * v_t[d];
                }
            }
        }
    }

    return final_output;
}