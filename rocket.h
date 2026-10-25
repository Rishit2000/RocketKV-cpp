#pragma once
#include <vector>

// Parameters mathematically derived at runtime
struct RocketBudgetParams {
    int token_budget;       // Target user-specified KV budget (e.g. 512, 1024, 2048)
    int sequence_length;    // Actual prompt sequence length (N)
    int max_new_tokens;     // Expected generation length (T)
    int capacity_budget;    // Total allowed KV capacity across prefill + decode
    int prompt_budget;      // Tokens retained after Stage 1 prefill eviction
    int chunk_size;         // Stage 2 chunk size (c)
    int r;                  // Stage 2 low-rank query projection dimension
    int k;                  // Stage 2 decode top-k token budget
    float compression_ratio; // Effective compression ratio (CR)
};

// Configuration container passed to prefill eviction and decode kernels
struct RocketConfig {
    // Model Architecture (probed directly from the GGUF model weights)
    int num_heads;
    int num_heads_kv;
    int dim_head;

    // Fixed Paper Hyperparameters
    int window_size = 32;       // Local observation window (recent tokens always kept)
    int kernel_size = 63;       // 1D max-pooling smoothing kernel size

    // Runtime Dynamic Allocation (populated by compute_rocket_params)
    int prompt_budget;      // Stage 1 tokens to retain
    int topk_decode;        // Stage 2 top-k tokens for sparse attention
    int chunk_size;         // Stage 2 chunk grouping size
    int r_dim;              // Stage 2 truncated query dimension
    float compression_ratio;
};

RocketBudgetParams compute_rocket_params(
    int token_budget, 
    int sequence_length, 
    int max_new_tokens, 
    int head_dim, 
    int window_size = 32);

std::vector<int> compute_prefill_eviction(
    const float* Q, const float* K, 
    int len_q, int len_k, 
    const RocketConfig& cfg);

std::vector<float> compute_hybrid_sparse_decode(
    const float* Q, const float* K, const float* V,
    int len_k, const RocketConfig& cfg);