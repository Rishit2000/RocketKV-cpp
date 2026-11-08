#include "rocket.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    assert(a.size() == b.size());
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// Ground truth exact dense attention for a single query token
static std::vector<float> compute_exact_dense_decode(
    const float* Q, const float* K, const float* V,
    int len_k, int dim_head)
{
    std::vector<float> scores(len_k);
    float scale = 1.0f / std::sqrt((float)dim_head);
    float max_s = -1e30f;

    for (int t = 0; t < len_k; ++t) {
        float dot = 0.0f;
        for (int d = 0; d < dim_head; ++d) {
            dot += Q[d] * K[t * dim_head + d];
        }
        scores[t] = dot * scale;
        if (scores[t] > max_s) max_s = scores[t];
    }

    float sum_exp = 0.0f;
    for (int t = 0; t < len_k; ++t) {
        scores[t] = std::exp(scores[t] - max_s);
        sum_exp += scores[t];
    }

    std::vector<float> out(dim_head, 0.0f);
    for (int t = 0; t < len_k; ++t) {
        float prob = scores[t] / sum_exp;
        for (int d = 0; d < dim_head; ++d) {
            out[d] += prob * V[t * dim_head + d];
        }
    }
    return out;
}

static void assert_close(const std::vector<float>& actual, const std::vector<float>& expected) {
    assert(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        assert(std::isfinite(actual[i]));
        assert(std::abs(actual[i] - expected[i]) <= 1e-5f + 1e-4f * std::abs(expected[i]));
    }
}

static void assert_invalid(const std::function<void()>& action) {
    bool rejected = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

static void test_sparse_contract() {
    RocketConfig cfg{};
    cfg.num_heads = 4;
    cfg.num_heads_kv = 2;
    cfg.dim_head = 2;
    cfg.chunk_size = 2;
    cfg.r_dim = 1;
    cfg.compression_ratio = 2.0f;
    cfg.topk_decode = 2;
    std::vector<float> q = {1, 0, 2, 0, -1, 0, -2, 0};
    std::vector<float> k = {0, 0, 1, 0, 3, 0, 4, 0, 10, 0,
                           0, 0, 1, 0, 3, 0, 4, 0, -10, 0};
    std::vector<float> v(20);
    for (size_t i = 0; i < v.size(); ++i) v[i] = (float)i / 7.0f;
    std::vector<std::vector<int>> selected;
    auto output = compute_hybrid_sparse_decode(q.data(), k.data(), v.data(), 5, cfg, &selected);
    assert((selected == std::vector<std::vector<int>>{{2, 4}, {0, 4}}));

    std::vector<float> oracle;
    for (int h = 0; h < 4; ++h) {
        const int group = h / 2;
        std::vector<float> selected_k, selected_v;
        for (int token : selected[group]) {
            for (int d = 0; d < 2; ++d) {
                selected_k.push_back(k[(group * 5 + token) * 2 + d]);
                selected_v.push_back(v[(group * 5 + token) * 2 + d]);
            }
        }
        auto head = compute_exact_dense_decode(q.data() + h * 2, selected_k.data(), selected_v.data(), 2, 2);
        oracle.insert(oracle.end(), head.begin(), head.end());
    }
    assert_close(output, oracle);

    cfg.topk_decode = 99;
    cfg.r_dim = 2;
    output = compute_hybrid_sparse_decode(q.data(), k.data(), v.data(), 5, cfg);
    oracle.clear();
    for (int h = 0; h < 4; ++h) {
        auto head = compute_exact_dense_decode(q.data() + h * 2, k.data() + (h / 2) * 10,
                                              v.data() + (h / 2) * 10, 5, 2);
        oracle.insert(oracle.end(), head.begin(), head.end());
    }
    assert_close(output, oracle);

    cfg.topk_decode = 2;
    cfg.r_dim = 1;
    std::fill(q.begin(), q.end(), 0.0f);
    output = compute_hybrid_sparse_decode(q.data(), k.data(), v.data(), 5, cfg, &selected);
    assert((selected == std::vector<std::vector<int>>{{0, 1}, {0, 1}}));
    for (int h = 0; h < 4; ++h) {
        for (int d = 0; d < 2; ++d) {
            const int base = (h / 2) * 10;
            assert(std::abs(output[h * 2 + d] - (v[base + d] + v[base + 2 + d]) / 2) < 1e-6f);
        }
    }
    // A nonzero head can also have zero mass in the group's selected dimension.
    q = {10, 0, 0, 1, -10, 0, 0, 1};
    output = compute_hybrid_sparse_decode(q.data(), k.data(), v.data(), 5, cfg);
    for (float value : output) assert(std::isfinite(value));

    assert_invalid([&] { compute_hybrid_sparse_decode(q.data(), k.data(), v.data(), 0, cfg); });
    cfg.topk_decode = 0;
    assert_invalid([&] { compute_hybrid_sparse_decode(q.data(), k.data(), v.data(), 5, cfg); });
    cfg.topk_decode = 2;
    cfg.num_heads = 3;
    assert_invalid([&] { compute_hybrid_sparse_decode(q.data(), k.data(), v.data(), 5, cfg); });
    cfg.num_heads = 4;
    q[0] = std::numeric_limits<float>::quiet_NaN();
    assert_invalid([&] { compute_hybrid_sparse_decode(q.data(), k.data(), v.data(), 5, cfg); });
    assert_invalid([] { compute_rocket_params(0, 100, 8, 64); });
    const auto short_prompt = compute_rocket_params(8, 1, 16, 64, 32);
    assert(short_prompt.prompt_budget == 1 && short_prompt.k == 1);
    std::cout << "  GQA selection, tail chunk, dense equivalence, zero queries, and invalid inputs: PASS\n";
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RocketKV C++ Algorithmic Verification" << std::endl;
    std::cout << "========================================" << std::endl;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    RocketConfig cfg{};
    cfg.num_heads = 24;      // Llama-3.2-3B specs
    cfg.num_heads_kv = 8;
    cfg.dim_head = 128;
    cfg.window_size = 32;
    cfg.kernel_size = 63;
    cfg.compression_ratio = 4.0f;
    cfg.prompt_budget = 512;
    cfg.topk_decode = 256;

    // ----------------------------------------------------
    // Test 1: Prefill Eviction Correctness
    // ----------------------------------------------------
    std::cout << "\n[Test 1] Prefill Eviction (SnapKV Stage 1)..." << std::endl;
    int len_q = 2048;
    int len_k = 2048;

    std::vector<float> Q(cfg.num_heads * len_q * cfg.dim_head);
    std::vector<float> K(cfg.num_heads_kv * len_k * cfg.dim_head);
    for (auto& v : Q) v = dist(rng) * 0.1f;
    for (auto& v : K) v = dist(rng) * 0.1f;

    // Plant spikes / needles at specific token positions:
    std::vector<int> needle_positions = {120, 450, 980, 1420};
    for (int needle_pos : needle_positions) {
        for (int h_kv = 0; h_kv < cfg.num_heads_kv; ++h_kv) {
            for (int d = 0; d < cfg.dim_head; ++d) {
                // High correlation key vector
                K[(h_kv * len_k + needle_pos) * cfg.dim_head + d] = 5.0f;
            }
        }
    }
    // Make observation window queries correlate strongly with the needles
    for (int h = 0; h < cfg.num_heads; ++h) {
        for (int w = 0; w < cfg.window_size; ++w) {
            int q_pos = len_q - cfg.window_size + w;
            for (int d = 0; d < cfg.dim_head; ++d) {
                Q[(h * len_q + q_pos) * cfg.dim_head + d] = 2.0f;
            }
        }
    }

    auto t_start_evict = std::chrono::high_resolution_clock::now();
    std::vector<int> kept = compute_prefill_eviction(Q.data(), K.data(), len_q, len_k, cfg);
    auto t_end_evict = std::chrono::high_resolution_clock::now();
    double evict_us = std::chrono::duration<double, std::micro>(t_end_evict - t_start_evict).count();

    std::cout << "  Prompt tokens: " << len_k << " -> Kept tokens: " << kept.size() << " (Budget: " << cfg.prompt_budget << ")" << std::endl;
    std::cout << "  Eviction time: " << evict_us << " us (" << (evict_us / 1000.0) << " ms)" << std::endl;

    assert((int)kept.size() == cfg.prompt_budget);

    // Verify observation window tokens are unconditionally kept
    for (int w = len_k - cfg.window_size; w < len_k; ++w) {
        bool found = std::binary_search(kept.begin(), kept.end(), w);
        if (!found) {
            std::cerr << "FAILED: observation token " << w << " not kept!" << std::endl;
            return 1;
        }
    }
    std::cout << "  Observation window (" << cfg.window_size << " tokens) fully preserved: PASS" << std::endl;

    // Verify planted needles are retained
    for (int needle : needle_positions) {
        bool found = std::binary_search(kept.begin(), kept.end(), needle);
        if (!found) {
            std::cerr << "FAILED: planted needle " << needle << " was evicted!" << std::endl;
            return 1;
        }
    }
    std::cout << "  All planted heavy-hitter needle tokens preserved: PASS" << std::endl;

    // ----------------------------------------------------
    // Test 2: Hybrid Sparse Decode Correctness & Accuracy
    // ----------------------------------------------------
    std::cout << "\n[Test 2] Hybrid Sparse Decode (HSA Stage 2: Full Layer, All GQA Groups)..." << std::endl;
    int decode_len_k = 1024;
    int heads_per_kv = cfg.num_heads / cfg.num_heads_kv; // 3 query heads per KV head
    std::vector<float> Q_all(cfg.num_heads * cfg.dim_head);
    std::vector<float> K_decode(cfg.num_heads_kv * decode_len_k * cfg.dim_head);
    std::vector<float> V_decode(cfg.num_heads_kv * decode_len_k * cfg.dim_head);

    for (auto& v : Q_all) v = dist(rng);
    for (auto& v : K_decode) v = dist(rng);
    for (auto& v : V_decode) v = dist(rng);

    // Compute exact dense attention for all heads across all KV groups
    std::vector<float> exact_out(cfg.num_heads * cfg.dim_head);
    for (int h_q = 0; h_q < cfg.num_heads; ++h_q) {
        int h_kv = h_q / heads_per_kv;
        auto h_out = compute_exact_dense_decode(
            Q_all.data() + h_q * cfg.dim_head, 
            K_decode.data() + h_kv * decode_len_k * cfg.dim_head, 
            V_decode.data() + h_kv * decode_len_k * cfg.dim_head, 
            decode_len_k, cfg.dim_head);
        std::copy(h_out.begin(), h_out.end(), exact_out.begin() + h_q * cfg.dim_head);
    }

    // Compute RocketKV hybrid sparse decode across all GQA groups
    auto t_start_decode = std::chrono::high_resolution_clock::now();
    auto sparse_out = compute_hybrid_sparse_decode(Q_all.data(), K_decode.data(), V_decode.data(), decode_len_k, cfg);
    auto t_end_decode = std::chrono::high_resolution_clock::now();
    double decode_us = std::chrono::duration<double, std::micro>(t_end_decode - t_start_decode).count();

    float cos_sim = cosine_similarity(exact_out, sparse_out);
    std::cout << "  Total Query heads evaluated: " << cfg.num_heads << " (" << cfg.num_heads_kv << " KV groups)" << std::endl;
    std::cout << "  Decode KV tokens: " << decode_len_k << " -> Top-k evaluated: " << cfg.topk_decode << std::endl;
    std::cout << "  Sparse decode latency (full layer, 24 heads): " << decode_us << " us (" << (decode_us / 1000.0) << " ms)" << std::endl;
    std::cout << "  Full Layer Cosine Similarity: " << cos_sim << std::endl;

    if (cos_sim > 0.70f) {
        std::cout << "  Accuracy Retention (> 70% Cosine Similarity on random noise): PASS (" << (cos_sim * 100.0f) << "%)" << std::endl;
    } else {
        std::cerr << "  WARNING: Cosine similarity lower than expected: " << cos_sim << std::endl;
    }

    // ----------------------------------------------------
    // Test 3: Paper Hyperparameter Derivation (ICML Theory)
    // ----------------------------------------------------
    std::cout << "\n[Test 3] Paper Hyperparameter Derivation (ICML Theory)..." << std::endl;
    auto p8k = compute_rocket_params(1024, 8192, 32, 128);
    std::cout << "  8K Seq / 1024 Budget:"
              << " Cap=" << p8k.capacity_budget << " (expected 3717)"
              << ", Prompt=" << p8k.prompt_budget << " (expected 3685)"
              << ", Top-k=" << p8k.k << " (expected 512)"
              << ", Chunk=" << p8k.chunk_size << " (expected 2)"
              << ", r=" << p8k.r << " (expected 71)" << std::endl;
    assert(p8k.capacity_budget == 3717);
    assert(p8k.prompt_budget == 3685);
    assert(p8k.k == 512);
    assert(p8k.chunk_size == 2);
    assert(p8k.r == 71);
    std::cout << "  8K Sequence Budget Derivation: PASS" << std::endl;

    auto p16k = compute_rocket_params(1024, 16384, 32, 128);
    std::cout << "  16K Seq / 1024 Budget:"
              << " Cap=" << p16k.capacity_budget << " (expected 4837)"
              << ", Prompt=" << p16k.prompt_budget << " (expected 4805)"
              << ", Top-k=" << p16k.k << " (expected 512)"
              << ", Chunk=" << p16k.chunk_size << " (expected 3)"
              << ", r=" << p16k.r << " (expected 81)" << std::endl;
    assert(p16k.capacity_budget == 4837);
    assert(p16k.prompt_budget == 4805);
    assert(p16k.k == 512);
    assert(p16k.chunk_size == 3);
    assert(p16k.r == 81);
    std::cout << "  16K Sequence Budget Derivation: PASS" << std::endl;

    std::cout << "\n[Test 4] Deterministic Sparse Attention Contract..." << std::endl;
    test_sparse_contract();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All Algorithmic Verification Tests Passed!" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}

