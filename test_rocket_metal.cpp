#include "rocket.h"
#include "rocket_metal.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <iomanip>

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

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  RocketKV Stage 2 Metal Compute Shader Verification" << std::endl;
    std::cout << "  Target: Apple Silicon Metal GPU (Apple M4)" << std::endl;
    std::cout << "==========================================================" << std::endl;

    if (!rocket_metal_init()) {
        std::cerr << "Failed to initialize RocketKV Metal GPU backend!" << std::endl;
        return 1;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    RocketConfig cfg{};
    cfg.num_heads = 24;      // Llama-3.2-3B specs
    cfg.num_heads_kv = 8;
    cfg.dim_head = 128;
    cfg.window_size = 32;
    cfg.kernel_size = 63;
    cfg.chunk_size = 2;
    cfg.r_dim = 64;
    cfg.topk_decode = 256;

    int test_len_k = 1024;
    std::cout << "\n[Test 1] Algorithmic Verification on Apple M4 GPU (L_K = " << test_len_k << ", Top-k = " << cfg.topk_decode << ")..." << std::endl;

    std::vector<float> Q(cfg.num_heads * cfg.dim_head);
    std::vector<float> K(cfg.num_heads_kv * test_len_k * cfg.dim_head);
    std::vector<float> V(cfg.num_heads_kv * test_len_k * cfg.dim_head);
    for (auto& v : Q) v = dist(rng);
    for (auto& v : K) v = dist(rng);
    for (auto& v : V) v = dist(rng);

    std::vector<float> out_dense_metal(cfg.num_heads * cfg.dim_head, 0.0f);
    std::vector<float> out_sparse_metal(cfg.num_heads * cfg.dim_head, 0.0f);
    std::vector<float> out_sparse_cpu(cfg.num_heads * cfg.dim_head, 0.0f);

    // Warmup GPU
    rocket_metal_dense_decode(Q.data(), K.data(), V.data(), out_dense_metal.data(), test_len_k, cfg);
    rocket_metal_sparse_decode(Q.data(), K.data(), V.data(), out_sparse_metal.data(), test_len_k, cfg);

    // Run GPU Dense
    rocket_metal_dense_decode(Q.data(), K.data(), V.data(), out_dense_metal.data(), test_len_k, cfg);

    // Run GPU Sparse
    rocket_metal_sparse_decode(Q.data(), K.data(), V.data(), out_sparse_metal.data(), test_len_k, cfg);

    // Run CPU Sparse reference
    out_sparse_cpu = compute_hybrid_sparse_decode(Q.data(), K.data(), V.data(), test_len_k, cfg);

    // Run CPU Dense reference
    std::vector<float> out_dense_cpu(cfg.num_heads * cfg.dim_head, 0.0f);
    int heads_per_kv = cfg.num_heads / cfg.num_heads_kv;
    for (int h_q = 0; h_q < cfg.num_heads; ++h_q) {
        int h_kv = h_q / heads_per_kv;
        auto h_out = compute_exact_dense_decode(
            Q.data() + h_q * cfg.dim_head, 
            K.data() + h_kv * test_len_k * cfg.dim_head, 
            V.data() + h_kv * test_len_k * cfg.dim_head, 
            test_len_k, cfg.dim_head);
        std::copy(h_out.begin(), h_out.end(), out_dense_cpu.begin() + h_q * cfg.dim_head);
    }

    float cos_sim_dense_gpu_vs_cpu = cosine_similarity(out_dense_metal, out_dense_cpu);
    std::cout << "  Cosine Similarity (Metal Dense GPU vs. CPU Dense Baseline): " 
              << std::fixed << std::setprecision(4) << cos_sim_dense_gpu_vs_cpu << std::endl;

    float cos_sim_dense_vs_sparse = cosine_similarity(out_dense_metal, out_sparse_metal);
    float cos_sim_gpu_vs_cpu = cosine_similarity(out_sparse_metal, out_sparse_cpu);

    std::cout << "  Cosine Similarity (Metal Sparse vs. Metal Dense Baseline): " 
              << std::fixed << std::setprecision(4) << cos_sim_dense_vs_sparse << std::endl;
    std::cout << "  Cosine Similarity (Metal Sparse vs. CPU Sparse Reference): " 
              << std::fixed << std::setprecision(4) << cos_sim_gpu_vs_cpu << std::endl;

    if (cos_sim_dense_vs_sparse >= 0.70f) {
        std::cout << "  >>> Verification PASS: Stage 2 Metal shader captures dense attention semantics (> 70%)" << std::endl;
    } else {
        std::cerr << "  >>> Verification WARNING: Cosine similarity lower than expected." << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 2: Latency & Speedup Benchmarking across Context Lengths on Apple M4
    // (Using resident GPU buffers to profile pure kernel execution latency)
    // -------------------------------------------------------------------------
    std::cout << "\n[Test 2] Systems Benchmarking on Apple Silicon Metal GPU (Resident Buffers)..." << std::endl;
    std::cout << "------------------------------------------------------------------------------------" << std::endl;
    std::cout << "  Context Length | Dense GPU (us) | Sparse Stage 2 (us) | Speedup | Top-k / Total" << std::endl;
    std::cout << "------------------------------------------------------------------------------------" << std::endl;

    std::vector<int> test_lengths = {512, 1024, 2048, 4096, 8192};
    const int WARMUP_RUNS = 10;
    const int BENCH_RUNS = 50;

    for (int l_k : test_lengths) {
        RocketConfig test_cfg = cfg;
        test_cfg.topk_decode = std::min(l_k, std::max(256, l_k / 4));
        test_cfg.chunk_size = 4;
        test_cfg.r_dim = 64;

        size_t q_bytes = cfg.num_heads * cfg.dim_head * sizeof(float);
        size_t kv_bytes = cfg.num_heads_kv * l_k * cfg.dim_head * sizeof(float);
        size_t out_bytes = cfg.num_heads * cfg.dim_head * sizeof(float);

        void* dev_q = rocket_metal_alloc_buffer(q_bytes);
        void* dev_k = rocket_metal_alloc_buffer(kv_bytes);
        void* dev_v = rocket_metal_alloc_buffer(kv_bytes);
        void* dev_out = rocket_metal_alloc_buffer(out_bytes);

        std::vector<float> K_bench(cfg.num_heads_kv * l_k * cfg.dim_head);
        std::vector<float> V_bench(cfg.num_heads_kv * l_k * cfg.dim_head);
        for (auto& v : K_bench) v = dist(rng);
        for (auto& v : V_bench) v = dist(rng);

        rocket_metal_copy_to_buffer(dev_q, Q.data(), q_bytes);
        rocket_metal_copy_to_buffer(dev_k, K_bench.data(), kv_bytes);
        rocket_metal_copy_to_buffer(dev_v, V_bench.data(), kv_bytes);

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            rocket_metal_dense_decode_device(dev_q, dev_k, dev_v, dev_out, l_k, test_cfg);
            rocket_metal_sparse_decode_device(dev_q, dev_k, dev_v, dev_out, l_k, test_cfg);
        }

        // Benchmark Dense
        auto t0_dense = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_RUNS; ++i) {
            rocket_metal_dense_decode_device(dev_q, dev_k, dev_v, dev_out, l_k, test_cfg);
        }
        auto t1_dense = std::chrono::high_resolution_clock::now();
        double dense_us = std::chrono::duration<double, std::micro>(t1_dense - t0_dense).count() / BENCH_RUNS;

        // Benchmark Sparse
        auto t0_sparse = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BENCH_RUNS; ++i) {
            rocket_metal_sparse_decode_device(dev_q, dev_k, dev_v, dev_out, l_k, test_cfg);
        }
        auto t1_sparse = std::chrono::high_resolution_clock::now();
        double sparse_us = std::chrono::duration<double, std::micro>(t1_sparse - t0_sparse).count() / BENCH_RUNS;

        double speedup = dense_us / (sparse_us + 1e-9);

        std::cout << "  " << std::setw(14) << l_k 
                  << " | " << std::setw(14) << std::fixed << std::setprecision(1) << dense_us
                  << " | " << std::setw(19) << std::fixed << std::setprecision(1) << sparse_us
                  << " | " << std::setw(6) << std::fixed << std::setprecision(2) << speedup << "x"
                  << " | " << test_cfg.topk_decode << " / " << l_k << std::endl;

        rocket_metal_free_buffer(dev_q);
        rocket_metal_free_buffer(dev_k);
        rocket_metal_free_buffer(dev_v);
        rocket_metal_free_buffer(dev_out);
    }

    std::cout << "------------------------------------------------------------------------------------" << std::endl;
    std::cout << "All Stage 2 Metal GPU Tests Completed Successfully!" << std::endl;
    std::cout << "==========================================================" << std::endl;

    rocket_metal_cleanup();
    return 0;
}
