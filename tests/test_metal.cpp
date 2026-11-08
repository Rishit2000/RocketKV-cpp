#include "rocket_metal.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>

void check(bool value, const char *message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void compare(RocketMetal &metal, int dim, int count, int chunk, int topk, bool zero_query = false,
             bool tied = false) {
    RocketConfig cfg{};
    cfg.dim_head = dim;
    cfg.num_heads = 6;
    cfg.num_heads_kv = 2;
    cfg.chunk_size = chunk;
    cfg.r_dim = dim / 2;
    cfg.compression_ratio = 2;
    cfg.topk_decode = topk;
    const int capacity = count * 2 + 3;
    ggml_init_params params{size_t(capacity) * dim * cfg.num_heads_kv * 8 + 1024 * 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    check(bool(ctx), "GGML allocation failed");
    auto *q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, dim, 1, cfg.num_heads);
    auto *k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, dim, capacity, cfg.num_heads_kv);
    auto *v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, dim, capacity, cfg.num_heads_kv);
    std::mt19937 rng(42 + count);
    std::uniform_real_distribution<float> random(-1, 1);
    auto *qp = static_cast<float *>(q->data);
    auto *kp = static_cast<ggml_fp16_t *>(k->data);
    auto *vp = static_cast<ggml_fp16_t *>(v->data);
    for (int i = 0; i < dim * cfg.num_heads; ++i) {
        qp[i] = zero_query ? 0 : random(rng);
    }
    for (int i = 0; i < dim * capacity * cfg.num_heads_kv; ++i) {
        kp[i] = ggml_fp32_to_fp16(tied ? 0.5f : random(rng));
        vp[i] = ggml_fp32_to_fp16(random(rng));
    }
    std::vector<int> slots(count);
    for (int i = 0; i < count; ++i) {
        slots[i] = 2 * i + 1;
    }
    std::vector<float> kc(size_t(count) * dim * cfg.num_heads_kv), vc(kc.size());
    for (int h = 0; h < cfg.num_heads_kv; ++h) {
        for (int t = 0; t < count; ++t) {
            for (int d = 0; d < dim; ++d) {
                size_t dst = (size_t(h) * count + t) * dim + d,
                       src = (size_t(h) * capacity + slots[t]) * dim + d;
                kc[dst] = ggml_fp16_to_fp32(kp[src]);
                vc[dst] = ggml_fp16_to_fp32(vp[src]);
            }
        }
    }
    std::vector<std::vector<int>> cpu_selection, gpu_selection;
    auto cpu = compute_hybrid_sparse_decode(qp, kc.data(), vc.data(), count, cfg, &cpu_selection);
    std::vector<float> gpu(cpu.size());
    metal.compute(q, k, v, slots, cfg, gpu.data(), &gpu_selection);
    check(cpu_selection == gpu_selection, "CPU/Metal selected different tokens");
    float max_error = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        check(std::isfinite(gpu[i]), "Non-finite output");
        max_error = std::max(max_error, std::abs(cpu[i] - gpu[i]));
    }
    check(max_error < 2e-4f, "CPU/Metal attention mismatch");
    std::cout << "dim=" << dim << " candidates=" << count << " topk=" << topk << " zero=" << zero_query
              << " ties=" << tied << " max_error=" << max_error << "\n";
    // Validate both shader-side finite checks and host-side index validation.
    qp[0] = std::numeric_limits<float>::quiet_NaN();
    bool failed = false;
    try {
        metal.compute(q, k, v, slots, cfg, gpu.data());
    } catch (const std::runtime_error &) {
        failed = true;
    }
    check(failed, "NaN input was accepted");
    qp[0] = 0;
    slots[0] = capacity;
    failed = false;
    try {
        metal.compute(q, k, v, slots, cfg, gpu.data());
    } catch (const std::invalid_argument &) {
        failed = true;
    }
    check(failed, "Invalid slot was accepted");
}
int main() {
    try {
        RocketMetal metal;
        compare(metal, 16, 1, 1, 1);
        compare(metal, 16, 5, 3, 3);
        compare(metal, 64, 257, 3, 17);
        compare(metal, 128, 513, 2, 256);
        compare(metal, 128, 4096, 3, 512);
        compare(metal, 128, 513, 3, 17, true);
        compare(metal, 64, 257, 3, 17, false, true);
        compare(metal, 64, 257, 3, 257);
        compare(metal, 96, 129, 2, 23);
        compare(metal, 256, 17, 2, 5);
        compare(metal, 16, 32768, 3, 17, true);
        std::cout << "Metal kernel tests passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
