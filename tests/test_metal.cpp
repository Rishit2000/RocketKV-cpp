#include "rocket_metal.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>

void check(bool value, const char *message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

void compare(RocketMetal &metal, int dim, int count, int chunk, int topk, bool zero_query = false,
             bool tied = false, float key_scale = 1.0f) {
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
        qp[i] = zero_query ? 0 : random(rng) / key_scale;
    }
    for (int i = 0; i < dim * capacity * cfg.num_heads_kv; ++i) {
        kp[i] = ggml_fp32_to_fp16((tied ? 0.5f : random(rng)) * key_scale);
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

void compare_cache(RocketMetal &metal, ggml_backend_t backend = nullptr) {
    metal.reset();
    RocketConfig cfg{};
    cfg.dim_head = 64;
    cfg.num_heads = 6;
    cfg.num_heads_kv = 2;
    cfg.chunk_size = 3;
    cfg.r_dim = 32;
    cfg.compression_ratio = 2;
    cfg.topk_decode = 3;
    const int capacity = 80;
    ggml_init_params params{1024 * 1024, nullptr, backend != nullptr};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    check(bool(ctx), "GGML allocation failed");
    auto *q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cfg.dim_head, 1, cfg.num_heads);
    std::array<ggml_tensor *, 2> keys, values;
    std::array<ggml_tensor *, 4> storage;
    std::mt19937 rng(87);
    std::uniform_real_distribution<float> random(-1, 1);
    for (int layer = 0; layer < 2; ++layer) {
        // Exercise interleaved K heads and transposed V dimensions.
        auto *k = storage[2 * layer] = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, cfg.dim_head, cfg.num_heads_kv, capacity + 2);
        auto *v = storage[2 * layer + 1] = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, capacity + 2, cfg.dim_head, cfg.num_heads_kv);
        k = ggml_view_3d(ctx.get(), k, cfg.dim_head, cfg.num_heads_kv, capacity, k->nb[1], k->nb[2], k->nb[2]);
        v = ggml_view_3d(ctx.get(), v, capacity, cfg.dim_head, cfg.num_heads_kv, v->nb[1], v->nb[2], sizeof(ggml_fp16_t));
        keys[layer] = ggml_permute(ctx.get(), k, 0, 2, 1, 3);
        values[layer] = ggml_permute(ctx.get(), v, 1, 0, 2, 3);
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        backend ? ggml_backend_alloc_ctx_tensors(ctx.get(), backend) : nullptr, ggml_backend_buffer_free);
    check(!backend || bool(buffer), "Metal allocation failed");
    check(!metal.can_bind(nullptr), "Null tensor was accepted for direct binding");
    for (auto *tensor : storage) {
        auto *data = static_cast<ggml_fp16_t *>(tensor->data);
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) data[i] = ggml_fp32_to_fp16(random(rng));
    }
    check(metal.can_bind(keys[0]) == bool(backend) && metal.can_bind(values[0]) == bool(backend),
          "Unexpected direct binding support");
    auto address = [](ggml_tensor *t, int d, int slot, int h) {
        return static_cast<char *>(t->data) + d * t->nb[0] + slot * t->nb[1] + h * t->nb[2];
    };
    std::vector<int> slots;
    for (int count : {4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 33, 7, 8, 9, 10, 11, 12, 13}) {
        slots.resize(count);
        for (int i = 0; i < count; ++i) slots[i] = 2 * i + 1;
        if (count == 9) std::swap(slots[0], slots[1]);
        if (count == 11) {
            metal.reset();
            // Reuse the same slots with different history after a new request.
            const auto changed = ggml_fp32_to_fp16(5.0f);
            std::memcpy(address(keys[0], 0, slots[0], 0), &changed, sizeof(changed));
        }
        if (count == 12) cfg.chunk_size = 2;
        if (count == 13) cfg.chunk_size = 1;
        cfg.topk_decode = count == 18 ? count : 3;
        auto *qp = static_cast<float *>(q->data);
        for (int i = 0; i < cfg.dim_head * cfg.num_heads; ++i) qp[i] = random(rng);
        for (int layer = 0; layer < 2; ++layer) {
            std::vector<float> k(size_t(count) * cfg.dim_head * cfg.num_heads_kv), v(k.size());
            for (int h = 0; h < cfg.num_heads_kv; ++h) {
                for (int t = 0; t < count; ++t) {
                    for (int d = 0; d < cfg.dim_head; ++d) {
                        ggml_fp16_t a, b;
                        std::memcpy(&a, address(keys[layer], d, slots[t], h), sizeof(a));
                        std::memcpy(&b, address(values[layer], d, slots[t], h), sizeof(b));
                        const size_t i = (size_t(h) * count + t) * cfg.dim_head + d;
                        k[i] = ggml_fp16_to_fp32(a);
                        v[i] = ggml_fp16_to_fp32(b);
                    }
                }
            }
            std::vector<std::vector<int>> expected, selected, uncached_selected;
            const auto cpu = compute_hybrid_sparse_decode(qp, k.data(), v.data(), count, cfg, &expected);
            std::vector<float> gpu(cpu.size()), uncached(cpu.size());
            metal.compute(q, keys[layer], values[layer], slots, cfg, gpu.data(), &selected, layer);
            metal.compute(q, keys[layer], values[layer], slots, cfg, uncached.data(), &uncached_selected);
            check(selected == expected && selected == uncached_selected, "Cached selection mismatch");
            for (size_t i = 0; i < gpu.size(); ++i) {
                check(std::isfinite(gpu[i]) && std::abs(gpu[i] - cpu[i]) < 2e-4f, "Cached attention mismatch");
            }
            check(gpu == uncached, "Cached/uncached Metal outputs differ");
            if (backend) {
                // The same host-readable V must also work through the copy fallback.
                auto host_v = *values[layer];
                host_v.buffer = nullptr;
                host_v.view_src = nullptr;
                check(!metal.can_bind(&host_v), "Host tensor was accepted for direct binding");
                metal.compute(q, keys[layer], &host_v, slots, cfg, uncached.data());
                check(gpu == uncached, "Direct/mixed binding outputs differ");
            }
        }
    }
    // Invalid new KV values must fail even when the historical prefix is cached.
    const auto bad = ggml_fp32_to_fp16(std::numeric_limits<float>::quiet_NaN());
    const auto good = ggml_fp32_to_fp16(0.25f);
    std::vector<float> output(cfg.num_heads * cfg.dim_head);
    cfg.chunk_size = 2;
    metal.compute(q, keys[0], values[0], slots, cfg, output.data(), nullptr, 0);
    for (auto *tensor : {keys[0], values[0]}) {
        slots.push_back(2 * int(slots.size()) + 1);
        std::memcpy(address(tensor, 0, slots.back(), 0), &bad, sizeof(bad));
        bool failed = false;
        try {
            metal.compute(q, keys[0], values[0], slots, cfg, output.data(), nullptr, 0);
        } catch (const std::runtime_error &) {
            failed = true;
        }
        check(failed, "Non-finite appended KV was accepted");
        std::memcpy(address(tensor, 0, slots.back(), 0), &good, sizeof(good));
        metal.compute(q, keys[0], values[0], slots, cfg, output.data(), nullptr, 0);
        std::vector<float> uncached(output.size());
        metal.compute(q, keys[0], values[0], slots, cfg, uncached.data());
        check(output == uncached, "Cache did not recover after invalid KV");
    }
    metal.reset();
    std::cout << "Incremental cache, strided KV, layer isolation and reset tests passed (direct=" << bool(backend) << ")\n";
}

void compare_native(RocketMetal &metal, ggml_backend_t backend) {
    metal.reset();
    RocketConfig cfg{};
    cfg.dim_head = 64;
    cfg.num_heads = 6;
    cfg.num_heads_kv = 2;
    cfg.chunk_size = 3;
    cfg.r_dim = 32;
    cfg.topk_decode = 17;
    std::vector<int> slots(257);
    std::iota(slots.begin(), slots.end(), 0);
    ggml_init_params params{1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    auto *q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cfg.dim_head, 1, cfg.num_heads);
    auto *k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, cfg.dim_head, slots.size(), cfg.num_heads_kv);
    auto *v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, cfg.dim_head, slots.size(), cfg.num_heads_kv);
    auto *mask = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, slots.size());
    std::mutex mutex;
    std::string error;
    struct Call {
        RocketMetal &metal;
        RocketConfig &cfg;
        std::vector<int> &slots;
        std::mutex &mutex;
        std::string &error;
        int layer;
    } calls[] = {{metal, cfg, slots, mutex, error, 0}, {metal, cfg, slots, mutex, error, 1}};
    auto encode = [](ggml_tensor *dst, void *encoder, void *userdata) {
        auto &call = *static_cast<Call *>(userdata);
        std::lock_guard<std::mutex> lock(call.mutex);
        try { call.metal.encode(dst, call.slots, call.cfg, encoder, call.layer); }
        catch (const std::exception &e) { call.error = e.what(); }
    };
    ggml_tensor *out[2];
    for (int layer = 0; layer < 2; ++layer) {
        auto *query = layer ? ggml_reshape_3d(ctx.get(), out[0], cfg.dim_head, 1, cfg.num_heads) : q;
        ggml_tensor *args[] = {query, k, v, mask};
        out[layer] = ggml_custom_4d(ctx.get(), GGML_TYPE_F32, cfg.dim_head, cfg.num_heads, 1, 1, args, 4,
                                   [](ggml_tensor *, int, int, void *) {}, 1, nullptr);
    }
    auto *graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out[1]);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    check(bool(buffer), "Native test allocation failed");
    for (int layer = 0; layer < 2; ++layer) {
        check(metal.attach(out[layer], encode, &calls[layer]), "Native Metal hook unavailable");
        check(ggml_backend_supports_op(backend, out[layer]), "Native Metal op unsupported");
    }
    std::mt19937 rng(321);
    std::uniform_real_distribution<float> random(-1, 1);
    for (int64_t i = 0; i < ggml_nelements(q); ++i) static_cast<float *>(q->data)[i] = random(rng);
    for (auto *tensor : {k, v}) {
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            static_cast<ggml_fp16_t *>(tensor->data)[i] = ggml_fp32_to_fp16(random(rng));
        }
    }
    std::memset(mask->data, 0, ggml_nbytes(mask));
    for (int step = 0; step < 3; ++step) {
        static_cast<float *>(q->data)[0] = random(rng);
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Native Metal graph failed");
        check(error.empty(), error.c_str());
        metal.check_errors();
        for (int layer = 0; layer < 2; ++layer) {
            std::vector<float> expected(ggml_nelements(out[layer])), actual(expected.size());
            metal.compute(out[layer]->src[0], k, v, slots, cfg, expected.data());
            ggml_backend_tensor_get(out[layer], actual.data(), 0, ggml_nbytes(out[layer]));
            check(expected == actual, "Native/synchronous Metal attention differs");
        }
    }
    static_cast<float *>(mask->data)[slots.back()] = -INFINITY;
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Masked graph failed");
    bool failed = false;
    try { metal.check_errors(); } catch (const std::runtime_error &) { failed = true; }
    check(failed, "Native Metal accepted a masked candidate");
    metal.reset();
    std::cout << "Native Metal graph dependencies, reuse and mask validation passed\n";
}

int main() {
    try {
        ggml_backend_load_all();
        auto reg = ggml_backend_reg_by_name("MTL");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            reg ? ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr) : nullptr, ggml_backend_free);
        check(!reg || bool(backend), "Metal backend initialization failed");
        RocketMetal metal;
        compare(metal, 16, 1, 1, 1);
        compare(metal, 16, 2, 1, 1);
        compare(metal, 16, 5, 3, 3);
        compare(metal, 64, 257, 3, 17);
        compare(metal, 128, 513, 2, 256);
        compare(metal, 128, 4096, 3, 512);
        compare(metal, 64, 4097, 3, 17);
        compare(metal, 128, 513, 3, 17, true);
        compare(metal, 64, 257, 3, 17, false, true);
        compare(metal, 64, 257, 3, 257);
        compare(metal, 96, 129, 2, 23);
        compare(metal, 256, 17, 2, 5);
        compare(metal, 16, 32768, 3, 17, true);
        compare(metal, 64, 257, 3, 17, false, false, 1e-6f);
        compare_cache(metal);
        if (backend) compare_cache(metal, backend.get());
        if (backend) compare_native(metal, backend.get());
        std::cout << "Metal kernel tests passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
