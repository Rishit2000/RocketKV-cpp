#include "rocket_metal.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

void check(bool value, const char *message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

// Metal may reorder floating-point reductions, so it is checked against the algorithm rather than
// bit-for-bit against the CPU reference: the selection must be well-formed and nearly identical,
// and the output must match fp64 attention over the tokens Metal actually selected.
// K/V are [kv_head][candidate][dim].
void check_hsa(const float *q, const std::vector<float> &k, const std::vector<float> &v, int count,
               const RocketConfig &cfg, const std::vector<std::vector<int>> &reference,
               const std::vector<std::vector<int>> &selection, const std::vector<float> &output, bool exact) {
    const int dim = cfg.dim_head, group = cfg.num_heads / cfg.num_heads_kv;
    const size_t want = size_t(std::min(cfg.topk_decode, count));
    check(selection.size() == size_t(cfg.num_heads_kv), "Wrong selection head count");
    double recall_sum = 0, recall_min = 1;
    for (int h = 0; h < cfg.num_heads_kv; ++h) {
        const auto &picked = selection[h];
        check(picked.size() == want, "Wrong selection size");
        for (size_t i = 0; i < picked.size(); ++i) {
            check(picked[i] >= 0 && picked[i] < count && (i == 0 || picked[i - 1] < picked[i]),
                  "Selection is not sorted, unique, and in range");
        }
        std::vector<int> common;
        std::set_intersection(picked.begin(), picked.end(), reference[h].begin(), reference[h].end(),
                              std::back_inserter(common));
        const double recall = double(common.size()) / double(want);
        recall_sum += recall;
        recall_min = std::min(recall_min, recall);
    }
    if (exact) check(selection == reference, "CPU/Metal selected different tokens");
    check(recall_sum / cfg.num_heads_kv >= 0.98 && recall_min >= 0.90, "CPU/Metal selection recall too low");
    for (int head = 0; head < cfg.num_heads; ++head) {
        const int h = head / group;
        const float *qh = q + size_t(head) * dim;
        std::vector<double> scores;
        double top = -INFINITY;
        for (int t : selection[h]) {
            double dot = 0;
            for (int d = 0; d < dim; ++d) dot += double(qh[d]) * k[(size_t(h) * count + t) * dim + d];
            scores.push_back(dot / std::sqrt(double(dim)));
            top = std::max(top, scores.back());
        }
        double total = 0;
        for (auto &s : scores) total += (s = std::exp(s - top));
        for (int d = 0; d < dim; ++d) {
            double expected = 0;
            for (size_t i = 0; i < scores.size(); ++i) {
                expected += scores[i] / total * v[(size_t(h) * count + selection[h][i]) * dim + d];
            }
            const float actual = output[size_t(head) * dim + d];
            check(std::isfinite(actual) && std::abs(actual - expected) <= 1e-3 * (1 + std::abs(expected)),
                  "Metal attention differs from fp64 attention over its selection");
        }
    }
}

void compare(RocketMetal &metal, int dim, int count, int chunk, int topk, bool zero_query = false,
             bool tied = false, float key_scale = 1.0f, bool tail_best = false) {
    RocketConfig cfg{};
    cfg.dim_head = dim;
    cfg.num_heads = 6;
    cfg.num_heads_kv = 2;
    cfg.chunk_size = chunk;
    cfg.r_dim = std::max(1, dim / 2);
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
    if (tail_best) {
        std::fill_n(qp, dim * cfg.num_heads, 1.0f);
        for (int h = 0; h < cfg.num_heads_kv; ++h) {
            for (int t = (count - 1) / chunk * chunk; t < count; ++t) {
                std::fill_n(kp + (size_t(h) * capacity + slots[t]) * dim, dim, ggml_fp32_to_fp16(10.0f));
            }
        }
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
    // Degenerate inputs have exactly tied approximate scores, so the tie order must match exactly.
    check_hsa(qp, kc, vc, count, cfg, cpu_selection, gpu_selection, gpu, zero_query || tied || tail_best);
    float max_error = 0;
    for (size_t i = 0; i < cpu.size(); ++i) max_error = std::max(max_error, std::abs(cpu[i] - gpu[i]));
    std::cout << "dim=" << dim << " candidates=" << count << " topk=" << topk << " zero=" << zero_query
              << " ties=" << tied << " max_error_vs_cpu=" << max_error << "\n";
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
            check(selected == uncached_selected && gpu == uncached, "Cached/uncached Metal outputs differ");
            check_hsa(qp, k, v, count, cfg, expected, selected, gpu, false);
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

void compare_prefill(RocketMetal &metal, int dim, int prefix, int window, int kernel, bool zero = false) {
    RocketConfig cfg{};
    cfg.num_heads = 6;
    cfg.num_heads_kv = 2;
    cfg.dim_head = dim;
    cfg.window_size = window;
    cfg.kernel_size = kernel;
    cfg.prompt_budget = window + prefix / 2;
    const int len_k = prefix + window, len_q = window + 2;
    std::mt19937 rng(173 + prefix + dim);
    std::uniform_real_distribution<float> random(-1, 1);
    std::vector<float> q(size_t(cfg.num_heads) * len_q * dim);
    std::vector<float> k(size_t(cfg.num_heads_kv) * len_k * dim);
    for (auto &x : q) x = zero ? 0.0f : random(rng);
    for (auto &x : k) x = random(rng);
    std::vector<float> cpu_scores, gpu_scores;
    const auto cpu = compute_prefill_eviction(q.data(), k.data(), len_q, len_k, cfg, &cpu_scores);
    const auto gpu = metal.prefill(q.data(), k.data(), len_q, len_k, cfg, &gpu_scores);
    check(cpu_scores.size() == gpu_scores.size(), "Stage 1 score shape mismatch");
    for (size_t i = 0; i < cpu_scores.size(); ++i) {
        check(std::isfinite(gpu_scores[i]) &&
              std::abs(cpu_scores[i] - gpu_scores[i]) <= 1e-5f + 1e-4f * std::abs(cpu_scores[i]),
              "Stage 1 CPU/Metal importance mismatch");
    }
    check(cpu == gpu, "Stage 1 CPU/Metal survivors differ");
    check(gpu.size() == size_t(cfg.prompt_budget), "Stage 1 budget mismatch");
    for (int i = 0; i < window; ++i) {
        check(gpu[gpu.size() - window + i] == prefix + i, "Stage 1 lost observation token");
    }
    // Reuse the same buffers after changing the query and budget.
    std::fill(q.begin(), q.end(), 0.0f);
    cfg.prompt_budget = window;
    check(metal.prefill(q.data(), k.data(), len_q, len_k, cfg) ==
          compute_prefill_eviction(q.data(), k.data(), len_q, len_k, cfg),
          "Stage 1 observation-only selection mismatch");
    cfg.prompt_budget = len_k + 1;
    check(metal.prefill(nullptr, nullptr, len_q, len_k, cfg, &gpu_scores) ==
          compute_prefill_eviction(nullptr, nullptr, len_q, len_k, cfg),
          "Stage 1 no-eviction path mismatch");
    check(gpu_scores.empty(), "Stage 1 skipped request left stale scores");
}

void test_prefill_errors(RocketMetal &metal) {
    RocketConfig cfg{};
    cfg.num_heads = 2;
    cfg.num_heads_kv = 1;
    cfg.dim_head = 3;
    cfg.window_size = 2;
    cfg.kernel_size = 3;
    cfg.prompt_budget = 3;
    std::vector<float> q(30, 0.1f), k(15, 0.2f);
    auto invalid = [&](const float *qp, const float *kp, int nq, int nk) {
        bool failed = false;
        try {
            metal.prefill(qp, kp, nq, nk, cfg);
        } catch (const std::invalid_argument &) {
            failed = true;
        }
        check(failed, "Invalid Stage 1 input accepted");
    };
    invalid(nullptr, k.data(), 5, 5);
    invalid(q.data(), nullptr, 5, 5);
    invalid(q.data(), k.data(), 0, 5);
    cfg.num_heads_kv = 3;
    invalid(q.data(), k.data(), 5, 5);
    cfg.num_heads_kv = 1;
    cfg.prompt_budget = 1;
    invalid(q.data(), k.data(), 5, 5);
    cfg.prompt_budget = 3;
    q[0] = std::numeric_limits<float>::quiet_NaN();
    invalid(q.data(), k.data(), 5, 5);
    q[0] = 0.1f;
    k.back() = std::numeric_limits<float>::infinity();
    invalid(q.data(), k.data(), 5, 5);
    std::fill(q.begin(), q.end(), 1e30f);
    std::fill(k.begin(), k.end(), 1e30f);
    bool failed = false;
    try {
        metal.prefill(q.data(), k.data(), 5, 5, cfg);
    } catch (const std::runtime_error &) {
        failed = true;
    }
    check(failed, "Overflowing Stage 1 dot product accepted");
    std::fill(q.begin(), q.end(), 0.1f);
    std::fill(k.begin(), k.end(), 0.2f);
    check(metal.prefill(q.data(), k.data(), 5, 5, cfg).size() == 3,
          "Stage 1 failed to recover after an error");
}

void compare_prefill_capture(RocketMetal &metal, ggml_backend_t backend, ggml_type type, bool direct) {
    RocketConfig cfg{};
    cfg.num_heads = 6;
    cfg.num_heads_kv = 2;
    cfg.dim_head = 7;
    cfg.window_size = 9;
    cfg.kernel_size = 3;
    cfg.prompt_budget = 19;
    const int nq = 23, nk = 37, batch = 8;
    ggml_init_params params{1024 * 1024, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    check(bool(ctx), "Capture tensor allocation failed");
    auto *qs = ggml_new_tensor_3d(ctx.get(), type, cfg.dim_head * 2, cfg.num_heads, batch + 2);
    auto *ks = ggml_new_tensor_3d(ctx.get(), type, cfg.dim_head * 2, cfg.num_heads_kv, batch + 2);
    const size_t element = ggml_type_size(type);
    auto make_view = [&](ggml_tensor *storage) {
        auto *view = ggml_view_3d(ctx.get(), storage, cfg.dim_head, storage->ne[1], batch,
                                  storage->nb[1], storage->nb[2], storage->nb[2] + element);
        view->nb[0] *= 2;
        return view;
    };
    auto *qv = make_view(qs), *kv = make_view(ks);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    check(bool(buffer), "Capture backend allocation failed");
    check(metal.can_bind(qv) == direct, "Unexpected capture binding route");
    std::mt19937 rng(173);
    std::uniform_int_distribution<int> random(-16, 16);
    std::vector<float> q(size_t(cfg.num_heads) * nq * cfg.dim_head);
    std::vector<float> k(size_t(cfg.num_heads_kv) * nk * cfg.dim_head);
    auto upload = [&](ggml_tensor *storage, ggml_tensor *view, const std::vector<float> &values, int total, int offset) {
        std::vector<unsigned char> raw(ggml_nbytes(storage), 0);
        view->ne[2] = std::min(batch, total - offset);
        for (int t = 0; t < view->ne[2]; ++t) {
            for (int h = 0; h < view->ne[1]; ++h) {
                for (int d = 0; d < cfg.dim_head; ++d) {
                    float x = values[(size_t(h) * total + offset + t) * cfg.dim_head + d];
                    auto *dst = raw.data() + storage->nb[2] + element +
                                t * view->nb[2] + h * view->nb[1] + d * view->nb[0];
                    if (type == GGML_TYPE_F32) std::memcpy(dst, &x, sizeof(x));
                    else if (type == GGML_TYPE_F16) {
                        const auto v = ggml_fp32_to_fp16(x);
                        std::memcpy(dst, &v, sizeof(v));
                    } else {
                        const auto v = ggml_fp32_to_bf16(x);
                        std::memcpy(dst, &v, sizeof(v));
                    }
                }
            }
        }
        ggml_backend_tensor_set(storage, raw.data(), 0, raw.size());
        ggml_backend_synchronize(backend);
    };
    // Validate even discarded Q positions, and recover by starting a new request.
    q[0] = std::numeric_limits<float>::quiet_NaN();
    upload(qs, qv, q, nq, 0);
    metal.begin_prefill(nq, nk, cfg);
    bool failed = false;
    try { metal.capture_prefill(qv, true, 0); }
    catch (const std::runtime_error &) { failed = true; }
    check(failed, "Non-finite discarded query accepted");
    for (int request = 0; request < 2; ++request) {
        for (auto &x : q) x = float(random(rng)) / 16;
        for (auto &x : k) x = float(random(rng)) / 16;
        if (request) {
            cfg.window_size = 3;
            cfg.prompt_budget = 7;
        }
        metal.begin_prefill(nq, nk, cfg);
        failed = false;
        try { metal.finish_prefill(); }
        catch (const std::runtime_error &) { failed = true; }
        check(failed, "Incomplete capture accepted");
        upload(qs, qv, q, nq, 0);
        failed = false;
        try { metal.capture_prefill(qv, true, 1); }
        catch (const std::invalid_argument &) { failed = true; }
        check(failed, "Out-of-order capture accepted");
        for (int offset = 0; offset < nk; offset += batch) {
            if (offset < nq) {
                upload(qs, qv, q, nq, offset);
                metal.capture_prefill(qv, true, offset);
                ggml_backend_tensor_memset(qs, 0, 0, ggml_nbytes(qs));
            }
            upload(ks, kv, k, nk, offset);
            metal.capture_prefill(kv, false, offset);
            ggml_backend_tensor_memset(ks, 0, 0, ggml_nbytes(ks));
        }
        std::vector<float> captured_scores, host_scores;
        const auto actual = metal.finish_prefill(&captured_scores);
        const auto stats = metal.capture_stats();
        check(stats.metal_calls == (direct ? 8 : 0) && stats.host_calls == (direct ? 0 : 8),
              "Capture route statistics incorrect");
        check(direct ? stats.host_bytes == 0 : stats.host_bytes > 0, "Capture host-byte accounting incorrect");
        failed = false;
        try { metal.finish_prefill(); }
        catch (const std::runtime_error &) { failed = true; }
        check(failed, "Completed capture was reused");
        const auto expected = metal.prefill(q.data(), k.data(), nq, nk, cfg, &host_scores);
        check(captured_scores == host_scores && actual == expected, "Resident/host capture scores differ");
        check(actual == compute_prefill_eviction(q.data(), k.data(), nq, nk, cfg),
              "Resident/CPU selection differs");
    }
    metal.reset();
    failed = false;
    try { metal.capture_prefill(qv, true, 0); }
    catch (const std::invalid_argument &) { failed = true; }
    check(failed, "Capture after reset accepted");
    std::cout << "Prefill strided capture, source reuse and reset passed (direct=" << direct
              << ", type=" << ggml_type_name(type) << ")\n";
}

// Stage 2 decode cost without a model: a chain of 28 HSA layers with Llama-3.2-3B shapes, encoded
// into GGML's Metal encoder like a decode step. One candidate is appended per step, as in decode.
void bench(RocketMetal &metal, ggml_backend_t backend) {
    constexpr int layers = 28, warmup = 20, steps = 200;
    for (int budget : {1024, 2048}) {
        for (int context : {2048, 8192, 16384, 32768}) {
            const auto params = compute_rocket_params(budget, context, 128, 128, 32);
            RocketConfig cfg{};
            cfg.dim_head = 128;
            cfg.num_heads = 24;
            cfg.num_heads_kv = 8;
            cfg.topk_decode = params.k;
            cfg.chunk_size = params.chunk_size;
            cfg.r_dim = params.r;
            const int capacity = params.prompt_budget + warmup + steps;
            std::vector<int> slots(params.prompt_budget);
            std::iota(slots.begin(), slots.end(), 0);
            ggml_init_params init{ggml_tensor_overhead() * 8 * layers + ggml_graph_overhead(), nullptr, true};
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(init), ggml_free);
            auto *q = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cfg.dim_head, 1, cfg.num_heads);
            auto *mask = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, capacity);
            std::mutex mutex;
            std::string error;
            struct Call {
                RocketMetal &metal;
                RocketConfig &cfg;
                std::vector<int> &slots;
                std::mutex &mutex;
                std::string &error;
                int layer;
            };
            std::vector<Call> calls;
            std::vector<ggml_tensor *> kv, out;
            for (int layer = 0; layer < layers; ++layer) {
                calls.push_back({metal, cfg, slots, mutex, error, layer});
                auto *k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, cfg.dim_head, capacity, cfg.num_heads_kv);
                auto *v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, cfg.dim_head, capacity, cfg.num_heads_kv);
                auto *query = layer ? ggml_reshape_3d(ctx.get(), out.back(), cfg.dim_head, 1, cfg.num_heads) : q;
                ggml_tensor *args[] = {query, k, v, mask};
                out.push_back(ggml_custom_4d(ctx.get(), GGML_TYPE_F32, cfg.dim_head, cfg.num_heads, 1, 1, args, 4,
                                             [](ggml_tensor *, int, int, void *) {}, 1, nullptr));
                kv.insert(kv.end(), {k, v});
            }
            auto *graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, out.back());
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
            check(bool(buffer), "Benchmark allocation failed");
            auto encode = [](ggml_tensor *dst, void *encoder, void *userdata) {
                auto &call = *static_cast<Call *>(userdata);
                std::lock_guard<std::mutex> lock(call.mutex);
                try { call.metal.encode(dst, call.slots, call.cfg, encoder, call.layer); }
                catch (const std::exception &e) { call.error = e.what(); }
            };
            for (int layer = 0; layer < layers; ++layer) {
                check(metal.attach(out[layer], encode, &calls[layer]), "Native Metal hook unavailable");
            }
            std::mt19937 rng(7);
            std::uniform_real_distribution<float> random(-1, 1);
            for (int64_t i = 0; i < ggml_nelements(q); ++i) static_cast<float *>(q->data)[i] = random(rng);
            for (auto *tensor : kv) {
                auto *data = static_cast<ggml_fp16_t *>(tensor->data);
                for (int64_t i = 0; i < ggml_nelements(tensor); ++i) data[i] = ggml_fp32_to_fp16(random(rng));
            }
            std::memset(mask->data, 0, ggml_nbytes(mask));
            metal.reset();
            std::vector<double> times;
            for (int step = 0; step < warmup + steps; ++step) {
                slots.push_back(int(slots.size()));
                static_cast<float *>(q->data)[step % ggml_nelements(q)] = random(rng);
                const auto start = std::chrono::steady_clock::now();
                check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Benchmark graph failed");
                const auto stop = std::chrono::steady_clock::now();
                check(error.empty(), error.c_str());
                if (step >= warmup) times.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
            }
            metal.check_errors();
            metal.reset();
            std::sort(times.begin(), times.end());
            std::cout << "budget " << budget << " context " << context << " candidates " << params.prompt_budget
                      << " k " << cfg.topk_decode << " chunk " << cfg.chunk_size << " r " << cfg.r_dim
                      << ": " << layers << " layers median " << times[times.size() / 2] << " ms, p10 "
                      << times[times.size() / 10] << ", p90 " << times[times.size() * 9 / 10] << "\n";
        }
    }
}

int main(int argc, char **argv) {
    try {
        ggml_backend_load_all();
        auto reg = ggml_backend_reg_by_name("MTL");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            reg ? ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr) : nullptr, ggml_backend_free);
        check(!reg || bool(backend), "Metal backend initialization failed");
        RocketMetal metal;
        if (argc > 1 && std::string(argv[1]) == "bench") {
            check(bool(backend), "bench requires the Metal backend");
            bench(metal, backend.get());
            return 0;
        }
        check(argc == 1, "Usage: test_metal [bench]");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_name("CPU", nullptr), ggml_backend_free);
        check(bool(cpu), "CPU backend initialization failed");
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
            compare_prefill_capture(metal, cpu.get(), type, false);
            if (backend) compare_prefill_capture(metal, backend.get(), type, true);
        }
        for (int dim : {1, 7, 128}) {
            for (int prefix : {1, 2, 3, 4, 31, 32, 33, 255, 256, 257, 513}) {
                compare_prefill(metal, dim, prefix, 3, 1);
            }
        }
        for (int count : {2048, 4096, 8192}) compare_prefill(metal, 128, count - 32, 32, 63);
        compare_prefill(metal, 7, 33, 3, 63);
        compare_prefill(metal, 7, 33, 3, 63, true);
        test_prefill_errors(metal);
        metal.reset();
        compare_prefill(metal, 7, 33, 3, 3);
        std::cout << "Stage 1 CPU/Metal scoring, selection, reuse and validation passed\n";
        compare(metal, 1, 17, 3, 7);
        compare(metal, 31, 33, 1, 17);
        compare(metal, 33, 35, 4, 10);
        compare(metal, 16, 1, 1, 1);
        compare(metal, 16, 2, 1, 1);
        compare(metal, 16, 5, 3, 3);
        for (int topk : {1, 3, 4, 6, 12}) compare(metal, 17, 13, 5, topk, false, false, 1.0f, true);
        compare(metal, 17, 13, 32, 5);
        compare(metal, 64, 4096, 1, 512);
        compare(metal, 64, 4097, 1, 512);
        compare(metal, 16, 4097, 3, 4097);
        compare(metal, 16, 1300, 3, 1299);
        compare(metal, 64, 257, 3, 17);
        compare(metal, 128, 513, 2, 256);
        compare(metal, 128, 4096, 3, 512);
        compare(metal, 64, 4097, 3, 17);
        compare(metal, 128, 513, 3, 17, true);
        compare(metal, 64, 257, 3, 17, false, true);
        compare(metal, 64, 257, 3, 257);
        // Selections whose last attention block gets no tokens (k = 600, 650, 656 split over 64 slices).
        for (int topk : {600, 650, 656}) compare(metal, 128, 1300, 1, topk);
        for (int topk : {600, 656}) compare(metal, 128, 1300, 2, topk);
        compare(metal, 96, 129, 2, 23);
        compare(metal, 256, 17, 2, 5);
        compare(metal, 16, 32768, 3, 17, true);
        // Stage 2 without eviction: the whole 32k/128k context is the candidate set.
        compare(metal, 128, 33000, 6, 512);
        compare(metal, 128, 131200, 12, 512);
        compare(metal, 16, 8193, 1, 511);
        compare(metal, 17, 8193, 1, 8192, true);
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
