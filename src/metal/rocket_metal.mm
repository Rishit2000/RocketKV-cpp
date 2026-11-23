#include "rocket_metal.h"
#include "ggml-metal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "rocket_metal_source.h"

namespace {
// Buffer indices below are offset by one in the shader (buffer 0 holds Params).
enum Buffer {
    RawQ,
    RawK,
    RawV,
    Slots,
    Query,
    Bounds,
    ChunkScores,
    TokenScores,
    Selected,
    Partials,
    Output,
    Error,
    BufferCount
};
enum Pipeline { Score, Attend,
                PrefillPack, PrefillDot, PrefillSoftmax, PrefillReduce, PrefillPool, PrefillCapture, PipelineCount };
enum PrefillBuffer { PrefillQ, PrefillK, PrefillPackedK, PrefillLogits, PrefillGroups,
                     PrefillScores, PrefillError, PrefillRaw, PrefillBufferCount };
struct PrefillParams {
    uint32_t dim, heads, kv_heads, len_k, obs, count, pad;
    float scale;
};
struct CaptureParams {
    uint32_t heads, tokens, offset, total, sd, sh, st, type, is_query;
};
struct Params {
    uint32_t dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded;
    uint32_t qd, qh, kd, kt, kh, vd, vt, vh;
    uint32_t cached_count, mask_type, mask_stride, bounds_stride;
};
uint32_t power_of_two(uint32_t n) {
    uint32_t p = 1;
    while (p < n) {
        p *= 2;
    }
    return p;
}
uint32_t stride(size_t n) {
    if (n > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("Metal HSA tensor stride exceeds 32 bits");
    }
    return uint32_t(n);
}
std::runtime_error metal_error(const char *message, NSError *error) {
    return std::runtime_error(std::string(message) +
                              (error ? ": " + std::string(error.localizedDescription.UTF8String) : ""));
}
} // namespace

struct RocketMetal::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::array<id<MTLComputePipelineState>, PipelineCount> pipelines;
    struct Cache {
        id<MTLBuffer> bounds = nil;
        std::array<id<MTLBuffer>, BufferCount> buffers{};
        std::vector<int> slots;
        uint32_t dim = 0, kv_heads = 0, chunk = 0, bounds_stride = 0;
    };
    std::vector<Cache> caches;
    Cache scratch;
    std::array<id<MTLBuffer>, PrefillBufferCount> prefill_buffers{};
    PrefillParams prefill_params{};
    RocketConfig prefill_config{};
    RocketCaptureStats capture_stats{};
    int prefill_len_q = 0, captured_q = 0, captured_k = 0;
    uint32_t score_threads = 0;
    bool capturing = false;
    ggml_backend_metal_get_tensor_buffer_t get_tensor_buffer = nullptr;
    ggml_backend_metal_set_custom_t set_custom = nullptr;
    Impl() {
        auto reg = ggml_backend_reg_by_name("MTL");
        if (reg) {
            set_custom = reinterpret_cast<ggml_backend_metal_set_custom_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_set_custom"));
            get_tensor_buffer = reinterpret_cast<ggml_backend_metal_get_tensor_buffer_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_get_tensor_buffer"));
        }
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            throw std::runtime_error("Metal HSA requires an available Metal device");
        }
        queue = [device newCommandQueue];
        MTLCompileOptions *options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError *error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithSource:[NSString stringWithUTF8String:rocket_metal_source]
                                 options:options
                                   error:&error];
        if (!library || !queue) {
            throw metal_error("Cannot initialize Metal HSA", error);
        }
        NSArray<NSString *> *names = @[
            @"score", @"attend", @"prefill_pack", @"prefill_dot", @"prefill_softmax",
            @"prefill_reduce", @"prefill_pool", @"prefill_capture"
        ];
        for (size_t i = 0; i < pipelines.size(); ++i) {
            pipelines[i] = [device newComputePipelineStateWithFunction:[library newFunctionWithName:names[i]]
                                                                 error:&error];
            if (!pipelines[i] || pipelines[i].maxTotalThreadsPerThreadgroup < (i == Attend ? 128 : 256) ||
                (i <= Attend && pipelines[i].threadExecutionWidth != 32)) {
                throw metal_error("Metal HSA pipeline does not support its threadgroup shape", error);
            }
        }
        score_threads = uint32_t(std::min<NSUInteger>(1024, pipelines[Score].maxTotalThreadsPerThreadgroup) / 128 * 128);
    }
    template <size_t N>
    void reserve(std::array<id<MTLBuffer>, N> &buffers, size_t index, size_t bytes, const void *source = nullptr) {
        if (!buffers[index] || buffers[index].length < bytes) {
            const size_t capacity = std::max({bytes, size_t(4096), size_t(buffers[index].length) * 2});
            buffers[index] = [device newBufferWithLength:capacity options:MTLResourceStorageModeShared];
            if (!buffers[index]) {
                throw std::runtime_error("Metal HSA buffer allocation failed");
            }
        }
        if (source) {
            std::memcpy(buffers[index].contents, source, bytes);
        }
    }
    bool tensor_buffer(const ggml_tensor *tensor, id<MTLBuffer> __strong &buffer, size_t &offset) const {
        void *handle = nullptr;
        if (!get_tensor_buffer || !get_tensor_buffer(tensor, &handle, &offset)) return false;
        buffer = (__bridge id<MTLBuffer>)handle;
        return [buffer.device isEqual:device] && offset <= buffer.length &&
               ggml_nbytes(tensor) <= buffer.length - offset;
    }
    void prepare_prefill(int len_q, int len_k, const RocketConfig &cfg, bool host_keys) {
        capturing = false;
        validate_prefill_config(len_q, len_k, cfg);
        if (len_k <= cfg.prompt_budget) throw std::invalid_argument("Resident prefill requires eviction");
        capture_stats = {};
        captured_q = captured_k = 0;
        prefill_len_q = len_q;
        prefill_config = cfg;
        auto &p = prefill_params;
        p.dim = cfg.dim_head;
        p.heads = cfg.num_heads;
        p.kv_heads = cfg.num_heads_kv;
        p.len_k = len_k;
        p.obs = std::min(cfg.window_size, std::min(len_q, len_k));
        p.count = len_k - p.obs;
        p.pad = cfg.kernel_size / 2;
        p.scale = 1.0f / std::sqrt(float(p.dim));
        auto bytes = [&](size_t elements) {
            if (elements > std::min(size_t(UINT32_MAX), size_t(device.maxBufferLength)) / sizeof(float)) {
                throw std::invalid_argument("Metal Stage 1 scratch exceeds buffer/address limits");
            }
            return elements * sizeof(float);
        };
        const std::array<size_t, PrefillBufferCount> sizes{
            bytes(size_t(p.heads) * p.obs * p.dim),
            host_keys ? bytes(size_t(p.kv_heads) * p.len_k * p.dim) : 0,
            bytes(size_t(p.kv_heads) * p.count * p.dim),
            bytes(size_t(p.heads) * p.obs * p.count),
            bytes(size_t(p.kv_heads) * p.count), bytes(p.count), sizeof(uint32_t), 0};
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (sizes[i]) reserve(prefill_buffers, i, sizes[i]);
            else prefill_buffers[i] = nil;
        }
        *static_cast<uint32_t *>(prefill_buffers[PrefillError].contents) = 0;
    }
};

RocketMetal::RocketMetal() {
    @autoreleasepool {
        impl_ = std::make_unique<Impl>();
    }
}
RocketMetal::~RocketMetal() = default;

void RocketMetal::reset() {
    impl_->caches.clear();
    impl_->scratch = {};
    impl_->prefill_buffers = {};
    impl_->capturing = false;
    impl_->capture_stats = {};
}

void RocketMetal::begin_prefill(int len_q, int len_k, const RocketConfig &cfg) {
    @autoreleasepool {
        impl_->prepare_prefill(len_q, len_k, cfg, false);
        impl_->capturing = true;
    }
}

void RocketMetal::capture_prefill(const ggml_tensor *tensor, bool is_query, int offset) {
    @autoreleasepool {
        auto &m = *impl_;
        const auto &p = m.prefill_params;
        const int total = is_query ? m.prefill_len_q : int(p.len_k);
        const uint32_t heads = is_query ? p.heads : p.kv_heads;
        int &captured = is_query ? m.captured_q : m.captured_k;
        if (!m.capturing || !tensor || !tensor->data || offset != captured ||
            tensor->ne[0] != p.dim || tensor->ne[1] != heads || tensor->ne[3] != 1 ||
            tensor->ne[2] <= 0 || tensor->ne[2] > total - captured ||
            (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16)) {
            throw std::invalid_argument("Invalid Metal prefill capture shape, type or offset");
        }
        const size_t span = ggml_nbytes(tensor);
        if (span > UINT32_MAX || span > m.device.maxBufferLength) {
            throw std::invalid_argument("Metal prefill capture exceeds buffer/address limits");
        }
        CaptureParams c{heads, uint32_t(tensor->ne[2]), uint32_t(offset), uint32_t(total),
                        stride(tensor->nb[0]), stride(tensor->nb[1]), stride(tensor->nb[2]),
                        tensor->type == GGML_TYPE_F32 ? 0u : (tensor->type == GGML_TYPE_F16 ? 1u : 2u),
                        uint32_t(is_query)};
        auto &b = m.prefill_buffers;
        id<MTLBuffer> source = nil;
        size_t source_offset = 0;
        const bool direct = m.tensor_buffer(tensor, source, source_offset);
        if (!direct) {
            if (!(tensor->view_src ? tensor->view_src->buffer : tensor->buffer)) {
                throw std::invalid_argument("Metal prefill capture requires a backend buffer");
            }
            m.reserve(b, PrefillRaw, span);
            source = b[PrefillRaw];
            ggml_backend_tensor_get(tensor, source.contents, 0, span);
            m.capture_stats.host_calls++;
            m.capture_stats.host_bytes += span;
        } else {
            m.capture_stats.metal_calls++;
        }
        id<MTLCommandBuffer> command = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) throw std::runtime_error("Cannot encode Metal prefill capture");
        [encoder setComputePipelineState:m.pipelines[PrefillCapture]];
        [encoder setBytes:&p length:sizeof(p) atIndex:0];
        for (size_t i = 0; i < b.size(); ++i) [encoder setBuffer:b[i] offset:0 atIndex:i + 1];
        [encoder setBuffer:source offset:source_offset atIndex:8];
        [encoder setBytes:&c length:sizeof(c) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake((size_t(c.tokens) * heads * p.dim + 255) / 256, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        [command commit];
        // The scheduler may overwrite the borrowed tensor as soon as the callback returns.
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw metal_error("Metal prefill capture command failed", command.error);
        }
        if (*static_cast<const uint32_t *>(b[PrefillError].contents)) {
            throw std::runtime_error("Non-finite Metal prefill activation");
        }
        captured += int(c.tokens);
    }
}

std::vector<int> RocketMetal::finish_prefill(std::vector<float> *importance) {
    auto &m = *impl_;
    if (!m.capturing || m.captured_q != m.prefill_len_q || m.captured_k != int(m.prefill_params.len_k)) {
        throw std::runtime_error("Incomplete Metal prefill capture");
    }
    m.capturing = false;
    return score_prefill(false, importance);
}

RocketCaptureStats RocketMetal::capture_stats() const {
    return impl_->capture_stats;
}

std::vector<int> RocketMetal::prefill(const float *q, const float *k, int len_q, int len_k,
                                     const RocketConfig &cfg, std::vector<float> *importance) {
    @autoreleasepool {
        validate_prefill_inputs(q, k, len_q, len_k, cfg);
        if (len_k <= cfg.prompt_budget) {
            return compute_prefill_eviction(q, k, len_q, len_k, cfg, importance);
        }
        auto &m = *impl_;
        m.prepare_prefill(len_q, len_k, cfg, true);
        const auto &p = m.prefill_params;
        auto &b = m.prefill_buffers;
        for (uint32_t h = 0; h < p.heads; ++h) {
            std::memcpy(static_cast<float *>(b[PrefillQ].contents) + size_t(h) * p.obs * p.dim,
                        q + (size_t(h) * len_q + len_q - p.obs) * p.dim,
                        size_t(p.obs) * p.dim * sizeof(float));
        }
        std::memcpy(b[PrefillK].contents, k, size_t(p.kv_heads) * p.len_k * p.dim * sizeof(float));
        return score_prefill(true, importance);
    }
}

std::vector<int> RocketMetal::score_prefill(bool pack_keys, std::vector<float> *importance) {
    @autoreleasepool {
        if (importance) importance->clear();
        auto &m = *impl_;
        const auto &p = m.prefill_params;
        auto &b = m.prefill_buffers;
        id<MTLCommandBuffer> command = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) throw std::runtime_error("Cannot encode Metal Stage 1");
        [encoder setBytes:&p length:sizeof(p) atIndex:0];
        for (size_t i = 0; i < b.size(); ++i) [encoder setBuffer:b[i] offset:0 atIndex:i + 1];
        auto dispatch = [&](Pipeline pipeline, size_t groups) {
            [encoder setComputePipelineState:m.pipelines[pipeline]];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        };
        if (pack_keys) dispatch(PrefillPack, (size_t(p.kv_heads) * p.count * p.dim + 255) / 256);
        dispatch(PrefillDot, (size_t(p.heads) * p.obs * p.count + 255) / 256);
        dispatch(PrefillSoftmax, size_t(p.heads) * p.obs);
        dispatch(PrefillReduce, (size_t(p.kv_heads) * p.count + 255) / 256);
        dispatch(PrefillPool, (p.count + 255) / 256);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw metal_error("Metal Stage 1 command failed", command.error);
        }
        if (*static_cast<const uint32_t *>(b[PrefillError].contents)) {
            throw std::runtime_error("Non-finite Metal Stage 1 score");
        }
        const auto *output = static_cast<const float *>(b[PrefillScores].contents);
        std::vector<float> scores(output, output + p.count);
        if (importance) *importance = scores;
        return select_prefill_tokens(scores, p.obs, m.prefill_config.prompt_budget);
    }
}

bool RocketMetal::can_bind(const ggml_tensor *tensor) const {
    id<MTLBuffer> buffer = nil;
    size_t offset = 0;
    return impl_->tensor_buffer(tensor, buffer, offset);
}

void RocketMetal::compute(const ggml_tensor *q, const ggml_tensor *k, const ggml_tensor *v,
                          const std::vector<int> &slots, const RocketConfig &cfg, float *output,
                          std::vector<std::vector<int>> *selected, int layer) {
    run(q, k, v, slots, cfg, output, selected, layer, nullptr, nullptr);
}

bool RocketMetal::attach(ggml_tensor *tensor, EncodeCallback callback, void *userdata) const {
    return impl_->set_custom && can_bind(tensor->src[1]) && can_bind(tensor->src[2]) &&
           impl_->set_custom(tensor, callback, userdata);
}

void RocketMetal::encode(ggml_tensor *dst, const std::vector<int> &slots, const RocketConfig &cfg,
                         void *encoder, int layer) {
    if (!encoder || layer < 0 || !ggml_is_contiguous(dst)) throw std::invalid_argument("Invalid Metal HSA encoder/output");
    run(dst->src[0], dst->src[1], dst->src[2], slots, cfg, nullptr, nullptr, layer, encoder, dst);
}

void RocketMetal::check_errors() const {
    for (const auto &cache : impl_->caches) {
        if (cache.buffers[Error] && *static_cast<const uint32_t *>(cache.buffers[Error].contents)) {
            throw std::runtime_error("Non-finite Metal HSA input/output or masked candidate");
        }
    }
}

void RocketMetal::run(const ggml_tensor *q, const ggml_tensor *k, const ggml_tensor *v,
                      const std::vector<int> &slots, const RocketConfig &cfg, float *output,
                      std::vector<std::vector<int>> *selected, int layer, void *native_encoder, ggml_tensor *dst) {
    @autoreleasepool {
        if (!q || !k || !v || !q->data || !k->data || !v->data || (!output && !dst) || q->type != GGML_TYPE_F32 ||
            k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || cfg.dim_head < 1 || cfg.dim_head > 256 ||
            cfg.num_heads_kv < 1 || cfg.num_heads < 1 || cfg.num_heads % cfg.num_heads_kv ||
            cfg.num_heads / cfg.num_heads_kv > 8 || slots.empty() || slots.size() > 32768 ||
            cfg.chunk_size < 1 || cfg.r_dim < 1 || cfg.topk_decode < 1 || layer < -1) {
            throw std::invalid_argument(
                "Metal HSA requires F32 Q/F16 KV, head dimension <=256, <=8 query heads per KV head and 1..32768 candidates");
        }
        if (q->ne[0] != cfg.dim_head || q->ne[1] != 1 || q->ne[2] != cfg.num_heads || q->ne[3] != 1 ||
            k->ne[0] != cfg.dim_head || k->ne[2] != cfg.num_heads_kv || k->ne[3] != 1 ||
            !ggml_are_same_shape(k, v)) {
            throw std::invalid_argument("Metal HSA tensor shape mismatch");
        }
        for (int slot : slots) {
            if (slot < 0 || slot >= k->ne[1]) {
                throw std::invalid_argument("Metal HSA slot out of bounds");
            }
        }
        Params p{};
        p.dim = cfg.dim_head;
        p.heads = cfg.num_heads;
        p.kv_heads = cfg.num_heads_kv;
        p.count = uint32_t(slots.size());
        p.chunk = cfg.chunk_size;
        p.rank = std::min(cfg.r_dim, cfg.dim_head);
        p.topk = std::min(uint32_t(cfg.topk_decode), p.count);
        p.chunks = 1 + (p.count - 1) / p.chunk;
        p.padded = power_of_two(p.count);
        p.bounds_stride = p.chunk > 1 ? power_of_two(p.chunks) : 0;
        p.qd = stride(q->nb[0]);
        p.qh = stride(q->nb[2]);
        p.kd = stride(k->nb[0]);
        p.kt = stride(k->nb[1]);
        p.kh = stride(k->nb[2]);
        p.vd = stride(v->nb[0]);
        p.vt = stride(v->nb[1]);
        p.vh = stride(v->nb[2]);
        // Shader address arithmetic uses uint offsets; reject spans that could overflow.
        if (ggml_nbytes(q) > UINT32_MAX || ggml_nbytes(k) > UINT32_MAX || ggml_nbytes(v) > UINT32_MAX) {
            throw std::invalid_argument("Metal HSA tensor exceeds 4 GiB");
        }
        auto &m = *impl_;
        if (layer >= 0 && size_t(layer) >= m.caches.size()) m.caches.resize(size_t(layer) + 1);
        auto &cache = layer >= 0 ? m.caches[layer] : m.scratch;
        const size_t bounds_bytes = p.topk < p.count && p.chunk > 1
                                        ? size_t(p.bounds_stride) * p.kv_heads * p.dim * 2 * sizeof(ggml_fp16_t)
                                        : 2 * sizeof(ggml_fp16_t);
        if (layer >= 0 && cache.dim == p.dim && cache.kv_heads == p.kv_heads && cache.chunk == p.chunk &&
            cache.bounds_stride == p.bounds_stride &&
            cache.bounds.length >= bounds_bytes && cache.slots.size() < slots.size() &&
            std::equal(cache.slots.begin(), cache.slots.end(), slots.begin())) {
            p.cached_count = uint32_t(cache.slots.size());
        }
        auto &buffers = native_encoder ? cache.buffers : m.scratch.buffers;
        buffers[Bounds] = cache.bounds;
        // A failed dispatch must not leave a reusable prefix.
        cache.slots.clear();
        if (!native_encoder) m.reserve(buffers, RawQ, ggml_nbytes(q), q->data);
        std::array<id<MTLBuffer>, 2> kv_buffers;
        std::array<size_t, 2> kv_offsets{};
        const ggml_tensor *kv[] = {k, v};
        for (size_t i = 0; i < 2; ++i) {
            if (!m.tensor_buffer(kv[i], kv_buffers[i], kv_offsets[i])) {
                if (native_encoder) throw std::runtime_error("Metal HSA requires native KV buffers");
                m.reserve(buffers, RawK + i, ggml_nbytes(kv[i]), kv[i]->data);
                kv_buffers[i] = buffers[RawK + i];
                kv_offsets[i] = 0;
            }
        }
        m.reserve(buffers, Slots, slots.size() * sizeof(int), slots.data());
        const size_t qbytes = size_t(p.heads) * p.dim * sizeof(float);
        if (qbytes > UINT32_MAX || bounds_bytes > UINT32_MAX) {
            throw std::invalid_argument("Metal HSA packed tensor exceeds 4 GiB");
        }
        m.reserve(buffers, Query, qbytes);
        m.reserve(buffers, Bounds, bounds_bytes);
        cache.bounds = buffers[Bounds];
        m.reserve(buffers, ChunkScores, size_t(p.heads) * p.chunks * sizeof(float));
        m.reserve(buffers, TokenScores, size_t(p.kv_heads) * p.padded * sizeof(float));
        m.reserve(buffers, Selected, size_t(p.kv_heads) * p.padded * sizeof(uint32_t));
        // Split-K slices of the exact attention: about 8 selected tokens per simdgroup,
        // 4 simdgroups per threadgroup, at most 64 slices so one threadgroup can merge them.
        const uint32_t units = (std::min(64u, (p.topk + 7) / 8) + 3) / 4 * 4;
        m.reserve(buffers, Partials, size_t(p.heads) * (units / 4) * (p.dim + 2) * sizeof(float));
        if (!native_encoder) m.reserve(buffers, Output, qbytes);
        // Error word, then score and attend tickets per KV head.
        const std::vector<uint32_t> zeros(1 + 2 * p.kv_heads, 0);
        m.reserve(buffers, Error, zeros.size() * sizeof(uint32_t), zeros.data());
        id<MTLCommandBuffer> command = native_encoder ? nil : [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = native_encoder ? (__bridge id<MTLComputeCommandEncoder>)native_encoder
                                                            : [command computeCommandEncoder];
        if (!encoder) {
            throw std::runtime_error("Cannot create Metal HSA command buffer");
        }
        for (size_t i = 0; i < buffers.size(); ++i) {
            if (i == RawK || i == RawV || (native_encoder && (i == RawQ || i == Output))) continue;
            [encoder setBuffer:buffers[i] offset:0 atIndex:i + 1];
        }
        for (size_t i = 0; i < 2; ++i) {
            [encoder setBuffer:kv_buffers[i] offset:kv_offsets[i] atIndex:RawK + i + 1];
        }
        if (native_encoder) {
            const ggml_tensor *tensors[] = {q, dst, dst->src[3]};
            const size_t bindings[] = {RawQ + 1, Output + 1, 18};
            for (size_t i = 0; i < 3; ++i) {
                id<MTLBuffer> buffer = nil;
                size_t offset = 0;
                if (!m.tensor_buffer(tensors[i], buffer, offset)) throw std::runtime_error("Cannot bind Metal HSA tensor");
                [encoder setBuffer:buffer offset:offset atIndex:bindings[i]];
            }
            const auto *mask = dst->src[3];
            if (mask->ne[0] != k->ne[1] || (mask->type != GGML_TYPE_F32 && mask->type != GGML_TYPE_F16) ||
                ggml_nbytes(mask) > UINT32_MAX) throw std::invalid_argument("Unsupported Metal HSA mask");
            p.mask_type = mask->type == GGML_TYPE_F32 ? 1 : 2;
            p.mask_stride = stride(mask->nb[0]);
        } else {
            [encoder setBuffer:buffers[RawQ] offset:0 atIndex:18];
        }
        [encoder setBytes:&p length:sizeof(p) atIndex:0];
        auto dispatch = [&](Pipeline kernel, uint32_t blocks, uint32_t threads) {
            [encoder setComputePipelineState:m.pipelines[kernel]];
            [encoder dispatchThreadgroups:MTLSizeMake(blocks, p.kv_heads, 1)
                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
        // Blocks of score_threads / 4 chunks per KV head (one block when every candidate is selected).
        const uint32_t per_block = m.score_threads / 4;
        dispatch(Score, p.topk < p.count ? (p.chunks + per_block - 1) / per_block : 1, m.score_threads);
        [encoder setBytes:&units length:sizeof(units) atIndex:16];
        dispatch(Attend, units / 4, 128);
        if (!native_encoder) {
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted) {
                throw metal_error("Metal HSA command failed", command.error);
            }
            if (*static_cast<uint32_t *>(buffers[Error].contents)) {
                throw std::runtime_error("Non-finite Metal HSA input or output");
            }
        }
        if (layer >= 0 && p.topk < p.count) {
            cache.slots = slots;
            cache.dim = p.dim;
            cache.kv_heads = p.kv_heads;
            cache.chunk = p.chunk;
            cache.bounds_stride = p.bounds_stride;
        }
        if (output) std::memcpy(output, buffers[Output].contents, qbytes);
        if (selected) {
            selected->resize(p.kv_heads);
            const auto *indices = static_cast<const uint32_t *>(buffers[Selected].contents);
            for (uint32_t h = 0; h < p.kv_heads; ++h) {
                (*selected)[h].assign(indices + h * p.padded, indices + h * p.padded + p.topk);
            }
        }
    }
}
