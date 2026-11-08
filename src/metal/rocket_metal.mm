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
    Dimensions,
    Signs,
    Scales,
    ChunkScores,
    TokenScores,
    Selected,
    ExactScores,
    Output,
    Error,
    BufferCount
};
enum Pipeline { Pack, Summarize, Prepare, Approximate, Softmax, Select, SelectLocal, Exact, WeightedSum, PipelineCount };
struct Params {
    uint32_t dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded, dim_padded, topk_padded;
    uint32_t qd, qh, kd, kt, kh, vd, vt, vh;
    uint32_t cached_count, mask_type, mask_stride;
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
        uint32_t dim = 0, kv_heads = 0, chunk = 0;
    };
    std::vector<Cache> caches;
    Cache scratch;
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
            @"pack", @"summarize", @"prepare", @"approximate", @"softmax", @"select_tokens", @"select_tokens_local", @"exact_scores",
            @"weighted_sum"
        ];
        for (size_t i = 0; i < pipelines.size(); ++i) {
            pipelines[i] = [device newComputePipelineStateWithFunction:[library newFunctionWithName:names[i]]
                                                                 error:&error];
            if (!pipelines[i] || pipelines[i].maxTotalThreadsPerThreadgroup < 256) {
                throw metal_error("Metal HSA requires 256 threads per threadgroup", error);
            }
        }
    }
    void reserve(std::array<id<MTLBuffer>, BufferCount> &buffers, size_t index, size_t bytes, const void *source = nullptr) {
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
            cfg.num_heads / cfg.num_heads_kv > 256 || slots.empty() || slots.size() > 32768 ||
            cfg.chunk_size < 1 || cfg.r_dim < 1 || cfg.topk_decode < 1 || layer < -1) {
            throw std::invalid_argument(
                "Metal HSA requires F32 Q/F16 KV, head dimension <=256 and 1..32768 candidates");
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
        p.dim_padded = power_of_two(p.dim);
        p.topk_padded = power_of_two(p.topk);
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
                                        ? size_t(p.chunks) * p.kv_heads * p.dim * 2 * sizeof(ggml_fp16_t)
                                        : 2 * sizeof(ggml_fp16_t);
        if (layer >= 0 && cache.dim == p.dim && cache.kv_heads == p.kv_heads && cache.chunk == p.chunk &&
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
        m.reserve(buffers, Dimensions, size_t(p.kv_heads) * p.dim * sizeof(uint32_t));
        m.reserve(buffers, Signs, size_t(p.kv_heads) * p.dim * sizeof(float));
        m.reserve(buffers, Scales, p.heads * sizeof(float));
        m.reserve(buffers, ChunkScores, size_t(p.heads) * p.chunks * sizeof(float));
        m.reserve(buffers, TokenScores, size_t(p.kv_heads) * p.padded * sizeof(float));
        m.reserve(buffers, Selected, size_t(p.kv_heads) * p.padded * sizeof(uint32_t));
        m.reserve(buffers, ExactScores, size_t(p.heads) * p.topk * sizeof(float));
        if (!native_encoder) m.reserve(buffers, Output, qbytes);
        uint32_t zero = 0;
        m.reserve(buffers, Error, sizeof(zero), &zero);
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
        auto dispatch = [&](size_t kernel, uint32_t groups) {
            [encoder setComputePipelineState:m.pipelines[kernel]];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
        auto softmax = [&](uint32_t width, uint32_t exact_mode) {
            [encoder setBytes:&width length:sizeof(width) atIndex:16];
            [encoder setBytes:&exact_mode length:sizeof(exact_mode) atIndex:17];
            dispatch(Softmax, p.heads);
        };
        dispatch(Pack, (std::max({p.heads * p.dim, p.kv_heads * (p.count - p.cached_count) * p.dim, p.count}) + 255) / 256);
        if (p.topk < p.count) {
            if (p.chunk > 1) {
                const uint32_t updated_chunks = p.chunks - p.cached_count / p.chunk;
                dispatch(Summarize, (updated_chunks * p.kv_heads * p.dim + 255) / 256);
            }
            dispatch(Prepare, p.kv_heads);
            dispatch(Approximate, (p.heads * p.chunks + 255) / 256);
            softmax(p.chunks, 0);
        }
        const size_t sort_bytes = std::max(size_t(16), size_t(p.padded) * sizeof(uint32_t));
        if (p.topk < p.count && 2 * sort_bytes <= m.device.maxThreadgroupMemoryLength) {
            [encoder setThreadgroupMemoryLength:sort_bytes atIndex:0];
            [encoder setThreadgroupMemoryLength:sort_bytes atIndex:1];
            dispatch(SelectLocal, p.kv_heads);
            [encoder setThreadgroupMemoryLength:0 atIndex:0];
            [encoder setThreadgroupMemoryLength:0 atIndex:1];
        } else {
            dispatch(Select, p.kv_heads);
        }
        dispatch(Exact, (p.heads * p.topk + 255) / 256);
        softmax(p.topk, 1);
        dispatch(WeightedSum, (p.heads * p.dim + 255) / 256);
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
