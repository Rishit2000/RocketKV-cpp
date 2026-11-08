#include "rocket_metal.h"
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
    Keys,
    Values,
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
enum Pipeline { Pack, Prepare, Approximate, Softmax, Select, Exact, WeightedSum, PipelineCount };
struct Params {
    uint32_t dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded, dim_padded, topk_padded;
    uint32_t qd, qh, kd, kt, kh, vd, vt, vh;
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
    std::array<id<MTLBuffer>, BufferCount> buffers;
    Impl() {
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
            @"pack", @"prepare", @"approximate", @"softmax", @"select_tokens", @"exact_scores",
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
    void reserve(size_t index, size_t bytes, const void *source = nullptr) {
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
};

RocketMetal::RocketMetal() {
    @autoreleasepool {
        impl_ = std::make_unique<Impl>();
    }
}
RocketMetal::~RocketMetal() = default;

void RocketMetal::compute(const ggml_tensor *q, const ggml_tensor *k, const ggml_tensor *v,
                          const std::vector<int> &slots, const RocketConfig &cfg, float *output,
                          std::vector<std::vector<int>> *selected) {
    @autoreleasepool {
        if (!q || !k || !v || !q->data || !k->data || !v->data || !output || q->type != GGML_TYPE_F32 ||
            k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || cfg.dim_head < 1 || cfg.dim_head > 256 ||
            cfg.num_heads_kv < 1 || cfg.num_heads < 1 || cfg.num_heads % cfg.num_heads_kv ||
            cfg.num_heads / cfg.num_heads_kv > 256 || slots.empty() || slots.size() > 32768 ||
            cfg.chunk_size < 1 || cfg.r_dim < 1 || cfg.topk_decode < 1) {
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
        m.reserve(RawQ, ggml_nbytes(q), q->data);
        m.reserve(RawK, ggml_nbytes(k), k->data);
        m.reserve(RawV, ggml_nbytes(v), v->data);
        m.reserve(Slots, slots.size() * sizeof(int), slots.data());
        const size_t qbytes = size_t(p.heads) * p.dim * sizeof(float),
                     kvbytes = size_t(p.kv_heads) * p.count * p.dim * sizeof(float);
        if (qbytes > UINT32_MAX || kvbytes > UINT32_MAX) {
            throw std::invalid_argument("Metal HSA packed tensor exceeds 4 GiB");
        }
        m.reserve(Query, qbytes);
        m.reserve(Keys, kvbytes);
        m.reserve(Values, kvbytes);
        m.reserve(Dimensions, size_t(p.kv_heads) * p.dim * sizeof(uint32_t));
        m.reserve(Signs, size_t(p.kv_heads) * p.dim * sizeof(float));
        m.reserve(Scales, p.heads * sizeof(float));
        m.reserve(ChunkScores, size_t(p.heads) * p.chunks * sizeof(float));
        m.reserve(TokenScores, size_t(p.kv_heads) * p.padded * sizeof(float));
        m.reserve(Selected, size_t(p.kv_heads) * p.padded * sizeof(uint32_t));
        m.reserve(ExactScores, size_t(p.heads) * p.topk * sizeof(float));
        m.reserve(Output, qbytes);
        uint32_t zero = 0;
        m.reserve(Error, sizeof(zero), &zero);
        id<MTLCommandBuffer> command = [m.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            throw std::runtime_error("Cannot create Metal HSA command buffer");
        }
        [encoder setBytes:&p length:sizeof(p) atIndex:0];
        for (size_t i = 0; i < m.buffers.size(); ++i) {
            [encoder setBuffer:m.buffers[i] offset:0 atIndex:i + 1];
        }
        auto dispatch = [&](size_t kernel, uint32_t groups) {
            [encoder setComputePipelineState:m.pipelines[kernel]];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        };
        auto softmax = [&](uint32_t width, uint32_t exact_mode) {
            [encoder setBytes:&width length:sizeof(width) atIndex:17];
            [encoder setBytes:&exact_mode length:sizeof(exact_mode) atIndex:18];
            dispatch(Softmax, p.heads);
        };
        dispatch(Pack, (std::max(p.heads * p.dim, p.kv_heads * p.count * p.dim) + 255) / 256);
        if (p.topk < p.count) {
            dispatch(Prepare, p.kv_heads);
            dispatch(Approximate, (p.heads * p.chunks + 255) / 256);
            softmax(p.chunks, 0);
        }
        dispatch(Select, p.kv_heads);
        dispatch(Exact, (p.heads * p.topk + 255) / 256);
        softmax(p.topk, 1);
        dispatch(WeightedSum, (p.heads * p.dim + 255) / 256);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            throw metal_error("Metal HSA command failed", command.error);
        }
        if (*static_cast<uint32_t *>(m.buffers[Error].contents)) {
            throw std::runtime_error("Non-finite Metal HSA input or output");
        }
        std::memcpy(output, m.buffers[Output].contents, qbytes);
        if (selected) {
            selected->resize(p.kv_heads);
            const auto *indices = static_cast<const uint32_t *>(m.buffers[Selected].contents);
            for (uint32_t h = 0; h < p.kv_heads; ++h) {
                (*selected)[h].assign(indices + h * p.padded, indices + h * p.padded + p.topk);
            }
        }
    }
}
