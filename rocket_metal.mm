#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "rocket_metal.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>

static id<MTLDevice>              g_metal_device = nil;
static id<MTLCommandQueue>         g_command_queue = nil;
static id<MTLComputePipelineState> g_pipe_dense = nil;
static id<MTLComputePipelineState> g_pipe_sparse = nil;
static bool                        g_metal_ready = false;

// Internal persistent buffers for cached host execution
static id<MTLBuffer> g_cached_q = nil;
static id<MTLBuffer> g_cached_k = nil;
static id<MTLBuffer> g_cached_v = nil;
static id<MTLBuffer> g_cached_out = nil;
static id<MTLBuffer> g_cached_scratch = nil;
static id<MTLBuffer> g_cached_params = nil;

struct MetalParams {
    int num_heads;
    int num_heads_kv;
    int heads_per_kv;
    int dim_head;
    int len_k;
    int chunk_size;
    int num_chunks;
    int r_dim;
    int topk;
    float inv_sqrt_d;
};

static id<MTLBuffer> get_or_grow_buffer(id<MTLBuffer>& buf, size_t required_bytes) {
    if (buf == nil || [buf length] < required_bytes) {
        size_t alloc_bytes = std::max(required_bytes, (size_t)4096);
        buf = [g_metal_device newBufferWithLength:alloc_bytes options:MTLResourceStorageModeShared];
    }
    return buf;
}

bool rocket_metal_init() {
    if (g_metal_ready) return true;

    @autoreleasepool {
        g_metal_device = MTLCreateSystemDefaultDevice();
        if (!g_metal_device) {
            std::cerr << "[RocketKV Metal] Error: No Metal device found in current process environment!" << std::endl;
            return false;
        }

        g_command_queue = [g_metal_device newCommandQueue];
        if (!g_command_queue) {
            std::cerr << "[RocketKV Metal] Error: Failed to create Metal command queue!" << std::endl;
            return false;
        }

        // Load shader source from rocket_stage2.metal with resilient search paths
        std::vector<std::string> shader_paths = {
            "rocket_stage2.metal",
            "RocketKV-cpp/rocket_stage2.metal",
            "../rocket_stage2.metal"
        };
        std::ifstream file;
        for (const auto & p : shader_paths) {
            file.open(p);
            if (file.is_open()) break;
        }
        if (!file.is_open()) {
            std::cerr << "[RocketKV Metal] Error: Could not locate rocket_stage2.metal!" << std::endl;
            return false;
        }

        std::stringstream ss;
        ss << file.rdbuf();
        std::string source_str = ss.str();

        NSError* error = nil;
        NSString* src_ns = [NSString stringWithUTF8String:source_str.c_str()];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = YES;
#pragma clang diagnostic pop

        id<MTLLibrary> library = [g_metal_device newLibraryWithSource:src_ns
                                                              options:options
                                                                error:&error];
        if (!library) {
            std::cerr << "[RocketKV Metal] Shader compilation error: "
                      << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        id<MTLFunction> fn_dense = [library newFunctionWithName:@"kernel_exact_dense_decode"];
        if (!fn_dense) {
            std::cerr << "[RocketKV Metal] Failed to find function: kernel_exact_dense_decode" << std::endl;
            return false;
        }
        g_pipe_dense = [g_metal_device newComputePipelineStateWithFunction:fn_dense error:&error];
        if (!g_pipe_dense) {
            std::cerr << "[RocketKV Metal] Pipeline creation error (dense): "
                      << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        id<MTLFunction> fn_sparse = [library newFunctionWithName:@"kernel_rocket_sparse_decode"];
        if (!fn_sparse) {
            std::cerr << "[RocketKV Metal] Failed to find function: kernel_rocket_sparse_decode" << std::endl;
            return false;
        }
        g_pipe_sparse = [g_metal_device newComputePipelineStateWithFunction:fn_sparse error:&error];
        if (!g_pipe_sparse) {
            std::cerr << "[RocketKV Metal] Pipeline creation error (sparse): "
                      << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }

        g_metal_ready = true;
        std::cout << "[RocketKV Metal] Initialized successfully on " 
                  << [[g_metal_device name] UTF8String] << "!" << std::endl;
        return true;
    }
}

void rocket_metal_cleanup() {
    @autoreleasepool {
        g_cached_q = nil;
        g_cached_k = nil;
        g_cached_v = nil;
        g_cached_out = nil;
        g_cached_scratch = nil;
        g_cached_params = nil;

        g_pipe_dense = nil;
        g_pipe_sparse = nil;
        g_command_queue = nil;
        g_metal_device = nil;
        g_metal_ready = false;
    }
}

bool is_rocket_metal_available() {
    return g_metal_ready;
}

// -----------------------------------------------------------------------------
// Persistent Buffer Management
// -----------------------------------------------------------------------------
void* rocket_metal_alloc_buffer(size_t bytes) {
    if (!g_metal_ready && !rocket_metal_init()) return nullptr;
    @autoreleasepool {
        id<MTLBuffer> buf = [g_metal_device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
#if __has_feature(objc_arc)
        return (__bridge_retained void*)buf;
#else
        [buf retain];
        return (void*)buf;
#endif
    }
}

void rocket_metal_free_buffer(void* buffer) {
    if (!buffer) return;
    @autoreleasepool {
#if __has_feature(objc_arc)
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)buffer;
        buf = nil;
#else
        id<MTLBuffer> buf = (id<MTLBuffer>)buffer;
        [buf release];
#endif
    }
}

void rocket_metal_copy_to_buffer(void* buffer, const void* src, size_t bytes) {
    if (!buffer || !src || bytes == 0) return;
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buffer;
    memcpy([mtl_buf contents], src, bytes);
}

void rocket_metal_copy_from_buffer(void* dst, const void* buffer, size_t bytes) {
    if (!buffer || !dst || bytes == 0) return;
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buffer;
    memcpy(dst, [mtl_buf contents], bytes);
}

// -----------------------------------------------------------------------------
// Direct Device Execution (Zero CPU allocation)
// -----------------------------------------------------------------------------
void rocket_metal_dense_decode_device(
    void* buf_q, void* buf_k, void* buf_v, void* buf_out,
    int len_k, const RocketConfig& cfg)
{
    if (!g_metal_ready && !rocket_metal_init()) return;

    @autoreleasepool {
        int heads_per_kv = cfg.num_heads / cfg.num_heads_kv;
        MetalParams params{};
        params.num_heads = cfg.num_heads;
        params.num_heads_kv = cfg.num_heads_kv;
        params.heads_per_kv = heads_per_kv;
        params.dim_head = cfg.dim_head;
        params.len_k = len_k;
        params.chunk_size = (cfg.chunk_size > 0) ? cfg.chunk_size : 2;
        params.num_chunks = (len_k + params.chunk_size - 1) / params.chunk_size;
        params.r_dim = (cfg.r_dim > 0) ? cfg.r_dim : 64;
        params.topk = (cfg.topk_decode > 0) ? cfg.topk_decode : len_k;
        params.inv_sqrt_d = 1.0f / std::sqrt((float)cfg.dim_head);

        id<MTLBuffer> mtl_q = (__bridge id<MTLBuffer>)buf_q;
        id<MTLBuffer> mtl_k = (__bridge id<MTLBuffer>)buf_k;
        id<MTLBuffer> mtl_v = (__bridge id<MTLBuffer>)buf_v;
        id<MTLBuffer> mtl_out = (__bridge id<MTLBuffer>)buf_out;
        id<MTLBuffer> mtl_p = get_or_grow_buffer(g_cached_params, sizeof(params));
        memcpy([mtl_p contents], &params, sizeof(params));

        id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

        [enc setComputePipelineState:g_pipe_dense];
        [enc setBuffer:mtl_q offset:0 atIndex:0];
        [enc setBuffer:mtl_k offset:0 atIndex:1];
        [enc setBuffer:mtl_v offset:0 atIndex:2];
        [enc setBuffer:mtl_out offset:0 atIndex:3];
        [enc setBuffer:mtl_p offset:0 atIndex:4];

        MTLSize grid_size = MTLSizeMake(cfg.num_heads, 1, 1);
        MTLSize tg_size = MTLSizeMake(32, 1, 1);
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];

        [enc endEncoding];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void rocket_metal_sparse_decode_device(
    void* buf_q, void* buf_k, void* buf_v, void* buf_out,
    int len_k, const RocketConfig& cfg)
{
    if (!g_metal_ready && !rocket_metal_init()) return;

    @autoreleasepool {
        int heads_per_kv = std::max(1, cfg.num_heads / cfg.num_heads_kv);
        int chunk_size = (cfg.chunk_size > 0) ? cfg.chunk_size : 2;
        int num_chunks = (len_k + chunk_size - 1) / chunk_size;

        MetalParams params{};
        params.num_heads = cfg.num_heads;
        params.num_heads_kv = cfg.num_heads_kv;
        params.heads_per_kv = heads_per_kv;
        params.dim_head = cfg.dim_head;
        params.len_k = len_k;
        params.chunk_size = chunk_size;
        params.num_chunks = num_chunks;
        params.r_dim = (cfg.r_dim > 0) ? cfg.r_dim : 64;
        params.topk = (cfg.topk_decode > 0) ? std::min(cfg.topk_decode, len_k) : len_k;
        params.inv_sqrt_d = 1.0f / std::sqrt((float)cfg.dim_head);

        size_t scratch_bytes = cfg.num_heads * num_chunks * sizeof(float);
        id<MTLBuffer> mtl_scratch = get_or_grow_buffer(g_cached_scratch, scratch_bytes);

        id<MTLBuffer> mtl_q = (__bridge id<MTLBuffer>)buf_q;
        id<MTLBuffer> mtl_k = (__bridge id<MTLBuffer>)buf_k;
        id<MTLBuffer> mtl_v = (__bridge id<MTLBuffer>)buf_v;
        id<MTLBuffer> mtl_out = (__bridge id<MTLBuffer>)buf_out;
        id<MTLBuffer> mtl_p = get_or_grow_buffer(g_cached_params, sizeof(params));
        memcpy([mtl_p contents], &params, sizeof(params));

        id<MTLCommandBuffer> cmd_buf = [g_command_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoder];

        [enc setComputePipelineState:g_pipe_sparse];
        [enc setBuffer:mtl_q offset:0 atIndex:0];
        [enc setBuffer:mtl_k offset:0 atIndex:1];
        [enc setBuffer:mtl_v offset:0 atIndex:2];
        [enc setBuffer:mtl_out offset:0 atIndex:3];
        [enc setBuffer:mtl_scratch offset:0 atIndex:4];
        [enc setBuffer:mtl_p offset:0 atIndex:5];

        MTLSize grid_size = MTLSizeMake(cfg.num_heads_kv, 1, 1);
        MTLSize tg_size = MTLSizeMake(128, 1, 1);
        [enc dispatchThreadgroups:grid_size threadsPerThreadgroup:tg_size];

        [enc endEncoding];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

// -----------------------------------------------------------------------------
// Host Pointer Convenience APIs (with internal buffer caching)
// -----------------------------------------------------------------------------
void rocket_metal_dense_decode(
    const float* Q, const float* K, const float* V,
    float* Out, int len_k, const RocketConfig& cfg)
{
    if (!g_metal_ready && !rocket_metal_init()) {
        std::cerr << "[RocketKV Metal] Cannot run: Metal runtime not initialized." << std::endl;
        return;
    }

    @autoreleasepool {
        size_t q_bytes = cfg.num_heads * cfg.dim_head * sizeof(float);
        size_t kv_bytes = cfg.num_heads_kv * len_k * cfg.dim_head * sizeof(float);
        size_t out_bytes = cfg.num_heads * cfg.dim_head * sizeof(float);

        id<MTLBuffer> buf_q = get_or_grow_buffer(g_cached_q, q_bytes);
        id<MTLBuffer> buf_k = get_or_grow_buffer(g_cached_k, kv_bytes);
        id<MTLBuffer> buf_v = get_or_grow_buffer(g_cached_v, kv_bytes);
        id<MTLBuffer> buf_out = get_or_grow_buffer(g_cached_out, out_bytes);

        memcpy([buf_q contents], Q, q_bytes);
        memcpy([buf_k contents], K, kv_bytes);
        memcpy([buf_v contents], V, kv_bytes);

        rocket_metal_dense_decode_device(
            (__bridge void*)buf_q, (__bridge void*)buf_k, (__bridge void*)buf_v, (__bridge void*)buf_out,
            len_k, cfg);

        memcpy(Out, [buf_out contents], out_bytes);
    }
}

void rocket_metal_set_kv(
    const float* K, const float* V, int len_k, const RocketConfig& cfg)
{
    if (!g_metal_ready && !rocket_metal_init()) {
        std::cerr << "[RocketKV Metal] Cannot bind KV: Metal runtime not initialized." << std::endl;
        return;
    }
    if (!K || !V || len_k <= 0) return;

    @autoreleasepool {
        size_t kv_bytes = cfg.num_heads_kv * len_k * cfg.dim_head * sizeof(float);
        id<MTLBuffer> buf_k = get_or_grow_buffer(g_cached_k, kv_bytes);
        id<MTLBuffer> buf_v = get_or_grow_buffer(g_cached_v, kv_bytes);

        memcpy([buf_k contents], K, kv_bytes);
        memcpy([buf_v contents], V, kv_bytes);
    }
}

void rocket_metal_sparse_decode_step(
    const float* Q, float* Out, int len_k, const RocketConfig& cfg)
{
    if (!g_metal_ready && !rocket_metal_init()) {
        std::cerr << "[RocketKV Metal] Cannot run decode step: Metal runtime not initialized." << std::endl;
        return;
    }
    if (!Q || !Out || len_k <= 0) return;

    @autoreleasepool {
        size_t q_bytes = cfg.num_heads * cfg.dim_head * sizeof(float);
        size_t out_bytes = cfg.num_heads * cfg.dim_head * sizeof(float);

        id<MTLBuffer> buf_q = get_or_grow_buffer(g_cached_q, q_bytes);
        id<MTLBuffer> buf_out = get_or_grow_buffer(g_cached_out, out_bytes);

        memcpy([buf_q contents], Q, q_bytes);

        rocket_metal_sparse_decode_device(
            (__bridge void*)buf_q,
            (__bridge void*)g_cached_k,
            (__bridge void*)g_cached_v,
            (__bridge void*)buf_out,
            len_k, cfg);

        memcpy(Out, [buf_out contents], out_bytes);
    }
}

void rocket_metal_sparse_decode(
    const float* Q, const float* K, const float* V,
    float* Out, int len_k, const RocketConfig& cfg)
{
    rocket_metal_set_kv(K, V, len_k, cfg);
    rocket_metal_sparse_decode_step(Q, Out, len_k, cfg);
}
