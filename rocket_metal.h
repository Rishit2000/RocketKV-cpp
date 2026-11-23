#pragma once
#include <vector>
#include <cstddef>
#include "rocket.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the Apple Metal GPU runtime and compile Stage 2 shaders
bool rocket_metal_init();

// Release Metal device, pipelines, and command queues
void rocket_metal_cleanup();

// Check if Metal runtime is ready
bool is_rocket_metal_available();

// Baseline exact dense attention executed directly on Apple Metal GPU
void rocket_metal_dense_decode(
    const float* Q, const float* K, const float* V,
    float* Out, int len_k, const RocketConfig& cfg);

// RocketKV Stage 2: Hybrid Sparse Attention executed on Apple Metal GPU
void rocket_metal_sparse_decode(
    const float* Q, const float* K, const float* V,
    float* Out, int len_k, const RocketConfig& cfg);

// Stage 2 Decode with One-Time KV Binding
void rocket_metal_set_kv(
    const float* K, const float* V, int len_k, const RocketConfig& cfg);

void rocket_metal_sparse_decode_step(
    const float* Q, float* Out, int len_k, const RocketConfig& cfg);

// Persistent GPU Buffer Management for Zero-Overhead Inference
void* rocket_metal_alloc_buffer(size_t bytes);
void  rocket_metal_free_buffer(void* buffer);
void  rocket_metal_copy_to_buffer(void* buffer, const void* src, size_t bytes);
void  rocket_metal_copy_from_buffer(void* dst, const void* buffer, size_t bytes);

// Fast decode executing directly on resident GPU buffers (zero host allocation overhead)
void rocket_metal_dense_decode_device(
    void* buf_q, void* buf_k, void* buf_v, void* buf_out,
    int len_k, const RocketConfig& cfg);

void rocket_metal_sparse_decode_device(
    void* buf_q, void* buf_k, void* buf_v, void* buf_out,
    int len_k, const RocketConfig& cfg);

#ifdef __cplusplus
}
#endif
