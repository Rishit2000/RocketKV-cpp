#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded, dim_padded, topk_padded;
    uint qd, qh, kd, kt, kh, vd, vt, vh;
};
#define ARGS                                                                                                 \
    constant Params &p [[buffer(0)]], device const uchar *qr [[buffer(1)]],                                  \
        device const uchar *kr [[buffer(2)]], device const uchar *vr [[buffer(3)]],                          \
        device const int *slots [[buffer(4)]], device float *q [[buffer(5)]], device float *k [[buffer(6)]], \
        device float *v [[buffer(7)]], device uint *dims [[buffer(8)]], device float *signs [[buffer(9)]],   \
        device float *scales [[buffer(10)]], device float *chunks [[buffer(11)]],                            \
        device float *scores [[buffer(12)]], device uint *indices [[buffer(13)]],                            \
        device float *exact [[buffer(14)]], device float *output [[buffer(15)]],                             \
        device atomic_uint *error [[buffer(16)]]

kernel void pack(ARGS, uint i [[thread_position_in_grid]]) {
    if (i < p.heads * p.dim) {
        float x = *reinterpret_cast<device const float *>(qr + (i / p.dim) * p.qh + (i % p.dim) * p.qd);
        q[i] = x;
        if (!isfinite(x)) {
            atomic_store_explicit(error, 1u, memory_order_relaxed);
        }
    }
    if (i >= p.kv_heads * p.count * p.dim) {
        return;
    }
    uint d = i % p.dim, t = (i / p.dim) % p.count, h = i / (p.dim * p.count), slot = uint(slots[t]);
    float a = float(*reinterpret_cast<device const half *>(kr + h * p.kh + slot * p.kt + d * p.kd));
    float b = float(*reinterpret_cast<device const half *>(vr + h * p.vh + slot * p.vt + d * p.vd));
    k[i] = a;
    v[i] = b;
    if (!isfinite(a) || !isfinite(b)) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
    }
}

inline bool better(float a, uint ai, float b, uint bi) {
    return a > b || (a == b && ai < bi);
}

kernel void prepare(ARGS, uint tid [[thread_index_in_threadgroup]],
                    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float magnitude[256];
    threadgroup uint order[256];
    uint nh = p.heads / p.kv_heads;
    if (tid < p.dim_padded) {
        float sum = 0, mag = 0;
        if (tid < p.dim) {
            for (uint h = 0; h < nh; ++h) {
                float x = q[(group * nh + h) * p.dim + tid];
                sum += x;
                mag += abs(x);
            }
            signs[group * p.dim + tid] = sum >= 0 ? 1.0f : -1.0f;
        }
        magnitude[tid] = tid < p.dim ? mag : -INFINITY;
        order[tid] = tid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint size = 2; size <= p.dim_padded; size *= 2) {
        for (uint stride = size / 2; stride; stride /= 2) {
            uint j = tid ^ stride;
            if (tid < p.dim_padded && j > tid) {
                bool swap = (tid & size) == 0 ? better(magnitude[j], order[j], magnitude[tid], order[tid])
                                              : better(magnitude[tid], order[tid], magnitude[j], order[j]);
                if (swap) {
                    float f = magnitude[tid];
                    magnitude[tid] = magnitude[j];
                    magnitude[j] = f;
                    uint u = order[tid];
                    order[tid] = order[j];
                    order[j] = u;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < p.dim) {
        dims[group * p.dim + tid] = p.rank == p.dim ? tid : order[tid];
    }
    if (tid < nh) {
        float all = 0, kept = 0;
        for (uint d = 0; d < p.dim; ++d) {
            all += abs(q[(group * nh + tid) * p.dim + d]);
        }
        for (uint r = 0; r < p.rank; ++r) {
            kept += abs(q[(group * nh + tid) * p.dim + (p.rank == p.dim ? r : order[r])]);
        }
        scales[group * nh + tid] = sqrt(float(p.dim) * (kept / (all + 1e-9f)));
    }
}

kernel void approximate(ARGS, uint i [[thread_position_in_grid]]) {
    if (i >= p.heads * p.chunks) {
        return;
    }
    uint h = i / p.chunks, c = i % p.chunks, group = h / (p.heads / p.kv_heads);
    float dot = 0;
    for (uint r = 0; r < p.rank; ++r) {
        uint d = dims[group * p.dim + r];
        bool positive = signs[group * p.dim + d] >= 0;
        float rep = positive ? -INFINITY : INFINITY;
        for (uint t = c * p.chunk; t < min((c + 1) * p.chunk, p.count); ++t) {
            float x = k[(group * p.count + t) * p.dim + d];
            rep = positive ? max(rep, x) : min(rep, x);
        }
        dot += q[h * p.dim + d] * rep;
    }
    chunks[i] = scales[h] > 0 ? dot / scales[h] : 0;
}

// Used for both approximate chunk probabilities and exact selected-token probabilities.
kernel void softmax(ARGS, constant uint &width [[buffer(17)]], constant uint &exact_mode [[buffer(18)]],
                    uint tid [[thread_index_in_threadgroup]], uint h [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[256];
    device float *row = (exact_mode ? exact : chunks) + h * width;
    float m = -INFINITY;
    for (uint i = tid; i < width; i += 256) {
        m = max(m, row[i]);
    }
    scratch[tid] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s; s /= 2) {
        if (tid < s) {
            scratch[tid] = max(scratch[tid], scratch[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    m = scratch[0];
    float sum = 0;
    for (uint i = tid; i < width; i += 256) {
        float x = exp(row[i] - m);
        row[i] = x;
        sum += x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s; s /= 2) {
        if (tid < s) {
            scratch[tid] += scratch[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = 1.0f / (scratch[0] + 1e-9f);
    for (uint i = tid; i < width; i += 256) {
        row[i] *= inverse;
    }
}

kernel void select_tokens(ARGS, uint tid [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]]) {
    device float *values = scores + group * p.padded;
    device uint *ids = indices + group * p.padded;
    if (p.topk == p.count) {
        for (uint i = tid; i < p.count; i += 256) {
            ids[i] = i;
        }
        return;
    }
    uint nh = p.heads / p.kv_heads;
    for (uint i = tid; i < p.padded; i += 256) {
        float score = 0;
        if (i < p.count) {
            for (uint h = 0; h < nh; ++h) {
                score += chunks[(group * nh + h) * p.chunks + i / p.chunk];
            }
        }
        values[i] = i < p.count ? score : -INFINITY;
        ids[i] = i;
    }
    threadgroup_barrier(mem_flags::mem_device);
    // One threadgroup owns each KV group's sort; no cross-threadgroup barrier is needed.
    for (uint size = 2; size <= p.padded; size *= 2) {
        for (uint stride = size / 2; stride; stride /= 2) {
            for (uint i = tid; i < p.padded; i += 256) {
                uint j = i ^ stride;
                if (j > i) {
                    bool swap = (i & size) == 0 ? better(values[j], ids[j], values[i], ids[i])
                                                : better(values[i], ids[i], values[j], ids[j]);
                    if (swap) {
                        float f = values[i];
                        values[i] = values[j];
                        values[j] = f;
                        uint u = ids[i];
                        ids[i] = ids[j];
                        ids[j] = u;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
    // Match the CPU reference's chronological order for exact attention accumulation.
    for (uint i = tid; i < p.topk_padded; i += 256) {
        if (i >= p.topk) {
            ids[i] = 0xffffffffu;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint size = 2; size <= p.topk_padded; size *= 2) {
        for (uint stride = size / 2; stride; stride /= 2) {
            for (uint i = tid; i < p.topk_padded; i += 256) {
                uint j = i ^ stride;
                if (j > i && ((i & size) == 0 ? ids[i] > ids[j] : ids[i] < ids[j])) {
                    uint u = ids[i];
                    ids[i] = ids[j];
                    ids[j] = u;
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
    }
}

kernel void exact_scores(ARGS, uint i [[thread_position_in_grid]]) {
    if (i >= p.heads * p.topk) {
        return;
    }
    uint h = i / p.topk, group = h / (p.heads / p.kv_heads), t = indices[group * p.padded + i % p.topk];
    float dot = 0;
    for (uint d = 0; d < p.dim; ++d) {
        dot += q[h * p.dim + d] * k[(group * p.count + t) * p.dim + d];
    }
    exact[i] = dot / sqrt(float(p.dim));
}

kernel void weighted_sum(ARGS, uint i [[thread_position_in_grid]]) {
    if (i >= p.heads * p.dim) {
        return;
    }
    uint h = i / p.dim, d = i % p.dim, group = h / (p.heads / p.kv_heads);
    float sum = 0;
    for (uint j = 0; j < p.topk; ++j) {
        sum += exact[h * p.topk + j] * v[(group * p.count + indices[group * p.padded + j]) * p.dim + d];
    }
    output[i] = sum;
    if (!isfinite(sum)) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
    }
}
