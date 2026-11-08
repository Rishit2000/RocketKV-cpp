#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded, dim_padded, topk_padded;
    uint qd, qh, kd, kt, kh, vd, vt, vh;
    uint cached_count, mask_type, mask_stride;
};
#define ARGS                                                                                                 \
    constant Params &p [[buffer(0)]], device const uchar *qr [[buffer(1)]],                                  \
        device const uchar *kr [[buffer(2)]], device const uchar *vr [[buffer(3)]],                          \
        device const int *slots [[buffer(4)]], device float *q [[buffer(5)]], device half2 *bounds [[buffer(6)]], \
        device uint *dims [[buffer(7)]], device float *signs [[buffer(8)]],                                 \
        device float *scales [[buffer(9)]], device float *chunks [[buffer(10)]],                             \
        device float *scores [[buffer(11)]], device uint *indices [[buffer(12)]],                            \
        device float *exact [[buffer(13)]], device float *output [[buffer(14)]],                             \
        device atomic_uint *error [[buffer(15)]]

inline float read_kv(device const uchar *raw, uint h, uint slot, uint d, uint hs, uint ts, uint ds) {
    return float(*reinterpret_cast<device const half *>(raw + h * hs + slot * ts + d * ds));
}

kernel void pack(ARGS, device const uchar *mask [[buffer(18)]], uint i [[thread_position_in_grid]]) {
    if (p.mask_type && i < p.count) {
        device const uchar *value = mask + uint(slots[i]) * p.mask_stride;
        float bias = p.mask_type == 1 ? *reinterpret_cast<device const float *>(value)
                                     : float(*reinterpret_cast<device const half *>(value));
        if (bias != 0) atomic_store_explicit(error, 1u, memory_order_relaxed);
    }
    if (i < p.heads * p.dim) {
        float x = *reinterpret_cast<device const float *>(qr + (i / p.dim) * p.qh + (i % p.dim) * p.qd);
        q[i] = x;
        if (!isfinite(x)) {
            atomic_store_explicit(error, 1u, memory_order_relaxed);
        }
    }
    uint added = p.count - p.cached_count;
    if (i >= p.kv_heads * added * p.dim) {
        return;
    }
    uint d = i % p.dim, t = p.cached_count + (i / p.dim) % added, h = i / (p.dim * added);
    float a = read_kv(kr, h, uint(slots[t]), d, p.kh, p.kt, p.kd);
    float b = read_kv(vr, h, uint(slots[t]), d, p.vh, p.vt, p.vd);
    if (!isfinite(a) || !isfinite(b)) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
    }
}

kernel void summarize(ARGS, uint i [[thread_position_in_grid]]) {
    uint first = p.cached_count / p.chunk;
    if (i >= (p.chunks - first) * p.kv_heads * p.dim) return;
    uint d = i % p.dim, h = (i / p.dim) % p.kv_heads, c = first + i / (p.dim * p.kv_heads);
    float lo = INFINITY, hi = -INFINITY;
    for (uint t = c * p.chunk; t < min((c + 1) * p.chunk, p.count); ++t) {
        float x = read_kv(kr, h, uint(slots[t]), d, p.kh, p.kt, p.kd);
        lo = min(lo, x);
        hi = max(hi, x);
    }
    // Chunk-major layout keeps completed chunks stable as candidates grow.
    bounds[(c * p.kv_heads + h) * p.dim + d] = half2(lo, hi);
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
        float rep;
        if (p.chunk == 1) {
            rep = read_kv(kr, group, uint(slots[c]), d, p.kh, p.kt, p.kd);
        } else {
            float2 extrema = float2(bounds[(c * p.kv_heads + group) * p.dim + d]);
            rep = positive ? extrema.y : extrema.x;
        }
        dot += q[h * p.dim + d] * rep;
    }
    chunks[i] = scales[h] > 0 ? dot / scales[h] : 0;
}

// Used for both approximate chunk probabilities and exact selected-token probabilities.
kernel void softmax(ARGS, constant uint &width [[buffer(16)]], constant uint &exact_mode [[buffer(17)]],
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

template <typename Scores, typename Indices>
inline void select_tokens_impl(constant Params &p, device const float *chunks, device uint *indices,
                               Scores values, Indices ids, uint tid, uint group, mem_flags memory) {
    if (p.topk == p.count) {
        for (uint i = tid; i < p.count; i += 256) {
            indices[group * p.padded + i] = i;
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
    threadgroup_barrier(memory);
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
            threadgroup_barrier(memory);
        }
    }
    // Match the CPU reference's chronological order for exact attention accumulation.
    for (uint i = tid; i < p.topk_padded; i += 256) {
        if (i >= p.topk) {
            ids[i] = 0xffffffffu;
        }
    }
    threadgroup_barrier(memory);
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
            threadgroup_barrier(memory);
        }
    }
    for (uint i = tid; i < p.topk; i += 256) indices[group * p.padded + i] = ids[i];
}

kernel void select_tokens(ARGS, uint tid [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]]) {
    select_tokens_impl(p, chunks, indices, scores + group * p.padded, indices + group * p.padded,
                       tid, group, mem_flags::mem_device);
}

kernel void select_tokens_local(ARGS, threadgroup float *values [[threadgroup(0)]],
                               threadgroup uint *ids [[threadgroup(1)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint group [[threadgroup_position_in_grid]]) {
    select_tokens_impl(p, chunks, indices, values, ids, tid, group, mem_flags::mem_threadgroup);
}

kernel void exact_scores(ARGS, uint i [[thread_position_in_grid]]) {
    if (i >= p.heads * p.topk) {
        return;
    }
    uint h = i / p.topk, group = h / (p.heads / p.kv_heads), t = indices[group * p.padded + i % p.topk];
    float dot = 0;
    for (uint d = 0; d < p.dim; ++d) {
        dot += q[h * p.dim + d] * read_kv(kr, group, uint(slots[t]), d, p.kh, p.kt, p.kd);
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
        uint slot = uint(slots[indices[group * p.padded + j]]);
        sum += exact[h * p.topk + j] * read_kv(vr, group, slot, d, p.vh, p.vt, p.vd);
    }
    output[i] = sum;
    if (!isfinite(sum)) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
    }
}
