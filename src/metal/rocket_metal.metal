#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded, dim_padded;
    uint qd, qh, kd, kt, kh, vd, vt, vh;
    uint cached_count, mask_type, mask_stride, bounds_stride;
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
    // Adjacent scoring lanes read adjacent chunks; a fixed stride preserves cached prefixes.
    bounds[(h * p.dim + d) * p.bounds_stride + c] = half2(lo, hi);
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
            float2 extrema = float2(bounds[(group * p.dim + d) * p.bounds_stride + c]);
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

inline uint selection_prefix(uint value, threadgroup uint *scratch, uint tid, uint width) {
    uint prefix = simd_prefix_exclusive_sum(value);
    uint group = tid / width;
    if (tid % width == width - 1) scratch[group] = prefix + value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < group; ++i) prefix += scratch[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return prefix;
}

template <typename Scores>
inline void select_tokens_impl(constant Params &p, device const float *chunks, device uint *indices,
                               Scores values, threadgroup atomic_uint *histogram, threadgroup uint *scratch,
                               device atomic_uint *error, uint tid, uint group, uint width, mem_flags memory) {
    if (p.topk == p.count) {
        for (uint i = tid; i < p.count; i += 256) {
            indices[group * p.padded + i] = i;
        }
        return;
    }
    uint nh = p.heads / p.kv_heads;
    for (uint i = tid; i < p.chunks; i += 256) {
        float score = 0;
        for (uint h = 0; h < nh; ++h) {
            score += chunks[(group * nh + h) * p.chunks + i];
        }
        if (!isfinite(score) || score < 0) atomic_store_explicit(error, 1u, memory_order_relaxed);
        values[i] = score;
    }
    threadgroup_barrier(memory);

    // Nonnegative F32 scores have the same ordering as their unsigned bit patterns.
    // Weight each chunk by its token count so the short tail cannot change top-k.
    uint cutoff = 0, mask = 0;
    if (tid == 0) scratch[0] = p.topk;
    for (int shift = 28; shift >= 0; shift -= 4) {
        if (tid < 16) atomic_store_explicit(histogram + tid, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint local[16] = {};
        for (uint i = tid; i < p.chunks; i += 256) {
            uint key = as_type<uint>(values[i]);
            if ((key & mask) == cutoff) {
                local[(key >> shift) & 15u] += min(p.chunk, p.count - i * p.chunk);
            }
        }
        for (uint bin = 0; bin < 16; ++bin) {
            if (local[bin]) atomic_fetch_add_explicit(histogram + bin, local[bin], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint rank = scratch[0];
            for (int bin = 15; bin >= 0; --bin) {
                uint count = atomic_load_explicit(histogram + bin, memory_order_relaxed);
                if (rank > count) {
                    rank -= count;
                } else {
                    scratch[0] = rank;
                    scratch[1] = cutoff | (uint(bin) << shift);
                    break;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cutoff = scratch[1];
        mask |= 15u << shift;
    }
    uint rank = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Contiguous chunk ranges plus integer prefix sums emit chronological IDs directly.
    uint block = (p.chunks + 255) / 256;
    uint begin = min(tid * block, p.chunks), end = min(begin + block, p.chunks);
    uint tied = 0;
    for (uint i = begin; i < end; ++i) {
        if (as_type<uint>(values[i]) == cutoff) tied += min(p.chunk, p.count - i * p.chunk);
    }
    uint tie_prefix = selection_prefix(tied, scratch, tid, width);
    uint selected = 0, remaining = rank > tie_prefix ? rank - tie_prefix : 0;
    for (uint i = begin; i < end; ++i) {
        uint key = as_type<uint>(values[i]), weight = min(p.chunk, p.count - i * p.chunk);
        if (key > cutoff) selected += weight;
        if (key == cutoff) {
            uint take = min(remaining, weight);
            selected += take;
            remaining -= take;
        }
    }
    uint output = selection_prefix(selected, scratch, tid, width);
    remaining = rank > tie_prefix ? rank - tie_prefix : 0;
    for (uint i = begin; i < end; ++i) {
        uint key = as_type<uint>(values[i]), weight = min(p.chunk, p.count - i * p.chunk);
        uint take = key > cutoff ? weight : key == cutoff ? min(remaining, weight) : 0;
        if (key == cutoff) remaining -= take;
        for (uint j = 0; j < take; ++j) indices[group * p.padded + output++] = i * p.chunk + j;
    }
}

kernel void select_tokens(ARGS, uint tid [[thread_index_in_threadgroup]],
                          uint group [[threadgroup_position_in_grid]], uint width [[threads_per_simdgroup]]) {
    threadgroup atomic_uint histogram[16];
    threadgroup uint scratch[256];
    select_tokens_impl(p, chunks, indices, scores + group * p.padded, histogram, scratch,
                       error, tid, group, width, mem_flags::mem_device);
}

kernel void select_tokens_local(ARGS, threadgroup float *values [[threadgroup(0)]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint group [[threadgroup_position_in_grid]], uint width [[threads_per_simdgroup]]) {
    threadgroup atomic_uint histogram[16];
    threadgroup uint scratch[256];
    select_tokens_impl(p, chunks, indices, values, histogram, scratch,
                       error, tid, group, width, mem_flags::mem_threadgroup);
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
    uint j = 0;
    for (; j + 3 < p.topk; j += 4) {
        uint s0 = uint(slots[indices[group * p.padded + j]]);
        uint s1 = uint(slots[indices[group * p.padded + j + 1]]);
        uint s2 = uint(slots[indices[group * p.padded + j + 2]]);
        uint s3 = uint(slots[indices[group * p.padded + j + 3]]);
        float v0 = read_kv(vr, group, s0, d, p.vh, p.vt, p.vd);
        float v1 = read_kv(vr, group, s1, d, p.vh, p.vt, p.vd);
        float v2 = read_kv(vr, group, s2, d, p.vh, p.vt, p.vd);
        float v3 = read_kv(vr, group, s3, d, p.vh, p.vt, p.vd);
        sum += exact[h * p.topk + j] * v0;
        sum += exact[h * p.topk + j + 1] * v1;
        sum += exact[h * p.topk + j + 2] * v2;
        sum += exact[h * p.topk + j + 3] * v3;
    }
    for (; j < p.topk; ++j) {
        uint slot = uint(slots[indices[group * p.padded + j]]);
        sum += exact[h * p.topk + j] * read_kv(vr, group, slot, d, p.vh, p.vt, p.vd);
    }
    output[i] = sum;
    if (!isfinite(sum)) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
    }
}
