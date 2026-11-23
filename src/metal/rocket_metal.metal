#include <metal_stdlib>
using namespace metal;

struct Params {
    uint dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded;
    uint qd, qh, kd, kt, kh, vd, vt, vh;
    uint cached_count, mask_type, mask_stride, bounds_stride;
};
#define ARGS                                                                                                 \
    constant Params &p [[buffer(0)]], device const uchar *qr [[buffer(1)]],                                  \
        device const uchar *kr [[buffer(2)]], device const uchar *vr [[buffer(3)]],                          \
        device const int *slots [[buffer(4)]], device float *q [[buffer(5)]], device half2 *bounds [[buffer(6)]], \
        coherent(device) device float *chunks [[buffer(7)]], device float *scores [[buffer(8)]],                              \
        device uint *indices [[buffer(9)]], coherent(device) device float *partials [[buffer(10)]],                           \
        device float *output [[buffer(11)]], device atomic_uint *error [[buffer(12)]]

// Decode kernels assume 32-wide simdgroups (checked on the host) and at most MAX_GROUP query heads
// per KV head; lanes own dimensions lane, lane + 32, ... (head dimension <= 32 * MAX_LANE_DIMS).
constant uint MAX_GROUP = 8, MAX_LANE_DIMS = 8;

inline float read_kv(device const uchar *raw, uint h, uint slot, uint d, uint hs, uint ts, uint ds) {
    return float(*reinterpret_cast<device const half *>(raw + h * hs + slot * ts + d * ds));
}

inline bool better(float a, uint ai, float b, uint bi) {
    return a > b || (a == b && ai < bi);
}

inline void normalize_row(device float *row, uint width, uint tid, threadgroup float *scratch) {
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


struct PrefillParams {
    uint dim, heads, kv_heads, len_k, obs, count, pad;
    float scale;
};
#define PREFILL_ARGS \
    constant PrefillParams &p [[buffer(0)]], device const float *q [[buffer(1)]], \
    device const float *k [[buffer(2)]], device float *packed [[buffer(3)]], \
    device float *logits [[buffer(4)]], device float *groups [[buffer(5)]], \
    device float *scores [[buffer(6)]], device atomic_uint *error [[buffer(7)]]

struct CaptureParams {
    uint heads, tokens, offset, total, sd, sh, st, type, is_query;
};

kernel void prefill_capture(constant PrefillParams &p [[buffer(0)]], device float *q [[buffer(1)]],
                           device float *packed [[buffer(3)]], device atomic_uint *error [[buffer(7)]],
                           device const uchar *raw [[buffer(8)]], constant CaptureParams &c [[buffer(9)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= c.tokens * c.heads * p.dim) return;
    uint d = i % p.dim, h = (i / p.dim) % c.heads, t = i / (p.dim * c.heads);
    device const uchar *src = raw + d * c.sd + h * c.sh + t * c.st;
    float value = c.type == 0 ? *reinterpret_cast<device const float *>(src)
                : c.type == 1 ? float(*reinterpret_cast<device const half *>(src))
                              : as_type<float>(uint(*reinterpret_cast<device const ushort *>(src)) << 16);
    if (!isfinite(value)) atomic_store_explicit(error, 1u, memory_order_relaxed);
    uint pos = c.offset + t;
    if (c.is_query) {
        if (pos >= c.total - p.obs) q[(h * p.obs + pos - (c.total - p.obs)) * p.dim + d] = value;
    } else if (pos < p.count) {
        packed[(h * p.dim + d) * p.count + pos] = value;
    }
}

kernel void prefill_pack(PREFILL_ARGS, uint i [[thread_position_in_grid]]) {
    if (i >= p.kv_heads * p.count * p.dim) return;
    uint t = i % p.count, d = (i / p.count) % p.dim, h = i / (p.count * p.dim);
    packed[i] = k[(h * p.len_k + t) * p.dim + d];
}

kernel void prefill_dot(PREFILL_ARGS, uint i [[thread_position_in_grid]]) {
    if (i >= p.heads * p.obs * p.count) return;
    uint t = i % p.count, row = i / p.count;
    uint group = (row / p.obs) / (p.heads / p.kv_heads);
    float dot = 0;
    for (uint d = 0; d < p.dim; ++d) {
        dot += q[row * p.dim + d] * packed[(group * p.dim + d) * p.count + t];
    }
    logits[i] = dot * p.scale;
    if (!isfinite(logits[i])) atomic_store_explicit(error, 1u, memory_order_relaxed);
}

kernel void prefill_softmax(PREFILL_ARGS, uint tid [[thread_index_in_threadgroup]],
                           uint row [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[256];
    normalize_row(logits + row * p.count, p.count, tid, scratch);
}

kernel void prefill_reduce(PREFILL_ARGS, uint i [[thread_position_in_grid]]) {
    if (i >= p.kv_heads * p.count) return;
    uint group = i / p.count, t = i % p.count;
    uint rows = p.heads / p.kv_heads * p.obs;
    float sum = 0;
    for (uint row = group * rows; row < (group + 1) * rows; ++row) {
        sum += logits[row * p.count + t];
    }
    groups[i] = sum;
    if (!isfinite(sum)) atomic_store_explicit(error, 1u, memory_order_relaxed);
}

kernel void prefill_pool(PREFILL_ARGS, uint t [[thread_position_in_grid]]) {
    if (t >= p.count) return;
    float sum = 0;
    for (uint h = 0; h < p.kv_heads; ++h) {
        float pooled = -INFINITY;
        uint start = t > p.pad ? t - p.pad : 0;
        uint end = t + min(p.pad, p.count - 1 - t);
        for (uint j = start; j <= end; ++j) pooled = max(pooled, groups[h * p.count + j]);
        sum += pooled;
    }
    scores[t] = sum;
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

// Per-head max or sum over the threadgroup for the nh heads of a GQA group; result[g] for all threads.
template <bool MAX>
inline void reduce_group(thread const float *value, uint nh, threadgroup float *partial, threadgroup float *result,
                         uint lane, uint sg, uint sgs) {
    for (uint g = 0; g < nh; ++g) {
        float x = MAX ? simd_max(value[g]) : simd_sum(value[g]);
        if (lane == 0) partial[g * 32 + sg] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (uint g = 0; g < nh; ++g) {
            float x = lane < sgs ? partial[g * 32 + lane] : (MAX ? -INFINITY : 0.0f);
            x = MAX ? simd_max(x) : simd_sum(x);
            if (lane == 0) result[g] = x;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The last threadgroup (or simdgroup) of a KV head to finish continues with the next phase.
// Tickets live after the error word and are zeroed by the host before every dispatch.
inline bool last_arrival(device atomic_uint *ticket, uint arrivals, uint tid, threadgroup bool *last) {
    atomic_thread_fence(mem_flags::mem_device, memory_order_release, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        *last = atomic_fetch_add_explicit(ticket, 1u, memory_order_relaxed) == arrivals - 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!*last) return false;
    atomic_thread_fence(mem_flags::mem_device, memory_order_acquire, thread_scope_device);
    return true;
}

// Per-head softmax over the group's chunk scores, GQA group sum, and a token-count-weighted
// top-k emitted as chronological candidate indices; run by one threadgroup per KV head.
inline void select_group(constant Params &p, coherent(device) device const float *chunks, device float *scores,
                         device uint *indices, device atomic_uint *error, uint h, uint tid, uint threads,
                         uint lane, uint sg, uint sgs, threadgroup float *partial, threadgroup float *stat,
                         threadgroup uint *scratch, threadgroup atomic_uint *histogram) {
    const uint nh = p.heads / p.kv_heads;
    coherent(device) device const float *raw = chunks + h * nh * p.chunks;
    float local[MAX_GROUP];
    for (uint g = 0; g < MAX_GROUP; ++g) local[g] = -INFINITY;
    for (uint c = tid; c < p.chunks; c += threads) {
        for (uint g = 0; g < nh; ++g) local[g] = max(local[g], raw[g * p.chunks + c]);
    }
    reduce_group<true>(local, nh, partial, stat, lane, sg, sgs);
    float top[MAX_GROUP];
    for (uint g = 0; g < MAX_GROUP; ++g) {
        top[g] = g < nh ? stat[g] : 0;
        local[g] = 0;
    }
    for (uint c = tid; c < p.chunks; c += threads) {
        for (uint g = 0; g < nh; ++g) local[g] += exp(raw[g * p.chunks + c] - top[g]);
    }
    reduce_group<false>(local, nh, partial, stat, lane, sg, sgs);
    float inverse[MAX_GROUP];
    for (uint g = 0; g < MAX_GROUP; ++g) inverse[g] = g < nh ? 1.0f / (stat[g] + 1e-9f) : 0;
    device float *values = scores + h * p.padded;
    for (uint c = tid; c < p.chunks; c += threads) {
        float v = 0;
        for (uint g = 0; g < nh; ++g) v += exp(raw[g * p.chunks + c] - top[g]) * inverse[g];
        if (!isfinite(v) || v < 0) atomic_store_explicit(error, 1u, memory_order_relaxed);
        values[c] = v;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Radix select on the bit patterns (nonnegative F32 orders like uint), 8 bits per pass.
    // Each chunk counts as its number of tokens so the short tail cannot change top-k.
    uint cutoff = 0, known = 0;
    if (tid == 0) scratch[0] = p.topk;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (uint bin = tid; bin < 256; bin += threads) atomic_store_explicit(histogram + bin, 0u, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < p.chunks; i += threads) {
            uint key = as_type<uint>(values[i]);
            if ((key & known) == cutoff) {
                atomic_fetch_add_explicit(histogram + ((key >> shift) & 255u), min(p.chunk, p.count - i * p.chunk),
                                          memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            // Lane l owns bins 255 - 8l down to 248 - 8l; find the bin holding the rank-th largest.
            uint rank = scratch[0], counts[8], owned = 0;
            for (uint k = 0; k < 8; ++k) {
                counts[k] = atomic_load_explicit(histogram + 255 - 8 * lane - k, memory_order_relaxed);
                owned += counts[k];
            }
            uint before = simd_prefix_exclusive_sum(owned);
            if (before < rank && rank <= before + owned) {
                rank -= before;
                for (uint k = 0; k < 8; ++k) {
                    if (rank <= counts[k]) {
                        scratch[0] = rank;
                        scratch[1] = cutoff | ((255u - 8 * lane - k) << shift);
                        break;
                    }
                    rank -= counts[k];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        cutoff = scratch[1];
        known |= 255u << shift;
    }
    uint rank = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Contiguous chunk ranges plus integer prefix sums emit chronological IDs directly.
    uint block = (p.chunks + threads - 1) / threads;
    uint begin = min(tid * block, p.chunks), end = min(begin + block, p.chunks);
    uint tied = 0;
    for (uint i = begin; i < end; ++i) {
        if (as_type<uint>(values[i]) == cutoff) tied += min(p.chunk, p.count - i * p.chunk);
    }
    uint tie_prefix = selection_prefix(tied, scratch, tid, 32);
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
    uint out = selection_prefix(selected, scratch, tid, 32);
    remaining = rank > tie_prefix ? rank - tie_prefix : 0;
    for (uint i = begin; i < end; ++i) {
        uint key = as_type<uint>(values[i]), weight = min(p.chunk, p.count - i * p.chunk);
        uint take = key > cutoff ? weight : key == cutoff ? min(remaining, weight) : 0;
        if (key == cutoff) remaining -= take;
        for (uint j = 0; j < take; ++j) indices[h * p.padded + out++] = i * p.chunk + j;
    }
}

// Stage 2 selection. Grid (blocks, KV heads); each threadgroup owns threads / 4 consecutive chunks
// of one KV head, four lanes per chunk. It recomputes the group's top-r query dimensions (largest group magnitude, ties to
// the lower index; cheaper than another dependent dispatch), refreshes summaries of its chunks
// that received new tokens, and writes approximate scores q·rep / scale, where rep is the chunk
// max (positive group sign) or min, or the key itself for 1-token chunks. The last block of each
// KV head then runs the selection. Input checks are spread over the grid.
kernel void score(ARGS, device const uchar *mask [[buffer(18)]],
                  uint tid [[thread_index_in_threadgroup]], uint2 tg [[threadgroup_position_in_grid]],
                  uint2 grid [[threadgroups_per_grid]], uint2 shape [[threads_per_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint sgs [[simdgroups_per_threadgroup]]) {
    const uint threads = shape.x;
    threadgroup float qg[MAX_GROUP * 256], magnitude[256], sign[256], scale[MAX_GROUP];
    threadgroup float partial[MAX_GROUP * 32], stat[MAX_GROUP];
    threadgroup uint order[256], scratch[32];
    threadgroup atomic_uint histogram[256];
    threadgroup bool last;
    const uint b = tg.x, h = tg.y, blocks = grid.x, nh = p.heads / p.kv_heads;
    if (p.mask_type) {
        uint stride = blocks * p.kv_heads * threads;
        for (uint i = (h * blocks + b) * threads + tid; i < p.count; i += stride) {
            device const uchar *value = mask + uint(slots[i]) * p.mask_stride;
            float bias = p.mask_type == 1 ? *reinterpret_cast<device const float *>(value)
                                         : float(*reinterpret_cast<device const half *>(value));
            if (bias != 0) atomic_store_explicit(error, 1u, memory_order_relaxed);
        }
    }
    for (uint i = tid; i < nh * p.dim; i += threads) {
        uint head = h * nh + i / p.dim, d = i % p.dim;
        float x = *reinterpret_cast<device const float *>(qr + head * p.qh + d * p.qd);
        qg[i] = x;
        if (b == 0) {
            if (!isfinite(x)) atomic_store_explicit(error, 1u, memory_order_relaxed);
            q[head * p.dim + d] = x;
        }
    }
    if (b == 0) {
        for (uint i = tid; i < (p.count - p.cached_count) * p.dim; i += threads) {
            uint slot = uint(slots[p.cached_count + i / p.dim]), d = i % p.dim;
            if (!isfinite(read_kv(kr, h, slot, d, p.kh, p.kt, p.kd)) ||
                !isfinite(read_kv(vr, h, slot, d, p.vh, p.vt, p.vd))) {
                atomic_store_explicit(error, 1u, memory_order_relaxed);
            }
        }
    }
    if (p.topk == p.count) {
        for (uint i = b * threads + tid; i < p.count; i += blocks * threads) indices[h * p.padded + i] = i;
        return;
    }
    const uint begin = b * (threads / 4), end = min(begin + threads / 4, p.chunks);
    if (p.chunk > 1) {
        // Bounds persist across steps; only chunks that received new tokens are recomputed.
        uint first = max(begin, p.cached_count / p.chunk);
        for (uint i = tid; first < end && i < (end - first) * p.dim; i += threads) {
            uint d = i % p.dim, c = first + i / p.dim;
            float lo = INFINITY, hi = -INFINITY;
            for (uint t = c * p.chunk; t < min((c + 1) * p.chunk, p.count); ++t) {
                float x = read_kv(kr, h, uint(slots[t]), d, p.kh, p.kt, p.kd);
                lo = min(lo, x);
                hi = max(hi, x);
            }
            bounds[(h * p.dim + d) * p.bounds_stride + c] = half2(lo, hi);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint d = tid; d < p.dim; d += threads) {
        float sum = 0, mag = 0;
        for (uint g = 0; g < nh; ++g) {
            sum += qg[g * p.dim + d];
            mag += abs(qg[g * p.dim + d]);
        }
        magnitude[d] = mag;
        sign[d] = sum >= 0 ? 1.0f : -1.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < p.dim; d += threads) {
        uint rank = 0;
        for (uint e = 0; e < p.dim; ++e) rank += better(magnitude[e], e, magnitude[d], d);
        order[rank] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg < nh) {
        float all = 0, kept = 0;
        for (uint d = lane; d < p.dim; d += 32) all += abs(qg[sg * p.dim + d]);
        for (uint r = lane; r < p.rank; r += 32) kept += abs(qg[sg * p.dim + order[r]]);
        all = simd_sum(all);
        kept = simd_sum(kept);
        if (lane == 0) scale[sg] = sqrt(float(p.dim) * (kept / (all + 1e-9f)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint c = begin + tid / 4;
    float dot[MAX_GROUP] = {};
    if (c < end) {
        for (uint r = tid % 4; r < p.rank; r += 4) {
            uint d = order[r];
            float rep;
            if (p.chunk == 1) {
                rep = read_kv(kr, h, uint(slots[c]), d, p.kh, p.kt, p.kd);
            } else {
                float2 extrema = float2(bounds[(h * p.dim + d) * p.bounds_stride + c]);
                rep = sign[d] >= 0 ? extrema.y : extrema.x;
            }
            for (uint g = 0; g < MAX_GROUP; ++g) {
                if (g < nh) dot[g] += qg[g * p.dim + d] * rep;
            }
        }
    }
    for (uint g = 0; g < nh; ++g) {
        dot[g] += simd_shuffle_xor(dot[g], 1);
        dot[g] += simd_shuffle_xor(dot[g], 2);
        if (c < end && tid % 4 == 0) chunks[(h * nh + g) * p.chunks + c] = scale[g] > 0 ? dot[g] / scale[g] : 0;
    }
    if (!last_arrival(error + 1 + h, blocks, tid, &last)) return;
    select_group(p, chunks, scores, indices, error, h, tid, threads, lane, sg, sgs, partial, stat, scratch, histogram);
}

// Exact attention over the selected tokens, split-K. Grid (blocks, KV heads); each simdgroup runs
// an online softmax over one slice of the head's selection for all query heads of the group. The
// simdgroups of a threadgroup merge in threadgroup memory into one partial per block, and the last
// block of each KV head merges the blocks.
kernel void attend(ARGS, constant uint &units [[buffer(16)]], uint tid [[thread_index_in_threadgroup]],
                   uint2 tg [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
                   uint2 shape [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint sgs [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint threads = shape.x;
    threadgroup float merged[MAX_GROUP * 256], stats[MAX_GROUP * 16], sums[MAX_GROUP * 16], weight[MAX_GROUP * 16],
        total[MAX_GROUP];
    threadgroup bool last;
    const uint h = tg.y, w = tg.x * sgs + sg, nh = p.heads / p.kv_heads, stride = p.dim + 2, blocks = grid.x;
    const uint per = (p.topk + units - 1) / units, begin = min(w * per, p.topk), end = min(begin + per, p.topk);
    const float scale = rsqrt(float(p.dim));
    float qv[MAX_GROUP][MAX_LANE_DIMS], acc[MAX_GROUP][MAX_LANE_DIMS], m[MAX_GROUP], l[MAX_GROUP];
    for (uint g = 0; g < MAX_GROUP; ++g) {
        m[g] = -INFINITY;
        l[g] = 0;
        for (uint j = 0; j < MAX_LANE_DIMS; ++j) {
            uint d = lane + j * 32;
            qv[g][j] = g < nh && d < p.dim ? q[(h * nh + g) * p.dim + d] * scale : 0;
            acc[g][j] = 0;
        }
    }
    // Lane j fetches the slot of token base + j; the simdgroup then walks the batch.
    for (uint base = begin; base < end; base += 32) {
        const uint batch = min(32u, end - base);
        const uint mine = lane < batch ? uint(slots[indices[h * p.padded + base + lane]]) : 0;
        for (uint i = 0; i < batch; ++i) {
            uint slot = simd_shuffle(mine, ushort(i));
            float kv[MAX_LANE_DIMS], vv[MAX_LANE_DIMS];
            for (uint j = 0; j < MAX_LANE_DIMS; ++j) {
                uint d = lane + j * 32;
                kv[j] = d < p.dim ? read_kv(kr, h, slot, d, p.kh, p.kt, p.kd) : 0;
                vv[j] = d < p.dim ? read_kv(vr, h, slot, d, p.vh, p.vt, p.vd) : 0;
            }
            for (uint g = 0; g < MAX_GROUP; ++g) {
                if (g >= nh) break;
                float dot = 0;
                for (uint j = 0; j < MAX_LANE_DIMS; ++j) dot += qv[g][j] * kv[j];
                dot = simd_sum(dot);
                float next = max(m[g], dot), keep = exp(m[g] - next), e = exp(dot - next);
                l[g] = l[g] * keep + e;
                for (uint j = 0; j < MAX_LANE_DIMS; ++j) acc[g][j] = acc[g][j] * keep + e * vv[j];
                m[g] = next;
            }
        }
    }
    // Merge the simdgroups of this block (an empty slice has m = -inf and weight 0).
    if (lane == 0) {
        for (uint g = 0; g < nh; ++g) {
            stats[(sg * MAX_GROUP + g) * 2] = m[g];
            stats[(sg * MAX_GROUP + g) * 2 + 1] = l[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < nh; ++g) {
        float top = -INFINITY, sum = 0;
        for (uint i = 0; i < sgs; ++i) top = max(top, stats[(i * MAX_GROUP + g) * 2]);
        for (uint i = 0; i < sgs; ++i) {
            float mi = stats[(i * MAX_GROUP + g) * 2];
            if (mi != -INFINITY) sum += exp(mi - top) * stats[(i * MAX_GROUP + g) * 2 + 1];
        }
        m[g] = top == -INFINITY ? 0.0f : exp(m[g] - top);  // this simdgroup's weight
        l[g] = top;
        if (sg == 0 && lane == 0) total[g] = sum;
    }
    for (uint i = 0; i < sgs; ++i) {
        if (sg == i) {
            for (uint g = 0; g < nh; ++g) {
                for (uint j = 0; j < MAX_LANE_DIMS; ++j) {
                    uint d = lane + j * 32;
                    if (d < p.dim) merged[g * p.dim + d] = (i ? merged[g * p.dim + d] : 0) + m[g] * acc[g][j];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < nh * (p.dim + 2); i += threads) {
        uint g = i / stride, d = i % stride;
        coherent(device) device float *part = partials + ((h * nh + g) * blocks + tg.x) * stride;
        part[d] = d == 0 ? l[g] : d == 1 ? total[g] : merged[g * p.dim + d - 2];
    }
    if (!last_arrival(error + 1 + p.kv_heads + h, blocks, tid, &last)) return;
    coherent(device) device const float *parts = partials + h * nh * blocks * stride;
    for (uint i = tid; i < nh * blocks; i += threads) {
        stats[i] = parts[i * stride];
        sums[i] = parts[i * stride + 1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < nh) {
        float top = -INFINITY, sum = 0;
        for (uint u = 0; u < blocks; ++u) top = max(top, stats[tid * blocks + u]);
        for (uint u = 0; u < blocks; ++u) {
            float keep = stats[tid * blocks + u] == -INFINITY ? 0.0f : exp(stats[tid * blocks + u] - top);
            weight[tid * blocks + u] = keep;
            sum += keep * sums[tid * blocks + u];
        }
        total[tid] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < nh * p.dim; i += threads) {
        uint g = i / p.dim, d = i % p.dim;
        float sum = 0;
        for (uint u = 0; u < blocks; ++u) sum += weight[g * blocks + u] * parts[(g * blocks + u) * stride + 2 + d];
        float out = sum / total[g];
        output[(h * nh + g) * p.dim + d] = out;
        if (!isfinite(out)) atomic_store_explicit(error, 1u, memory_order_relaxed);
    }
}
