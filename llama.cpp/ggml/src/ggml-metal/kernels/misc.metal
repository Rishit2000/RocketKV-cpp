#include "common.h"

kernel void kernel_argmax_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint  tgpig[[threadgroup_position_in_grid]],
        uint  tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint    ntg[[threads_per_threadgroup]]) {
    device const float * x_row = (device const float *) ((device const char *) src0 + tgpig * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = tpitg; i00 < args.ne00; i00 += ntg) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    device int32_t * dst_i32 = (device int32_t *) dst;

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        dst_i32[tgpig] = arg_val_reduced;

        return;
    }

    dst_i32[tgpig] = arg_val;
}

kernel void kernel_argmax_part_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * src0,
        device         char * tmp,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  ntg[[threads_per_threadgroup]]) {
    const int row  = tgpig.y;
    const int part = tgpig.x;
    const int ne00 = (int) args.ne00;
    const int npart = (int) args.npart;
    const int chunk = (ne00 + npart - 1) / npart;
    const int start = part * chunk;
    const int end   = min(start + chunk, ne00);

    device const float * x_row = (device const float *) ((device const char *) src0 + row * args.nb01);

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i00 = start + tpitg.x; i00 < end; i00 += ntg.x) {
        if (x_row[i00] > lmax) {
            lmax = x_row[i00];
            larg = i00;
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        max_val = max_val_reduced;
        arg_val = arg_val_reduced;
    }

    if (tpitg.x == 0) {
        device float * tmp_max = (device float *) tmp;
        device int32_t * tmp_arg = (device int32_t *) tmp + npart * (int) args.ne01;

        tmp_max[row*npart + part] = max_val;
        tmp_arg[row*npart + part] = arg_val;
    }
}

kernel void kernel_argmax_final_f32(
        constant ggml_metal_kargs_argmax & args,
        device   const char * tmp,
        device         char * dst,
        threadgroup    char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint3  ntg[[threads_per_threadgroup]]) {
    const int row = tgpig.x;
    const int npart = (int) args.npart;

    device const float * tmp_max = (device const float *) tmp;
    device const int32_t * tmp_arg = (device const int32_t *) tmp + npart * (int) args.ne01;

    float   lmax = -INFINITY;
    int32_t larg = -1;

    for (int i = tpitg.x; i < npart; i += ntg.x) {
        if (tmp_max[row*npart + i] > lmax) {
            lmax = tmp_max[row*npart + i];
            larg = tmp_arg[row*npart + i];
        }
    }

    // find the argmax value in the block
    float max_val = simd_max(lmax);
    int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));

    threadgroup   float * shared_maxval = (threadgroup   float *) shmem;
    threadgroup int32_t * shared_argmax = (threadgroup int32_t *) shmem + N_SIMDWIDTH;

    if (ntg.x > N_SIMDWIDTH) {
        if (sgitg == 0) {
            shared_maxval[tiisg] = -INFINITY;
            shared_argmax[tiisg] = -1;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            shared_maxval[sgitg] = max_val;
            shared_argmax[sgitg] = arg_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = shared_maxval[tiisg];
        arg_val = shared_argmax[tiisg];

        float max_val_reduced   = simd_max(max_val);
        int32_t arg_val_reduced = simd_max(select(-1, arg_val, max_val == max_val_reduced));

        max_val = max_val_reduced;
        arg_val = arg_val_reduced;
    }

    if (tpitg.x == 0) {
        ((device int32_t *) dst)[row] = arg_val;
    }
}

kernel void kernel_diag_f32(
        constant ggml_metal_kargs_diag & args,
        device   const char * src0,
        device         char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;

    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *)(src0 +                i2*args.nb02 + i3*args.nb03);
    device       float * dst_ptr  = (device       float *)(dst  + i1*args.nb01 + i2*args.nb2  + i3*args.nb3);

    for (int i0 = tiitg; i0 < args.ne0; i0 += NW) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}

kernel void kernel_roll_f32(
    constant ggml_metal_kargs_roll & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    device const float * src0_ptr = (device const float *) src0;
    device       float * dst_ptr  = (device       float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        // apply shifts and wrap around
        int64_t i00 = i0 - args.s0;
        int64_t i01 = i1 - args.s1;
        int64_t i02 = i2 - args.s2;
        int64_t i03 = i3 - args.s3;

        if (i00 < 0) { i00 += args.ne00; } else if (i00 >= args.ne00) { i00 -= args.ne00; }
        if (i01 < 0) { i01 += args.ne01; } else if (i01 >= args.ne01) { i01 -= args.ne01; }
        if (i02 < 0) { i02 += args.ne02; } else if (i02 >= args.ne02) { i02 -= args.ne02; }
        if (i03 < 0) { i03 += args.ne03; } else if (i03 >= args.ne03) { i03 -= args.ne03; }

        int64_t src_idx = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00 + i00;
        int64_t dst_idx = i3 *args.ne2 *args.ne1 *args.ne0  + i2 *args.ne1 *args.ne0  + i1 *args.ne0  + i0;

        dst_ptr[dst_idx] = src0_ptr[src_idx];
    }
}

constant bool FC_pad_circular [[function_constant(FC_PAD + 0)]];

// circular means on a torus, so the coordinates wrap around
static inline int32_t wrap_around(int32_t coord, int32_t size) {
    return (coord + size) % size;
}

kernel void kernel_pad_f32(
    constant ggml_metal_kargs_pad & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {
    const int32_t i3 = tgpig.z;
    const int32_t i2 = tgpig.y;
    const int32_t k0 = tgpig.x/args.ne1;
    const int32_t i1 = tgpig.x - k0*args.ne1;

    const int32_t ne00 = args.ne00;
    const int32_t ne01 = args.ne01;
    const int32_t ne02 = args.ne02;
    const int32_t ne03 = args.ne03;

    int32_t i01 = i1 - args.lp1;
    int32_t i02 = i2 - args.lp2;
    int32_t i03 = i3 - args.lp3;

    if (FC_pad_circular) {
        i01 = wrap_around(i01, ne01);
        i02 = wrap_around(i02, ne02);
        i03 = wrap_around(i03, ne03);
    }

    device float * dst_ptr = (device float *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1);

    // the row lies in the padded region, so no source row backs it
    if (i01 < 0 || i01 >= ne01 ||
        i02 < 0 || i02 >= ne02 ||
        i03 < 0 || i03 >= ne03) {
        for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
            const int32_t i0 = k0*1024 + tpitg.x + l0;
            if (i0 >= args.ne0) {
                break;
            }

            dst_ptr[i0] = 0.0f;
        }

        return;
    }

    device const char * src0_row = src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01;

    for (int32_t l0 = 0; l0 < 1024; l0 += ntg.x) {
        const int32_t i0 = k0*1024 + tpitg.x + l0;
        if (i0 >= args.ne0) {
            break;
        }

        int32_t i00 = i0 - args.lp0;

        if (FC_pad_circular) {
            i00 = wrap_around(i00, ne00);
        }

        dst_ptr[i0] = i00 >= 0 && i00 < ne00 ? *((device const float *) (src0_row + i00*args.nb00)) : 0.0f;
    }
}

// TODO: this is slow - optimize
kernel void kernel_pad_reflect_1d_f32(
    constant   ggml_metal_kargs_pad_reflect_1d & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3  tgpg[[threadgroups_per_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const int64_t i3 = tgpig.z;
    const int64_t i2 = tgpig.y;
    const int64_t i1 = tgpig.x;

    const int64_t i03 = i3;
    const int64_t i02 = i2;
    const int64_t i01 = i1;

    device const float * src0_ptr = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       float * dst_ptr  = (device       float *) (dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1);

    if (i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
        for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
            if (i0 < args.p0) {
                dst_ptr[i0] = src0_ptr[args.p0 - i0];
            } else if (i0 < args.ne0 - args.p1) {
                dst_ptr[i0] = src0_ptr[i0 - args.p0];
            } else {
                dst_ptr[i0] = src0_ptr[(args.ne0 - args.p1 - args.p0) - (args.p1 + 1 - (args.ne0 - i0)) - 1];
            }
        }
    }
}

kernel void kernel_arange_f32(
    constant   ggml_metal_kargs_arange & args,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    device float * dst_ptr = (device float *) dst;

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_ptr[i0] = args.start + args.step * i0;
    }
}

kernel void kernel_timestep_embedding_f32(
    constant  ggml_metal_kargs_timestep_embedding & args,
    device  const char * src0,
    device        char * dst,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    int i = tgpig.x;
    device float * embed_data = (device float *)(dst + i*args.nb1);

    int half_ = args.dim / 2;
    for (int j = tpitg.x; j < half_; j += ntg.x) {
        float timestep = ((device float *)src0)[i];
        float freq = (float)exp(-log((float)args.max_period) * j / half_);
        float arg = timestep * freq;
        embed_data[j        ] = cos(arg);
        embed_data[j + half_] = sin(arg);
    }

    if (args.dim % 2 != 0 && tpitg.x == 0) {
        embed_data[2 * half_] = 0.f;
    }
}

kernel void kernel_opt_step_adamw_f32(
        constant    ggml_metal_kargs_opt_step_adamw & args,
        device       float * x,
        device const float * g,
        device       float * g_m,
        device       float * g_v,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    const float alpha  = pars[0];
    const float beta1  = pars[1];
    const float beta2  = pars[2];
    const float eps    = pars[3];
    const float wd     = pars[4];
    const float beta1h = pars[5];
    const float beta2h = pars[6];

    const float gi = g[gid];
    const float gmi = g_m[gid] * beta1 +      gi * (1.0f - beta1);
    const float gvi = g_v[gid] * beta2 + gi * gi * (1.0f - beta2);

    g_m[gid] = gmi;
    g_v[gid] = gvi;

    const float mh =      gmi * beta1h;
    const float vh = sqrt(gvi * beta2h) + eps;

    x[gid] = x[gid] * (1.0f - alpha * wd) - alpha * mh / vh;
}

kernel void kernel_opt_step_sgd_f32(
        constant    ggml_metal_kargs_opt_step_sgd & args,
        device       float * x,
        device const float * g,
        device const float * pars,
        uint        gid[[thread_position_in_grid]]) {

    if (gid >= args.np) {
        return;
    }

    x[gid] = x[gid] * (1.0f - pars[0] * pars[1]) - pars[0] * g[gid];
}

template<typename T>
kernel void kernel_memset(
        constant ggml_metal_kargs_memset & args,
        device T * dst,
        uint tpig[[thread_position_in_grid]]) {
    dst[tpig] = args.val;
}

typedef decltype(kernel_memset<int64_t>) kernel_memset_t;

template [[host_name("kernel_memset_i64")]] kernel kernel_memset_t kernel_memset<int64_t>;

constant short FC_count_equal_nsg [[function_constant(FC_COUNT_EQUAL + 0)]];

template<typename T>
kernel void kernel_count_equal(
        constant ggml_metal_kargs_count_equal & args,
        device   const char * src0,
        device   const char * src1,
        device   atomic_int * dst,
        threadgroup int32_t * shmem_i32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const short NSG = FC_count_equal_nsg;

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = tgpig.x;

    if (i3 >= args.ne03 || i2 >= args.ne02 || i1 >= args.ne01) {
        return;
    }

    int sum = 0;

    device const char * base0 = src0 + i1*args.nb01 + i2*args.nb02 + i3*args.nb03;
    device const char * base1 = src1 + i1*args.nb11 + i2*args.nb12 + i3*args.nb13;

    for (int64_t i0 = tpitg.x; i0 < args.ne00; i0 += ntg.x) {
        const T v0 = *(device const T *)(base0 + i0*args.nb00);
        const T v1 = *(device const T *)(base1 + i0*args.nb10);
        sum += (v0 == v1);
    }

    sum = simd_sum(sum);

    if (tiisg == 0) {
        shmem_i32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        float v = 0.0f;
        if (tpitg.x < NSG) {
            v = shmem_i32[tpitg.x];
        }

        float total = simd_sum(v);
        if (tpitg.x == 0) {
            atomic_fetch_add_explicit(dst, (int32_t) total, memory_order_relaxed);
        }
    }
}

typedef decltype(kernel_count_equal<int32_t>) kernel_count_equal_t;

template [[host_name("kernel_count_equal_i32")]] kernel kernel_count_equal_t kernel_count_equal<int32_t>;

template <typename T>
kernel void kernel_snake(
        constant ggml_metal_kargs_snake & args,
        device const T     * x,
        device const float * a,
        device const float * inv_b,
        device       T     * dst,
        uint         tgpig [[threadgroup_position_in_grid]],
        uint         tpitg [[thread_position_in_threadgroup]],
        uint         ntg   [[threads_per_threadgroup]]) {

    const int idx = tgpig * ntg + tpitg;
    if (idx >= args.T * args.C) {
        return;
    }

    const int   c  = idx / args.T;  // x is [T, C], a / inv_b collapse to [1, C]
    const float xi = float(x[idx]);
    const float si = sin(a[c] * xi);
    dst[idx] = T(xi + si * si * inv_b[c]);
}

template [[host_name("kernel_snake_f32")]]  kernel void kernel_snake<float>(constant ggml_metal_kargs_snake &, device const float *, device const float *, device const float *, device float *, uint, uint, uint);
template [[host_name("kernel_snake_f16")]]  kernel void kernel_snake<half>(constant ggml_metal_kargs_snake &, device const half *, device const float *, device const float *, device half *, uint, uint, uint);
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_snake_bf16")]] kernel void kernel_snake<bfloat>(constant ggml_metal_kargs_snake &, device const bfloat *, device const float *, device const float *, device bfloat *, uint, uint, uint);
#endif

template<int N, typename src_t>
kernel void kernel_fwht(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NW;

    const float scale = 1.0f / sqrt((float) N);

    const int sg_per_tg = ntg.x / NW;
    const int64_t r = tgpig.x * sg_per_tg + sgitg;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int lane = tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NW + lane])*scale;
    }
    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = val2 - val + 2*((lane & i) == 0)*val;
        }
    }

    for (int i = NW; i < N; i *= 2) {
        const int step = i / NW;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NW + lane] = reg[i];
    }
}

// Wide blocks: one row per threadgroup instead of per simdgroup, so each thread keeps
// N/NT values rather than N/32. Butterflies below the simdgroup width still shuffle;
// those up to NT go through threadgroup memory; the rest stay in registers.
// TODO: try avoiding branch https://github.com/ggml-org/llama.cpp/pull/29094#discussion_r4049563223
// TODO: try to unroll loops
template<int N, int NT, typename src_t>
kernel void kernel_fwht_tg(
        constant ggml_metal_kargs_fwht & args,
        device const src_t * src,
        device float * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort3  ntg[[threads_per_threadgroup]]) {

    constexpr int NW = N_SIMDWIDTH;
    constexpr int NE = N / NT;

    threadgroup float shmem[N];

    const float scale = 1.0f / sqrt((float) N);

    const int64_t r = tgpig.x;
    if (r >= args.nrows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const int tid = sgitg * NW + tiisg;

    float reg[NE];
    for (int i = 0; i < NE; i++) {
        reg[i] = float(src[i*NT + tid])*scale;
    }

    for (int i = 1; i < NW; i *= 2) {
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = simd_shuffle_xor(val, i);
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
    }

    for (int i = NW; i < NT; i *= 2) {
        for (int j = 0; j < NE; j++) {
            shmem[j*NT + tid] = reg[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = shmem[j*NT + (tid ^ i)];
            reg[j] = (tid & i) == 0 ? val2 + val : val2 - val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int i = NT; i < N; i *= 2) {
        const int step = i / NT;
        for (int j = 0; j < NE; j += (2 * step)) {
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k ];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    for (int i = 0; i < NE; i++) {
        dst[i*NT + tid] = reg[i];
    }
}

typedef decltype(kernel_fwht<64, float>) kernel_fwht_f32_t;
typedef decltype(kernel_fwht<64, half>)  kernel_fwht_f16_t;

template [[host_name("kernel_fwht_f32_64")]]  kernel kernel_fwht_f32_t kernel_fwht<64,  float>;
template [[host_name("kernel_fwht_f32_128")]] kernel kernel_fwht_f32_t kernel_fwht<128, float>;
template [[host_name("kernel_fwht_f32_256")]] kernel kernel_fwht_f32_t kernel_fwht<256, float>;

template [[host_name("kernel_fwht_f16_64")]]  kernel kernel_fwht_f16_t kernel_fwht<64,  half>;
template [[host_name("kernel_fwht_f16_128")]] kernel kernel_fwht_f16_t kernel_fwht<128, half>;
template [[host_name("kernel_fwht_f16_256")]] kernel kernel_fwht_f16_t kernel_fwht<256, half>;

template [[host_name("kernel_fwht_f32_512")]]  kernel kernel_fwht_f32_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_1024")]] kernel kernel_fwht_f32_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_2048")]] kernel kernel_fwht_f32_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_4096")]] kernel kernel_fwht_f32_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, float>;
template [[host_name("kernel_fwht_f32_8192")]] kernel kernel_fwht_f32_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, float>;

template [[host_name("kernel_fwht_f16_512")]]  kernel kernel_fwht_f16_t kernel_fwht_tg<512,  GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_1024")]] kernel kernel_fwht_f16_t kernel_fwht_tg<1024, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_2048")]] kernel kernel_fwht_f16_t kernel_fwht_tg<2048, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_4096")]] kernel kernel_fwht_f16_t kernel_fwht_tg<4096, GGML_METAL_FWHT_TG_NT, half>;
template [[host_name("kernel_fwht_f16_8192")]] kernel kernel_fwht_f16_t kernel_fwht_tg<8192, GGML_METAL_FWHT_TG_NT, half>;

constant int FC_dsv4_hc_n_hc [[function_constant(FC_DSV4_HC + 0)]];

kernel void kernel_dsv4_hc_comb_f32(
        constant ggml_metal_kargs_dsv4_hc_comb & args,
        device const char * mixes,
        device const char * scale,
        device const char * base,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;
    constexpr ushort comb_offset = 2*hc;

    const int it = tgpig.x*ntg.y + sgitg;
    if (it >= args.n_tokens) {
        return;
    }

    float scale_lane = 0.0f;
    if (tiisg == 0) {
        scale_lane = *(device const float *) (scale + 2*args.nb_s0);
    }
    const float scale_comb = simd_shuffle(scale_lane, 0);

    float v = 0.0f;
    if (tiisg < hc*hc) {
        v = *(device const float *) (mixes + (comb_offset + tiisg)*args.nb_m0 + it*args.nb_m1)*scale_comb
          + *(device const float *) (base   + (comb_offset + tiisg)*args.nb_b0);
    }

    // Softmax across destinations (the four contiguous lanes for each source).
    float vmax = max(v, simd_shuffle_xor(v, 1));
    vmax = max(vmax, simd_shuffle_xor(vmax, 2));
    v = exp(v - vmax);

    float sum = v + simd_shuffle_xor(v, 1);
    sum += simd_shuffle_xor(sum, 2);
    v = v/sum + args.eps;

    // Normalize columns: equal destination indices are four lanes apart.
    sum = v + simd_shuffle_xor(v, 4);
    sum += simd_shuffle_xor(sum, 8);
    v /= sum + args.eps;

    for (int i = 1; i < args.n_iter; ++i) {
        sum = v + simd_shuffle_xor(v, 1);
        sum += simd_shuffle_xor(sum, 2);
        v /= sum + args.eps;

        sum = v + simd_shuffle_xor(v, 4);
        sum += simd_shuffle_xor(sum, 8);
        v /= sum + args.eps;
    }

    if (tiisg < hc*hc) {
        const ushort idst = tiisg & 3;
        const ushort isrc = tiisg >> 2;
        *(device float *) (dst + idst*args.nb_d0 + isrc*args.nb_d1 + it*args.nb_d2) = v;
    }
}

kernel void kernel_dsv4_hc_pre_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * weights,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x + i0*args.nb_x0 + it*args.nb_x2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        const float wv = *(device const float *) (weights + ih*args.nb_w0 + it*args.nb_w1);
        result = fma(xv, wv, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_pre_gated_f32(
        constant ggml_metal_kargs_dsv4_hc_pre & args,
        device const char * x,
        device const char * gate,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    if (i0 >= args.n_embd) {
        return;
    }

    device const char * xb = x    + i0*args.nb_x0 + it*args.nb_x2;
    device const char * gb = gate + i0*args.nb_w0 + it*args.nb_w2;
    float result = 0.0f;
    FOR_UNROLL (int ih = 0; ih < FC_dsv4_hc_n_hc; ++ih) {
        const float g  = 1.0f/(1.0f + exp(-*(device const float *) (gb + ih*args.nb_w1)));
        const float xv = *(device const float *) (xb + ih*args.nb_x1);
        result = fma(xv, g, result);
    }

    *(device float *) (dst + i0*args.nb_d0 + it*args.nb_d1) = args.scale*result;
}

kernel void kernel_dsv4_hc_post_nocomb_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float post_lane = 0.0f;
    if (tiisg < hc) {
        post_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    }

    float post_reg[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(post_lane, idst);
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        const float rv = *(device const float *) (rb + idst*args.nb_r1);
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = xv*post_reg[idst] + rv;
    }
}

kernel void kernel_dsv4_hc_post_f32(
        constant ggml_metal_kargs_dsv4_hc_post & args,
        device const char * x,
        device const char * residual,
        device const char * post,
        device const char * comb,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    constexpr ushort hc = 4;

    const int it = tgpig.y;
    const int i0 = ((int) tgpig.x*ntg.y + sgitg)*32 + tiisg;

    float coeff_lane = 0.0f;
    if (tiisg < hc) {
        coeff_lane = *(device const float *) (post + tiisg*args.nb_p0 + it*args.nb_p1);
    } else if (tiisg < hc + hc*hc) {
        const ushort idx  = tiisg - hc;
        const ushort idst = idx & 3;
        const ushort isrc = idx >> 2;
        coeff_lane = *(device const float *) (comb + idst*args.nb_c0 + isrc*args.nb_c1 + it*args.nb_c2);
    }

    float post_reg[hc];
    float comb_reg[hc][hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        post_reg[idst] = simd_shuffle(coeff_lane, idst);
    }
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            comb_reg[isrc][idst] = simd_shuffle(coeff_lane, hc + idst + hc*isrc);
        }
    }

    if (i0 >= args.n_embd) {
        return;
    }

    const float xv = *(device const float *) (x + i0*args.nb_x0 + it*args.nb_x1);
    float result[hc];
    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        result[idst] = xv*post_reg[idst];
    }

    device const char * rb = residual + i0*args.nb_r0 + it*args.nb_r2;
    FOR_UNROLL (ushort isrc = 0; isrc < hc; ++isrc) {
        const float rv = *(device const float *) (rb + isrc*args.nb_r1);
        FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
            result[idst] = fma(rv, comb_reg[isrc][idst], result[idst]);
        }
    }

    FOR_UNROLL (ushort idst = 0; idst < hc; ++idst) {
        *(device float *) (dst + i0*args.nb_d0 + idst*args.nb_d1 + it*args.nb_d2) = result[idst];
    }
}

constant bool FC_top_k_mask_has_score  [[function_constant(FC_TOP_K_MASK + 0)]];
constant bool FC_top_k_mask_has_blocks [[function_constant(FC_TOP_K_MASK + 1)]];
constant bool FC_top_k_mask_has_extra  [[function_constant(FC_TOP_K_MASK + 2)]];
constant bool FC_top_k_mask_vec        [[function_constant(FC_TOP_K_MASK + 3)]];

// the fill runs before the scatter, both are fully parallel over the whole dst and over the picks of one row
template <typename T, typename T4>
kernel void kernel_top_k_mask_fill(
        constant ggml_metal_kargs_top_k_mask & args,
        device char * dst,
        uint  tid[[thread_position_in_grid]],
        uint ntot[[threads_per_grid]]) {
    const T ninf = T(-INFINITY);

    if (FC_top_k_mask_vec) {
        device T4 * dst_4 = (device T4 *) dst;
        for (int64_t i = tid; i < args.n_elems/4; i += ntot) {
            dst_4[i] = T4(ninf);
        }
    } else {
        device T * dst_t = (device T *) dst;
        for (int64_t i = tid; i < args.n_elems; i += ntot) {
            dst_t[i] = ninf;
        }
    }
}

typedef decltype(kernel_top_k_mask_fill<half,  half4>)  kernel_top_k_mask_fill_h_t;
typedef decltype(kernel_top_k_mask_fill<float, float4>) kernel_top_k_mask_fill_f_t;

template [[host_name("kernel_top_k_mask_fill_f16")]] kernel kernel_top_k_mask_fill_h_t kernel_top_k_mask_fill<half,  half4>;
template [[host_name("kernel_top_k_mask_fill_f32")]] kernel kernel_top_k_mask_fill_f_t kernel_top_k_mask_fill<float, float4>;

// one threadgroup per row: copy the mask cells of the live picks and of the extra cells over the fill
template <typename T, typename T4>
kernel void kernel_top_k_mask(
        constant ggml_metal_kargs_top_k_mask & args,
        device const  char * mask,
        device const  char * top_k,
        device const  char * score,
        device const  char * blocks,
        device const  char * extra,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        uint3   tpitg[[thread_position_in_threadgroup]],
        uint3 ntg_max[[threads_per_threadgroup]]) {
    const int i1 = tgpig.x;
    const int i2 = tgpig.y;
    const int i3 = tgpig.z;

    const int   tiitg = tpitg.x;
    const int     ntg = ntg_max.x;

    device const char * mask_row  = mask  + (i1 % args.m1)*args.nbm1 + (i2 % args.m2)*args.nbm2 + (i3 % args.m3)*args.nbm3;
    device       char * dst_row   = dst   + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;
    device const char * top_k_row = top_k + i1*args.nbk1 + i2*args.nbk2 + i3*args.nbk3;

    device const T * mask_row_t = (device const T *) mask_row;
    device       T * dst_row_t  = (device       T *) dst_row;

    for (int i = tiitg; i < args.n_top; i += ntg) {
        const int32_t p = *(device const int32_t *) (top_k_row + i*args.nbk0);
        if (p < 0 || p >= args.n_pick) {
            continue;
        }
        if (FC_top_k_mask_has_score && !(*(device const float *) (score + p*args.nbs0 + i1*args.nbs1 + i2*args.nbs2 + i3*args.nbs3) > -INFINITY)) {
            continue;
        }
        if (FC_top_k_mask_has_blocks) {
            FOR_UNROLL (int k = 0; k < args.n_blk; ++k) {
                const int32_t c = *(device const int32_t *) (blocks + k*args.nbb0 + p*args.nbb1);
                if (c >= 0 && c < args.n_kv) {
                    dst_row_t[c] = mask_row_t[c];
                }
            }
        } else if (p < args.n_kv) {
            dst_row_t[p] = mask_row_t[p];
        }
    }

    if (FC_top_k_mask_has_extra) {
        for (int i = tiitg; i < args.n_extra; i += ntg) {
            const int32_t c = *(device const int32_t *) (extra + i*args.nbe0 + i1*args.nbe1 + i2*args.nbe2 + i3*args.nbe3);
            if (c >= 0 && c < args.n_kv) {
                dst_row_t[c] = mask_row_t[c];
            }
        }
    }
}

typedef decltype(kernel_top_k_mask<half,  half4>)  kernel_top_k_mask_h_t;
typedef decltype(kernel_top_k_mask<float, float4>) kernel_top_k_mask_f_t;

template [[host_name("kernel_top_k_mask_f16")]] kernel kernel_top_k_mask_h_t kernel_top_k_mask<half,  half4>;
template [[host_name("kernel_top_k_mask_f32")]] kernel kernel_top_k_mask_f_t kernel_top_k_mask<float, float4>;

// All HSA kernels keep selection on the GPU; no last-layer or previous-step state.
inline float hsa_sum(float v, threadgroup float * scratch, uint lane, uint sg) {
    float sum = simd_sum(v);
    if (lane == 0) scratch[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sum = simd_sum(lane < 8 ? scratch[lane] : 0.f);
        if (lane == 0) scratch[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void kernel_hsa_compact(
        constant ggml_metal_kargs_hsa & a,
        device const float * mask,
        device int * order,
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint offsets[8];
    threadgroup uint count, base;
    threadgroup int invalid;
    if (tid == 0) { count = 0; invalid = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int start = 0; start < a.N; start += 256) {
        int t = start + tid;
        uint valid = t < a.N && isfinite(mask[t]);
        uint prefix = simd_prefix_exclusive_sum(valid);
        uint total = simd_sum(valid);
        if (lane == 0) offsets[sg] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            base = count;
            uint n = 0;
            for (int s = 0; s < 8; ++s) { uint v = offsets[s]; offsets[s] = n; n += v; }
            count += n;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) order[base + offsets[sg] + prefix] = t;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0 && count < uint(a.N)) {
        for (int t = 0; t < a.N; ++t) if (!isfinite(mask[t])) { invalid = t; break; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = count + tid; t < uint(a.N); t += 256) order[t] = invalid;
}

kernel void kernel_hsa_query(
        constant ggml_metal_kargs_hsa & a,
        device const float * q,
        device float * out,
        uint kv [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float importance[256], sign[256];
    if (tid < uint(a.D)) {
        float imp = 0, sum = 0;
        for (int h = 0; h < a.H; ++h) { float v = q[(kv*a.H+h)*a.D+tid]; imp += abs(v); sum += v; }
        importance[tid] = imp;
        sign[tid] = sum > 0 ? 1.f : -1.f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < uint(a.D)) {
        int rank = 0;
        for (int j = 0; j < a.D; ++j) rank += importance[j] > importance[tid] || (importance[j] == importance[tid] && j < int(tid));
        if (rank >= a.r) sign[tid] = 0.f;
        out[(kv*(a.H+1)+a.H)*a.D+tid] = sign[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Each SIMD group normalizes one query head independently.
    for (uint h = sg; h < uint(a.H); h += 8) {
        float total = 0.f, kept = 0.f;
        for (uint d = lane; d < uint(a.D); d += 32) {
            float value = abs(q[(kv*a.H+h)*a.D+d]);
            total += value;
            if (sign[d] != 0.f) kept += value;
        }
        total = simd_sum(total);
        kept = simd_sum(kept);
        float scale = sqrt(a.D*kept/total);
        for (uint d = lane; d < uint(a.D); d += 32) out[(kv*(a.H+1)+h)*a.D+d] = q[(kv*a.H+h)*a.D+d]*sign[d]/scale;
    }

}

template <ushort NH>
kernel void kernel_hsa_logits(
        constant ggml_metal_kargs_hsa & a,
        device const float * query,
        device const char * k,
        device const int * order,
        device const float * mask,
        device float * out,
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const int H = NH == 0 ? a.H : NH;
    // Four independent SIMD groups score four chunks per threadgroup.
    int kv = group.y, start = (group.x*4+sg)*a.chunk, end = min(a.N,start+a.chunk);
    if (start >= a.N) return;
    int C=(a.N+a.chunk-1)/a.chunk, c=start/a.chunk;
    if (lane==0) {
        int count=0;
        for(int t=start;t<end;++t) count+=isfinite(mask[order[t]]);
        out[(kv*(a.H+1)+a.H)*C+c]=count;
    }
    if (!isfinite(mask[order[start]])) {
        if(lane==0) for (int h=0;h<H;++h) out[(kv*(a.H+1)+h)*C+c]=-INFINITY;
        return;
    }
    float dots[NH == 0 ? 32 : NH];
    for (int h = 0; h < H; ++h) dots[h] = 0.f;
    if (a.D % 4 == 0) {
        // Adjacent dimensions are contiguous; vector loads avoid sparse scalar reads.
        for (int d = 4*lane; d < a.D; d += 128) {
            float4 sign = *(device const float4 *)(query+(kv*(a.H+1)+a.H)*a.D+d);
            float4 mx = float4(-INFINITY);
            for (int t = start; t < end; ++t) {
                int row = order[t];
                if (!isfinite(mask[row])) continue;
                device const char * ptr = k+row*a.nb2+kv*a.nb1+d*a.nb0;
                float4 value = a.k_f16 ? float4(*(device const half4 *)ptr) : *(device const float4 *)ptr;
                mx = max(mx,sign*value);
            }
            for (int h = 0; h < H; ++h) dots[h] += dot(mx,*(device const float4 *)(query+(kv*(a.H+1)+h)*a.D+d));
        }
    } else {
    for (int d = lane; d < a.D; d += 32) {
        float sign = query[(kv*(a.H+1)+a.H)*a.D+d];
        if (sign == 0.f) continue;
        float mx = -INFINITY;
        for (int t = start; t < end; ++t) {
            int row = order[t];
            if (!isfinite(mask[row])) continue;
            device const char * ptr = k + row*a.nb2 + kv*a.nb1 + d*a.nb0;
            float value = a.k_f16 ? float(*(device const half *)ptr) : *(device const float *)ptr;
            mx = max(mx,sign*value);
        }
        for (int h = 0; h < H; ++h) dots[h] += mx*query[(kv*(a.H+1)+h)*a.D+d];
    }
    }
    for (int h = 0; h < H; ++h) {
        float dot = simd_sum(dots[h]);
        if(lane==0) out[(kv*(a.H+1)+h)*C+c]=dot;
    }
    (void)tid;
}

typedef decltype(kernel_hsa_logits<0>) kernel_hsa_logits_t;
template [[host_name("kernel_hsa_logits_h0")]] kernel kernel_hsa_logits_t kernel_hsa_logits<0>;
template [[host_name("kernel_hsa_logits_h1")]] kernel kernel_hsa_logits_t kernel_hsa_logits<1>;
template [[host_name("kernel_hsa_logits_h2")]] kernel kernel_hsa_logits_t kernel_hsa_logits<2>;
template [[host_name("kernel_hsa_logits_h3")]] kernel kernel_hsa_logits_t kernel_hsa_logits<3>;
template [[host_name("kernel_hsa_logits_h4")]] kernel kernel_hsa_logits_t kernel_hsa_logits<4>;
template [[host_name("kernel_hsa_logits_h8")]] kernel kernel_hsa_logits_t kernel_hsa_logits<8>;
template [[host_name("kernel_hsa_logits_h16")]] kernel kernel_hsa_logits_t kernel_hsa_logits<16>;
template [[host_name("kernel_hsa_logits_h32")]] kernel kernel_hsa_logits_t kernel_hsa_logits<32>;

inline float hsa_max(float v, threadgroup float * scratch, uint lane, uint sg) {
    float value = simd_max(v);
    if (lane == 0) scratch[sg] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        value = simd_max(lane < 8 ? scratch[lane] : -INFINITY);
        if (lane == 0) scratch[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float result = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

kernel void kernel_hsa_scores(
        constant ggml_metal_kargs_hsa & a,
        device const float * logits,
        device float * out,
        uint kv [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    device const float * counts=logits+(kv*(a.H+1)+a.H)*a.N;
    for(int c=tid;c<a.N;c+=256) out[kv*a.D+c*a.chunk]=0.f;
    for(int h=0;h<a.H;++h) {
        device const float * row=logits+(kv*(a.H+1)+h)*a.N;
        float mx=-INFINITY;
        for(int c=tid;c<a.N;c+=256) mx=max(mx,row[c]);
        mx=hsa_max(mx,scratch,lane,sg);
        float sum=0.f;
        for(int c=tid;c<a.N;c+=256) sum+=exp(row[c]-mx)*counts[c];
        sum=hsa_sum(sum,scratch,lane,sg);
        for(int c=tid;c<a.N;c+=256) out[kv*a.D+c*a.chunk]+=exp(row[c]-mx)/(sum*a.H);
    }
    for(int c=tid;c<a.N;c+=256) {
        float value=out[kv*a.D+c*a.chunk];
        for(int j=0;j<a.chunk && c*a.chunk+j<a.D;++j) out[kv*a.D+c*a.chunk+j]=j < int(counts[c]) ? value : -INFINITY;
    }
}

kernel void kernel_hsa_gather(
        constant ggml_metal_kargs_hsa & a,
        device const char * k,
        device const char * v,
        device const int * order,
        device const int * rows,
        device const float * mask,
        device half * out,
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    int i = group.y*a.N+group.x, row = order[rows[group.y*a.H+group.x]];
    int block = a.D*a.N*a.Hkv;
    if (tid == 0) out[2*block+i] = half(mask[row]);
    for (int d = tid; d < a.D; d += 64) {
        if (!isfinite(mask[row])) {
            out[i*a.D+d] = half(0.f);
            out[block+i*a.D+d] = half(0.f);
            continue;
        }
        device const char * pk = k + row*a.nb2 + group.y*a.nb1 + d*a.nb0;
        device const char * pv = v + row*a.vb2 + group.y*a.vb1 + d*a.vb0;
        out[i*a.D+d] = a.k_f16 ? *(device const half *)pk : half(*(device const float *)pk);
        out[block+i*a.D+d] = a.v_f16 ? *(device const half *)pv : half(*(device const float *)pv);
    }
}


// Weighted radix selection treats a chunk as count identical token scores.
// Output starts with C scratch floats followed by keep compact row indices.
kernel void kernel_hsa_select(
        constant ggml_metal_kargs_hsa & a,
        device const float * logits,
        device int * packed,
        uint kv [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scratch[8];
    threadgroup atomic_uint histogram[256], emitted;
    threadgroup uint bucket, above, n_above;
    device float * scores=(device float *)(packed+kv*(a.N+a.D));
    device int * rows=packed+kv*(a.N+a.D)+a.N;
    device const float * counts=logits+(kv*(a.H+1)+a.H)*a.N;
    float n_live=0.f;
    for(int c=tid;c<a.N;c+=256) { scores[c]=0.f; n_live+=counts[c]; }
    uint k=min(uint(a.D),uint(hsa_sum(n_live,scratch,lane,sg)));
    if(k==0) {
        for(int i=tid;i<a.D;i+=256) rows[i]=a.r-1;
        return;
    }
    for(int h=0;h<a.H;++h) {
        device const float * row=logits+(kv*(a.H+1)+h)*a.N;
        float mx=-INFINITY;
        for(int c=tid;c<a.N;c+=256) mx=max(mx,row[c]);
        mx=hsa_max(mx,scratch,lane,sg);
        float sum=0.f;
        for(int c=tid;c<a.N;c+=256) sum+=exp(row[c]-mx)*counts[c];
        sum=hsa_sum(sum,scratch,lane,sg);
        for(int c=tid;c<a.N;c+=256) scores[c]+=exp(row[c]-mx)/(sum*a.H);
    }
    threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
    uint prefix=0,desired=k;
    for(int shift=24;shift>=0;shift-=8) {
        atomic_store_explicit(&histogram[tid],0u,memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint hi_mask=shift==24 ? 0u : (0xFFFFFFFFu << uint(shift+8));
        for(int c=tid;c<a.N;c+=256) {
            uint key=as_type<uint>(scores[c]),count=uint(counts[c]);
            if(count && (key & hi_mask)==(prefix & hi_mask)) atomic_fetch_add_explicit(&histogram[(key>>uint(shift)) & 255u],count,memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid<32) {
            uint top=255-8*tid,sum=0,values[8];
            for(uint j=0;j<8;++j) { values[j]=atomic_load_explicit(&histogram[top-j],memory_order_relaxed); sum+=values[j]; }
            uint acc=simd_prefix_exclusive_sum(sum);
            const simd_vote::vote_t hit=simd_vote::vote_t(simd_ballot(acc+sum>=desired));
            if(tid==uint(ctz(hit))) {
                uint b=top;
                for(uint j=0;j<8;++j) {
                    if(acc+values[j]>=desired) { b=top-j; break; }
                    acc+=values[j];
                }
                bucket=b; above=acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix|=bucket<<uint(shift);
        desired-=above;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0) atomic_store_explicit(&emitted,0u,memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int c=tid;c<a.N;c+=256) {
        uint count=uint(counts[c]);
        if(count && as_type<uint>(scores[c])>prefix) {
            uint start=atomic_fetch_add_explicit(&emitted,count,memory_order_relaxed);
            for(uint j=0;j<count;++j) rows[start+j]=c*a.chunk+j;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0) { n_above=atomic_load_explicit(&emitted,memory_order_relaxed); atomic_store_explicit(&emitted,0u,memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(int c=tid;c<a.N;c+=256) {
        uint count=uint(counts[c]);
        if(count && as_type<uint>(scores[c])==prefix) {
            uint start=n_above+atomic_fetch_add_explicit(&emitted,count,memory_order_relaxed);
            for(uint j=0;j<count && start+j<k;++j) rows[start+j]=c*a.chunk+j;
        }
    }
    for(uint i=k+tid;i<uint(a.D);i+=256) rows[i]=a.r-1;
}


kernel void kernel_hsa_bounds(
        constant ggml_metal_kargs_hsa & a,
        device const char * previous,
        device char * minimum,
        device const char * keys,
        device const int * order,
        device const float * mask,
        device char * maximum,
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int live;
    if(tid==0) {
        int n=a.N;
        while(n>0 && !isfinite(mask[order[n-1]])) --n;
        live=n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(live==0) return;
    int c=a.r ? (live-1)/a.chunk : int(group.x), kv=group.y;
    for(int d=tid;d<a.D;d+=256) {
        float hi=-INFINITY,lo=INFINITY;
        for(int t=c*a.chunk;t<min(live,(c+1)*a.chunk);++t) {
            device const char * ptr=keys+order[t]*a.nb2+kv*a.nb1+d*a.nb0;
            float value=a.k_f16 ? float(*(device const half *)ptr) : *(device const float *)ptr;
            hi=max(hi,value);lo=min(lo,value);
        }
        uint64_t offset=kv*a.vb2+c*a.vb1+d*a.vb0;
        if(a.v_f16) { *(device half *)(maximum+offset)=half(hi);*(device half *)(minimum+offset)=half(lo); }
        else { *(device float *)(maximum+offset)=hi;*(device float *)(minimum+offset)=lo; }
    }
    (void)previous;
}

template <ushort NH>
kernel void kernel_hsa_bounds_logits(
        constant ggml_metal_kargs_hsa & a,
        device const float * query,
        device char * maximum,
        device char * minimum,
        device const int * order,
        device const float * mask,
        device const char * keys,
        device float * out,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const int H=NH ? NH : a.H;
    int c=group.x*4+sg,start=c*a.chunk,C=(a.N+a.chunk-1)/a.chunk,kv=group.y;
    if(start>=a.N) return;
    int count=0;
    for(int t=start;t<min(a.N,start+a.chunk);++t) count+=isfinite(mask[order[t]]);
    if(lane==0) out[(kv*(H+1)+H)*C+c]=count;
    bool update=a.r && count && (start+a.chunk>=a.N || !isfinite(mask[order[start+a.chunk]]));
    float dots[NH ? NH : 32];
    for(int h=0;h<H;++h) dots[h]=0.f;
    if(count && a.D%4==0) for(int d=4*lane;d<a.D;d+=128) {
        uint64_t offset=kv*a.nb2+c*a.nb1+d*a.nb0;
        float4 hi,lo;
        if(update) {
            hi=float4(-INFINITY);lo=float4(INFINITY);
            for(int t=start;t<min(a.N,start+a.chunk);++t) {
                if(!isfinite(mask[order[t]])) continue;
                device const char * p=keys+order[t]*a.vb2+kv*a.vb1+d*a.vb0;
                float4 value=a.v_f16 ? float4(*(device const half4 *)p) : *(device const float4 *)p;
                hi=max(hi,value);lo=min(lo,value);
            }
            if(a.k_f16) { *(device half4 *)(maximum+offset)=half4(hi);*(device half4 *)(minimum+offset)=half4(lo); }
            else { *(device float4 *)(maximum+offset)=hi;*(device float4 *)(minimum+offset)=lo; }
        } else {
            hi=a.k_f16 ? float4(*(device const half4 *)(maximum+offset)) : *(device const float4 *)(maximum+offset);
            lo=a.k_f16 ? float4(*(device const half4 *)(minimum+offset)) : *(device const float4 *)(minimum+offset);
        }
        float4 sign=*(device const float4 *)(query+(kv*(H+1)+H)*a.D+d);
        float4 key=select(-lo,hi,sign>0.f);
        for(int h=0;h<H;++h) dots[h]+=dot(key,*(device const float4 *)(query+(kv*(H+1)+h)*a.D+d));
    }
    if(count && a.D%4!=0) for(int d=lane;d<a.D;d+=32) {
        uint64_t offset=kv*a.nb2+c*a.nb1+d*a.nb0;
        float hi,lo;
        if(update) {
            hi=-INFINITY;lo=INFINITY;
            for(int t=start;t<min(a.N,start+a.chunk);++t) {
                if(!isfinite(mask[order[t]])) continue;
                device const char * p=keys+order[t]*a.vb2+kv*a.vb1+d*a.vb0;
                float value=a.v_f16 ? float(*(device const half *)p) : *(device const float *)p;
                hi=max(hi,value);lo=min(lo,value);
            }
            if(a.k_f16) { *(device half *)(maximum+offset)=half(hi);*(device half *)(minimum+offset)=half(lo); }
            else { *(device float *)(maximum+offset)=hi;*(device float *)(minimum+offset)=lo; }
        } else {
            hi=a.k_f16 ? float(*(device const half *)(maximum+offset)) : *(device const float *)(maximum+offset);
            lo=a.k_f16 ? float(*(device const half *)(minimum+offset)) : *(device const float *)(minimum+offset);
        }
        float sign=query[(kv*(H+1)+H)*a.D+d];
        if(sign==0.f) continue;
        float key=sign>0.f ? hi : -lo;
        for(int h=0;h<H;++h) dots[h]+=key*query[(kv*(H+1)+h)*a.D+d];
    }
    for(int h=0;h<H;++h) {
        float value=simd_sum(dots[h]);
        if(lane==0) out[(kv*(H+1)+h)*C+c]=count ? value : -INFINITY;
    }
}
typedef decltype(kernel_hsa_bounds_logits<0>) kernel_hsa_bounds_logits_t;
template [[host_name("kernel_hsa_bounds_logits_h0")]] kernel kernel_hsa_bounds_logits_t kernel_hsa_bounds_logits<0>;
template [[host_name("kernel_hsa_bounds_logits_h1")]] kernel kernel_hsa_bounds_logits_t kernel_hsa_bounds_logits<1>;
template [[host_name("kernel_hsa_bounds_logits_h2")]] kernel kernel_hsa_bounds_logits_t kernel_hsa_bounds_logits<2>;
template [[host_name("kernel_hsa_bounds_logits_h3")]] kernel kernel_hsa_bounds_logits_t kernel_hsa_bounds_logits<3>;
template [[host_name("kernel_hsa_bounds_logits_h4")]] kernel kernel_hsa_bounds_logits_t kernel_hsa_bounds_logits<4>;
template [[host_name("kernel_hsa_bounds_logits_h8")]] kernel kernel_hsa_bounds_logits_t kernel_hsa_bounds_logits<8>;
template [[host_name("kernel_hsa_bounds_logits_h16")]] kernel kernel_hsa_bounds_logits_t kernel_hsa_bounds_logits<16>;
template [[host_name("kernel_hsa_bounds_logits_h32")]] kernel kernel_hsa_bounds_logits_t kernel_hsa_bounds_logits<32>;
