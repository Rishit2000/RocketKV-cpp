# HSA 优化实验记录

- 分支：`xiang/hsa-optimize`
- 起点：`95dea0953b11f21eb3f09f73fd959b11f3793282`
- 环境：Apple M2 Ultra；Llama-3.2-3B-Instruct-Q8_0；F16 KV；全 GPU offload；16 CPU threads；budget=1024。
- 规则：一次一个改动；与上一保留版交替运行；正确性失败、无可重复收益或明显回退的改动不保留。
- 性能口径：固定 token IDs 1000..1032，共 32 次 decode forward，关闭 trace，报告 ms/forward 中位数。临时 runner 仅解除 forced_tokens 必须启用 trace 的限制，正式 runner 不变。每项记录全部单次测量值。
- 输入：`results/runtime_probe/samples.json` 的 sample 201，重复八次后按现有中间截断逻辑截到指定长度。此实验衡量执行速度，不衡量生成质量。
- 正确性：选中 token 与 CPU reference 一致；attention 数值误差满足原有阈值；最终执行模型集成、UBSan 和 Metal shader validation。
- 每项记录其相对上一版的增量收益；最终另测相对起点的累计收益。历史测量不代替当前交替 A/B。
- 长输入检查发现 F32 重排会产生超阈值漂移后，最终组合只采用保持原浮点运算顺序的改法。E3 原组合虽通过当时对比，仍不纳入这一更保守的最终组合。

## 实验汇总

**最终保留链：E1 → E6 → E7 → E8。** 表中其余项的历史收益不计入最终结果。

最终版本相对起点 HSA 的累计耗时下降（本轮重新交替测量，不乘算增量百分比）：

| 输入 | 起点 HSA ms/步 | 最终 HSA ms/步 | 下降 |
|---|---:|---:|---:|
| 8192 | 32.40634 | 21.29306 | 34.29% |
| 16384 | 50.30534 | 21.41044 | 57.44% |
| 32768 | 50.85794 | 21.77200 | 57.19% |

以下为各项实验的增量结果；E2-E5 不计入最终累计收益。

| 编号 | 改动 | 8K 增量耗时下降 | 16K 增量耗时下降 | 决定 |
|---|---|---:|---:|---|
| E1 | 对 chunk 排序，再展开 token | 17.77% | 46.73% | 保留 |
| E2 | 加权求和沿 selected KV 并行归约 | 23.56% | 24.13% | 撤回：长输入数值漂移 |
| E3 | 近似评分使用 SIMD 并行点积 | 8.42% | 8.60% | 不纳入：最终保留原 FP 运算顺序 |
| E4 | 精确 QK 使用 SIMD 并行点积 | 4.17% | 4.06% | 撤回：长输入数值漂移 |
| E5 | 排序中的 SIMD 内交换使用 shuffle | 1.78% | 1.75% | 原组合撤回；安全组合重测见 E8 |
| E6 | 加权求和使用 64 线程组，保留累加顺序 | 1.83% | 2.17% | 保留 |
| E7 | 提前加载四个 V，按原顺序累加 | 17.93% | 17.11% | 保留 |
| E8 | 在安全组合上重测 SIMD shuffle 排序 | 1.20% | 1.69% | 保留 |

## 历史参照（不用于本轮增量收益判定）

| 输入长度 | 原生 Metal off | 仅 Stage 1 | 起点 HSA |
|---|---:|---:|---:|
| 16,384 | 10.400 | 9.053 | 50.314 |
| 32,768 | 13.729 | 9.721 | 50.830 |

单位为 ms/forward，三轮中位数。原始记录：`results/stage2-perf/native-long-context.json`。

## E1：按 chunk 排序，展开 token（保留）

同一 chunk 的 token 分数相同，只对 chunk 排序；按 score、chunk ID 保留稳定顺序，展开时处理不足一个 chunk 的尾部，再按 token ID 排序。保持原算法选择结果和精确 attention 累加顺序。线程组内存按排序宽度与 top-k 宽度的较大者分配。

| 输入 | 修改前各轮 ms/步 | 修改后各轮 ms/步 | 前中位数 | 后中位数 | 下降 |
|---|---|---|---:|---:|---:|
| 8192 | 32.46719, 32.46559, 32.42375 | 26.71372, 26.69697, 26.66134 | 32.46559 | 26.69697 | 17.77% |
| 16384 | 50.25419, 50.29953, 50.28675 | 26.77997, 26.80806, 26.78578 | 50.28675 | 26.78578 | 46.73% |

原有 Metal kernel/reference、native graph 测试通过；新增最高分短尾 chunk、top-k 跨 chunk、单 chunk 与 4096/4097 边界覆盖。

<details><summary>本项源代码差异</summary>

```diff
--- before/src/metal/rocket_metal.mm
+++ after/src/metal/rocket_metal.mm
@@ -34,7 +34,7 @@
 struct Params {
     uint32_t dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded, dim_padded, topk_padded;
     uint32_t qd, qh, kd, kt, kh, vd, vt, vh;
-    uint32_t cached_count, mask_type, mask_stride;
+    uint32_t cached_count, mask_type, mask_stride, sort_padded;
 };
 uint32_t power_of_two(uint32_t n) {
     uint32_t p = 1;
@@ -210,6 +210,7 @@
         p.padded = power_of_two(p.count);
         p.dim_padded = power_of_two(p.dim);
         p.topk_padded = power_of_two(p.topk);
+        p.sort_padded = power_of_two(p.chunks);
         p.qd = stride(q->nb[0]);
         p.qh = stride(q->nb[2]);
         p.kd = stride(k->nb[0]);
@@ -319,8 +320,8 @@
             dispatch(Approximate, (p.heads * p.chunks + 255) / 256);
             softmax(p.chunks, 0);
         }
-        const size_t sort_bytes = std::max(size_t(16), size_t(p.padded) * sizeof(uint32_t));
-        if (p.topk < p.count && 2 * sort_bytes <= m.device.maxThreadgroupMemoryLength) {
+        const size_t sort_bytes = std::max(size_t(16), size_t(std::max(p.sort_padded, p.topk_padded)) * sizeof(uint32_t));
+        if (p.topk < p.count && 2 * sort_bytes + m.pipelines[SelectLocal].staticThreadgroupMemoryLength <= m.device.maxThreadgroupMemoryLength) {
             [encoder setThreadgroupMemoryLength:sort_bytes atIndex:0];
             [encoder setThreadgroupMemoryLength:sort_bytes atIndex:1];
             dispatch(SelectLocal, p.kv_heads);
--- before/src/metal/rocket_metal.metal
+++ after/src/metal/rocket_metal.metal
@@ -4,7 +4,7 @@
 struct Params {
     uint dim, heads, kv_heads, count, chunk, rank, topk, chunks, padded, dim_padded, topk_padded;
     uint qd, qh, kd, kt, kh, vd, vt, vh;
-    uint cached_count, mask_type, mask_stride;
+    uint cached_count, mask_type, mask_stride, sort_padded;
 };
 #define ARGS                                                                                                 \
     constant Params &p [[buffer(0)]], device const uchar *qr [[buffer(1)]],                                  \
@@ -186,21 +186,21 @@
         return;
     }
     uint nh = p.heads / p.kv_heads;
-    for (uint i = tid; i < p.padded; i += 256) {
+    for (uint i = tid; i < p.sort_padded; i += 256) {
         float score = 0;
-        if (i < p.count) {
+        if (i < p.chunks) {
             for (uint h = 0; h < nh; ++h) {
-                score += chunks[(group * nh + h) * p.chunks + i / p.chunk];
-            }
-        }
-        values[i] = i < p.count ? score : -INFINITY;
+                score += chunks[(group * nh + h) * p.chunks + i];
+            }
+        }
+        values[i] = i < p.chunks ? score : -INFINITY;
         ids[i] = i;
     }
     threadgroup_barrier(memory);
     // One threadgroup owns each KV group's sort; no cross-threadgroup barrier is needed.
-    for (uint size = 2; size <= p.padded; size *= 2) {
+    for (uint size = 2; size <= p.sort_padded; size *= 2) {
         for (uint stride = size / 2; stride; stride /= 2) {
-            for (uint i = tid; i < p.padded; i += 256) {
+            for (uint i = tid; i < p.sort_padded; i += 256) {
                 uint j = i ^ stride;
                 if (j > i) {
                     bool swap = (i & size) == 0 ? better(values[j], ids[j], values[i], ids[i])
@@ -218,27 +218,38 @@
             threadgroup_barrier(memory);
         }
     }
-    // Match the CPU reference's chronological order for exact attention accumulation.
+    // Tokens within a chunk share a score. Expand ranked chunks, accounting for the short tail.
+    const uint missing = p.chunks * p.chunk - p.count;
+    if (!missing) {
+        if (tid == 0) values[0] = float(p.count);
+    } else {
+        for (uint i = tid; i < p.chunks; i += 256) {
+            if (ids[i] == p.chunks - 1) values[0] = float(i * p.chunk);
+        }
+    }
+    threadgroup_barrier(memory);
+    uint tail_start = uint(values[0]);
+    threadgroup_barrier(memory);
     for (uint i = tid; i < p.topk_padded; i += 256) {
-        if (i >= p.topk) {
-            ids[i] = 0xffffffffu;
-        }
-    }
-    threadgroup_barrier(memory);
+        uint j = i + (i >= tail_start + p.chunk - missing ? missing : 0);
+        values[i] = i < p.topk ? float(ids[j / p.chunk] * p.chunk + j % p.chunk) : INFINITY;
+    }
+    threadgroup_barrier(memory);
+    // Candidate IDs <=32768 are exact in F32; reuse scores to sort IDs chronologically.
     for (uint size = 2; size <= p.topk_padded; size *= 2) {
         for (uint stride = size / 2; stride; stride /= 2) {
             for (uint i = tid; i < p.topk_padded; i += 256) {
                 uint j = i ^ stride;
-                if (j > i && ((i & size) == 0 ? ids[i] > ids[j] : ids[i] < ids[j])) {
-                    uint u = ids[i];
-                    ids[i] = ids[j];
-                    ids[j] = u;
+                if (j > i && ((i & size) == 0 ? values[i] > values[j] : values[i] < values[j])) {
+                    float value = values[i];
+                    values[i] = values[j];
+                    values[j] = value;
                 }
             }
             threadgroup_barrier(memory);
         }
     }
-    for (uint i = tid; i < p.topk; i += 256) indices[group * p.padded + i] = ids[i];
+    for (uint i = tid; i < p.topk; i += 256) indices[group * p.padded + i] = uint(values[i]);
 }
 
 kernel void select_tokens(ARGS, uint tid [[thread_index_in_threadgroup]],
```

</details>

## E2：加权求和沿 selected KV 并行归约（撤回：长输入数值漂移）

每个线程组处理一个 head 的 32 个输出维度，将所选 token 分给 8 组线程，再归约部分和。缩短串行循环并增加线程组数量。改变 F32 累加顺序，但不改变所选 token 或数学公式。扩展内核边界测试及全部 16 项集成/脚本测试通过。

| 输入 | 修改前各轮 ms/步 | 修改后各轮 ms/步 | 前中位数 | 后中位数 | 下降 |
|---|---|---|---:|---:|---:|
| 8192 | 26.63244, 26.79597, 26.64991 | 20.37150, 20.15944, 20.40972 | 26.64991 | 20.37150 | 23.56% |
| 16384 | 26.66409, 26.90900, 26.80247 | 20.33528, 20.40741, 20.32225 | 26.80247 | 20.33528 | 24.13% |

<details><summary>本项源代码差异</summary>

```diff
--- before/src/metal/rocket_metal.mm
+++ after/src/metal/rocket_metal.mm
@@ -332,7 +332,7 @@
         }
         dispatch(Exact, (p.heads * p.topk + 255) / 256);
         softmax(p.topk, 1);
-        dispatch(WeightedSum, (p.heads * p.dim + 255) / 256);
+        dispatch(WeightedSum, p.heads * ((p.dim + 31) / 32));
         if (!native_encoder) {
             [encoder endEncoding];
             [command commit];
--- before/src/metal/rocket_metal.metal
+++ after/src/metal/rocket_metal.metal
@@ -277,18 +277,25 @@
     exact[i] = dot / sqrt(float(p.dim));
 }
 
-kernel void weighted_sum(ARGS, uint i [[thread_position_in_grid]]) {
-    if (i >= p.heads * p.dim) {
-        return;
-    }
-    uint h = i / p.dim, d = i % p.dim, group = h / (p.heads / p.kv_heads);
+kernel void weighted_sum(ARGS, uint tid [[thread_index_in_threadgroup]],
+                         uint block [[threadgroup_position_in_grid]]) {
+    threadgroup float partial[256];
+    uint blocks = (p.dim + 31) / 32;
+    uint h = block / blocks, d = (block % blocks) * 32 + tid % 32;
+    uint group = h / (p.heads / p.kv_heads);
     float sum = 0;
-    for (uint j = 0; j < p.topk; ++j) {
-        uint slot = uint(slots[indices[group * p.padded + j]]);
-        sum += exact[h * p.topk + j] * read_kv(vr, group, slot, d, p.vh, p.vt, p.vd);
-    }
-    output[i] = sum;
-    if (!isfinite(sum)) {
-        atomic_store_explicit(error, 1u, memory_order_relaxed);
-    }
-}
+    if (d < p.dim) {
+        for (uint j = tid / 32; j < p.topk; j += 8) {
+            uint slot = uint(slots[indices[group * p.padded + j]]);
+            sum += exact[h * p.topk + j] * read_kv(vr, group, slot, d, p.vh, p.vt, p.vd);
+        }
+    }
+    partial[tid] = sum;
+    threadgroup_barrier(mem_flags::mem_threadgroup);
+    if (tid < 32 && d < p.dim) {
+        sum = 0;
+        for (uint j = 0; j < 8; ++j) sum += partial[j * 32 + tid];
+        output[h * p.dim + d] = sum;
+        if (!isfinite(sum)) atomic_store_explicit(error, 1u, memory_order_relaxed);
+    }
+}
```

</details>

## E3：近似评分使用 SIMD 并行点积（不纳入最终版本）

将每个 query/chunk 的截断维度点积分给一个 SIMD group，使用 simd_sum 合并部分和；主机按 pipeline 的 threadExecutionWidth 计算分组数。内核/reference 与 16 项集成/脚本测试通过。另在 8K、16K 真实模型上固定 4 次 forward 比较，本项前后 attention/logits 最大绝对差均为 0。

| 输入 | 修改前各轮 ms/步 | 修改后各轮 ms/步 | 前中位数 | 后中位数 | 下降 |
|---|---|---|---:|---:|---:|
| 8192 | 20.32347, 20.39719, 20.36409 | 18.64844, 18.73681, 18.61119 | 20.36409 | 18.64844 | 8.42% |
| 16384 | 20.45400, 20.42169, 20.41575 | 18.59178, 18.71541, 18.66584 | 20.42169 | 18.66584 | 8.60% |

<details><summary>本项源代码差异</summary>

```diff
--- before/src/metal/rocket_metal.mm
+++ after/src/metal/rocket_metal.mm
@@ -317,7 +317,8 @@
                 dispatch(Summarize, (updated_chunks * p.kv_heads * p.dim + 255) / 256);
             }
             dispatch(Prepare, p.kv_heads);
-            dispatch(Approximate, (p.heads * p.chunks + 255) / 256);
+            const uint32_t chunks_per_group = 256 / m.pipelines[Approximate].threadExecutionWidth;
+            dispatch(Approximate, (p.heads * p.chunks + chunks_per_group - 1) / chunks_per_group);
             softmax(p.chunks, 0);
         }
         const size_t sort_bytes = std::max(size_t(16), size_t(std::max(p.sort_padded, p.topk_padded)) * sizeof(uint32_t));
--- before/src/metal/rocket_metal.metal
+++ after/src/metal/rocket_metal.metal
@@ -116,13 +116,15 @@
     }
 }
 
-kernel void approximate(ARGS, uint i [[thread_position_in_grid]]) {
+kernel void approximate(ARGS, uint tid [[thread_position_in_grid]],
+                        uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
+    uint i = tid / width;
     if (i >= p.heads * p.chunks) {
         return;
     }
     uint h = i / p.chunks, c = i % p.chunks, group = h / (p.heads / p.kv_heads);
     float dot = 0;
-    for (uint r = 0; r < p.rank; ++r) {
+    for (uint r = lane; r < p.rank; r += width) {
         uint d = dims[group * p.dim + r];
         bool positive = signs[group * p.dim + d] >= 0;
         float rep;
@@ -134,7 +136,8 @@
         }
         dot += q[h * p.dim + d] * rep;
     }
-    chunks[i] = scales[h] > 0 ? dot / scales[h] : 0;
+    dot = simd_sum(dot);
+    if (lane == 0) chunks[i] = scales[h] > 0 ? dot / scales[h] : 0;
 }
 
 // Used for both approximate chunk probabilities and exact selected-token probabilities.
```

</details>

## E4：精确 QK 使用 SIMD 并行点积（撤回：长输入数值漂移）

将每个 query/selected-token 的完整维度点积分给一个 SIMD group，再归约；主机使用 pipeline 的 threadExecutionWidth。改变 F32 求和顺序。内核/reference 与 16 项集成/脚本测试通过。此项在 E1-E3 后重新实测，收益不沿用旧同步实现上的结论。

| 输入 | 修改前各轮 ms/步 | 修改后各轮 ms/步 | 前中位数 | 后中位数 | 下降 |
|---|---|---|---:|---:|---:|
| 8192 | 18.78759, 18.64938, 18.76266 | 18.06978, 17.97956, 17.91222 | 18.76266 | 17.97956 | 4.17% |
| 16384 | 18.65062, 18.64444, 18.75803 | 17.96841, 17.89284, 17.88216 | 18.65062 | 17.89284 | 4.06% |

<details><summary>本项源代码差异</summary>

```diff
--- before/src/metal/rocket_metal.mm
+++ after/src/metal/rocket_metal.mm
@@ -331,7 +331,8 @@
         } else {
             dispatch(Select, p.kv_heads);
         }
-        dispatch(Exact, (p.heads * p.topk + 255) / 256);
+        const uint32_t tokens_per_group = 256 / m.pipelines[Exact].threadExecutionWidth;
+        dispatch(Exact, (p.heads * p.topk + tokens_per_group - 1) / tokens_per_group);
         softmax(p.topk, 1);
         dispatch(WeightedSum, p.heads * ((p.dim + 31) / 32));
         if (!native_encoder) {
--- before/src/metal/rocket_metal.metal
+++ after/src/metal/rocket_metal.metal
@@ -268,16 +268,19 @@
     select_tokens_impl(p, chunks, indices, values, ids, tid, group, mem_flags::mem_threadgroup);
 }
 
-kernel void exact_scores(ARGS, uint i [[thread_position_in_grid]]) {
+kernel void exact_scores(ARGS, uint tid [[thread_position_in_grid]],
+                         uint lane [[thread_index_in_simdgroup]], uint width [[threads_per_simdgroup]]) {
+    uint i = tid / width;
     if (i >= p.heads * p.topk) {
         return;
     }
     uint h = i / p.topk, group = h / (p.heads / p.kv_heads), t = indices[group * p.padded + i % p.topk];
     float dot = 0;
-    for (uint d = 0; d < p.dim; ++d) {
+    for (uint d = lane; d < p.dim; d += width) {
         dot += q[h * p.dim + d] * read_kv(kr, group, uint(slots[t]), d, p.kh, p.kt, p.kd);
     }
-    exact[i] = dot / sqrt(float(p.dim));
+    dot = simd_sum(dot);
+    if (lane == 0) exact[i] = dot / sqrt(float(p.dim));
 }
 
 kernel void weighted_sum(ARGS, uint tid [[thread_index_in_threadgroup]],
```

</details>

## E5：排序中的 SIMD 内交换使用 shuffle（原组合撤回；重测见 E8）

同 SIMD group 内的 bitonic compare/exchange 使用 simd_shuffle_xor；跨组交换及每个 merge 边界保留线程组屏障。选择排序和 token ID 排序均覆盖。三轮 8K/16K 的所有配对都改善，但幅度较小，因此额外执行了 8K、128 forward、五轮交替确认。正确性与 16 项集成/脚本测试通过。

| 输入 | 修改前各轮 ms/步 | 修改后各轮 ms/步 | 前中位数 | 后中位数 | 下降 |
|---|---|---|---:|---:|---:|
| 8192 | 17.98125, 18.12738, 17.87644 | 17.66150, 17.61722, 17.74244 | 17.98125 | 17.66150 | 1.78% |
| 16384 | 17.96700, 17.83872, 17.82528 | 17.42634, 17.54306, 17.52622 | 17.83872 | 17.52622 | 1.75% |

<details><summary>本项源代码差异</summary>

```diff
--- before/src/metal/rocket_metal.metal
+++ after/src/metal/rocket_metal.metal
@@ -181,7 +181,7 @@
 
 template <typename Scores, typename Indices>
 inline void select_tokens_impl(constant Params &p, device const float *chunks, device uint *indices,
-                               Scores values, Indices ids, uint tid, uint group, mem_flags memory) {
+                               Scores values, Indices ids, uint tid, uint group, uint width, mem_flags memory) {
     if (p.topk == p.count) {
         for (uint i = tid; i < p.count; i += 256) {
             indices[group * p.padded + i] = i;
@@ -205,7 +205,15 @@
         for (uint stride = size / 2; stride; stride /= 2) {
             for (uint i = tid; i < p.sort_padded; i += 256) {
                 uint j = i ^ stride;
-                if (j > i) {
+                if (stride < width) {
+                    float value = values[i], other = simd_shuffle_xor(value, stride);
+                    uint id = ids[i], other_id = simd_shuffle_xor(id, stride);
+                    bool descending = ((i & size) == 0) == ((i & stride) == 0);
+                    if (descending ? better(other, other_id, value, id) : better(value, id, other, other_id)) {
+                        values[i] = other;
+                        ids[i] = other_id;
+                    }
+                } else if (j > i) {
                     bool swap = (i & size) == 0 ? better(values[j], ids[j], values[i], ids[i])
                                                 : better(values[i], ids[i], values[j], ids[j]);
                     if (swap) {
@@ -218,7 +226,7 @@
                     }
                 }
             }
-            threadgroup_barrier(memory);
+            if (stride >= width || stride == 1) threadgroup_barrier(memory);
         }
     }
     // Tokens within a chunk share a score. Expand ranked chunks, accounting for the short tail.
@@ -243,29 +251,33 @@
         for (uint stride = size / 2; stride; stride /= 2) {
             for (uint i = tid; i < p.topk_padded; i += 256) {
                 uint j = i ^ stride;
-                if (j > i && ((i & size) == 0 ? values[i] > values[j] : values[i] < values[j])) {
+                if (stride < width) {
+                    float value = values[i], other = simd_shuffle_xor(value, stride);
+                    bool ascending = ((i & size) == 0) == ((i & stride) == 0);
+                    values[i] = ascending ? min(value, other) : max(value, other);
+                } else if (j > i && ((i & size) == 0 ? values[i] > values[j] : values[i] < values[j])) {
                     float value = values[i];
                     values[i] = values[j];
                     values[j] = value;
                 }
             }
-            threadgroup_barrier(memory);
+            if (stride >= width || stride == 1) threadgroup_barrier(memory);
         }
     }
     for (uint i = tid; i < p.topk; i += 256) indices[group * p.padded + i] = uint(values[i]);
 }
 
 kernel void select_tokens(ARGS, uint tid [[thread_index_in_threadgroup]],
-                          uint group [[threadgroup_position_in_grid]]) {
+                          uint group [[threadgroup_position_in_grid]], uint width [[threads_per_simdgroup]]) {
     select_tokens_impl(p, chunks, indices, scores + group * p.padded, indices + group * p.padded,
-                       tid, group, mem_flags::mem_device);
+                       tid, group, width, mem_flags::mem_device);
 }
 
 kernel void select_tokens_local(ARGS, threadgroup float *values [[threadgroup(0)]],
                                threadgroup uint *ids [[threadgroup(1)]],
                                uint tid [[thread_index_in_threadgroup]],
-                               uint group [[threadgroup_position_in_grid]]) {
-    select_tokens_impl(p, chunks, indices, values, ids, tid, group, mem_flags::mem_threadgroup);
+                               uint group [[threadgroup_position_in_grid]], uint width [[threads_per_simdgroup]]) {
+    select_tokens_impl(p, chunks, indices, values, ids, tid, group, width, mem_flags::mem_threadgroup);
 }
 
 kernel void exact_scores(ARGS, uint tid [[thread_position_in_grid]],
```

</details>

### E5 小收益追加确认

8K，max_tokens=129、固定 token IDs 1000..1128，即 128 次 forward，五轮交替。其他设置相同。步数变化会改变 Stage 1 保留数量，因此这组绝对时间只在组内比较。

| 版本 | 五轮 ms/步 | 中位数 |
|---|---|---:|
| exact | 17.44084, 17.41299, 17.44680, 17.43545, 17.52749 | 17.44084 |
| shuffle | 17.11245, 17.12345, 17.12775, 17.14937, 17.10084 | 17.12345 |

五对测量全部下降，中位数下降 **1.82%**，当时作为候选保留；原组合随后撤回，最终在 E8 中重新验证。

## 长输入最终检查与回退（覆盖前面的候选保留决定）

小模型和内核测试通过不代表长输入一定满足本轮数值阈值。使用相同 4 次 teacher-forced forward、比较每层 attention 与 logits 后发现：

| 对比 | 8K attention 最大绝对差 | 8K logits 最大绝对差 | 结论 |
|---|---:|---:|---|
| 起点 → E1 | 0 | 0 | E1 保留；16K 也完全一致 |
| 起点 → E2 组合 | 0.00268519 | 0.00189757 | 超出逐元素 `2e-4 + 2e-4*abs(reference)`，撤回 E2 |
| E3 → E4 | 0.02570379 | 0.04091632 | 超阈值，撤回 E4 |
| E4 → E5 | 0 | 0 | E5 本身无数值变化，但需在新的保留基线上复测 |

E3/E5 的原组合性能数字仅代表当时实验，不能计入最终累计收益。代码先恢复到 E1，再逐项重建安全组合；不放宽误差阈值。

## E6：加权求和使用 64 线程组，保留累加顺序（保留）

恢复原始串行加权求和公式，只将其 dispatch 的线程组由 256 缩小为 64。扩大可调度的线程组数量，不重排浮点加法。扩展内核/reference 测试通过；8K/16K 真实模型与 E1 的 attention/logits 最大绝对差均为 0。

| 输入 | 修改前各轮 ms/步 | 修改后各轮 ms/步 | 前中位数 | 后中位数 | 下降 |
|---|---|---|---:|---:|---:|
| 8192 | 26.60816, 26.66834, 26.60859 | 26.12525, 26.10684, 26.12062 | 26.60859 | 26.12062 | 1.83% |
| 16384 | 26.84113, 26.82869, 26.89063 | 26.42209, 26.25769, 26.25569 | 26.84113 | 26.25769 | 2.17% |

<details><summary>本项源代码差异</summary>

```diff
--- before/src/metal/rocket_metal.mm
+++ after/src/metal/rocket_metal.mm
@@ -299,10 +299,10 @@
             [encoder setBuffer:buffers[RawQ] offset:0 atIndex:18];
         }
         [encoder setBytes:&p length:sizeof(p) atIndex:0];
-        auto dispatch = [&](size_t kernel, uint32_t groups) {
+        auto dispatch = [&](size_t kernel, uint32_t groups, uint32_t threads = 256) {
             [encoder setComputePipelineState:m.pipelines[kernel]];
             [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
-                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
+                    threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
             [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
         };
         auto softmax = [&](uint32_t width, uint32_t exact_mode) {
@@ -332,7 +332,7 @@
         }
         dispatch(Exact, (p.heads * p.topk + 255) / 256);
         softmax(p.topk, 1);
-        dispatch(WeightedSum, (p.heads * p.dim + 255) / 256);
+        dispatch(WeightedSum, (p.heads * p.dim + 63) / 64, 64);
         if (!native_encoder) {
             [encoder endEncoding];
             [command commit];
```

</details>

## E7：提前加载四个 V，按原顺序累加（保留）

每次循环预先读取四个 selected KV 的 V 值，再依照原始 j 顺序执行四次 sum += probability * value，余数走原循环。没有并行归约，也不改变浮点加法顺序。扩展内核/reference 测试通过；8K/16K 真实模型与 E6 的 attention/logits 最大绝对差均为 0。

| 输入 | 修改前各轮 ms/步 | 修改后各轮 ms/步 | 前中位数 | 后中位数 | 下降 |
|---|---|---|---:|---:|---:|
| 8192 | 26.16825, 26.14809, 26.19050 | 21.45947, 21.47613, 21.64675 | 26.16825 | 21.47613 | 17.93% |
| 16384 | 26.24678, 26.30613, 26.14537 | 21.75716, 21.80222, 21.67922 | 26.24678 | 21.75716 | 17.11% |

<details><summary>本项源代码差异</summary>

```diff
--- before/src/metal/rocket_metal.metal
+++ after/src/metal/rocket_metal.metal
@@ -283,7 +283,22 @@
     }
     uint h = i / p.dim, d = i % p.dim, group = h / (p.heads / p.kv_heads);
     float sum = 0;
-    for (uint j = 0; j < p.topk; ++j) {
+    uint j = 0;
+    for (; j + 3 < p.topk; j += 4) {
+        uint s0 = uint(slots[indices[group * p.padded + j]]);
+        uint s1 = uint(slots[indices[group * p.padded + j + 1]]);
+        uint s2 = uint(slots[indices[group * p.padded + j + 2]]);
+        uint s3 = uint(slots[indices[group * p.padded + j + 3]]);
+        float v0 = read_kv(vr, group, s0, d, p.vh, p.vt, p.vd);
+        float v1 = read_kv(vr, group, s1, d, p.vh, p.vt, p.vd);
+        float v2 = read_kv(vr, group, s2, d, p.vh, p.vt, p.vd);
+        float v3 = read_kv(vr, group, s3, d, p.vh, p.vt, p.vd);
+        sum += exact[h * p.topk + j] * v0;
+        sum += exact[h * p.topk + j + 1] * v1;
+        sum += exact[h * p.topk + j + 2] * v2;
+        sum += exact[h * p.topk + j + 3] * v3;
+    }
+    for (; j < p.topk; ++j) {
         uint slot = uint(slots[indices[group * p.padded + j]]);
         sum += exact[h * p.topk + j] * read_kv(vr, group, slot, d, p.vh, p.vt, p.vd);
     }
```

</details>

## E8：在安全组合上重测 SIMD shuffle 排序（保留）

重新应用 E5 的排序交换优化，但基线改为 E1+E6+E7，不含改变浮点求和顺序的候选。8K/16K 六对测量全部下降，幅度约 1%-2%；两档真实模型 attention/logits 与 E7 最大绝对差均为 0。这里只使用新基线测得的增量收益。

| 输入 | 修改前各轮 ms/步 | 修改后各轮 ms/步 | 前中位数 | 后中位数 | 下降 |
|---|---|---|---:|---:|---:|
| 8192 | 21.57734, 21.57825, 21.62569 | 21.31903, 21.30541, 21.34031 | 21.57825 | 21.31903 | 1.20% |
| 16384 | 21.76597, 21.72072, 21.64997 | 21.35416, 21.26131, 21.39897 | 21.72072 | 21.35416 | 1.69% |

<details><summary>本项源代码差异</summary>

```diff
--- before/src/metal/rocket_metal.metal
+++ after/src/metal/rocket_metal.metal
@@ -178,7 +178,7 @@
 
 template <typename Scores, typename Indices>
 inline void select_tokens_impl(constant Params &p, device const float *chunks, device uint *indices,
-                               Scores values, Indices ids, uint tid, uint group, mem_flags memory) {
+                               Scores values, Indices ids, uint tid, uint group, uint width, mem_flags memory) {
     if (p.topk == p.count) {
         for (uint i = tid; i < p.count; i += 256) {
             indices[group * p.padded + i] = i;
@@ -202,7 +202,15 @@
         for (uint stride = size / 2; stride; stride /= 2) {
             for (uint i = tid; i < p.sort_padded; i += 256) {
                 uint j = i ^ stride;
-                if (j > i) {
+                if (stride < width) {
+                    float value = values[i], other = simd_shuffle_xor(value, stride);
+                    uint id = ids[i], other_id = simd_shuffle_xor(id, stride);
+                    bool descending = ((i & size) == 0) == ((i & stride) == 0);
+                    if (descending ? better(other, other_id, value, id) : better(value, id, other, other_id)) {
+                        values[i] = other;
+                        ids[i] = other_id;
+                    }
+                } else if (j > i) {
                     bool swap = (i & size) == 0 ? better(values[j], ids[j], values[i], ids[i])
                                                 : better(values[i], ids[i], values[j], ids[j]);
                     if (swap) {
@@ -215,7 +223,7 @@
                     }
                 }
             }
-            threadgroup_barrier(memory);
+            if (stride >= width || stride == 1) threadgroup_barrier(memory);
         }
     }
     // Tokens within a chunk share a score. Expand ranked chunks, accounting for the short tail.
@@ -240,29 +248,33 @@
         for (uint stride = size / 2; stride; stride /= 2) {
             for (uint i = tid; i < p.topk_padded; i += 256) {
                 uint j = i ^ stride;
-                if (j > i && ((i & size) == 0 ? values[i] > values[j] : values[i] < values[j])) {
+                if (stride < width) {
+                    float value = values[i], other = simd_shuffle_xor(value, stride);
+                    bool ascending = ((i & size) == 0) == ((i & stride) == 0);
+                    values[i] = ascending ? min(value, other) : max(value, other);
+                } else if (j > i && ((i & size) == 0 ? values[i] > values[j] : values[i] < values[j])) {
                     float value = values[i];
                     values[i] = values[j];
                     values[j] = value;
                 }
             }
-            threadgroup_barrier(memory);
+            if (stride >= width || stride == 1) threadgroup_barrier(memory);
         }
     }
     for (uint i = tid; i < p.topk; i += 256) indices[group * p.padded + i] = uint(values[i]);
 }
 
 kernel void select_tokens(ARGS, uint tid [[thread_index_in_threadgroup]],
-                          uint group [[threadgroup_position_in_grid]]) {
+                          uint group [[threadgroup_position_in_grid]], uint width [[threads_per_simdgroup]]) {
     select_tokens_impl(p, chunks, indices, scores + group * p.padded, indices + group * p.padded,
-                       tid, group, mem_flags::mem_device);
+                       tid, group, width, mem_flags::mem_device);
 }
 
 kernel void select_tokens_local(ARGS, threadgroup float *values [[threadgroup(0)]],
                                threadgroup uint *ids [[threadgroup(1)]],
                                uint tid [[thread_index_in_threadgroup]],
-                               uint group [[threadgroup_position_in_grid]]) {
-    select_tokens_impl(p, chunks, indices, values, ids, tid, group, mem_flags::mem_threadgroup);
+                               uint group [[threadgroup_position_in_grid]], uint width [[threads_per_simdgroup]]) {
+    select_tokens_impl(p, chunks, indices, values, ids, tid, group, width, mem_flags::mem_threadgroup);
 }
 
 kernel void exact_scores(ARGS, uint i [[thread_position_in_grid]]) {
```

</details>

## 最终累计结果的原始测量值

| 输入 | 起点三轮 ms/步 | 最终三轮 ms/步 |
|---|---|---|
| 8192 | 32.40634, 32.44309, 32.39078 | 21.29200, 21.29306, 21.29491 |
| 16384 | 50.28106, 50.31519, 50.30534 | 21.39506, 21.41044, 21.46291 |
| 32768 | 50.66600, 51.07469, 50.85794 | 21.71359, 21.79787, 21.77200 |

## 最终验证

真实模型固定 4 次 teacher-forced forward，逐层比较 attention 与 logits；不同于性能测试的 32 次 forward。逐元素阈值始终为 `2e-4 + 2e-4*abs(reference)`，未放宽。

| 输入 | attention 最大绝对差 | logits 最大绝对差 | 超阈值元素 |
|---|---:|---:|---:|
| 8192 | 0.0 | 0.0 | 0 |
| 16384 | 0.0 | 0.0 | 0 |
| 32768 | 0.0 | 0.0 | 0 |

- `PATH="$PWD/.venv/bin:$PATH" ROCKET_TEST_METAL=1 bash scripts/stage2.sh test`：通过，含算法测试、16 项 Python 集成/脚本测试、Metal 内核测试、UBSan 与 12 项 sanitizer 集成测试。
- `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./build/test_metal`：通过，覆盖尾 chunk、4096/4097、单维/非 SIMD 整数倍维度、近全选、strided KV、缓存、native graph 与 mask 校验。
- 正式 `src/runner.cpp`、配置、预算和模型未改动。临时测速 runner 仅在副本中将 `!trace_logits || forced_tokens.size()` 改为 `forced_tokens.size()`，允许关闭 trace 时使用固定 token。
- 所有测速临时二进制、输入和大日志在记录完成后清理；本文件保留每项源代码差异与全部用于判定的耗时测量。
- 这些是同机、两/三档固定输入的性能测试，不是完整 LongBench 质量评估，也不是原生 dense attention 的加速声明。

## 测试与文档改动

最终生产内核可按 E1、E6、E7、E8 的源代码差异从起点重建。以下补充对应的测试与 README 差异。

```diff
diff --git a/README.md b/README.md
index cae6fc6..a796bc7 100644
--- a/README.md
+++ b/README.md
@@ -151,7 +151,7 @@ ROCKET_TEST_METAL=1 bash scripts/stage2.sh test
 
 The Metal implementation is isolated in `src/metal/`; `src/rocket.cpp` remains the CPU reference. Kernels are embedded into the binary at build time. With compatible shared Metal buffers, HSA encodes into GGML's existing command encoder, reads Q/K/V and writes its output directly, and checks GPU errors after decode synchronization. Scratch buffers are isolated per layer; the scheduler tracks the complete tensor dependencies. CPU inputs and unsupported buffers retain synchronous staging, with direct K/V binding where available. `scripts/stage2.sh build` applies the optional buffer and custom-op bridges from `patches/`.
 
-GPU kernels read selected F16 K/V with F32 accumulation. Per-layer chunk min/max summaries reuse immutable KV prefixes and update the tail on append; request resets, changed slots or summary shapes rebuild them. Single-token chunks read K directly. Top-K sorting uses threadgroup memory when it fits, with the original device-memory path for larger candidate sets. Both paths preserve score/index tie-breaking and chronological accumulation order.
+GPU kernels read selected F16 K/V with F32 accumulation. Per-layer chunk min/max summaries reuse immutable KV prefixes and update the tail on append; request resets, changed slots or summary shapes rebuild them. Single-token chunks read K directly. Top-K ranks chunks and expands their token IDs, using threadgroup memory when it fits and device memory otherwise. Both paths preserve score/index tie-breaking and chronological selected-token order. Per-change measurements and code differences are recorded in [HSA_OPTIMIZATION_LOG.md](HSA_OPTIMIZATION_LOG.md).
 
 Supported Metal shapes are head dimensions 1–256, up to 256 query heads per KV group, and 1–32,768 live candidates, with F32 Q and F16 K/V. Unsupported shapes and non-finite input/output fail explicitly. The shader uses F32 accumulation and deterministic score/index tie-breaking; floating-point reduction order can differ from the CPU reference, particularly near selection boundaries. Full-model quantized CPU/Metal math also differs. Tests compare selected indices and attention on controlled inputs, and layer outputs/logits in real-model integration tests.
 
diff --git a/tests/test_metal.cpp b/tests/test_metal.cpp
index 3aeb816..e6845ab 100644
--- a/tests/test_metal.cpp
+++ b/tests/test_metal.cpp
@@ -20,13 +20,13 @@ void check(bool value, const char *message) {
 }
 
 void compare(RocketMetal &metal, int dim, int count, int chunk, int topk, bool zero_query = false,
-             bool tied = false, float key_scale = 1.0f) {
+             bool tied = false, float key_scale = 1.0f, bool tail_best = false) {
     RocketConfig cfg{};
     cfg.dim_head = dim;
     cfg.num_heads = 6;
     cfg.num_heads_kv = 2;
     cfg.chunk_size = chunk;
-    cfg.r_dim = dim / 2;
+    cfg.r_dim = std::max(1, dim / 2);
     cfg.compression_ratio = 2;
     cfg.topk_decode = topk;
     const int capacity = count * 2 + 3;
@@ -52,6 +52,14 @@ void compare(RocketMetal &metal, int dim, int count, int chunk, int topk, bool z
     for (int i = 0; i < count; ++i) {
         slots[i] = 2 * i + 1;
     }
+    if (tail_best) {
+        std::fill_n(qp, dim * cfg.num_heads, 1.0f);
+        for (int h = 0; h < cfg.num_heads_kv; ++h) {
+            for (int t = (count - 1) / chunk * chunk; t < count; ++t) {
+                std::fill_n(kp + (size_t(h) * capacity + slots[t]) * dim, dim, ggml_fp32_to_fp16(10.0f));
+            }
+        }
+    }
     std::vector<float> kc(size_t(count) * dim * cfg.num_heads_kv), vc(kc.size());
     for (int h = 0; h < cfg.num_heads_kv; ++h) {
         for (int t = 0; t < count; ++t) {
@@ -301,9 +309,18 @@ int main() {
             reg ? ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr) : nullptr, ggml_backend_free);
         check(!reg || bool(backend), "Metal backend initialization failed");
         RocketMetal metal;
+        compare(metal, 1, 17, 3, 7);
+        compare(metal, 31, 33, 1, 17);
+        compare(metal, 33, 35, 4, 10);
         compare(metal, 16, 1, 1, 1);
         compare(metal, 16, 2, 1, 1);
         compare(metal, 16, 5, 3, 3);
+        for (int topk : {1, 3, 4, 6, 12}) compare(metal, 17, 13, 5, topk, false, false, 1.0f, true);
+        compare(metal, 17, 13, 32, 5);
+        compare(metal, 64, 4096, 1, 512);
+        compare(metal, 64, 4097, 1, 512);
+        compare(metal, 16, 4097, 3, 4097);
+        compare(metal, 16, 1300, 3, 1299);
         compare(metal, 64, 257, 3, 17);
         compare(metal, 128, 513, 2, 256);
         compare(metal, 128, 4096, 3, 512);
```

## 最终文件指纹

| 文件 | SHA-256 |
|---|---|
| `src/metal/rocket_metal.mm` | `ec0d55119bf99a04f27d2d97a3b514a26e4151938f99e7d82732f1e8377e1a27` |
| `src/metal/rocket_metal.metal` | `679e9f6b3a480663d58eb7f2f712bca35528d8da7d628e2c0ccfbdfc91e00294` |
| `tests/test_metal.cpp` | `3c5e22b064c56b611eedf16f54e77c004e0d661d8a90d0aa55b04c0c1e9c1cbb` |
| `README.md` | `fc5e30d515e77173c7852a4fc85b6cb423d22c6d5163936988936ba920b337ae` |

## 性能实验复现脚本

以下为本轮实际使用的临时脚本。可将其另存到临时目录，在仓库根目录执行：对每个待比较版本运行 `python3 <临时目录>/bench.py build <版本名>`，然后运行 `python3 <临时目录>/bench.py compare <修改前> <修改后> 8192 16384 32768`。它复制源码、构建独立二进制并交替计时；正式 runner 无需修改。

<details><summary>bench.py</summary>

```python
from pathlib import Path
import sys,json,subprocess,statistics,shutil,difflib
ROOT=Path.cwd();W=Path(__file__).resolve().parent
FILES=['src/metal/rocket_metal.mm','src/metal/rocket_metal.metal']
def build(name):
 d=W/name;d.mkdir(exist_ok=True)
 for f in FILES:
  t=d/f;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,t)
 (d/'rocket_metal_source.h').write_text('static const char* rocket_metal_source = R"MSL('+Path(FILES[1]).read_text()+')MSL";\n')
 (d/'runner.cpp').write_text(Path('src/runner.cpp').read_text().replace('!trace_logits || forced_tokens.size()', 'forced_tokens.size()'))
 flags=['clang++','-std=c++17','-O3','-Wno-unknown-pragmas','-Iinclude','-I'+str(d),'-Illama.cpp/include','-Illama.cpp/ggml/include','-Illama.cpp/vendor']
 subprocess.run(flags+['-fobjc-arc','-c',str(d/FILES[0]),'-o',str(d/'metal.o')],check=True)
 subprocess.run(flags+[str(d/'runner.cpp'),'src/rocket.cpp','src/rocket_adapter.cpp',str(d/'metal.o'),'-o',str(d/'runner'),'-Lllama.cpp/build/bin','-Wl,-rpath,'+str(ROOT/'llama.cpp/build/bin'),'-lllama','-lggml','-lggml-base','-lggml-cpu','-framework','Metal','-framework','Foundation'],check=True)
def compare(before,after,lengths):
 prompt=next(r['prompt'] for r in json.loads(Path('results/runtime_probe/samples.json').read_text()) if r['sample_id']==201)
 inp=W/'input.json';inp.write_text(json.dumps([dict(sample_id=201,prompt='\n\n'.join([prompt]*8),max_tokens=33,forced_tokens=list(range(1000,1033)))]))
 result={'before':before,'after':after,'samples':[]}
 for length in lengths:
  records={before:[],after:[]}
  for trial in range(3):
   for name in ([before,after] if trial%2==0 else [after,before]):
    out=W/f'{before}-{after}-{length}-{trial}-{name}.json'
    cmd=[str(W/name/'runner'),'--model',str(ROOT/'models/Llama-3.2-3B-Instruct-Q8_0.gguf'),'--batch',str(inp),'--output',str(out),'--rocket-mode','hsa','--stage2-backend','metal','--gpu-layers','99','--budget','1024','--kv-type','f16','--n-ctx',str(length+1024),'--context-len',str(length),'--threads','16']
    with out.with_suffix('.log').open('w') as log:proc=subprocess.run(cmd,stdout=log,stderr=log,timeout=300)
    if proc.returncode:raise RuntimeError(out.with_suffix('.log').read_text()[-5000:])
    r=json.loads(out.read_text())[0];m=r['metrics'];assert r['status']=='ok' and m['decode_forward_calls']==32 and m['prompt_tokens']==length
    records[name].append(m['decode_ms_per_forward']);print(length,name,trial,round(records[name][-1],4),flush=True)
  b=statistics.median(records[before]);a=statistics.median(records[after]);result['samples'].append({'context':length,'runs':records,'before_ms':b,'after_ms':a,'reduction_pct':100*(1-a/b)})
  (W/f'{before}-{after}.json').write_text(json.dumps(result,indent=2));print(result['samples'][-1],flush=True)
if sys.argv[1]=='build':build(sys.argv[2])
elif sys.argv[1]=='compare':compare(sys.argv[2],sys.argv[3],list(map(int,sys.argv[4:])))
```

</details>

输入样本文件 SHA-256：`5aa7b81a0e705aa9a529dd07969f9c722763b8b0a51f5d46fd9401daa8eb4c01`。
