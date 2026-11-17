# Stage 1 与 Stage 2：从删掉部分 KV 到每步只读部分 KV

当前实现的主线是：**Stage 1 在 prefill 后淘汰部分历史 KV，Stage 2 在每次 decode attention 中临时选择部分候选 KV。** 最新修改进一步优化了 Stage 1 的 CPU 数据布局，两个阶段的选择策略保持不变。

更新日期：2026-11-17（北京时间）。本文以 `main`（`330d43e`）为历史起点，讲解当前 `xiang/dev`（`402ee70`）的完整两阶段实现。代码路径相对仓库根目录，行号对应 `402ee70`。

阅读路线：第 3 节讲 Stage 1 的打分、淘汰和 CPU 优化；第 4–6 节讲 Stage 2 的粗筛、精算、模型接入和 Metal 优化；最后保留检查要点与五道自测题。最新实验保留了 Stage 1 的四路 K 打包，撤回了没有稳定收益的 Stage 2 softmax 改动。

## 1. 背景：两个阶段分别解决什么问题

### 整体背景：熟悉 KV cache 可以跳过

模型推理分为 **prefill** 和 **decode**：前者处理输入提示词，后者逐步生成后续 token。KV cache 保存历史 token 的 Key 和 Value，供后续 attention 使用。

可以把一次 attention 理解为：当前 Query 给历史 Key 打分，再按分数对 Value 加权求和。

```text
scores = Q @ K.T / sqrt(head_dim)
output = softmax(scores) @ V
```

历史越长，需要考虑的 KV 就越多。这个项目用两个不同层次的操作减少工作量：

- **Stage 1 决定哪些历史 KV 留在候选池里。** Prefill 后做一次淘汰。
- **Stage 2 决定这次 attention 使用候选池里的哪些 KV。** 每个 decode forward、每层重新选择。

对应实现：`src/runner.cpp:346`、`src/rocket.cpp:240`。

### 本次修改的直接背景

原来的 `main` **已经有两个算法函数**，但 runner 只调用 Stage 1。Stage 2 函数存在，不等于模型生成时使用了它。

| 部分 | 原来的 `main` | 当前实现 |
|---|---|---|
| Stage 1 | 已计算保留位置并删除 KV | 保留策略，补强数据读取、校验、四路点积和 K 打包 |
| Stage 2 | 有独立算法函数，未接入 runner 的模型 attention | 接入每层计算图，影响后续生成 |
| GPU HSA | 没有当前 Metal 实现 | 有 Metal kernels、缓存及图内执行 |

历史调用点是 `main` 的 `runner.cpp:244` 和 `runner.cpp:315`；当前接入点见 `src/runner.cpp:386`、`src/rocket_adapter.cpp:87`。

## 2. 核心直觉：用一个小例子串起来

以下保留数量和分数都是**教学用假设**，不是一次实测结果。

假设输入有 10 个 token，原始位置为：

```text
prompt_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

Stage 1 保留 5 个，其中最后 2 个属于必须保留的观察窗口：

```text
retained_positions = [0, 3, 6, 8, 9]
```

位置 `1、2、4、5、7` 从当前序列的有效 KV 中移除，后续 Stage 2 不能再选到它们。删除逻辑见 `src/runner.cpp:371`。

下一次 decode forward 处理位置 `10` 的 token，将它的 KV 加入候选池：

```text
candidate_positions = [0, 3, 6, 8, 9, 10]
```

假设这一层、这一组 attention heads 最终选中：

```text
selected_positions = [6, 8, 9]
```

这次只对这 3 个位置计算精确 attention，但 `[0, 3, 10]` **仍留在候选池里**，下一步可能被选中。候选追加和选择逻辑分别在 `src/rocket_adapter.cpp:115`、`src/rocket.cpp:409`。

**因此，Stage 1 是淘汰；Stage 2 是临时选择。Stage 2 的 `topk_decode` 不是 KV cache 的存储容量。**

## 3. 代码导览：Stage 1 改了什么

### 3.1 保留原来的选择策略

Stage 1 的核心思路是：**用提示词末尾几个 token 的 Query，判断前面的哪些 token 值得保留。** 当前 runner 用最后一层的信息做这件事，沿用原来的选择策略。

#### 第一步：指定并收集最后一层的 Q/K

```cpp
g_acts.target_layer = llama_model_n_layer(model) - 1;
```

这行代码只是指定捕获哪一层，**本身不读取张量，也不选择 token**。假设模型有 32 层，层编号从 0 开始，这里指定的就是第 31 层。见 `src/runner.cpp:319`。

随后执行 prefill 时，回调只接收这一层经过 RoPE（位置编码）后的 Q/K，并存入：

```cpp
g_acts.q  // 最后一层各 Query head、各 token 的 Q
g_acts.k  // 最后一层各 KV head、各 token 的 K
```

捕获逻辑见 `src/runner.cpp:63`。这里只用 Q/K 判断重要性，不需要 V。Prefill 完成后，runner 把这些数据传给 `compute_prefill_eviction()`，才开始选择保留位置。见 `src/runner.cpp:354`。

#### 第二步：用观察窗口的 Query 给历史候选打分

沿用第 2 节的例子：提示词有 10 个 token，观察窗口为 2，希望最终保留 5 个。

```text
历史候选位置 = [0, 1, 2, 3, 4, 5, 6, 7]
观察窗口位置 = [8, 9]  # 直接保留，不参与淘汰
```

虽然收集了全部 Q，但打分只用最后两个位置的 Q。GQA 中，多个 Query heads 共享同一个 KV head，构成一个 KV group。对每个 Query head，计算：

```python
# 示意代码：Q、K 均来自最后一层。
# group 是当前 Query head 所属的 KV group。
p8 = softmax(Q[head, 8] @ K[group, 0:8].T / sqrt(head_dim))
p9 = softmax(Q[head, 9] @ K[group, 0:8].T / sqrt(head_dim))

head_scores[group] += p8 + p9
```

再累加这个 KV group 中其他 Query heads 的贡献。越受末尾这些 Query 关注的历史位置，累计分数越高。

当前实现的 softmax **只在观察窗口之前的历史候选中归一化**，不包含位置 `8、9`。因此，这里是在重新计算用于淘汰的分数，不是直接读取模型完整 attention 矩阵的一部分。见 `src/rocket.cpp:135`、`src/rocket.cpp:176`。

#### 第三步：局部最大值池化，再汇总各组分数

为了让高分位置附近的 token 也有机会留下，每个 KV group 内先做局部最大值池化。假设 `kernel_size = 3`：

```python
# 示意代码：边界位置只使用存在的邻居。
neighbors = range(max(0, i - 1), min(non_obs_k, i + 2))
pooled[group, i] = max(head_scores[group, j] for j in neighbors)

# 各 KV group 汇总成一份全局 token 分数。
score[i] = sum(pooled[group, i] for group in kv_groups)
```

所以高分可能来自这个位置本身，也可能来自它的邻居。注意执行顺序是：先在每组内部池化，再跨组求和。见 `src/rocket.cpp:189`、`src/rocket.cpp:212`。

#### 第四步：取高分历史 token，加上观察窗口

保留预算是 5，观察窗口已经占了 2，所以历史部分只选 3 个：

```python
history_keep = prompt_budget - obs_w  # 本例为 5 - 2 = 3
# obs_w 是按实际输入长度截断后的观察窗口大小。
```

为单独展示 Top-K，这个分数示例取 `kernel_size = 1`，即不向邻居扩展分数；以下是假设的汇总分数：

```text
历史位置:  0     1     2     3     4     5     6     7
最终分数: 0.9   0.2   0.1   0.8   0.3   0.2   0.7   0.1

历史 Top-3 = [0, 3, 6]
观察窗口   = [8, 9]
最终保留   = [0, 3, 6, 8, 9]
```

代码用 `std::nth_element` 取高分候选，再加入观察窗口，最后按原始位置排序。见 `src/rocket.cpp:219`。如果提示词本身没有超过保留预算，就直接保留全部位置，无须淘汰。

runner 随后移除其他位置的 KV。**各层删的是相同位置，但各层保留的仍是自己原来的 K/V 数值**，不会把最后一层的 KV 复制给其他层。见 `src/runner.cpp:371`。

这就得到一份所有层共享的保留名单，而不是每层或每个 head 各选一份。概括起来：**最后一层负责给历史 token 的重要性打分，产生全局位置名单，再用这份名单统一裁剪各层 KV cache。**

### 3.2 把“读到 Q/K”变成可靠的数据交接

原来读取 prefill 张量时，长度为 1 的小批次会被跳过。现在按张量的实际字节步长读取，并检查形状、类型和最终捕获数量。见 `src/runner.cpp:76`、`src/runner.cpp:342`。

为什么重要？假设提示词分成 `8 + 1` 两批处理，最后一个 token 仍然属于 prefill。漏掉它，就可能让 Stage 1 用不完整的 Q/K 做淘汰。

这次还把保留位置显式交给 Stage 2：

```cpp
adapter->initialize(config, kept_positions, full_attention);
```

位置：`src/runner.cpp:386`。

回到例子，这一步交出去的是 `[0, 3, 6, 8, 9]`，而不只是“剩下 5 个 token”这个数字。

### 3.3 CPU 优化：四个独立点积，加上连续的数据布局

这部分经过两步优化。提交 `c8bf4ee` 先把一次算一个 Key，改为同时推进四个独立点积；最新提交 `402ee70` 则让这四个点积需要的数据在内存中排得更合适。

#### 原来：四个点积已经交错执行，但 K 仍按 token 存储

假设四个 token 的 Key 都只有三个维度，原始布局是：

```text
K0 = [a0, a1, a2]
K1 = [b0, b1, b2]
K2 = [c0, c1, c2]
K3 = [d0, d1, d2]

内存顺序 = [a0, a1, a2, b0, b1, b2, c0, c1, c2, d0, d1, d2]
```

计算第 0 个维度时，需要读取 `a0、b0、c0、d0`，但这些数不是连续存放的。独立运算已经存在，数据访问仍有改进空间。

#### 现在：每四个 token 打包一次，相同维度放在一起

最新实现为每个 KV group 建立临时 `packed_keys`，把上述数据变为：

```text
打包后 = [a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2]
```

这对应 `[token_block, dim, lane]` 的布局，每块有四个 token，`lane` 表示块内第几个 token。实际写入代码是：

```cpp
dst[4 * d + lane] = src[lane * cfg.dim_head + d];
```

位置：`src/rocket.cpp:123`。打包发生在 Query head 和观察窗口循环之外，因此同一组历史 K 可以被多个 Query 重复使用。

点积循环随后连续读取同一维度上的四个 Key：

```cpp
const float q = q_ptr[d];
dot0 += q * k_ptr[4 * d];
dot1 += q * k_ptr[4 * d + 1];
dot2 += q * k_ptr[4 * d + 2];
dot3 += q * k_ptr[4 * d + 3];
```

位置：`src/rocket.cpp:147`。

直觉是：读取一个 `q[d]`，就拿它和相邻的四个 Key 值相乘，使编译器更容易使用 SIMD，也就是用一条向量指令处理多个独立数据。

**每个 token 仍有自己独立的完整点积，按原来的维度顺序累加。** 不足四个 Key 的尾部继续读取原始 K；softmax、池化和 Top-K 不变。临时打包也不会修改模型实际存储的 KV cache。见 `src/rocket.cpp:166`。

代价是多一次打包和一块临时缓冲区。8k、`head_dim=128` 时，每个正在处理的 KV group 约增加 4 MiB；当前默认构建按组串行处理，启用 OpenMP 后应计入并发组的内存。边界覆盖见 `tests/test_rocket.cpp:152`。

### 3.4 最新优化的实际收益

本轮在 Apple M2、16 GiB 内存上，使用 Llama-3.2-3B-Instruct-Q8_0、F16 KV、Metal 全层 offload，固定 `token_budget=1024`、观察窗口 32、池化宽度 63、4 CPU threads。

LongBench 取 `multi_news`、`trec`、`passage_retrieval_en`、`repobench-p` 各前两条，保留任务标准生成上限。2k/4k/8k 是输入上限，长输入做中间截断，短输入不填充。起点、仅 E1、E1+E2 三个版本共完成 72 次配对运行；这里比较起点 `e492819` 与只含四路 K 打包的 E1（最终保留版本 `402ee70`）。

| 输入上限 | Stage 1 淘汰耗时降低 | 生成文本与分数 |
|---|---:|---|
| 2048 | 8.60% | 全部一致 |
| 4096 | 4.07% | 全部一致 |
| 8192 | 7.98% | 全部一致 |

百分比是逐样本耗时比的中位数：`100 * (1 - median(E1_ms / baseline_ms))`。收益集中在 Stage 1 淘汰计算；4k 的改善幅度仍有较大测量噪声，整段推理尚未确认稳定加速。算法级对照和固定 token 的真实模型检查也验证了保留位置、attention 和 logits 的一致性。

实验细节和原始数据入口见 `docs/optimize_exp_1116/e1.md`、`docs/optimize_exp_1116/results.md`。这些结论适用于本轮模型、硬件和任务子集。

## 4. 代码导览：Stage 2 怎样选择 KV

### 4.1 先粗筛，再精算

Stage 2 的 HSA，即混合稀疏注意力，分成两个层次：

1. 用较便宜的近似分数选 token。
2. 对选中的 token，用完整维度重新计算 attention。

**“精确”指选中子集上的 attention，并不意味着结果等于完整候选池的 dense attention。** 见 `src/rocket.cpp:335`、`src/rocket.cpp:426`。

### 4.2 粗筛如何变便宜

这里有两个缩减：少看一些 Query 维度，把多个候选 token 合成一个 chunk。

**GQA group** 是共享同一组 K/V 的多个 Query heads。代码先汇总组内各 Query 在每个维度上的绝对值，选出重要的 `r` 个维度。这里是**维度选择**，不是新增一个学习出来的投影矩阵。见 `src/rocket.cpp:277`。

然后，按候选池顺序分 chunk。沿用我们的例子，设 `chunk_size = 2`：

```text
chunk 0: [0, 3]
chunk 1: [6, 8]
chunk 2: [9, 10]
```

注意：分块依据是候选数组中的相邻关系；原始 token 位置不必连续。

CPU reference 按需要计算 chunk 在选中维度上的 Key 极值；Metal 路径缓存各维度的最小值和最大值，再根据 Query 的组内符号选择代表值。用这些代表值计算近似分数，经过每个 Query head 的 softmax，再在组内汇总。见 `src/rocket.cpp:296`、`src/rocket.cpp:335`、`src/rocket.cpp:374`、`src/metal/rocket_metal.metal:49`。

假设最后得到：

```text
chunk 0 score = 0.1
chunk 1 score = 0.6
chunk 2 score = 0.3
topk_decode = 3
```

同一 chunk 的 token 共享近似分数，因此选中 chunk 1 的两个 token，再从 chunk 2 选一个。同分时位置靠前的候选优先，得到 `[6, 8, 9]`。见 `src/rocket.cpp:394`、`src/rocket.cpp:409`。

### 4.3 精算不会继续使用粗筛分数

选好位置后，代码用完整的 `head_dim` 重算 Q/K 点积，再做 softmax 和 V 加权：

```cpp
for (int d = 0; d < cfg.dim_head; ++d) {
    exact_dot += q_h[d] * k_t[d];
}
```

位置：`src/rocket.cpp:440`。

可以把粗筛理解成“挑出要认真阅读的资料”，精算则是“读原始资料后重新给出判断”。粗筛分数只负责入选，不直接充当最终 attention 权重。

## 5. 最关键的工程变化：让模型真正使用 Stage 2

### 5.1 替换 attention 节点，而不是在生成后额外算一次

只在 `llama_decode()` 返回后调用 HSA，无法改变已经计算好的模型输出。因此这里修改了 llama.cpp 的 attention 构图路径：启用 HSA 时，使用自定义 attention 节点的输出，再继续原来的后续计算。

构图入口在 `patches/llama-stage2.patch:196`，自定义节点在 `src/rocket_adapter.cpp:133`。

runner 每步的实际调用顺序是：

```cpp
if (enable_hsa) adapter->begin_step(n_cur);
if (llama_decode(ctx, batch_decode) != 0) throw std::runtime_error("Decode failed at position " + std::to_string(n_cur));
llama_synchronize(ctx);
if (enable_hsa) adapter->finish_step();
```

位置：`src/runner.cpp:451`。

HSA 就发生在中间的 `llama_decode()` 内。`finish_step()` 检查本步是否执行完所有模型层，以及 GPU 是否报告错误。见 `src/rocket_adapter.cpp:125`。

还有一个容易混淆的细节：**第一个生成 token 来自 prefill logits。** 它作为下一次 decode 的输入时，HSA 才开始工作。见 `src/runner.cpp:421`。

### 5.2 区分原始位置、候选下标和物理槽位

我们的候选池是：

```text
candidate_positions = [0, 3, 6, 8, 9, 10]
```

选中候选下标 `2`，代表原始位置 `6`；它在 KV buffer 中的物理槽位又可能是另一个数字。

`RocketAdapter::refresh_slots()` 从实际 KV cache 读取位置，建立候选下标到物理槽位的映射，并核对是否与候选列表一致。见 `src/rocket_adapter.cpp:93`。

GPU 最终也是通过映射读 KV：

```cpp
read_kv(kr, group, uint(slots[t]), d, p.kh, p.kt, p.kd)
```

位置：`src/metal/rocket_metal.metal:300`。

这一步保证“选中了正确的历史 token”和“读到了那个 token 的 KV”是同一件事。

## 6. Metal 优化：减少搬运、重复计算和排序

### 6.1 不只是把一个函数放到 GPU 上

当前 Metal 实现包含摘要、Query 维度选择、近似打分、Top-K、精确打分、softmax 和加权求和。调度顺序在 `src/metal/rocket_metal.mm:313`。

在支持的 buffer 路径上，这些操作直接编码进 GGML 已有的 Metal command encoder，也就是记录 GPU 工作的命令编码器，读取模型的 Q/K/V，并写回模型需要的输出。

这样就不必每层都单独提交一份 HSA 工作，再等待和搬运结果。同步路径仍为不能直接绑定的输入保留。见 `src/metal/rocket_metal.mm:278`、`src/metal/rocket_metal.mm:334`。

### 6.2 缓存旧 chunk 的摘要

生成时通常是在候选池尾部追加 token，前面的完整 chunk 不变。当前实现因此复用旧摘要，更新未满的尾 chunk 和新增 chunk。

但复用有前提：层、维度、chunk 大小、存储步长和已有槽位前缀必须匹配；不匹配就重建。见 `src/metal/rocket_metal.mm:226`、`src/metal/rocket_metal.mm:231`。

提交 `6a6a8c7` 还调整了摘要布局：让相邻 GPU 线程读取相邻 chunk 时，访问的地址也相邻。使用容量步长而不是每步变化的 chunk 数作为步长，避免追加一个 chunk 就让全部旧摘要地址改变。容量跨越扩容边界、步长改变时，摘要会重建。见 `src/metal/rocket_metal.metal:60`、`src/metal/rocket_metal.mm:212`。

### 6.3 从完整排序改成找到入选门槛

提交 `e492819` 将 Metal token 选择从完整排序改成 **weighted radix selection**：按分数的二进制位逐轮筛选，定位第 K 个 token 对应的分数门槛。

直觉上，我们只要知道谁能入选，不需要知道所有候选的完整名次。

其中的 **weighted** 很关键：一个 chunk 代表多个 token，所以计数时用该 chunk 的实际 token 数作权重。尾 chunk 如果只有一个 token，就只能贡献一个名额。

找到门槛后，高于门槛的全部入选，等于门槛的按候选顺序补足，直接输出时间顺序的下标。见 `src/metal/rocket_metal.metal:190`。

**这项优化保持的是“对相同近似分数执行相同的 Top-K 选择”，不是把近似分数变成精确分数。**

### 6.4 加权求和：提前加载 V，仍按原顺序累加

Stage 2 的最后一步是将选中 token 的 V 按精确 attention 权重求和。当前 Metal kernel 每次先加载四个 V，再依次累加四项；不足四项的尾部单独处理。见 `src/metal/rocket_metal.metal:305`。

调度时，该 kernel 使用 64 个线程组成一个 threadgroup。见 `src/metal/rocket_metal.mm:333`。这里保留每个输出元素的原始求和顺序，没有把一个长和拆成多份后再合并；历史实验中，改变求和顺序的方案曾出现长输入数值漂移。已有优化的取舍记录在 `HSA_OPTIMIZATION_LOG.md`。

### 6.5 最新尝试：softmax 的 SIMD 归约已撤回

本轮还尝试把 Stage 2 softmax 二叉归约的最后五层改成 SIMD shuffle，在保持相加配对顺序的同时减少线程组同步。候选 E2 包含 E1，因此比较 `E1 -> E1+E2`，单独衡量 Stage 2 的增量效果。

在第 3.4 节的同一组 LongBench 评测中，decode 耗时的配对中位降低分别为：

```text
2k: -0.03%   # 负数表示候选略慢
4k:  1.19%
8k: -0.24%
```

三个长度的 95% 区间都跨过零，没有确认稳定收益，因此撤回候选代码及专用测试。**撤回的只是这次 softmax 优化；Stage 2 的图内 HSA、摘要缓存、weighted radix selection 和已有加权求和优化全部保留。** 当前 softmax 仍使用原来的线程组归约，见 `src/metal/rocket_metal.metal:141`。

尝试过程、数值一致性检查和弃用 patch 保存在 `docs/optimize_exp_1116/e2.md` 与同目录 `e2.patch`。本轮保留的修改只涉及 Stage 1 的项目侧代码和测试，没有进一步修改 `llama.cpp/`。

## 7. 阅读这类修改时，要重点检查什么

### 7.1 运行过函数，不等于接入了模型

应检查 HSA 输出是否真的通向最终 logits。仓库专门比较全选和稀疏选择后的 logits，避免自定义节点只是无效的旁路计算。见 `tests/test_stage2_integration.py:153`。

### 7.2 优化必须保留边界规则

Stage 1 要检查四路打包的尾部和非向量宽度的 head dimension；Stage 2 要检查 chunk 尾部不足、同分和接近全选。这里应同时核对选中下标和 attention 输出，而不是只看生成文本是否像样。见 `tests/test_rocket.cpp:152`、`tests/test_metal.cpp:318`。

### 7.3 缓存和异步执行增加了状态管理责任

上一请求的摘要、上一层的临时空间、旧的槽位映射都不能被错误复用。当前实现有请求 reset、逐层缓存和同步后错误检查；修改这些路径时要一起追踪它们的生命周期。见 `src/rocket_adapter.cpp:53`、`src/metal/rocket_metal.mm:226`。

### 7.4 减少有效 KV 数，不等于立即减少已分配内存

Stage 1 删除的是有效序列位置；llama.cpp 可能仍保留预分配的 buffer。当前代码已经把 KV 大小指标注明为逻辑估算，不能直接当作进程或 GPU 实际内存节省。最新 K 打包还有单独的临时 CPU 内存开销。见 `src/runner.cpp:542`、`src/rocket.cpp:124`。

## 8. 五道自测题

### 题 1：Stage 2 本步没有选中位置 3，下一步怎么办？

- A. 位置 3 已永久删除。
- B. 位置 3 仍在候选池，下一步可以被选中。
- C. 必须重新运行 prefill 才能使用它。
- D. 只有其他层能够继续使用它。

<details>
<summary>答案</summary>

**B。** Stage 2 每步选择 attention 子集，不删除未选中的候选。A 混淆了 Stage 1 的淘汰与 Stage 2 的选择。

</details>

### 题 2：为什么不能在 `llama_decode()` 返回后才计算 HSA，来实现这次 attention 替换？

- A. HSA 只能接收整数输入。
- B. 返回后就无法读取任何 KV。
- C. 本次模型输出已经计算完成，必须把 HSA 输出接进计算图。
- D. HSA 必须替换整个 Transformer 层。

<details>
<summary>答案</summary>

**C。** 关键是输出依赖关系。返回后可能仍能读取 KV，但额外计算不会自动改变已经得到的 logits；也不需要像 D 那样替换整个层。

</details>

### 题 3：一个 chunk 有 2 个 token，尾 chunk 只有 1 个；为什么 radix selection 要按 token 数加权？

- A. 最终预算限制 token 数，而不是 chunk 数。
- B. 尾 chunk 必须永远优先入选。
- C. 权重用于改变最终精确 attention 的 softmax。
- D. 权重用于把 F16 KV 转为 F32。

<details>
<summary>答案</summary>

**A。** 把每个 chunk 都算作同样大小，可能选不满或选错 Top-K。C 容易混淆“选择阶段的计数权重”和“最终 attention 权重”，两者用途不同。

</details>

### 题 4：最新 Stage 1 K 打包优化改变了什么，又保留了什么？

- A. 只看原来四分之一的维度。
- B. 四个 Key 共用一个最终分数。
- C. 让四个 Key 的相同维度连续存储，每个点积仍按原维度顺序累加。
- D. 淘汰逻辑从 CPU 移到了 GPU。

<details>
<summary>答案</summary>

**C。** 原来已经有四个独立累加器，最新修改进一步改善它们读取的数据布局。每个 token 仍使用完整维度、得到独立分数，各自的求和顺序不变。A 混淆了这项 CPU 优化与 Stage 2 的维度粗筛。

</details>

### 题 5：设置 `--hsa-full`，应优先与哪个路径比较，判断接入是否正确？

- A. 未淘汰任何 KV 的 `off` 路径。
- B. 保留相同 Stage 1 位置、使用原生 attention 的路径。
- C. 不同预算、但生成长度相同的 HSA 路径。
- D. 只比较最终生成文本，不比较 attention 或 logits。

<details>
<summary>答案</summary>

**B。** 全选 HSA 应与同一候选池上的原生 attention 对齐。A 的候选池可能不同，即使接入完全正确也会产生不同结果；现有对照见 `tests/test_stage2_integration.py:135`。

</details>
