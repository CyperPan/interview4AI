# 推理优化

> 文档定位：面向 LLM 推理面试的高频原理速记。重点不是背定义，而是回答时能够说清楚 `为什么有效`、`优化了哪个指标`、`新的 trade-off 是什么`。

## 目录

- [面试官最关心的指标](#面试官最关心的指标)
- [回答模板](#回答模板)
- [Continuous Batching](#continuous-batching)
- [PagedAttention](#pagedattention)
- [PD 分离](#pd-分离)
- [投机解码](#投机解码)
- [量化策略](#量化策略)
- [算子融合](#算子融合)
- [面试金句](#面试金句)

---

## 面试官最关心的指标

- **TTFT（Time To First Token）**：用户多久能看到第一个 token
- **TPOT / ITL（Time Per Output Token / Inter-Token Latency）**：后续 token 生成得是否流畅
- **Throughput**：单位时间能处理多少请求或多少 token
- **Accuracy**：量化、裁剪、蒸馏后模型效果是否还能接受

回答时不要只说“更快了”，要说明到底是 **TTFT 变短了、TPOT 变小了、吞吐提升了，还是显存压力下降了**。

---

## 回答模板

建议按这个顺序回答面试题：

1. **先给结论**：一句话回答它是什么
2. **再讲原因**：为什么它在 LLM 推理里重要
3. **点出瓶颈**：是 compute-bound、memory-bound、communication-bound 还是调度问题
4. **补 trade-off**：会引入什么额外复杂度或副作用

---

## Continuous Batching

### 什么是 Continuous Batching（持续批处理）？为什么它对 LLM 推理很重要？

**答：**

**核心定义：**

Continuous Batching（也叫 In-flight Batching）不是按“请求”组批，而是按“每一步 token 生成”动态调度 batch。某个请求一旦生成结束，系统会立刻把它移出，并把新的等待请求插入下一轮迭代。

**为什么重要：**

传统 Static Batching 必须等 batch 里最长的那个请求结束，短请求即使已经完成，也会继续占着 batch 位置，造成 GPU 空转和吞吐浪费。

```text
Static Batching:
时间 →
请求1: [==========]
请求2: [===============================]
请求3: [======]
           ↑
        短请求提前结束，但资源无法立刻复用
```

Continuous Batching 可以让 GPU 在绝大多数 step 都维持较高利用率，尤其适合 **输入长度和输出长度差异很大** 的在线推理场景。

```text
Continuous Batching:
时间 →
Iter1: [请求1][请求2][请求3]
Iter2: [请求1][请求2][请求3]
Iter3: [请求2][请求4]
Iter4: [请求2][请求4][请求5]
```

**它主要优化什么：**

- 更直接提升 **Throughput**
- 通常也会改善 GPU 利用率
- 但 **不保证 TTFT 一定变好**，因为调度策略、排队规则和公平性设计同样重要

**trade-off：**

- 调度器实现更复杂
- 需要更好的 KV Cache 生命周期管理
- 还要处理长短请求混跑时的公平性和饥饿问题

---

## PagedAttention

### 什么是 PagedAttention？它解决了什么问题？

**答：**

**核心问题：**

LLM 在 Decode 阶段要持续增长 KV Cache。传统做法经常按最大长度给每个请求预留连续显存，这会带来两类浪费：

- **内部碎片**：请求实际没用满预留空间
- **外部碎片**：明明还有空闲显存，但因为不连续而无法分配给新请求

```text
传统分配：
显存: [████    ] [████    ] [████    ]
      请求1(512)  请求2(1024) 请求3(512)
```

**PagedAttention 的做法：**

它借鉴了操作系统分页思想，把 KV Cache 切成固定大小的 Block，再通过 Block Table 管理逻辑地址到物理地址的映射。

```text
PagedAttention:
逻辑视角: 请求1 -> [Block0][Block1][Block2]
          请求2 -> [Block3][Block4]

物理视角: 显存 -> [B0][B3][B1][B4][B2][空闲]
```

**为什么有效：**

- 基本消除了外部碎片
- 内部碎片只会出现在最后一个未填满的 block 中，浪费量取决于 block 大小
- 更容易支持变长请求、请求 churn 和 continuous batching

**它主要优化什么：**

- 提高 KV Cache 的显存利用率
- 提高可支持的并发数和上下文长度
- 让调度器更容易回收和复用 KV 空间

**trade-off：**

- 需要维护额外的 Block Table 和索引逻辑
- 实现复杂度更高
- 在小模型、短上下文、低并发场景下，收益不一定显著

---

## PD 分离

### 为什么很多高性能推理系统会考虑做 Prefill-Decode Disaggregation（PD 分离）？

**答：**

**根本原因：**

Prefill 和 Decode 的算力特征不同：

| 阶段 | 计算特征 | 常见瓶颈 |
| --- | --- | --- |
| **Prefill** | 一次性处理整段 prompt，矩阵乘更大 | 更偏 **compute-bound** |
| **Decode** | 每次只生成一个 token，但要反复读取权重和 KV | 更偏 **memory-bound** |

如果把这两者混在同一批 GPU 上，长 prompt 的 Prefill 很容易打断正在进行的 Decode，导致 **尾延迟抖动**。

```text
混部时间线：
时间:  [====][====][========][====][====]
       D1    D2    P1(新请求) D1    D2
                   ↑
            Decode 被长 Prefill 干扰
```

**PD 分离方案：**

把 Prefill 和 Decode 放到不同 GPU 或不同机器上：

1. Prefill 节点专门处理长 prompt
2. Prefill 结束后，把 KV Cache 传给 Decode 节点
3. Decode 节点持续处理流式生成

**它主要优化什么：**

- 提升混合流量下的 **P99 延迟稳定性**
- 改善 Decode 的 **TPOT**
- 让两类节点可以分别按瓶颈优化

**trade-off：**

- 需要额外的 KV 传输链路，网络成本不可忽略
- 系统架构更复杂
- 如果流量模式简单或负载不高，收益未必能覆盖复杂度

---

## 投机解码

### 什么情况下投机解码（Speculative Decoding）能够带来明显收益？

**答：**

**核心逻辑：**

投机解码用一个更小、更快的 Draft 模型先猜出多个 token，再让大模型一次性验证；如果前面多个 token 都被接受，就相当于减少了大模型逐 token 解码的次数。

```text
Step 1: Draft 模型快速生成 K 个候选 Token
        [Token1][Token2][Token3][Token4]

Step 2: Target 模型并行验证
        [✓][✓][✓][✗]

Step 3: 接受前 3 个，从第 4 个开始重新生成
```

**什么情况下收益明显：**

- Draft 模型的 **Acceptance Rate** 足够高
- 大模型还有一定富余算力，可以承接“并行验证”
- workload 输出模式相对稳定，比如代码补全、模板化文本生成

**它主要优化什么：**

- 更直接改善 Decode 阶段的 **TPOT / ITL**
- 在合适负载下，也可能提升整体吞吐

**什么时候可能不划算：**

- 系统已经非常满载，尤其是 Decode 已经严重 memory-bound
- Draft 模型命中率不高
- 额外的 Draft 推理和控制逻辑抵消了节省下来的 Decode 成本

**面试中不要说得过于绝对：**

不要说“Speculative Decoding 只适合低 batch”。更准确的说法是：**它更适合有富余计算资源、Acceptance Rate 较高的场景；在高负载或强 memory-bound 的系统里，收益会明显下降，甚至可能负优化。**

---

## 量化策略

### 量化为什么能提速？INT4 和 FP8 你怎么选？为什么有些量化方案精度掉得很厉害？

**答：**

**为什么能提速：**

LLM 推理尤其是 Decode 阶段，很多时候是 **memory-bound** 的。量化把权重和激活变小后：

- 显存占用下降
- HBM 读写流量下降
- 在有对应硬件和 kernel 支持时，算子吞吐也可能提升

所以量化最稳定的收益通常先体现在 **省显存、提并发、降 TPOT**，而不一定总是直接把单请求延迟大幅打下来。

**怎么选 FP8、INT8、INT4：**

| 类型 | 更适合的场景 | 特点 |
| --- | --- | --- |
| **FP8** | Hopper 等有原生支持的 GPU，且希望兼顾速度和精度 | 动态范围较好，硬件支持强，部署路径相对自然 |
| **INT8** | 需要较成熟的工程方案，兼顾精度和性能 | 工业界方案成熟，很多框架支持较完整 |
| **INT4 / Weight-Only** | VRAM 或并发能力是首要瓶颈 | 压缩率高，但依赖 kernel 和反量化开销控制 |

**为什么有时掉点很厉害：**

很多 LLM 存在明显的激活或权重 outlier。如果直接做朴素 PTQ，量化范围会被极少数大值主导，导致大量正常值被压缩得过于粗糙，精度明显下降。

**常见改进方案：**

- **SmoothQuant**：把一部分量化难度从激活平滑地转移到权重
- **AWQ**：优先保护重要权重，做更适合 LLM 的 weight-only 量化
- **GPTQ**：通过近似二阶信息降低 PTQ 误差

**面试标准回答的关键：**

不要只说“INT4 更省显存，FP8 更准”。要补一句：**选型取决于硬件是否有原生支持、kernel 是否成熟、目标是极致容量还是稳健精度。**

---

## 算子融合

### 算子融合（Kernel Fusion）为什么重要？

**答：**

**核心原因：**

很多 LLM 推理热点，尤其是 Decode 阶段，瓶颈不在纯算力，而在 **HBM 访存和频繁的 kernel launch**。

如果一个逻辑公式被拆成多个 kernel：

1. 每次 launch 都有额外开销，通常是 **微秒级**
2. 中间结果需要写回 HBM
3. 下一个 kernel 又得把这些中间结果从 HBM 读出来

这会让本来就 memory-bound 的路径更慢。

**算子融合的思路：**

把多步操作放进同一个 kernel 中，让中间结果尽量停留在寄存器或 shared memory，而不是写回显存。

```cuda
// 融合前
// Kernel 1: tmp = x * w
// Kernel 2: tmp = tmp + b
// Kernel 3: out = activation(tmp)

// 融合后
// 在一个 kernel 内完成，减少中间结果回写 HBM
```

**为什么重要：**

- 减少 HBM 读写
- 减少 kernel launch 开销
- 提高热点路径的端到端效率

FlashAttention、融合 RMSNorm、融合反量化 GEMM 都是典型例子。

**trade-off：**

- kernel 更复杂，开发和调试成本更高
- 不一定所有场景都值得手写 fusion
- 需要结合 profiling 先确认热点路径再动手

---

## 面试金句

> "Continuous Batching 的核心不是把 batch 做大，而是让 GPU 在每一轮 token 调度里都尽量保持高利用率。"

> "PagedAttention 本质上是在用更灵活的 KV 内存管理，换更高的并发和更少的显存碎片。"

> "PD 分离的本质是把 compute-bound 的 prefill 和 memory-bound 的 decode 拆开，避免它们互相干扰。"

> "投机解码能不能加速，关键不只是小模型够不够快，更要看 acceptance rate 和系统是否还有富余计算资源。"

> "量化的稳定收益通常先体现在省显存和提并发，而不是任何场景下都直接换来线性提速。"
