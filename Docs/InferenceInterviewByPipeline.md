# 按 LLM 推理链路整理的面试问题与答案

> 文档定位：把 LLM 推理面试题按“真实推理链路”组织，而不是按零散主题堆放。这样更适合建立整体系统观，也更方便在面试中按阶段展开。

---



在线 LLM 推理链路可以整理成：

1. **请求进入与 Tokenization**
2. **调度与 Batching**
3. **Prefill**
4. **Attention 与 KV Cache**
5. **FFN / MoE**
6. **Decode**
7. **Speculative Decoding（可选）**
8. **Sampling 与 Output**
9. **跨阶段优化与指标**

---

## 目录

- [模块 1：请求进入与 Tokenization](#模块-1请求进入与-tokenization)
- [模块 2：调度与 Batching](#模块-2调度与-batching)
- [模块 3：Prefill](#模块-3prefill)
- [模块 4：Attention 与 KV Cache](#模块-4attention-与-kv-cache)
- [模块 5：FFN / MoE](#模块-5ffn--moe)
- [模块 6：Decode](#模块-6decode)
- [模块 7：Speculative Decoding（可选）](#模块-7speculative-decoding可选)
- [模块 8：Sampling 与 Output](#模块-8sampling-与-output)
- [模块 9：跨阶段优化与指标](#模块-9跨阶段优化与指标)

---

## 模块 1：请求进入与 Tokenization

### 这一模块在干什么

用户请求先经过 API 接入、排队、文本预处理和 tokenization，之后才会进入真正的模型计算。这个阶段常被忽略，但在短请求和高 QPS 场景下，它会直接影响 TTFT。

### 高频问题 1：为什么推理岗位也会问 tokenizer，而不只是问模型和 kernel？

**答：**

因为 tokenizer 虽然在模型之前，但它会直接影响请求的首 token 延迟。短请求场景下，tokenization 在 TTFT 里的占比并不低；高并发场景下，它还可能成为 CPU 侧瓶颈。所以面试官问 tokenizer，本质上是在看你是不是把推理链路理解成一个完整系统，而不是只盯着 GPU kernel。

### 高频问题 2：为什么有些系统会把 tokenization 放在 CPU 侧并做并行化？

**答：**

因为 tokenization 逻辑复杂、分支多，不适合放到 GPU 上做；而且 CPU 可以多线程并行处理多个请求，并和 GPU 推理做 overlap。这样设计的目标不是让 tokenizer 更“高级”，而是避免 GPU 等待输入准备，降低端到端 TTFT。

### 高频问题 3：一次完整的 LLM 在线推理链路怎么讲？

**答：**

可以按下面这个顺序回答：

`请求进入 -> tokenization -> scheduler 排队 -> prefill 生成 KV cache -> decode 循环执行 attention/FFN -> sampling -> 流式输出或结束`

这样回答的好处是，你既交代了模型内部计算，也交代了 serving 侧的调度点。面试里最好顺手补一句：`prefill 更偏 compute-bound，decode 更偏 memory-bound`，这样层次会更完整。

---

## 模块 2：调度与 Batching

### 这一模块在干什么

调度器负责决定请求什么时候进入 GPU、如何组成 batch、长短请求如何共存、系统在延迟和吞吐之间如何取舍。这个模块决定了线上服务是不是“稳”。

### 高频问题 1：静态 batching 和 continuous batching 的区别？

**答：**

静态 batching 是先攒一批请求，再固定 batch 跑完；continuous batching 则允许请求在运行过程中动态加入和退出。LLM 服务里请求长短差异很大，所以静态 batch 很容易被长请求拖住，而 continuous batching 可以更充分地利用 GPU，提高吞吐和资源利用率。

### 高频问题 2：Continuous batching 为什么适合 LLM serving？

**答：**

因为在线 LLM 服务里，请求的 prompt 长度和输出长度都高度不均匀。continuous batching 的价值在于让 GPU 的每个 step 都尽量有活干，而不是因为某几个长尾请求导致整个 batch 空转。它最直接改善的是 throughput，但不保证 TTFT 一定更优，因为调度公平性和排队策略也很关键。

### 高频问题 3：什么是 in-flight batching？

**答：**

in-flight batching 本质上就是 continuous batching 的另一种说法，强调“一个正在运行的 batch 可以持续插入新请求”。如果面试官提这个词，不要被术语绕住，直接回答：**它就是 token 级动态调度 batch 的实现方式。**

### 高频问题 4：为什么 admission control 在多租户推理系统里是必需的？

**答：**

因为在线服务不是只看“能不能跑”，还要看不同租户之间的资源隔离、优先级和过载保护。如果没有 admission control，某个大流量租户可能会把队列打爆，导致高优先级请求的 TTFT 和 P99 延迟一起恶化。

---

## 模块 3：Prefill

### 这一模块在干什么

Prefill 指的是把整段 prompt 一次性送入模型，完成所有 layer 的前向计算，并生成初始 KV Cache。它通常是长 prompt 场景下的主要计算来源。

### 高频问题 1：什么是 prefill，什么是 decode？

**答：**

Prefill 是把整段 prompt 一次性过模型，算出各层 KV Cache；decode 是后续逐 token 生成，每一步只输入最新 token，但会读取历史 KV。面试里这题的关键不是只会下定义，而是要补一句：**prefill 更偏 compute-bound，decode 更偏 memory-bound，所以优化手段不一样。**

### 高频问题 2：为什么 prefill 更偏 compute-bound？

**答：**

因为 prefill 会同时处理整段输入序列，矩阵规模更大，算子的 arithmetic intensity 更高，更容易把 Tensor Core 和算力资源吃满。相比之下，decode 每次只处理一个新 token，计算量不大，但要频繁搬运权重和 KV，所以更受带宽限制。

### 高频问题 2.5：用 Roofline 模型怎么分析 prefill 和 decode？

**答：**

Roofline 模型横轴是算术强度（FLOPs/Byte），纵轴是实际吞吐（FLOPs/s）。拐点 = 峰值算力 / 峰值带宽。

```
性能 (FLOPs/s)
    │        _______________  ← 峰值算力
    │       /     ★ Prefill (compute-bound)
    │      /
    │     /
    │    / ★ Decode batch=1 (memory-bound)
    │   /
    └──────────────────────── 算术强度 (FLOPs/Byte)
```

**Prefill：** 处理整段 prompt（长度 s），做大规模 GEMM。算术强度 ≈ O(s)，随 prompt 长度线性增长，通常落在 **compute-bound 区域**。

**Decode：** 每步只处理 1 个 token（batch=1 时是 GEMV）。算术强度 ≈ O(1)，非常低，落在 **memory-bound 区域**。

**H100 SXM 数值示例：**
- BF16 峰值算力 ~990 TFLOPS，HBM 带宽 ~3.35 TB/s → 拐点 ≈ **295 FLOPs/Byte**
- Decode (batch=1, 7B)：算术强度 ≈ 1~2 → 极度 memory-bound
- Decode (batch=64)：算术强度 ≈ 64~128 → 接近拐点
- Prefill (seq_len=2048)：算术强度 > 300 → compute-bound

**面试关键点：** 增大 batch size 可以把 decode 从 memory-bound 区域往 compute-bound 方向推，这也是 continuous batching 的价值所在。

### 高频问题 3：什么是 chunked prefill？

**答：**

chunked prefill 是把很长的 prompt 切成多个块分步执行，而不是让单个请求一次性独占 GPU 很久。它的核心作用不是让单个请求绝对更快，而是改善混合流量下的尾延迟和公平性，让 decode 请求不会一直被超长 prefill 卡住。

### 高频问题 4：Prefix caching 是什么？它主要优化什么？

**答：**

Prefix caching 是多个请求拥有相同前缀时，直接复用已有前缀 KV，而不是每次都重新做 prefill。它最直接改善的是 TTFT，因为节省了重复 prompt 的前向计算；在系统提示词固定、RAG 模板固定、多轮对话前缀高度重复的业务里收益很明显。

---

## 模块 4：Attention 与 KV Cache

### 这一模块在干什么

Attention 是 Transformer 的核心计算；KV Cache 则是把历史 token 的 K/V 保存下来，避免 decode 时重复计算。这个模块是理解长上下文推理和 decode 瓶颈的关键。

### 高频问题 1：为什么 KV Cache 是推理优化核心？

**答：**

因为自回归生成每出一个新 token，都要依赖之前所有 token 的 K/V；如果不缓存，每一步都要重算整段上下文，计算量会爆炸。KV Cache 让计算省下来了，但也把显存容量和带宽压力推高了，所以 PagedAttention、GQA、prefix caching 这些优化本质上都在处理 KV 问题。

### 高频问题 2：什么是 PagedAttention？它解决了什么问题？

**答：**

PagedAttention 的核心是把 KV Cache 按固定大小 block 管理，而不是给每个请求预留一整段连续显存。这样可以显著缓解显存碎片问题，更好支持变长请求和 continuous batching。它主要改善的是 KV 利用率、可服务并发和长上下文能力。

### 高频问题 3：MHA、MQA、GQA 的区别是什么？为什么 GQA/MQA 对推理更有利？

**答：**

MHA 是每个 Q head 都有自己的 K/V head；MQA 是多个 Q head 共享一组 K/V；GQA 则是在两者之间做分组共享。GQA/MQA 对推理更友好，是因为 decode 阶段的热点通常是读 KV，而不是做大矩阵计算；减少 K/V head 数量，就等于直接减少 KV Cache 体积和访存压力。

### 高频问题 4：FlashAttention 和 PagedAttention 的区别是什么？

**答：**

FlashAttention 优化的是 **Attention 计算过程中的 HBM 读写**，本质是 IO-aware kernel；PagedAttention 优化的是 **KV Cache 的内存分配与管理**，本质是更高效的显存布局。一个偏计算 kernel，一个偏内存管理，两者不是替代关系，而是可以叠加使用。

### 高频问题 5：如果现场让你手撕 `MHA / linear attention / MLA`，面试官真正想看什么？

**答：**

不是单纯看你能不能把代码写出来，而是看你有没有“结构 + 复杂度 + 访存”三层意识。`MHA` 要写清 `Q / K / V` 投影、mask 和多头 reshape；`linear attention` 要体现把二次复杂度改写成可累计形式；`MLA` 要体现“先压缩再参与注意力”这件事。面试里如果只会抄公式，不会解释它们分别在省什么，分数通常不会高。

### 高频问题 6：`MHA`、`GQA/MQA`、`MLA` 在 decode 阶段最大的差别是什么？

**答：**

最大的差别通常不在“这一步多了多少 FLOPs”，而在 `KV Cache` 体积和读取带宽。`MHA` 需要为每个 head 保留完整 K/V；`GQA/MQA` 通过共享 K/V 头减少缓存；`MLA` 进一步把缓存压到更紧凑的 latent 表示上，所以长上下文时更容易降低显存占用和 decode 访存压力。面试里最好直接点明：**decode 阶段最怕的是搬不动 KV，不是算不动 attention。**

### 高频问题 7：MLA（Multi-head Latent Attention）的原理是什么？

**答：**

MLA 是 DeepSeek-V2 提出的 KV Cache 压缩方法。核心思想是将 KV 压缩到低维 latent 空间，只缓存 latent 向量。

```
传统 MHA/GQA: KV Cache = [K_1,...,K_H, V_1,...,V_H]   (大)
MLA:         hidden → 下投影 → latent vector c (维度 d_c << H×d_head)
             KV Cache = [c]   (极小)
             推理时: c → 上投影恢复 K,V → 做 attention
```

通过数学变换，可以将上投影 fuse 进 Q 的投影中，避免显式恢复 K/V。DeepSeek-V2 的 KV Cache 压缩到 GQA 的约 **1/5~1/10**。

### 高频问题 7.5：MHA → MQA → GQA → MLA 的 KV Cache 压缩对比

**答：**

| 方法 | KV head 数 | KV Cache 大小（相对 MHA） | 表达能力 | 代表模型 |
|------|-----------|-------------------------|---------|---------|
| **MHA** | H | 1x | 最强 | GPT-3, BERT |
| **MQA** | 1 | 1/H | 最弱 | PaLM, StarCoder |
| **GQA** | G (1<G<H) | G/H | 较强 | LLaMA 2/3, Qwen2 |
| **MLA** | latent | ~1/5~1/10 x GQA | 强 | DeepSeek-V2/V3 |

面试里最好直接点明：**所有 KV 压缩手段的核心目标都是减少 decode 时的 KV 读取量，缓解 memory-bound 瓶颈。**

### 高频问题 8：为什么很多 attention 题最后都会追问”访存怎么估”？

**答：**

因为线上推理性能往往不是输在公式推不出来，而是输在你不知道系统到底在读什么。attention 的计算只是表面，真正决定 decode TPOT 的，往往是历史 `K/V`、权重和中间张量的读取量。所以面试官追问访存，本质上是在确认你能不能把算法题落到 GPU 和服务系统里。

---

## 模块 5：FFN / MoE

### 这一模块在干什么

在标准 dense Transformer 中，Attention 后面会接 FFN；在 MoE 模型里，这一部分会变成“router + expert”的稀疏计算路径。所以这里要先明确：**MoE 是 FFN 的一种变体，不是 attention 后面的独立大阶段。**

### 高频问题 1：MoE 在推理链路中的位置是什么？

**答：**

MoE 出现在 Transformer layer 内部，通常替代原本的 dense FFN。也就是说，模型在每一层里先做 attention，再根据 router 选择少量 expert 做后续前向，而不是跑完所有 attention 之后，再统一进入一个独立的“MoE 阶段”。

### 高频问题 2：为什么 MoE 说是“算力省了，但通信炸了”？

**答：**

因为 MoE 每个 token 只激活少量 expert，从纯计算角度看比 dense FFN 更省；但这些 expert 往往分布在不同 GPU 上，token 需要根据 router 结果在多卡之间做 AllToAll 分发。这样就把问题从“算不算得动”变成了“网络和调度扛不扛得住”。

### 高频问题 3：为什么 MoE 场景更容易出通信问题？

**答：**

因为 token 到 expert 的映射是动态的，负载不均和跨卡分发都比较严重。只要 AllToAll 频繁发生，网络就容易成为瓶颈。面试里如果被追问，可以继续答 expert placement、capacity factor、router 负载均衡和拓扑感知调度。

### 高频问题 4：FFN 为什么常常是参数量大头？

**答：**

在标准 Transformer 里，FFN 往往包含两个大矩阵，参数量通常比 attention 投影更大。所以很多模型从“参数分布”角度看，FFN 才是大头；而从“长序列计算”角度看，attention 又经常是热点。面试里不要把“参数量最大”和“最慢”混为一谈。

### 高频问题 5：如果现场让你手撕 `MoE`，标准回答应该包含哪些部分？

**答：**

最少要把 `router -> top-k -> dispatch -> expert forward -> weighted combine` 这条链路写完整，并说明 token 不是过所有 expert，而是只过被选中的少数 expert。如果面试官继续追问，下一层通常就是：负载怎么均衡、capacity factor 有什么用、为什么会引入 `AllToAll` 通信。也就是说，`MoE` 题写代码只是起点，真正要看的是你能不能把稀疏激活的系统代价讲出来。

---

## 模块 6：Decode

### 这一模块在干什么

Decode 是每生成一个 token 就重复执行一次的循环：读取最新输入 token、读历史 KV、做 attention/FFN、得到 logits，再进入 sampling。这个阶段是在线服务里最容易成为性能瓶颈的地方。

### 高频问题 1：为什么 decode 往往更难优化？

**答：**

因为 decode 每一步只生成极少量 token，单次计算量不大，但要频繁读取大体积权重和 KV Cache，算力利用率不高，却很容易把显存带宽打满。所以 decode 的核心矛盾通常不是 FLOPs 不够，而是带宽、缓存和调度效率不够。

### 高频问题 2：为什么说 decode 常常是 memory-bound？

**答：**

因为每一步 decode 都要重新读取大部分模型权重，并访问不断增长的 KV Cache，而真正新增的计算量却很小。用更直白的话说，就是“搬数据的成本大于算数据的成本”，所以优化 decode 时，量化、KV 压缩、访存模式和 batching 往往比单纯提高峰值算力更重要。

### 高频问题 3：P/D 分离为什么会在 decode 阶段特别重要？

**答：**

因为 decode 对尾延迟非常敏感，而长 prompt 的 prefill 很容易打断它。P/D 分离本质上是把 compute-bound 的 prefill 和 memory-bound 的 decode 拆到不同资源池里，让 decode 的 TPOT 和 P99 更稳定。它不一定白送吞吐，但常常能换来更稳的线上体验。

### 高频问题 4：decode 很慢但 GPU 利用率不高，优先怀疑什么？

**答：**

优先看四件事：

1. batch 太小，调度没把 GPU 喂饱
2. CPU 侧 tokenizer 或调度器拖住了 GPU
3. 实际瓶颈在 HBM 带宽，而不是算力
4. 多卡通信或 NCCL 等待拉长了 step

如果面试官要你继续展开，最好回答“先看 timeline，再看带宽和 SM 利用率，再看 queue depth 和 KV 利用率”。

---

## 模块 7：Speculative Decoding（可选）

### 这一模块在干什么

Speculative Decoding 不是所有服务都会开启，但它是 decode 阶段最常见的高级加速技巧之一。它的核心思想是先让小模型或草稿机制生成多个 token，再让大模型批量验证。

### 高频问题 1：什么是 speculative decoding？

**答：**

Speculative decoding 是先用一个更快的 draft 模型猜一串 token，再让目标大模型一次性验证。如果前面若干 token 被接受，就等于减少了大模型逐 token 解码的次数。它的本质是把串行 decode 变成“猜测 + 批量验证”的组合。

### 高频问题 2：它主要优化什么指标？

**答：**

它更直接改善的是 decode 阶段的 TPOT / ITL，因为目标是减少大模型逐 token 走完整路径的次数。在某些负载下，它也可能提升整体吞吐，但它不是天然提升 TTFT 的工具。

### 高频问题 3：为什么 speculative decoding 不一定总能加速？

**答：**

因为收益高度依赖 acceptance rate。如果 draft 猜得不准，大模型还是要重算，大量验证工作就白做了；如果系统已经处在强 memory-bound 或高负载状态，引入 draft 模型还可能额外抢占资源，导致整体吞吐下降。

### 高频问题 4：什么情况下 speculative decoding 更容易有效？

**答：**

通常是：

- 输出模式比较稳定，例如代码补全、模板化文本
- acceptance rate 较高
- 目标模型还有一定富余计算资源

面试里不要说“它只适合低 batch”，更准确的说法是：**它更适合有空余计算预算、且猜中率较高的场景。**

### 高频问题 5：Lookahead Decoding 是什么？

**答：**

Lookahead 不需要额外的 draft 模型，利用 Jacobi 迭代的思想在单个大模型上并行生成多个 token。维护一个 n-token 窗口，每个位置同时猜测，每步用上一步的猜测值做 parallel forward，位置猜测收敛即接受。优点是不需要 draft 模型，缺点是加速比通常低于 draft model 方案（约 1.5-2x）。

### 高频问题 6：N-gram Speculative Decoding 怎么做？

**答：**

用 n-gram 统计作为 draft 策略：在生成过程中维护已生成 token 的 n-gram 表，当前 context 末尾匹配到 n-gram 时，直接用后续 token 作为 draft，再由大模型验证。极其轻量（零额外参数），但只在重复性内容（代码、模版文本）中有效。

### 高频问题 7：Medusa 是什么？

**答：**

Medusa 在大模型的 LM head 旁边加多个额外的"Medusa head"，每个 head 预测未来第 k 个 token。

```
             ┌→ Medusa Head 1 → 预测 t+1
hidden state ├→ Medusa Head 2 → 预测 t+2
             ├→ Medusa Head 3 → 预测 t+3
             └→ Original Head → 预测 t+1（验证用）
```

用 **tree attention** 组合多种候选序列，大模型一次 forward 验证整棵候选树，接受最长正确路径。不需要独立 draft 模型，只增加少量参数（~几百 MB），但需要额外训练 Medusa head。典型加速比 2-2.5x。

### 高频问题 8：EAGLE 是什么？和 Medusa 有什么区别？

**答：**

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）在**特征级别**做 draft：用一个轻量自回归网络（如单层 Transformer）预测大模型下一步的 hidden state，然后通过 LM head 得到 token。

**与 Medusa 的核心区别：**
- Medusa 每个 head **独立**预测，不考虑已预测 token 之间的依赖
- EAGLE 在 feature space **自回归**，能捕捉 token 间依赖 → acceptance rate 更高

典型加速比 **2.5-3.5x**（高于 Medusa 的 2-2.5x）。

### 高频问题 9：投机解码方案对比

| 方法 | 需要额外模型 | 额外参数量 | 典型加速比 | 适用场景 |
|------|------------|-----------|-----------|---------|
| **Draft Model** | 是（小模型） | 完整小模型 | 2-3x | 通用 |
| **Lookahead** | 否 | 0 | 1.5-2x | 不想维护额外模型 |
| **N-gram** | 否 | 0 | 1.2-2x | 重复性内容 |
| **Medusa** | 否（加 head） | ~几百 MB | 2-2.5x | 不想维护独立 draft |
| **EAGLE** | 否（加小网络） | ~几百 MB | 2.5-3.5x | 追求高 acceptance rate |

---

## 模块 8：Sampling 与 Output

### 这一模块在干什么

模型在每一步 decode 后会输出 logits，接下来要做采样、约束解码、流式返回，直到遇到 EOS 或 stop condition。这个模块直接决定用户看到的内容质量、结构化程度和交互体验。

### 高频问题 1：Sampling 在推理链路中的位置是什么？

**答：**

Sampling 发生在每一步 decode 之后。模型先输出 logits，再经过 temperature、top-k、top-p、repetition penalty 等策略选择下一个 token，接着把这个 token 回填到下一轮 decode。也就是说，sampling 不是模型外面的装饰，而是自回归生成闭环的一部分。

### 高频问题 2：temperature、top-k、top-p 各自怎么理解？

**答：**

- **temperature**：调节分布平滑程度，越低越保守，越高越随机
- **top-k**：只在概率最高的前 k 个 token 里采样
- **top-p**：只在累计概率达到阈值 p 的 token 集合里采样

面试里更好的回答方式不是逐个背定义，而是补一句：**它们本质上都在控制输出的随机性和稳定性，只是裁剪分布的方式不同。**

### 高频问题 3：为什么 JSON mode / grammar constrained decoding 常常变慢？

**答：**

因为它要求系统在每一步都根据语法规则重新计算“哪些 token 合法”，这会增加额外的 mask 计算和约束逻辑，某些情况下还会降低缓存命中率。换句话说，它是用更强的输出可控性，换更高的每步解码开销。

### 高频问题 4：流式输出为什么重要？

**答：**

因为用户并不只在乎总时延，还在乎“模型什么时候开始说话”。流式输出能把 TTFT 之后的结果尽早展示出来，显著改善交互体验。很多时候，业务侧感知的“快”，并不来自绝对总耗时最短，而是来自首 token 快、后续输出连续。

---

## 模块 9：跨阶段优化与指标

### 为什么要单独有这个模块

量化、FlashAttention、CUDA Graph、benchmark、profiling 这些内容并不只属于某一个阶段，而是跨阶段影响整个推理链路。所以更适合单独归类。

### 高频问题 1：为什么量化能提升推理效率？

**答：**

因为 LLM 推理尤其是 decode 阶段，经常受显存容量和带宽限制。量化把权重和激活变小后，显存占用下降、HBM 读写流量下降，在有硬件支持的情况下算子吞吐也可能提升。所以量化最稳定的收益通常先体现在“省显存、提并发、降 TPOT”，而不是任何场景下都线性提速。

### 高频问题 2：量化一定更快吗？

**答：**

不一定。它是否更快取决于目标硬件有没有对应低精度支持、kernel 是否成熟、以及 dequantization 开销是否把收益吃掉。所以面试里如果只说“INT4 一定比 BF16 快”，通常会被追问。

### 高频问题 3：FlashAttention 解决了什么问题？

**答：**

FlashAttention 没改 attention 的数学形式，而是通过 IO-aware 的执行方式减少显存读写和中间张量搬运。它主要解决的是 attention 计算中的 HBM 瓶颈，尤其适合长序列和高注意力开销场景。

### 高频问题 4：做推理 benchmark 最先要定义什么？

**答：**

首先要定义目标：到底是比最大吞吐、最低 TTFT、最低 ITL，还是某延迟约束下的 throughput；然后要固定 workload：prompt 长度、输出长度、并发数、采样参数、batch 策略。如果这两件事不先定好，benchmark 很容易变成“换了问题还在比答案”。

### 高频问题 5：Profiling 时怎么区分 compute、memory、communication 瓶颈？

**答：**

可以按这个顺序判断：

1. 看热点算子和 timeline
2. 看 SM 利用率和 HBM 带宽利用率
3. 看 NCCL / 等待对端数据的时间占比

如果 SM 很忙但带宽没打满，通常更偏 compute-bound；如果带宽接近打满而算力利用一般，更偏 memory-bound；如果大量时间卡在 collectives 上，就是 communication-bound。

### 高频问题 6：如果面试官要求你手算某种结构的参数量和单步 decode 访存量，应该怎么答？

**答：**

先分三层：`总参数量`、`单 token 激活参数量`、`单步 decode 读取量`。对 dense 模型，decode 时近似要读一遍主干权重，再读当前上下文对应的历史 `KV`；对 `MoE`，不是所有 expert 都会被激活，所以单 token 激活参数量通常远小于总参数量；对 `MLA / GQA`，关键不是参数少多少，而是 `KV` 侧缓存和带宽少多少。只要你先把这三件事分开，后面的公式就不容易乱。

### 高频问题 7：CUDA Graph 在推理中的作用是什么？

**答：**

CUDA Graph 把一段相对固定的 GPU 执行图（一系列 kernel launch 和它们之间的依赖）capture 下来，之后每次 replay 就不需要 CPU 反复发射和调度。在 decode 阶段，每步的计算图几乎一样（shape 固定），所以 CUDA Graph 能显著减少 CPU 侧 kernel launch overhead 和调度抖动。它最适合 **decode 阶段**，因为 decode 每步计算量小、kernel 多但 shape 不变；对 prefill 来说，由于 prompt 长度动态变化，CUDA Graph 的收益有限。

### 高频问题 8：KV Cache 量化是什么？为什么近年越来越受关注？

**答：**

KV Cache 量化是把缓存的 K/V 张量从 FP16/BF16 压缩到 FP8 或 INT8，以减少显存占用和访存带宽。之所以越来越重要，是因为长上下文场景下 KV Cache 已经成为显存的绝对大头——一个 7B 模型 128K 上下文的 KV Cache 可能占到几十 GB。和权重量化不同，KV 量化的挑战在于每层 K/V 的数值分布不一样，需要 per-channel 或 per-token 的 scale 策略。面试中可以补一句：权重量化省的是模型参数显存，KV 量化省的是上下文显存，两者可以叠加。

### 高频问题 9：长上下文推理（128K+）面临的主要挑战是什么？

**答：**

主要有三个方面：

1. **KV Cache 显存爆炸**：上下文越长，缓存的 K/V 越多，显存占用线性增长，很快超出单卡容量
2. **Attention 计算量二次增长**：标准 attention 的计算量是 O(n²d)，128K 上下文的 prefill 计算量远超短上下文
3. **位置编码外推**：RoPE 等位置编码在训练长度之外的外推能力有限，需要 NTK-aware scaling、YaRN 等技术

解决方案包括：FlashAttention（减少 HBM 访问）、Ring Attention / Sequence Parallelism（跨卡切分序列维度）、KV Cache 量化/压缩（减少存储）。面试里点出"显存、计算、编码"三个维度就够了。

### 高频问题 10：推理中的 Tensor Parallelism 和 Pipeline Parallelism 怎么选？

**答：**

当模型太大一张卡放不下时，推理也需要并行：

- **TP（张量并行）**：把同一层的权重切到多卡上，每步需要 AllReduce 通信，但延迟增加较少。适合放在 **机内 NVLink 互连** 的卡之间，因为通信量大但延迟要求高。
- **PP（流水并行）**：把不同层放到不同卡上，按流水线执行。通信量更小（只传激活），但会引入 pipeline bubble。

推理中通常优先用 TP，因为推理 batch 小、延迟敏感，PP 的 bubble 开销占比更大。经验法则：机内用 TP，跨机用 PP，能用 DP（模型副本）尽量先用 DP。

### 高频问题 11：FlashAttention 的核心原理是什么？为什么能同时省显存和提速？

**答：**

FlashAttention 的核心是 **tiling + online softmax**：

1. 把 Q/K/V 分成小块（tile），每次只加载一块到 SRAM 中计算
2. 用 online softmax 算法（基于 Milakov-Gimelshein 技巧）在分块过程中逐步更新 softmax 的分母，避免需要先算完全部 QK^T 再做 softmax
3. 中间的 N×N attention 矩阵不需要写回 HBM，只在 SRAM 里暂存

省显存是因为不需要 materialze O(N²) 的 attention 矩阵；提速是因为大幅减少了 HBM 读写次数。标准 attention 的 HBM 访问量是 O(N²d)，FlashAttention 降到 O(N²d²/M)（M 是 SRAM 大小）。

### 高频问题 12：RoPE 为什么适合推理中的 KV Cache？

**答：**

RoPE（Rotary Position Embedding）的关键性质是：位置信息被编码到 Q 和 K 中，而不是加在输入上。这意味着 KV Cache 中存的 K 已经包含了位置信息，decode 时新 token 的 Q 也自然带有自己的位置信息，两者做点积时自动产生相对位置的效果。这比绝对位置编码更适合 KV Cache 复用和 prefix caching，因为相同前缀的 KV 不会因为绝对位置不同而需要重新计算。

### 高频问题 13：`DeepSeek-V3` 这类题如果让你现场写伪代码和手算，最稳的回答框架是什么？

**答：**

最稳的方式不是背某个固定数字，而是按模块拆：

`Embedding -> [Attention / MLA -> MoE / FFN -> Residual / Norm] x L -> LM Head`

然后分别估：

- 主干 dense 部分的参数和权重读取
- `MLA` 对 KV 缓存与带宽的压缩
- `MoE` 的总 expert 参数量、单 token 激活 expert 参数量
- 多卡场景下 token 分发带来的通信代价

这样即使你不背具体公开配置，也能把题讲得结构清楚、逻辑正确。

### 高频问题 14：LLaMA 1 → 2 → 3 的架构演变

**答：**

| 维度 | LLaMA 1 (2023.02) | LLaMA 2 (2023.07) | LLaMA 3/3.1 (2024) |
|------|-------------------|-------------------|---------------------|
| 模型大小 | 7B, 13B, 33B, 65B | 7B, 13B, **70B** | **8B**, **70B**, **405B** |
| 上下文长度 | 2K | **4K** | **8K**（3.1: **128K**） |
| Attention | MHA | **GQA**（70B: 8 KV head） | GQA（全系列） |
| Tokenizer | SentencePiece, 32K vocab | SentencePiece, 32K vocab | **tiktoken, 128K vocab** |
| 训练数据 | 1.4T tokens | **2T tokens** | **15T+ tokens** |
| 其他 | RoPE + RMSNorm + SwiGLU | 增加 RLHF（Chat） | RoPE scaling 做 128K |

**关键演变总结：**
- LLaMA 1→2：GQA 减少 KV Cache，更多训练数据，RLHF 对齐
- LLaMA 2→3：128K 词表提升多语言和 token 效率，数据量暴增到 15T+，架构本身非常稳定
- 整体趋势是**架构不变、scaling up + 更多数据**

### 高频问题 15：Qwen3 的 MoE 怎么做的？

**答：**

Qwen3-MoE 采用 **Fine-grained Expert（细粒度专家）** 设计：

```
传统 MoE (Mixtral): 8 个大 expert，每 token 激活 top-2
Qwen3-MoE:         128 个小 expert，每 token 激活 top-8 + 1 个 shared expert（必经）
```

| 特性 | 内容 |
|------|------|
| Expert 数量 | 128 个小 expert（vs 传统 8/16 个大 expert） |
| 激活 expert 数 | 每 token 激活 8 个 |
| Shared Expert | 有（所有 token 必经的共享专家，保证基础能力不退化） |
| Router | Top-K routing with load balancing loss |

**为什么 fine-grained？** 更多 expert → routing 粒度更细 → 每个 token 获得更专业化处理；激活更多小 expert → 组合灵活性更高。

### 高频问题 16：DeepSeek 的 MTP（Multi-Token Prediction）是什么？

**答：**

传统 LLM 每步只预测下一个 token。MTP 在训练时同时预测未来多个 token。

```
传统:    hidden_state → LM Head → predict token t+1
DeepSeek MTP:
         hidden_state → LM Head 0  → predict t+1
                      → MTP Head 1 → predict t+2
                      → MTP Head 2 → predict t+3
```

每个 MTP module 包含：Embedding 投影 + 一层 Transformer Block + 共享 LM Head，接收上一步预测 embedding 和 main model 的 hidden state。

**MTP 的三重作用：**
1. **训练信号更丰富**：每个位置提供多个梯度信号，提高数据效率
2. **推理加速**：训练好的 MTP head 天然可做 speculative decoding 的 draft
3. **更好的表征学习**：迫使模型学习更远的依赖关系

**与 Medusa 的区别：** MTP 在**预训练阶段**就参与训练，head 质量更高、acceptance rate 更好；Medusa 是在已训练好的模型上**后期加装** head。

---

## 推荐使用方式

如果你要准备推理面试，可以按这个顺序读：

1. 先读 `模块 3 + 模块 4 + 模块 6`，建立 prefill / attention / decode 主链路
2. 再读 `模块 2 + 模块 7`，理解调度和高级加速
3. 如果面试偏系统，补 `模块 5 + 模块 9`
4. 如果面试偏产品或应用，重点看 `模块 8`

---

## 对照到旧文档

- 如果想背诵 30 秒答案，看 [QuickInterviewAnswers.md](./QuickInterviewAnswers.md)
- 如果想看推理优化速记，看 [InferenceOptimization.md](./InferenceOptimization.md)
- 如果想看更资深的追问，看 [SeniorInterviewQuestions.md](./SeniorInterviewQuestions.md)
