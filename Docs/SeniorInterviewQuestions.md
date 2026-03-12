# 资深面试官补充追问（高级篇）

> 来源：牛客网大模型推理加速 & AI Infra 实习岗位面试题汇总
> 
> 这一章补的是"更像资深面试官会继续往下深挖"的问题

---

## 目录

- [Tokenizer 与输入链路](#tokenizer-与输入链路)
- [模型加载与冷启动](#模型加载与冷启动)
- [LoRA / 多租户服务](#lora--多租户服务)
- [多模态 / VLM 推理](#多模态--vlm-推理)
- [Structured Output / Tool Use](#structured-output--tool-use)
- [线上故障与可观测性](#线上故障与可观测性)
- [正确性验证与回归测试](#正确性验证与回归测试)
- [SLO、成本与容量规划](#slo成本与容量规划)
- [硬件认知与系统边界](#硬件认知与系统边界)
- [开放式系统设计](#开放式系统设计)
- [项目与方法论追问](#项目与方法论追问)
- [反问式陷阱题](#反问式陷阱题)

---

## Tokenizer 与输入链路

### 1. 为什么推理岗位也会问 tokenizer，而不只是问模型和 kernel？

**答：**

Tokenizer 虽然是预处理阶段，但在高 QPS 场景下：
- **CPU 瓶颈**：BPE 编码计算量不容小觑
- **延迟占比**：短请求中 tokenizer 可能占 TTFT 的 30%+
- **并发处理**：需要多线程/进程并行化

### 2. BPE、WordPiece、SentencePiece 的核心区别

**答：**

| 算法 | 特点 | 代表 |
|-----|------|------|
| **BPE** | 合并频率最高的字符对 | GPT、LLaMA |
| **WordPiece** | 基于语言模型概率合并 | BERT |
| **SentencePiece** | 不依赖预分词，直接学 subword | T5、XLNet |

### 3. 为什么有些系统会把 tokenization 放在 CPU 侧并做并行化？

**答：**

- Tokenizer 逻辑复杂，不适合 GPU 执行
- 可以异步预处理，与 GPU 计算 overlap
- 多核 CPU 并行处理多个请求的 tokenization

---

## 模型加载与冷启动

### 4. 模型启动时，真正耗时的部分是哪些？

**答：**

| 阶段 | 耗时因素 |
|-----|---------|
| **读盘** | 模型文件大小（数十到数百 GB） |
| **反序列化** | 格式解析（safetensors 比 pickle 快） |
| **搬到 GPU** | PCIe/NVLink 带宽 |
| **初始化** | CUDA context、graph capture |

### 5. 预热（warmup）除了把 kernel 热起来，还有哪些作用？

**答：**

- **CUDA context 初始化**：避免首次请求卡顿
- **内存分配器预热**：让 allocator 建立缓存池
- **Graph capture**：捕获优化后的执行图
- **精度校准**：量化模型的 warmup 校准

### 6. 如果同一模型要在多实例反复拉起，你会怎么优化加载路径？

**答：**

- **共享权重**：多进程共享同一份内存映射
- **模型缓存池**：预加载常用模型，按需分配
- **懒加载**：只加载需要的层
- **Checkpoint 复用**：从内存直接恢复而不是磁盘

---

## LoRA / 多租户服务

### 7. 为什么现在很多推理岗位会追问 LoRA serving？

**答：**

- **个性化需求**：不同用户需要不同适配器
- **显存优化**：LoRA 参数量小（<1%），可动态加载
- **多租户场景**：一个 base model + 多个 adapter

### 8. 多 LoRA 并发挂载的主要系统挑战

**答：**

| 挑战 | 说明 |
|-----|------|
| **显存管理** | 哪些 adapter 常驻，哪些按需加载 |
| **Batch 内混合** | 同一 batch 不同请求用不同 adapter |
| **切换开销** | Adapter 切换的延迟 |
| **KV Cache 共享** | Base model 相同，KV 可复用 |

### 9. LoRA adapter 应该在请求级切换、batch 内混合，还是做实例级隔离？

**答：**

| 策略 | 适用场景 | 特点 |
|-----|---------|------|
| **请求级切换** | 低并发、adapter 多 | 灵活但切换开销大 |
| **Batch 内混合** | 高并发、adapter 少 | 效率高但实现复杂 |
| **实例级隔离** | 大流量、固定 adapter | 最简单但资源浪费 |

### 10. 多租户下为什么 admission control 是必需而不是可选？

**答：**

- **资源隔离**：防止单个租户占满资源
- **QoS 保障**：确保高优先级租户体验
- **过载保护**：系统接近容量时拒绝低优先级请求
- **公平性**：防止 noisyn neighbor

---

## 多模态 / VLM 推理

### 11. VLM 推理相比纯文本 LLM，链路上多了哪些阶段？

**答：**

```
图像输入 → Vision Encoder (ViT) → Patch Embeddings 
                                   ↓
文本输入 → Tokenizer → Text Embeddings
                                   ↓
                          投影/对齐层
                                   ↓
                         统一 LLM 解码
```

### 12. Vision encoder 的开销通常更像 prefill 还是 decode？

**答：**

更像 **prefill** —— 一次性处理整张图片，计算密集。

### 13. 多图输入为什么容易把 TTFT 拉高？

**答：**

- 每张图都要过 vision encoder
- 图片 token 数远多于文本
- Prefill 阶段计算量爆炸

### 14. 多模态场景里，prefix cache 的适用性会发生什么变化？

**答：**

- **适用**：系统提示词、固定图片模板
- **不适用**：用户上传的个性化图片
- **新挑战**：图片 embedding 的缓存和匹配

---

## Structured Output / Tool Use

### 15. 为什么 JSON mode / grammar constrained decoding 常常变慢？

**答：**

- **Mask 计算**：每步都要计算合法的 next token
- **分支约束**：某些路径被强制剪枝
- **缓存失效**：约束条件变化时 cache 命中率下降

### 16. function calling / tool calling 对普通聊天推理链路增加了哪些复杂度？

**答：**

```
1. 用户输入 → 模型判断是否需要调用工具
2. 生成 tool call 参数（JSON）
3. 系统执行工具 → 获取结果
4. 结果返回给模型 → 继续生成回复

复杂度：
- 多轮交互的上下文管理
- 工具 schema 的 prompt 注入
- 并行 tool call 的处理
- 错误处理和重试机制
```

### 17. agent 场景为什么更依赖 prefix cache 和 prompt 复用？

**答：**

- Agent 有固定的 system prompt 和工具描述
- 多轮对话中历史上下文高度相似
- ReAct / CoT 模板固定，可缓存

---

## 线上故障与可观测性

### 18. 某次版本上线后 TTFT 突然飙高，排查顺序

**答：**

```
1. 查看变更列表 → 确定相关改动
2. 检查 tokenizer → 是否有新逻辑拖慢
3. 查看 scheduler 日志 → 队列是否积压
4. 检查 prefix cache 命中率 → 是否失效
5. 对比模型版本 → 是否有结构变化
6. 查看网络/存储 → 是否有外部依赖变慢
```

### 19. 线上只在长 prompt 场景超时，短 prompt 正常，先看哪里？

**答：**

- **Prefill 阶段瓶颈**：可能是 attention 计算或 vision encoder
- **显存不足**：长序列导致 OOM 或 swapping
- **Chunked prefill 配置**：chunk size 是否合理

### 20. decode 很慢但 GPU 利用率不高，优先怀疑什么？

**答：**

| 可能原因 | 排查方法 |
|---------|---------|
| **CPU 瓶颈** | tokenizer 或数据预处理 |
| **调度问题** | batch 太小或请求饥饿 |
| **通信等待** | NCCL 同步问题 |
| **内存带宽** | 实际是 memory-bound 而非 compute-bound |

### 21. 你认为 LLM serving 面板里最关键的 10 个指标

**答：**

```
1. TTFT (p50/p95/p99)
2. ITL (p50/p95/p99)
3. Throughput (tokens/s)
4. QPS
5. GPU Utilization
6. Memory Utilization
7. KV Cache Utilization
8. Prefix Cache Hit Rate
9. Queue Depth / Wait Time
10. Error Rate
```

---

## 正确性验证与回归测试

### 22. 推理优化为什么不能只跑 benchmark，还必须做 correctness check？

**答：**

- **数值精度**：量化、FlashAttention 等会改变数值
- **随机性**：Sampling、Dropout 导致非确定性
- **长期影响**：微小误差在长序列中会累积

### 23. 量化、FlashAttention、speculative decoding 各自更适合用什么 correctness 标准？

**答：**

| 优化 | 验证标准 |
|-----|---------|
| **量化** | 下游任务准确率、Perplexity |
| **FlashAttention** | 数值误差 < 1e-5（相对 FP32） |
| **Speculative Decoding** | 输出分布一致性、BLEU/ROUGE |

### 24. 同一个请求在不同 batch 条件下输出不一致，可能原因

**答：**

- **Floating point 累加顺序**：不同并行策略导致
- **随机性**：Sampling 种子、Dropout
- **数值误差**：量化、低精度计算
- **边界条件**：Padding 处理差异

---

## SLO、成本与容量规划

### 25. 如何给一个在线 LLM 服务定义 SLO？

**答：**

```
核心 SLO：
- TTFT p99 < 500ms
- ITL p99 < 100ms  
- Availability > 99.9%
- Error Rate < 0.1%

扩展 SLO：
- Cost per 1M tokens
- Max concurrent requests
- Max context length supported
```

### 26. 为什么 cost/token 已经变成推理岗位高频追问？

**答：**

- **商业化**：API 定价需要成本核算
- **优化目标**：从"更快"到"更便宜"
- **容量规划**：预测运营支出

### 27. 吞吐最大化有时反而会让单位成本变差？

**答：**

- 极致吞吐可能需要过度 batch，导致延迟恶化
- 用户体验差 → 用户流失 → 实际收入下降
- 需要平衡吞吐、延迟、成本

---

## 硬件认知与系统边界

### 28. 为什么推理岗面试会追问 HBM 带宽、L2、shared memory？

**答：**

因为推理瓶颈经常在**访存**而非算力：
- Decode 阶段是 memory-bound
- KV Cache 管理直接影响 HBM 访问
- Kernel 优化需要理解内存层级

### 29. HBM 带宽不足和 Tensor Core 算力不足，在 profiler 上分别长什么样？

**答：**

| 瓶颈 | Profiler 表现 |
|-----|--------------|
| **HBM 带宽** | Memory throughput 高，SM 利用率低 |
| **Tensor Core** | SM 利用率高，但 memory 不饱和 |

### 30. Hopper 相比 Ampere，对推理优化最值得关心的变化

**答：**

- **FP8 原生支持**：推理量化收益更大
- **Transformer Engine**：自动管理精度
- **新的内存层级**：更大 L2，更好 locality

---

## 开放式系统设计

### 31. 设计一个支持 1000 并发、长短请求混合的大模型在线服务

**答（框架）：**

```
架构分层：

接入层：
- API Gateway（限流、鉴权、路由）
- Load Balancer

调度层：
- Global Scheduler（跨实例调度）
- Local Scheduler（实例内 continuous batching）

推理层：
- Prefill Pool（计算密集型）
- Decode Pool（访存密集型）
- P/D 分离可选

存储层：
- KV Cache Manager（PagedAttention）
- Prefix Cache Index

监控层：
- Metrics（Prometheus）
- Tracing（Jaeger）
- Logging
```

### 32. 如果目标是"首 token 尽量快"，架构如何调整？

**答：**

- **PD 分离**：Prefill 独立扩缩容
- **Prefix Cache**：减少重复计算
- **Tokenizer 并行**：异步预处理
- **Admission Control**：优先处理短请求

### 33. 如果目标是"单位成本最低"，架构怎么改？

**答：**

- **极致 batching**：最大化 GPU 利用率
- **量化**：INT8/FP8 降低显存和计算
- **动态扩缩容**：按需启停实例
- **冷热分离**：热门模型常驻，冷模型按需加载

---

## 项目与方法论追问

### 34. 你怎么判断一个优化值得做成论文，还是更适合做工程特性？

**答：**

| 维度 | 论文 | 工程 |
|-----|------|------|
| **创新性** | 新方法、新理论 | 已有方法的工程实现 |
| **通用性** | 跨模型、跨场景有效 | 特定场景优化 |
| **可复现性** | 需要严格实验设计 | 可快速验证上线 |

### 35. 你怎么在"局部算子更快"和"端到端更快"之间做取舍？

**答：**

- **先端到端**：确定真正的瓶颈点
- **局部优化**：针对瓶颈点深入优化
- **持续验证**：每次优化后回归端到端测试

### 36. 你如何向面试官证明自己不仅会调参数，而且真的理解系统瓶颈？

**答：**

- **Profiler 证据**：展示 timeline 分析
- **瓶颈定位**：明确说出是 compute/memory/communication 哪类
- **优化逻辑**：解释为什么这个优化能解决这个瓶颈
- **Trade-off 分析**：说明优化的代价和适用场景

---

## 反问式陷阱题

### 37. 你说 decode 是 memory-bound，那为什么我 profiler 里看到 Tensor Core 很忙？

**答：**

可能的原因：
- **Batch size 足够大**：矩阵够大，能利用 Tensor Core
- **GQA/MQA**：减少了 KV 读取，计算占比上升
- **Attention 优化**：FlashAttention 减少了访存，计算占比上升
- **混合负载**：部分请求 compute-bound，部分 memory-bound

### 38. 你说量化更快，那为什么我的 INT4 比 BF16 还慢？

**答：**

可能原因：
- **Dequantization 开销**：INT4 需要解包回 FP16
- **Kernel 未优化**：没有专门的 INT4 Tensor Core kernel
- **Batch 太小**：小矩阵下开销占比大
- **硬件不支持**：GPU 没有原生 INT4 支持

### 39. 你说 P/D 分离好，那为什么官方也说不一定提 throughput？

**答：**

P/D 分离的核心收益是**延迟可控性**和**资源隔离**，而非吞吐：
- **通信开销**：KV 传输占用带宽
- **资源碎片**：Prefill/Decode 资源不能完全互补利用
- **调度复杂度**：增加系统开销

### 40. 你说自己是做 LLM inference 的，那在你心里最核心的资源到底是什么？

**答（参考）：**

> 我认为最核心的资源是**显存带宽**。因为 LLM 推理（尤其是 decode）本质是 memory-bound，算力往往过剩，而 KV Cache 的读写、权重的加载都受限于 HBM 带宽。所有优化——无论是量化、GQA、PagedAttention 还是 PD 分离——最终都是在解决"如何更高效地利用有限的显存带宽"。

---

## 面试建议

### 资深岗位的核心考察点

1. **系统思维**：不只看局部优化，能看端到端链路
2. **工程权衡**：理解各种 trade-off，会取舍
3. **线上经验**：有真实的故障排查和优化经验
4. **业务理解**：技术优化能回归业务价值

### 回答策略

- **STAR 法则**：Situation → Task → Action → Result
- **数据说话**：用具体数字支撑观点
- **承认局限**：不知道就坦诚，但尝试分析
- **展示思考过程**：比标准答案更重要的是思考逻辑
