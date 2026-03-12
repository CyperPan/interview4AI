# 📚 大模型 AI Infra 面试基础知识总结

<p align="center">
  <a href="https://github.com/huihut/interview"><img src="https://img.shields.io/badge/Reference-huihut%2Finterview-blue.svg" alt="Reference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="LICENSE"></a>
</p>

---

## 💡 关于

📚 本仓库是面向大模型 AI Infra 方向校招求职者、初学者的基础知识总结，包括分布式训练、推理优化、通信优化、显存优化、多模态训练等知识及面试经验。

💡 建议阅读方式：使用支持 Markdown 目录的编辑器（如 VSCode、Typora）打开，或使用 GitHub 的 TOC 导航。

🙏 仓库内容如有错误或改进欢迎 issue 或 pr。由于本人水平有限，仓库中的知识点有来自本人原创、读书笔记、技术博客等，非原创均已标明出处。

本仓库遵循 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)（署名 - 非商业性使用 - 相同方式共享）协议。

---

## 📋 目录

- [🔥 分布式训练基础](#-分布式训练基础)
  - [数据并行 vs 模型并行 vs 流水线并行](#数据并行-vs-模型并行-vs-流水线并行)
  - [DDP 原理与梯度同步](#ddp-原理与梯度同步)
  - [3D 并行策略设计](#3d-并行策略设计)
- [⚡ 训练效率优化](#-训练效率优化)
  - [影响吞吐的主要因素](#影响吞吐的主要因素)
  - [GPU 利用率排查](#gpu-利用率排查)
  - [梯度累积](#梯度累积)
- [📡 通信优化](#-通信优化)
  - [通信原语对比](#通信原语对比)
  - [通信与计算重叠](#通信与计算重叠)
- [💾 显存优化与训练框架](#-显存优化与训练框架)
  - [ZeRO 系列详解](#zero-系列详解)
  - [激活检查点](#激活检查点)
  - [混合精度训练](#混合精度训练)
  - [DeepSpeed vs Megatron vs FSDP](#deepspeed-vs-megatron-vs-fsdp)
- [🚀 推理优化](#-推理优化)
  - [Continuous Batching](#continuous-batching)
  - [PagedAttention](#pagedattention)
  - [PD 分离](#pd-分离)
  - [投机解码](#投机解码)
  - [量化策略](#量化策略)
- [🎨 多模态训练](#-多模态训练)
  - [多模态训练挑战](#多模态训练挑战)
  - [视频模型优化](#视频模型优化)
- [🎯 SFT / RL 训练](#-sft--rl-训练)
  - [三阶段训练差异](#三阶段训练差异)
  - [RLHF 复杂度分析](#rlhf-复杂度分析)
- [🔧 排障与实战](#-排障与实战)
  - [系统排查方法论](#系统排查方法论)
  - [常见故障定位](#常见故障定位)
- [📝 手撕代码](#-手撕代码)
  - [注意力机制实现](#注意力机制实现)
  - [CUDA 基础](#cuda-基础)
  - [C++ 高并发](#c-高并发)
- [💯 面试实战话术](#-面试实战话术)
- [🔥 牛客网面经汇总](#-牛客网面经汇总)
  - [NVIDIA/HPC 岗](#nvidiahpc-岗)
  - [30 秒口语答案](#30-秒口语答案)
  - [资深面试官追问](#资深面试官追问)
  - [各公司面经](#各公司面经)
- [📚 推荐资源](#-推荐资源)

---

## 🔥 分布式训练基础

### 数据并行 vs 模型并行 vs 流水线并行

| 并行方式 | 解决问题 | 切分维度 | 通信特点 |
|---------|---------|---------|---------|
| **数据并行 (DP)** | 数据量太大，单卡算得慢 | 数据切分，模型复制 | All-Reduce 梯度同步 |
| **模型并行/张量并行 (TP)** | 单层参数过大，单卡显存装不下 | 层内切分（矩阵切片） | 高频 All-Reduce，需 NVLink |
| **流水线并行 (PP)** | 整体模型太深，单卡装不下 | 层间切分 | P2P 通信，通信量小，适合跨机 |

**如何选择？**

采用 **3D 混合并行** 策略：
- **单机内 (Intra-node)**：机器内带宽极高（NVLink），跑 TP（通常 TP=4 或 8），切分大矩阵
- **跨机器 (Inter-node)**：机器间带宽较慢（RDMA），跑 PP，按层分配到不同机器
- **全局**：在 TP 和 PP 的组别之外套上 DP/ZeRO，增加副本数以消化海量数据

### DDP 原理与梯度同步

**DDP (DistributedDataParallel) 核心机制：**

1. **多进程架构**：每个 GPU 一个进程
2. **初始化时 Broadcast**：保证各卡权重一致
3. **Bucket (分桶) 机制**：反向传播时，一旦某几个层的梯度填满一个 Bucket，就立刻触发 All-Reduce，实现网络通信和后续反向计算的 **Overlap（重叠）**
4. **Ring-AllReduce 算法**：高效的梯度同步

**为什么纯 DP 在大模型训练中会遇到瓶颈？**

- **显存墙 (OOM)**：每张卡需完整保存模型权重 + 梯度 + 优化器状态（Adam 占大头）
- 70B 模型单是这些状态就需要上 TB 显存，远超单卡 H100（80GB）极限
- 参数量太大导致 All-Reduce 通信延迟极高

### 3D 并行策略设计

**流水线并行中的 Bubble 问题：**

- **Bubble（气泡）**：后面层的 GPU 在等待前面层 GPU 计算结果时的闲置空转时间
- **解决方案**：
  - 引入 **Micro-batching（微批次）**
  - 采用 **1F1B（一前一后）** 调度策略
  - 使用 **Interleaved 1F1B（交错式）** 进一步缩小气泡比例

---

## ⚡ 训练效率优化

### 影响吞吐的主要因素

1. **计算限制**：算子未优化（如没用 FlashAttention）
2. **显存带宽限制**：Memory-bound，频繁读写 HBM
3. **通信瓶颈**：NCCL All-Reduce 慢
4. **数据 I/O 瓶颈**：Dataloader 跟不上 GPU 速度

### GPU 利用率排查

**排查优先级：**

1. 看 `top`：CPU 和 IO 是否打满（Dataloader 瓶颈）
2. 看 `nvidia-smi`：显存和算力利用率
3. 用 **Nsight Systems**：看时间都花在 compute、nccl_all_reduce 还是 cudaMemcpy 上

**关键指标：**
- SM 利用率（算力是否吃满）
- Kernel 耗时 vs NCCL 耗时比例
- DataLoader Wait Time

### 梯度累积

**原理：** 连续进行 N 次前向+反向传播，累加梯度但不更新权重，最后才执行一次 `optimizer.step()`

| 方面 | 说明 |
|-----|------|
| **好处** | 显存不变的情况下等效扩大了全局 Batch Size |
| **代价** | 缓存微批次的激活值会占用少许额外显存，且不能减少总的计算量 |

**吞吐与收敛的矛盾：**
- 极端增大 Batch Size 提升吞吐，但会导致模型陷入局部最优，泛化变差
- 使用更低精度（如 FP8）能极大提速，但会引入数值下溢，导致 Loss 爆炸/NAN

---

## 📡 通信优化

### 通信原语对比

| 原语 | 功能 | 典型应用场景 |
|-----|------|------------|
| **All-Reduce** | 所有卡求和，每张卡得到完整结果 | DDP 梯度同步 |
| **Reduce-Scatter** | 所有卡求和，结果切片分发 | ZeRO 替代方案 |
| **All-Gather** | 每张卡广播切片，拼接完整数据 | ZeRO 参数收集 |
| **All-to-All** | 每个节点发送数据到所有节点 | MoE Token 路由 |
| **Send/Recv** | 点对点通信 | PP 激活值传递 |

### 通信与计算重叠

**为什么重要？**

GPU 的计算单元（SM）和通信单元（如 NVLink/网卡 DMA）是物理独立的。重叠就是让 GPU 算第 N-1 层梯度的同时，网卡发送第 N 层的梯度，能把通信耗时从总时间中"隐藏"掉。

**多机训练 vs 单机多卡：**

| 维度 | 单机多卡 | 多机训练 |
|-----|---------|---------|
| 带宽 | 400GB/s (NVLink) | 50GB/s (RDMA) |
| 延迟 | 低 | 高 |
| 挑战 | 较小 | 硬件/网络故障导致的掉线重启 |

---

## 💾 显存优化与训练框架

### ZeRO 系列详解

**FSDP/ZeRO 核心思想：用通信换显存（数据切片）**

传统的 DP 每张卡存全量模型状态；FSDP/ZeRO 将参数、梯度、优化器状态等价分片到 N 张卡上，单卡只存 1/N。需要计算时，再通过 All-Gather 临时拿回完整参数，算完立刻丢弃。

| Stage | 分片内容 | 显存节省 |
|-------|---------|---------|
| **Stage 1** | 优化器状态 (Optimizer States) | 4x |
| **Stage 2** | 优化器状态 + 梯度 (Gradients) | 8x |
| **Stage 3** | 优化器状态 + 梯度 + 参数 (Parameters) | 与数据并行度线性相关 |

### 激活检查点

**原理：** 反向传播需要用到前向传播的激活值。不保存所有中间激活值，只保存某几个层的"Checkpoint"。反向计算需要时，临时从最近的 Checkpoint 重新前向计算一次。

** trade-off：** 以 ~33% 的额外计算时间，换取指数级的显存节省

### 混合精度训练

**为什么能提升效率？**

- FP16/BF16 把权重和激活砍半，省 50% 读写带宽
- 激活硬件 Tensor Cores，算力翻倍

**潜在问题：**

| 精度 | 问题 | 解决方案 |
|-----|------|---------|
| FP16 | 容易 Underflow（下溢出）导致梯度为 0 | 动态 Loss Scaling |
| BF16 | 精度较低，极少数任务难收敛 | 直接用 BF16（范围更大） |

**训练显存爆了，解决方向：**

1. 调小 Batch Size + 开启梯度累积
2. 开启 Activation Checkpointing（立竿见影）
3. 提升 ZeRO 等级（Stage 2 升到 3）
4. ZeRO-Offload（把优化器状态赶到 CPU 内存）
5. 采用 FlashAttention 省略注意力矩阵显存

### DeepSpeed vs Megatron vs FSDP

| 框架 | 适合场景 | 特点 |
|-----|---------|------|
| **Megatron** | 超大规模（万卡集群）、追求极限 MFU | 3D 并行原生支持，代码侵入性强 |
| **DeepSpeed** | 资源有限、需要 CPU Offload | ZeRO 护城河深，插件化好 |
| **FSDP** | 不需要复杂 TP，靠数据并行 | PyTorch 原生，生态好，代码干净 |

---

## 🚀 推理优化

### Continuous Batching

**痛点：** 传统 Static Batching 必须等 Batch 中最长的句子生成完毕，导致提前生成完的短句子对应的 GPU 算力处于"空转（Bubble）"状态。

**原理：** 也叫 In-flight Batching，打破请求级边界，实现 **Token 级调度**。当某个请求遇到 `<EOS>` 时，调度器会立刻将它踢出，并从等待队列中塞入新请求。

**重要性：** 将 GPU 吞吐量提升数倍，是 vLLM、TensorRT-LLM 等现代推理引擎的标配。

### PagedAttention

**痛点：** LLM 解码时 KV Cache 随序列动态增大。传统框架按最大长度预分配连续显存，导致巨大内部碎片和外部碎片，超一半显存被浪费。

**解决方案：** 借鉴操作系统虚拟内存分页机制：
- 将 KV Cache 划分为固定大小的 Block（如每 Block 存 16 个 Token）
- 逻辑上连续，物理上离散存储
- 通过 BlockTable 记录逻辑到物理的映射

**收益：** 彻底消除外部碎片，内部碎片 < 4%，让 GPU 能扛住 3-4 倍并发请求。

### PD 分离

**根本原因（算力特征冲突）：**

| 阶段 | 计算特征 | 瓶颈 |
|-----|---------|------|
| **Prefill** | 一次性处理长 Prompt，GEMM | Compute-bound（算力密集） |
| **Decode** | 每次吐一个 Token，GEMV | Memory-bound（访存密集） |

**混部问题：** 新请求做 Prefill 时会抢占算力，导致正在 Decode 的请求被"卡住"，引起严重尾延迟（Tail Latency 抖动）。

**PD 分离方案：** 将 Prefill 和 Decode 拆分到不同 GPU/机器（如 Kimi 的 Mooncake 架构）。Prefill 节点算完 Prompt，通过 RDMA 把 KV Cache 传给 Decode 节点。

### 投机解码

**原理：** "小模型猜，大模型判"。用小的 Draft 模型快速猜出几个 Token，再用大模型（Target）一次性验证。

**适用场景：**
- 输出重复度高、有强模板的场景（代码补全、固定格式 JSON）
- **必须在 Low Batch Size 下使用**

**不适用场景：**
- 高 QPS 满载状态（会抢占主模型带宽）
- 输出极短的场景（如审核业务）

### 量化策略

**为什么能提速？**

LLM 推理（尤其是 Decode）是 Memory-bound，算力空转等待权重从 HBM 搬运。量化把数据体积缩减 2-4 倍，极大降低访存延迟。

| 量化类型 | 适用场景 | 特点 |
|---------|---------|------|
| **FP8** | Hopper 架构 (H100) | 硬件原生支持，保留指数位，动态范围大，精度保护好 |
| **INT8** | 通用场景 | 平衡速度和精度 |
| **INT4 (AWQ/GPTQ)** | 极致压榨单卡并发 | Weight-Only，需要解包开销 |

**为什么有些量化掉点厉害？**

LLM 参数量超过 6.7B 后，激活值中出现大量 Outliers（异常极大值）。朴素 PTQ 会截断这些值，导致精度断崖式下跌。

**解决方案：** SmoothQuant（把激活难度转移到权重）、AWQ（保留 1% 最重要权重不量化）

---

## 🎨 多模态训练

### 多模态训练挑战

1. **Token 数量暴增**：图像/视频编码后序列过长
2. **负载不均衡**：视觉编码器（ViT）和语言模型（LLM）计算特征不同
3. **跨模态同步**：Cross-Modal 对齐涉及复杂 Contrastive Loss，增加跨卡全局特征同步开销

### 视频模型优化

**为什么资源压力更大？**

视频引入时间维度（T×H×W），导致：
- 3D Attention 序列长度呈指数级爆炸（可达 100K 级别）
- 显存 OOM 和计算二次方复杂度墙

**优化方向：**
- **Context Parallel / Ring Attention**：长序列并行
- **Token 压缩/池化**：丢弃视频背景冗余 Token
- **Chunked Prefill**：分块预填充

### Omni 模型训练难点

**异构负载不均衡（Straggler Effect）**：一段数据的音频/视频/文本比例完全不同，导致不同 GPU 计算时间差异极大。在 Pipeline 或 TP 同步时，快的卡被迫等待慢的卡。

---

## 🎯 SFT / RL 训练

### 三阶段训练差异

| 阶段 | 优化重点 | 特点 |
|-----|---------|------|
| **Pretrain** | 极限吞吐（MFU）和长时间稳定性 | 静态图，追求极致性能 |
| **SFT** | 避免无效 Padding | 数据长度极不固定，依赖 Sequence Packing |
| **RL** | 生成 + 训练循环 | Rollout 生成阶段极其耗时 |

### RLHF 复杂度分析

**为什么比 SFT 复杂？**

1. **多模型维护**：PPO 同时需要 4 个模型（Actor, Critic, Reward, Reference），内存常态爆炸
2. **推理与训练冲突**：Actor 生成（推理需要 KV-Cache）和参数更新（训练引擎）底层逻辑冲突
3. **框架拆分**：通常需要把生成和训练拆分到两个框架（如 vLLM + DeepSpeed）

**RL 训练效率瓶颈：**
- Actor 生成经验的速度（Rollout 阶段）
- CPU 与 GPU 之间的 Experience Buffer 拷贝
- Reward 模型打分的串行阻塞等待

---

## 🔧 排障与实战

### 系统排查方法论

**8 卡训练任务吞吐比预期低，如何定位？**

1. **看 top**：CPU 和 IO 是否打满（Dataloader 瓶颈）
2. **看 nvidia-smi**：显存和算力利用率
3. **开 PyTorch Profiler 或 Nsight Systems**：看时间花在 compute、nccl_all_reduce 还是 cudaMemcpy 上

**loss 正常下降但训练速度慢，优先优化什么？**

1. 确定是 Compute Bound 还是 Memory Bound
2. 确认是否开启 FlashAttention 和 TF32/BF16
3. 尝试增大 Batch Size 跑满流处理器
4. 排查是否有频繁的小数据 Host/Device 拷贝

### 常见故障定位

**增加 GPU 数量后吞吐无线性提升？**

- 阿姆达尔定律（Amdahl's Law）体现
- 节点增加导致跨机 RDMA 通信开销激增
- 全局 Batch Size 没有等比例放大，单卡 Micro-batch 太小

**Pipeline 某个 stage 特别慢？**

- 短板效应严重
- **解决**：重新做负载均衡（Stage Imbalance Profiling），给慢的阶段少分配层数

**训练中出现 occasional hang（偶尔卡住）？**

- 网络层面的丢包或死锁（如 RoCE 交换机的 PFC 风暴）
- 某一台机器的 GPU 硬件降频（过热或 ECC 错误）
- 某一个 step 的文本数据异常的长，没做好截断

---

## 📝 手撕代码

### 注意力机制实现

**Multi-Head Attention / Grouped-Query Attention (GQA)**

考察点：
- 矩阵维度变换（Transpose/Reshape）
- Mask 的应用位置
- Softmax 前除以 √dk（防止梯度消失/Softmax 溢出）
- GQA 中 KV 的广播（Broadcast）处理

### CUDA 基础

**必会题目：**

| 题目 | 考察点 |
|-----|-------|
| **并行归约 (Parallel Reduction)** | Shared Memory 树状规约、__syncthreads()、Warp-level primitives |
| **矩阵乘法 GEMM** | Tiling（分块）思想、Shared Memory 优化 |
| **RMSNorm 算子** | 计算公式、variance 内存读取优化 |
| **Softmax (数值稳定版)** | 先减 max 再 exp，防止指数溢出 |

### C++ 高并发

**必会题目：**

| 题目 | 考察点 |
|-----|-------|
| **无锁环形缓冲区 (Lock-free Ring Buffer)** | std::atomic、Memory Order (acquire/release)、SPSC 队列 |
| **线程池 (Thread Pool)** | std::thread、std::mutex、std::condition_variable、虚假唤醒处理 |

---

## 💯 面试实战话术

### Q: 你做过分布式训练吗？具体用过哪些并行方式？

> 我熟悉 3D 并行架构。用过基于 DDP/ZeRO 的数据并行解决数据和显存瓶颈；了解基于 Megatron 的张量并行（TP）处理单机内的大矩阵；也了解流水线并行（PP）处理跨机器的深层网络调度。

### Q: DDP 和 FSDP 的主要区别是什么？

> DDP 核心是全量参数复制，通过 Ring-AllReduce 同步梯度，优点是速度快，缺点是模型不能超出单卡显存。FSDP（基于 ZeRO-3）是参数切片，模型被拆解到各个卡上，运算时实时 All-Gather 拿回来，算完丢掉，极大地省了显存，牺牲了一点网络通信。

### Q: 为什么模型越大，通信问题越突出？

> 第一，大模型的显存装不下，逼迫我们使用 TP 和 PP，这两种并行自身就带有海量通信。第二，模型增大使单卡上的 Batch Size 被迫缩小，导致计算时间变短，无法有效 Overlap 越来越大的参数同步通信时间。

### Q: 如果 8 张卡扩到 64 张卡，吞吐上不去，你怎么分析？

> 这是典型的扩展性折损问题。首先用 Nsight 看通信占比，64 卡跨机 RDMA 速度远不如机内 NVLink，可能全在等网络 All-Reduce；其次，看 Global Batch Size 有没有等比例放大，如果没有，单卡分到的数据少了，GPU 就吃不满了。

### Q: 你怎么判断一个训练任务的瓶颈在算力、通信还是数据？

> 一看 top，CPU 满就是数据加载瓶颈；二看 Nsight Systems 的 Timeline：如果 GPU 呈现大块绿色的 Kernel 执行，就是算力瓶颈；如果大块红色的 NCCL Wait 或 cudaMemcpy，那就是通信或访存瓶颈。

### Q: 如果让你做一个训练优化项目，你会怎么设计实验验证收益？

> **控制变量法与 A/B Test。** 首先跑一个 Baseline 锚定 Loss 和 MFU。加入优化后，首先要验证"数学等价性"（在相同 seed 下跑到第 1000 步 Loss 是否能对齐小数点后三位）。如果正确，再去对比 Tokens/sec 的吞吐提升，以及通过 Profiler 证明瓶颈点真的被消除了。

---

## 🔥 牛客网面经汇总

> 以下内容整理自牛客网大模型推理加速 & AI Infra 实习岗位面试题汇总

### NVIDIA/HPC 岗

**C++ 基础：**
- 头文件作用、static 关键字、单例模式实现
- C++ 类型转换、浅拷贝 vs 深拷贝
- 友元类设计

**量化：**
- SmoothQuant 原理（缓解 INT8 精度下降）
- BF16 vs FP16 对比
- 精度与速度的 trade-off 设计

**CUDA 优化：**
- Shared Memory Bank Conflict 场景与解决
- cudaMalloc 二级指针设计原因
- 访存效率优化手段（coalesced access、shared memory、向量化）
- 计算效率优化（occupancy、kernel fusion、Tensor Core）

**大模型理论：**
- Encoder-only / Decoder-only / Encoder-Decoder 结构对比
- FlashAttention vs PagedAttention 本质区别
- 算子融合典型场景

### 30 秒口语答案

为面试高频题提供标准化的 30 秒回答模板，结构：**定义 → 重要性 → Trade-off**

涵盖 44 道高频题：
- **推理流程**：Prefill/Decode、TTFT/ITL、完整链路
- **KV Cache**：PagedAttention、Prefix Caching、MHA/MQA/GQA
- **调度**：Continuous Batching、Chunked Prefill、P/D 分离
- **并行**：DP/TP/PP/EP、通信原语
- **Kernel**：FlashAttention、CUDA Graph、Triton
- **量化与投机解码**：收益与局限
- **Benchmark 与框架对比**

### 资深面试官追问

涵盖高级面试题，拉开差距的 12 个专题：
- **Tokenizer 与输入链路**：BPE/WordPiece、CPU 并行化
- **模型加载与冷启动**：权重格式、warmup 策略
- **LoRA/多租户服务**：Adapter 切换策略、QoS 设计
- **多模态/VLM 推理**：Vision Encoder、长上下文处理
- **Structured Output**：JSON mode、Function Calling
- **线上故障排查**：TTFT 飙高、GPU 利用率低、可观测性指标
- **正确性验证**：量化验证标准、A/B 实验设计
- **SLO/成本规划**：cost/token 模型、autoscaling 策略
- **硬件认知**：HBM/Tensor Core、Hopper vs Ampere
- **系统设计题**：1000 并发服务设计、成本优化架构
- **项目方法论**：论文 vs 工程、瓶颈定位证明
- **反问陷阱题**：Decode memory-bound 但 Tensor Core 忙？

### 各公司面经

| 公司 | 重点内容 |
|-----|---------|
| **美团北斗** | Transformer 架构、GPU 规格、PD 分离、PagedAttention、手撕代码（链表反转、CUDA prefix sum、GEMM） |
| **混元** | FP4/低比特量化、Roofline 性能分析、矩阵维度选择 |
| **网易** | 项目介绍 STAR、微调方式对比、歌词生成系统设计、推理加速技术 |
| **快手** | LangGraph vs LangChain、向量数据库、RAG 切片策略、CoT、幻觉缓解 |
| **AI Infra 人才库** | 训练时间估算、Megatron 通信优化、DeepSeek 优化点、NCCL 原语、NVSHMEM |

**手撕代码高频题：**
- K 个一组翻转链表
- CUDA Prefix Sum（Blelloch Scan）
- CUDA GEMM（Tiling + Shared Memory 优化）
- RMSNorm / Online Softmax

---

## 📚 推荐资源

### 论文
- [FlashAttention](https://arxiv.org/abs/2205.14135) - 快速内存高效注意力
- [ZeRO](https://arxiv.org/abs/1910.02054) - 大规模模型训练优化
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) - LLM 推理优化

### 开源框架
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 微软分布式训练框架
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) - NVIDIA 大模型训练框架
- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA 推理优化库

### 学习资料
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)

---

## 📜 License

本仓库遵循 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 协议。

转载请注明出处，不得用于商业目的。

---

<p align="center">
  如果这个项目对你有帮助，欢迎 ⭐ Star 支持！
</p>
