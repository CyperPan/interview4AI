# 分布式训练基础

## 目录

- [数据并行、模型并行、流水线并行](#数据并行模型并行流水线并行)
- [DDP 基本原理](#ddp-基本原理)
- [Tensor Parallel vs Pipeline Parallel](#tensor-parallel-vs-pipeline-parallel)
- [流水线并行中的 Bubble](#流水线并行中的-bubble)
- [3D 混合并行策略](#3d-混合并行策略)
- [Megatron-LM vs DeepSpeed](#megatron-lm-vs-deepspeed)
- [通信技术全栈：NCCL / RDMA / GPU Direct / MPI](#通信技术全栈nccl--rdma--gpu-direct--mpi)
- [计算与通信 Overlap](#计算与通信-overlap)

---

## 数据并行、模型并行、流水线并行

### 1. 你如何理解数据并行、模型并行、流水线并行？它们分别解决什么问题？

**答：**

| 并行方式 | 核心思想 | 解决问题 |
|---------|---------|---------|
| **数据并行 (DP)** | 模型复制多份，数据切分 | 数据量太大，单卡算得慢 |
| **模型并行/张量并行 (TP)** | 模型单层（如矩阵乘法）切分到多卡 | 单层参数过大，单卡显存装不下 |
| **流水线并行 (PP)** | 模型按深度（层）切分到多卡 | 整体模型太深，单卡装不下，且跨节点通信量小 |

### 2. 为什么单纯使用数据并行在大模型训练中会遇到瓶颈？

**答：** 主要是显存墙（OOM）。纯 DP 要求每张卡完整保存一份：
- 模型权重
- 梯度
- 优化器状态（Adam 占大头）

大模型（如 70B）单是这些状态就需要上 TB 显存，远超单卡 H100（80GB）极限。其次是参数量太大导致的 All-Reduce 通信延迟极高。

---

## DDP 基本原理

### DDP 的基本原理是什么？训练过程中梯度同步是怎么做的？

**答：**

DDP 采用**多进程架构**：
1. **初始化时 Broadcast**：保证各卡权重一致
2. **训练时**：各卡独立前向/反向计算
3. **梯度同步**：采用 **Ring-AllReduce 算法**
4. **Bucket（分桶）机制**：反向传播时，一旦某几个层的梯度填满了一个 Bucket，就立刻触发 All-Reduce，实现网络通信和后续反向计算的 **Overlap（重叠）**

---

## Tensor Parallel vs Pipeline Parallel

### 什么是模型并行？Tensor Parallel 和 Pipeline Parallel 有什么区别？

**答：**

模型并行就是"把大模型拆开算"。

| 特性 | Tensor Parallel (TP) | Pipeline Parallel (PP) |
|-----|---------------------|----------------------|
| **切分维度** | 层内切分（对算子如 Linear 的权重矩阵切片） | 层间切分（把前面的层和后面的层分给不同 GPU） |
| **通信方式** | 极密集的 All-Reduce 通信 | 只传递激活值（P2P 通信） |
| **适用场景** | 只能在单机内（依赖 NVLink） | 通信量极小，适合跨机器使用 |
| **通信频率** | 每次前向后向都需要 | 只传递激活值 |

---

## 流水线并行中的 Bubble

### 流水线并行中的 bubble 是什么？怎么减少 bubble？

**答：**

**Bubble（气泡）**是指后面层的 GPU 在等待前面层 GPU 计算结果时的**闲置空转时间**。

**减少 Bubble 的方法：**

1. **Micro-batching（微批次）**：将大 batch 切分成多个 micro-batch
2. **1F1B（一前一后）调度策略**：让各阶段紧凑交替执行
3. **Interleaved 1F1B（交错式）**：进一步缩小气泡比例

```
传统流水线：
GPU0: [F1][F2][F3][F4]      [B4][B3][B2][B1]
GPU1:      [F1][F2][F3][F4]      [B4][B3][B2][B1]
            ↑←← Bubble →→↑

1F1B 调度：
GPU0: [F1][F2][B1][F3][B2][F4][B3]   [B4]
GPU1:      [F1][F2][B1][F3][B2][F4][B3][B4]
            Bubble 大幅减少（startup/drain 阶段仍有少量 bubble）
```

---

## Expert Parallelism（EP / 专家并行）

### 什么是专家并行？它和 TP/PP 有什么区别？

**答：**

**EP 专用于 MoE 模型**，把不同的 expert 分配到不同的 GPU。

```
GPU 0: Expert 0, 1, 2, 3
GPU 1: Expert 4, 5, 6, 7

Token routing:
Token A → Expert 2 (GPU 0) + Expert 5 (GPU 1)  ← AllToAll
Token B → Expert 0 (GPU 0) + Expert 7 (GPU 1)  ← AllToAll
```

| 维度 | TP | PP | EP |
|------|----|----|-----|
| 切分对象 | 同一层权重 | 不同层 | 不同 expert |
| 通信原语 | AllReduce | P2P Send/Recv | AllToAll |
| 适用模型 | Dense 和 MoE | Dense 和 MoE | **仅 MoE** |
| 通信频率 | 每层 | 每 stage | 每个 MoE 层 |

### 在 vLLM 中如何实现各种并行？

**答：**

| 并行方式 | vLLM 实现 | 使用方式 |
|---------|----------|---------|
| **DP** | 多 worker 进程，各加载完整模型，请求负载均衡分发 | 模型单卡能放下时首选，零通信开销 |
| **TP** | `--tensor-parallel-size N`，自动切分 QKV/FFN 权重，NCCL AllReduce 同步 | 需要 NVLink 高带宽互连 |
| **PP** | `--pipeline-parallel-size N`，不同层分配到不同卡，micro-batching 减少 bubble | 适合跨机部署 |
| **EP** | 与 TP 结合，MoE 层自动做 expert 分片和 token 路由 | MoE 模型使用 |

**选型经验法则：**
- 模型单卡能放下 → **DP**
- 需要 2-8 卡（机内） → **TP**
- 需要跨机 → TP（机内）+ PP（跨机）
- MoE 模型 → TP + EP

**推理和训练并行的区别：** 推理 batch 小、延迟敏感，首选 DP > TP > PP；训练关心总吞吐和显存，常用 3D 并行（DP+TP+PP）。

---

## 3D 混合并行策略

### 如果让你为一个超大模型设计并行策略，你会如何在 DP、TP、PP 之间做权衡？

**答：**

采用 **3D 混合并行**：

```
┌─────────────────────────────────────────┐
│           全局 DP / ZeRO                 │  ← 数据并行，增加副本数
│  ┌─────────────────────────────────┐    │
│  │      跨机器 PP (Inter-node)      │    │  ← 机器间带宽较慢 (RDMA)
│  │  ┌─────────────────────────┐    │    │
│  │  │  单机内 TP (Intra-node)  │    │    │  ← 机器内带宽极高 (NVLink)
│  │  │  GPU0 GPU1 GPU2 GPU3     │    │    │
│  │  └─────────────────────────┘    │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

**具体策略：**
- **单机内（Intra-node）**：机器内带宽极高（NVLink），跑 TP（通常 TP=4 或 8），切分大矩阵
- **跨机器（Inter-node）**：机器间带宽较慢（RDMA），跑 PP，按层分配到不同机器
- **全局**：在 TP 和 PP 的组别之外套上 DP / ZeRO，增加副本数以消化海量数据

---

## Megatron-LM vs DeepSpeed

### Megatron-LM 和 DeepSpeed 的区别，能不能同时用

**答：**

**一句话区分：** Megatron 做**模型并行**（怎么把模型切到多张卡上），DeepSpeed 做**数据并行优化**（怎么让每张卡少存一点东西）。

| 维度 | Megatron-LM | DeepSpeed |
|------|-------------|-----------|
| **核心贡献** | TP（层内 Column/Row 切分）+ PP（层间流水线） | ZeRO（切分优化器状态/梯度/参数） |
| **解决问题** | 单模型太大，一张卡放不下 | 数据并行时冗余存储太多 |
| **通信类型** | AllReduce（TP）、Send/Recv（PP） | AllGather / ReduceScatter（ZeRO） |
| **代码侵入性** | 高（需要改模型代码） | 低（插件式，改配置即可） |
| **适用规模** | 万卡级集群 | 中小规模也友好 |

**能同时用，这就是 3D 并行：**

```
节点内 8 卡：TP=8（Megatron 的张量并行，走 NVLink）
节点间 8 组：PP=8（Megatron 的流水线并行，走 RDMA）
多组副本间：ZeRO-1 数据并行（DeepSpeed，切分优化器状态）

总卡数 = TP × PP × DP = 8 × 8 × 16 = 1024 张卡
```

### TP 下 AllReduce 的通信分析：为什么每层恰好 2 次

**答：**

**Column Parallel + Row Parallel 的配对设计：**

以 MLP 为例（TP=2）：

```
gate_up 用 ColumnParallelLinear（按列切权重）：
  卡0: x @ W_gate[:, :mid/2] → 得到一半中间结果
  卡1: x @ W_gate[:, mid/2:] → 得到另一半
  ← 各卡拿完整 x，独立算各自那半输出，不需要通信

down 用 RowParallelLinear（按行切权重）：
  卡0: half_0 @ W_down[:mid/2, :] → 局部结果
  卡1: half_1 @ W_down[mid/2:, :] → 局部结果
  ← 两卡的局部结果加起来才是正确答案 → AllReduce
```

**Attention 同理：** QKV 用 ColumnParallel 按 head 切（不通信），O 用 RowParallel（AllReduce）。

**规律：** ColumnParallel 切输出维度 → 不通信；RowParallel 切输入维度 → 需 AllReduce 求和。一个 Column 接一个 Row，中间不通信，Row 后通信一次。每层 Attention + MLP = **2 次 AllReduce**。

### Ring AllReduce 详解

**答：**

**场景：** 4 张卡，每张卡有一份局部数据，要得到全局求和。

```
阶段一 ReduceScatter（N-1=3 轮）：
  把数据分成 4 块（chunk 0-3），沿环传递
  每轮每张卡接收一块并累加
  3 轮后：每张卡持有 1 个 chunk 的完整求和结果

阶段二 AllGather（N-1=3 轮）：
  每张卡把自己持有的完整 chunk 沿环广播
  3 轮后：所有卡都有全部 4 个 chunk 的完整求和
```

**关键性质：**
- 每张卡总通信量 ≈ **2 × 数据大小**，跟卡数无关（带宽不是瓶颈）
- 但轮次 = 2×(N-1)，卡越多**延迟越高**（延迟是瓶颈）
- 这就是 TP 不能无限扩、不适合跨节点的原因

**注意：** AllReduce ≠ 把数据发到 rank 0 求和再广播（那叫 Reduce + Broadcast，效率极低）。

---

## 通信技术全栈：NCCL / RDMA / GPU Direct / MPI

### 四层通信技术栈

**答：**

从上层到底层：

```
应用层：  PyTorch dist.all_reduce()
            ↓
通信库层： NCCL（GPU 专用） / MPI（通用）
            ↓
传输技术层：RDMA + GPU Direct（绕过 CPU）
            ↓
硬件层：   NVLink（节点内） / InfiniBand（跨节点）
```

### RDMA（Remote Direct Memory Access）

**答：**

```
传统网络：应用 → 内核 → 驱动 → 网卡 → 网线 → 网卡 → 驱动 → 内核 → 应用
          CPU 全程参与，多次内存拷贝，延迟 ~几十微秒

RDMA：   应用 → 网卡 → 网线 → 网卡 → 远程内存
          CPU 几乎不参与，零拷贝，延迟 ~几微秒
```

最常见实现：**InfiniBand 网络 + RDMA**，带宽 200-400 Gbps。

### GPU Direct（逐级缩短路径）

**答：**

```
最原始：GPU → CPU内存 → 内核 → 网卡 → 网线 → 网卡 → 内核 → CPU内存 → GPU
加RDMA：GPU → CPU内存 → 网卡 → 网线 → 网卡 → CPU内存 → GPU    （省掉内核）
加GDR： GPU → 网卡 → 网线 → 网卡 → GPU                        （省掉CPU内存拷贝）
节点内： GPU → NVLink → GPU                                    （最短路径）
```

| 级别 | 含义 | 省掉了什么 |
|------|------|-----------|
| **GPU Direct P2P** | 节点内 GPU 直接通过 NVLink/PCIe 传数据 | CPU 内存中转 |
| **GPU Direct RDMA (GDR)** | 网卡直接读写 GPU 显存 | CPU 内存拷贝 |
| **GPU Direct Storage** | NVMe SSD 直接搬数据到 GPU | CPU 内存中转 |

### NCCL vs MPI

**答：**

| 维度 | NCCL | MPI |
|------|------|-----|
| **定位** | NVIDIA GPU 专用集合通信库 | 通用分布式通信标准 |
| **优化** | 深度利用 NVLink、GPU Direct、IB | 通用实现，GPU 性能次之 |
| **支持硬件** | 仅 NVIDIA GPU | CPU + 任意 GPU |
| **API 风格** | AllReduce、AllGather 等集合操作 | Send/Recv + 集合操作 |
| **AI 场景** | **首选**（PyTorch 默认 backend） | 有时用于进程管理（mpirun） |

NCCL 自动检测硬件拓扑，选择最优路径和算法。开发者调 `dist.all_reduce()` → PyTorch 调 NCCL → NCCL 调 GPU Direct RDMA + InfiniBand。

---

## 计算与通信 Overlap

### TP 的 AllReduce 怎么跟计算做 Overlap

**答：**

**实现机制：CUDA 多 Stream**

GPU 有独立的计算单元（SM/Tensor Core）和通信单元（NVLink/PCIe 的 DMA 引擎），物理上可同时工作。

```
无 Overlap（默认，单 stream）：
Stream 0: [Linear 计算] → [AllReduce 等待] → [下一层计算] → [AllReduce 等待]
                           ^^^ GPU 计算单元闲等

有 Overlap（多 stream）：
Stream 0 (计算): [Linear A] → [Linear B] → ...
Stream 1 (通信):    [AllReduce A] → [AllReduce B] → ...
                 计算和通信同时进行
```

**三种 Overlap 策略：**

| 策略 | 做法 | 适用场景 |
|------|------|---------|
| **层间 Overlap** | 当前层 AllReduce 与下一层 RMSNorm+QKV 并行 | 需要改模型结构 |
| **拆分 AllReduce** | AllReduce = ReduceScatter + 中间计算 + AllGather | 中间有可用计算时 |
| **分块流水线** | 大 AllReduce 拆成多个小块，逐块通信+计算 | 通用 |

**本质：** 让通信延迟被计算时间遮盖，CUDA 多 stream 是实现手段。

---

## 面试金句

> "我熟悉 3D 并行架构。用过基于 DDP/ZeRO 的数据并行解决数据和显存瓶颈；了解基于 Megatron 的张量并行（TP）处理单机内的大矩阵；也了解流水线并行（PP）处理跨机器的深层网络调度。"

> "DDP 核心是全量参数复制，它通过 Ring-AllReduce 同步梯度，优点是速度快，缺点是模型不能超出单卡显存。FSDP（基于 ZeRO-3）是参数切片，模型被拆解到各个卡上，运算时实时 All-Gather 拿回来，算完丢掉，极大地省了显存，牺牲了一点网络通信。"

> "Megatron 做模型并行（TP/PP），DeepSpeed 做数据并行优化（ZeRO）。两者互补，3D 并行 = 节点内 TP + 节点间 PP + 多组间 ZeRO。"

> "NCCL 是 GPU 集合通信首选，底层跨节点走 InfiniBand + GPU Direct RDMA 绕过 CPU 内存。节点内走 NVLink。Ring AllReduce 每卡通信量恒定，但轮次随卡数增加，延迟是 TP 不能无限扩的瓶颈。"

> "计算与通信 Overlap 靠 CUDA 多 Stream 实现——计算任务和通信任务放不同 Stream，利用计算单元和 DMA 引擎物理独立的特性并行执行。"
