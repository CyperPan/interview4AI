# 各公司面经汇总（字节、美团、混元、网易、快手等）

> 来源：牛客网多岗位汇总

---

## 目录

- [美团北斗 AI Infra](#美团北斗-ai-infra)
- [字节 AI Infra 实习二面](#字节-ai-infra-实习二面)
- [混元 AI Infra](#混元-ai-infra)
- [网易大模型应用开发](#网易大模型应用开发)
- [快手 AI 应用开发](#快手-ai-应用开发)
- [AI Infra 人才库面经](#ai-infra-人才库面经)

---

## 美团北斗 AI Infra

### Transformer 架构相关问题

#### 1. 介绍一下 Transformer 的整体架构，相比传统 RNN 有哪些优势？

**答：**

**Transformer 架构：**
```
Input → Embedding + Positional Encoding
        ↓
    [Encoder × N]  (Self-Attention + FFN)
        ↓
    [Decoder × N]  (Masked Self-Attention + Cross-Attention + FFN)
        ↓
Output → Linear + Softmax
```

**相比 RNN 的优势：**

| 特性 | RNN | Transformer |
|-----|-----|-------------|
| **并行性** | 序列化计算 | 完全并行 |
| **长距离依赖** | 梯度消失，难以捕捉 | Self-Attention 直接连接 |
| **训练速度** | 慢 | 快（适合大规模预训练） |
| **位置信息** | 天然有序列信息 | 需要 Position Encoding |

#### 2. Transformer 中参数主要分布在哪些模块？参数量最大的是哪一部分？计算量最大的是哪一部分？

**答：**

**参数分布：**

| 模块 | 参数量占比 | 说明 |
|-----|-----------|------|
| **Embedding** | ~20% | 词表 × 维度 |
| **Attention** | ~10% | Q/K/V/O 投影矩阵 |
| **FFN** | ~60% | 两个大矩阵 (4d × d) |
| **LayerNorm** | ~1% | 可忽略 |

**参数量最大：FFN**（约占总参数 60-70%）

**计算量最大：**
- **Prefill**：Attention（O(n²) 复杂度）
- **Decode**：Attention 或 FFN（取决于实现）

### GPU 基础

#### 3. 了解 GPU 的 CUDA Core、Tensor Core 吗？常用 GPU 型号及规格？

**答：**

| GPU | CUDA Cores | Tensor Cores | 显存 | 显存带宽 |
|-----|-----------|--------------|------|---------|
| **A100** | 6912 | 3rd Gen | 40/80 GB | 2039 GB/s |
| **H100** | ~15000 | 4th Gen | 80 GB | 3350 GB/s |
| **4090** | 16384 | 4th Gen | 24 GB | 1008 GB/s |

**CUDA Core**：通用浮点运算单元
**Tensor Core**：专门加速矩阵乘法的硬件单元（支持 FP16/BF16/INT8/FP8）

### 量化与推理优化

#### 4. 讲讲大模型的量化思路及常见量化策略

**答：**

**量化思路：**
1. **降低精度**：FP32 → FP16/BF16 → INT8/FP8 → INT4
2. **减少存储**：权重占用减半或更少
3. **加速计算**：利用低精度 Tensor Core

**常见策略：**

| 策略 | 特点 |
|-----|------|
| **Weight-only** | 只量化权重，激活保持 FP16 |
| **W8A8** | 权重和激活都量化到 INT8 |
| **W4A16** | 权重 INT4，激活 FP16 |
| **FP8** | H100 原生支持，精度损失小 |

### PD 分离与 PagedAttention

#### 5. 详细讲一讲「参数/数据 (PD) 分离」的思路和收益

**答：**

详见 [InferenceOptimization.md](./InferenceOptimization.md#pd-分离)

#### 6. 详细讲一下 PagedAttention

**答：**

详见 [InferenceOptimization.md](./InferenceOptimization.md#pagedattention)

### 手撕代码

#### 7. LeetCode：K 个一组翻转链表

```cpp
ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode dummy(0, head);
    ListNode* prev = &dummy;
    
    while (true) {
        // 检查是否还有 k 个节点
        ListNode* tail = prev;
        for (int i = 0; i < k && tail; ++i) {
            tail = tail->next;
        }
        if (!tail) break;
        
        // 翻转这 k 个节点
        ListNode* next_group = tail->next;
        ListNode* p = prev->next;
        ListNode* q = tail->next;
        
        // 头插法翻转
        while (p != q) {
            ListNode* next = p->next;
            p->next = prev->next;
            prev->next = p;
            p = next;
        }
        
        prev->next->next = next_group;
        prev = tail;
    }
    
    return dummy.next;
}
```

#### 8. 手写 CUDA 算子：前缀和 (prefix sum)

```cuda
// Baseline
__global__ void prefix_sum_baseline(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i <= idx; ++i) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

// 优化：双调排序 / Blelloch Scan
__global__ void prefix_sum_blelloch(float* data, int n) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    
    // Load to shared memory
    temp[tid] = (tid < n) ? data[tid] : 0;
    __syncthreads();
    
    // Up-sweep phase
    for (int stride = 1; stride < n; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }
    
    // Down-sweep phase
    if (tid == 0) temp[n - 1] = 0;
    __syncthreads();
    
    for (int stride = n / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            float t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }
    
    // Write back
    if (tid < n) data[tid] = temp[tid];
}
```

#### 9. 手写 CUDA GEMM（矩阵乘）

```cuda
// 基础版本
__global__ void gemm_baseline(const float* A, const float* B, float* C, 
                               int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 优化版本：Tiling + Shared Memory
#define TILE_SIZE 16

__global__ void gemm_optimized(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0;
    
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Load tile to shared memory
        if (row < M && tile * TILE_SIZE + tx < K) {
            s_A[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        } else {
            s_A[ty][tx] = 0;
        }
        
        if (tile * TILE_SIZE + ty < K && col < N) {
            s_B[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            s_B[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## 字节 AI Infra 实习二面

### 这一轮题目的特点

这组题明显不是普通的“概念快答”，而是直接考你能不能把推理结构、稀疏架构、访存估算和通信优化写出来、算出来、讲明白。

### Attention / MoE / 模型结构手撕

#### 1. 手撕 `linear attention`

**考察重点：**

- 会不会把 `softmax(QK^T)V` 改写成可前缀累计的形式
- 知道它不是“白送更快”，而是用近似或特征映射换复杂度
- 能说清它和标准 attention 在长序列下的复杂度与适用场景差异

#### 2. 手撕 `MHA`

**考察重点：**

- `Q / K / V` 投影和多头 reshape
- causal mask 的位置
- 为什么 decode 阶段更关心 `KV Cache` 而不是单纯 FLOPs

#### 3. 手撕 `MLA`

**考察重点：**

- 知道它不是普通 `MHA` 改名，而是通过 latent 压缩来降低 KV 侧缓存和带宽
- 伪代码里要体现 “先压缩缓存，再恢复 K/V 或等价表示参与注意力”
- 能和 `GQA / MQA` 做对比，而不是孤立解释

#### 4. 手撕 `MoE`

**考察重点：**

- `router -> top-k -> dispatch -> expert -> combine` 整条链路
- 区分总参数量和单 token 激活参数量
- 能解释为什么 MoE 经常是“计算省了，但通信更难”

#### 5. 手撕 `DeepSeek-V3` 结构化伪代码

**考察重点：**

- 主干要写出 `Attention / MLA + MoE + 残差 + Norm`
- 解释哪些部分决定长上下文成本，哪些部分决定专家通信成本
- 面试里比起背官网表格，更重要的是把模块关系讲对

### 参数量与访存量手算

#### 6. 手算上面结构的参数量和 decode 访存量

**考察重点：**

- `MHA / MLA / MoE / linear attention` 不只是会写，还要会估算
- 区分 `prefill` 的大矩阵计算和 `decode` 的权重 / KV 读取
- 说清 `参数量大` 不等于 `单步推理一定慢`

#### 7. 手算 `DeepSeek-V3` 每次推理的主要访存开销

**考察重点：**

- dense backbone 的权重读取
- `MLA` 对 KV 侧开销的压缩
- `MoE` 的激活 expert 权重读取与跨卡分发

### 算法与通信手撕

#### 8. 算法题：堆排序

**考察重点：**

- `heapify` 和建堆 / 取顶过程是否写得稳
- 时间复杂度 `O(n log n)` 和原地排序特性

#### 9. 手撕 `reduce` 并继续讲优化

**考察重点：**

- shared memory 规约
- warp shuffle / 两级规约
- global memory coalescing、bank conflict、同步开销

#### 10. 继续追问 collective reduction 优化

**考察重点：**

- `all-reduce / reduce-scatter` 的拆分方式
- bucket、overlap、拓扑感知
- 通信原语为什么会成为训练或 MoE 的瓶颈

### 推荐跳转

- 手撕实现：见 [CodingProblems.md](./CodingProblems.md)
- 参数量 / 访存量手算：见 [LLMMathDerivations.md](./LLMMathDerivations.md)
- 按推理链路整理：见 [InferenceInterviewByPipeline.md](./InferenceInterviewByPipeline.md)

---

## 混元 AI Infra

### FP4 / 低比特量化

#### 1. FP4 / 低比特量化的基本概念与典型实现方式

**答：**

**FP4 格式：**
- 1 bit sign
- 2 bits exponent
- 1 bit mantissa

**实现方式：**
- **Microscaling**：分组缩放因子
- **Block-wise quantization**：块级量化减少误差

**对比：**

| 格式 | 精度 | 速度 | 适用场景 |
|-----|------|------|---------|
| **FP8** | 高 | 快 | 通用推理 |
| **INT8** | 中高 | 快 | 权重/激活量化 |
| **INT4** | 中 | 中等 | 极致压缩 |
| **FP4** | 低 | 需硬件支持 | 研究探索 |

### GPU 性能分析

#### 2. GPU 性能分析时，你会如何定位系统瓶颈？

**答：**

**Roofline Model：**
```
Compute Roofline:
    ↑
  P |     _______  <-- 峰值算力
  e |    /
  a |   /
  k |  /
    | / Memory Bandwidth Bound
    |/________________→ Arithmetic Intensity
```

**分析步骤：**
1. **计算 Arithmetic Intensity** (FLOPs / Bytes)
2. **与 Roofline 对比** 确定瓶颈区域
3. **Profiler 验证**：
   - Compute-bound：SM 利用率高
   - Memory-bound：内存带宽饱和

---

## 网易大模型应用开发

### 项目与工程

#### 1. 请详细介绍一个与大模型相关的项目

**答（框架）：**

```
项目介绍 STAR 法则：

Situation（背景）：
- 业务场景：如智能客服、内容审核、代码生成
- 技术挑战：延迟要求高、上下文长、准确性要求

Task（任务）：
- 负责模块：模型选型、推理优化、服务部署
- 目标指标：TTFT < 200ms，Throughput > 1000 tokens/s

Action（行动）：
- 技术方案：vLLM + PagedAttention + Continuous Batching
- 优化手段：量化、KV Cache 优化、PD 分离
- 工程实现：Kubernetes 部署、Auto-scaling

Result（结果）：
- 性能提升：延迟降低 50%，成本降低 30%
- 业务价值：支持 X 并发，服务 X 用户
```

### 微调方式对比

#### 2. 几种微调方式对比：Full Fine-tuning / LoRA / Adapter / P-Tuning

**答：**

| 方法 | 训练参数 | 显存占用 | 适用场景 |
|-----|---------|---------|---------|
| **Full FT** | 100% | 高 | 数据充足、算力充足 |
| **LoRA** | <1% | 低 | 快速适配、多任务 |
| **Adapter** | ~2-4% | 中 | 模块化、可组合 |
| **P-Tuning** | 可忽略 | 极低 | 特定任务快速验证 |

### 推理加速技术

#### 3. 大模型推理加速技术：量化、动态批处理、FlashAttention

**答：**

详见各专题文档：
- [量化](./MemoryOptimization.md#混合精度训练)
- [动态批处理](./InferenceOptimization.md#continuous-batching)
- [FlashAttention](./NvidiaHPCInterview.md#cuda-与-gpu-优化)

### 系统设计

#### 4. 为网易云音乐设计「AI 歌词生成系统」

**答（思路）：**

```
1. 数据层：
   - 高质量歌词语料
   - 风格标签（古风、流行、说唱等）
   - 韵律、押韵标注

2. 模型层：
   - 基础模型：开源中文 LLM
   - 微调策略：领域 SFT + RLHF
   - 韵律控制：Constrained Decoding

3. 推理层：
   - vLLM 加速
   - 流式输出
   - 缓存热门风格

4. 产品层：
   - 主题输入 → 风格选择 → 歌词生成
   - 人工反馈闭环
```

---

## 快手 AI 应用开发

### LangChain / LangGraph

#### 1. LangGraph 相比于 LangChain 有哪些优势？

**答：**

| 特性 | LangChain | LangGraph |
|-----|-----------|-----------|
| **架构** | 链式调用 | 图结构 |
| **状态管理** | 简单上下文 | 持久化状态快照 |
| **循环/条件** | 有限支持 | 原生支持 |
| **适用场景** | 简单 pipeline | 复杂 Agent 工作流 |

**LangGraph 状态快照：**
```python
# 允许随时保存和恢复执行状态
state = graph.get_state(thread_id)
graph.update_state(thread_id, new_state)
```

### RAG 与向量数据库

#### 2. RAG 中，文档切片粒度如何选择？

**答：**

| 粒度 | 优点 | 缺点 |
|-----|------|------|
| **大段落** | 上下文完整 | 可能包含无关信息 |
| **小段落** | 精确匹配 | 可能丢失上下文 |
| **重叠窗口** | 平衡 | 增加存储 |

**选择策略：**
- 根据问题和文档特性动态调整
- 大段落做粗排，小段落做精排

#### 3. 向量数据库索引：IVF_FLAT 与 HNSW 的区别

**答：**

| 特性 | IVF_FLAT | HNSW |
|-----|----------|------|
| **构建时间** | 快 | 慢 |
| **查询速度** | 中等 | 快 |
| **内存占用** | 小 | 大 |
| **召回率** | 高（精确） | 高（近似） |
| **适用规模** | 中小规模 | 大规模 |

### 大模型应用

#### 4. 什么是 CoT（Chain-of-Thought）？

**答：**

**定义：** 让模型在给出最终答案前，先显式输出推理过程。

**示例：**
```
Q: 一个农场有鸡和兔，共 35 个头，94 只脚，各多少只？

CoT 回答：
设鸡 x 只，兔 y 只。
1. x + y = 35（头的数量）
2. 2x + 4y = 94（脚的数量）
从方程1：x = 35 - y
代入方程2：2(35-y) + 4y = 94
70 - 2y + 4y = 94
2y = 24
y = 12
所以兔子 12 只，鸡 23 只。
```

**为什么有效：**
- 分解复杂问题为简单步骤
- 减少推理错误累积
- 提高可解释性

#### 5. 大模型应用中常见的「幻觉」类型及缓解方法

**答：**

| 幻觉类型 | 表现 | 缓解方法 |
|---------|------|---------|
| **事实性幻觉** | 编造不存在的事实 | RAG、事实核验 |
| **忠实性幻觉** | 回答与问题无关 | Prompt 约束、RLHF |
| **逻辑幻觉** | 推理步骤错误 | CoT、Self-consistency |

---

## AI Infra 人才库面经

### 训练时间估算

#### 1. 给定训练所需的 token 总量，如何估算模型训练所需的完整时间？

**答：**

**公式：**
```
Training Time = (Total Tokens × 6 × Parameters) / (GPU Flops × GPU Count × MFU × 1e12)

其中：
- 6 × Parameters：每个 token 前向+反向的 FLOPs
- MFU：Model FLOPs Utilization（通常 30-50%）
```

**需要考虑的因素：**
- 模型参数量
- 总 token 数
- GPU 算力（A100/H100）
- GPU 数量
- 并行策略效率
- 通信 overhead
- Checkpoint 和故障恢复时间

### Megatron-LM 通信优化

#### 2. 在 Megatron-LM 中，通信优化是如何做的？

**答：**

**张量并行（TP）：**
- All-Reduce 通信
- 使用 NCCL 优化
- 尽可能在机内（NVLink）完成

**流水线并行（PP）：**
- P2P 通信（send/recv）
- 1F1B 调度减少 bubble
- 异步通信 overlap 计算

**优化手段：**
- **Bucket**：聚合小通信
- **Overlap**：通信与计算重叠
- **拓扑感知**：NVLink > InfiniBand > Ethernet

### DeepSeek 相关

#### 3. DeepSeek-V3 的主要优化点有哪些？

**答：**

- **MoE 架构**：Sparse activation，降低推理成本
- **MLA（Multi-head Latent Attention）**：压缩 KV Cache
- **FP8 训练**：全流程 FP8，节省显存和计算
- **DualPipe**：高效的流水线并行调度

#### 4. DeepSeek-DSA 与 NSA、MoBA 的区别

**答：**

| 方法 | 核心思想 |
|-----|---------|
| **DSA** | Dynamic Sparse Attention，动态稀疏注意力 |
| **NSA** | Native Sparse Attention，原生稀疏架构 |
| **MoBA** | Mixture of Block Attention，分块混合注意力 |

**共同点：** 都在探索如何在长上下文下降低 Attention 复杂度。

### NCCL 通信

#### 5. NCCL 中有哪些常见通信原语？一次 all-reduce 参数更新需要几次通信？

**答：**

**通信原语：**
- `ncclAllReduce`：全规约
- `ncclAllGather`：全收集
- `ncclReduceScatter`：规约分散
- `ncclBroadcast`：广播
- `ncclAllToAll`：全对全

**All-Reduce 通信次数（Ring 算法）：**
- 2 × (N-1) 次通信（Scatter-Reduce + All-Gather）
- N 为 GPU 数量

### NVSHMEM

#### 6. 在小数据量场景下使用 NVSHMEM，相比 ring all-reduce 的好处和代价

**答：**

**好处：**
- 直接读取其他 GPU 数据，减少同步
- 适合不规则访问模式

**代价：**
- 编程复杂度更高
- 需要显式管理内存一致性
- 大数据量下性能可能不如优化后的 NCCL

---

## 面试金句

> "我在美团实习时做的推理优化项目，核心是用 vLLM 的 PagedAttention 和 Continuous Batching，把服务的 P99 延迟从 2s 降到了 500ms，同时支持了 3 倍的并发。"

> "对于长上下文场景，我一般会先看 KV Cache 占用，然后考虑 MLA 或 GQA 压缩，最后看是否需要 Context Parallel。"

> "在做混元的 FP4 量化时，我们发现主要挑战是精度损失，最后采用了 block-wise quantization + 微缩放因子的方案。"
