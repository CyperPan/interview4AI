# NVIDIA / HPC 岗面经（牛客网汇总）

> 来源：牛客网 AI深度学习推理加速HPC类岗位-Nvidia面经

---

## 目录

- [C++ 基础与工程](#c-基础与工程)
- [量化 Quantization](#量化-quantization)
- [CUDA 与 GPU 优化](#cuda-与-gpu-优化)
- [大模型理论与推理优化](#大模型理论与推理优化)
- [GPU 硬件与编程模型补充](#gpu-硬件与编程模型补充)

---

## C++ 基础与工程

### 1. 为什么我们做 C++ 项目的时候，需要写头文件？

**答：**

头文件（.h/.hpp）的主要作用是：
- **接口声明**：暴露类的定义、函数签名、宏定义给使用者
- **编译分离**：实现编译期类型检查，支持分别编译
- **避免重复定义**：配合 include guard 或 pragma once
- **代码组织**：将实现细节隐藏在 .cpp 中，提高封装性

### 2. 讲出 static 关键字的一种应用场景

**答：**

| 应用场景 | 说明 |
|---------|------|
| **函数内 static 变量** | 保持状态跨调用，如计数器 |
| **文件作用域 static** | 限制符号只在当前文件可见，避免命名冲突 |
| **类内 static 成员** | 所有对象共享，如单例模式 |
| **类内 static 方法** | 无需对象实例即可调用 |

**典型例子 - 单例模式：**
```cpp
class Singleton {
public:
    static Singleton& getInstance() {
        static Singleton instance;  // C++11 线程安全
        return instance;
    }
private:
    Singleton() = default;
};
```

### 3. 单例模式如何实现？请写出或描述典型实现方式，并说明线程安全问题

**答：**

**Meyers' Singleton（推荐，C++11 起线程安全）：**

```cpp
class Singleton {
public:
    static Singleton& getInstance() {
        static Singleton instance;  // 线程安全初始化
        return instance;
    }
    
    // 禁止拷贝
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;

private:
    Singleton() = default;
    ~Singleton() = default;
};
```

**双检查锁（DCL，C++11 前）：**

```cpp
class Singleton {
public:
    static Singleton* getInstance() {
        if (instance_ == nullptr) {           // 第一次检查
            std::lock_guard<std::mutex> lock(mutex_);
            if (instance_ == nullptr) {       // 第二次检查
                instance_ = new Singleton();
            }
        }
        return instance_;
    }

private:
    static std::atomic<Singleton*> instance_;
    static std::mutex mutex_;
};
```

**线程安全说明：**
- C++11 起，函数内 static 变量初始化是线程安全的（由编译器保证）
- DCL 需要配合 memory barrier，否则可能看到未构造完成的对象

### 4. C++ 中有哪几种类型转换？各自适用场景

**答：**

| 转换类型 | 适用场景 | 安全性 |
|---------|---------|--------|
| **static_cast** | 相关类型转换（如 int→float、基类→派生类指针） | 编译期检查 |
| **dynamic_cast** | 多态类型向下转换 | 运行时检查，失败返回 nullptr |
| **const_cast** | 添加/移除 const/volatile | 可能未定义行为 |
| **reinterpret_cast** | 底层位重新解释（如指针↔整数） | 最危险 |

**推理场景例子：**
```cpp
// static_cast: 精度转换
float f = static_cast<float>(int_val);

// reinterpret_cast: CUDA 指针转换
void* d_ptr;
cudaMalloc(&d_ptr, size);
float* f_ptr = reinterpret_cast<float*>(d_ptr);
```

### 5. 拷贝构造函数中浅拷贝和深拷贝的区别？在什么场景下必须做深拷贝？

**答：**

| 特性 | 浅拷贝 | 深拷贝 |
|-----|--------|--------|
| **行为** | 复制指针值，共享资源 | 分配新内存，复制内容 |
| **问题** | 双重释放、悬空指针 | 内存开销增加 |
| **触发条件** | 默认拷贝构造 | 自定义拷贝构造 |

**必须深拷贝的场景：**
- 类包含指针成员（如动态数组）
- 资源需要独立管理（如文件句柄、GPU 显存指针）

```cpp
class Tensor {
    float* data_;
    size_t size_;
public:
    // 深拷贝
    Tensor(const Tensor& other) : size_(other.size_) {
        data_ = new float[size_];
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    }
    
    ~Tensor() { delete[] data_; }
};
```

### 6. 一个类要去访问另一个类的 private 数据成员，该如何设计？

**答：**

| 方法 | 适用场景 |
|-----|---------|
| **friend 友元** | 紧密耦合的类，如迭代器访问容器 |
| **公共接口（getter）** | 一般情况，保持封装 |
| **内部嵌套类** | 天然的访问权限 |

**推荐优先使用公共接口**，保持封装性。

---

## 量化 Quantization

### 1. 说说你知道的针对大模型/LLM的量化技术或方案

**答：**

| 方法 | 类型 | 特点 |
|-----|------|------|
| **GPTQ** | PTQ | 逐层权重重建，适合离线量化 |
| **AWQ** | PTQ | 保护显著权重，精度友好 |
| **SmoothQuant** | PTQ | 平滑激活异常值，适合 INT8 |
| **LLM.int8()** | PTQ | 分离异常值处理 |
| **FP8/FP4** | 低精度 | 硬件原生支持，动态范围好 |

### 2. SmoothQuant 为什么可以缓解 INT8 LLM 精度下降的问题？

**答：**

**核心问题：** LLM 激活值中存在大量 **Outliers（异常极大值）**，导致朴素 INT8 量化时大部分值被压缩到很小范围，精度损失严重。

**SmoothQuant 思想：**
```
原始: Y = X · W
平滑: Y = (X · s) · (W / s) = X' · W'

其中 s 是 per-channel 的缩放因子，把激活的难度转移到权重上
```

**为什么有效：**
- 权重分布通常比激活更平滑，对量化更友好
- 通过数学等价变换，不改变输出结果
- 让激活和权重都落入适合 INT8 量化的范围

### 3. bfloat16 和 fp16 在相同内存占用下，主要优缺点分别是什么？

**答：**

| 特性 | FP16 | BF16 |
|-----|------|------|
| **指数位** | 5 bit | 8 bit |
| **尾数位** | 10 bit | 7 bit |
| **动态范围** | 较小 (±65504) | 较大 (±3.4e38) |
| **精度** | 较高 | 较低 |

**优缺点：**

**FP16：**
- ✅ 精度更高
- ❌ 容易 overflow/underflow，需要 Loss Scaling

**BF16：**
- ✅ 动态范围与 FP32 相同，数值稳定性好
- ❌ 精度略低，极少数任务难收敛

**使用场景：**
- **训练：** BF16 更稳定，FP16 需配合 Loss Scaling
- **推理：** 两者皆可，BF16 更适合对数值敏感的场景

### 3.5 LLM.int8() 的原理是什么？

**答：**

**核心思想：** 混合精度分解，将 outlier（异常大值）单独用 FP16 处理。

1. 按列检测激活中的 outlier（绝对值 > 阈值，如 6.0）
2. Outlier 列用 FP16 精度计算
3. 非 outlier 列用 INT8 量化计算
4. 两部分结果相加

```
X @ W = X_outlier @ W_outlier (FP16) + X_normal @ W_normal (INT8)
```

- ✅ 几乎零精度损失的 INT8 推理
- ❌ 需要动态检测 outlier，增加实现复杂度；混合精度会降低 Tensor Core 利用率

### 3.6 AWQ（Activation-aware Weight Quantization）的原理？

**答：**

**核心思想：** 不是所有权重都同等重要，保护"重要"权重。

1. 通过校准数据统计每个权重通道对应的激活幅度
2. 激活幅度大的通道 → 对应权重更重要 → 量化时给更大的 scale
3. 等价变换：`W_q = Quantize(W * s) / s`，s 基于激活幅度搜索最优值

- ✅ Weight-only 量化（W4A16），适合极致压缩权重、省显存
- ❌ 只量化权重，激活仍是 FP16

### 3.7 GPTQ 的原理？

**答：**

**核心思想：** 基于二阶信息（Hessian 近似）的逐层权重量化。

1. 逐层处理：对每一层的权重矩阵做量化
2. 逐列量化：量化一列后，用 Hessian 信息补偿其他列的误差
3. 目标：最小化量化后的输出误差 $||WX - \hat{W}X||^2$
4. 利用 OBS（Optimal Brain Surgeon）的近似方法

- ✅ 量化精度高（尤其 4-bit），一次量化多次推理
- ❌ 量化过程较慢（需逐层优化），需要校准数据

### 3.8 量化方案对比

| 方法 | 类型 | 量化对象 | 精度 | 适用场景 |
|------|------|---------|------|---------|
| **LLM.int8()** | PTQ | W8A8（mixed） | 高 | 零损失推理，但速度一般 |
| **SmoothQuant** | PTQ | W8A8 | 高 | INT8 全量化，追求吞吐 |
| **AWQ** | PTQ | W4A16 | 较高 | 极致压缩权重，省显存 |
| **GPTQ** | PTQ | W4A16 / W3A16 | 较高 | 离线量化，追求低 bit 精度 |
| **FP8** | PTQ/QAT | W8A8 (FP8) | 高 | Hopper 原生支持，部署自然 |

### 4. 在实际工程中，量化如何平衡「精度」和「速度/显存」？

**答：**

**评估指标设计：**

| 维度 | 指标 |
|-----|------|
| **精度** | Perplexity、下游任务准确率、人工评估 |
| **速度** | Tokens/s、Latency (TTFT/ITL) |
| **显存** | 峰值显存、并发能力 |
| **成本** | $/1M tokens |

**Trade-off 策略：**

1. **分层量化：** Attention 层用 FP16，FFN 层用 INT8
2. **混合精度：** Weight 用 INT4，Activation 用 FP16
3. **自适应量化：** 根据 layer 敏感度选择精度

**实验设计：**
- 先在小数据集验证精度损失可接受
- 再在全量数据评估端到端性能
- A/B 测试对比用户感知质量

---

## CUDA 与 GPU 优化

### 1. 讲讲 shared memory bank conflict 的发生场景？出现 bank conflict 会有什么后果？

**答：**

**Bank Conflict 发生场景：**

```cuda
// 假设 bank 宽度为 4 bytes，32 个 banks
__shared__ float shared[256];

// 情况 1: 无 conflict - 每个线程访问不同 bank
float x = shared[threadIdx.x];  // stride=1

// 情况 2: 2-way conflict - 相邻线程访问同一 bank 不同地址
float x = shared[threadIdx.x * 2];  // stride=2

// 情况 3: 32-way conflict - 所有线程访问同一 bank
float x = shared[threadIdx.x * 32];  // stride=32
```

**后果：**
- 同一 warp 内访问同一 bank 不同地址时，访问被串行化
- 32-way conflict 会导致 32 倍延迟！

**解决方法：**
1. **改变访问模式：** 使用 stride 避免冲突
2. **Padding：** `shared[256][33]` 而非 `[256][32]`
3. **广播访问：** 所有线程访问同一地址是广播，无 conflict

### 2. CUDA 里面如何分配 GPU 显存？为什么相关 API 里参数是二级指针？

**答：**

**分配方式：**

```cuda
// 方式 1: cudaMalloc - 设备内存
float* d_ptr;
cudaMalloc(&d_ptr, size);  // 二级指针

// 方式 2: cudaMallocHost - 锁页主机内存
float* h_ptr;
cudaMallocHost(&h_ptr, size);

// 方式 3: cudaMallocManaged - 统一内存
float* u_ptr;
cudaMallocManaged(&u_ptr, size);
```

**为什么是二级指针？**

```cpp
// cudaMalloc 内部需要修改调用者的指针值
// 一级指针无法修改调用者的指针（传值）
// 二级指针可以修改指针本身的值（传址）

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    void* ptr = internal_alloc(size);  // 内部分配
    *devPtr = ptr;  // 通过二级指针修改外部指针
    return cudaSuccess;
}
```

### 2.5 GPU 线程层次结构：Grid / Block / Thread / Warp

**答：**

```
Grid（网格）
├── Block 0（线程块 / CTA）
│   ├── Warp 0: Thread 0~31
│   ├── Warp 1: Thread 32~63
│   └── ...
├── Block 1
│   └── ...
└── ...
```

| 概念 | 说明 |
|------|------|
| **Thread** | GPU 最小执行单元，每个 thread 执行相同的 kernel 代码 |
| **Warp** | 32 个连续 thread 组成，**SIMT 执行的基本单位**，warp 内 thread 执行相同指令 |
| **Block（CTA）** | 多个 warp 组成，共享 Shared Memory，在一个 SM 上执行，block 内可同步（`__syncthreads()`） |
| **Grid** | 所有 block 的集合，对应一次 kernel launch |

**高频追问：**
- **Warp divergence**：同一 warp 内 thread 走不同 if/else 分支 → GPU 串行执行两个分支，浪费算力
- **一个 block 最多 1024 个 thread**，实际要根据 register 和 shared memory 用量考虑 occupancy
- **一个 block 完整运行在一个 SM 上**，一个 SM 可以同时运行多个 block

### 2.6 GPU 存储层次结构：L0/L1/L2/Global Memory、SRAM/HBM

**答：**

```
┌────────────────────────────────────────────┐
│              Global Memory (HBM)            │  ← 容量大(80GB)，带宽 3.35TB/s，延迟 ~400 cycles
│  ┌──────────────────────────────────────┐  │
│  │           L2 Cache (SRAM)             │  │  ← 全 GPU 共享，50MB (H100)
│  │  ┌────────────────────────────────┐  │  │
│  │  │  Per-SM: L1 Cache /            │  │  │  ← 每个 SM 私有
│  │  │  Shared Memory (SRAM)          │  │  │  ← 可配置划分，延迟 ~20-30 cycles
│  │  │  ┌──────────────────────────┐  │  │  │
│  │  │  │   Register File          │  │  │  │  ← 每个 thread 私有，延迟 ~1 cycle
│  │  │  │   L0 Instruction Cache   │  │  │  │  ← Warp scheduler 指令缓存
│  │  │  └──────────────────────────┘  │  │  │
│  │  └────────────────────────────────┘  │  │
│  └──────────────────────────────────────┘  │
└────────────────────────────────────────────┘
```

| 存储层级 | 位置 | 容量 | 延迟 | 作用域 |
|---------|------|------|------|--------|
| **Register** | SM 内 | 每 SM 256KB | ~1 cycle | Thread 私有 |
| **L0 Instruction Cache** | SM 内 | 小 | 极低 | Warp scheduler 用 |
| **L1 / Shared Memory** | SM 内（SRAM） | ~192-256KB/SM | ~20-30 cycles | Block 内共享（Shared Memory）或自动缓存（L1） |
| **L2 Cache** | 全芯片（SRAM） | ~50MB (H100) | ~200 cycles | 全 GPU 共享 |
| **Global Memory (HBM)** | 芯片外 | 80GB (H100) | ~400 cycles | 全 GPU + Host 可访问 |

**SRAM vs HBM 核心区别：**
- **SRAM**（L1/Shared/L2）：on-chip，速度极快（数十 TB/s），容量极小（KB~MB）
- **HBM**（Global Memory）：off-chip，速度相对慢（几 TB/s），容量大（数十 GB）
- **FlashAttention 的核心就是让 attention 中间结果留在 SRAM 中，避免写回 HBM**

### 2.7 Tensor Core

**答：**

Tensor Core 是 GPU 中专门用于矩阵乘加（MMA）的硬件单元，每个时钟周期可完成一个小矩阵乘法。

| 架构 | 支持精度 | 说明 |
|------|---------|------|
| Volta (V100) | FP16 | 首次引入 Tensor Core |
| Ampere (A100) | FP16, BF16, TF32, INT8, INT4 | 扩展精度支持 |
| Hopper (H100) | 增加 FP8 | 990 TFLOPS (BF16) vs 67 TFLOPS (FP32 CUDA Core) |
| Blackwell (B200) | FP4, FP6 等 | 更多低精度支持 |

**为什么 Tensor Core 对 decode 帮助有限？** 因为 decode（batch=1）做的是 GEMV（矩阵×向量），规模太小无法喂满 Tensor Core。Tensor Core 在大规模 GEMM（如 prefill）中收益更大。

### 2.8 TMA（Tensor Memory Accelerator）

**答：**

TMA 是 Hopper 架构引入的硬件单元，在 HBM 和 Shared Memory 之间高效搬运多维张量。

**核心能力：**
- 支持多维张量的**异步拷贝**（不阻塞 SM 计算）
- 自动处理地址计算、边界检查、padding
- 支持 multicast（一次读取分发给多个 SM）

```
传统方式：Thread 手动计算地址 → 发起 load → 等待 → SM 大量时间浪费在地址计算和等待上
TMA 方式：TMA 单元接收描述符 → 自动搬运整块数据到 Shared Memory → 计算与搬运完全 overlap
```

**FlashAttention-3 利用 TMA 实现更高效的 tiling，让 SM 专注于计算。**

### 3. 为了优化 CUDA 程序的访存效率，你可以想到哪些手段？

**答：**

| 优化手段 | 说明 |
|---------|------|
| **Coalesced Memory Access** | 确保线程按顺序访问连续地址，合并内存事务 |
| **Shared Memory** | 利用片上高速缓存，减少全局内存访问 |
| **缓存重用** | 设计算法提高数据局部性 |
| **对齐访问** | 确保访问地址对齐到 128 bytes |
| **向量化加载** | 使用 float4/int4 等宽向量指令 |
| **零拷贝** | 使用 Unified Memory 或 Zero Copy |

### 4. 为了优化 CUDA 程序的计算效率，你又可以想到哪些？

**答：**

| 优化手段 | 说明 |
|---------|------|
| **Occupancy** | 提高 SM 占用率，隐藏延迟 |
| **指令级并行 (ILP)** | 一个线程执行多条独立指令 |
| **Kernel Fusion** | 合并小 kernel，减少 launch overhead |
| **减少分支发散** | 避免 warp 内线程走不同分支 |
| **使用 Tensor Core** | 对矩阵运算使用 WMMA/mma.sync |
| **循环展开** | 减少循环控制开销 |

---

## 大模型理论与推理优化

### 1. 说出你知道的典型 encoder-only / decoder-only / encoder-decoder 结构的模型，各自优缺点

**答：**

| 架构 | 代表模型 | Attention 类型 | 典型任务 |
|-----|---------|---------------|---------|
| **Encoder-only** | BERT, RoBERTa, ALBERT, DeBERTa | 双向（Bidirectional）全注意力 | 文本分类、NER、句子相似度 |
| **Decoder-only** | GPT 系列, LLaMA, Qwen, DeepSeek | 因果（Causal）单向注意力 | 文本生成、对话、代码补全 |
| **Encoder-Decoder** | T5, BART, mBART, Flan-T5 | Encoder 双向 + Decoder 因果 + Cross Attention | 翻译、摘要、问答 |

**Encoder-only 优缺点：**
- ✅ 双向上下文建模能力强，适合理解类任务；模型相对较小，推理快
- ❌ 不擅长生成任务（没有自回归机制）；在生成式 AI 浪潮中逐渐被取代

**Decoder-only 优缺点：**
- ✅ 天然适合自回归生成，scaling law 表现最好；涌现能力（ICL, CoT）主要在此架构上观察到
- ❌ 单向注意力在纯理解任务上不如双向模型；长序列 KV Cache 显存线性增长

**Encoder-Decoder 优缺点：**
- ✅ 输入输出可以有不同长度和结构，天然适合 seq2seq 任务；Encoder 做理解、Decoder 做生成，分工明确
- ❌ 架构更复杂，参数效率不如 decoder-only；scaling 方面不如 decoder-only 有优势

**为什么 Decoder-only 成了主流：**
1. Scaling law 在 decoder-only + next-token prediction 上表现最好
2. 统一的自回归范式可以同时处理理解和生成
3. 工程上 KV Cache、continuous batching 等优化都是针对 decoder-only 设计

| 维度 | Encoder-only | Decoder-only | Encoder-Decoder |
|------|-------------|-------------|----------------|
| 生成能力 | 弱 | 强 | 强 |
| 理解能力 | 强 | 中等（大模型后追上来了） | 强 |
| Scaling 表现 | 一般 | 最好 | 一般 |
| KV Cache | 不需要 | 需要 | Encoder KV + Decoder KV |
| 推理优化生态 | 较少 | 最成熟 | 中等 |
| 当前主流地位 | 衰退 | 绝对主流 | 衰退 |

### 2. 随着序列长度增加，encoder-only 和 decoder-only 模型的计算量与访存量变化趋势

**答：**

| 模型类型 | 计算复杂度 | 访存趋势 |
|---------|-----------|---------|
| **Encoder-only** | O(n²) - 一次性算完 | 随序列增长，但无 KV Cache 累积 |
| **Decoder-only** | O(n²) 每步，总 O(n³) | KV Cache 线性增长，带宽瓶颈 |

**Decoder-only 的特殊性：**

```
Step 1: 计算 token 1 的 KV
Step 2: 计算 token 2 的 KV，读取 token 1 的 KV
Step 3: 计算 token 3 的 KV，读取 token 1,2 的 KV
...
Step n: 读取 n-1 个 KV

总计算量: O(n³)
KV Cache: O(n) 存储，但 O(n) 带宽每步
```

### 3. 说说你知道的大模型训练或推理的常见优化手段

**答：**

| 类别 | 方法 |
|-----|------|
| **并行策略** | 流水线并行、张量并行、序列并行 |
| **显存优化** | 量化、KV Cache 优化、Activation Checkpointing |
| **计算优化** | FlashAttention、算子融合、CUDA Graph |
| **调度优化** | Continuous Batching、PD 分离、Speculative Decoding |

### 4. 一般会对哪些大模型中的算子做「算子融合」？举几个典型例子

**答：**

**典型融合场景：**

| 融合算子 | 融合内容 | 收益 |
|---------|---------|------|
| **RMSNorm** | 1/x² → multiply → scale | 减少 kernel launch，中间结果存 SRAM |
| **GeLU/SiLU** | Linear → Activation | 避免写回 HBM |
| **QKV Projection** | 三个 Linear 合并 | 减少启动开销 |
| **Attention** | QK^T → Softmax → V 合并 | FlashAttention 核心 |

### 5. 请讲讲 FlashAttention 的原理？为什么它能极大提升速度并节省显存？

**答：**

**核心思想：IO-Aware 计算**

```
传统 Attention:
HBM: Q, K, V (大矩阵)
  ↓ 加载到 SRAM
SRAM: 计算 S = QK^T
  ↓ 写回 HBM
HBM: S (大矩阵)
  ↓ 加载到 SRAM
SRAM: 计算 P = softmax(S)
  ↓ 写回 HBM
HBM: P (大矩阵)
... 大量 HBM 读写

FlashAttention:
HBM: Q, K, V
  ↓ 分块加载
SRAM: 每次处理小块，在 SRAM 内完成所有计算
      只输出最终结果到 HBM
      使用 Online Softmax 技巧
```

**收益来源：**
1. **减少 HBM 访问** - 从 O(N²d) 降到 O(N²d²/M)（M 为 SRAM 大小），不需要将 N×N 注意力矩阵写回 HBM
2. **节省显存** - 不需要存储中间注意力矩阵
3. **速度提升** - HBM 带宽不再是瓶颈

### 5.5 FlashAttention v1 → v2 → v3 的演进

**答：**

**FlashAttention v2 改进（在 v1 基础上）：**
- **更好的并行化**：外层循环改为遍历 Q blocks（v1 是遍历 K/V blocks），每个 Q block 结果在 SRAM 中累积完后一次性写回，减少 HBM 读写
- **减少非矩阵乘 FLOPs**：优化 softmax 统计量更新，减少 warp 间同步
- **更好的 work partitioning**：在 warp 之间更均匀分配工作
- 速度比 v1 快约 **2x**，达到理论 FLOPS 的 50-73%

**FlashAttention v3 改进（面向 Hopper 架构）：**
- **利用 TMA**：用 Tensor Memory Accelerator 做异步数据搬运，解放 SM
- **Warp Specialization**：不同 warp 分别负责数据加载和计算，流水线化
- **FP8 支持**：原生支持 FP8 精度
- **Ping-pong scheduling**：双缓冲调度，让 TMA 加载和 WGMMA 计算完全重叠
- 在 H100 上比 v2 快约 **1.5-2x**

| 维度 | FA v1 | FA v2 | FA v3 |
|------|-------|-------|-------|
| 核心思想 | Tiling + Online Softmax | 更好的并行化 | Hopper 硬件特性利用 |
| 目标硬件 | A100 | A100/H100 | H100 (Hopper) |
| 数据搬运 | Thread 手动加载 | Thread 手动加载 | TMA 异步搬运 |
| FP8 支持 | 无 | 无 | 有 |
| 相对速度 | 1x | ~2x | ~3-4x |

**总结：v1→v2 是算法层面改进（更好的并行化），v2→v3 是硬件层面利用（TMA、Warp Specialization）。**

### 5.6 什么是 FlashDecoding？

**答：**

**问题：** FlashAttention 并行维度是 batch × head × Q_block。Decode 阶段 Q 只有 1 个 token，如果 batch 小、head 少，并行度不足以喂满 GPU。

**FlashDecoding 做法：** 在 **KV sequence 维度** 上增加并行。

```
传统 decode attention: 一个 thread block 串行遍历整个 KV 序列

FlashDecoding: 将 K/V 序列切成多个分片 → 每个分片由不同 thread block 并行处理
→ 每个分片独立计算 partial softmax 和 partial output
→ 最后一次 reduction 合并（用 online softmax 技巧保证数值正确）
```

- 长序列 decode 速度提升 **8x+**
- 特别适合 long context + small batch 场景

### 6. 请讲讲 PagedAttention 的原理？为什么它能极大提升推理速度？与 FlashAttention 的本质区别？

**答：**

**PagedAttention 核心：**

```
传统 KV Cache 分配:
请求1: [████████████████] 预分配 2048
请求2: [████████████████] 预分配 2048
       实际只用 512      实际只用 1024
       浪费 75%          浪费 50%

PagedAttention:
┌─────────────────────────────────────┐
│ 请求1: [Block0][Block1][空闲]        │
│ 请求2: [Block3][空闲][Block4]        │
│ 物理:  [B0][B3][B1][B4][空闲]        │
└─────────────────────────────────────┘
按需分配 Block，消除碎片
```

**提升速度的原因：**
1. **更高显存利用率** - 可以服务更多并发请求
2. **支持 Continuous Batching** - 动态插入新请求
3. **减少显存碎片** - 高效利用每一字节

**与 FlashAttention 的区别：**

| 特性 | FlashAttention | PagedAttention |
|-----|----------------|----------------|
| **优化目标** | 单步 Attention 计算效率 | KV Cache 显存管理 |
| **技术层次** | Kernel 层优化 | 系统/调度层优化 |
| **解决的问题** | Attention 计算中的 HBM 瓶颈 | KV Cache 分配碎片 |
| **关系** | 互补，可叠加使用 | 互补，可叠加使用 |

---

## GPU 硬件与编程模型补充

### H100 相比 A100 有哪些改进

**答：**

| 维度 | A100 (Ampere) | H100 (Hopper) | 改进倍数 |
|------|--------------|---------------|---------|
| **Tensor Core** | 第 3 代（FP16/BF16/TF32/INT8） | 第 4 代（新增 **FP8**） | ~3x FLOPS |
| **BF16 峰值算力** | 312 TFLOPS | 990 TFLOPS | ~3.2x |
| **FP8 峰值算力** | 不支持 | 1979 TFLOPS | 全新 |
| **NVLink** | 第 3 代，600 GB/s | 第 4 代，**900 GB/s** | 1.5x |
| **HBM** | HBM2e，2.0 TB/s，80GB | HBM3，**3.35 TB/s**，80GB | 1.67x |
| **L2 Cache** | 40 MB | **50 MB** | 1.25x |
| **SM 数量** | 108 | 132 | 1.22x |

**Hopper 架构关键新特性：**

1. **Transformer Engine**：硬件级自动 FP8 混合精度，自动选择 FP8/FP16 粒度
2. **Thread Block Clusters**：多个 Block 可组成 Cluster，共享跨 SM 的数据（Distributed Shared Memory）
3. **TMA（Tensor Memory Accelerator）**：硬件异步搬运多维张量，解放 SM 做计算
4. **DPX 指令**：加速动态规划类算法
5. **Confidential Computing**：硬件级 AI 安全保护

**面试重点：** H100 对 LLM 最重要的改进是 FP8 Tensor Core + TMA + 更高 NVLink 带宽，使得 FlashAttention-3 和 FP8 训练成为可能。

### Triton 和 CUDA 区别

**答：**

| 维度 | CUDA | Triton |
|------|------|--------|
| **开发者** | NVIDIA | OpenAI |
| **编程语言** | C/C++ + CUDA 扩展 | **Python** |
| **抽象级别** | 低（手动管理线程、共享内存、同步） | 高（自动 tiling、自动共享内存管理） |
| **控制粒度** | Thread 级别 | **Block (tile) 级别** |
| **内存管理** | 手动（shared memory load/store） | 自动（编译器优化 tiling 和缓存） |
| **编译目标** | PTX → SASS | MLIR → PTX → SASS |
| **学习曲线** | 陡峭 | 平缓 |
| **性能上限** | 最高（手工极致优化） | 接近 CUDA（通常 80-95%） |
| **调试难度** | 高 | 中等 |
| **生态** | 最成熟，文档/工具丰富 | 快速发展，PyTorch 生态集成 |

**Triton 核心思想：**
```python
# Triton: 以 block/tile 为粒度编程
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)  # 获取 block ID
    offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 自动向量化
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)  # 自动合并访存
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
```

**选择建议：**
- **写自定义 kernel（Flash Attention 变体、融合算子）**：优先 Triton（开发效率高）
- **极致性能优化（最后 5-10%）**：CUDA
- **需要 Warp 级控制（warp shuffle、inline PTX）**：只能 CUDA

### BLOCK SIZE 选择原则

**答：**

BLOCK SIZE（每个 Block 的线程数）直接影响 GPU 利用率，通常选择 **64、128 或 256**。

**权衡因素：**

| 因素 | BLOCK SIZE 过小 | BLOCK SIZE 过大 |
|------|----------------|----------------|
| **Occupancy** | Wave 内 block 多，但每 block 线程少 | 每 block 用 register/shared memory 多，SM 能放的 block 少 |
| **Register 压力** | 低 | 高（可能溢出到 local memory） |
| **Shared Memory** | 用量少 | 用量大，限制并发 block 数 |
| **Warp 数量** | 少，难以隐藏延迟 | 多，更好隐藏 memory latency |
| **计算粒度** | Tile 小，数据复用低 | Tile 大，数据复用好 |

**实践指南：**

```
1. 确保 BLOCK SIZE 是 Warp 大小(32) 的倍数
2. 查看 SM 资源限制：
   - Register: 每 SM 65536 个，每 thread 使用 N 个 → 最多 65536/N 个 thread
   - Shared Memory: 每 SM 48-164KB
3. 使用 CUDA Occupancy Calculator 计算最优值

常见选择：
- GEMM kernel: 128 或 256（需要大 tile 做数据复用）
- Elementwise kernel: 256 或 512（简单计算，追求高占用率）
- Reduction kernel: 256（平衡并行度和同步开销）
```

### CUDA Tile 编程模型

**答：**

Tiling 是 GPU 编程的核心思想：将大规模计算分解为适合 SRAM 的小块（tile），减少对全局内存的访问。

**基本模式：**

```
全局内存 (HBM)          共享内存 (SRAM)          寄存器
┌──────────┐           ┌──────────┐           ┌──────┐
│ A[M×K]   │ ──load──→ │ As[T×T]  │ ──load──→ │ 计算  │
│ B[K×N]   │           │ Bs[T×T]  │           │ 结果  │
│ C[M×N]   │ ←─store── │ Cs[T×T]  │ ←─store── │      │
└──────────┘           └──────────┘           └──────┘
```

**GEMM Tiling 经典实现（C = A × B）：**

```cuda
// 每个 Block 计算 C 的一个 TILE_M × TILE_N 子块
__global__ void gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];  // A 的 tile 缓存
    __shared__ float Bs[TILE_K][TILE_N];  // B 的 tile 缓存
    float acc = 0.0f;  // 寄存器累加

    for (int t = 0; t < K; t += TILE_K) {
        // Step 1: 从全局内存加载 tile 到共享内存
        As[ty][tx] = A[row * K + (t + tx)];
        Bs[ty][tx] = B[(t + ty) * N + col];
        __syncthreads();  // 等待所有线程加载完成

        // Step 2: 在共享内存中计算（数据复用）
        for (int k = 0; k < TILE_K; k++)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();  // 等待计算完成再加载下一个 tile
    }

    // Step 3: 写回全局内存
    C[row * N + col] = acc;
}
```

**为什么 Tiling 有效：**
- 全局内存访问 A 的同一行被 TILE_N 个线程复用 → 减少 TILE_N 倍访存
- 共享内存带宽 ~19 TB/s vs HBM ~3.35 TB/s → 快 ~5-6 倍
- FlashAttention 本质就是 Attention 的 Tiling 实现

**高级 Tiling 技巧：**
1. **双缓冲（Double Buffering）**：加载下一个 tile 同时计算当前 tile
2. **寄存器分块（Register Tiling）**：每个线程计算多个输出元素
3. **Warp 级 Tiling**：利用 Tensor Core 的 wmma 指令

### 介绍下 Ray 框架

**答：**

**Ray** 是 UC Berkeley RISELab（现 Anyscale 公司）开发的分布式计算框架，核心特点是**统一的分布式 API**。

**核心组件：**

| 组件 | 功能 | 典型用途 |
|------|------|---------|
| **Ray Core** | Task（无状态函数）+ Actor（有状态对象）的分布式抽象 | 通用分布式计算 |
| **Ray Train** | 分布式训练 | 多 GPU/多节点训练 |
| **Ray Serve** | 模型推理服务 | 在线 serving |
| **Ray Data** | 分布式数据处理 | 数据预处理流水线 |
| **RLlib** | 分布式强化学习 | RL 训练 |

**为什么在 LLM 领域重要：**

```
传统 RL 训练：
手动管理 Actor/Critic/Reward/Reference 4 个模型的分布式调度 → 工程量巨大

Ray 方案（如 OpenRLHF）：
@ray.remote(num_gpus=4)
class ActorWorker:
    def generate(self, prompts): ...
    def train_step(self, batch): ...

@ray.remote(num_gpus=2)
class RewardWorker:
    def score(self, responses): ...

# Ray 自动处理调度、通信、容错
futures = [actor.generate.remote(batch) for batch in batches]
results = ray.get(futures)
```

**Ray 的核心优势：**
1. **动态资源调度**：按需分配 GPU，训练和推理资源可动态切换
2. **容错恢复**：Worker 失败自动重启
3. **灵活编排**：不同模型可以用不同并行策略
4. **Python 原生**：API 简洁，学习成本低
5. **生态丰富**：与 PyTorch、vLLM、DeepSpeed 等集成

**在 RLHF 中的应用：** OpenRLHF、veRL 等框架使用 Ray 编排 PPO/GRPO 训练中的多模型协作，Actor 生成、Reward 打分、Critic 评估可以在不同 GPU 组上并行执行。

---

## 面试金句

> "FlashAttention 解决的是 Attention 计算中的 HBM 瓶颈，通过分块计算减少显存访问；PagedAttention 解决的是 KV Cache 显存分配碎片问题，通过虚拟内存分页机制提高利用率。两者互补，可叠加使用。"

> "SmoothQuant 的核心思想是通过数学等价变换，将激活中的异常值平滑到权重上，让两者都落入适合 INT8 量化的范围。"

> "BF16 和 FP16 内存占用相同，但 BF16 有更大的动态范围（与 FP32 相同），数值稳定性更好，训练时不需要 Loss Scaling。"

> "H100 相比 A100 的核心提升：FP8 Tensor Core（算力 2x）、第 4 代 NVLink（900GB/s）、HBM3（3.35TB/s）、TMA 异步搬运，使 FlashAttention-3 和 FP8 训练成为可能。"

> "Triton 是 Python 级的 GPU 编程语言，以 Block/Tile 为粒度编程，自动处理 tiling 和内存管理，开发效率远高于 CUDA，性能可达 CUDA 的 80-95%。"

> "CUDA Tile 编程的核心：将大计算分解为适合 SRAM 的小块，从全局内存加载 → 共享内存计算 → 写回全局内存，通过数据复用减少访存次数。"
