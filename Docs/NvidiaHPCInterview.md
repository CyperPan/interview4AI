# NVIDIA / HPC 岗面经（牛客网汇总）

> 来源：牛客网 AI深度学习推理加速HPC类岗位-Nvidia面经

---

## 目录

- [C++ 基础与工程](#c-基础与工程)
- [量化 Quantization](#量化-quantization)
- [CUDA 与 GPU 优化](#cuda-与-gpu-优化)
- [大模型理论与推理优化](#大模型理论与推理优化)

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

### 1. 说出你知道的典型 encoder-only / decoder-only / encoder-decoder 结构的模型

**答：**

| 架构 | 代表模型 | 特点 |
|-----|---------|------|
| **Encoder-only** | BERT、RoBERTa | 双向注意力，适合理解任务 |
| **Decoder-only** | GPT、LLaMA、Qwen | 自回归生成，适合生成任务 |
| **Encoder-Decoder** | T5、BART | 编码器+解码器，适合翻译/摘要 |

**LLM Serving 主要是 Decoder-only：**
- 自回归生成是逐 token 的
- 需要 KV Cache 优化
- Continuous Batching 等优化都是针对 Decoder

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
1. **减少 HBM 访问** - 从 O(N²) 降到 O(N)
2. **节省显存** - 不需要存储中间注意力矩阵
3. **速度提升** - HBM 带宽不再是瓶颈

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

## 面试金句

> "FlashAttention 解决的是 Attention 计算中的 HBM 瓶颈，通过分块计算减少显存访问；PagedAttention 解决的是 KV Cache 显存分配碎片问题，通过虚拟内存分页机制提高利用率。两者互补，可叠加使用。"

> "SmoothQuant 的核心思想是通过数学等价变换，将激活中的异常值平滑到权重上，让两者都落入适合 INT8 量化的范围。"

> "BF16 和 FP16 内存占用相同，但 BF16 有更大的动态范围（与 FP32 相同），数值稳定性更好，训练时不需要 Loss Scaling。"
