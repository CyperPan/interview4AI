# 手撕代码

> 文档定位：给出面试里常见的“能手写出来”的标准版本代码，并解释每段代码为什么这么写。这里优先追求 **正确、易讲、能在白板或在线 IDE 中复现**，而不是追求生产环境里的极致性能。

## 目录

- [注意力机制实现](#注意力机制实现)
- [CUDA 基础](#cuda-基础)
- [C++ 高并发](#c-高并发)
- [PagedAttention 内存分配器](#pagedattention-内存分配器)

---

## 注意力机制实现

### 手撕 Multi-Head Attention (MHA) 或 Grouped-Query Attention (GQA)

**题目：** 请用纯 NumPy 或 PyTorch 实现一个标准的缩放点积注意力（Scaled Dot-Product Attention），并支持 Causal Mask（因果掩码）。

**考察点：**
- 矩阵维度的变换（Transpose/Reshape）
- Mask 的应用位置
- Softmax 前为什么要除以 √dk（防止梯度消失/Softmax 溢出）
- GQA 中 KV 的广播（Broadcast）处理

**参考实现：**

```python
import torch
import torch.nn as nn
import math

def make_causal_mask(seq_len, device):
    # True 表示当前位置可见，False 表示需要屏蔽
    return torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, n_heads, seq_len, d_k)
        mask: (batch, 1, seq_len, seq_len) 或 (1, 1, seq_len, seq_len)
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        if mask is None:
            mask = make_causal_mask(seq_len, x.device)

        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_o(attn_output)


# GQA 实现：多个 Query 头共享一组 KV 头
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # 每个 KV 头被重复的次数
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k)  # 更少的 KV
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
    
    def repeat_kv(self, x, n_rep):
        """复制 KV 头以匹配 Query 头数量"""
        batch, n_kv_heads, seq_len, d_k = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(
            batch, n_kv_heads, n_rep, seq_len, d_k
        ).reshape(batch, n_kv_heads * n_rep, seq_len, d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        if mask is None:
            mask = make_causal_mask(seq_len, x.device)

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        K = self.repeat_kv(K, self.n_rep)
        V = self.repeat_kv(V, self.n_rep)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.W_o(output)
```

**代码讲解：**

1. 先把输入 `x` 线性投影成 `Q / K / V`，再 reshape 成多头格式 `(batch, heads, seq_len, head_dim)`。
2. 注意力分数的计算是 `Q @ K^T / sqrt(dk)`，这里除以 `sqrt(dk)` 是为了避免点积值过大，导致 Softmax 饱和。
3. 因果掩码必须在 **Softmax 之前** 应用，把未来位置写成 `-inf`，这样 Softmax 后对应概率才会变成 0。
4. MHA 的每个 Q 头都有独立的 K/V 头；GQA 则让多个 Q 头共享一组 K/V 头，所以 `K / V` 投影出来后要重复到和 Q 头数一致。
5. 面试里如果被追问，重点讲清楚“维度变化、Mask 位置、为什么 GQA 更省 KV Cache”这三点。

---

## CUDA 基础

### 1. 手撕 CUDA 并行归约（Parallel Reduction）

**题目：** 写一个 CUDA Kernel，求一个长度为 N 的数组的最大值或总和。

**考察点：**
- Shared Memory 中的树状规约（Tree-based reduction）
- `__syncthreads()` 同步
- Warp-level primitives 优化（`__shfl_down_sync`）

**参考实现：**

```cuda
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// 基础版：每个 block 处理 2 * BLOCK_SIZE 个元素
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    float sum = 0.0f;
    if (gid < n) {
        sum += input[gid];
    }
    if (gid + blockDim.x < n) {
        sum += input[gid + blockDim.x];
    }

    shared[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 优化版：先做 grid-stride 累加，再做 warp-level 归约
__global__ void reduce_sum_warp_kernel(const float* input, float* output, int n) {
    __shared__ float warp_sums[BLOCK_SIZE / 32];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    float val = 0.0f;
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        val += input[i];
    }

    val = warp_reduce_sum(val);

    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < BLOCK_SIZE / 32) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            output[blockIdx.x] = block_sum;
        }
    }
}
```

**代码讲解：**

1. 基础版先把每个线程负责的数据累加到寄存器，再写进 shared memory，接着做标准树状规约。
2. `__syncthreads()` 不能省，因为每一轮 stride 归约都依赖上一轮 shared memory 的结果。
3. 优化版先让每个线程做 `grid-stride loop`，减少全局访存发射次数，再用 `__shfl_down_sync` 做 warp 内规约。
4. 原理上是“两级规约”：先每个 warp 得到一个局部和，再让第 0 个 warp 把所有 warp 的结果规约成 block 结果。
5. 面试里要主动指出：`warp-level primitive` 主要是为了减少 shared memory 读写和同步开销。

### 2. 手撕数值稳定的 Softmax

**题目：** 用 C++ 写一个 Softmax 函数。

**踩坑警告：** 绝对不能直接写 `exp(xi) / sum(exp(xi))`，会指数溢出（Overflow）！

**满分答案：**

```cpp
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    
    // 1. 找最大值（数值稳定性关键）
    float max_val = *std::max_element(input.begin(), input.end());
    
    // 2. 计算 exp(xi - max_val)，防止溢出
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // 3. 归一化
    for (size_t i = 0; i < output.size(); ++i) {
        output[i] /= sum;
    }
    
    return output;
}
```

**代码讲解：**

1. Softmax 最大的坑是数值溢出，所以第一步必须先找到最大值 `max_val`。
2. 把每个元素改写成 `exp(x_i - max_val)`，不会改变最终结果，因为分子分母同时乘了同一个常数。
3. 第一轮循环算指数和总和，第二轮循环再做归一化，逻辑最清楚，也适合白板手写。
4. 如果面试官继续追问，可以补一句：LogSumExp 技巧和 FlashAttention 在线 Softmax 的数值稳定思路是一脉相承的。

### 3. 手撕 RMSNorm

**题目：** 简历写了优化 RMSNorm，请用代码展示 RMSNorm 的计算公式。

**考察点：**
- 计算公式
- CUDA 中计算 variance 时内存读取优化

```cpp
#include <vector>
#include <cmath>
#include <stdexcept>

struct RMSNorm {
    explicit RMSNorm(size_t hidden_size, float eps_ = 1e-6f)
        : eps(eps_), weight(hidden_size, 1.0f) {}

    std::vector<float> forward(const std::vector<float>& x) const {
        if (x.size() != weight.size()) {
            throw std::invalid_argument("input size must match weight size");
        }

        float sum_squares = 0.0f;
        for (float val : x) {
            sum_squares += val * val;
        }

        const float mean_square = sum_squares / static_cast<float>(x.size());
        const float inv_rms = 1.0f / std::sqrt(mean_square + eps);

        std::vector<float> output(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            output[i] = x[i] * inv_rms * weight[i];
        }

        return output;
    }
 
    float eps;
    std::vector<float> weight;
};
```

**代码讲解：**

1. RMSNorm 不减去均值，只计算均方根，所以它比 LayerNorm 少了一步“减 mean”。
2. 先算 `mean(x^2)`，再算 `1 / sqrt(mean(x^2) + eps)`，最后乘上可学习参数 `weight`。
3. 这里把 `inv_rms` 提前算出来，是为了避免循环里重复做除法。
4. 面试里可以顺手补一句：RMSNorm 在实现上更简单，访存和计算都比 LayerNorm 更轻一些，所以常出现在 LLM 中。

---

## C++ 高并发

### 1. 手撕无锁环形缓冲区（Lock-free Ring Buffer / SPSC Queue）

**题目：** 用 C++11 的 `std::atomic` 实现一个单生产者单消费者（SPSC）的无锁队列。

**考察点：**
- `std::atomic` 使用
- Memory Order（`memory_order_acquire` / `memory_order_release`）
- 防止 CPU 指令重排

**参考实现：**

```cpp
#include <atomic>
#include <vector>

template<typename T>
class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t capacity)
        : capacity_(capacity + 1), buffer_(capacity_) {}

    bool push(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) % capacity_;

        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;
        }

        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);

        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false;
        }

        item = buffer_[current_head];
        head_.store((current_head + 1) % capacity_, std::memory_order_release);
        return true;
    }

private:
    size_t capacity_;
    std::vector<T> buffer_;

    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};
```

**Memory Order 解释：**

```cpp
// Acquire：在这个操作之后的读写操作不能被重排到它之前
// 消费者读取 tail_：确保看到生产者已经写好的数据
tail_.load(std::memory_order_acquire);

// Release：在这个操作之前的读写操作不能被重排到它之后
// 生产者发布 tail_：确保 buffer_ 写入先于 tail_ 更新对外可见
tail_.store(next_tail, std::memory_order_release);
```

**代码讲解：**

1. 这是单生产者单消费者队列，所以只有生产者写 `tail_`，只有消费者写 `head_`，实现上比 MPMC 简单得多。
2. 这里故意把底层数组开成 `capacity + 1`，留出一个空槽位，用来区分“队列满”和“队列空”。
3. `push` 里先写数据，再用 `release` 发布新的 `tail_`；`pop` 里先用 `acquire` 看到最新 `tail_`，再读数据。
4. 如果面试官追问为什么不能全用 `memory_order_relaxed`，回答重点是：那样可能会出现“索引更新已可见，但数据本体还没对另一线程可见”的重排问题。

### 2. 手撕线程池（Thread Pool）

**题目：** 用 C++ `std::thread`, `std::mutex`, `std::condition_variable` 实现一个包含固定数量 Worker 的线程池。

**考察点：**
- 生产者-消费者模型
- 虚假唤醒（Spurious Wakeup）的处理
- `cv.wait` 必须搭配 lambda

**参考实现：**

```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>
#include <stdexcept>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);

                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty()) {
                            return;
                        }

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }

        cv_.notify_all();

        for (auto& worker : workers_) {
            worker.join();
        }
    }

    template<typename F>
    void submit(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool is stopped");
            }
            tasks_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_;
};
```

**代码讲解：**

1. 整体就是标准生产者-消费者模型：主线程提交任务，worker 线程阻塞等待并消费任务。
2. `cv_.wait(lock, predicate)` 比手写 `while` 更适合面试表达，因为它把“防虚假唤醒”的条件写得更清楚。
3. 任务一定要在锁外执行，否则一个慢任务会把整个线程池的任务队列都堵住。
4. 析构时先把 `stop_` 置为 `true`，再 `notify_all()` 唤醒所有 worker，让它们自行退出。

---

## PagedAttention 内存分配器

### 现场模拟 KV-Cache 或 PagedAttention 的分配逻辑

**题目：** 给定一个 GPU 显存池（表示为一个大数组）和一个 BlockSize，请用 C++ 实现一个简化版的 BlockTable 内存分配器。要求支持：
1. 新请求进来时分配 Block
2. 请求生成结束时释放 Block
3. 显存不足时触发 OOM 拒绝

**考察点：**
- 操作系统虚拟内存映射机制的理解
- 链表或位图（Bitmap）管理空闲内存

**参考实现：**

```cpp
#include <vector>
#include <list>
#include <unordered_map>
#include <cstdint>
#include <memory>
#include <stdexcept>

class PagedAttentionAllocator {
public:
    struct Block {
        int block_id;
        bool allocated;
    };
    
    struct Request {
        int request_id;
        std::vector<int> block_table;  // 逻辑到物理的映射
    };

    PagedAttentionAllocator(int num_blocks, int block_size) 
        : num_blocks_(num_blocks), block_size_(block_size) {
        // 初始化所有 block 为空闲
        for (int i = 0; i < num_blocks; ++i) {
            blocks_.push_back({i, false});
            free_blocks_.push_back(i);
        }
    }
    
    // 为请求分配初始 block
    bool allocate_request(int request_id, int num_initial_blocks) {
        if (free_blocks_.size() < num_initial_blocks) {
            return false;  // OOM
        }
        
        Request req;
        req.request_id = request_id;
        
        for (int i = 0; i < num_initial_blocks; ++i) {
            int block_id = free_blocks_.front();
            free_blocks_.pop_front();
            
            blocks_[block_id].allocated = true;
            req.block_table.push_back(block_id);
        }
        
        requests_[request_id] = std::move(req);
        return true;
    }
    
    // 为已有请求追加 block（Token 生成过程中）
    bool append_block(int request_id) {
        if (free_blocks_.empty()) {
            return false;  // OOM，需要等待或抢占
        }
        
        auto it = requests_.find(request_id);
        if (it == requests_.end()) {
            return false;
        }
        
        int block_id = free_blocks_.front();
        free_blocks_.pop_front();
        blocks_[block_id].allocated = true;
        
        it->second.block_table.push_back(block_id);
        return true;
    }
    
    // 释放请求的所有 block
    void free_request(int request_id) {
        auto it = requests_.find(request_id);
        if (it == requests_.end()) {
            return;
        }
        
        for (int block_id : it->second.block_table) {
            blocks_[block_id].allocated = false;
            free_blocks_.push_back(block_id);
        }
        
        requests_.erase(it);
    }
    
    // 获取 block 在物理显存中的位置
    void* get_block_ptr(int block_id, void* base_ptr) {
        return static_cast<char*>(base_ptr) + block_id * block_size_;
    }
    
    // 查询请求的逻辑到物理映射
    const std::vector<int>& get_block_table(int request_id) const {
        auto it = requests_.find(request_id);
        if (it == requests_.end()) {
            throw std::out_of_range("request_id not found");
        }
        return it->second.block_table;
    }

private:
    int num_blocks_;
    int block_size_;
    
    std::vector<Block> blocks_;
    std::list<int> free_blocks_;  // 空闲 block 链表
    std::unordered_map<int, Request> requests_;
};
```

**代码讲解：**

1. `free_blocks_` 维护所有空闲物理 block；`block_table` 记录某个请求的“逻辑块 -> 物理块”映射。
2. 新请求进来时，从空闲链表里取 block；请求结束时，再把这些 block 放回空闲链表。
3. 这和操作系统分页的思路一致：逻辑上连续，不代表物理上必须连续。
4. 面试里被追问时，可以继续讲：真实系统还会有 `block 共享、引用计数、抢占和换出`，这里只是最小可讲清楚的版本。

---

## 面试金句

> "Softmax 必须先遍历一遍数组找到最大值 max_val，然后计算 `exp(xi - max_val)`，再求和归一化。这是 FlashAttention 的底层算术基础。"

> "无锁队列必须用 `memory_order_acquire` 在 load 时，`memory_order_release` 在 store 时，防止 CPU 指令重排。"

> "线程池的 `cv.wait` 必须搭配 lambda 表达式检查队列是否为空，这是为了处理虚假唤醒（Spurious Wakeup）。"
