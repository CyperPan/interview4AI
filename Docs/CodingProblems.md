# 手撕代码

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
        mask: (batch, 1, seq_len, seq_len) 或 None
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: (batch, n_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head: (batch, n_heads, seq_len, d_k)
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
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        # 复制 KV 头
        K = self.repeat_kv(K, self.n_rep)
        V = self.repeat_kv(V, self.n_rep)
        
        # 后续与 MHA 相同...
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.W_o(output)
```

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

// 基础版：使用 Shared Memory
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    __shared__ float shared[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到 Shared Memory
    shared[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();
    
    // 树状归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    
    // 写回结果
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

// 优化版：使用 Warp-level primitives
__global__ void reduce_sum_warp_kernel(const float* input, float* output, int n) {
    __shared__ float shared[32];  // 只需要 32 个，一个 warp
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (gid < n) ? input[gid] : 0.0f;
    
    // Warp 内归约
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    
    // 每个 warp 的第一个线程写结果
    if (tid % 32 == 0) {
        shared[tid / 32] = val;
    }
    __syncthreads();
    
    // 最后 32 个值的归约
    if (tid < 32) {
        val = shared[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }
    
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}
```

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

### 3. 手撕 RMSNorm

**题目：** 简历写了优化 RMSNorm，请用代码展示 RMSNorm 的计算公式。

**考察点：**
- 计算公式
- CUDA 中计算 variance 时内存读取优化

```cpp
// C++ 版本
struct RMSNorm {
    float eps;
    std::vector<float> weight;
    
    std::vector<float> forward(const std::vector<float>& x) {
        // 1. 计算 RMS (Root Mean Square)
        float sum_squares = 0.0f;
        for (float val : x) {
            sum_squares += val * val;
        }
        float rms = std::sqrt(sum_squares / x.size() + eps);
        
        // 2. 归一化并缩放
        std::vector<float> output(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            output[i] = (x[i] / rms) * weight[i];
        }
        
        return output;
    }
};
```

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
#include <optional>

template<typename T>
class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t capacity) 
        : capacity_(capacity), buffer_(capacity) {}
    
    // 生产者调用（单线程）
    bool push(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) % capacity_;
        
        // 检查队列是否满
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;  // 队列满
        }
        
        buffer_[current_tail] = item;
        
        // Release：确保先写入数据，再更新 tail
        // 防止编译器重排：保证 buffer_ 写入在 tail_ 更新之前可见
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    // 消费者调用（单线程）
    std::optional<T> pop() {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        // 检查队列是否空
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return std::nullopt;  // 队列空
        }
        
        T item = buffer_[current_head];
        const size_t next_head = (current_head + 1) % capacity_;
        
        // Release：确保先读取数据，再更新 head
        head_.store(next_head, std::memory_order_release);
        return item;
    }

private:
    size_t capacity_;
    std::vector<T> buffer_;
    
    // head_ 和 tail_ 用原子变量
    // 用 alignas 避免 false sharing
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};
```

**Memory Order 解释：**

```cpp
// Acquire：在这个操作之后的读写操作不能被重排到它之前
// 消费者用：确保看到生产者写入的数据
head_.load(std::memory_order_acquire);

// Release：在这个操作之前的读写操作不能被重排到它之后  
// 生产者用：确保数据写入对消费者可见
tail_.store(next_tail, std::memory_order_release);
```

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
#include <atomic>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop_(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        
                        // 等待条件：队列非空 或 线程池停止
                        // 必须用 while 防止虚假唤醒！
                        cv_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });
                        
                        // 线程池停止且队列为空，退出
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    
                    // 执行任务（在锁外执行）
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
        
        cv_.notify_all();  // 唤醒所有线程
        
        for (auto& worker : workers_) {
            worker.join();
        }
    }
    
    // 提交任务
    template<typename F>
    void submit(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool is stopped");
            }
            tasks_.emplace(std::forward<F>(f));
        }
        cv_.notify_one();  // 唤醒一个线程
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_;
};
```

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
    const std::vector<int>& get_block_table(int request_id) {
        return requests_[request_id].block_table;
    }

private:
    int num_blocks_;
    int block_size_;
    
    std::vector<Block> blocks_;
    std::list<int> free_blocks_;  // 空闲 block 链表
    std::unordered_map<int, Request> requests_;
};
```

---

## 面试金句

> "Softmax 必须先遍历一遍数组找到最大值 max_val，然后计算 `exp(xi - max_val)`，再求和归一化。这是 FlashAttention 的底层算术基础。"

> "无锁队列必须用 `memory_order_acquire` 在 load 时，`memory_order_release` 在 store 时，防止 CPU 指令重排。"

> "线程池的 `cv.wait` 必须搭配 lambda 表达式检查队列是否为空，这是为了处理虚假唤醒（Spurious Wakeup）。"
