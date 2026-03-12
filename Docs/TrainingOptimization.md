# 训练效率优化

## 目录

- [影响吞吐的主要因素](#影响吞吐的主要因素)
- [GPU 利用率排查](#gpu-利用率排查)
- [梯度累积](#梯度累积)
- [吞吐与收敛的矛盾](#吞吐与收敛的矛盾)

---

## 影响吞吐的主要因素

### 影响大模型训练吞吐的主要因素有哪些？

**答：**

| 因素 | 说明 | 排查工具 |
|-----|------|---------|
| **计算限制** | 算子未优化，如没用 FlashAttention | Nsight Compute |
| **显存带宽限制** | Memory-bound，频繁读写 HBM | Nsight Systems |
| **通信瓶颈** | NCCL All-Reduce 慢 | Nsight Systems (NCCL 时间占比) |
| **数据 I/O 瓶颈** | Dataloader 跟不上 GPU 速度 | top, iostat |

---

## GPU 利用率排查

### GPU 利用率低通常可能有哪些原因？你会怎么排查？

**答：**

**排查优先级：**

1. **优先看 CPU/IO 是否成为瓶颈**
   - Dataloader 慢导致 GPU 空等
   - 数据预处理在 CPU 上成为瓶颈

2. **看 Batch Size 是否太小**
   - 导致算力没跑满
   - 检查 GPU-Util 和功耗

3. **看网络通信占比**
   - 用 Nsight 看 NCCL 同步时间
   - 通信和计算是否有效 Overlap

### 如果训练过程中 GPU 经常"吃不满"，你会优先检查哪些指标？

**答：**

| 优先级 | 指标 | 检查方法 |
|-------|------|---------|
| 1 | Volatile GPU-Util | `nvidia-smi` |
| 2 | 功耗 | `nvidia-smi` |
| 3 | SM 利用率 | Nsight Systems |
| 4 | Kernel 耗时 vs NCCL 耗时 | Nsight Systems |
| 5 | DataLoader Wait Time | PyTorch Profiler |

### 如何提升训练中的资源利用率？

**答：**

- 扩大 Micro-batch Size
- 使用梯度累积（Gradient Accumulation）
- 开启混合精度（BF16/FP16），开启 Tensor Core 加速
- 使用融合算子（Kernel Fusion，如 RMSNorm）
- 优化通信拓扑，提升 Overlap 效率

---

## 梯度累积

### 什么是梯度累积？它的好处和代价分别是什么？

**答：**

**原理：** 连续进行 N 次前向+反向传播，累加梯度但不更新权重，最后才执行一次 `optimizer.step()`。

```python
# 伪代码
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()  # 累加梯度
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # 更新权重
        optimizer.zero_grad()
```

| 方面 | 说明 |
|-----|------|
| **好处** | 显存不变的情况下等效扩大了全局 Batch Size |
| **代价** | 1. 缓存微批次的激活值会占用少许额外显存<br>2. 不能减少总的计算量 |

---

## 吞吐与收敛的矛盾

### 训练吞吐提升和收敛效果之间可能有什么矛盾？

**答：**

| 优化手段 | 吞吐提升 | 潜在问题 |
|---------|---------|---------|
| 极端增大 Batch Size | ✅ | 模型陷入局部最优，泛化变差 |
| 使用更低精度（FP8） | ✅ 极大提速 | 引入数值下溢，导致 Loss 爆炸/NAN |
| 大量 Checkpointing | ✅ 省显存提吞吐 | 增加前向重算时间 |

**平衡策略：**
- 逐步增大 Batch Size，监控验证集 Loss
- FP8 需要配合适当的 Scaling 策略
- Checkpointing 层数选择要权衡计算和显存

---

## 面试金句

> "先看 top，CPU 满就是数据加载瓶颈；二看 Nsight Systems 的 Timeline：如果 GPU 呈现大块绿色的 Kernel 执行，就是算力瓶颈；如果大块红色的 NCCL Wait 或 cudaMemcpy，那就是通信或访存瓶颈。"

> "先确定是 Compute Bound 还是 Memory Bound。优先确认是否开启了 FlashAttention 和 TF32/BF16；如果开启了，尝试增大 Batch Size 跑满流处理器；排查是否有频繁的小数据 Host/Device 拷贝打断了 CUDA 流。"
