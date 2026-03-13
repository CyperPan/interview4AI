# 显存优化与训练框架

## 目录

- [ZeRO 系列详解](#zero-系列详解)
- [激活检查点](#激活检查点)
- [混合精度训练](#混合精度训练)
- [训练显存爆炸解决方案](#训练显存爆炸解决方案)
- [框架对比](#框架对比)

---

## ZeRO 系列详解

### 为什么 FSDP / ZeRO 能够节省显存？它们的核心思想是什么？

**答：**

**核心思想：用通信换显存（数据切片）**

```
传统 DP：                    ZeRO-3：
GPU0: [完整参数]             GPU0: [参数切片 0]
GPU1: [完整参数]             GPU1: [参数切片 1]  
GPU2: [完整参数]             GPU2: [参数切片 2]
GPU3: [完整参数]             GPU3: [参数切片 3]
         ↓                          ↓
   显存 = 4 × 模型大小          显存 = 1 × 模型大小 / 4
```

- 传统的 DP 每张卡存全量模型状态
- FSDP/ZeRO 将**参数、梯度、优化器状态**等价分片到 N 张卡上，单卡只存 1/N
- 需要计算时，再通过 All-Gather 临时拿回完整参数，算完立刻丢弃

### ZeRO Stage1、Stage2、Stage3 分别在分什么？

**答：**

| Stage | 分片内容 | 显存节省（N 卡时） |
|-------|---------|---------|
| **Stage 1** | 优化器状态 (Optimizer States) | 优化器状态部分被 N 卡均摊（N 足够大时最多约 4x） |
| **Stage 2** | 优化器状态 + 梯度 (Gradients) | 优化器+梯度被均摊（N 足够大时最多约 8x） |
| **Stage 3** | 优化器状态 + 梯度 + 参数 (Parameters) | 全部状态被 N 卡均摊，显存 ≈ 16/N bytes/参数 |

**详细说明：**

```
混合精度训练的模型状态组成（以 Adam 优化器为例）：
- FP16 参数 (Parameters): 2 bytes
- FP16 梯度 (Gradients): 2 bytes
- 优化器状态（共 12 bytes）:
  - FP32 Master Copy: 4 bytes
  - FP32 Momentum (一阶动量): 4 bytes
  - FP32 Variance (二阶动量): 4 bytes

总显存 = 2 + 2 + 12 = 16 bytes / 参数

ZeRO-3 分片后：
- 每张卡只存 1/N 的全部状态
- 显存 ≈ 16/N bytes / 参数
```

---

## 激活检查点

### 激活检查点（Activation Checkpointing）是什么？为什么能省显存？

**答：**

**原理：**

反向传播需要用到前向传播的激活值（Activation）。

```
传统方式：
前向: [Layer1] -> [Layer2] -> [Layer3] -> [Output]
保存:   [激活1]    [激活2]    [激活3]
反向:   [梯度1] <- [梯度2] <- [梯度3] <- [Loss]

检查点方式：
前向: [Layer1] -> [Layer2] -> [Layer3] -> [Output]
保存:   [✓]        [✗]        [✓]        
                ↑ 只保存检查点
反向: [重算L2] <- [梯度2] <- [重算L3] <- [Loss]
       ↑ 从L1重算      ↑ 从L3重算
```

**Trade-off：**
- 以 ~33% 的额外计算时间
- 将激活显存从 O(L) 降至 O(√L)（L 为层数），显著节省显存

---

## 混合精度训练

### 混合精度训练为什么能提升效率？可能带来哪些问题？

**答：**

**效率提升原因：**

1. **显存减半**：FP16/BF16 把权重和激活砍半，省 50% 读写带宽
2. **Tensor Core 加速**：激活硬件 Tensor Cores，算力翻倍

**潜在问题：**

| 精度 | 问题 | 原因 | 解决方案 |
|-----|------|------|---------|
| **FP16** | Underflow（下溢出） | 范围只有 6 万多，下限高，极小梯度变 0 | 动态 Loss Scaling |
| **BF16** | 精度较低 | 尾数位少 | 直接用（范围大，对大模型友好） |

**Loss Scaling 原理：**

```python
# 前向：放大 Loss
scaled_loss = loss * scale_factor
scaled_loss.backward()

# 反向：检查梯度是否溢出
if grad.is_inf_or_nan():
    scale_factor /= 2  # 减小缩放因子
else:
    optimizer.step()
    scale_factor *= 2  # 增大缩放因子
```

---

## 训练显存爆炸解决方案

### 训练显存爆了，从哪些方向解决？

**答：**

**按优先级排序：**

| 优先级 | 方案 | 效果 | 代价 |
|-------|------|------|------|
| 1 | 调小 Batch Size + 梯度累积 | 立竿见影 | 可能略微降低效率 |
| 2 | 开启 Activation Checkpointing | 极大节省 | ~33% 额外计算 |
| 3 | 提升 ZeRO 等级 (Stage 2→3) | 显著节省 | 增加通信 |
| 4 | ZeRO-Offload | 极大节省 | CPU-GPU 传输开销 |
| 5 | FlashAttention | 省 Attention 显存 | 无代价（推荐必开） |

---

## 框架对比

### DeepSpeed、Megatron、FSDP 各自适合的场景？

**答：**

| 框架 | 适合场景 | 特点 | 优缺点 |
|-----|---------|------|-------|
| **Megatron** | 超大规模（万卡集群）、追求极限 MFU | 3D 并行原生支持 | ✅ MFU 最高<br>❌ 代码侵入性强 |
| **DeepSpeed** | 资源有限、需要 CPU Offload | ZeRO 护城河深，插件化好 | ✅ 救场神器<br>✅ 社区活跃 |
| **FSDP** | 不需要复杂 TP，靠数据并行 | PyTorch 原生，生态好 | ✅ 代码最干净<br>✅ 易上手 |

**选择建议：**
- **追求极致性能**：Megatron
- **快速验证/资源紧张**：DeepSpeed
- **生产环境/易维护**：FSDP

---

## 面试金句

> "核心思想是用通信换显存。传统的 DP 每张卡存全量模型状态；FSDP/ZeRO 将参数、梯度、优化器状态等价分片到 N 张卡上，单卡只存 1/N。"

> "训练显存爆了，优先级：1. 调小 Batch + 梯度累积；2. 开 Activation Checkpointing；3. 提升 ZeRO 等级；4. ZeRO-Offload；5. FlashAttention。"

> "Megatron 适合超大规模追求极限 MFU，DeepSpeed 适合资源有限需要 Offload，FSDP 是 PyTorch 原生适合生产环境。"
