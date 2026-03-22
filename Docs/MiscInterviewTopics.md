# 综合面试题（杂项）

> 文档定位：覆盖不属于单一类别的高频面试题，包括 Agent、解码策略、损失函数、Scaling Law、大模型常见问题及解决方案等。

---

## 目录

- [Agent 与应用](#agent-与应用)
- [解码策略](#解码策略)
- [损失函数与基础](#损失函数与基础)
- [Scaling Law](#scaling-law)
- [参数量计算](#参数量计算)
- [DeepSeek R1](#deepseek-r1)
- [Adam 与 AdamW](#adam-与-adamw)
- [大模型常见问题与解决方案](#大模型常见问题与解决方案)
- [超长上下文](#超长上下文)
- [量化进阶](#量化进阶)
- [混合精度训练详解](#混合精度训练详解)
- [编译器与计算库基础](#编译器与计算库基础)

---

## Agent 与应用

### Agent 的思想是什么，Agent 包含哪些部分

**答：**

**Agent = LLM（大脑） + Tools（工具） + Memory（记忆） + Planning（规划）**

```
                    ┌─────────────┐
                    │   Planning   │  ← 任务分解、反思、调整策略
                    │  (规划能力)   │
                    └──────┬──────┘
                           │
┌──────────┐    ┌──────────┴──────────┐    ┌──────────┐
│  Memory  │ ←→ │        LLM         │ ←→ │  Tools   │
│ (记忆)   │    │   (推理引擎/大脑)    │    │ (工具)   │
│ 短期+长期 │    └──────────┬──────────┘    │ 代码/搜索 │
└──────────┘               │               │ API/数据库│
                    ┌──────┴──────┐        └──────────┘
                    │ Environment │
                    │  (环境交互)  │
                    └─────────────┘
```

**四大组件详解：**

| 组件 | 作用 | 实现方式 |
|------|------|---------|
| **LLM** | 推理、决策、理解 | GPT-4, Claude, Qwen 等 |
| **Tools** | 执行具体操作 | 代码执行、搜索引擎、API 调用、计算器 |
| **Memory** | 记住历史信息 | 短期：对话上下文；长期：向量数据库 |
| **Planning** | 任务分解与反思 | Chain-of-Thought, ReAct, Reflection |

**核心循环（ReAct 范式）：**
```
Observe → Think → Act → Observe → Think → Act → ...
观察环境 → 推理思考 → 执行动作 → 观察结果 → 继续思考 → ...
```

**主流框架：** LangChain, LangGraph, AutoGPT, CrewAI, OpenAI Function Calling

---

## 解码策略

### 常用的解码策略，TopP 和 TopK 的具体细节

**答：**

| 策略 | 原理 | 特点 |
|------|------|------|
| **Greedy** | 每步选概率最高的 token | 确定性，质量一般 |
| **Beam Search** | 维护 B 个最优候选序列 | 质量较好，但缺乏多样性 |
| **Top-K** | 从概率最高的 K 个 token 中采样 | 固定候选集大小 |
| **Top-P (Nucleus)** | 从累积概率 ≥ P 的最小 token 集中采样 | 自适应候选集大小 |
| **Temperature** | logits / T，T>1 更随机，T<1 更确定 | 控制分布平滑度 |

**Top-K 详解：**
```
原始概率: [0.3, 0.2, 0.15, 0.1, 0.08, 0.05, ...]
Top-K=3:  [0.3, 0.2, 0.15, 0,    0,    0,   ...]
重归一化:  [0.46, 0.31, 0.23, 0,   0,    0,   ...]
从前 3 个中采样

问题：K 是固定的，对于高确定性分布（一个 token 概率 0.99）K=50 会引入噪声
      对于低确定性分布（平坦分布）K=3 可能过于限制
```

**Top-P 详解：**
```
排序后概率: [0.3, 0.2, 0.15, 0.1, 0.08, ...]
P=0.7:    累积 0.3 → 0.5 → 0.65 → 0.75 > 0.7 → 选前 4 个
重归一化后从这 4 个中采样

优势：自适应候选集大小
- 高确定性分布 → 候选少 → 更确定
- 低确定性分布 → 候选多 → 更多样
```

**实际使用：** 通常组合使用：先 Top-K 截断 → 再 Top-P 筛选 → Temperature 调节

---

## 损失函数与基础

### SFT 损失及其公式

**答：**

SFT 使用标准的 **交叉熵损失（Cross-Entropy Loss）** 做下一个 token 预测：

```
L_SFT = -(1/T) × Σ_{t ∈ response} log P_θ(y_t | x, y_{<t})

其中：
x = prompt（输入）
y = response（目标输出）
T = response 中的 token 数
P_θ = 模型预测的 token 概率
```

**关键细节：只在 response token 上计算 loss，不计算 prompt token 的 loss。**

```
Prompt:    "请翻译以下文本：Hello world"
Response:  "你好世界"

Loss mask: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
                                  ↑ 只计算这些 token 的 loss
```

### 对比学习原理和 Loss

**答：**

**核心思想：** 拉近相似样本、推远不相似样本在表示空间中的距离。

**InfoNCE Loss（对比学习标准损失）：**
```
L = -log( exp(sim(z_i, z_j⁺) / τ) / Σ_k exp(sim(z_i, z_k) / τ) )

其中：
z_i = anchor 的表示
z_j⁺ = 正样本（与 anchor 相似）的表示
z_k = 所有样本（正 + 负）的表示
sim = 相似度函数（通常是余弦相似度）
τ = 温度参数（越小越尖锐）
```

**直觉理解：** 像 softmax 分类一样，把正样本当"正确类"，负样本当"错误类"。

**代表方法：**
- **SimCLR**：同一图片两种增强 → 正样本对
- **CLIP**：图片-文本配对 → 正样本对
- **DPR**：问题-段落配对 → 用于检索

**温度 τ 的影响：** τ 小 → 更关注困难负样本；τ 大 → 更均匀地对待所有负样本

### 口述交叉熵公式，口述自注意力过程

**答：**

**交叉熵：**
```
H(p, q) = -Σ_x p(x) log q(x)

在 LLM 中：
L = -Σ_{t=1}^{T} log P_model(y_t | y_{<t})
  = -log P_model(正确token)

直觉：正确 token 的预测概率越高，loss 越低
```

**自注意力过程（口述版）：**

```
输入：一组 token 的向量表示 X ∈ R^{n×d}

第一步：通过三个线性变换得到 Q、K、V
  Q = X · W_Q    ← "我在找什么"
  K = X · W_K    ← "我有什么"
  V = X · W_V    ← "我的内容"

第二步：计算注意力分数
  Scores = Q · K^T / √d_k    ← 每对 token 的相关性

第三步：因果掩码（decoder-only）
  Scores = Scores.masked_fill(causal_mask, -inf)

第四步：Softmax 归一化
  Weights = softmax(Scores)    ← 每行和为 1

第五步：加权聚合
  Output = Weights · V    ← 用注意力权重聚合各 token 的 V

第六步：输出投影
  Output = Output · W_O
```

---

## Scaling Law

### SFT 训练时数据规模和模型大小的 Scaling Law

**答：**

**Chinchilla Scaling Law（DeepMind, 2022）：**

```
最优训练配置：
  最优 token 数 D ≈ 20 × N（N 为参数量）

计算预算固定时：
  C ≈ 6 × N × D（C 为总 FLOPs）

损失预测：
  L(N, D) = A/N^α + B/D^β + E
  其中 α ≈ 0.34, β ≈ 0.28
```

**实际应用：**

| 模型大小 | 最优 token 数 | 实际训练 token |
|---------|-------------|--------------|
| 7B | 140B | 1-2T（过度训练） |
| 13B | 260B | 1-2T |
| 70B | 1.4T | 2-15T |

**为什么实际训练远超最优 token 数？**
- Chinchilla 优化的是训练效率（总 FLOPs 最小化）
- 实际关注推理效率：更小模型 + 更多数据 → 推理更快
- **Over-training 策略**：用更多数据训练较小模型，牺牲训练效率换推理效率

**SFT 阶段的 Scaling：**
- SFT 数据量通常远小于预训练（几万到几十万条）
- 数据质量 > 数据数量
- 过多 SFT 数据可能导致"对齐税"（alignment tax）——通用能力下降

---

## 参数量计算

### Transformer 参数量怎么计算

**答：**

**单层 Transformer Block 参数量：**

```
MHA (Multi-Head Attention):
  W_Q: d × d = d²
  W_K: d × d = d²
  W_V: d × d = d²
  W_O: d × d = d²
  小计: 4d²

FFN (SwiGLU):
  W_gate: d × (8d/3) = 8d²/3
  W_up:   d × (8d/3) = 8d²/3
  W_down: (8d/3) × d = 8d²/3
  小计: 8d²  (≈ 3 × 8d²/3)

RMSNorm × 2:
  2 × d = 2d  (可忽略)

单层总计 ≈ 4d² + 8d² = 12d²
```

**整个模型参数量：**
```
模型参数 = L × 12d² + V × d + d
                       ↑ Embedding  ↑ Final Norm
其中 L=层数, d=hidden_dim, V=词表大小

验证（LLaMA-7B: L=32, d=4096, V=32000）：
= 32 × 12 × 4096² + 32000 × 4096
= 32 × 12 × 16.7M + 131M
= 6.44B + 0.13B ≈ 6.6B  ✓
```

**如果用 GQA（KV heads < Q heads）：**
```
W_Q: d × d = d²
W_K: d × d_kv = d × (n_kv × d_h)  ← 更小
W_V: d × d_kv = d × (n_kv × d_h)  ← 更小
W_O: d × d = d²
```

---

## DeepSeek R1

### 了解 DeepSeek R1 吗，介绍一下

**答：**

**DeepSeek-R1** 是 DeepSeek 推出的推理增强型大模型，核心特色是通过 RL 训练获得长链推理能力。

**训练流程：**
```
Step 1: SFT 冷启动
  基于 DeepSeek-V3 做少量 SFT
  教会基本输出格式（<think>...</think> 标签）

Step 2: RL 训练（GRPO）
  使用规则化奖励（Rule-based Reward）：
  - 答案正确性（数学/代码可验证）
  - 格式合规性（有思考过程 + 最终答案）

  不使用 Reward Model，避免 reward hacking

Step 3: 拒绝采样 + SFT
  用 RL 模型生成大量推理数据
  筛选高质量样本做 SFT（蒸馏 RL 能力到更稳定的模型）
```

**R1-Zero（纯 RL 版本）：**
- 直接从 base 模型做 RL，不经过 SFT
- 出现了涌现行为：自我验证、反思、尝试不同方法
- 但可读性差、语言混乱 → 需要 SFT 冷启动来改善

**关键创新：**
1. **GRPO**：无 Critic 的 RL，用组内相对奖励
2. **规则化奖励**：不依赖 RM，用代码运行/数学验证等硬标准
3. **涌现推理**：RL 过程中自发学会 chain-of-thought
4. **开源**：完整公开模型权重和训练方法

### R1 在 SFT 时冷启动的目的是什么

**答：**

**冷启动 SFT 的三个目的：**

1. **教会输出格式**：
   ```
   <think>
   让我一步一步思考这个问题...
   首先...
   然后...
   所以答案是...
   </think>

   最终答案是 42。
   ```
   没有冷启动，模型不知道要用 `<think>` 标签分隔推理和答案

2. **提供初始推理能力**：
   - RL 需要一个起点——如果模型完全不会推理，RL 探索效率极低
   - 冷启动给出"初始策略"，RL 在此基础上改进

3. **改善可读性**：
   - R1-Zero（无冷启动）的推理过程语言混乱、多语言混杂
   - 冷启动确保输出是流畅的目标语言

**冷启动数据量很小**（几千条），目的不是教知识，而是教格式和基本推理框架。真正的推理能力来自后续的 RL 训练。

---

## Adam 与 AdamW

### Adam, AdamW 原理

**答：**

**Adam（Adaptive Moment Estimation）：**

```python
# 核心算法
m_t = β₁ × m_{t-1} + (1 - β₁) × g_t        # 一阶动量（梯度均值的指数移动平均）
v_t = β₂ × v_{t-1} + (1 - β₂) × g_t²       # 二阶动量（梯度方差的指数移动平均）

# 偏差校正（训练初期 m, v 偏向 0）
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# 参数更新
θ_{t+1} = θ_t - lr × m̂_t / (√v̂_t + ε)

# 典型超参数：β₁=0.9, β₂=0.999, ε=1e-8
```

**直觉理解：**
- m_t：梯度的"方向"（动量，平滑掉随机波动）
- v_t：梯度的"幅度"（每个参数的历史梯度大小）
- `m/√v`：大梯度参数 → 缩小步长；小梯度参数 → 放大步长 → **自适应学习率**

**AdamW（Decoupled Weight Decay）：**

```
Adam 中的 L2 正则化：
  θ -= lr × (m̂/(√v̂ + ε) + λ × θ)
  → weight decay 被自适应学习率缩放 → 正则化效果不均匀

AdamW 的改进：
  θ -= lr × m̂/(√v̂ + ε)    # 先做 Adam 更新
  θ -= lr × λ × θ            # 再独立做 weight decay

  → weight decay 不受 Adam 的自适应影响 → 正则化更均匀
```

**为什么 LLM 都用 AdamW：** 因为 AdamW 的 weight decay 是真正的"让权重衰减"，而 Adam + L2 的 weight decay 会被梯度大小调节，对大梯度参数几乎无效。

---

## 大模型常见问题与解决方案

### 大模型灾难性遗忘是什么，如何解决

**答：**

**灾难性遗忘（Catastrophic Forgetting）：** 在新任务上微调后，模型在旧任务上的能力严重退化。

```
Base Model:     数学 90分 + 代码 85分 + 翻译 88分
SFT on 医疗:    医疗 92分 + 数学 60分↓ + 代码 55分↓ + 翻译 50分↓
               ← 灾难性遗忘
```

**解决方案：**

| 方法 | 原理 | 效果 |
|------|------|------|
| **LoRA** | 冻结原始权重，只训练低秩增量 | ⭐⭐⭐ 最常用 |
| **数据混合（Replay）** | 微调时混入旧任务数据 | ⭐⭐⭐ 简单有效 |
| **正则化（EWC）** | 对重要参数施加更强约束 | ⭐⭐ 理论优雅 |
| **渐进式训练** | 逐步增加新任务比例 | ⭐⭐ 需要调参 |
| **多任务学习** | 同时训练所有任务 | ⭐⭐⭐ 但需要所有数据 |
| **模块化方法** | 每个任务一个 Adapter/LoRA | ⭐⭐⭐ 互不干扰 |

### 为什么会有复读机问题，业内解决方法有哪些

**答：**

**复读机问题（Repetition）：** 模型反复生成相同的词、短语或句子。

**根本原因：**
1. **正反馈循环**：生成 token A → A 进入上下文 → 增加再次生成 A 的概率 → 循环
2. **训练数据中的重复模式**：数据本身有重复 → 模型学到重复是"正常的"
3. **注意力模式坍缩**：Attention 集中在某些 token → 输出单调
4. **decode 策略**：Greedy/Beam Search 更容易陷入重复（无随机性）

**解决方法：**

| 方法 | 原理 | 实现位置 |
|------|------|---------|
| **Repetition Penalty** | 降低已生成 token 的概率（乘以惩罚因子） | 采样时 |
| **Frequency Penalty** | 根据 token 出现频率递增惩罚 | 采样时（OpenAI API） |
| **Presence Penalty** | token 出现过就固定惩罚 | 采样时（OpenAI API） |
| **N-gram Blocking** | 禁止生成已出现过的 n-gram | 采样时 |
| **Temperature 采样** | 增加随机性，避免陷入局部最优 | 采样时 |
| **RLHF** | 奖励模型惩罚重复 | 训练时 |
| **数据去重** | 减少训练数据中的重复模式 | 数据处理 |
| **对比解码** | 用小模型的输出作为"惩罚基线" | 推理时 |

### 什么是大模型幻觉，如何缓解

**答：**

**幻觉（Hallucination）：** 模型自信地生成事实错误或完全虚构的内容。

**两种类型：**
- **事实幻觉**：编造不存在的事实（"爱因斯坦在 1920 年发明了互联网"）
- **忠实度幻觉**：与输入/上下文矛盾（摘要时添加原文没有的信息）

**根本原因：**
1. 模型学到的是"统计模式"而非"事实知识"
2. 训练数据中的错误信息
3. 知识截止日期后的信息空白
4. 过度自信的生成策略

**缓解方法：**

| 方法 | 原理 | 效果 |
|------|------|------|
| **RAG** | 检索相关文档作为上下文，基于证据生成 | ⭐⭐⭐ 最有效 |
| **RLHF 对齐** | 训练模型说"我不知道"而非编造 | ⭐⭐⭐ |
| **Chain-of-Thought** | 逐步推理减少跳跃性错误 | ⭐⭐ |
| **自一致性检查** | 多次生成，取一致的答案 | ⭐⭐ |
| **工具调用** | 不确定的查搜索引擎/计算器 | ⭐⭐⭐ |
| **引用标注** | 要求模型给出信息来源 | ⭐⭐ |
| **事实微调** | 在高质量事实数据上 SFT | ⭐⭐ |
| **置信度校准** | 训练模型输出不确定性估计 | ⭐ 研究中 |

---

## 超长上下文

### 超长上下文业界一般怎么做

**答：**

**主要技术路线：**

**1. 位置编码扩展：**

| 方法 | 原理 | 代表模型 |
|------|------|---------|
| **NTK-aware Interpolation** | 调整 RoPE 频率基数（高频外推，低频内插） | CodeLlama |
| **YaRN** | NTK + attention scaling factor | Qwen |
| **Dynamic NTK** | 根据实际序列长度动态调整基数 | 多种模型 |
| **Position Interpolation** | 线性缩放位置 ID 到训练范围 | Meta |

**YaRN 原理：**
```
标准 RoPE: θ_i = 10000^{-2i/d}

YaRN 三步：
1. NTK-aware: 根据频率区间做不同处理
   - 高频维度（变化快）：不缩放（外推）
   - 低频维度（变化慢）：线性内插
   - 中间维度：混合策略

2. Attention Scaling: 乘以 √(s)（s=扩展比例）
   补偿外推时的注意力分数偏移

3. 少量长文本微调（~1000 步）
```

**2. 注意力机制优化：**
- **Sliding Window Attention**（Mistral）：每层只关注局部窗口
- **Sparse Attention**（BigBird）：局部 + 全局 + 随机稀疏
- **Ring Attention**：跨 GPU 切分序列维度，各 GPU 处理一段
- **FlashAttention**：减少 HBM 访问，支持更长序列

**3. 系统层面：**
- **Sequence Parallelism**：序列维度切分到多卡
- **KV Cache 量化**：FP8/INT8 压缩缓存
- **KV Cache 压缩**：MLA、GQA 减少缓存量
- **Chunked Processing**：长输入分段 prefill

**4. 检索增强（非纯模型方案）：**
- 超长文档 → 分段检索 → 只把相关段落放入上下文
- 实际应用中最常用的方案

---

## 量化进阶

### 对称量化和非对称量化的区别

**答：**

| 维度 | 对称量化 | 非对称量化 |
|------|---------|-----------|
| **参数** | 只需 scale | 需要 scale + zero_point |
| **公式** | x_q = round(x / scale) | x_q = round(x / scale) + zero_point |
| **反量化** | x ≈ x_q × scale | x ≈ (x_q - zero_point) × scale |
| **零点映射** | 0 → 0（零点固定） | 0 可映射到任意位置 |
| **INT8 范围** | [-127, 127] | [0, 255] |

**各自适合的场景：**
- **权重**：通常对称分布（接近正态，均值 ≈ 0）→ **对称量化**，实现简单，MatMul kernel 不需处理 zero_point
- **Activation**：经常不对称（ReLU 后全正，某些 channel 偏移大）→ **非对称量化**，精度更好

**常见组合：W8A8 = 权重对称量化 + activation 非对称量化。**

### AWQ 详解：为什么看 Activation 而不是权重大小

**答：**

**AWQ = Activation-aware Weight Quantization**，核心看的是 activation magnitude，不是权重本身的大小。

**原因：** 量化误差的影响 = 权重误差 × activation。同样的量化误差，activation 大的 channel 输出偏差更大。所以要保护的不是"大权重"，而是"被大 activation 乘的权重"。

**做法：**
```
1. 跑校准数据 → 统计每个 channel 的 activation magnitude
2. magnitude 大的 channel → 对应权重乘 scale 放大
3. 量化放大后的权重（相对精度损失更小）
4. 推理时 activation 侧除以同一个 scale 补偿

数学等价：W_q = Quantize(W × s) / s
         Y = X @ W = (X / s) @ (W × s) → 结果不变
```

---

## 混合精度训练详解

### 混合精度训练原理和 Loss Scaling

**答：**

**标准做法：FP16/BF16 + FP32 混合（不是 INT8/INT4，那是推理量化）。**

```
每一步训练：
1. FP32 权重 → 拷贝一份转 FP16
2. 用 FP16 做前向和反向 → 得到 FP16 梯度（快，省显存）
3. FP16 梯度转回 FP32
4. 用 FP32 梯度更新 FP32 权重主副本（精度保障）
5. 回到 1
```

**为什么不能全用 FP16：**
1. **梯度下溢**：FP16 最小正数 ~5.96e-8，很多梯度（如 1e-10）直接变 0，模型学不动
2. **更新被吞掉**：权重=1.0，更新量=1e-7 → FP16 下 1.0+1e-7=1.0（精度不够，被舍入）
3. **Loss 溢出**：FP16 最大值 ~65504，中间激活值可能超出变成 inf

**Loss Scaling 解决下溢：**
```python
# 前向：放大 loss
scaled_loss = loss * scale_factor  # 如 ×1024
scaled_loss.backward()              # 梯度也被放大 1024 倍 → 小梯度不下溢

# 更新前：缩回来
if grad.is_inf_or_nan():
    scale_factor /= 2              # 溢出了就减小
else:
    grads /= scale_factor           # 恢复真实值
    optimizer.step()
    scale_factor *= 2              # 没问题就逐步放大
```

**BF16 趋势：** 8-bit 指数 → 范围与 FP32 相同，不容易溢出，大多数情况不需要 Loss Scaling。现代大模型训练主流用 **BF16 + FP32 混合**。

---

## 编译器与计算库基础

### TVM 在 AI 推理中的角色

**答：**

```
类比：
  C 代码 → GCC/Clang 编译器 → x86 机器码
  PyTorch 模型 → TVM 编译器 → CUDA kernel / 昆仑芯片指令 / ARM 指令
```

| 维度 | 手写 CUDA | TVM |
|------|----------|-----|
| **性能上限** | 最高 | 接近（90-95%） |
| **开发效率** | 极低（几百行） | 高（自动生成） |
| **可移植性** | 只能跑 NVIDIA | 可编译到任何硬件 |
| **适用场景** | 极致优化热点算子 | 快速适配多种硬件 |

**TVM 的两级优化：**
- **图级别**：算子融合、常量折叠、死代码消除
- **算子级别**：自动搜索最优 tiling/loop 策略（AutoTVM / Ansor）

**为什么对百度重要：** 百度有自研昆仑芯片，TVM 类编译器可以让同一模型自动编译到 GPU 和昆仑上，大幅降低适配成本。

### cuBLAS 和主流计算库速记

**答：**

| 库 | 硬件 | 用途 | 典型调用 |
|----|------|------|---------|
| **cuBLAS** | NVIDIA GPU | 矩阵乘法（GEMM） | PyTorch `F.linear()` 底层 |
| **cuDNN** | NVIDIA GPU | DL 原语（Conv, BN, Pooling） | PyTorch Conv2d 底层 |
| **CUTLASS** | NVIDIA GPU | 可定制的 GEMM 模板库 | 融合 dequant + MatMul |
| **MKL** | Intel CPU | CPU 数学加速 | NumPy 底层 |
| **OpenBLAS** | CPU（通用） | 开源 CPU 线性代数 | MKL 的开源替代 |
| **Eigen** | CPU | C++ 模板线性代数 | 小矩阵、CPU 端推理 |
| **MIOpen** | AMD GPU | AMD 版 cuDNN | ROCm 生态 |

**面试追问"既然 cuBLAS 很快为什么还要自己写"的答法：**
通用 MatMul 确实用 cuBLAS，但某些场景需要自定义 kernel：
- dequant + MatMul 融合（cuBLAS 不支持）
- 小 batch decode 的针对性优化
- 非标准 layout 的矩阵运算

---

## 面试金句

> "Agent 的核心是 LLM + Tools + Memory + Planning，本质是把 LLM 从'回答问题'升级为'完成任务'。"

> "Top-P 比 Top-K 更灵活：高确定性时候选少，低确定性时候选多，自适应调节多样性。"

> "大模型幻觉的最有效缓解方案是 RAG——让模型基于检索到的事实回答，而非凭记忆编造。"

> "超长上下文三条路：位置编码扩展（YaRN）、注意力优化（稀疏/滑窗）、系统层面（序列并行/KV 压缩）。工程中最常用的其实是 RAG。"

> "Chinchilla Scaling Law 说最优是 20 倍 token/参数，但实际训练远超此数——因为我们优化的是推理效率，不是训练效率。"

> "AWQ 看的是 activation magnitude 而非权重大小。同样的量化误差，被大 activation 乘的权重影响更大，所以要保护的是'被大 activation 乘的权重'。"

> "混合精度训练 = FP16/BF16 做前向反向提速 + FP32 保留权重和优化器状态保精度。Loss Scaling 解决 FP16 梯度下溢。BF16 因范围大（同 FP32）逐渐成为主流。"

> "Roofline Model：算出 kernel 的 arithmetic intensity，跟硬件算力带宽比比较。Decode Linear AI≈1 远低于 A100 拐点 156，是 memory-bound。"
