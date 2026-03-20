# SFT / RL 训练

## 目录

- [三阶段训练差异](#三阶段训练差异)
- [RLHF 复杂度分析](#rlhf-复杂度分析)
- [RL 训练效率瓶颈](#rl-训练效率瓶颈)
- [强化学习面试高频题](#强化学习面试高频题)

---

## 三阶段训练差异

### 预训练、SFT、RL 训练在系统优化上的不同？

**答：**

| 阶段 | 优化重点 | 核心特点 |
|-----|---------|---------|
| **Pretrain** | 极度追求静态图下的极限吞吐（MFU）和长时间稳定性 | 追求极致性能，长时间运行 |
| **SFT** | 数据长度极不固定，避免无效 Padding | 极度依赖 **Sequence Packing（拼装）** |
| **RL** | "生成 + 训练" 的循环 | 不仅卡训练，更卡推理（Rollout 生成阶段极其耗时） |

**详细说明：**

**Pretrain：**
- 数据长度相对固定
- 可以充分优化静态图
- 追求极致 MFU（Model FLOPs Utilization）
- 需要长时间稳定运行（数周甚至数月）

**SFT：**
```
数据长度分布：
[████████][█][██][██████████][███]
   1024    64 128    2048      256

未 Packing：
Batch: [████    ][█       ][██      ]  ← 大量 Padding 浪费
       1024(Pad) 64      128

Packing 后：
Batch: [████████][████████]          ← 紧凑拼接，无浪费
       1024+64   128+2048+256
```

**RL：**
- 是"生成 + 训练"的循环
- Rollout 生成阶段是推理过程（自回归生成）
- 训练阶段是标准的反向传播
- 两个阶段底层逻辑冲突，难以高效融合

---

## RLHF 复杂度分析

### RLHF / PPO 类训练为什么比普通 SFT 复杂？

**答：**

**PPO 需要同时维护 4 个模型：**

| 模型 | 作用 | 是否更新参数 |
|-----|------|------------|
| **Actor** | 生成响应的策略模型 | ✅ 训练更新 |
| **Critic** | 评估响应的价值模型 | ✅ 训练更新 |
| **Reward** | 给响应打分的奖励模型 | ❌ 冻结 |
| **Reference** | 参考策略，防止 Actor 偏离太远 | ❌ 冻结 |

**内存常态爆炸：** 4 个模型同时驻留显存

**Actor 生成与模型参数更新的冲突：**

```
生成阶段（推理）：
- 需要 KV-Cache
- 自回归解码
- 底层逻辑：逐个 Token 生成

训练阶段（训练）：
- 标准反向传播
- 底层逻辑：批量梯度更新

冲突点：
- 生成和训练的内存布局不同
- 通常需要把生成和训练拆分到两个框架（如 vLLM + DeepSpeed）
- 需要解决内存状态流转问题
```

---

## RL 训练效率瓶颈

### RL 训练效率瓶颈可能不只在 GPU 算力，还可能在哪？

**答：**

| 瓶颈点 | 说明 |
|-------|------|
| **Rollout 阶段** | Actor 生成经验的速度（推理阶段） |
| **Experience Buffer 拷贝** | CPU 与 GPU 之间的大量数据传输开销 |
| **Reward 模型打分** | 串行阻塞等待，无法并行 |

**优化方向：**

1. **加速 Rollout**：使用 vLLM 等高效推理引擎
2. **异步 Experience Buffer**：减少 CPU-GPU 拷贝阻塞
3. **Reward 模型批量化**：批量打分，提高吞吐

---

## 强化学习面试高频题

### RLHF 的训练流程

**答：**

RLHF（Reinforcement Learning from Human Feedback）分三个阶段：

```
阶段 1: SFT（Supervised Fine-Tuning）
Base Model + 高质量指令数据 → SFT Model
目的：教会模型基本的指令遵循能力

阶段 2: Reward Model 训练
SFT Model 对同一 prompt 生成多个回复 → 人工标注偏好对 (chosen > rejected)
→ 训练 Reward Model（Bradley-Terry 模型）
Loss = -E[log σ(r(chosen) - r(rejected))]

阶段 3: RL 训练（PPO/GRPO）
┌─────────────────────────────────────────────┐
│ 循环：                                       │
│ 1. Actor 对 prompt 生成回复（Rollout）         │
│ 2. Reward Model 对回复打分                    │
│ 3. Critic Model 估计价值（PPO）/ 组内对比（GRPO）│
│ 4. 计算优势函数 Advantage                     │
│ 5. PPO/GRPO 更新 Actor（+ KL 惩罚防偏离）      │
└─────────────────────────────────────────────┘
```

### SFT 和 RLHF 的作用有什么区别

**答：**

| 维度 | SFT | RLHF |
|------|-----|------|
| **学什么** | 格式、风格、指令遵循 | 价值观、偏好、质量判断 |
| **数据来源** | 人工标注的 (input, output) 对 | 人工偏好排序 (chosen > rejected) |
| **优化目标** | 最大化 P(正确答案\|输入) | 最大化人类满意度（奖励） |
| **学习方式** | 模仿学习（照抄好答案） | 探索学习（自己生成并改进） |
| **泛化能力** | 受限于训练数据覆盖 | 可泛化到训练数据未覆盖的场景 |
| **比喻** | 学做菜（照着菜谱做） | 学品味（学会判断好坏并创新） |

**关键区别：** SFT 只能学"怎么做"，RLHF 能学"什么是好的"。SFT 模型可能生成格式正确但内容有害的回复，RLHF 能对齐人类价值观。

### Critic Model 和 Reward Model 的区别

**答：**

| 维度 | Reward Model | Critic Model (Value Function) |
|------|-------------|------------------------------|
| **评估时机** | 回复完成后，给整体打分 | 每个 token 位置，估计未来累积奖励 |
| **输出含义** | 绝对质量分数 r(x, y) | 状态价值 V(s_t) = E[未来总回报] |
| **训练数据** | 人工偏好对 | RL 过程中的经验数据 |
| **是否更新** | ❌ 冻结 | ✅ 训练更新 |
| **作用** | 提供最终奖励信号 | 作为 baseline 减少方差 |

**优势函数：** `A_t = R - V(s_t)`（Advantage = 实际回报 - 预期回报）

**组内相对优势（GRPO）：**
```
对同一 prompt 生成 G 个回复，计算各自 reward r_i
Advantage_i = (r_i - mean(r)) / std(r)
```

**组内差距大的影响：**
- std(r) 大 → 归一化后 advantage 适中，训练稳定
- 如果某些回复 reward 极高/极低 → advantage 极端值 → 梯度波动大
- 需要 PPO 的 clip 机制或 GRPO 的组归一化来控制更新幅度

### DPO、PPO、GRPO、DAPO 原理、公式和对比

**答：**

**PPO（Proximal Policy Optimization）：**
```
L_PPO = E[min(r(θ) · A, clip(r(θ), 1-ε, 1+ε) · A)]

其中：r(θ) = π_θ(y|x) / π_old(y|x)  （新旧策略概率比）
      A = 优势函数（由 Critic 估计）
      ε = 裁剪范围（通常 0.1-0.2）

完整目标：L = L_PPO - β · KL(π_θ || π_ref)
需要：Actor + Critic + Reward + Reference = 4 个模型
```

**DPO（Direct Preference Optimization）：**
```
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

其中：y_w = 偏好回复（chosen），y_l = 非偏好回复（rejected）
      β = 温度参数
      π_ref = 参考策略（SFT 模型）

核心思想：将 RM + RL 合并为单步优化
需要：Actor + Reference = 2 个模型（无需 RM 和 Critic）
```

**GRPO（Group Relative Policy Optimization）：**
```
L_GRPO = E[min(r(θ) · Â, clip(r(θ), 1-ε, 1+ε) · Â)] - β · KL

其中：对每个 prompt 采样 G 个回复 {y_1, ..., y_G}
      计算 reward {r_1, ..., r_G}
      Â_i = (r_i - mean(r)) / std(r)  ← 组内归一化替代 Critic

需要：Actor + Reward + Reference = 3 个模型（无需 Critic）
特点：DeepSeek-R1 使用此方法
```

**DAPO（Decoupled Alignment from Policy Optimization）：**
```
基于 GRPO 的改进：
1. Dynamic Sampling：过滤太简单/太难的样本（全对或全错的 prompt 不参与训练）
2. Token-level Loss：按 token 计算 loss（而非 sequence-level），避免长回复被低估
3. Overlong Reward Shaping：对过长回复给予软惩罚而非截断
4. 去掉 KL 惩罚：允许更大的探索空间

需要：Actor + Reward = 2 个模型（无 Reference，无 Critic）
```

**对比总结：**

| 维度 | PPO | DPO | GRPO | DAPO |
|------|-----|-----|------|------|
| **模型数** | 4 | 2 | 3 | 2 |
| **需要 RM** | ✅ | ❌ | ✅ | ✅ |
| **需要 Critic** | ✅ | ❌ | ❌ | ❌ |
| **在线/离线** | On-policy | Off-policy | On-policy | On-policy |
| **实现复杂度** | 最高 | 最低 | 中等 | 中等 |
| **探索能力** | 强 | 弱 | 中 | 最强 |
| **训练稳定性** | 中（需调参） | 高 | 较高 | 较高 |
| **适用场景** | 通用 | 简单对齐 | 推理任务 | 开放式推理 |

### DPO 的原理和相比 PPO 的优势

**答：**

**DPO 核心推导：**

从 RLHF 目标出发：`max E[r(x,y)] - β·KL(π_θ || π_ref)`

最优策略的闭式解：`π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)`

反解 reward：`r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β·log Z(x)`

将 reward 代入 Bradley-Terry 偏好模型，Z(x) 消掉：

`P(y_w > y_l) = σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))`

**DPO 相比 PPO 的优势：**
1. **无需训练 RM**：直接从偏好数据优化策略
2. **无需在线采样**：不用 rollout 生成（省大量推理资源）
3. **实现简单**：只需 2 个模型（Actor + Reference），标准 SFT 训练流程
4. **训练更稳定**：没有 RL 训练的不稳定性

**DPO 的劣势：**
- **Off-policy**：使用固定数据集，存在分布偏移
- **探索不足**：无法发现训练数据未覆盖的好回复
- **对数据质量敏感**：偏好标注质量直接影响结果

### PPO 算法中为什么有 Reward Model 又有 Critic Model

**答：**

**RM 给出稀疏信号（最终评分），Critic 给出稠密信号（过程评估）。**

```
Actor 生成回复: "The answer is 42 because ..."
                token_1  token_2  token_3 ... token_n

Reward Model:   只在最后给一个分数 → r = 0.85
                                         ↑ 整体质量分

Critic Model:   V(s_1)=0.3, V(s_2)=0.5, ... V(s_n)=0.8
                ↑ 每个位置的未来预期值

Advantage:      A_t = R_total - V(s_t)
                ↑ "实际表现 - 预期表现" = 这个 token 的贡献
```

**为什么不能只用 RM？**
- RM 只给最终分，无法判断哪些 token 好、哪些不好（Credit Assignment 问题）
- 如果只用最终 reward，方差极大，训练不稳定
- Critic 的价值估计作为 baseline，大幅减少梯度方差

**类比：** RM 是考试总分，Critic 是每道题的预估分。有了每道题的预估，你就知道哪道题超常发挥（advantage > 0），哪道题失误了（advantage < 0）。

### 奖励模型的训练

**答：**

**数据构造：**
```
1. 对每个 prompt，用 SFT 模型生成多个回复
2. 人工标注偏好排序：y_1 > y_2 > y_3 > y_4
3. 构造偏好对：(prompt, chosen=y_i, rejected=y_j) where i < j
```

**模型架构：**
- 基于 SFT 模型初始化
- 去掉 LM Head，换成标量奖励头：`hidden_state → Linear(d, 1) → scalar reward`
- 通常取最后一个 token 的 hidden state 作为序列表示

**训练目标（Bradley-Terry 模型）：**
```
L_RM = -E[log σ(r_θ(x, y_w) - r_θ(x, y_l))]

含义：让 chosen 的奖励高于 rejected 的奖励
σ 是 sigmoid 函数，将分差映射到 [0,1] 概率
```

**训练注意事项：**
1. 数据多样性：不同难度、不同领域的偏好对
2. 奖励校准：不同 prompt 间的奖励可比性
3. 防止 Reward Hacking：模型可能找到 RM 的漏洞（如更长 = 更高分）
4. 定期更新：RM 应随策略模型更新而重训

### On-policy 和 Off-policy 的不同和优缺点

**答：**

| 维度 | On-policy | Off-policy |
|------|-----------|------------|
| **数据来源** | 当前策略生成 | 其他策略或历史数据 |
| **每次更新后** | 必须重新采样 | 可复用旧数据 |
| **样本效率** | 低（用完即弃） | 高（可重复使用） |
| **稳定性** | 较好（数据匹配策略） | 可能不稳定（分布偏移） |
| **计算开销** | 高（需要频繁 rollout） | 低（不需要在线生成） |
| **代表算法** | PPO, GRPO, DAPO | DPO, offline RL |

**在 LLM 对齐中：**
- **PPO/GRPO**（On-policy）：每轮用当前 Actor 生成回复 → RM 打分 → 更新 Actor → 重新生成
- **DPO**（Off-policy）：使用固定的偏好数据集，不需要在线生成

**Trade-off：** On-policy 质量更高但开销大（每次都要 rollout），Off-policy 高效但分布偏移可能导致性能天花板低。

### 如何理解强化学习熵的概念，如何在训练中保持较高水平

**答：**

**熵的定义：** `H(π) = -Σ π(a|s) log π(a|s)`

**直觉理解：**
- 高熵 → 策略输出分布均匀 → 生成多样、有探索性
- 低熵 → 策略输出集中在少数 token → 生成确定、缺乏创新
- 熵坍塌（Entropy Collapse）→ 模型反复生成相同回复

**保持高熵的方法：**

| 方法 | 原理 | 应用 |
|------|------|------|
| **熵正则化** | 在 loss 中加 `+ α·H(π)` | PPO 常用 |
| **KL 惩罚** | `- β·KL(π_θ \|\| π_ref)` 保持接近多样的 reference | PPO/GRPO |
| **Temperature Scaling** | 提高采样温度增加随机性 | 推理和训练 |
| **去掉 KL 约束** | DAPO 的做法，允许更大探索 | 需要其他机制防止崩溃 |
| **多样化数据** | Rollout 时使用多种 prompt | 增加经验多样性 |

**实际影响：** 熵过低 → 复读机、模式坍缩；熵过高 → 输出随机、质量差。需要在探索（exploration）和利用（exploitation）之间平衡。

### RL 训练领域尚存在哪些问题

**答：**

| 问题 | 说明 |
|------|------|
| **Reward Hacking** | 模型找到 RM 漏洞，获得高分但实际质量差（如更长=更高分） |
| **RM 泛化性** | RM 在分布外 prompt 上不可靠，可能给出错误信号 |
| **训练不稳定** | RL 训练容易出现 loss 震荡、突然崩溃 |
| **样本效率低** | On-policy 方法需要大量 rollout，计算开销巨大 |
| **模式坍塌** | 模型收敛到少数固定回复模式，多样性丧失 |
| **评估困难** | 如何衡量对齐效果？人工评估成本高，自动指标不可靠 |
| **Reward-KL 权衡** | 追求高 reward 可能导致偏离预训练分布太远 |
| **长度偏差** | RM 倾向给长回复高分，导致 RL 后回复越来越长 |
| **长序列 RL** | 长 chain-of-thought 的 credit assignment 困难 |
| **基础设施复杂** | 4 个模型协调、推理+训练混合调度、多框架协作 |
| **可扩展性** | 从 7B 到 70B+，RL 训练的工程复杂度指数增长 |

**当前研究方向：**
- GRPO/DAPO 简化 PPO 流程
- 规则化奖励（DeepSeek-R1）替代 RM
- Process Reward Model（过程奖励）替代 Outcome RM
- 异步 RL 训练（如 Ray + vLLM）提高效率

---

## 面试金句

> "RLHF 比 SFT 复杂得多，因为 PPO 同时需要维护 4 个模型（Actor, Critic, Reward, Reference），内存常态爆炸；而且 Actor 生成和模型参数更新底层逻辑冲突，通常需要把生成和训练拆分到两个框架。"

> "RL 训练不仅要优化训练引擎，更要重度优化推理引擎（Actor 生成经验）。把 vLLM 的生成能力和 DeepSpeed 的训练能力捏合在一个显存池里并且不冲突，工程难度极大。"

> "DPO 的核心洞察：最优策略可以从 RLHF 目标中推导出闭式解，从而将 RM + RL 两步合并为一步偏好优化。优势是简单稳定，劣势是 off-policy 缺乏探索。"

> "GRPO 用组内相对奖励替代 Critic Model，省去一个模型的同时保持了 on-policy 的探索能力，特别适合推理类任务（DeepSeek-R1 的核心方法）。"
