# LLM 数学推导（详细版）

> 文档定位：把 LLM 里最常见、最容易在面试中被追问的数学公式单独抽出来，并按“详细推导版”来写。每个模块都按四件事组织：
>
> 1. 公式是什么  
> 2. 在 LLM 中起什么作用  
> 3. 数学推导或直觉推导  
> 4. PyTorch 代码展示

---

这份文档默认不是背诵提纲，而是完整展开版。重点是把：

- 公式从哪里来
- 为什么这样设计
- 梯度和复杂度怎么推
- PyTorch 里对应哪段实现

全部放在一起。

---

## 目录

- [为什么要单独学 LLM 数学](#为什么要单独学-llm-数学)
- [1. Embedding 与输出投影](#1-embedding-与输出投影)
- [2. Positional Encoding 与 RoPE](#2-positional-encoding-与-rope)
- [3. Softmax 与 Cross Entropy](#3-softmax-与-cross-entropy)
- [4. Scaled Dot-Product Attention](#4-scaled-dot-product-attention)
- [5. Multi-Head Attention 与 GQA](#5-multi-head-attention-与-gqa)
- [6. FFN 与 SwiGLU](#6-ffn-与-swiglu)
- [7. LayerNorm 与 RMSNorm](#7-layernorm-与-rmsnorm)
- [8. Residual Connection](#8-residual-connection)
- [9. Transformer Block 的前向传播](#9-transformer-block-的前向传播)
- [10. 反向传播与链式法则](#10-反向传播与链式法则)
- [11. Adam 与 AdamW](#11-adam-与-adamw)
- [12. MoE Router 与 Top-k Gating](#12-moe-router-与-top-k-gating)
- [13. 参数量、FLOPs 与 KV Cache 估算](#13-参数量flops-与-kv-cache-估算)
- [14. 访存量、激活参数量与 DeepSeek 类结构手算](#14-访存量激活参数量与-deepseek-类结构手算)

---

## 为什么要单独学 LLM 数学

很多面试会问你：

- 注意力公式为什么要除以 `sqrt(d_k)`？
- FFN 为什么通常是参数量大头？
- LayerNorm 和 RMSNorm 到底差在哪？
- Adam 里的一阶、二阶矩估计是什么？
- 反向传播里梯度是怎么一路传回去的？
- RoPE 为什么能编码相对位置？
- MoE 为什么“算力省了，通信炸了”？

如果你只会背模块名，不会写公式，也不会说梯度和复杂度，面试官通常会继续深挖。所以这份文档的重点不是“数学炫技”，而是把 **能直接帮助你解释 LLM 行为的公式** 讲清楚。

---

## 1. Embedding 与输出投影

### 公式

给定词表大小 `V`、隐藏维度 `d_model`，Embedding 矩阵记作：

`E ∈ R^(V × d_model)`

如果输入 token id 为 `t`，它对应的向量就是：

`x = E[t]`

模型最后输出 logits 时，通常会做：

`logits = h W_out + b`

其中：

- `h ∈ R^(d_model)` 是最后一层 hidden state
- `W_out ∈ R^(d_model × V)`

很多 LLM 会做 **weight tying**，即：

`W_out = E^T`

### 在 LLM 中的作用

- Embedding 负责把离散 token id 映射到连续向量空间
- 输出投影负责把 hidden state 映射回词表概率空间
- weight tying 可以减少参数量，并让输入输出语义空间更一致

### 详细推导

Embedding 本质上不是“查字典”之外的更复杂操作，它就是一个 one-hot 向量和矩阵相乘：

如果 `e_t ∈ R^V` 是 token `t` 的 one-hot 表示，那么：

`x = e_t^T E`

因为 one-hot 只有第 `t` 个位置是 1，所以结果恰好就是 `E` 的第 `t` 行。

输出层也是类似的线性分类器。对每个词表项 `i`：

`logit_i = h · W_out[:, i]`

也就是说，logit 本质上是“当前隐藏状态和每个词向量方向的匹配程度”。

如果使用 weight tying：

`W_out = E^T`

那么：

`logit_i = h · E[i]`

这说明输出层其实是在问：当前 hidden state `h` 和词表中第 `i` 个词向量是否对齐。  
从优化角度看，这会把输入表征空间和输出分类空间绑在一起，通常能减少参数量，也常能带来更稳定的训练。

### PyTorch 代码

```python
import torch
import torch.nn as nn

vocab_size = 32000
d_model = 4096

embedding = nn.Embedding(vocab_size, d_model)
lm_head = nn.Linear(d_model, vocab_size, bias=False)

# weight tying
lm_head.weight = embedding.weight

token_ids = torch.tensor([[1, 42, 256]])
x = embedding(token_ids)          # (batch, seq, d_model)
h = x[:, -1, :]                   # 取最后一个 token 的 hidden state
logits = lm_head(h)               # (batch, vocab_size)
```

---

## 2. Positional Encoding 与 RoPE

### 公式

Transformer 本身对输入顺序不敏感，所以需要额外引入位置信息。

经典正弦位置编码：

`PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))`

`PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))`

RoPE（Rotary Positional Embedding）则把位置编码写成二维平面的旋转：

对一对特征 `(x_1, x_2)`，位置 `m` 的旋转结果为：

`[x_1', x_2'] = [x_1 cos θ_m - x_2 sin θ_m, x_1 sin θ_m + x_2 cos θ_m]`

### 在 LLM 中的作用

- 让模型知道“哪个 token 在前、哪个 token 在后”
- RoPE 特别适合 decoder-only LLM，因为它天然把相对位置信息编码进 attention 点积里

### 详细推导

正弦位置编码的关键性质是：

`PE(pos + k)` 可以由 `PE(pos)` 线性组合出来

这让模型更容易从绝对位置中恢复相对位移关系。

更具体一点，利用三角恒等式：

`sin(a + b) = sin a cos b + cos a sin b`

`cos(a + b) = cos a cos b - sin a sin b`

可以看到 `PE(pos + k)` 能由 `PE(pos)` 和只依赖偏移量 `k` 的系数组合出来。  
这就是“绝对位置编码里隐含相对位移信息”的来源。

RoPE 的核心更直接。若对 Query 和 Key 都做相同频率的旋转：

`q_m = R_m q`

`k_n = R_n k`

那么它们的点积满足：

`q_m^T k_n = q^T R_(n-m) k`

也就是说，attention 分数只和相对位置 `(n - m)` 有关，而不是绝对位置本身。这就是为什么 RoPE 特别适合长上下文和自回归建模。

若把二维旋转矩阵写出来：

`R_m = [[cos θ_m, -sin θ_m], [sin θ_m, cos θ_m]]`

那么：

`q_m^T k_n = q^T R_m^T R_n k`

因为旋转矩阵满足：

`R_m^T = R_(-m)`

所以：

`R_m^T R_n = R_(n-m)`

这一步就是 RoPE 最关键的数学结论：**相对位置被自然编码进了 Query-Key 点积里。**

### PyTorch 代码

```python
import torch

def apply_rope(x, cos, sin):
    # x: (batch, heads, seq, dim)
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    out = torch.stack([x_rot_even, x_rot_odd], dim=-1)
    return out.flatten(-2)
```

---

## 3. Softmax 与 Cross Entropy

### 公式

给定 logits `z_i`，softmax 定义为：

`p_i = exp(z_i) / Σ_j exp(z_j)`

若正确类别为 `y`，cross entropy loss 为：

`L = -log p_y`

代入 softmax 可得：

`L = -z_y + log Σ_j exp(z_j)`

### 在 LLM 中的作用

- softmax 把 logits 变成词表上的概率分布
- cross entropy 是语言模型训练中最常见的目标函数

### 详细推导

cross entropy 的一个极重要结果是它对 logits 的梯度非常简洁：

`∂L / ∂z_i = p_i - 1(i = y)`

其中 `1(i = y)` 是 one-hot 标签。

这个结果意味着：

- 对正确类别，梯度是 `p_y - 1`
- 对错误类别，梯度是 `p_i`

也就是说，模型会自动“压低错误类别，抬高正确类别”。

这个梯度之所以重要，是因为它让 softmax + cross entropy 的反向传播既稳定又高效。

下面把这个梯度推出来。

先写：

`p_i = exp(z_i) / Σ_j exp(z_j)`

loss 为：

`L = -log p_y = -z_y + log Σ_j exp(z_j)`

对任意 `z_i` 求导：

`∂L / ∂z_i = ∂(-z_y) / ∂z_i + ∂ log Σ_j exp(z_j) / ∂z_i`

第一项：

- 当 `i = y` 时是 `-1`
- 当 `i != y` 时是 `0`

也就是：

`∂(-z_y) / ∂z_i = -1(i = y)`

第二项：

`∂ log Σ_j exp(z_j) / ∂z_i = exp(z_i) / Σ_j exp(z_j) = p_i`

所以最终得到：

`∂L / ∂z_i = p_i - 1(i = y)`

这也是为什么在实现里经常把 `softmax + nll_loss` 融合成一个 kernel：  
前向和反向公式都非常规整，数值稳定版本也容易统一处理。

### PyTorch 代码

```python
import torch
import torch.nn.functional as F

logits = torch.randn(2, 5, requires_grad=True)
targets = torch.tensor([1, 3])

loss = F.cross_entropy(logits, targets)
loss.backward()

print(loss.item())
print(logits.grad)  # 梯度大致对应 p - y_one_hot
```

---

## 4. Scaled Dot-Product Attention

### 公式

注意力的核心公式：

`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`

其中：

- `Q ∈ R^(n × d_k)`
- `K ∈ R^(n × d_k)`
- `V ∈ R^(n × d_v)`

### 在 LLM 中的作用

- 让当前 token 能看到历史 token 的信息
- 是 Transformer 能建模长距离依赖的核心

### 详细推导

未缩放时，`QK^T` 的每个元素是长度为 `d_k` 的点积。

如果 `q_i, k_i` 独立且方差为 1，那么：

`Var(q · k) = d_k`

也就是说，点积的方差会随着维度线性增长。`d_k` 一大，softmax 输入就容易过大，导致分布过尖、梯度变差。

所以要除以：

`sqrt(d_k)`

使得点积的量级更稳定。

把这件事写得更严格一些。

设：

- `q = (q_1, ..., q_d)`
- `k = (k_1, ..., k_d)`

并假设每个分量独立、零均值、单位方差：

`E[q_i] = E[k_i] = 0`

`Var(q_i) = Var(k_i) = 1`

则点积：

`s = q · k = Σ_i q_i k_i`

因为独立且零均值：

`E[s] = 0`

又因为：

`Var(q_i k_i) = E[q_i^2] E[k_i^2] = 1`

所以：

`Var(s) = Σ_i Var(q_i k_i) = d_k`

这意味着 `s` 的典型量级会随着 `sqrt(d_k)` 增长。  
如果直接把这样的 `s` 丢进 softmax，当 `d_k` 很大时，softmax 会快速饱和，注意力分布接近 one-hot，梯度变得很差。

将其缩放成：

`s' = s / sqrt(d_k)`

后就有：

`Var(s') = 1`

因此分数分布在不同 head_dim 下更稳定。

### PyTorch 代码

```python
import math
import torch

def scaled_dot_product_attention(q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    return out, probs
```

---

## 5. Multi-Head Attention 与 GQA

### 公式

多头注意力：

`head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)`

`MHA(Q, K, V) = Concat(head_1, ..., head_h) W^O`

GQA（Grouped Query Attention）中，多个 Query 头共享较少的 KV 头。

### 在 LLM 中的作用

- MHA 让模型从不同子空间学习不同关系
- GQA / MQA 主要是为了减少 KV Cache 体积，降低 decode 带宽压力

### 详细推导

MHA 的本质不是“把 attention 做很多遍”这么简单，而是：

- 每个头的投影矩阵不同
- 所以每个头在不同的表示子空间中做匹配

GQA 的关键计算收益来自 KV cache 大小：

若 attention 头数为 `h_q`，KV 头数为 `h_kv`，则 KV cache 大小大致与 `h_kv` 成正比。

所以当 `h_kv << h_q` 时，decode 的访存开销会显著下降。

更具体地，如果每层 KV cache 大小近似为：

`KV_bytes_per_layer ≈ seq_len * h_kv * d_head * 2 * bytes_per_elem`

那么：

- 对 MHA，`h_kv = h_q`
- 对 MQA，`h_kv = 1`
- 对 GQA，`1 < h_kv < h_q`

因此 GQA / MQA 的收益不是来自“attention 算法公式变了”，而是来自 **K/V 存储和读取量变小了**。  
这也是为什么它们对 decode 更重要，而不是对 prefill 一样重要。

### PyTorch 代码

```python
import torch
import torch.nn as nn

class GQAProjection(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v
```

---

## 6. FFN 与 SwiGLU

### 公式

标准 FFN：

`FFN(x) = W_2 σ(W_1 x + b_1) + b_2`

常见激活函数可以是 ReLU、GELU。

SwiGLU 常写成：

`SwiGLU(x) = (x W_a) ⊙ swish(x W_b)`

其中：

`swish(t) = t * sigmoid(t)`

### 在 LLM 中的作用

- FFN 负责逐 token 的非线性变换
- 在很多 LLM 中，FFN 参数量通常比 attention 还大
- SwiGLU 往往比普通 ReLU / GELU FFN 表达能力更强

### 详细推导

FFN 可以理解为“逐 token 的 MLP”。attention 负责信息混合，FFN 负责在每个 token 的特征维度上做非线性变换。

SwiGLU 的关键是门控：

- 一路做特征变换
- 一路做门控权重
- 两路逐元素相乘

这比单一路径激活函数更灵活，因此很多现代 LLM 采用 `SwiGLU / GeGLU / GLU` 变体。

参数量为什么 FFN 常是大头？

设隐藏维度为 `d`，中间维度为 `d_ff`。  
标准 FFN 两层线性层参数量近似为：

`params_FFN ≈ d * d_ff + d_ff * d = 2 d d_ff`

如果 `d_ff = 4d`，则：

`params_FFN ≈ 8 d^2`

而 attention 的 Q/K/V/O 四个投影总参数量大致为：

`params_attn ≈ 4 d^2`

所以在很多 Transformer 里，FFN 参数量约为 attention 的两倍量级。  
这也是“FFN 往往是参数量大头”的数学来源。

### PyTorch 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w_a = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_b = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_out = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.w_b(x))
        value = self.w_a(x)
        return self.w_out(value * gate)
```

---

## 7. LayerNorm 与 RMSNorm

### 公式

LayerNorm：

`μ = (1 / d) Σ_i x_i`

`σ^2 = (1 / d) Σ_i (x_i - μ)^2`

`LN(x)_i = γ_i * (x_i - μ) / sqrt(σ^2 + ε) + β_i`

RMSNorm：

`RMS(x) = sqrt((1 / d) Σ_i x_i^2 + ε)`

`RMSNorm(x)_i = γ_i * x_i / RMS(x)`

### 在 LLM 中的作用

- 归一化有助于稳定训练
- LayerNorm 更完整，RMSNorm 更轻量
- 很多现代 LLM 更偏向 RMSNorm

### 详细推导

LayerNorm 做了两件事：

1. 减均值
2. 除标准差

RMSNorm 只保留第二步的“按尺度归一化”思想，不减均值。

在很多 LLM 场景里，减均值并不是最关键的，真正重要的是控制激活尺度，所以 RMSNorm 常能用更简单的计算达到相近效果。

还可以把 LayerNorm 的方差写开：

`σ^2 = E[x^2] - (E[x])^2`

而 RMSNorm 用的是：

`RMS(x)^2 = E[x^2]`

也就是说，RMSNorm 省掉的是 `(E[x])^2` 这部分居中操作。  
如果模型训练过程中，真正更关键的是“防止激活尺度失控”，而不是“强制零均值”，那么 RMSNorm 就可能足够用了。

从实现角度看，RMSNorm：

- 少一次均值减法
- 少一次方差中心化
- kernel 更简单

这也是它在 LLM 中流行的工程原因。

### PyTorch 代码

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

---

## 8. Residual Connection

### 公式

残差连接最常见的形式：

`y = x + F(x)`

### 在 LLM 中的作用

- 帮助深层网络训练
- 让梯度更容易传播
- 避免层数一深就退化

### 数学推导 / 直觉

看梯度：

`∂y / ∂x = I + ∂F(x) / ∂x`

即使 `∂F/∂x` 很小，恒等映射 `I` 仍然给梯度保留了一条“直通路径”。这就是为什么残差连接对深层 Transformer 非常关键。

### PyTorch 代码

```python
def residual_block(x, sublayer):
    return x + sublayer(x)
```

---

## 9. Transformer Block 的前向传播

### 公式

以 Pre-Norm Transformer Block 为例：

`h_1 = x + Attention(Norm_1(x))`

`h_2 = h_1 + FFN(Norm_2(h_1))`

### 在 LLM 中的作用

- 这是现代 decoder-only LLM 最常见的基本骨架
- 把 attention、FFN、norm、residual 串成完整前向路径

### 详细推导

Transformer block 的关键不是某一个单独公式，而是：

- norm 控尺度
- attention 做 token 间信息混合
- FFN 做 token 内非线性变换
- residual 保证深层稳定性

如果写成 decoder-only pre-norm block 的更完整形式：

`u = Norm_1(x)`

`a = Attention(u)`

`h = x + a`

`v = Norm_2(h)`

`f = FFN(v)`

`y = h + f`

这样拆开看更清楚：

- `Norm_1 / Norm_2` 负责把输入尺度拉回稳定区间
- `Attention` 负责 token 间信息交换
- `FFN` 负责逐 token 特征变换
- 两次 residual 确保梯度路径不断裂

### PyTorch 代码

```python
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim, attn, ffn, norm_cls=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_cls(dim)
        self.norm2 = norm_cls(dim)
        self.attn = attn
        self.ffn = ffn

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## 10. 反向传播与链式法则

### 公式

链式法则：

若 `z = f(y)`，`y = g(x)`，则

`∂z / ∂x = (∂z / ∂y) (∂y / ∂x)`

以线性层为例：

`y = xW + b`

若上游梯度为 `G = ∂L / ∂y`，则：

`∂L / ∂W = x^T G`

`∂L / ∂x = G W^T`

### 在 LLM 中的作用

- backward 就是把 loss 的梯度沿计算图一层层传回去
- 训练的所有参数更新都依赖这些梯度

### 详细推导

这部分最容易被问的不是“会不会推一大页矩阵求导”，而是：

1. 你知不知道梯度是怎么传回去的
2. 你能不能说清哪些量需要缓存给 backward
3. 你是否理解为什么反向传播显存开销大

例如线性层：

`y_j = Σ_i x_i W_ij + b_j`

所以：

`∂L / ∂W_ij = x_i * ∂L / ∂y_j`

写成矩阵形式就是：

`∂L / ∂W = x^T G`

如果把 batch 维一起考虑，设：

- `X ∈ R^(B × d_in)`
- `W ∈ R^(d_in × d_out)`
- `Y = XW`
- `G = ∂L / ∂Y ∈ R^(B × d_out)`

则：

`∂L / ∂W = X^T G`

`∂L / ∂X = G W^T`

这两个式子非常重要，因为它解释了两件事：

1. backward 需要拿到前向输入 `X`，所以前向中间结果要缓存
2. 梯度计算本身也是矩阵乘法，所以 backward 往往同样是高代价算子

这也是为什么训练时显存不仅存参数，还要存激活。

### PyTorch 代码

```python
import torch

x = torch.randn(2, 4, requires_grad=True)
linear = torch.nn.Linear(4, 3)
target = torch.randn(2, 3)

out = linear(x)
loss = ((out - target) ** 2).mean()
loss.backward()

print(linear.weight.grad.shape)  # (3, 4)
print(x.grad.shape)              # (2, 4)
```

---

## 11. Adam 与 AdamW

### 公式

给定梯度 `g_t`：

一阶矩估计：

`m_t = β_1 m_(t-1) + (1 - β_1) g_t`

二阶矩估计：

`v_t = β_2 v_(t-1) + (1 - β_2) g_t^2`

偏差修正：

`m̂_t = m_t / (1 - β_1^t)`

`v̂_t = v_t / (1 - β_2^t)`

参数更新：

`θ_t = θ_(t-1) - α * m̂_t / (sqrt(v̂_t) + ε)`

AdamW 会把 weight decay 和梯度更新解耦：

`θ_t = θ_(t-1) - α * m̂_t / (sqrt(v̂_t) + ε) - α λ θ_(t-1)`

### 在 LLM 中的作用

- Adam / AdamW 是训练 LLM 最常见的优化器之一
- 它能对不同参数维度自适应调整学习率
- AdamW 的 decoupled weight decay 在大模型训练中更常用

### 详细推导

Adam 里的两个统计量：

- `m_t` 类似“梯度的滑动平均”，让更新方向更平滑
- `v_t` 类似“梯度平方的滑动平均”，用来估计每个参数方向上的梯度尺度

所以 Adam 的更新可以理解为：

“方向上参考一阶动量，步长上参考历史波动做自适应缩放”

偏差修正则是因为 `m_t, v_t` 在训练初期从 0 开始，会系统性偏小。

把偏差修正写开更清楚。

由于初始时 `m_0 = 0`，递推展开：

`m_t = (1 - β_1) Σ_(i=1)^t β_1^(t-i) g_i`

如果梯度在统计意义上近似平稳，`E[g_i] = μ`，则：

`E[m_t] = (1 - β_1) Σ_(i=1)^t β_1^(t-i) μ = (1 - β_1^t) μ`

所以 `m_t` 比真实均值 `μ` 少了一个因子 `(1 - β_1^t)``，训练初期会偏小。  
因此要除以这个因子，得到：

`m̂_t = m_t / (1 - β_1^t)`

`v_t` 的偏差修正同理。

为什么 AdamW 比 Adam 更常见？

因为传统 Adam 如果把 L2 正则直接混进梯度，会和自适应学习率缩放耦合在一起；AdamW 则把 weight decay 单独拆开：

`θ <- θ - α * update - α λ θ`

这样正则项不会被二阶矩缩放干扰，通常更符合“参数衰减”的原始意图。

### PyTorch 代码

```python
import torch
import torch.nn as nn

model = nn.Linear(16, 8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

x = torch.randn(4, 16)
y = torch.randn(4, 8)

pred = model(x)
loss = ((pred - y) ** 2).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 12. MoE Router 与 Top-k Gating

### 公式

设输入 token hidden state 为 `x`，router logits 为：

`r = W_r x`

router 概率：

`p = softmax(r)`

若采用 top-k routing，则只保留概率最大的 `k` 个 expert。

### 在 LLM 中的作用

- 决定每个 token 该送到哪些 expert
- 让模型在参数总量很大时，单次前向仍只激活少数 expert

### 详细推导

MoE 的核心不是“参数很多”，而是“激活稀疏”：

- 总参数量可以很大
- 但每个 token 只走少数 expert

这样单 token 的理论计算量更低，但代价是 token 分发会引入额外通信和负载均衡问题。

若一共有 `N` 个 expert，每个 token 只选 `k` 个 expert，则每个 token 的 FFN 计算量大致从：

`O(N * FFN_cost)` 的“全激活”

变成：

`O(k * FFN_cost)`

其中通常 `k << N`。

但系统侧的新问题是：

1. router 必须先算出每个 token 去哪
2. token 要被重新分发到对应 expert
3. 不同 expert 负载可能严重不均

所以 MoE 节省的是 **理论算力**，未必直接节省 **端到端时间**。

### PyTorch 代码

```python
import torch
import torch.nn.functional as F

def topk_router(x, router_weight, k=2):
    # x: (batch, seq, dim)
    logits = torch.matmul(x, router_weight)          # (batch, seq, n_experts)
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
    return topk_probs, topk_idx
```

---

## 13. 参数量、FLOPs 与 KV Cache 估算

### 公式

#### 参数量

一个线性层 `W ∈ R^(d_in × d_out)` 的参数量约为：

`params = d_in * d_out`

#### Attention 计算复杂度

attention 分数矩阵 `QK^T` 的复杂度大致为：

`O(n^2 d)`

其中：

- `n` 是序列长度
- `d` 是 head_dim 或隐藏维度相关量

#### FFN 复杂度

FFN 通常是两个大矩阵乘法，复杂度近似：

`O(n d d_ff)`

#### KV Cache 大小

`KV_bytes ≈ batch * seq_len * num_layers * num_kv_heads * head_dim * 2 * bytes_per_elem`

### 在 LLM 中的作用

- 帮你判断哪一部分是参数大头
- 帮你估算长上下文显存压力
- 帮你解释为什么 prefill 和 decode 的瓶颈不一样

### 详细推导

attention 的 `n^2` 来自哪里？

因为长度为 `n` 的序列中，每个 token 都要和另外 `n` 个 token 做匹配，构成一个 `n × n` 的注意力分数矩阵。

为什么 decode 常常更 memory-bound？

因为 decode 每一步新增的计算很少，但要不断读取庞大的权重和 KV Cache。也就是说，它不是“算不完”，而是“搬不动”。

再把 prefill 和 decode 的差别写得更数学一点：

- prefill 处理整段长度为 `n` 的序列，attention 会形成 `n × n` 分数矩阵
- decode 每一步只新增 1 个 token，对已有 `n` 个历史 token 做注意力

所以：

- prefill 的 attention 计算更像大矩阵乘法，算力利用率高
- decode 的 attention 更像“读很多历史 KV，再做较小计算”，更容易受带宽限制

KV cache 公式：

`KV_bytes ≈ batch * seq_len * num_layers * num_kv_heads * head_dim * 2 * bytes_per_elem`

中：

- `batch * seq_len` 决定 token 总数
- `num_layers` 表示每层都要存 K 和 V
- `num_kv_heads * head_dim` 是每个 token 每层的 KV 向量大小
- `2` 对应 K 和 V 两份缓存

这也是为什么长上下文时，KV cache 会迅速成为显存主压力之一。

### PyTorch 代码

```python
def estimate_kv_cache_bytes(
    batch,
    seq_len,
    num_layers,
    num_kv_heads,
    head_dim,
    bytes_per_elem=2,
):
    return (
        batch
        * seq_len
        * num_layers
        * num_kv_heads
        * head_dim
        * 2
        * bytes_per_elem
    )

bytes_used = estimate_kv_cache_bytes(
    batch=8,
    seq_len=4096,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
)

print(bytes_used / (1024 ** 3), "GB")
```

---

## 14. 访存量、激活参数量与 DeepSeek 类结构手算

### 公式

#### 标准 MHA / GQA 的 decode 侧 KV 读取

对单个生成步、单个样本、单层来说，历史 KV 的读取量可近似写成：

`KV_read_bytes_per_step_per_layer ≈ seq_len * 2 * num_kv_heads * head_dim * bytes_per_elem`

把层数和 batch 带上：

`KV_read_bytes_per_step ≈ batch * num_layers * seq_len * 2 * num_kv_heads * head_dim * bytes_per_elem`

#### MLA 的缓存量

如果系统缓存的是更紧凑的 latent 表示，而不是完整 K/V，那么单层单 token 的缓存量更接近：

`MLA_cache_bytes_per_token_per_layer ≈ d_latent * bytes_per_elem`

相比标准 KV：

`MHA_cache_bytes_per_token_per_layer ≈ 2 * num_kv_heads * head_dim * bytes_per_elem`

#### Linear Attention 的状态量

若使用特征映射后的线性 attention，并维护前缀状态：

`S_t = Σ_{i<=t} phi(k_i) v_i^T`

`z_t = Σ_{i<=t} phi(k_i)`

则每层缓存状态大小近似为：

`Linear_state_bytes_per_layer ≈ num_heads * (d_phi * d_v + d_phi) * bytes_per_elem`

它随序列长度增长得更慢，甚至可以做到对 `seq_len` 不敏感。

#### MoE 的总参数量与激活参数量

若一个 MoE 层有 `N` 个 expert，每个 expert 参数量为 `P_expert`，共享部分参数量为 `P_shared`，router 参数量为 `P_router`，则：

`P_total = P_shared + P_router + N * P_expert`

若每个 token 只激活 `k` 个 expert，则单 token 激活参数量近似：

`P_active = P_shared + P_router + k * P_expert`

#### DeepSeek 类 `MLA + MoE` 结构的单步 decode 读取

把它写成最通用的面试模板：

`Bytes_decode_step ≈ Bytes_dense_weight + Bytes_active_expert_weight + Bytes_cache_read + Bytes_comm`

其中：

- `Bytes_dense_weight`：主干 dense 部分权重读取
- `Bytes_active_expert_weight`：本步被激活 expert 的权重读取
- `Bytes_cache_read`：历史 attention 状态读取，`MHA / GQA / MLA / linear attention` 形式不同
- `Bytes_comm`：多卡 MoE token 分发或聚合通信

### 在 LLM 中的作用

- 帮你区分“模型总参数很大”和“单步真正读了多少”不是一回事
- 帮你解释为什么 `MoE` 看起来参数很大，但单 token 激活的只是其中一小部分
- 帮你解释为什么 `MLA / GQA / linear attention` 的价值主要体现在缓存和带宽，而不是只看理论 FLOPs
- 帮你在面试里用统一框架手算 `DeepSeek-V3` 这类结构，而不是死记一个公开数字

### 详细推导

很多人手算时最容易把三件事混在一起：

1. **总参数量**
2. **单 token 激活参数量**
3. **单步 decode 访存量**

这三者必须先拆开。

先看标准 `MHA`。decode 第 `t` 步时，新 token 需要和前 `t` 个历史 token 的 `K/V` 做注意力，所以单层至少要把历史 `K` 和 `V` 各读一遍。于是有：

`KV_read_bytes_per_step_per_layer ≈ t * 2 * num_kv_heads * head_dim * bytes_per_elem`

当 `t` 很大时，这个量会线性增长。也就是说，长上下文里 decode 慢，很多时候不是算不动，而是历史 `KV` 越读越多。

`GQA / MQA` 为什么有效？因为它们直接把 `num_kv_heads` 变小了。公式里别的量不变，只有 `num_kv_heads` 缩小，所以缓存和读取量会按比例下降。

`MLA` 再进一步。它的关键不是“把 attention 换成另一个公式”，而是让缓存更接近一个低维 latent 表示。于是每个 token、每层缓存的不是完整 `K/V`，而是更小的 latent。手算时最重要的是比较：

`2 * num_kv_heads * head_dim`

和

`d_latent`

谁更大。只要 `d_latent` 显著更小，长上下文缓存和读取压力就会明显下降。

`Linear attention` 又是另一个方向。它不再显式保存所有历史 token 的 `K/V`，而是把历史压缩成前缀状态 `S_t` 和 `z_t`。所以它的状态量更像：

`num_heads * (d_phi * d_v + d_phi)`

而不再是：

`seq_len * 2 * num_kv_heads * head_dim`

这也是为什么 linear attention 在超长序列题里经常被拿来和标准 attention 对比。

再看 `MoE`。很多人一看到 expert 数量大，就直接说“推理一定更慢”，这是不对的。因为总参数量：

`P_total = P_shared + P_router + N * P_expert`

但单 token 激活的通常只有：

`P_active = P_shared + P_router + k * P_expert`

其中 `k << N`。  
所以 `MoE` 的问题不只是“算了多少”，而是“这些被选中的 expert 在哪里、要不要跨卡取、通信是不是成为瓶颈”。

把这些合起来，`DeepSeek-V3` 这类 `MLA + MoE` 结构在面试里最稳的手算法就是：

1. 先估主干 dense 部分参数和每步权重读取
2. 再估 `MLA` 相比标准 `MHA/GQA` 少掉多少缓存和读取
3. 再估 `MoE` 的总 expert 参数量，以及每步只激活 `k` 个 expert 时的读取量
4. 最后补一个系统项：如果 expert 分布在多卡，还要加上 `AllToAll` 或类似 token 分发通信

这样你的答案就会从“背模型名词”变成“能把结构翻译成带宽、显存和通信”。

### PyTorch 代码

```python
def estimate_mha_kv_read_bytes(
    batch,
    seq_len,
    num_layers,
    num_kv_heads,
    head_dim,
    bytes_per_elem=2,
):
    return (
        batch
        * num_layers
        * seq_len
        * 2
        * num_kv_heads
        * head_dim
        * bytes_per_elem
    )


def estimate_mla_cache_bytes(
    batch,
    seq_len,
    num_layers,
    d_latent,
    bytes_per_elem=2,
):
    return batch * seq_len * num_layers * d_latent * bytes_per_elem


def estimate_linear_attention_state_bytes(
    batch,
    num_layers,
    num_heads,
    d_phi,
    d_v,
    bytes_per_elem=2,
):
    return (
        batch
        * num_layers
        * num_heads
        * (d_phi * d_v + d_phi)
        * bytes_per_elem
    )


def estimate_moe_params(p_shared, p_router, p_expert, n_experts, top_k):
    p_total = p_shared + p_router + n_experts * p_expert
    p_active = p_shared + p_router + top_k * p_expert
    return p_total, p_active


def estimate_deepseek_like_decode_bytes(
    dense_weight_bytes,
    active_expert_weight_bytes,
    cache_read_bytes,
    comm_bytes=0,
):
    return (
        dense_weight_bytes
        + active_expert_weight_bytes
        + cache_read_bytes
        + comm_bytes
    )


mha_kv = estimate_mha_kv_read_bytes(
    batch=1,
    seq_len=8192,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
)

mla_cache = estimate_mla_cache_bytes(
    batch=1,
    seq_len=8192,
    num_layers=32,
    d_latent=512,
)

moe_total, moe_active = estimate_moe_params(
    p_shared=400_000_000,
    p_router=10_000_000,
    p_expert=120_000_000,
    n_experts=64,
    top_k=2,
)

print("MHA decode KV read (GB):", mha_kv / (1024 ** 3))
print("MLA cache (GB):", mla_cache / (1024 ** 3))
print("MoE total params:", moe_total)
print("MoE active params per token:", moe_active)
```

---

## 推荐使用方式

如果你是为面试准备这份文档，建议按下面顺序读：

1. 先读 `3, 4, 6, 7, 11`
2. 再读 `2, 5, 9, 10`
3. 最后补 `12, 13`

因为：

- `softmax / attention / FFN / norm / Adam` 是最高频
- `RoPE / GQA / backward` 是常见追问
- `MoE / FLOPs / KV cache` 是资深面试官加深题

---

## 和现有文档的关系

- 如果你想看面试快答，读 [QuickInterviewAnswers.md](./QuickInterviewAnswers.md)
- 如果你想看推理链路，读 [InferenceInterviewByPipeline.md](./InferenceInterviewByPipeline.md)
- 如果你想看训练链路，读 [TrainingInterviewByPipeline.md](./TrainingInterviewByPipeline.md)
- 如果你想看代码实现，读 [CodingProblems.md](./CodingProblems.md)
