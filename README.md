# 📚 大模型 AI Infra 面试知识库

<p align="center">
  <a href="https://github.com/huihut/interview"><img src="https://img.shields.io/badge/Reference-huihut%2Finterview-blue.svg" alt="Reference"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="LICENSE"></a>
</p>

---

## 💡 关于

这个仓库先按 **LLM 推理主线** 和 **LLM 训练主线** 来组织。

如果你是第一次看这个仓库，建议先读：

1. [LLM 推理主线](#-llm-推理主线)
2. [LLM 训练主线](#-llm-训练主线)
3. [LLM 数学推导](#-llm-数学推导)
4. [公司面试与快答资料](#-公司面试与快答资料)

---

## 📋 导航

- [🚀 LLM 推理主线](#-llm-推理主线)
  - [总览](#总览)
  - [1. 请求进入与 Tokenization](#1-请求进入与-tokenization)
  - [2. 调度与 Batching](#2-调度与-batching)
  - [3. Prefill](#3-prefill)
  - [4. Attention 与 KV Cache](#4-attention-与-kv-cache)
  - [5. FFN / MoE](#5-ffn--moe)
  - [6. Decode](#6-decode)
  - [7. Speculative Decoding（可选）](#7-speculative-decoding可选)
  - [8. Sampling 与 Output](#8-sampling-与-output)
  - [9. 跨阶段优化与指标](#9-跨阶段优化与指标)
- [⚡ LLM 训练主线](#-llm-训练主线)
  - [总览](#总览-1)
  - [1. 数据准备与 Tokenization](#1-数据准备与-tokenization)
  - [2. Batch 组织与 DataLoader](#2-batch-组织与-dataloader)
  - [3. Forward](#3-forward)
  - [4. Loss 计算](#4-loss-计算)
  - [5. Backward](#5-backward)
  - [6. 梯度同步与并行通信](#6-梯度同步与并行通信)
  - [7. Optimizer Step 与参数更新](#7-optimizer-step-与参数更新)
  - [8. 显存与吞吐优化](#8-显存与吞吐优化)
  - [9. Checkpoint、Eval 与稳定性](#9-checkpointeval-与稳定性)
  - [10. SFT 与 RLHF 的流程差异](#10-sft-与-rlhf-的流程差异)
- [🧮 LLM 数学推导](#-llm-数学推导)
  - [Embedding 与输出投影](./Docs/LLMMathDerivations.md#1-embedding-与输出投影)
  - [Positional Encoding 与 RoPE](./Docs/LLMMathDerivations.md#2-positional-encoding-与-rope)
  - [Softmax 与 Cross Entropy](./Docs/LLMMathDerivations.md#3-softmax-与-cross-entropy)
  - [Scaled Dot-Product Attention](./Docs/LLMMathDerivations.md#4-scaled-dot-product-attention)
  - [FFN 与 SwiGLU](./Docs/LLMMathDerivations.md#6-ffn-与-swiglu)
  - [LayerNorm 与 RMSNorm](./Docs/LLMMathDerivations.md#7-layernorm-与-rmsnorm)
  - [Adam 与 AdamW](./Docs/LLMMathDerivations.md#11-adam-与-adamw)
- [📝 手撕代码与实现](#-手撕代码与实现)
- [💯 公司面试与快答资料](#-公司面试与快答资料)
- [📚 专题补充资料](#-专题补充资料)
- [🔗 推荐阅读顺序](#-推荐阅读顺序)

---

## 🚀 LLM 推理主线

### 总览

更合理的在线 LLM 推理链路是：

`请求进入 -> tokenization -> scheduler / batching -> prefill -> attention / KV cache -> FFN / MoE -> decode -> speculative decoding(可选) -> sampling / output`

对应主文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md)

---

### 1. 请求进入与 Tokenization

这一段重点理解：

- 为什么 tokenizer 也会影响 TTFT
- 为什么短请求里 CPU 侧开销不能忽略
- 为什么推理不是只看 GPU kernel

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-1请求进入与-tokenization)
- [SeniorInterviewQuestions.md](./Docs/SeniorInterviewQuestions.md#tokenizer-与输入链路)

---

### 2. 调度与 Batching

这一段重点理解：

- 静态 batch 和 continuous batching 的区别
- in-flight batching 的本质
- admission control、公平性、尾延迟

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-2调度与-batching)
- [InferenceOptimization.md](./Docs/InferenceOptimization.md#continuous-batching)
- [QuickInterviewAnswers.md](./Docs/QuickInterviewAnswers.md)

---

### 3. Prefill

这一段重点理解：

- prefill 为什么更偏 compute-bound
- chunked prefill 的作用
- prefix caching 为什么主要改善 TTFT

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-3prefill)
- [InferenceOptimization.md](./Docs/InferenceOptimization.md#pd-分离)
- [QuickInterviewAnswers.md](./Docs/QuickInterviewAnswers.md)

---

### 4. Attention 与 KV Cache

这一段重点理解：

- attention 为什么是 Transformer 核心
- KV cache 为什么是推理优化中心
- PagedAttention、MHA、MQA、GQA 各自解决什么问题
- FlashAttention 和 PagedAttention 的区别

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-4attention-与-kv-cache)
- [InferenceOptimization.md](./Docs/InferenceOptimization.md#pagedattention)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#4-scaled-dot-product-attention)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#5-multi-head-attention-与-gqa)

---

### 5. FFN / MoE

这一段重点理解：

- FFN 为什么常常是参数量大头
- MoE 在 Transformer block 里的真实位置
- 为什么 MoE 常常是“算力省了，但通信炸了”

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-5ffn--moe)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#6-ffn-与-swiglu)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#12-moe-router-与-top-k-gating)
- [CommunicationOptimization.md](./Docs/CommunicationOptimization.md)

---

### 6. Decode

这一段重点理解：

- decode 为什么更难优化
- 为什么 decode 常常是 memory-bound
- 为什么 PD 分离主要是为 decode 稳定性服务

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-6decode)
- [InferenceOptimization.md](./Docs/InferenceOptimization.md#pd-分离)
- [SeniorInterviewQuestions.md](./Docs/SeniorInterviewQuestions.md#线上故障与可观测性)

---

### 7. Speculative Decoding（可选）

这一段重点理解：

- speculative decoding 是什么
- acceptance rate 为什么关键
- 为什么它不是天然提升 TTFT 的工具

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-7speculative-decoding可选)
- [InferenceOptimization.md](./Docs/InferenceOptimization.md#投机解码)

---

### 8. Sampling 与 Output

这一段重点理解：

- temperature / top-k / top-p 的作用
- constrained decoding 为什么会变慢
- 流式输出为什么能显著改善用户体验

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-8sampling-与-output)
- [SeniorInterviewQuestions.md](./Docs/SeniorInterviewQuestions.md#structured-output--tool-use)

---

### 9. 跨阶段优化与指标

这一段重点理解：

- TTFT、TPOT / ITL、Throughput、Accuracy
- 量化、FlashAttention、profiling、benchmark
- 为什么 tokens/s 不能代表全部用户体验

对应文档：

- [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md#模块-9跨阶段优化与指标)
- [InferenceOptimization.md](./Docs/InferenceOptimization.md)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#13-参数量flops-与-kv-cache-估算)

---

## ⚡ LLM 训练主线

### 总览

更合理的训练主线是：

`数据准备 -> batch 组织 -> forward -> loss -> backward -> 梯度同步 / 并行通信 -> optimizer step -> 显存 / 吞吐优化 -> checkpoint / eval`

如果是 RLHF，还要插入：

`rollout -> reward / advantage -> 再进入训练更新`

对应主文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md)

---

### 1. 数据准备与 Tokenization

这一段重点理解：

- 数据长度分布为什么影响训练吞吐
- tokenizer 为什么会影响 attention 成本
- 为什么 SFT 比预训练更依赖样本组织

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-1数据准备与-tokenization)

---

### 2. Batch 组织与 DataLoader

这一段重点理解：

- GPU 吃不满时为什么先查 DataLoader
- sequence packing 的作用
- 梯度累积和等效 batch size 的关系

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-2batch-组织与-dataloader)
- [TrainingOptimization.md](./Docs/TrainingOptimization.md)

---

### 3. Forward

这一段重点理解：

- training forward 和 inference prefill 的相似点
- Transformer 里 attention 和 FFN 的参数 / 计算分工
- FlashAttention 为什么能同时影响速度和显存

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-3forward)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#9-transformer-block-的前向传播)

---

### 4. Loss 计算

这一段重点理解：

- 为什么训练不只看吞吐，还看收敛
- softmax + cross entropy 的数学意义
- loss scaling 为什么是混合精度训练的关键

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-4loss-计算)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#3-softmax-与-cross-entropy)

---

### 5. Backward

这一段重点理解：

- backward 为什么通常更贵
- activation checkpointing 为什么能省显存
- 为什么训练显存里激活值经常是大头

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-5backward)
- [MemoryOptimization.md](./Docs/MemoryOptimization.md)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#10-反向传播与链式法则)

---

### 6. 梯度同步与并行通信

这一段重点理解：

- DDP 梯度同步怎么做
- DP / TP / PP / ZeRO 各解决什么问题
- 通信为什么成为瓶颈
- overlap 和 AllReduce / AllGather / AllToAll 的作用

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-6梯度同步与并行通信)
- [DistributedTraining.md](./Docs/DistributedTraining.md)
- [CommunicationOptimization.md](./Docs/CommunicationOptimization.md)

---

### 7. Optimizer Step 与参数更新

这一段重点理解：

- Adam / AdamW 在做什么
- 为什么优化器状态会吃掉大量显存
- ZeRO / FSDP 为什么本质上是“用通信换显存”

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-7optimizer-step-与参数更新)
- [MemoryOptimization.md](./Docs/MemoryOptimization.md)
- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md#11-adam-与-adamw)

---

### 8. 显存与吞吐优化

这一段重点理解：

- compute-bound / memory-bound / communication-bound / input-bound
- mixed precision、checkpointing、梯度累积
- 吞吐和收敛之间的矛盾

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-8显存与吞吐优化)
- [TrainingOptimization.md](./Docs/TrainingOptimization.md)
- [MemoryOptimization.md](./Docs/MemoryOptimization.md)

---

### 9. Checkpoint、Eval 与稳定性

这一段重点理解：

- 为什么预训练强调长时间稳定运行
- checkpoint 为什么是训练系统核心设计点
- 为什么 benchmark 不能替代真实训练验证

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-9checkpointeval-与稳定性)
- [Troubleshooting.md](./Docs/Troubleshooting.md)

---

### 10. SFT 与 RLHF 的流程差异

这一段重点理解：

- pretrain / SFT / RLHF 各自系统重点不同
- RLHF 为什么是“推理链路 + 训练链路”
- rollout、reward、advantage、PPO 更新的整体关系

对应文档：

- [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md#模块-10sft-与-rlhf-的流程差异)
- [RLTraining.md](./Docs/RLTraining.md)

---

## 🧮 LLM 数学推导

如果你想从公式层面把 LLM 讲明白，直接看：

- [LLMMathDerivations.md](./Docs/LLMMathDerivations.md)

当前已经单独整理的数学部分包括：

- Embedding 与输出投影
- Positional Encoding 与 RoPE
- Softmax 与 Cross Entropy
- Scaled Dot-Product Attention
- Multi-Head Attention 与 GQA
- FFN 与 SwiGLU
- LayerNorm 与 RMSNorm
- Residual Connection
- Transformer Block 前向传播
- 反向传播与链式法则
- Adam 与 AdamW
- MoE Router 与 Top-k Gating
- 参数量、FLOPs 与 KV Cache 估算

---

## 📝 手撕代码与实现

如果你要准备代码题或现场手写：

- [CodingProblems.md](./Docs/CodingProblems.md)

覆盖内容：

- Attention / GQA
- CUDA Reduction / Softmax / RMSNorm
- C++ SPSC Ring Buffer / Thread Pool
- PagedAttention Block Allocator

---

## 💯 公司面试与快答资料

如果你需要按公司或者按“快答背诵”来准备：

- [QuickInterviewAnswers.md](./Docs/QuickInterviewAnswers.md)
- [SeniorInterviewQuestions.md](./Docs/SeniorInterviewQuestions.md)
- [NvidiaHPCInterview.md](./Docs/NvidiaHPCInterview.md)
- [CompanyInterviews.md](./Docs/CompanyInterviews.md)
- [CompanyInterviewByCompany.md](./Docs/CompanyInterviewByCompany.md)

---

## 📚 专题补充资料

如果你已经建立了流程视角，再回头看专题资料会更高效：

- [DistributedTraining.md](./Docs/DistributedTraining.md)
- [TrainingOptimization.md](./Docs/TrainingOptimization.md)
- [CommunicationOptimization.md](./Docs/CommunicationOptimization.md)
- [MemoryOptimization.md](./Docs/MemoryOptimization.md)
- [InferenceOptimization.md](./Docs/InferenceOptimization.md)
- [MultimodalTraining.md](./Docs/MultimodalTraining.md)
- [InterviewTips.md](./Docs/InterviewTips.md)
- [InferenceInterviewReview.md](./Docs/InferenceInterviewReview.md)
- [Resources.md](./Docs/Resources.md)

---

## 🔗 推荐阅读顺序

### 如果你主攻 LLM 推理面试

1. [InferenceInterviewByPipeline.md](./Docs/InferenceInterviewByPipeline.md)
2. [LLMMathDerivations.md](./Docs/LLMMathDerivations.md)
3. [InferenceOptimization.md](./Docs/InferenceOptimization.md)
4. [CodingProblems.md](./Docs/CodingProblems.md)
5. [QuickInterviewAnswers.md](./Docs/QuickInterviewAnswers.md)
6. [SeniorInterviewQuestions.md](./Docs/SeniorInterviewQuestions.md)

### 如果你主攻训练 / 分布式面试

1. [TrainingInterviewByPipeline.md](./Docs/TrainingInterviewByPipeline.md)
2. [LLMMathDerivations.md](./Docs/LLMMathDerivations.md)
3. [DistributedTraining.md](./Docs/DistributedTraining.md)
4. [CommunicationOptimization.md](./Docs/CommunicationOptimization.md)
5. [MemoryOptimization.md](./Docs/MemoryOptimization.md)
6. [RLTraining.md](./Docs/RLTraining.md)

### 如果你主攻公司面试准备

1. [CompanyInterviewByCompany.md](./Docs/CompanyInterviewByCompany.md)
2. [CompanyInterviews.md](./Docs/CompanyInterviews.md)
3. [NvidiaHPCInterview.md](./Docs/NvidiaHPCInterview.md)
4. [QuickInterviewAnswers.md](./Docs/QuickInterviewAnswers.md)
5. [SeniorInterviewQuestions.md](./Docs/SeniorInterviewQuestions.md)

---

本仓库遵循 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)（署名 - 非商业性使用 - 相同方式共享）协议。
