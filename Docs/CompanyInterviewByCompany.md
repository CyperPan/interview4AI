# 按公司整理的面试内容总览

> 文档定位：把仓库里已经出现的“公司面试”内容单独抽出来，按公司维度整理，方便快速查找和针对性准备。
>
> 使用方式：先看“公司总览”判断该公司重点考什么，再进入对应小节看高频题方向和原始文档。

---

## 公司总览

| 公司 / 方向 | 面试侧重点 | 适合优先准备的能力 |
| --- | --- | --- |
| **NVIDIA / HPC** | C++ 基础、CUDA、量化、FlashAttention、PagedAttention、GPU 性能优化 | CUDA 编程、kernel 视角、硬件理解 |
| **美团北斗 AI Infra** | Transformer 基础、GPU 基础、量化、PD 分离、PagedAttention、手撕代码 | 推理优化基础、工程表达、CUDA 手写 |
| **混元 AI Infra** | FP4 / 低比特量化、Roofline、GPU 性能分析 | 低比特量化、性能瓶颈判断 |
| **网易大模型应用开发** | 项目介绍、微调方式、推理加速、系统设计 | 项目表述、方案设计、产品落地 |
| **快手 AI 应用开发** | LangGraph、RAG、向量数据库、CoT、幻觉治理 | 应用链路设计、RAG 工程、Agent 工作流 |
| **AI Infra 人才库** | 训练时间估算、Megatron 通信、DeepSeek、NCCL、NVSHMEM | 分布式训练、通信优化、系统级理解 |

---

## NVIDIA / HPC

### 这类岗位通常怎么问

这类面试更偏底层系统和高性能计算，不会只停留在“大模型原理”。面试官通常会沿着 `C++ 基础 -> CUDA 访存与并行 -> 量化与 kernel 优化 -> LLM 推理机制` 逐层追问。

### 高频题方向

- C++ 工程基础：头文件、`static`、单例、类型转换、深浅拷贝
- 量化：GPTQ、AWQ、SmoothQuant、BF16 vs FP16、精度与速度权衡
- CUDA：shared memory bank conflict、显存分配、访存优化、计算优化
- 推理优化：FlashAttention、PagedAttention、算子融合、长序列性能

### 代表题目

- 为什么 C++ 项目需要头文件？
- SmoothQuant 为什么能缓解 INT8 精度下降？
- shared memory bank conflict 是怎么发生的？
- 如何优化 CUDA 程序的访存效率？
- FlashAttention 和 PagedAttention 的区别是什么？

### 准备建议

- 不要只会“背概念”，要能说清 `性能瓶颈在哪`、`硬件资源怎么用`、`优化后代价是什么`
- 量化题要能讲出 `为什么更快`，而不是只说“位宽更低”
- CUDA 题尽量能从寄存器、shared memory、HBM、coalesced access 这些层次往下讲

### 原始文档

- [NvidiaHPCInterview.md](./NvidiaHPCInterview.md)

---

## 美团北斗 AI Infra

### 这类岗位通常怎么问

这类题目更像“AI Infra 基础面”。会先看你是不是理解 Transformer、GPU 和推理优化，再通过手撕代码判断工程基本功。

### 高频题方向

- Transformer 架构与参数分布
- GPU 基础：CUDA Core、Tensor Core、常见卡型规格
- 量化思路与常见策略
- PD 分离、PagedAttention
- LeetCode 与 CUDA 手写题

### 代表题目

- Transformer 相比 RNN 的优势是什么？
- Transformer 中参数量最大的是哪一部分？计算量最大的是哪一部分？
- 常见大模型量化策略有哪些？
- PD 分离为什么有收益？
- 手写 CUDA prefix sum 或 GEMM

### 准备建议

- 基础题回答要稳定，不要在 Transformer 和 GPU 基础题上失分
- 推理优化题要能把 `prefill / decode` 和 `compute-bound / memory-bound` 区分清楚
- 手撕代码至少要准备一道链表题和一道 CUDA 基础题

### 原始文档

- [CompanyInterviews.md](./CompanyInterviews.md#美团北斗-ai-infra)

---

## 混元 AI Infra

### 这类岗位通常怎么问

更关注低比特量化和性能分析能力，题量不一定多，但问得会更集中、更偏“你能不能判断瓶颈”。

### 高频题方向

- FP4 / 低比特量化基础
- block-wise quantization、microscaling
- Roofline Model
- GPU 性能分析与瓶颈定位

### 代表题目

- FP4 的基本概念和典型实现方式是什么？
- FP8、INT8、INT4、FP4 分别适合什么场景？
- 用 Roofline Model 怎么定位系统瓶颈？
- 如何区分 compute-bound 和 memory-bound？

### 准备建议

- 量化题不要只说格式定义，要能回答 `为什么这个格式更适合某种硬件或目标`
- 性能分析题尽量从 `Arithmetic Intensity -> Roofline -> Profiler 验证` 这一条链路来答

### 原始文档

- [CompanyInterviews.md](./CompanyInterviews.md#混元-ai-infra)

---

## 网易大模型应用开发

### 这类岗位通常怎么问

更偏“做过什么项目，怎么把模型落地到业务里”。既看你会不会讲项目，也看你能不能把推理优化、微调和系统设计串起来。

### 高频题方向

- 项目经历介绍
- 微调方式对比：Full FT、LoRA、Adapter、P-Tuning
- 推理加速技术：量化、动态批处理、FlashAttention
- 系统设计：业务场景的 LLM 应用设计

### 代表题目

- 请详细介绍一个大模型相关项目
- LoRA、Adapter 和 Full Fine-tuning 怎么比较？
- 大模型推理加速有哪些常见方法？
- 如何设计一个 AI 歌词生成系统？

### 准备建议

- 项目题尽量用 `背景 -> 目标 -> 方案 -> 指标结果` 的结构讲
- 不要只讲模型，要补充部署、指标、吞吐、延迟和成本
- 系统设计题要同时覆盖数据、模型、推理、产品四层

### 原始文档

- [CompanyInterviews.md](./CompanyInterviews.md#网易大模型应用开发)

---

## 快手 AI 应用开发

### 这类岗位通常怎么问

这类岗位更偏应用层，重点是 Agent、RAG、向量检索和大模型产品问题，不会像 AI Infra 那样重 kernel 和 CUDA。

### 高频题方向

- LangChain 与 LangGraph
- RAG 文档切片策略
- 向量数据库索引：IVF_FLAT、HNSW
- CoT
- 幻觉类型与缓解方法

### 代表题目

- LangGraph 相比 LangChain 有什么优势？
- RAG 中文档切片粒度怎么定？
- IVF_FLAT 和 HNSW 的区别是什么？
- CoT 为什么有效？
- 常见幻觉有哪些，怎么缓解？

### 准备建议

- 回答时要偏“工程链路”，不是只解释名词
- RAG 题最好顺手带上 `召回率、延迟、存储成本、上下文完整性`
- 幻觉题要能区分事实性、忠实性和逻辑性问题

### 原始文档

- [CompanyInterviews.md](./CompanyInterviews.md#快手-ai-应用开发)

---

## AI Infra 人才库

### 这类岗位通常怎么问

这部分更像“偏研究和底层的 AI Infra 综合面”，容易出现训练估算、分布式通信、DeepSeek 架构优化、NCCL 和 NVSHMEM 这类系统题。

### 高频题方向

- 训练时间估算
- Megatron-LM 中的通信优化
- DeepSeek-V3 / DSA / NSA / MoBA
- NCCL 常见通信原语
- NVSHMEM 与 ring all-reduce 对比

### 代表题目

- 已知 token 总量，如何估算完整训练时间？
- Megatron-LM 如何做通信优化？
- DeepSeek-V3 的主要优化点有哪些？
- all-reduce 一次参数更新需要几次通信？
- NVSHMEM 在小数据量场景下有什么优势和代价？

### 准备建议

- 训练估算题要能写出公式，并解释每个变量的意义
- 通信题不要只背 API，要能说明使用场景和性能 trade-off
- 如果提到 DeepSeek，最好区分架构优化、注意力优化和并行调度

### 原始文档

- [CompanyInterviews.md](./CompanyInterviews.md#ai-infra-人才库面经)

---

## 推荐使用顺序

如果你是按公司准备面试，可以按这个顺序读：

1. 先看本页，判断该公司更偏底层、Infra 还是应用
2. 再进入对应原始文档，逐题准备
3. 如果公司更偏推理优化，补读 [InferenceOptimization.md](./InferenceOptimization.md)
4. 如果公司更偏口语快答，补读 [QuickInterviewAnswers.md](./QuickInterviewAnswers.md)

---

## 对照关系

为了避免重复阅读，这里给出最简对照：

- **底层 / CUDA / HPC 强相关**：优先看 `NVIDIA / HPC`
- **推理优化 + AI Infra 基础**：优先看 `美团北斗 AI Infra`
- **低比特量化 / 性能分析**：优先看 `混元 AI Infra`
- **项目落地 / 系统设计**：优先看 `网易大模型应用开发`
- **RAG / Agent / 应用开发**：优先看 `快手 AI 应用开发`
- **训练系统 / 通信 / 分布式**：优先看 `AI Infra 人才库`
