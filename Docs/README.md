# Docs 目录说明

本目录包含大模型 AI Infra 面试知识库的详细文档。

## 文档索引

### 核心知识点

| 文档 | 内容 |
|------|------|
| [DistributedTraining.md](./DistributedTraining.md) | 分布式训练基础：DP/TP/PP、DDP 原理、3D 并行策略 |
| [TrainingOptimization.md](./TrainingOptimization.md) | 训练效率优化：吞吐因素、GPU 利用率排查、梯度累积 |
| [CommunicationOptimization.md](./CommunicationOptimization.md) | 通信优化：原语对比、通信与计算重叠、多机训练挑战 |
| [MemoryOptimization.md](./MemoryOptimization.md) | 显存优化：ZeRO 详解、激活检查点、混合精度、框架对比 |
| [InferenceOptimization.md](./InferenceOptimization.md) | 推理优化：Continuous Batching、PagedAttention、PD 分离、投机解码、量化 |
| [InferenceInterviewReview.md](./InferenceInterviewReview.md) | 推理面试审查：题目质量、标准答案补充、结构建议 |
| [MultimodalTraining.md](./MultimodalTraining.md) | 多模态训练：挑战、视频模型优化、Omni 模型难点 |
| [RLTraining.md](./RLTraining.md) | SFT/RL 训练：三阶段差异、RLHF 复杂度分析 |
| [Troubleshooting.md](./Troubleshooting.md) | 排障实战：系统排查方法论、故障定位、工程价值观 |

### 面试专项

| 文档 | 内容 |
|------|------|
| [CodingProblems.md](./CodingProblems.md) | 手撕代码：Attention、CUDA、C++ 高并发、PagedAttention 分配器 |
| [InterviewTips.md](./InterviewTips.md) | 面试实战话术：常见问答模板、回答技巧 |
| [QuickInterviewAnswers.md](./QuickInterviewAnswers.md) | 30 秒口语答案：44 道高频题标准回答 |
| [SeniorInterviewQuestions.md](./SeniorInterviewQuestions.md) | 资深面试官追问：高级话题、系统设计、故障排查 |
| [NvidiaHPCInterview.md](./NvidiaHPCInterview.md) | NVIDIA/HPC 岗面经：C++、量化、CUDA、大模型理论 |
| [CompanyInterviews.md](./CompanyInterviews.md) | 各公司面经汇总：美团、混元、网易、快手、AI Infra 人才库 |
| [CompanyInterviewByCompany.md](./CompanyInterviewByCompany.md) | 按公司整理的面试总览：NVIDIA、美团、混元、网易、快手、AI Infra |

### 资源

| 文档 | 内容 |
|------|------|
| [Resources.md](./Resources.md) | 推荐资源：论文、开源框架、官方文档、技术博客 |

## 推荐阅读顺序

### 第一遍：建立知识框架
1. DistributedTraining.md - 理解分布式基础
2. MemoryOptimization.md - 显存优化核心
3. InferenceOptimization.md - 推理优化核心

### 第二遍：面试准备
4. QuickInterviewAnswers.md - 背诵 30 秒答案
5. InterviewTips.md - 学习回答技巧
6. CodingProblems.md - 准备手撕代码

### 第三遍：进阶提升
7. SeniorInterviewQuestions.md - 应对资深面试官追问
8. CompanyInterviews.md - 了解各公司面试风格
9. Troubleshooting.md - 提升排障能力

### 持续学习
10. Resources.md - 深入阅读论文和官方文档
