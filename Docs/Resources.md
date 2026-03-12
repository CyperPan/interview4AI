# 推荐资源

## 目录

- [论文](#论文)
- [开源框架](#开源框架)
- [官方文档](#官方文档)
- [技术博客](#技术博客)
- [在线课程](#在线课程)

---

## 论文

### 训练优化

| 论文 | 作者/机构 | 核心贡献 |
|-----|----------|---------|
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | Microsoft | ZeRO 系列，用通信换显存 |
| [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | NVIDIA | 张量并行和流水线并行 |
| [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | Stanford | IO-Aware 的 Attention 优化 |
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | Stanford | 进一步优化 |

### 推理优化

| 论文 | 作者/机构 | 核心贡献 |
|-----|----------|---------|
| [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) | NVIDIA | 3D 并行最佳实践 |
| [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | UC Berkeley | PagedAttention |
| [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670) | Duke/SJTU | PD 分离 |
| [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) | Princeton | 投机解码 |
| [SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification](https://arxiv.org/abs/2305.09781) | CMU | 投机推理 |

### 量化与压缩

| 论文 | 作者/机构 | 核心贡献 |
|-----|----------|---------|
| [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) | Meta | INT8 量化 |
| [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) | MIT | 激活平滑量化 |
| [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) | MIT | 保护显著权重 |
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | IST Austria | 训练后量化 |

---

## 开源框架

### 训练框架

| 框架 | 链接 | 适用场景 |
|-----|------|---------|
| **DeepSpeed** | https://github.com/microsoft/DeepSpeed | 微软分布式训练框架，ZeRO 优化 |
| **Megatron-LM** | https://github.com/NVIDIA/Megatron-LM | NVIDIA 大模型训练，3D 并行 |
| **FSDP** | PyTorch 原生 | PyTorch 原生分布式训练 |
| **Colossal-AI** | https://github.com/hpcaitech/Colossal-AI | 统一的大模型训练系统 |
| **Ray Train** | https://github.com/ray-project/ray | 分布式训练编排 |

### 推理框架

| 框架 | 链接 | 特点 |
|-----|------|------|
| **vLLM** | https://github.com/vllm-project/vllm | PagedAttention，Continuous Batching |
| **TensorRT-LLM** | https://github.com/NVIDIA/TensorRT-LLM | NVIDIA 推理优化，FP8 支持 |
| **Text Generation Inference** | https://github.com/huggingface/text-generation-inference | HuggingFace 推理服务 |
| **lmdeploy** | https://github.com/InternLM/lmdeploy | 国产推理框架，TurboMind 引擎 |
| **llama.cpp** | https://github.com/ggerganov/llama.cpp | 纯 C++ 实现，CPU/GPU 混合推理 |

### 编译器/优化工具

| 工具 | 链接 | 用途 |
|-----|------|------|
| **Triton** | https://github.com/openai/triton | Python 编写高性能 GPU Kernel |
| **TVM** | https://github.com/apache/tvm | 深度学习编译器 |
| **XLA** | TensorFlow/PyTorch 集成 | 线性代数编译器 |
| **ONNX Runtime** | https://github.com/microsoft/onnxruntime | 跨平台推理加速 |

---

## 官方文档

### NVIDIA 生态

| 资源 | 链接 | 内容 |
|-----|------|------|
| CUDA Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | CUDA 编程权威指南 |
| NCCL Documentation | https://docs.nvidia.com/deeplearning/nccl/ | 集合通信库 |
| cuBLAS/cuDNN | NVIDIA 开发者网站 | GPU 数学库 |
| Nsight Systems | NVIDIA 开发者网站 | 性能分析工具 |

### PyTorch 生态

| 资源 | 链接 | 内容 |
|-----|------|------|
| PyTorch Distributed | https://pytorch.org/tutorials/beginner/dist_overview.html | 分布式训练入门 |
| FSDP Tutorial | PyTorch 官方文档 | FSDP 详细教程 |
| PyTorch Profiler | PyTorch 官方文档 | 性能分析 |
| torch.compile | PyTorch 官方文档 | 图编译加速 |

---

## 技术博客

### 公司/机构博客

| 博客 | 链接 | 特点 |
|-----|------|------|
| NVIDIA Developer Blog | https://developer.nvidia.com/blog | GPU 优化权威 |
| PyTorch Blog | https://pytorch.org/blog/ | PyTorch 最新特性 |
| Microsoft Research | https://www.microsoft.com/en-us/research/research-area/artificial-intelligence/ | DeepSpeed 相关 |
| Google AI Blog | https://ai.googleblog.com/ | Transformer 家族 |
| EleutherAI Blog | https://www.eleuther.ai/ | 开源大模型研究 |

### 个人/社区博客

| 博主 | 链接/渠道 | 内容方向 |
|-----|----------|---------|
| **Lilian Weng** | https://lilianweng.github.io/ | OpenAI 研究员，深度学习理论 |
| **Tim Dettmers** | https://timdettmers.com/ | 量化、显卡选购指南 |
| **Papers with Code** | https://paperswithcode.com/ | 论文+代码 |
| **Hugging Face Blog** | https://huggingface.co/blog | 大模型应用 |

### 中文资源

| 来源 | 链接/渠道 | 内容方向 |
|-----|----------|---------|
| **苏剑林博客** | https://spaces.ac.cn/ | 科学空间，数学+算法 |
| **李沐** | B站/YouTube | 动手学深度学习 |
| **跟李沐学AI** | B站 | 论文精读系列 |
| **朱小厮** | 知乎/CSDN | 分布式系统 |

---

## 在线课程

### 免费课程

| 课程 | 平台 | 内容 |
|-----|------|------|
| **CS217: Parallel Computing** | Stanford | 并行计算基础 |
| **CUDA on NVIDIA GPUs** | NVIDIA | CUDA 编程 |
| **Deep Learning Specialization** | Coursera (Andrew Ng) | 深度学习基础 |
| **Introduction to Parallel Programming** | Udacity (NVIDIA) | 并行编程入门 |

### 付费/专业课程

| 课程 | 平台 | 内容 |
|-----|------|------|
| **大规模机器学习系统** | 各大在线教育平台 | 分布式 ML 系统设计 |
| **高性能 CUDA 编程** | NVIDIA DLI | CUDA 进阶 |

---

## 社区与论坛

| 社区 | 链接 | 用途 |
|-----|------|------|
| **GitHub Issues** | 各开源框架 | 问题排查 |
| **Stack Overflow** | stackoverflow.com | 编程问题 |
| **Reddit r/MachineLearning** | reddit.com/r/MachineLearning | 最新讨论 |
| **Discord (Hugging Face)** | discord.gg/hugging-face | 社区交流 |
| **Paper Reading Groups** | 各大公司/学校 | 论文讨论 |

---

## 工具推荐

### 性能分析

| 工具 | 用途 |
|-----|------|
| **Nsight Systems** | 全系统性能分析 |
| **Nsight Compute** | Kernel 级性能分析 |
| **PyTorch Profiler** | PyTorch 模型分析 |
| **TensorBoard** | 训练可视化 |
| **Weights & Biases** | 实验管理 |

### 开发环境

| 工具 | 用途 |
|-----|------|
| **nvidia-smi** | GPU 状态监控 |
| **ncu (NVIDIA Compute Profiler)** | CUDA Kernel 分析 |
| **htop/top** | CPU/内存监控 |
| **iostat** | IO 监控 |
| **perf** | Linux 性能分析 |

---

<p align="center">
  持续更新中，欢迎推荐优质资源！
</p>
