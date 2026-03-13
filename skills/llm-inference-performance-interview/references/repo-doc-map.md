# Repo Doc Map

Use this file to decide which existing repository docs to load next. Read only the topic that matches the user request.

## Core Topics

- `Transformer inference, continuous batching, PagedAttention, PD separation, speculative decoding, quantization`
  - Read [Docs/InferenceOptimization.md](../../../Docs/InferenceOptimization.md)
- `DDP, TP, PP, 3D parallelism, distributed systems basics`
  - Read [Docs/DistributedTraining.md](../../../Docs/DistributedTraining.md)
- `GPU utilization, throughput tuning, overlap, profiling mindset`
  - Read [Docs/TrainingOptimization.md](../../../Docs/TrainingOptimization.md)
- `collectives, NVLink vs RDMA, communication overlap`
  - Read [Docs/CommunicationOptimization.md](../../../Docs/CommunicationOptimization.md)
- `ZeRO, activation checkpointing, mixed precision, memory tradeoffs`
  - Read [Docs/MemoryOptimization.md](../../../Docs/MemoryOptimization.md)

## Interview-Specific Materials

- `high-frequency concise answers`
  - Read [Docs/QuickInterviewAnswers.md](../../../Docs/QuickInterviewAnswers.md)
- `behavioral framing and answer polish`
  - Read [Docs/InterviewTips.md](../../../Docs/InterviewTips.md)
- `senior-level probing, system design, troubleshooting`
  - Read [Docs/SeniorInterviewQuestions.md](../../../Docs/SeniorInterviewQuestions.md)
- `NVIDIA or HPC flavored questions, CUDA, C++, quantization`
  - Read [Docs/NvidiaHPCInterview.md](../../../Docs/NvidiaHPCInterview.md)
- `coding exercises such as attention or allocator design`
  - Read [Docs/CodingProblems.md](../../../Docs/CodingProblems.md)

## Suggested Loading Order

Choose the smallest set that covers the question:

1. Start from [Docs/InferenceOptimization.md](../../../Docs/InferenceOptimization.md) for most deployment questions.
2. Add [Docs/NvidiaHPCInterview.md](../../../Docs/NvidiaHPCInterview.md) when the role is GPU, CUDA, or systems heavy.
3. Add [Docs/CommunicationOptimization.md](../../../Docs/CommunicationOptimization.md) or [Docs/MemoryOptimization.md](../../../Docs/MemoryOptimization.md) when the tradeoff depends on scaling or VRAM.
4. Add [Docs/QuickInterviewAnswers.md](../../../Docs/QuickInterviewAnswers.md) only when the user wants a short spoken answer.
