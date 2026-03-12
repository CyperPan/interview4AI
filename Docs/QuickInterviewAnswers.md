# 30 秒标准口语答案（面试背诵版）

> 来源：牛客网大模型推理加速 & AI Infra 实习岗位面试题汇总
> 
> 建议每题用「一句定义 → 一句说明为什么重要 → 一句 trade-off」的结构来回答

---

## 第一部分：推理总流程

### 1. 什么是 prefill，什么是 decode？

> **Prefill** 就是把整段输入 prompt 一次性过模型，算出各层的 KV cache；**decode** 是后续逐 token 生成，每步只输入最新 token，但要读取之前所有历史 KV。它们的重要性在于 prefill 更偏计算密集、decode 更偏访存密集，所以优化手段和调度策略完全不同。trade-off 在于：prefill 适合用大 batch 吃满算力，而 decode 要在延迟、带宽和并发之间平衡。

### 2. 为什么 decode 往往更难优化？

> 因为 decode 每次只生成很少 token，单步算力利用率不高，却要频繁读取大体积 KV cache，常常更受显存带宽和访存模式限制。它的重要性在于在线服务的大部分时间其实都花在 decode，而不是 prefill 上。trade-off 是：你可以通过 KV 管理、batching、spec decoding 来降 decode 成本，但往往会增加实现复杂度和系统开销。

### 3. TTFT 和 ITL 分别是什么？

> **TTFT**（Time To First Token）是从请求进入到用户看到首 token 的时间，**ITL**（Inter-Token Latency）是后续 token 之间的平均间隔。它们的重要性在于共同决定用户"感觉快不快"：TTFT 决定"开口速度"，ITL 决定"说话是否流畅"。trade-off 是：追求极致吞吐时往往会牺牲 TTFT/ITL，需要在吞吐和用户体验之间做选择。

### 4. 为什么不能只看 tokens/s？

> Tokens/s 更像系统吞吐指标，反映 GPU 被利用得有多满，但不直接代表单个请求体验。它的重要性在于：很多优化只提高了 tokens/s，却让 TTFT、尾延迟或稳定性变差。trade-off 在于：离线场景可以更重视 tokens/s，在线场景必须同时看 TTFT、ITL、p95/p99 latency。

### 5. 一次完整的 LLM 在线推理链路怎么讲？

> 可以简单描述为：请求进来做 **tokenization**，进入 **scheduler** 排队，执行 **prefill** 生成 KV cache，然后在 **decode** 阶段反复"读 KV → 算 attention/FFN → sampling → 输出 token"，直到 EOS 或 stop 条件。它的重要性在于让面试官看到你既懂模型内部，又懂 serving 侧的调度和流式输出。trade-off 是：回答时不要陷太细 kernel 细节，而要把"阶段+瓶颈+调度点"讲清楚。

---

## 第二部分：KV Cache 与 Attention

### 6. 为什么 KV cache 是推理优化核心？

> 因为自回归生成里，每个新 token 都要依赖之前所有 token 的 K/V，如果不缓存就得每步重算整段上下文，算力开销爆炸。它的重要性在于：开启 KV cache 后，计算大幅减少，但显存和带宽成了新的瓶颈，所以几乎所有优化都围绕 KV 做文章。trade-off 是：你用 KV cache 换来了算力节省，但要用更复杂的内存布局、压缩和调度去对冲它带来的资源压力。

### 7. 什么是 PagedAttention？

> **PagedAttention** 的核心是把 KV cache 像操作系统分页一样按固定大小 block 管理，而不是给每个请求分一整段连续显存。它的重要性在于显著缓解显存碎片，更好支持变长请求和 continuous batching，是 vLLM 的招牌特性。trade-off 是：实现更复杂，增加了索引和寻址开销，但在高并发和长上下文下整体收益更大。

### 8. PagedAttention 的收益主要体现在哪？

> 主要体现在三点：**减碎片、提利用率、稳调度**——也就是更高的 KV 利用率、更大的可服务上下文和更平滑的请求生命周期管理。它的重要性在于直接决定"同样一块卡能同时撑多少上下文、多高并发"。trade-off 是：在小模型、短上下文、低并发下，收益可能没那么明显。

### 9. Prefix caching 是什么？

> **Prefix caching** 是指多个请求前缀相同时直接复用已有的前缀 KV cache，而不是每次都从头做 prefill。它的重要性在系统提示词固定、RAG 模板固定、多轮对话前缀重用度高的业务场景里，能显著降 TTFT 和显存压力。trade-off 是：需要维护 prefix 索引和安全隔离，命中率低时性价比就会下降。

### 10. Prefix caching 主要优化什么指标？

> 它更直接地优化 **TTFT**，因为省掉了重复 prefill 的时间，同时也能降低一部分计算和显存占用。重要性在于对长前缀、模版化很强的 workload 体验提升非常直观。trade-off 是：对自由度很高、个性化强的 prompt，命中率有限，不能指望它通吃所有场景。

### 11. 为什么 prefix caching 不一定总是收益很大？

> 因为它的收益高度依赖前缀复用率；如果业务里每个 prompt 都差别很大，那 cache 命中率会很低，维护 cache 的元数据和调度逻辑反而成了额外负担。重要性在于你要能根据 workload 判断"值不值当"。trade-off 是：可以把 prefix cache 设计成按路由、模版、租户做分层，而不是一刀切开关。

### 12. MHA、MQA、GQA 的区别是什么？

> **MHA** 是每个 Q head 有对应的 K/V head，**MQA** 是多个 Q head 共享一组 K/V，**GQA** 是在两者之间做分组共享。它们的重要性在于都在用"减少 K/V 头数"来换更小的 KV cache、更低的带宽开销，对推理友好。trade-off 是：K/V 越共享，表达能力和灵活性越可能受限，需要在模型效果和系统效率之间做平衡。

### 13. 为什么 GQA/MQA 对推理更有利？

> 因为 decode 阶段的热点是"读 KV"，而不是"算 QK^T"，减小 K/V head 数就等于直接减小 KV 体积和访问量。它的重要性在于使长上下文推理更可行、更省显存，很多新模型默认就用 GQA/MQA 设计。trade-off 是：训练时可能要更仔细调参，避免表达能力下降。

---

## 第三部分：Batching 与调度

### 14. 静态 batching 和 continuous batching 的区别？

> **静态 batching** 是先攒一批请求，固定 batch 跑完再处理下一批；**continuous batching** 则允许 batch 在运行中动态加入和移除请求。它们的重要性在于 LLM 请求长度差异巨大，静态 batch 容易被最长请求拖死，而 continuous batching 能显著提高整体 GPU 利用率。trade-off 是：continuous batching 调度复杂度更高，需要精心设计 scheduler 和 KV 管理。

### 15. Continuous batching 为什么适合 LLM serving？

> 因为 LLM serving 的典型特征就是"长短请求混在一起、输出长度不可预测"，静态 batch 的 padding 和等待成本会非常高。continuous batching 的重要性在于让 GPU 尽可能保持"有人干活"，而不是被长尾请求占住整个 step。trade-off 是：要防止饥饿和不公平，同时保证实现可维护。

### 16. 什么是 in-flight batching？

> **In-flight batching** 本质上就是 continuous batching 的一种叫法，特别强调"正在执行的 batch 中可以动态插入新请求"。它的重要性在 NVIDIA TensorRT-LLM 等框架中，是达成高吞吐和良好延迟的关键特性。trade-off 是：对 runtime 控制面和调试复杂度的要求更高。

### 17. 什么是 chunked prefill？

> **Chunked prefill** 就是把很长的 prompt prefill 切成多个小块按 step 执行，而不是一次性跑完。它的重要性在于避免单个长 prompt 把整张卡"卡死"，让 decode 和其他请求有机会插队，提高系统整体稳定性和尾延迟表现。trade-off 是：单个长 prompt 的理论最短时间可能略有拉长，但换来了整体更稳的服务质量。

### 18. Chunked prefill 主要改善什么？

> 它更偏向改善系统的**尾延迟**和多租户公平性，而不是追求单个长请求的极限时延。重要性在混合流量场景里特别明显，可以防止少量极长 prompt 把所有人拖慢。trade-off 是：如果场景全是均匀短请求，chunk 带来的调度开销可能不划算。

### 19. 什么是 P/D 分离？

> **P/D 分离** 是把 prefill 和 decode 阶段拆到不同进程甚至不同机器上运行，针对各自负载特性独立扩缩容和调优。它的重要性在于 prefill 更 compute-bound，decode 更 memory-bound，把它们拆开可以更细粒度地控制延迟和资源利用。trade-off 是：增加了 KV 传输、架构复杂度和一致性维护成本。

### 20. P/D 分离一定能提高吞吐吗？

> **不一定**，它更像是为"可控延迟和资源隔离"买单，而不是免费吞吐加速器。重要性在于很多官方文档都强调"它不保证 throughput 提升，而是给你更好的延迟和稳定性控制手段"。trade-off 是：如果 workload 很简单或量级不大，直接在一体化 runtime 里优化可能更划算。

---

## 第四部分：并行与通信

### 21. DP、TP、PP、EP 各是什么？

> **DP** 是数据并行，多副本模型分摊不同请求；**TP** 是张量并行，把同一层权重在多卡切块；**PP** 是流水并行，把不同层放到不同卡按流水线跑；**EP** 是专家并行，主要用于 MoE，把不同 expert 分配到不同 GPU。重要性在于它们共同构成大模型"怎么塞进多卡"的基本工具箱。trade-off 是：每种并行在显存、通信和负载均衡上的代价完全不同，需要按模型和场景选型。

### 22. 为什么推理和训练不能机械用同一套并行策略？

> 因为训练更关心总算力效率和显存能否装下，而推理更关心单请求延迟、动态 batch 和线上稳定性。重要性在于很多训练时期看起来很香的并行策略，一搬到小 batch、在线 decode 场景下就会被通信和调度开销吃掉。trade-off 是：推理往往宁可多开副本做 DP，也不愿意过度细粒度切 TP/PP。

### 23. 为什么多卡不一定更快？

> 因为多卡除了分摊计算，还引入同步、通信、负载不均和更多 kernel launch / NCCL 调度开销。重要性在于你必须能解释"8 卡比 4 卡还慢"的真实原因，而不是只会说"可能实现有 bug"。trade-off 是：在 batch 小或 decode 碎片化的时候，宁可少卡高效也比多卡低效强。

### 24. AllReduce、AllGather、AllToAll 分别怎么理解？

> **AllReduce** 用于把各卡的梯度或分片结果求和并广播，常见于 TP/DP；**AllGather** 是把各卡分片结果串联起来形成完整张量；**AllToAll** 是 MoE/EP 里典型的 token 重分发模式。重要性在于大多数通信瓶颈都可以映射到这几个原语上来分析。trade-off 是：AllToAll 功能最灵活，但对网络和拓扑要求也最高。

### 25. 为什么 MoE 场景更容易出通信问题？

> 因为 token 需要根据 router 结果被分发到不同 expert，而这些 expert 通常分布在多卡甚至多机上，频繁的 AllToAll 很容易把网络打满。重要性在于 MoE 常常"算力省了，但通信炸了"。trade-off 是：需要通过更好的 expert placement、路由策略和通信优化来抵消这部分成本。

---

## 第五部分：Kernel、CUDA、Triton

### 26. 为什么推理优化面试一定会问 CUDA/Triton？

> 因为很多性能瓶颈已经不在 Python 层或框架调度上，而落在 attention、layer norm、softmax、quantization、MoE dispatch 这些 kernel 上。重要性在于：没有 kernel 视角，很难真正解释"瓶颈在哪里"和"下一步怎么优化"。trade-off 是：工程上不能凡事都自己手写 kernel，要挑热点路径下手。

### 27. FlashAttention 解决了什么问题？

> FlashAttention 没改 attention 数学形式，而是通过 **IO-aware** 的执行方式来减少高代价的显存读写和中间张量搬运。它的重要性在于显著提高长序列下的速度和显存效率，已经是很多框架里的默认实现。trade-off 是：实现更复杂，对 head dim、布局和硬件特性敏感，不是"一行代码换个函数"那么简单。

### 28. 什么是 kernel fusion？

> **Kernel fusion** 是把多步算子合并到一个 kernel 里执行，以减少中间结果写回显存和 kernel launch 开销。重要性在于 LLM 推理 batch 经常不大，launch overhead 和 IO 成本比例更高，fusion 能明显提升端到端性能。trade-off 是：融合太重会让 kernel 变得难以维护和调优，并不总是收益最大。

### 29. CUDA Graph 为什么有用？

> **CUDA Graph** 能把一段相对固定的 GPU 执行图 capture 下来复用，减少 CPU 端发射 kernel 的开销和调度抖动。它的重要性在于在高 QPS 或复杂 pipeline 下，CPU 很容易成为隐藏瓶颈。trade-off 是：对动态 shape 和控制流不友好，需要在静态性和灵活性之间平衡。

### 30. Triton 和 CUDA 的差别怎么答？

> **Triton** 更像专门为深度学习 kernel 设计的 DSL，开发效率高、易于跨硬件调优；**CUDA** 则给你底层全部控制权，性能天花板更高。重要性在于很多团队会先用 Triton 快速验证和优化，再对个别关键 kernel 用 CUDA 手工榨干性能。trade-off 是：Triton 不一定覆盖所有硬件特性，极端场景仍然要回到 CUDA。

---

## 第六部分：量化与 Speculative Decoding

### 31. 为什么量化能提升推理效率？

> 量化通过降低权重和激活的比特宽度，减少显存占用和内存带宽压力，在硬件友好的场景下还能直接提升算子吞吐。它的重要性在于"让更大的模型跑在有限的卡上，并且服务更多并发"。trade-off 是：不同方法对模型质量、稳定性和硬件依赖程度差异很大，不能只看显存节省。

### 32. 量化一定更快吗？

> **不一定**，速度收益取决于硬件支持（比如是否有 INT4/INT8 Tensor Core）、kernel 实现和 dequant 带来的额外开销。重要性在于你要能解释"显存省了一半但几乎没提速"的现象。trade-off 是：量化最稳定的价值往往先体现在"省显存、提并发"，而不是单请求延迟翻倍优化。

### 33. 什么是 speculative decoding？

> **Speculative decoding** 是先用一个更快的 draft 机制（小模型或其他策略）猜一串 token，再让大模型批量验证，从而减少逐 token decode 的次数。它的重要性在于 decode 是很多系统的硬伤，spec 可以在不改大模型权重的前提下显著加速。trade-off 是：收益高度依赖 acceptance rate 和 workload 分布。

### 34. Speculative decoding 主要优化什么？

> 主要优化的是 decode 阶段的 token 生成效率，更直接改善 **ITL**，在高 QPS 或长回答场景中效果明显。重要性在于它补齐了"光靠 KV/调度还不够快"的那段空间。trade-off 是：不一定提升 TTFT，而且在 acceptance 低或 draft 很慢时甚至可能负优化。

### 35. 为什么 speculative decoding 不一定总能加速？

> 因为如果 draft 猜得不准，验证阶段就会做大量无用功，整体工作量未必下降，加上额外的控制逻辑本身也有成本。重要性在于你要知道 spec 是"机会主义加速"，不是稳定增益。trade-off 是：要根据模型、采样参数和 workload 实测 acceptance 决定是否开启以及怎么调。

---

## 第七部分：Benchmark 与 Profiling

### 36. 做推理 benchmark 最先要定义什么？

> 第一要定义**目标**：是看最大吞吐、最低 TTFT/ITL，还是某延迟约束下的 throughput；第二要固定 **workload**：prompt 长度、输出长度、并发数、采样参数、batch 策略。重要性在于没有明确定义就容易"换了问题还在比答案"。trade-off 是：要在实验可操作性和贴近真实场景之间取一个中间点。

### 37. 为什么 benchmark 不能只跑单 batch？

> 因为真实线上是动态请求流，而不是一个静态大 batch；很多系统优化（continuous batching、paged KV、prefix cache、P/D 分离）只有在动态并发下才真正发挥作用。重要性在于你需要设计"代表线上压测形态"的 benchmark，而不是只做 toy 实验。trade-off 是：更真实的 benchmark 更复杂、更难复现，但说服力要强得多。

### 38. Profiling 时怎么区分是 compute、memory 还是 communication 瓶颈？

> 可以按"算子时间 → 带宽利用 → 通信占比"三步来：kernel 算子时间长且 SM 忙但带宽富裕，多半 **compute-bound**；带宽打满但算力利用一般，多半 **memory-bound**；timeline 上大量 NCCL 或等待对端数据，多半 **communication-bound**。重要性在于只有先分清这三类，你后面的优化方向才不会跑偏。trade-off 是：profiling 要控制粒度和环境，否则很容易被噪音干扰结论。

---

## 第八部分：框架对比

### 39. vLLM 的关键词怎么背？

> 我一般会总结成：**PagedAttention、continuous batching、CUDA graph、prefix caching、chunked prefill、speculative decoding**。重要性在于它基本代表了现代高性能开源 LLM serving 的"标配能力"。trade-off 是：vLLM 偏通用高性能，极致深度优化或特定硬件生态下，可能还要配合其他方案。

### 40. SGLang 的关键词怎么背？

> 可以记成：**RadixAttention 做 prefix caching、zero-overhead CPU scheduler、prefill-decode 分离、continuous batching、paged attention、量化和多种并行支持**。重要性在于它在 runtime 和 structured/agent 场景里走得比较前。trade-off 是：生态相对更前沿、迭代快，稳定性和长期维护也要综合考虑。

### 41. TensorRT-LLM 的关键词怎么背？

> 我会记成：**in-flight batching、paged KV、chunked prefill、强量化能力、自定义 attention kernel、benchmark/serve 工具链一体**。重要性在于在 NVIDIA 生态和追求极致性能的场景下，它往往是第一选择。trade-off 是：对硬件和工具链绑定更深，迁移性和开发门槛要提前评估。

### 42. 面试官问"你更推荐哪个框架"怎么答？

> 我会说：做研究和通用验证时更偏 **vLLM/SGLang**，看重开发效率和灵活探索；在 NVIDIA 生态里要追求生产级极致性能，会重点考虑 **TensorRT-LLM**。重要性在于展示你能"按需求选型"，不是盲目吹某一个框架。trade-off 是：任何选型都要结合现有硬件、模型类型、团队经验和运维成本来权衡。

---

## 背诵建议

### 记忆技巧

1. **关键词提取**：每道题记住 3-5 个核心关键词
2. **结构模板**：定义 → 重要性 → Trade-off
3. **场景联想**：结合实际项目经历来记忆
4. **对比记忆**：相似概念对比记忆（如 FP16 vs BF16）

### 面试输出节奏

- **15秒**：给出核心定义
- **10秒**：说明重要性
- **5秒**：补充 trade-off

### 注意事项

- 不要死记硬背，要理解后用自己的话表达
- 结合自己的项目经历来举例说明
- 遇到不会的问题，坦诚说明并尝试关联已知知识点
