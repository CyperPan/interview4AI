# Metrics and Formulas

Use this file when the user needs calculations, tradeoff framing, or polished interview answers.

## Core Metrics

- `TTFT`: Queueing + prefill + first decode step. Improve it with shorter queues, faster prefill, prompt caching, and better scheduler isolation.
- `TPOT`: Steady-state decode latency per generated token. Improve it with lower memory traffic, better KV handling, efficient decode kernels, and better batching.
- `Throughput`: Requests or tokens per second under a load profile. Improve it with higher occupancy, less wasted padding, better batching, and higher concurrency.
- `Accuracy`: Perplexity, task score, or benchmark score after optimization. Track this when quantizing, pruning, or distilling.

## Quick Decision Table

| Situation | Likely bottleneck | First levers |
| --- | --- | --- |
| Long prompt, short output | Compute-bound prefill | FlashAttention, FP8/BF16 Tensor Core kernels, prompt caching |
| Short prompt, long output | Memory-bound decode | Quantization, KV cache efficiency, continuous batching |
| Latency spikes during mixed traffic | Scheduler-bound | PD separation, queue isolation, admission control |
| TP across many GPUs | Communication-bound | Reduce TP width, prefer NVLink islanding, fuse collectives |
| GPU OOM at moderate QPS | Capacity-bound | Weight-only quantization, PagedAttention, shorter max context |

## Back-of-the-Envelope Formulas

### Weight memory

Approximate model weight memory:

`weight_bytes ~= parameter_count * bytes_per_weight`

Examples:

- `7B` at `BF16`: about `14 GB`
- `7B` at `INT8`: about `7 GB`
- `70B` at `INT4`: about `35 GB` before metadata, scales, and runtime overhead

Add runtime headroom for activations, workspace, fragmentation, allocator slack, and KV cache.

### KV cache memory

For grouped-query or multi-query attention, use KV heads, not attention heads:

`kv_bytes ~= batch * seq_len * num_layers * num_kv_heads * head_dim * 2 * bytes_per_element`

Where:

- `2` accounts for keys and values
- `seq_len` means cached tokens, not total max model length unless fully used

If the user only knows hidden size and total heads:

`head_dim = hidden_size / num_attention_heads`

For rough reasoning, KV often dominates serving memory at long context even when weights dominate short-context single-request inference.

### Arithmetic intensity

Use:

`arithmetic_intensity = flops / bytes_moved`

Interpretation:

- low intensity means memory-bound
- high intensity means compute-bound

Prefill has much higher intensity than decode because decode runs small matrix-vector style work against large weights.

### Bandwidth-limited decode intuition

If decode is purely bandwidth-limited:

`tokens_per_second ~= effective_memory_bandwidth / bytes_read_per_token`

This is deliberately rough. Use it to explain why lower precision weights or better cache reuse can help even when peak FLOPs stay unchanged.

### Tensor parallel communication

For tensor parallel layers, each step may require collective communication on activations or partial outputs:

`step_time ~= compute_time + collective_time`

If `collective_time` is not hidden by overlap, scaling saturates quickly outside fast interconnect domains.

## Interview Templates

### Quantization choice

Use this shape:

1. State the goal: latency, throughput, or capacity.
2. State the regime: prefill or decode.
3. Explain the hardware path.
4. Choose the quantization method.
5. State the accuracy risk and kernel dependency.

Example skeleton:

"For decode on bandwidth-limited GPUs, I first ask whether VRAM or latency is the main limit. If capacity is the problem, I favor weight-only INT4 such as AWQ because it cuts memory traffic and improves concurrency. If I am on Hopper with mature kernels and need safer accuracy, I prefer FP8 because it uses native Tensor Core support with lower calibration risk."

### Speculative decoding

Use this shape:

1. Mention acceptance rate.
2. Mention spare compute.
3. Mention when it backfires.

Example skeleton:

"Speculative decoding helps when the draft model predicts many accepted tokens and the target model still has spare compute. It often helps low-to-moderate load scenarios more than saturated systems. If acceptance rate is low or memory bandwidth is already maxed out, the extra draft pass can reduce net throughput."

### PD separation

Use this shape:

1. Contrast prefill and decode bottlenecks.
2. Explain interference.
3. Explain why split deployment improves tail latency.

## Common Follow-ups

- Why does this help `TPOT` more than `TTFT`?
- What changes on `H100` versus `A100` versus consumer GPUs?
- What if the interconnect is `PCIe` instead of `NVLink`?
- What accuracy loss would you accept for a production chatbot?
- How would you validate that the optimization really moved the bottleneck?
