# vLLM 完整执行流程详解

本文档详细描述一个请求从输入到输出的完整执行流程，包括每一步的代码位置和关键细节。

## 示例场景

假设我们使用以下代码进行推理：

```python
from vllm import LLM, SamplingParams

# 初始化
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

# 生成
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

## 完整执行流程

### 阶段 1: 用户入口 - LLM.generate()

**文件**: `vllm/entrypoints/llm.py`

**步骤 1.1**: 用户调用 `llm.generate(prompts, sampling_params)`
- **位置**: `LLM.generate()` (约第 400 行)
- **操作**:
  1. 验证模型类型（必须是生成式模型）
  2. 处理采样参数
  3. 调用 `_validate_and_add_requests()` 添加请求
  4. 调用 `_run_engine()` 运行引擎

**步骤 1.2**: 添加请求到引擎
- **位置**: `LLM._validate_and_add_requests()` (约第 500 行)
- **操作**:
  1. 为每个 prompt 创建唯一的 `request_id`
  2. 调用 `self.llm_engine.add_request()` 添加请求

---

### 阶段 2: 引擎层 - LLMEngine.add_request()

**文件**: `vllm/v1/engine/llm_engine.py`

**步骤 2.1**: 处理输入请求
- **位置**: `LLMEngine.add_request()` (约第 222 行)
- **操作**:
  1. 验证 `request_id` 类型
  2. 如果 prompt 是字符串，通过 `InputProcessor` 处理
  3. 创建 `EngineCoreRequest` 对象
  4. 调用 `self.engine_core.add_request()` 添加到核心引擎

**步骤 2.2**: 输入处理
- **文件**: `vllm/v1/engine/input_processor.py`
- **位置**: `InputProcessor.process_prompts()`
- **操作**:
  1. **Tokenization**: 将文本 prompt 转换为 token IDs
     - 使用 HuggingFace tokenizer
     - 处理特殊 token（如 BOS、EOS）
  2. **多模态处理**: 如果有图像/视频等，进行编码
  3. 返回 `TextPrompt` 或 `TokensPrompt` 对象

**步骤 2.3**: 添加到 EngineCore
- **文件**: `vllm/v1/engine/core_client.py` 或 `vllm/v1/engine/core.py`
- **操作**:
  1. 将请求添加到调度器的等待队列
  2. 请求状态: `WAITING` → 等待调度

---

### 阶段 3: 引擎步进 - LLMEngine.step()

**文件**: `vllm/v1/engine/llm_engine.py`

**步骤 3.1**: 调用引擎步进
- **位置**: `LLM._run_engine()` → `LLMEngine.step()` (约第 285 行)
- **操作**:
  1. 调用 `self.engine_core.get_output()` 获取输出
  2. 通过 `OutputProcessor` 处理输出
  3. 处理需要中止的请求
  4. 记录统计信息

**步骤 3.2**: EngineCore 步进
- **文件**: `vllm/v1/engine/core.py`
- **位置**: `EngineCore.step()` (约第 336 行)
- **操作**:
  1. **检查请求**: `if not self.scheduler.has_requests(): return {}, False`
  2. **调度**: `scheduler_output = self.scheduler.schedule()`
  3. **执行模型**: `future = self.model_executor.execute_model(scheduler_output, non_block=True)`
  4. **采样**: `model_output = self.model_executor.sample_tokens(grammar_output)`
  5. **更新调度器**: `engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)`

---

### 阶段 4: 调度器 - Scheduler.schedule()

**文件**: `vllm/v1/scheduler/` (具体实现取决于调度器类型)

**步骤 4.1**: 调度请求
- **位置**: `Scheduler.schedule()`
- **操作**:
  1. **选择请求**: 从等待队列中选择要执行的请求
     - 考虑优先级、SLA、资源限制等
  2. **批处理**: 将多个请求组合成一个批次
     - Prefill 请求（新请求的首次处理）
     - Decode 请求（继续生成 token）
  3. **分配 KV Cache**: 为每个序列分配 KV cache 块
  4. **创建 SchedulerOutput**:
     - `scheduled_seq_groups`: 调度的序列组
     - `num_scheduled_tokens`: 每个序列的 token 数量
     - `block_tables`: KV cache 块表
     - `slot_mapping`: token 到 cache slot 的映射

**步骤 4.2**: KV Cache 分配
- **文件**: `vllm/v1/kv_cache_interface.py` 或相关实现
- **操作**:
  1. **计算所需块数**: 根据序列长度和 block_size 计算
  2. **分配块**: 从空闲块池中分配
  3. **更新块表**: 记录每个序列使用的块
  4. **创建 slot_mapping**: 将 token 位置映射到 cache slot

**SchedulerOutput 关键字段**:
```python
{
    'scheduled_seq_groups': [...],  # 调度的序列组
    'num_scheduled_tokens': {req_id: num_tokens},  # 每个请求的 token 数
    'block_tables': {req_id: [block_ids]},  # KV cache 块表
    'slot_mapping': tensor([...]),  # token 到 cache slot 的映射
    'attn_metadata': AttentionMetadata,  # 注意力元数据
}
```

---

### 阶段 5: 模型执行器 - Executor.execute_model()

**文件**: `vllm/v1/executor/`

**步骤 5.1**: Executor 分发
- **位置**: `Executor.execute_model()` (抽象基类)
- **操作**:
  1. 根据配置选择执行器类型:
     - `UniProcExecutor`: 单进程执行
     - `MultiprocExecutor`: 多进程执行
     - `RayDistributedExecutor`: Ray 分布式执行
  2. 调用 `collective_rpc("execute_model", args=(scheduler_output,))`

**步骤 5.2**: Worker 执行
- **文件**: `vllm/v1/worker/worker_base.py`
- **位置**: `WorkerWrapperBase.execute_model()` (约第 360 行)
- **操作**:
  1. 应用多模态缓存（如果有）
  2. 调用 `self.worker.execute_model(scheduler_output)`

**步骤 5.3**: Worker.execute_model()
- **文件**: `vllm/v1/worker/gpu_worker.py`
- **位置**: `Worker.execute_model()` (约第 575 行)
- **操作**:
  1. 处理流水线并行（如果需要）
  2. 调用 `self.model_runner.execute_model(scheduler_output, intermediate_tensors)`

---

### 阶段 6: Model Runner - GPUModelRunner.execute_model()

**文件**: `vllm/v1/worker/gpu_model_runner.py`

**步骤 6.1**: 准备输入
- **位置**: `GPUModelRunner.execute_model()` (约第 2930 行)
- **操作**:
  1. **准备输入批次**:
     ```python
     input_ids = prepare_input_tensors(scheduler_output)
     positions = prepare_position_tensors(scheduler_output)
     ```
  2. **准备注意力元数据**:
     ```python
     attn_metadata = prepare_attn_metadata(scheduler_output)
     # 包含: block_tables, seq_lens, slot_mapping 等
     ```
  3. **准备 KV Cache**:
     ```python
     kv_cache = self.kv_cache.get_cache()  # 获取 KV cache 张量
     ```

**步骤 6.2**: 设置前向上下文
- **位置**: 约第 3103 行
- **操作**:
  ```python
  with set_forward_context(
      attn_metadata,
      self.vllm_config,
      num_tokens=num_tokens_padded,
      ...
  ):
      # 这个上下文设置全局的注意力元数据，供模型层使用
  ```

**步骤 6.3**: 执行模型前向传播
- **位置**: 约第 3116 行
- **操作**:
  ```python
  model_output = self._model_forward(
      input_ids=input_ids,
      positions=positions,
      inputs_embeds=inputs_embeds,
      **model_kwargs,
  )
  ```

---

### 阶段 7: 模型前向传播 - Model.forward()

**文件**: `vllm/model_executor/models/llama.py` (以 Llama 为例)

**步骤 7.1**: 模型入口
- **位置**: `LlamaForCausalLM.forward()` 或类似方法
- **操作**:
  1. **嵌入层**: `input_embeds = self.embed_tokens(input_ids)`
  2. **位置编码**: `positions = self.rotary_emb(positions)`
  3. **逐层处理**: 遍历所有 Transformer 层

**步骤 7.2**: Transformer 层处理
- **位置**: `LlamaDecoderLayer.forward()`
- **操作** (对每一层):
  1. **自注意力**:
     ```python
     residual = hidden_states
     hidden_states = self.input_layernorm(hidden_states)
     hidden_states = self.self_attn(
         hidden_states,
         kv_cache=kv_cache,
         attn_metadata=attn_metadata,
     )
     hidden_states = residual + hidden_states
     ```
  2. **MLP**:
     ```python
     residual = hidden_states
     hidden_states = self.post_attention_layernorm(hidden_states)
     hidden_states = self.mlp(hidden_states)
     hidden_states = residual + hidden_states
     ```

---

### 阶段 8: 注意力层 - AttentionLayer.forward()

**文件**: `vllm/v1/attention/layer.py` 或 `vllm/model_executor/layers/attention.py`

**步骤 8.1**: 计算 Q, K, V
- **位置**: `AttentionLayer.forward()`
- **操作**:
  ```python
  qkv = self.qkv_proj(hidden_states)
  q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
  q = q.view(..., num_heads, head_size)
  k = k.view(..., num_kv_heads, head_size)
  v = v.view(..., num_kv_heads, head_size)
  ```

**步骤 8.2**: 调用注意力后端
- **位置**: `AttentionLayer.forward()` 继续
- **操作**:
  ```python
  # 根据配置选择后端（FlashAttention, PagedAttention 等）
  attn_output = self.attn_impl.forward(
      layer=self,
      query=q,
      key=k,
      value=v,
      kv_cache=kv_cache,  # KV cache 张量
      attn_metadata=attn_metadata,  # 包含 block_tables, slot_mapping 等
  )
  ```

---

### 阶段 9: 注意力后端 - AttentionImpl.forward()

**文件**: `vllm/v1/attention/backends/` (以 FlashAttention 为例)

**步骤 9.1**: 写入 KV Cache
- **文件**: `vllm/v1/attention/backends/flash_attn.py`
- **位置**: `FlashAttentionImpl.forward()` (约第 574 行)
- **操作**:
  ```python
  # 分离 key_cache 和 value_cache
  key_cache, value_cache = kv_cache.unbind(0)
  
  # 将新的 K, V 写入 cache
  reshape_and_cache_flash(
      key,           # [num_tokens, num_kv_heads, head_size]
      value,         # [num_tokens, num_kv_heads, head_size]
      key_cache,     # [num_blocks, num_kv_heads, head_size/x, block_size, x]
      value_cache,   # [num_blocks, num_kv_heads, head_size, block_size]
      attn_metadata.slot_mapping,  # [num_tokens] - 每个 token 对应的 cache slot
      self.kv_cache_dtype,
      layer._k_scale,
      layer._v_scale,
  )
  ```

**步骤 9.2**: 调用 PagedAttention 内核
- **位置**: 约第 667 行之后
- **操作**:
  ```python
  # 调用 FlashAttention 或 PagedAttention 内核
  flash_attn_with_kvcache(
      query,              # [num_tokens, num_heads, head_size]
      key_cache,          # KV cache
      value_cache,        # KV cache
      block_tables,       # [num_seqs, max_blocks_per_seq]
      seq_lens,           # [num_seqs] - 每个序列的长度
      ...
  )
  ```

---

### 阶段 10: CUDA 内核 - PagedAttention Kernel

**文件**: `csrc/attention/paged_attention_v1.cu` 或 `csrc/attention/attention_kernels.cu`

**步骤 10.1**: 内核启动
- **位置**: `paged_attention_v1()` 函数
- **操作**:
  1. **计算网格和块大小**:
     ```cuda
     dim3 grid(num_heads, num_seqs, num_partitions);
     dim3 block(THREAD_GROUP_SIZE);
     ```
  2. **启动内核**:
     ```cuda
     paged_attention_kernel<<<grid, block>>>(
         exp_sums, max_logits, out,
         q, k_cache, v_cache,
         num_kv_heads, scale,
         block_tables, seq_lens,
         ...
     );
     ```

**步骤 10.2**: 内核执行 (GPU)
- **位置**: `paged_attention_kernel()` (在 `csrc/attention/attention_kernels.cuh`)
- **操作** (每个线程组处理一个 (head, seq, partition)):
  1. **读取 Query**:
     ```cuda
     // 从全局内存读取 query
     float q_vec[HEAD_SIZE];
     load_q(q_vec, q, seq_idx, head_idx);
     ```
  2. **遍历 KV Cache 块**:
     ```cuda
     for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
         int physical_block = block_table[block_idx];
         // 从 KV cache 读取 K, V
         load_k(k_vec, k_cache, physical_block, ...);
         load_v(v_vec, v_cache, physical_block, ...);
         
         // 计算 attention score
         float score = dot_product(q_vec, k_vec) * scale;
         // 更新 max_logit, exp_sum
         // 累加 weighted value
     }
     ```
  3. **写入输出**:
     ```cuda
     // 归一化并写入输出
     out = weighted_sum / exp_sum;
     ```

**步骤 10.3**: KV Cache 写入内核
- **文件**: `csrc/cache_kernels.cu`
- **位置**: `reshape_and_cache_kernel()`
- **操作**:
  ```cuda
  // 将 K, V 重塑并写入对应的 cache slot
  // slot_mapping[i] 指定第 i 个 token 的 K, V 应该写入哪个 slot
  for (int i = 0; i < num_tokens; i++) {
      int slot = slot_mapping[i];
      int block_id = slot / block_size;
      int block_offset = slot % block_size;
      // 写入 key_cache[block_id][..., block_offset, ...]
      // 写入 value_cache[block_id][..., block_offset]
  }
  ```

---

### 阶段 11: 后处理 - 计算 Logits 和采样

**文件**: `vllm/v1/worker/gpu_model_runner.py`

**步骤 11.1**: 计算 Logits
- **位置**: 约第 3151 行
- **操作**:
  ```python
  # 只对需要采样的 token 计算 logits（通常是每个序列的最后一个 token）
  sample_hidden_states = hidden_states[logits_indices]
  logits = self.model.compute_logits(sample_hidden_states)
  # logits: [num_seqs, vocab_size]
  ```

**步骤 11.2**: 采样
- **文件**: `vllm/v1/worker/gpu_model_runner.py`
- **位置**: `GPUModelRunner.sample_tokens()` (约第 950 行)
- **操作**:
  1. **获取 logits**:
     ```python
     logits = self.execute_model_state[2]  # 从 execute_model 保存的状态
     ```
  2. **应用 logits processor**:
     ```python
     logits = apply_logits_processors(logits, sampling_params)
     ```
  3. **采样**:
     ```python
     # 根据 sampling_params 进行采样
     # - temperature sampling
     # - top_p sampling
     # - top_k sampling
     # - beam search
     sampled_token_ids = sample(logits, sampling_params)
     ```

**步骤 11.3**: 创建 ModelRunnerOutput
- **位置**: 约第 1000 行
- **操作**:
  ```python
  output = ModelRunnerOutput(
      req_ids=[...],
      token_ids=sampled_token_ids,  # [num_seqs] - 每个序列新生成的 token
      logprobs=logprobs,  # 可选
      hidden_states=hidden_states,  # 用于下一次迭代
  )
  ```

---

### 阶段 12: 调度器更新 - Scheduler.update_from_output()

**文件**: `vllm/v1/scheduler/`

**步骤 12.1**: 更新序列状态
- **位置**: `Scheduler.update_from_output()`
- **操作**:
  1. **更新序列长度**: 每个序列长度 +1
  2. **检查停止条件**:
     - EOS token
     - 最大长度
     - 停止字符串
  3. **更新 KV Cache 块表**: 如果序列增长，可能需要分配新块
  4. **标记完成的序列**: 如果满足停止条件

**步骤 12.2**: 创建 EngineCoreOutput
- **操作**:
  ```python
  for seq_group in scheduled_seq_groups:
      for seq in seq_group.seqs:
          output = EngineCoreOutput(
              request_id=seq.request_id,
              finished=seq.is_finished(),
              new_token_id=sampled_token_ids[seq.idx],
              new_token_logprob=logprobs[seq.idx],
              ...
          )
  ```

---

### 阶段 13: 输出处理 - OutputProcessor.process_outputs()

**文件**: `vllm/v1/engine/output_processor.py`

**步骤 13.1**: 处理输出
- **位置**: `OutputProcessor.process_outputs()` (约第 442 行)
- **操作**:
  1. **更新序列状态**:
     ```python
     for engine_core_output in engine_core_outputs:
         req_state = self.request_states[output.request_id]
         req_state.add_new_token(output.new_token_id)
     ```
  2. **Detokenization**: 将 token ID 转换为文本
     ```python
     text = self.tokenizer.decode([token_ids])
     ```
  3. **检查停止字符串**: 如果生成的文本包含停止字符串，标记为完成
  4. **创建 RequestOutput**:
     ```python
     request_output = RequestOutput(
         request_id=req_id,
         prompt=original_prompt,
         outputs=[CompletionOutput(
             index=0,
             text=generated_text,
             token_ids=all_token_ids,
             finish_reason=finish_reason,
         )],
         finished=is_finished,
     )
     ```

**步骤 13.2**: 返回结果
- **位置**: `LLMEngine.step()` (约第 319 行)
- **操作**:
  ```python
  return processed_outputs.request_outputs
  # 返回给 LLM.generate()
  ```

---

### 阶段 14: 循环执行

**文件**: `vllm/entrypoints/llm.py`

**步骤 14.1**: 继续生成
- **位置**: `LLM._run_engine()` (约第 600 行)
- **操作**:
  ```python
  while self.llm_engine.has_unfinished_requests():
      step_outputs = self.llm_engine.step()
      # 收集输出
      for output in step_outputs:
          if output.finished:
              final_outputs.append(output)
  ```

**步骤 14.2**: 下一次迭代
- 对于未完成的序列，重复阶段 3-13，直到:
  - 生成 EOS token
  - 达到最大长度
  - 遇到停止字符串

---

## 关键数据结构

### 1. EngineCoreRequest
```python
{
    'request_id': str,
    'prompt': TextPrompt | TokensPrompt,
    'sampling_params': SamplingParams,
    'arrival_time': float,
}
```

### 2. SchedulerOutput
```python
{
    'scheduled_seq_groups': List[SequenceGroup],
    'num_scheduled_tokens': Dict[req_id, int],
    'block_tables': Dict[req_id, List[int]],
    'slot_mapping': torch.Tensor,
    'attn_metadata': AttentionMetadata,
}
```

### 3. AttentionMetadata
```python
{
    'block_tables': torch.Tensor,  # [num_seqs, max_blocks]
    'seq_lens': torch.Tensor,      # [num_seqs]
    'slot_mapping': torch.Tensor,  # [num_tokens]
    'query_start_loc': torch.Tensor,  # [num_seqs + 1]
    'context_lens': torch.Tensor,  # [num_seqs]
}
```

### 4. KV Cache 布局
```python
# Key Cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
# Value Cache: [num_blocks, num_kv_heads, head_size, block_size]
# block_size: 通常为 16
# 每个 block 存储 16 个 token 的 KV
```

---

## 性能优化点

1. **PagedAttention**: 减少内存碎片，提高内存利用率
2. **Continuous Batching**: 动态批处理，提高 GPU 利用率
3. **CUDA Graph**: 减少 CPU-GPU 同步开销
4. **Kernel Fusion**: 融合多个操作，减少内存访问
5. **Quantization**: 使用 INT8/FP8 减少内存和计算

---

## 调试建议

1. **设置日志级别**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **使用 Profiler**:
   ```python
   from vllm.profiler import Profiler
   profiler = Profiler()
   # ... 运行推理
   profiler.print_stats()
   ```

3. **检查关键变量**:
   - `scheduler_output.block_tables`: KV cache 分配
   - `attn_metadata.slot_mapping`: Token 到 cache 的映射
   - `model_output.token_ids`: 生成的 token

---

## 总结

一个完整的推理步骤包含以下主要阶段：

1. **输入处理**: Tokenization, 多模态编码
2. **调度**: 选择请求，批处理，分配 KV cache
3. **模型执行**: 前向传播，注意力计算
4. **KV Cache**: 写入和读取
5. **CUDA 内核**: PagedAttention 计算
6. **采样**: Logits 计算和 token 采样
7. **输出处理**: Detokenization, 停止检查
8. **循环**: 重复直到完成

整个过程高度优化，充分利用 GPU 并行性和内存管理技术，实现高性能的 LLM 推理。

