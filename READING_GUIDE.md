# nanochat 系统学习与源码阅读指南

适用背景：已经接触过 Llama 相关源码，了解 Transformer、RoPE、注意力与自回归生成。接触过不等于已经掌握，可通过下面的前置检查按需补齐。

阅读主线：追踪一个模型如何从原始文本训练成聊天助手。先打通数据与执行流程，再深入架构改动和性能优化。本文依据当前工作区代码整理；源码更新后，具体实现可能变化。

## 学习目标与前置检查

这份指南的目标是建立一个 decoder-only 语言模型从数据、训练到聊天推理与评估的完整认识，并能迁移到另一个类似项目。它不是覆盖所有大模型研究方向的课程；多模态、检索增强、复杂并行和底层内核可以在主线完成后独立学习。

先尝试回答下表问题。不熟悉的内容在进入对应阶段前补齐即可，不必先停下来学习一整套数学或 PyTorch 课程。

| 基础 | 最小掌握要求 | 在哪里用到 |
| --- | --- | --- |
| 张量与线性代数 | 能说明 `(B,T,C) @ (C,V)` 的结果；理解转置、广播、最后一维归一化 | GPT、注意力 |
| 概率与损失 | 能区分 logits、softmax 概率、采样；解释目标概率越低，负对数损失越大 | 生成、预训练 |
| 自动微分与优化 | 区分参数、激活、梯度；知道 backward 计算梯度，step 更新参数，梯度需要清零 | 训练循环 |
| Python 与 PyTorch | 理解生成器的暂停/恢复、切片、原地写入、device/dtype，以及 eval 与禁用梯度的不同职责 | 数据加载、缓存、推理 |
| 注意力与位置 | 能用 Q/K/V 解释“当前位置读取哪些历史信息”，知道位置编码提供顺序信息 | mask、RoPE、KV cache |

不要用“认得这些名词”作为通过标准。例如，给定两个 logits `[0,0]`，应能解释概率为什么都是 `1/2`、目标是其中一个时损失为什么是 `ln(2)`，以及仅对 logits 取 argmax 与按概率采样的区别。

## 贯穿项目的知识框架

所有章节围绕同一个目标：根据前缀预测下一个 token。对一段序列，模型把联合概率分解为各位置条件概率的乘积；对数把乘积变成求和，训练就可以逐位置计算负对数似然。

```text
条件分布：pθ(x[t+1] | x[0], ..., x[t])
训练目标：让真实的下一个 token 在这个分布中的概率变大
生成过程：从这个分布选出一个 token，接到前缀上，再计算下一步
```

| 层次 | 它解决什么问题 | 与其他层的连接 |
| --- | --- | --- |
| 表示与协议 | 文本和角色如何表示成 ID？ | tokenizer 定义输入空间和聊天边界；模型本身读取的是 ID |
| 模型与因果性 | 如何由可见前缀得到 logits？ | embedding、Block、位置与 mask 决定信息流 |
| 学习目标 | 哪些预测算对错，如何汇总？ | target shift 与 loss mask 指定监督；梯度把误差传回模型 |
| 执行与状态 | 如何重复训练或持续生成？ | 优化器、数据位置、checkpoint、KV cache 各自管理不同状态 |
| 测量与改进 | 学得怎样，运行成本怎样？ | 验证损失、任务评分、延迟和显存衡量不同维度 |

训练和推理共享模型，但输入来源和状态变化不同：

- **训练**：使用数据中的真实前缀，同时计算多个位置的预测与 loss，再更新参数。这解释了为什么训练能够并行处理多个位置，却仍然需要因果 mask。
- **生成**：模型参数通常固定，后续前缀包含模型自己选出的 token；每次选择都会改变下一步输入。生成错误也可能改变后续上下文。
- **缓存**：在参数固定且位置、窗口和其他状态正确的前提下复用历史计算，目标是在数值容差内保持对应预测，而不是改变学习目标。
- **SFT**：仍然训练下一 token 预测，但改变数据组织和参与监督的位置，使模型学习对话行为。

读每项优化时先判断它属于哪类：KV cache 与融合注意力主要改变计算执行方式；滑动窗口改变可见信息范围；smear 等架构改动改变模型计算；温度与 top-k 改变生成选择规则。它们不能都用“只是加速”来解释。

还要区分三类常见状态：模型参数是训练得到并用于推理的知识载体；优化器状态服务于继续训练；KV cache 是当前上下文的临时计算结果。加载同一份模型不意味着恢复了优化器或某段对话的缓存。

## 从你当前的位置开始

你已经读到了 `GPT.forward`、`CausalSelfAttention` 和朴素的 `GPT.generate`，也开始关注因果 mask。现在适合先把模型计算与生成过程连接起来，再进入完整训练链路。无需重新逐行阅读已经理解的 `gpt.py`。

建议顺序如下；其中 2A、2B 是模型章节后的两段衔接阅读：

```text
1. 全流程地图（快速浏览）
   ↓
2. GPT 主干与朴素 generate（回顾，补齐张量形状）
   ↓
2A. flash_attention：因果性、窗口、位置对齐 ← 当前建议从这里开始
   ↓
2B. KVCache 与 Engine：先理解纯文本生成
   ↓
3. tokenizer：普通文本编码
   ↓
4. 预训练：数据 → batch → loss → 参数更新 → checkpoint
   ↓
5. SFT：对话格式、target 对齐与 loss mask
   ↓
6. 聊天推理：回到 Engine，补齐角色、停止条件与工具调用
   ↓
7. 评估 → 8. 架构复读、优化器、分布式、精度与 RL 等选修
```

学习依赖与实际训练顺序不同：读懂推理可以先于读懂训练脚本。此时把 tokenizer 暂时视为“文本与 ID 之间的转换器”，不妨碍理解注意力和缓存。

每次阅读只追踪一条路径，记录四件事：输入和输出形状、位置含义、修改了哪些状态、调用方依赖什么约定。遇到优化代码，先写出它对应的普通计算，再研究如何加速。检查点能用自己的话回答后即可进入下一阶段，不要求第一遍理解每一行。

## 1. 建立全流程地图

先读 [runs/speedrun.sh](runs/speedrun.sh)，只关注脚本调用顺序和参数，不需要实际启动训练。

```text
下载数据 → 训练 tokenizer → 预训练 → 基座评估 → SFT → 对话评估
```

接着快速浏览 [runs/miniseries.sh](runs/miniseries.sh)，了解如何组织不同模型规模的实验；小规模运行入口可参考 [runs/runcpu.sh](runs/runcpu.sh)。

检查点：

- [ ] 能说出每个阶段对应的 Python 入口。
- [ ] 能区分基座模型与 SFT 后的聊天模型。
- [ ] 知道当前 speedrun 默认流程不包含 RL。

## 2. 理解模型主干

文件：[nanochat/gpt.py](nanochat/gpt.py)。建议按符号跳转阅读，而不是从第一行顺读到底：

```text
GPTConfig → GPT.forward → Block → CausalSelfAttention → MLP
          → GPT.__init__ → init_weights
```

先追踪张量流：

```text
token ids [B,T] → embedding [B,T,C] → 多层 Block
               → logits [B,T,V] → 有 targets 时计算交叉熵
```

结合已有的 Llama 知识，重点关注本项目的实现：

| 实现 | 阅读问题 |
| --- | --- |
| 无可学习参数的 RMSNorm | `norm()` 如何实现？归一化放在哪些位置？ |
| ReLU² MLP | 两层线性层如何连接，与 SwiGLU 的门控分支有何区别？ |
| QK norm | RoPE 后如何处理 Q、K？ |
| 滑动窗口注意力 | `window_pattern` 如何决定各层窗口？ |
| GQA 支持 | Q 与 KV 的头数如何配置？默认二者相等意味着什么？ |
| 自定义 `Linear` | fp32 权重如何与计算 dtype 配合？ |
| Value embedding、smear、backout、残差系数 | 各自增加或修改了哪条信息路径？ |

第一遍可以先标记最后一行的架构改动，第二遍再细读。FLOPs 估算和优化器细节暂时跳过。

检查点：

- [ ] 能画出一个 Block 的计算路径，并标注张量形状。
- [ ] 能区分训练 forward 与带 KV cache 的推理 forward。
- [ ] 理解 meta device 构建与 `init_weights()` 初始化的分工。

补读 `GPT.generate`，把一次 forward 与完整生成循环区分开：`forward` 同时输出各输入位置的下一 token 分数，`generate` 只取最后一个位置，然后采样、追加 ID、重新 forward。当前无缓存路径要求输入长度大于 1；它不会自动检测 EOS，也不自动裁剪过长上下文。

建议产出：一张标注 `B`（批量）、`T`（当前输入长度）、`C`（隐藏维度）、`V`（词表大小）的形状图，再用三个输入 ID 手动追踪两轮生成。此时无需研究采样效果调参。

## 2A. 现在读注意力接口：mask 到底在哪里

文件：[nanochat/flash_attention.py](nanochat/flash_attention.py)。本轮目标是理解注意力的可见范围与张量布局，不要求掌握 Flash Attention 内核算法。

按调用关系阅读：

```text
CausalSelfAttention.forward
  → flash_attn_func
    → FA3：传递 causal=True 与 window_size
    → SDPA fallback：转置布局 → _sdpa_attention → 转回布局
```

先看 `flash_attn_func`，再细读 `_sdpa_attention`，最后读 `flash_attn_with_kvcache`。文件开头的内核加载、设备检测和后端选择先略读，知道存在 FA3 与 SDPA 两条路径即可。

| 情况 | Q/K 序列长度 | 当前 SDPA 路径如何限制可见范围 |
| --- | --- | --- |
| 完整序列，窗口不限制上下文 | `Tq == Tk` | `is_causal=True`，由底层处理因果遮罩 |
| 带历史缓存的单 token decode | `Tq=1`，通常 `Tk>1` | K/V 只包含历史和当前位置，窗口有限时先裁剪，再用 `is_causal=False` |
| 需要限制的滑动窗口或带缓存的多 token chunk | 可能 `Tq != Tk` | 显式构造布尔 mask，同时考虑绝对位置与左侧窗口 |

分支按源码顺序匹配，例如 `Tq=Tk=1` 且窗口无限时，首先进入完整序列分支。不要只根据一个条件猜测实际执行路径。

重点理解以下细节：

- 外部 Q 的布局是 `(B,Tq,Hq,D)`，K/V 是 `(B,Tk,Hkv,D)`；进入 SDPA 前交换序列维与头维。GQA 时 `Hq` 与 `Hkv` 可以不同，不要把 Q 的头数直接套给 K/V。
- `causal=True` 表示禁止看到未来位置，不要求调用方分配一个完整的三角 mask 张量。
- 在这里传给 SDPA 的布尔 `attn_mask` 中，`True` 表示允许关注。不要把其他接口中布尔 mask 的约定直接搬过来。
- 单 token decode 使用 `is_causal=False` 的前提是当前提供的 K/V 没有未来 token；不是“推理可以忽略因果性”。
- 窗口参数 `(left, right)` 中，当前调用使用 `right=0`；左窗口为 `W` 时包含至多 `W` 个历史位置和当前位置，共 `W+1` 个 key。全上下文 `(-1,0)` 仍然是因果注意力。
- 当前 fallback 没有把公开接口的 `causal` 参数传给 `_sdpa_attention`，而是按项目的因果场景实现；不能仅凭接口默认值 `causal=False` 就推断 fallback 支持双向注意力。缓存 fallback 还假设 batch 内位置一致，使用 `cache_seqlens[0]`。

### 手算一次缓存下的 mask

假设缓存已有 3 个 token，现在输入 2 个新 token，则 `Tq=2`、`Tk=5`。query 的绝对位置是 3、4，而不是 0、1：

```text
row_idx = (Tk - Tq) + arange(Tq) = [3, 4]
col_idx = arange(Tk)            = [0, 1, 2, 3, 4]

仅因果约束：
query 3 → [1, 1, 1, 1, 0]
query 4 → [1, 1, 1, 1, 1]

再限制 left=2：
query 3 → [0, 1, 1, 1, 0]
query 4 → [0, 0, 1, 1, 1]
```

源码中的 `col_idx <= row_idx` 排除未来，`row_idx - col_idx <= window` 排除太远的历史。这个例子说明：有缓存时，必须按 query 在完整序列中的位置构造 mask，不能直接套一个从位置 0 开始的三角形。

结合 [tests/test_attention_fallback.py](tests/test_attention_fallback.py) 阅读基本因果注意力、滑动窗口、GQA、prefill 和单 token 缓存案例。先读测试输入和断言；FA3 对比测试有设备与内核要求，不必为了阅读立即配置对应硬件。

检查点：

- [ ] 能解释 `gpt.py` 没有显式创建 mask，为什么仍然满足因果性。
- [ ] 能独立画出上面两种 mask，并解释位置偏移 `Tk-Tq`。
- [ ] 能区分窗口允许关注的范围、缓存已使用长度和缓存分配容量。
- [ ] 能解释普通完整序列和单 token decode 为什么使用不同的 `is_causal` 设置。

## 2B. 接着读 KV cache：把生成过程串起来

文件：[nanochat/engine.py](nanochat/engine.py)。本轮先忽略计算器与工具执行细节，关注 `KVCache`、`sample_next_token` 和 `Engine.generate` 的纯文本路径。

阅读顺序：

1. `KVCache.__init__ / get_layer_cache / get_pos / advance`：每层缓存什么，逻辑位置如何记录。
2. `flash_attn_with_kvcache`：新 K/V 写入哪个区间，本次注意力读取哪个区间。
3. 回到 `GPT.forward` 与 `CausalSelfAttention.forward`：RoPE 偏移、smear 的 `prev_embedding`、最后一层推进缓存位置。
4. `Engine.generate`：先做 batch=1 的 prompt prefill，再复制缓存以生成多个样本，之后逐 token decode。
5. `sample_next_token`：与已经读过的 `GPT.generate` 对照采样逻辑。

| 时刻 | 传入模型的 ID | 本次计算的新 Q/K/V | 可供注意力读取的 K/V |
| --- | --- | --- | --- |
| 空缓存 prefill | 整段提示词，长度 P | P 个位置 | 提示词的 P 个位置，内部仍需因果约束 |
| 下一次 decode | 刚采样得到的一个 ID | 1 个位置 | P 个历史位置 + 当前位置 |
| 再下一次 decode | 新采样得到的一个 ID | 1 个位置 | P+1 个历史位置 + 当前位置 |

prefill 最后一个位置的 logits 已能采样第一个新 token。采样得到该 ID 时，它尚未经过下一次 forward，因此尚未写入 K/V 缓存；不要把“已输出 token 数”直接当成“缓存长度”。

缓存复用的是各层历史 K/V，并没有免除当前 query 对历史 key 的注意力计算。当前实现还单独保存 smear 所需的前一 token embedding。窗口限制也不意味着缓存张量自动缩小成窗口大小。

结合 [tests/test_engine.py](tests/test_engine.py) 的缓存、随机种子、多样本和生成长度测试理解契约。注意其中使用了 mock 模型：这些测试有助于理解引擎控制流程，但不能单独证明真实 GPT 的缓存计算与完整重算数值一致。

检查点：

- [ ] 能说明为什么缓存位置在最后一层之后统一推进，而不是每层都加一次。
- [ ] 能手动记录长度为 3 的 prompt、首个采样结果和两次 decode 的输入及缓存位置。
- [ ] 能解释为什么通常不缓存历史 Q，以及新 token 的 Q 仍需重新计算。
- [ ] 能区分 K/V 缓存与 smear 的 `prev_embedding`。

完成后继续第 3 阶段；遇到聊天角色、工具状态等问题先记下来，第 6 阶段再处理。

## 3. 理解 tokenizer：从文本到 token

在进入预训练前，单独读一遍 tokenizer 的训练与编码流程。阅读顺序：

1. [scripts/tok_train.py](scripts/tok_train.py)：训练文本来源、词表大小、tokenizer 训练与保存。
2. [nanochat/tokenizer.py](nanochat/tokenizer.py)：重点看 `RustBPETokenizer.train_from_iterator`、`encode`、`decode`、`save`、`from_directory` 和特殊 token 定义。
3. [scripts/tok_eval.py](scripts/tok_eval.py)：了解如何评估分词的压缩效率。

先串起两条路径：

```text
训练文本 → 训练 BPE → 保存 tokenizer → 加载 tokenizer
输入文本 → encode → token ids → decode → 文本
```

结合 Llama 的阅读经验，重点看这里如何训练自己的词表，以及 tokenizer 的词表大小如何连接模型的 embedding 和输出层。先掌握项目使用的接口，BPE 底层算法实现可以后续深入。

这一轮先理解普通文本编码和特殊 token 定义；第 5 阶段再读对话序列化与 loss mask。

检查点：

- [ ] 能区分训练 tokenizer 与使用 tokenizer 编码文本。
- [ ] 能说明 tokenizer 如何保存、加载，以及词表大小如何传给模型。
- [ ] 能追踪一段文本的编码与解码，区分普通文本 token 和特殊 token。
- [ ] 能解释 token 数与文本字节数的关系，以及压缩效率衡量什么。

## 4. 追踪一个预训练 batch

以 [scripts/base_train.py](scripts/base_train.py) 的训练循环为中心，按需要跳转：

1. [nanochat/dataset.py](nanochat/dataset.py)：文本读取与数据分片。
2. [nanochat/tokenizer.py](nanochat/tokenizer.py)：回顾编码接口，关注数据加载器如何调用它。
3. [nanochat/dataloader.py](nanochat/dataloader.py)：分布式数据分配、文档打包、inputs 和 targets 构造。
4. [nanochat/gpt.py](nanochat/gpt.py) 的 `forward` 和 `setup_optimizer`：loss 与参数分组。
5. 回到训练循环：梯度累积、反向传播、学习率调度与参数更新。
6. [nanochat/checkpoint_manager.py](nanochat/checkpoint_manager.py)：保存、加载及恢复训练。

先理解 Muon 和 AdamW 分别处理哪些参数，暂时不深入 Muon 的矩阵运算。

检查点：

- [ ] 能解释 `x` 与 `y` 如何错开一个 token，以及模型内部是否还会 shift。
- [ ] 能计算单卡 batch、序列长度、进程数、梯度累积与全局 token batch 的关系。
- [ ] 能追踪 `--depth` 如何影响宽度、训练规模和超参数。
- [ ] 能说明恢复训练除了权重还需要哪些状态。

建议手算一个最小例子：原始 ID 为 `[a,b,c,d]`，则 `inputs=[a,b,c]`、`targets=[b,c,d]`。`GPT.forward` 接收的 targets 已经对齐，不再内部 shift。训练时各位置可并行计算，但因果 mask 保证位置 `a` 不能借助后面的 `b,c` 预测答案；生成时则需要先得到下一个 ID，才能继续下一步。

全局每步 token 数可写为：`device_batch_size × max_seq_len × world_size × grad_accum_steps`。例如 `2×8×1×4=64`；结合循环解释为什么每个 micro-batch 的 loss 要除以累积步数。

第一遍先按单进程、非 FP8 路径理解训练循环，再补分布式通信、编译与混合精度。读 checkpoint 时区分“恢复模型用于推理”和“恢复优化过程”；当前 dataloader 的恢复是近似恢复，不应直接理解为逐 token 完全复现未中断运行。

## 5. 理解 SFT 的对话格式与 loss mask

这是 tokenizer 的第二轮阅读，重点从普通文本编码转向聊天协议与训练目标。

阅读顺序：

1. [tasks/common.py](tasks/common.py) 与 [tasks/smoltalk.py](tasks/smoltalk.py)：任务接口、数据混合与对话样本。
2. [nanochat/tokenizer.py](nanochat/tokenizer.py) 的 `render_conversation` 与 `render_for_completion`：对话序列化、mask，以及训练格式与推理 prompt 的对齐。
3. [scripts/chat_sft.py](scripts/chat_sft.py) 的 `sft_data_generator_bos_bestfit`：打包、错位和屏蔽 targets。
4. SFT 训练循环：与预训练对比数据来源和训练目标。

手动追踪一条对话：

```text
user / assistant 消息 → 特殊 token + 文本 token → ids 和 mask
                     → inputs / targets 错位
                     → 不参与 loss 的 targets 设为 -1
                     → 交叉熵忽略对应位置
```

检查点：

- [ ] 能指出用户文本、助手回复、工具输出及各类特殊 token 是否参与 loss。
- [ ] 能解释为什么 mask 也要取 `[:, 1:]` 与 targets 对齐。
- [ ] 能区分 loss mask 与 attention mask：不计算 loss 不代表不能被注意到。
- [ ] 能解释 padding 如何被排除出训练目标。
- [ ] 能说明角色边界、结束标记与推理时的助手回复前缀如何编码。

建议产出一张逐 token 对照表，列为：`位置 | token/角色 | 原始 loss mask | input | target | target 是否计入 loss`。至少包含用户消息、助手回复和结束标记；再增加一条工具消息，回到 `render_conversation` 核对每种特殊 token 的处理，避免简单地认为“所有特殊 token 都不训练”。

另一个需要检查的边界：BOS 是一个 token，不天然等于 attention 隔离墙。多个文档或对话拼在同一行时，要看注意力是否真的接收了文档边界 mask，不能仅凭有 BOS 就推断它们互不可见。

## 6. 打通聊天推理

这是第 2B 阶段的第二遍阅读：缓存与采样已经理解，这次主要补齐“聊天协议如何驱动生成”。

阅读顺序：

```text
scripts/chat_cli.py → tokenizer.render_for_completion
                    → Engine.generate → KVCache → GPT.forward
```

文件：[chat_cli.py](scripts/chat_cli.py)、[tokenizer.py](nanochat/tokenizer.py)、[engine.py](nanochat/engine.py)、[gpt.py](nanochat/gpt.py)。

结合 [tests/test_engine.py](tests/test_engine.py) 理解推理实现的预期行为。

检查点：

- [ ] 能区分 prefill 与逐 token decode 的输入和计算过程。
- [ ] 能指出 KV cache 位置何时推进，RoPE 如何取得对应位置。
- [ ] 能追踪采样、停止条件以及工具返回内容如何进入序列。
- [ ] 能区分 `GPT.generate` 与 `Engine.generate` 的职责。

## 7. 理解评估结果

阅读顺序：

1. [nanochat/loss_eval.py](nanochat/loss_eval.py)：bits per byte。
2. [scripts/base_eval.py](scripts/base_eval.py) 与 [nanochat/core_eval.py](nanochat/core_eval.py)：基座模型评估。
3. [scripts/chat_eval.py](scripts/chat_eval.py) 与具体 [tasks](tasks/)：聊天模型任务评估。

检查点：

- [ ] 能区分训练 loss、bits per byte、CORE 与任务正确率。
- [ ] 能解释为什么评估时不能只看训练 loss。
- [ ] 能追踪一个任务从输入构造到预测判分的完整过程。

建议先挑一个任务完整追踪输入、预测和评分，再横向比较其他任务。记录它用的是候选答案打分还是自由生成，以及停止条件、答案提取和样本数量如何影响结果。评估 loss 与 tokenization 有联系，不能脱离词表和数据直接比较；BPB 则把信息量按文本字节数归一化，阅读时追踪实际分母的构造。

## 8. 按兴趣深入 RL 与性能优化

| 方向 | 阅读入口 | 重点 |
| --- | --- | --- |
| RL | [scripts/chat_rl.py](scripts/chat_rl.py)、[tasks/gsm8k.py](tasks/gsm8k.py) | 采样 → reward → 组内减均值的 advantage → 策略梯度更新；注意生成 token 的 mask |
| 优化器 | [nanochat/optim.py](nanochat/optim.py) | Muon、AdamW、参数分组与分布式更新 |
| 注意力性能进阶 | [nanochat/flash_attention.py](nanochat/flash_attention.py)、[tests/test_attention_fallback.py](tests/test_attention_fallback.py) | 2A 已完成语义阅读；此时再研究后端选择、数值容差，以及融合计算如何减少中间张量与访存 |
| 数值精度 | [nanochat/common.py](nanochat/common.py)、[nanochat/fp8.py](nanochat/fp8.py) | 计算 dtype、权重 dtype 与 FP8 训练路径 |
| 推理性能 | [scripts/infer_bench.py](scripts/infer_bench.py) | prefill、decode 的延迟、吞吐与显存 |
| 规模实验 | [runs/scaling_laws.sh](runs/scaling_laws.sh)、[runs/miniseries.sh](runs/miniseries.sh) | 模型规模、训练预算与评估结果之间的关系 |

需要确认边界行为时，再查阅相应的 [tests](tests/)。

建议在选修前回读一次 `gpt.py`，逐个研究 value embedding、smear、backout 与残差系数：先画信息路径，再找参数初始化和优化器分组，最后提出可验证的消融问题。代码能说明“它做了什么”，但单靠代码不能证明“它为什么提高效果”。

分布式优化是独立的第二遍阅读：从 `common.py` 的进程初始化和 rank 信息，追到 dataloader 分片，再到 `optim.py` 的梯度归约、参数更新与结果收集。不要预设它一定使用你熟悉的 DDP 包装模式，实际跟踪通信调用。

RL 放在 SFT、生成与任务评分之后。先沿 `chat_rl.py` 搞清楚哪些序列由当前策略采样、reward 如何计算、advantage 如何构造、哪些 token 参与更新，再比较其他 RL 方法；不要只根据脚本名称套用 PPO 等算法的完整结构。

## 学习节奏与最小练习

下面按每次约 45–90 分钟安排阅读单元，不是硬性日程。某一阶段检查点没有通过，就用下一次阅读补齐，不必同时打开所有文件。

| 阅读单元 | 内容 | 完成时留下的结果 |
| --- | --- | --- |
| 1（回顾） | 全流程地图、GPT 主干、朴素生成 | 一张张量形状图和训练入口清单 |
| 2 | 第 2A 阶段：注意力接口与 mask | 完整序列、单 token、chunk 三种可见范围 |
| 3–4 | 第 2B 阶段：缓存与生成 | prefill + 两次 decode 的状态表 |
| 5 | tokenizer 第一遍 | 文本到 ID 的路径、词表与特殊 token 笔记 |
| 6–8 | 预训练数据、训练循环、checkpoint | 一个 batch 从读取到参数更新的调用链 |
| 9–10 | SFT 格式、打包与 loss mask | 一条对话的 token/target/mask 对照表 |
| 11 | 聊天推理第二遍 | 用户输入到回复结束的流程，含一条工具路径 |
| 12 | 评估 | 一个任务的输入、预测与判分说明 |
| 后续 | 架构复读及选修 | 每次选一个可回答的问题或小实验 |

阅读不依赖完整训练。优先完成上述手算与状态追踪，环境可用后再选择小实验：

- **因果性**：固定模型与前缀，仅修改后缀，在一致的推理设置下比较前缀位置 logits，检查其是否在数值容差内保持一致。
- **缓存语义**：对同一有效序列，比较完整 forward 的最后位置 logits 与逐步缓存计算的对应 logits。先用全上下文窗口，注意 RoPE 位置和 smear 状态；比较 logits 比比较随机采样文本更容易定位问题。
- **loss mask**：手动构造一小行 logits 和 targets，核对设置为 `-1` 的 target 不参与交叉熵；同时解释该位置作为输入仍可影响后续预测。

这些是可选学习练习，不需要先下载训练集或训练一个模型。运行已有测试前先读文件顶部的依赖和跳过条件；“测试被跳过”不代表已验证该能力。

当前工作区使用 Windows PowerShell，`runs/*.sh` 是 Bash 脚本，应先把它们当作流程说明阅读；若要执行，需要对应的 Bash/Linux 环境并核对依赖。`runcpu.sh` 也包含数据下载和实际训练，并非瞬间完成的单元测试。完整 speedrun 留到理解训练参数、数据路径与硬件要求后再运行。

## 建议的阅读产出

每一阶段留下一张流程图或一小段笔记，记录“输入是什么、输出是什么、状态在哪里更新”。优先完成以下三项：

- [ ] 一张 `GPT.forward` 张量形状图。
- [ ] 一条预训练 batch 从文本到参数更新的调用链。
- [ ] 一条 SFT 对话的 token、target 和 mask 对照表。

完成这三项后，再研究架构改动、Muon 和 FP8，会更容易理解它们影响的是训练流程中的哪一部分。

## 阶段验收：怎样判断自己真正理解了

每个阶段采用同一套递进方法：先合上源码解释概念，再用一个小例子手算，最后回到源码定位证据。环境可用时，用小实验验证预测。仅仅能复述注释，还不足以判断已经掌握。

| 验收节点 | 独立完成的任务 | 如果卡住，回到哪里 |
| --- | --- | --- |
| 模型与朴素生成后 | 画出 ID 到 logits 的形状；解释最后位置预测谁、为什么不能读取未来；手算两次追加 | 第 2 阶段、条件分布与张量基础 |
| 注意力与缓存后 | 给定缓存长度 3、新输入长度 2，画 mask、标 RoPE 位置和缓存写入区间；解释各层为何共享同一个起始位置 | 第 2A、2B 阶段 |
| 预训练后 | 从四个 ID 构造 x/y，说明一次 loss、backward、step 的职责，并计算全局 token batch | 第 3、4 阶段 |
| SFT 与聊天后 | 对一条对话标注哪些 target 被监督，解释推理前缀为何要与训练格式匹配 | 第 5、6 阶段 |
| 评估后 | 解释 loss 改善但某项任务分数不变的可能原因，并提出一个能区分原因的检查 | 第 7 阶段 |

检查点不要求背诵行号或 API 参数。允许查具体符号，但应能独立解释因果关系；例如“使用了 causal=True”还不够，还应说明遮住了哪些位置、否则为什么会泄露预测目标。

每次阅读开始用 5 分钟回忆上一阶段的图或状态表，结束时记录一个仍未回答的问题。完成后续两三个阶段再回看旧笔记，补上模块间的连接。遇到卡点，先判断是数学概念、张量形状、状态变化还是项目约定，再只回读相关部分。

可复用的简短笔记模板：

```text
模块解决的问题：
输入/输出及形状：
依赖的约定与前提：
修改的状态和修改时机：
最小例子与预期结果：
与上游/下游的连接：
已从源码确认的事实：
尚待实验验证的解释：
```

## 主线结束后的综合练习

选择一条简短的用户/助手对话，写一份从文本到回复的完整说明。先使用符号 ID 进行手算，之后环境具备时再核对真实 tokenizer 的输出；不要编造真实 token ID。

1. **表示**：区分原始文本、角色标记、序列化后的 ID，并标出推理时的助手前缀。
2. **监督**：构造 inputs、targets 和对应 loss mask，说明哪个输入位置预测哪个目标，哪些目标被忽略。
3. **计算**：画出经过模型时的主要形状，并指出 loss mask 与 attention mask 分别作用在哪里。
4. **学习**：说明这些 loss 如何经 backward 产生梯度、优化器如何更新参数；列出继续训练需要恢复的状态。
5. **推理**：使用同样的聊天协议构造 prompt，追踪 prefill、首个采样、两次 decode 和停止条件。
6. **评估**：选择一种可核对的评分方式，说明它评什么、不能说明什么；将质量指标与性能指标分开记录。

再回答四个变化题，检查知识能否迁移：

- 把一个用户 token 的 loss mask 从 0 改成 1：增加了哪个预测目标？为什么这不等于改变 attention 可见范围？
- 把上下文窗口改小：哪些 query/key 关系发生变化？缓存分配是否随之自动变小？
- 把 temperature 改为 0：生成选择如何变化？训练好的参数与 forward logits 是否因此改变？
- 换用另一个 tokenizer：为什么词表大小一致也不能直接保证原模型权重仍然适用？从 ID 对应的语义解释。

如果要通过实验研究模型改动，每次只改变一个因素，记录假设、基线、数据、配置、评价指标和结果；保持训练预算与评估条件可比较。一个小样本结果只能支持有限结论，不能直接证明某架构普遍更好。

完成以上综合练习后，尝试阅读另一个 decoder-only 项目，只寻找 tokenizer、forward、训练目标、生成和评估这五条路径。能用相同框架解释它的共同点与差异，才说明知识已经从“记住 nanochat 文件”迁移到了“理解语言模型系统”。
