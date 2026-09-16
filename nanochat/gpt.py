"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    # x 的形状为 (B, T, H, D)：批大小、序列长度、注意力头数、每个头的通道数。
    # d = D/2 是通道对的数量；这里要求 D 为偶数。
    d = x.shape[3] // 2
    # 注意：变量 x1、x2 各自是一整个半向量，不是名为“第 1、第 2 通道”的两个标量。
    # 用 a0...a7 表示 D=8 时的单个通道，避免与变量名混淆：
    #   x  = [a0, a1, a2, a3, a4, a5, a6, a7]
    #   x1 = [a0, a1, a2, a3]，x2 = [a4, a5, a6, a7]
    # x1、x2 的形状均为 (B, T, H, D/2)，对应位置组成一对。
    x1, x2 = x[..., :d], x[..., d:]
    # cos/sin 的形状为 (1, T, 1, D/2)，在批次和注意力头维度上广播。
    # 同一位置的 x1[j] 与 x2[j] 共用角度 theta_j；不同 j 可以有不同角度。
    # 下方 * 是逐元素乘法：y1[j] = x1[j]*cos(theta_j) + x2[j]*sin(theta_j)，
    #                       y2[j] = -x1[j]*sin(theta_j) + x2[j]*cos(theta_j)。
    # 例如 j=0：y1[0] = a0*c0 + a4*s0，y2[0] = -a0*s0 + a4*c0。
    # 因此共用角度的通道对为 (a0,a4)、(a1,a5)、(a2,a6)、(a3,a7)。
    # 每对应用矩阵 [[cos, sin], [-sin, cos]]，即旋转 -theta，保持该二维向量的长度。
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    # 将旋转后的两个半向量按原布局拼回 (B, T, H, D)，不是将通道对交错排列。
    # 相邻通道配对也是一种 RoPE 布局，但不能在已有模型中直接替换而不调整 Q/K 权重。
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    # 构建模型结构：创建嵌入表、Transformer 层、输出头、可学习系数和 RoPE buffer。
    # config 控制模型规模及窗口模式；pad_vocab_size_to 控制词表尺寸对齐。
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.

        meta 设备上的张量只有形状和类型，没有实际数据；这里主要定义模型结构。
        在实际设备上分配存储后，还需要调用 init_weights() 填入参数和缓存数据。
        """
        super().__init__()
        self.config = config
        # config 保存模型超参数：n_embd 是隐藏向量宽度，n_layer 是 Transformer 层数。
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        # 向上补齐到 pad_vocab_size_to 的整数倍；新增行只用于计算对齐，不代表新词元。
        # 公式：V_pad = ceil(V / P) * P，其中 V 为真实词表大小，P 默认为 64。
        # 例如 V=100、P=64 时分配 128 行；若 V 已是 P 的整数倍，则无需增加行。
        # Tokenizer 的 ID 映射不变：约定真实 token ID 位于 [0, V)，直接作为嵌入表行号。
        # 例如 token ID=3 就读取 embedding.weight[3]，因此真实 token 使用前 V 行。
        # 补齐行也会初始化，但正常输入不会查到它们；Embedding 本身不判断 token 是否真实，
        # 例如在 128 行表中手动输入 ID=110 仍可查表，只是该行没有对应的真实 token。
        # wte、value_embeds 和 lm_head 均使用补齐尺寸；forward 会将 logits 裁回真实词表大小。
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            # wte：按 token ID 查表，得到 n_embd 维向量；h：依次执行的 Transformer 层。
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        # 将隐藏向量映射为词表中每个 token 的分数；与输入嵌入 wte 不共享权重。
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # 每层进入 block 前：x = resid_lambdas[i] * x + x0_lambdas[i] * x0。
        # resid_lambdas 缩放当前残差流；x0_lambdas 控制重新注入初始表示的比例。
        # 下方 ones/zeros 只是构造时的占位值，实际初值由 init_weights() 设置。
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        # smear_gate 用当前向量的前 24 个通道生成门控；smear_lambda 控制混入总强度。
        # 设 x_t 为归一化后、尚未混入前一个 token 信息的向量，混入公式为：
        #   g_t = smear_lambda * sigmoid(W_gate @ x_t[:24])
        #   x_t_mixed = x_t + g_t * x_{t-1}
        # W_gate 的形状为 (1, 24)，将当前 token 的前 24 个通道映射为一个标量；
        # g_t 随 token 变化，并广播到前一个向量的所有通道，不是逐通道门控。
        # sigmoid 输出在 (0, 1) 内，但 smear_lambda 不受正负限制，因此也可学到相减。
        # smear_lambda 初始为 0，故初始 x_t_mixed = x_t；没有前一个 token 时保持原样。
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        # backout_lambda 是可学习的扣除系数：最终表示减去该系数乘以中间层表示。
        # 公式：x_out = RMSNorm(x_final - backout_lambda * x_mid)。
        # x_final 是最后一个 block 的输出；x_mid 是索引 n_layer // 2 的 block 输出。
        # 层索引从 0 开始，例如 12 层模型缓存的是第 7 层输出。
        # 该参数是所有 token、所有通道共享的标量；初始在归一化前执行 x_final - 0.2 * x_mid。
        # 系数没有范围限制，训练后也可为负；其作用是调整中间表示对最终输出的贡献，
        # “去除低层特征”是设计意图，并不保证只减去某一类特征。
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        # head_dim 是单个注意力头的宽度；kv_dim 是所有 K/V 头拼接后的宽度。
        # value_embeds 按 token ID 提供额外的 V 向量，通过门控加到注意力的 V 上。
        # 仅隔层启用且包含最后一层；ModuleDict 用字符串层号索引各层独立的嵌入表。
        # 第 i 层的查表公式：ve_t^(i) = E^(i)[token_id_t]。
        # E^(i) 的形状为 (padded_vocab_size, kv_dim)，各层独立学习；
        # 同一层内相同 token ID 查出的向量相同，随后通过依赖当前输入的门控调整贡献。
        # 查表输出 (B, T, kv_dim)，在注意力中拆成 (B, T, n_kv_head, head_dim)。
        # 设 x_t 为进入注意力模块的归一化表示，实际混入公式为：
        #   g_t^(i) = 3 * sigmoid(W_gate^(i) @ x_t[:12])
        #   v_mixed[t, h] = (W_V^(i) @ x_t)[h] + g_t^(i)[h] * ve_t^(i)[h]
        # 这里 h 表示拆分后的 K/V 头；每个头有一个 (0, 3) 内的门值，广播到该头所有通道。
        # has_ve 的条件为 i % 2 == (n_layer - 1) % 2，即隔层启用且包含最后一层。
        # 例如 12 层模型启用索引 1、3、5、7、9、11（第 2、4、6、8、10、12 层）。
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        # RoPE 的 cos/sin 是固定位置编码缓存，不是可学习参数；预留长度不等于训练上下文长度。
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    # 在分配实际存储后初始化全模型的参数和位置编码缓存，并调整嵌入表的数据类型。
    # 此函数会覆盖已有权重；加载训练好的权重后，不应再调用它重新初始化模型。
    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=0.8
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=0.4/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # 统一初始化真实数据；@torch.no_grad() 避免将这些赋值记录到梯度计算图。
        # 输入嵌入随后会做 RMSNorm；输出头用很小的标准差，使初始 logits 接近零。
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer 各层使用均匀分布初始化，基础目标标准差为 1/sqrt(n_embd)。
        n_embd = self.config.n_embd
        # 原因：线性层的一个输出 y = sum_j(w_j * x_j)，需要累加 n_embd 个输入分量。
        # 假设初始权重相互独立、均值为 0；输入经 RMSNorm 后 sum_j(x_j²) 约为 n_embd，
        # 则给定输入时 Var(y | x) = sum_j(x_j²) * Var(w_j) ≈ n_embd * Var(w_j)。
        # 令 Var(w_j) = 1/n_embd（即 std = 1/sqrt(n_embd)），可使初始输出方差约为 1，
        # 避免输出幅度随模型宽度增大。例如 n_embd=768 时，权重 std≈0.036，输出 std≈1。
        # U(-s, s) 的标准差为 s/sqrt(3)，因此取 s = sqrt(3/n_embd)。
        # 这里直接设定均匀分布的尺度；代码中没有使用该标准差的高斯初始化。
        # 下方 Q/K/V 使用此尺度，mlp.c_fc 再乘 0.4，其标准差为 0.4/sqrt(n_embd)。
        # 0.4 是进一步缩小 MLP 初始化尺度的选择，并非上述方差推导的必然要求。
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            # Q/K/V 投影使用有界均匀分布；MLP 扩展层 c_fc 再缩小到 0.4 倍。
            # 两个 c_proj 置零，使注意力和 MLP 残差分支初始输出为零，之后可逐步学到非零值。
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        # 多层时从浅到深线性递减：resid 为 1.15 → 1.05，x0 为 0.20 → 0.05。
        # max(n_layer - 1, 1) 防止单层模型除零；单层分别取 1.15 和 0.20。
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        # smear_lambda=0 使前一 token 的混入初始关闭；backout 初始扣除 20% 的中间表示。
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # 门控权重取小的正数；实际门值还取决于输入的正负，不能保证高于中点。
        # 注意力中 gate = 3 * sigmoid(ve_gate(...))，输入接近零时 gate 接近 1.5。
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        # 在实际设备上重新生成 RoPE 缓存，替换构造时可能存在的 meta 占位张量。
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        # 嵌入表可用计算类型节省显存；fp16 路径保留 fp32 权重，以兼容梯度缩放。
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    # 为 seq_len 个位置预计算 RoPE 的 cos/sin，返回两个 (1, seq_len, 1, head_dim/2) 张量。
    # base 控制旋转频率尺度；未指定 device 时使用输入嵌入表所在设备，不产生可学习参数。
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # RoPE 每两个通道组成一对，共用一个旋转角度，因此 head_dim 个通道只需 head_dim/2 个频率。
        # 步长 2 用来生成频率公式中的 2j：inv_freq[j] = 1 / base^(2j/head_dim)，
        # 并不是跳过奇数通道、不对它们编码；这里的 channel_range 用于计算频率，不用于索引输入。
        # 本项目 apply_rotary_emb 将向量分成前后两半配对，而不是将相邻通道配对。
        # 例如 head_dim=8：channel_range=[0,2,4,6]，生成 4 个频率，依次用于
        # (x0,x4)、(x1,x5)、(x2,x6)、(x3,x7)，所以全部 8 个通道都参与旋转。
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # inv_freq：各旋转维度对的角频率；t：位置编号；freqs：位置 × 角频率得到的角度。
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    # 将 window_pattern 循环展开为每层的 (左侧窗口, 右侧窗口) 列表，供注意力计算使用。
    # S 使用短窗口，L 使用 sequence_len 长窗口；右侧为 0，最后一层强制使用长窗口。
    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    # 返回输入嵌入权重所在的 torch.device，便于在相同设备上创建输入和随机数生成器。
    def get_device(self):
        return self.transformer.wte.weight.device

    # 估算训练时每个 token 的浮点运算次数（前向 + 反向），用于计算训练成本或吞吐指标。
    # 包含线性层矩阵乘法和按各层窗口估算的注意力计算；不是运行时间或精确的算子统计。
    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    # 统计所有 Linear 权重的元素总数，作为矩阵乘法计算量估算的基础。
    # 嵌入查表和独立标量不计入；这里返回的是参数数量，不是 FLOPs。
    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    # 估算推理时生成单个 token 的前向 FLOPs，context_len 表示此次注意力的上下文长度。
    # 将线性层计算与各层窗口内的注意力计算相加，不包含反向传播。
    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    # 估算一次处理 num_tokens 个提示词 token（prefill，预填充）的总前向 FLOPs。
    # 因果注意力中，靠前位置可见的 token 较少；可见数量逐渐增长，到窗口上限后保持不变。
    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    # 计算单条序列每增加一个 token，全部层的 K/V 缓存需要存储多少字节。
    # 按层数、K/V 头数、每头维度和计算类型计数；不含批大小及其他缓存对象的开销。
    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    # 估算单条序列一次增量解码读取的 K/V 缓存字节数，各层只计入窗口内的上下文。
    # 这是用于成本分析的逻辑数据量，不是对 GPU 实际显存流量的测量。
    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    # 按输入嵌入、Value Embedding、输出头、Transformer 层参数和其余参数分别计数。
    # 返回包含各组数量及 total 的字典，供 scaling law（模型规模与效果关系）分析使用。
    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    # 将参数分组并创建 MuonAdamW 优化器：嵌入表、输出头和独立系数等使用 AdamW，
    # Transformer 层内矩阵使用 Muon，并按形状分组以便批量处理。
    # 各组设置不同的学习率和权重衰减，部分 AdamW 学习率随模型宽度缩放；返回优化器对象。
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters()) #list会不断调用next()把迭代器中的元素全部取出来
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # 执行模型前向计算：token 嵌入 → smear → 多层 Transformer → backout → 输出词表分数。
    # 提供 targets 时返回交叉熵损失（聚合方式由 loss_reduction 决定）；否则返回 (B,T,vocab_size) logits。
    # 可传入 kv_cache 复用推理上下文；该路径会更新 K/V 缓存、位置及前一个 token 的嵌入。
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # idx 是形状 (B, T) 的 token ID；B 为批大小，T 为本次输入长度。
        # 生成时 idx 也不是 None；T 不包含已缓存的历史长度。
        # 常规缓存生成先输入完整提示词（prefill），随后生成循环每次只传入新 token（decode）。
        # 历史 K/V 可复用，因此不用反复输入整段前文；kv_cache 本身不会自动截短 idx。
        # targets 提供监督标签（-1 忽略）；kv_cache 在推理时复用已有的 K/V。
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        # T0 是本次输入在完整序列中的起始位置，增量解码时据此选取正确的 RoPE 位置。
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        # x 的形状为 (B, T, C)。用当前 token 的前 24 维计算 gate，再混入前一个 token：
        # x'_t = x_t + gate(x_t) * x_{t-1}；所有混合均使用未经 smear 的原始 embedding。
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            # 第一个 token 没有前驱，故从第二个开始计算 gate；lambda 缩放后 gate 不一定在 (0, 1)。
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            # 当前位置 x[:, 1:] 与前驱 x[:, :-1] 对齐；保留首 token，沿序列维拼回 (B, T, C)。
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # 先引用旧缓存供本次使用，再缓存本次最后一个 token 供下一步使用。
            # prev_embedding 是 smear 额外保存的 embedding，不是注意力层的 K/V。
            x_pre_smear = kv_cache.prev_embedding
            # -1: 保留序列维，形状为 (B, 1, C)；此时保存的是 smear 之前的值。
            # 属性重新赋值不改变 x_pre_smear 的指向；后续 x = ... 也不会原地修改旧张量。
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                # 常见于一次输入完整提示词；本分支首 token 保持不变，不使用旧 embedding 缓存。
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                # elif 仅在 T > 1 不成立时检查；非空输入下即 T == 1，两个分支不会同时执行。
                # 这里判断的是单 token 且有历史 embedding；Decode 指常见使用阶段，不是网络模块。
                # 当前唯一 token 用全部序列位置 [:, :, :24] 计算 gate；[:, 1:, :24] 会得到空切片。
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                # 例如当前输入 d、旧缓存为 c：结果是 d + gate(d) * c，缓存中已保存原始 d。
                x = x + gate * x_pre_smear
            # 若 T == 1 且旧缓存为 None，则不混合，但上面仍已保存当前 embedding 供下一步使用。

        # Forward the trunk of the Transformer
        x0 = x  # 初始表示，形状 (B, T, n_embd)：经过归一化和 smear，供各层重新注入。
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None  # 缓存索引为 backout_layer 的 block 输出，供末尾扣除。
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # ve 的形状为 (B, T, kv_dim)；未启用 value embedding 的层传入 None。
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        # logits 是未归一化的词元分数；softcap 用 tanh 平滑限制幅度，避免分数过大。
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    # 简单的单序列自回归生成器：接收 token ID 列表，最多逐个 yield max_tokens 个新 ID。
    # 每步重算完整序列，不使用 KV 缓存；temperature>0 时采样，否则取最高分 token。
    # top_k 可限制候选数量，seed 控制采样随机种子；此函数不检查 EOS 来提前停止。
    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        朴素的自回归流式生成：用已有序列预测下一个 token，再把它接回序列。

        参数：
            tokens: 提示词的 token ID 列表，不是文本；只支持一条序列（B=1）。
                当前 forward 的无 KV 缓存分支要求 T > 1，故生成时至少需两个 ID。
            max_tokens: 最多生成多少个新 token，不包含提示词的长度。
            temperature: 温度；>0 时按概率采样，<=0 时直接选分数最高的 token。
                正温度越低，分布越集中；越高，分布越平缓。1.0 保持原始 logits。
            top_k: 正整数时按最高的 k 个分数筛选候选；None 或非正数不筛选。
            seed: 本次调用使用的局部随机数生成器的种子，辅助复现采样结果。
                不重置全局随机状态，也不保证跨设备或不同运行环境完全一致。

        产出：
            每次 yield 一个 Python int（新 token 的 ID），不包含输入提示词。
            含 yield 的函数是生成器：调用时返回生成器对象，迭代时才执行函数体；
            调用方可逐个解码这些 ID，也可用 list(model.generate(...)) 收集结果。

        注意：每步重新计算完整序列，不使用 KV 缓存，序列越长重复计算越多。
        不检测 EOS（结束标记），不自动截断上下文；调用方可提前停止迭代，
        并需确保每次 forward 的输入长度不超过模型支持的范围。
        inference_mode 在迭代执行时关闭梯度记录等开销，但不会自动调用 model.eval()。
        """
        # 这里只检查容器类型；不会验证每个 ID 的类型、范围或提示词长度。
        assert isinstance(tokens, list)
        # 输入张量和采样用的随机数生成器都放在模型所在设备上。
        device = self.get_device()
        rng = None
        if temperature > 0:
            # 每次 generate 调用在首次迭代时初始化一次；之后各步使用同一随机数流。
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        # 外层 [] 增加 batch 维：(T,) -> (1, T)；long 是嵌入层所需的整数索引类型。
        # 新建张量保存上下文，后面拼接不会修改调用方传入的 tokens 列表。
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            # T = 提示词长度 + 已生成数量；每个位置的 logits 都预测其后的 token。
            logits = self.forward(ids) # (B, T, vocab_size)
            # 只取最后一个位置的预测，即完整当前上下文之后的下一个 token 分数。
            # logits 是未归一化分数，还不是概率；此处 B 始终为 1。
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                # 默认沿最后一维选出最大的 k 个分数，并按降序排列。
                # min 防止 k 超过词表大小；v 为 (B, k)，丢弃不需要的词表索引。
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # v[:, [-1]] 取第 k 大分数，保留 (B, 1) 形状以广播到整个词表。
                # 低于阈值的分数设为 -inf，softmax 后概率为 0，不会被采样。
                # 使用严格小于：若阈值处有并列分数，实际保留的候选可能多于 k 个。
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                # 温度缩放改变分数差距，不改变大小排序，也不改变 top-k 候选集合。
                logits = logits / temperature
                # 沿词表维归一化，各候选概率之和为 1，形状仍是 (B, vocab_size)。
                probs = F.softmax(logits, dim=-1)
                # 每条序列按概率抽取一个 ID，结果形状为 (B, 1)，不是取概率最大项。
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                # 贪心解码，无需随机数；keepdim=True 保留 (B, 1)，方便后续拼接。
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            # 沿序列维追加新 ID：(B, T) + (B, 1) -> (B, T+1)。
            # 下一轮会把刚生成的 token 也作为上下文，这就是“自回归”。
            ids = torch.cat((ids, next_ids), dim=1)
            # B=1 时 next_ids 只有一个元素；item() 将张量转成 Python int。
            token = next_ids.item()
            # 交出一个 ID 并暂停；调用方请求下一项时，从这里继续下一轮循环。
            yield token
