"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on incompatible CUDA GPUs, MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F

# 阅读导航（建议顺序）：
# 1.【精读】flash_attn_func：统一接口、张量布局转换、GQA 入口。
# 2.【精读】_sdpa_attention：因果 mask、窗口、query/key 位置对齐。
# 3.【精读】flash_attn_with_kvcache：新 K/V 写入和有效历史的读取。
# 4.【略读】后端加载、选择与末尾导出：理解职责即可，不必深入硬件适配。
# 本文件是后端适配层；真正的 FA3 内核和 PyTorch SDPA 实现在外部，首轮不必追入。
# fallback 中的 mask 与缓存决定模型能读取哪些位置，仍是需要精读的模型语义。
# Q/K/V 投影、RoPE 和输出投影在 gpt.py 中完成，本文件接收已处理好的 Q/K/V。


# =============================================================================
# Detection: Try to load FA3 on CUDA GPUs
# =============================================================================
# 【略读：后端加载】成功返回 FA3 接口对象；不可用或加载失败则返回 None。
# 内核来源与设备兼容细节可暂时跳过；下面的硬件注释反映编写时情况。
def _load_flash_attention_3():
    """Try to load Flash Attention 3."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are currently compiled for Hopper (sm90), Ada (sm89) and Ampere (sm80/sm86)
        # Blackwell (sm100) needs SDPA fallback until FA3 is recompiled or FA4 is released
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        if major == 9:
            hf_kernel = "varunneal/flash-attention-3"
            return get_kernel(hf_kernel).flash_attn_interface
        else:
            hf_kernel = "kernels-community/flash-attn3"
            if has_kernel(hf_kernel):
                return get_kernel(hf_kernel).flash_attn_interface
            else:
                return None

    # 将加载过程的异常转换为“FA3 不可用”，让后续代码选择 SDPA。
    # 这里没有记录错误原因，因此 None 本身不能说明是哪一步失败。
    except Exception:
        return None


# 模块导入时尝试加载一次；HAS_FA3 表示加载成功，不代表最终一定使用 FA3。
_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


# 【略读：选择策略】结合测试覆盖开关、可用性和 COMPUTE_DTYPE 决定后端。
# 本项目的自动选择仅在 FA3 可用且计算 dtype 为 bfloat16 时使用 FA3。
def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

# 此处计算并保存选择结果，并不是每次 attention 调用都重新检测。
# 之后只修改 _override_impl 不会自动更新 USE_FA3，需要重新解析并赋值。
USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
# 【精读 2：注意力可见范围】先区分三条分支，再手算最后一条分支的 mask。
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    # 此时 q 为 (B,Hq,Tq,D)，k/v 为 (B,Hkv,Tk,D)。输出对应 query 的位置。
    # 缓存推理时，Q 只包含当前输入；K/V 包含历史和当前输入，所以 Tq 可以小于 Tk。
    # 本 helper 按因果场景实现，只读取左窗口；不提供任意右窗口或双向注意力语义。
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]  # 左侧最多可见多少个位置；负数表示不限制历史范围。

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        # 分支 1：Q/K 从同一起点对齐，且窗口不限制上下文。
        # 用底层因果模式，使 query i 只能读 key 0..i，无需这里手动创建三角矩阵。
        # “全上下文”仍不允许读取未来；Tq=Tk=1 且窗口无限也会先匹配此分支。
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        # 分支 2：唯一 query 位于有效 K/V 序列的末尾，提供的 K/V 不包含未来。
        # 因而可直接关注全部有效 key；窗口有限时先截取最近的历史和当前位置。
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            # 例如 Tk=5、window=2，取位置 [2,3,4]：2 个历史位置 + 当前位置。
            # 这里切片限制本次读取范围，不会缩小或清空调用方的缓存分配。
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        # False 不代表允许看未来，而是此处已没有未来 key 需要遮挡。
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    # 分支 3：显式构造 (Tq,Tk) 的可见关系，所有 batch/head 共用并广播。
    # 约定当前 queries 对应 K/V 序列最后 Tq 个位置；Tk-Tq 是历史长度。
    # 例如已有 3 个历史 token，新输入 2 个：Tq=2、Tk=5，query 位置为 [3,4]。
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)  # (Tq,1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)  # (1,Tk)
    mask = col_idx <= row_idx  # 广播比较得到 (Tq,Tk)；此处 True 表示允许关注。
    # 上例为 [[1,1,1,1,0], [1,1,1,1,1]]，禁止 query 3 读取位置 4。

    # sliding window (left)
    if window >= 0 and window < Tk:
        # 同时满足“不是未来”与“历史距离不超过 window”。
        # 上例 window=2 时，变为 [[0,1,1,1,0], [0,0,1,1,1]]。
        mask = mask & ((row_idx - col_idx) <= window)

    # 已通过 attn_mask 完整表达因果性和窗口，不再额外指定 is_causal=True。
    # 被禁止的位置在 softmax 后权重为 0；具体数值计算由 PyTorch 完成。
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
# 【精读 1：无缓存入口】训练和 GPT.generate 的完整序列重算都可以走此路径。
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    # FA3 分支只转发参数；第一次阅读理解接口即可，不必进入外部内核。
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    # 交换序列维和头维，适配 SDPA；不改变每个 query/key/value 的数值含义。
    # q: (B,T,Hq,D) -> (B,Hq,T,D)，k/v 同理，但头数为 Hkv。
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    # Q 头数与 KV 头数不同则启用 GQA，让多组 query 头共享 KV 头。
    # 合法的头数关系（例如 Hq 可被 Hkv 整除）由调用方配置保证。
    enable_gqa = q.size(1) != k.size(1)
    # 注意：当前 fallback 没有传递 causal 参数，而是使用 helper 的因果逻辑。
    # 因此不能因公开签名默认 causal=False，就认为该路径实现了双向注意力。
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


# 【精读 3：缓存入口】先写入本次 K/V，再用当前 Q 读取有效历史；支持 prefill/decode。
def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    # T_new 是本次输入长度：prefill 通常是整个 prompt，逐 token decode 通常为 1。
    # H 是 query 头数；缓存的头数 H_kv 在 GQA 下可以更少。
    B, T_new, H, D = q.shape
    # pos 是已使用长度/本次写入起点，不是缓存总容量。
    # 此 fallback 只读取第一个 batch 元素的位置，假设整个 batch 的位置一致。
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        # 原地写入调用方持有的缓存，不重新拼接分配历史 K/V。
        # 例如 pos=3、T_new=2，写入下标 3、4；调用方需保证缓存容量足够。
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    # 只取历史 + 本次输入，不能把尚未使用的预分配位置也交给注意力。
    # 本项目正常调用会提供新 k/v；若省略，调用方须保证读取范围已经有有效数据。
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    # q 的长度是 T_new，k/v 的长度是 pos+T_new；helper 由此推导 query 的位置偏移。
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    # 此函数更新 K/V 内容，但不递增 cache_seqlens。
    # gpt.py 在最后一层处理完后统一 advance(T_new)，使各层使用相同的写入起点。
    # 输出只有本次 query 的结果：(B,T_new,Hq,D)，不会输出全部历史位置。
    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
# 【略读：接口导出】提供 flash_attn.flash_attn_func(...) 这样的统一调用方式。
# SimpleNamespace 只是属性容器，不是可训练层，也不会在此执行注意力计算。
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)
