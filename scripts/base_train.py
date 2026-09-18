"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

# 阅读导航：先读 single training step，再追踪数据和模型，最后回看配置与性能优化。
# 主线：解析参数 → 初始化设备/模型 → 优化器/数据 → 训练更新 → 评估与保存。
# 模型内部计算见 nanochat/gpt.py；第一遍可跳过 FP8、compile、性能统计和 GC。
# 阅读行号索引（当前版本；后续增删代码后行号可能变化）：
# ① 核心训练更新：第 752–851 行。先理解 loss → backward → step → zero_grad。
# ② 数据输入与标签：第 413–422 行。理解 x、y 的形状和错位预测。
# ③ 模型构建与恢复：第 203–241 行。理解模型尺寸、初始化和 checkpoint 加载。
# ④ 梯度累积与循环状态：第 481–561 行。区分 micro_step 与 step。
# ⑤ 优化器与精度缩放：第 389–412 行；步数与调度器：第 424–479 行。
# ⑥ 完整循环控制、评估与保存：第 562–751 行。关注 last_step 与 break。
# ⑦ 参数入口：第 57–157 行；设备和日志初始化：第 159–194 行。
# ⑧ tokenizer 初始化：第 196–201 行；训练规模推导：第 329–387 行。
# ⑨ 进阶性能部分：FP8 第 243–319 行，compile 第 321–327 行。
# ⑩ 日志与状态更新：第 852–958 行；统计与清理：第 959–975 行。
# 推荐顺序：① → ② → ③ → ④ → ⑤ → ⑥ → ⑦ → ⑧ → ⑨ → ⑩。
# 行号包含各段注释；模型内部前向计算请继续阅读 nanochat/gpt.py 的 GPT.forward。
# -----------------------------------------------------------------------------
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI arguments
# CLI（命令行接口）：例如 python -m scripts.base_train --depth 12 --fp8。
# add_argument(...) 注册参数规则，此时还没有读取命令行输入。
# 第一个字符串是选项名；type 指定输入转换类型（str 字符串、int 整数、float 浮点数）；
# default 是未传入时的默认值；help 是 --help 显示的说明，不承担输入校验。
# 带值选项支持 --depth 12 或 --depth=12；选项名中的 - 在 args 属性中变为 _。
# -1 没有内置的特殊含义，禁用或自动计算等行为由后面的训练代码实现。
# 创建解析器并赋给 parser；description 是帮助页面中显示的脚本简介。
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
# 字符串，默认 dummy：关闭真正的 wandb 日志；其他值作为实验名称，主进程负责记录。
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
# 字符串，默认空字符串表示自动检测设备；可指定 cuda、cpu 或 mps（Apple GPU 后端）。
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# FP8 training
# action 指定解析器遇到选项时执行什么操作；type 则指定如何转换选项后面的输入值。
# 不写 action 时默认为 "store"（保存输入值），如 --depth 12 经 type=int 转换后保存为整数 12。
# store_true 是布尔开关：不写 --fp8 为 False，写了为 True，后面不需要再跟值。
# 示例：python -m scripts.base_train --fp8 得到 args.fp8 == True；不要写成 --fp8 True。
# 其他常见 action："store_false" 出现时保存 False（默认 True）；"append" 将重复选项的值
# 追加到列表（如 --tag a --tag b）；"count" 统计出现次数（如注册 -v 后使用 -vvv 得到 3）。
# 启用符合条件的线性层的 FP8 训练，需要支持的 GPU；非 CUDA 设备会忽略此开关。
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires H100+ GPU)")
# 字符串，默认 tensorwise；choices 列表限制合法输入，其他值会在解析时被拒绝。
# tensorwise 按整个张量缩放，rowwise 按行缩放；仅启用 FP8 时使用此配置。
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)")
# Model architecture
# 整数，默认 20：Transformer block 的层数。
parser.add_argument("--depth", type=int, default=20, help="depth of the Transformer model")
# 整数，默认 64：先用 depth * aspect_ratio 计算宽度，再向上对齐到 head_dim 的整数倍。
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
# 整数，默认 128：每个注意力头的维度；头数 = 对齐后的模型宽度 // head_dim。
parser.add_argument("--head-dim", type=int, default=128, help="target head dimension for attention")
# 整数，默认 2048：最大上下文长度，也是训练序列长度，单位为 token，不是字符。
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
# 字符串，默认 SSSL：逐层循环使用窗口模式；S 为半上下文窗口，L 为完整上下文窗口。
# 两者都遵循因果注意力；模型实现还会强制最后一层使用长窗口。
parser.add_argument("--window-pattern", type=str, default="SSSL", help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')")
# Training horizon (only one used, in order of precedence)
# 以下三项按顺序选择第一个正值来决定训练步数，并非 argparse 互斥参数。
# 整数，默认 -1 表示不直接指定；正值指定优化器更新总步数，一步可能包含多次梯度累积。
parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
# 浮点数，默认 -1.0 表示不采用；按总 FLOPs 预算 / 每步 FLOPs 估算步数，可传 1e18。
parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
# 浮点数输入，默认 12：目标训练 token 数 = 此比例 * 缩放参数量（Transformer 矩阵 + 输出层）。
# 再用目标 token 数 // 全局批量 token 数得到步数；此比例也参与自动批量和权重衰减计算，
# 因此即使显式指定了训练步数，也不能理解为它在后续计算中完全失效。
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)")
# Optimization
# 整数，默认 32：每个设备一次前向/反向传播处理的序列条数；显存不足时可减小。
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.")
# 整数，默认 -1 自动估算：每次优化器更新处理的全局 token 数，包含所有设备和梯度累积。
# 必须是 device_batch_size * max_seq_len * 进程数的整数倍，商就是梯度累积次数。
parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)")
# 浮点数，默认 0.3：嵌入参数的基础学习率；嵌入层把 token ID 转换成向量。
# 各学习率还会受到批量缩放、部分参数组的宽度/分组缩放以及训练调度的影响。
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
# 浮点数，默认 0.008：输出层 lm_head 的基础学习率，该层将隐藏向量映射为词表分数。
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="learning rate for unembedding parameters (Adam)")
# 浮点数，默认 0.28：Muon 参数组的基础权重衰减系数，用于约束权重增长。
# 后面还会按训练规模缩放，并在训练过程中按余弦曲线衰减；不是所有参数组统一使用此值。
parser.add_argument("--weight-decay", type=float, default=0.28, help="cautious weight decay for the Muon optimizer (for weights)")
# 浮点数，默认 0.02：Transformer 矩阵参数使用的 Muon 基础学习率。
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
# 浮点数，默认 0.5：resid_lambdas、x0_lambdas 等可学习缩放系数的基础学习率。
parser.add_argument("--scalar-lr", type=float, default=0.5, help="learning rate for scalars (resid_lambdas, x0_lambdas)")
# 整数，默认 40：学习率预热步数，前 40 步的学习率乘数从 1/40 线性增加到 1。
parser.add_argument("--warmup-steps", type=int, default=40, help="number of steps for LR warmup")
# 浮点数，默认 0.65：最后约 65% 的训练步数用于线性降低学习率，也影响 Muon 动量调度。
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="ratio of iterations for LR warmdown")
# 浮点数，默认 0.05：学习率下降的目标为完整学习率的 5%；最后一次更新通常略高于目标。
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="final LR as fraction of initial LR")
# 整数，默认 -1 从头训练；其他值指定要加载的 checkpoint 步数，恢复模型、优化器和数据状态。
# num_iterations 仍表示训练终点，例如从 2000 恢复并训练到 3000，只再更新 1000 步。
parser.add_argument("--resume-from-step", type=int, default=-1, help="resume training from this step (-1 = disable)")
# Evaluation
# 整数，默认每 250 步评估验证集 bpb（每字节比特数，越低越好）；-1 关闭。
# 启用时，第 0 步和训练结束也会评估。
parser.add_argument("--eval-every", type=int, default=250, help="evaluate val bpb every N steps (-1 = disable)")
# 整数；80*524288 是先计算乘法再传入默认值，即每次验证预算为 41,943,040 token。
# 实际评估量向下取整到完整批次；它不控制训练数据总量。
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="number of tokens to evaluate val loss on")
# 整数，默认每 2000 步和训练结束评估 CORE；-1 关闭，正常情况下第 0 步不执行。
# CORE 对多个任务的准确率进行随机基线校正，再求平均。
parser.add_argument("--core-metric-every", type=int, default=2000, help="evaluate CORE metric every N steps (-1 = disable)")
# 整数，默认每个 CORE 任务最多评估 500 条样本；非正值在评估实现中表示不限制样本数。
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="examples per task for CORE metric")
# 整数，默认每 2000 步和训练结束用固定提示词生成文本，观察模型效果；仅主进程执行，-1 关闭。
parser.add_argument("--sample-every", type=int, default=2000, help="sample from model every N steps (-1 = disable)")
# 整数，正值表示周期性保存 checkpoint；默认 -1 仅在训练结束保存，不是完全不保存。
# 周期性保存会跳过第 0 步以及刚恢复的那一步。
parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
# Output
# 字符串输入，默认 None（表示没有值，不是字符串 "None"）：自定义 checkpoint 子目录名。
# 未指定时使用 d{depth}，例如 d20；最终位于 <base_dir>/base_checkpoints/<目录名>。
parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")
# 真正读取命令行，转换输入类型并填入默认值，返回 Namespace 对象；如 args.device_batch_size。
args = parser.parse_args()
# vars(args) 取得属性字典，copy() 浅复制一份用于日志和 checkpoint；后续自动计算不会更新此快照。
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# Flash Attention status
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3: efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model

# 模型入口：depth 决定层数，depth * aspect_ratio 给出基础宽度，再对齐到 head_dim 的倍数。
# meta 设备只记录形状和类型；后续 to_empty 分配存储，init_weights 初始化权重。
def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# Build the model, move to device, init the weights
model = build_model_meta(args.depth) # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights() # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# Compile the model

# orig_model 与编译后的 model 使用同一套参数，不是两份独立训练的模型。
# 训练输入形状固定，适合编译；生成和部分评估使用原模型以避免形状变化引起重编译。
orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
# 本段推导：参数量 → 目标 token 数 → 全局批量 → 学习率与权重衰减。
# 初读时先跟住 target_tokens 和 total_batch_size，再研究缩放公式。
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # optimal tokens for the model we are about to train

# Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
d12_ref = build_model_meta(12) # creates the model on meta device
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # compute-optimal d12 training horizon in tokens (measured empirically)
B_REF = 2**19 # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nanochat)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
# 具体分组见 GPT.setup_optimizer：矩阵参数使用 Muon，其他参数组使用 AdamW。
# 这里设置基础超参数；训练循环还会按 step 调整学习率、Muon 动量和权重衰减。
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
# 每次 next(train_loader) 返回输入 x、标签 y 和用于续训的数据加载状态。
# x、y 都是 [B, T] 的 token ID 张量：B=device_batch_size，T=max_seq_len。
# 标签相对输入错开一个 token，例如 x=[BOS, 我, 喜欢]，y=[我, 喜欢, 学习]。
# 因此每个位置学习预测下一个 token，标签直接来自原始文本。
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
# num_iterations 是训练更新次数，不是读取批次的次数，也不是 epoch 数。
# 下面依次优先使用：显式步数、FLOPs 预算、目标 token 数除以每步 token 数。
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Learning rate schedule (linear warmup, constant, linear warmdown)
# 返回基础学习率的乘数：线性预热 → 保持 → 线性下降。
# 实际学习率在主循环中设为 initial_lr * 此乘数。
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# Training loop

# 【1. 初始化训练进度】这些变量会随训练变化，因此续训时也需要恢复。
# step 是已执行的训练迭代数；通常一次迭代更新一次参数，FP16 溢出跳过更新时仍会递增。
# val_bpb 保存最近一次验证结果，未验证时为 None；min_val_bpb 从正无穷开始方便比较。
# smooth_train_loss 保存未做偏差修正的 EMA；total_training_time 只累计后面选中的训练耗时。
# Loop state (variables updated by the training loop)
# 未指定续训时，创建一套新的循环状态。
if not resuming:
    # 将训练迭代计数初始化为零。
    step = 0
    # bpb（bits per byte）不是字节本身的位数；一个字节始终等于 8 bit。
    # 它衡量：按模型给出的概率编码真实文本时，平均每个原始字节需要多少理论比特。
    # 对真实 token 的预测概率为 p 时，信息成本为 -log2(p)：p=1/2 对应 1 bit，p=1/4 对应 2 bit。
    # 真实 token 的概率越高，模型越不意外，编码成本越低，因此 bpb 越低越好。
    # 公式：bpb = sum(-log2 P(真实 token | 前文)) / 计入评估的文本字节总数。
    # 同一段 1000 字节的文本，理论成本从 2000 bit 降至 1500 bit，bpb 就从 2.0 降至 1.5。
    # 改变的是模型的预测概率及理论编码成本，不是原始文本大小；这里也没有实际生成压缩文件。
    # loss_eval.evaluate_bpb 用自然对数累计交叉熵，再计算 total_nats / (log(2) * total_bytes)。
    # 除以 log(2) 将单位从 nat 转成 bit，再除以字节数，得到按字节归一化的交叉熵。
    # 评估会排除不计入指标的特殊 token 和被忽略的标签；字节数也不等于字符数或 token 数。
    # 按字节归一化能减少 tokenizer 切分粒度对比较的影响；bpb 不是准确率，也不代表所有任务能力。
    # 所以 val_bpb 记录最近结果，min_val_bpb 记录历史最低值，即该指标上的历史最佳表现。
    # 尚未评估验证集，用 None 表示没有验证结果。
    val_bpb = None # will be set if eval_every > 0
    # 历史最低 bpb 先设为正无穷，方便首次结果参与比较。
    min_val_bpb = float("inf")
    # 初始化训练损失的指数移动平均累积值。
    smooth_train_loss = 0 # EMA of training loss
    # 初始化累计训练计时，单位为秒。
    total_training_time = 0 # total wall-clock time of training
# 已有 checkpoint，进入恢复训练状态的分支。
else:
    # 续训沿用原 step 和统计量，不从零计数；num_iterations 仍然是总终点。
    # 例如从 step=2000 恢复到 num_iterations=3000，后面只再执行 1000 次迭代。
    # 从 checkpoint 元数据恢复迭代计数。
    step = meta_data["step"]
    # 取出 checkpoint 中保存的循环状态字典。
    # loop_state 是本项目定义的普通 Python 字典，不是 Python/PyTorch 的特殊机制。
    # 保存 checkpoint 时，将 min_val_bpb、smooth_train_loss、total_training_time 装进该字典。
    # 此处从 meta_data 取出它，再把其中的值赋回下面三个训练变量，相当于恢复数据的中转容器。
    # 从头训练不需要这个容器：上面的 if 分支直接给同样的三个变量赋初始值即可。
    # 两条分支最终都准备好训练循环需要的变量，区别只在值来自初始化还是 checkpoint。
    # 循环中更新的是独立变量，不是 loop_state 字典；这些数值重新赋值后不会自动同步回字典。
    # 下次保存时，save_checkpoint 的参数中会用变量的最新值重新构造一个 loop_state 字典。
    # step 和 val_bpb 也是训练状态，只是作者将它们放在 meta_data 外层，属于存储结构的选择。
    loop_state = meta_data["loop_state"]
    # 恢复最近一次验证结果，可能仍为 None。
    val_bpb = meta_data["val_bpb"]
    # 恢复历史最低验证 bpb。
    min_val_bpb = loop_state["min_val_bpb"]
    # 恢复 EMA 累积值，使损失曲线延续之前的历史。
    smooth_train_loss = loop_state["smooth_train_loss"]
    # 恢复此前累计的有效训练耗时。
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
# 梯度累积：多次小批次前向/反向后才更新一次参数，以控制单次显存占用。
# 注意单位：device_batch_size 是序列条数，total_batch_size 是全局 token 数。
# 全局 token 数 = 每卡序列数 × 序列长度 × 进程数 × 累积次数。
# 例如 4 × 1024 × 2 × 8 = 65536 token，累积 8 次后更新一次。
# 计算单个进程一次前向和反向处理的 token 数：序列条数乘序列长度。
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
# 乘以进程总数，得到所有卡一个微步合计处理的 token 数。
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
# 要求一次全局更新恰好包含整数个微批次，否则无法按当前设置完成梯度累积。
# 检查全局 batch 是否能由整数个微步组成，不满足时立即报错。
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
# 用全局目标 token 数整除每个微步的 token 数，得到累积次数。
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
# 由主进程输出单卡微批次的尺寸和 token 数，冒号后的逗号表示千位分隔。
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
# 由主进程输出所有卡一个微步的 token 总数。
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
# 输出全局 batch 对应的梯度累积次数，便于核对配置。
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
# step 表示已完成的训练更新次数；循环开头评估和保存的是当前参数。
# 最后一次进入循环只做最终评估/保存，随后 break，不会多更新一次参数。
# 持续执行循环，直到内部的 break 明确结束训练。
while True:
    # 【2. 循环入口】先检查当前参数的效果并按需保存，再决定是否还需要训练。
    # 从零开始时循环进入 N+1 次，但训练部分只执行 N 次；最后一次只做收尾。
    # 判断是否已经达到指定的总迭代数，供最终评估、保存和退出共用。
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    # 这是按 token 数估算的累计训练计算量，不含评估/生成，且不扣除溢出跳过的更新。
    # 估计当前累计训练 FLOPs：每 token 计算量乘每步 token 数再乘已执行步数。
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 【3. 验证集评估】eval_every > 0 才启用；第 0 步、每隔 N 步和最终步都会评估。
    # 所有 rank 都参与：evaluate_bpb 内部需要汇总各卡的损失与字节数。
    # bpb 是每字节比特数，越低越好；它与下面训练日志中的平均 token 交叉熵不是同一指标。
    # once in a while: evaluate the val bpb (all ranks participate)
    # 只有启用验证且到达间隔或终点时进入；第 0 步也满足取余条件。
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        # 切换模型到评估模式；此调用本身不关闭自动求导。
        model.eval()
        # eval() 只切换模块行为，本身不会关闭梯度；evaluate_bpb 的 no_grad 装饰器负责禁用梯度。
        # 每次重新创建验证加载器，使评估从该加载器的起点开始，而不是接着上次读取。
        # 为本次验证新建数据加载器。
        val_loader = build_val_loader()
        # 每轮评估批次覆盖 B*T*world_size 个 token；整除向下取整，因此实际数量可能小于预算。
        # 把验证 token 预算换算成完整的多卡批次数，余数部分不评估。
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        # 上下文内临时替换 FP8 线性层，退出时恢复；未启用 FP8 时不做替换。
        # 在本作用域内临时关闭模型的 FP8 线性层实现，结束后恢复。
        with disable_fp8(model):
            # 执行指定批次数的验证，并接收跨卡汇总后的每字节比特数。
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        # 打印验证结果：步数补零到五位，bpb 保留六位小数。
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        # 只记录历史最低验证值；这里不会立即保存“最佳模型”，保存由后面的独立条件控制。
        # 比较本次验证结果是否优于历史最低值。
        if val_bpb < min_val_bpb:
            # 将本次更低的验证结果记录为历史最佳值。
            min_val_bpb = val_bpb
        # 构造并提交本次指标字典；DummyWandb 会忽略此调用。
        wandb_run.log({
            # 记录迭代索引，用于对齐训练进度
            "step": step,
            # 记录累计训练计算量估计
            "total_training_flops": flops_so_far,
            # 记录累计有效训练秒数
            "total_training_time": total_training_time,
            # 记录本次验证集 bpb
            "val/bpb": val_bpb,
        # 结束指标字典，并完成本次日志提交。
        })
        # 结束评估或采样，恢复模型训练模式。
        model.train()

    # 【4. 任务评估】与验证损失不同，CORE 用一组任务衡量模型能力。
    # 正常训练时跳过 step=0，在指定间隔和最后一步执行；禁用时连最终评估也跳过。
    # orig_model 与 model 共享参数，使用原模型可避免变化的输入形状触发反复编译。
    # 每段评估结束后调用 train()，恢复后续训练的模块模式。
    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    # 先清空本轮任务评估结果，避免沿用上轮数据。
    results = {}
    # 启用 CORE 后，在非零间隔步或最终步执行任务评估。
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        # 切换模型到评估模式；此调用本身不关闭自动求导。
        model.eval()
        # 临时关闭原模型的 FP8 实现，供任务评估或文本生成使用。
        with disable_fp8(orig_model):
            # 运行 CORE 任务评估，并用 max_per_task 限制每个任务的样本数量。
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        # 由主进程打印 CORE 综合指标，保留四位小数。
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        # 构造并提交本次指标字典；DummyWandb 会忽略此调用。
        wandb_run.log({
            # 记录迭代索引，用于对齐训练进度
            "step": step,
            # 记录累计训练计算量估计
            "total_training_flops": flops_so_far,
            # 记录 CORE 综合分数
            "core_metric": results["core_metric"],
            # 记录按随机基线校正后的各任务结果
            "centered_results": results["centered_results"],
        # 结束指标字典，并完成本次日志提交。
        })
        # 结束评估或采样，恢复模型训练模式。
        model.train()

    # 【5. 文本采样】只让主进程打印固定提示词的续写，方便观察训练前后的变化。
    # 这段只生成文本，不调用 backward 或 optimizer.step，不用生成结果训练模型。
    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    # 启用采样后，仅主进程在非零间隔步或最终步生成示例。
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        # 切换模型到评估模式；此调用本身不关闭自动求导。
        model.eval()
        # 创建固定提示词列表，便于比较不同训练阶段的续写。
        prompts = [
            # 地理事实提示：让模型续写法国首都。
            "The capital of France is",
            # 化学事实提示：让模型续写金的元素符号。
            "The chemical symbol of gold is",
            # 日期推理提示：让模型推断明天是星期几。
            "If yesterday was Friday, then tomorrow will be",
            # 反义词提示：让模型续写 hot 的反义词。
            "The opposite of hot is",
            # 列表续写提示：让模型列出太阳系行星。
            "The planets of the solar system are:",
            # 开放式续写提示，用于观察自然文本生成。
            "My favorite color is",
            # 简单代数提示：让模型续写方程的解。
            "If 5*x + 3 = 13, then x is",
        # 结束提示词列表的定义。
        ]
        # 用原模型和 tokenizer 创建生成引擎，避免变长生成反复触发模型编译。
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        # 依次对每一个固定提示词生成一条示例。
        for prompt in prompts:
            # 先把字符串编码为 token ID，并在开头添加序列起始标记 BOS。
            # 将提示词编码为 token ID，并在前面加入 BOS 起始标记。
            tokens = tokenizer(prompt, prepend="<|bos|>")
            # 临时关闭原模型的 FP8 实现，供任务评估或文本生成使用。
            with disable_fp8(orig_model):
                # 生成一条序列，最多新增 16 个 token；temperature=0 使用贪心选择。
                # sample[0] 是第一条结果，后续 decode 把 token ID 还原成可读文本。
                # 生成一条最多新增 16 个 token 的结果；温度为零采用贪心选择，忽略第二个返回值。
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            # 把第一条生成结果解码为文本并打印。
            print0(tokenizer.decode(sample[0]))
        # 结束评估或采样，恢复模型训练模式。
        model.train()

    # 【6. 保存续训材料】最终步总会保存；周期保存需开启，并跳过第 0 步和刚恢复的步。
    # 所有 rank 都调用：save_checkpoint 内部仅由 rank 0 保存权重和元数据，各 rank 保存各自优化器分片。
    # 权重用于恢复预测能力；优化器状态用于恢复动量等；元数据用于恢复配置和训练进度。
    # val_bpb 是最近一次验证值，不保证恰好在当前保存步重新评估过。
    # 数据状态记录读取位置，但加载器采用近似恢复，并未完整保存预取批次和内部缓冲区。
    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    # 最终步必保存；周期保存需满足正步数、不是刚恢复的步、已启用且达到间隔。
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        # 调用保存函数，下面依次传入目录、步数、模型状态、优化器状态和元数据。
        save_checkpoint(
            # 指定 checkpoint 文件写入的目录。
            checkpoint_dir,
            # 将当前迭代数作为 checkpoint 标识，参与保存文件命名。
            step,
            # 提取原模型参数及持久缓冲区，避免编译包装影响状态键名。
            orig_model.state_dict(), # model parameters
            # 提取优化器参数组和动量等状态，分布式时保存本进程对应的状态。
            optimizer.state_dict(), # optimizer state
            # 开始构造用于保存 JSON 元数据的字典。
            { # metadata saved as json
                # 记录迭代索引，用于对齐训练进度
                "step": step,
                # 保存最近一次验证结果
                "val_bpb": val_bpb, # loss at last step
                # 保存模型结构配置，供恢复时重建模型
                "model_config": model_config_kwargs,
                # 保存启动时的参数快照，不是所有自动推导后的最终值
                "user_config": user_config, # inputs to the training script
                # 保存每个设备的微批次序列条数
                "device_batch_size": args.device_batch_size,
                # 保存训练序列长度
                "max_seq_len": args.max_seq_len,
                # 保存实际使用的全局 token 批量
                "total_batch_size": total_batch_size,
                # 保存加载器读取进度，供近似恢复数据位置
                "dataloader_state_dict": dataloader_state_dict,
                # 开始保存随训练变化的循环状态子字典
                "loop_state": { # all loop state (other than step) so that we can resume training
                    # 保存历史最低验证 bpb
                    "min_val_bpb": min_val_bpb,
                    # 保存尚未做偏差修正的 EMA 累积值
                    "smooth_train_loss": smooth_train_loss,
                    # 记录累计有效训练秒数
                    "total_training_time": total_training_time,
                # 结束当前嵌套字典，逗号用于继续外层字段或函数参数。
                },
            # 结束当前嵌套字典，逗号用于继续外层字段或函数参数。
            },
            # 传入当前进程编号，使保存函数区分公共文件与各 rank 的优化器文件。
            rank=ddp_rank,
        # 结束 save_checkpoint 的多行参数列表并完成调用。
        )

    # 【7. 到达终点就退出】必须放在最终评估和保存之后、下一次训练更新之前。
    # termination conditions (TODO: possibly also add loss explosions etc.)
    # 评估和保存完成后，再检查是否应终止循环。
    if last_step:
        # 跳出 while 循环，进入文件末尾的统计和清理代码。
        break

    # -------------------------------------------------------------------------
    # 核心阅读区：前向算 loss → 反向累积梯度 → 更新参数 → 清空梯度。
    # GPT.forward(x, y) 内部完成词表预测和交叉熵计算，因此 model(x, y) 直接返回标量 loss。
    # single training step
    # evaluate the gradient
    # CUDA 上等待设备任务完成，使计时准确；本脚本在其他设备上使用空操作。
    synchronize()
    # 前面的 synchronize 在 CUDA 上等待已有工作完成，避免把之前异步任务算进本步耗时。
    # 计时范围包括梯度累积、取下一批数据和优化器更新，不包括循环前面的评估/保存。
    # 记录本次训练更新开始的墙钟时间，单位为秒。
    t0 = time.time()
    # 【8. 累积本步梯度】首次 x、y 已在循环外取得；后续由每次迭代末尾的 next 提供。
    # 各微批次之间不清空梯度，也不更新参数，因此它们都使用本次更新前的权重。
    # 依次处理本次更新所需的微批次，循环内只累积梯度。
    for micro_step in range(grad_accum_steps):
        # 用当前输入和下一 token 标签前向计算，得到平均交叉熵标量。
        loss = model(x, y)
        # detach 为日志取值分离计算图；原来的 loss 仍用于反向传播。
        # 保留未除以累积次数的损失用于日志，并将其与计算图分离。
        train_loss = loss.detach() # for logging
        # backward 会累加梯度；除以累积次数，使结果对应各微批次损失的平均。
        # backward 只计算梯度，后面的 optimizer.step 才更新参数。
        # 按累积次数缩小每份损失，使各微批次的梯度之和对应平均损失。
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        # 若已创建 FP16 梯度缩放器，则走带缩放和溢出检测的分支。
        if scaler is not None:
            # FP16 分支先放大损失再反向，减轻小梯度下溢；真正更新前还要还原梯度尺度。
            # 先放大损失再反向传播，把缩放后的梯度累加到参数的 grad。
            scaler.scale(loss).backward()
        # 未使用梯度缩放器，直接对损失进行反向传播。
        else:
            # 直接反向传播，把本微批次的梯度累加到参数的 grad。
            loss.backward()
        # 提前取得下一微批次；最后一次也会取，因此本步结束时下一步的数据已准备好。
        # GPU 工作异步提交后，CPU 可继续准备数据；实际重叠程度取决于加载和拷贝实现。
        # 读取下一批输入、标签和加载状态，供下一微步或下一训练步使用。
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    # 所有微批次完成后，设置本步超参数，并进行一次优化器更新。
    # step the optimizer
    # 【9. 设置本步更新规则】调度器用当前 step 计算学习率乘数、Muon 动量和权重衰减。
    # 计算当前步的学习率乘数。
    lrm = get_lr_multiplier(step)
    # 计算当前步 Muon 参数组的动量系数。
    muon_momentum = get_muon_momentum(step)
    # 计算当前步 Muon 参数组的权重衰减系数。
    muon_weight_decay = get_weight_decay(step)
    # 遍历优化器的各参数组，为不同类别参数设置更新规则。
    for group in optimizer.param_groups:
        # 始终从 initial_lr 计算当前学习率，避免在上一步 lr 上重复相乘导致意外衰减。
        # 只为 kind=muon 的参数组更新下方动量和权重衰减，其他组保留自己的设置。
        # 将该组初始学习率乘以当前调度倍数，写入本步实际学习率。
        group["lr"] = group["initial_lr"] * lrm
        # 只有 Muon 参数组才更新下面的动量和权重衰减设置。
        if group['kind'] == 'muon':
            # 为当前 Muon 参数组写入本步动量。
            group["momentum"] = muon_momentum
            # 为当前 Muon 参数组写入本步权重衰减。
            group["weight_decay"] = muon_weight_decay
    # 若已创建 FP16 梯度缩放器，则走带缩放和溢出检测的分支。
    if scaler is not None:
        # 【10. 执行更新】先把放大后的梯度还原，并检查是否出现 inf/nan。
        # 原地还原梯度尺度，并记录梯度是否包含 inf 或 nan。
        scaler.unscale_(optimizer)
        # In distributed training, all ranks must agree on whether to skip the step.
        # Each rank may independently encounter inf/nan gradients, so we all-reduce
        # the found_inf flag (MAX = if any rank found inf, all ranks skip).
        # 多卡必须一致决定是否跳过：任一卡溢出，就将所有卡的 found_inf 标志置为真。
        # 这里同步的是溢出标志；梯度归约和参数同步由本项目的分布式优化器处理。
        # 只有分布式进程组已初始化时，才执行跨进程溢出标志同步。
        if is_ddp_initialized():
            # 遍历此优化器各设备的溢出标志张量；这里调用的是 GradScaler 私有接口。
            for v in scaler._found_inf_per_device(optimizer).values():
                # 对标志取跨进程最大值，任一进程溢出就让所有进程统一跳过更新。
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        # 梯度有限时调用优化器；出现溢出时跳过更新，避免污染参数。
        # 没有溢出才调用优化器更新参数；溢出时跳过本次参数更新。
        scaler.step(optimizer)
        # 根据溢出情况调整下一步的缩放因子；后面的 step 仍会递增，即使此次跳过更新。
        # 根据检测结果调整后续训练使用的梯度缩放因子。
        scaler.update()
    # 未使用梯度缩放器，直接调用优化器更新参数。
    else:
        # 使用已累积的梯度更新参数；分布式优化器内部同时负责相关通信。
        optimizer.step()
    # 更新后清空梯度，防止本次梯度带入下一次更新；下一轮重新累积。
    # 设置 grad=None，而不是给梯度张量填零；下一次 backward 会重新产生梯度。
    # 将模型参数的梯度设为 None，防止本次梯度被下次更新重复使用。
    model.zero_grad(set_to_none=True)
    # 日志取最后一个微批次的 loss，并不是所有微批次或所有卡的平均损失。
    # item() 将设备上的标量取成 Python 数字，在 CUDA 上需要等待相关计算完成。
    # 将最后一个微批次的损失取为 Python 数值，CUDA 上会等待相关结果就绪。
    train_loss_f = train_loss.item() # .item() is a CPU-GPU sync point
    # CUDA 上等待设备任务完成，使计时准确；本脚本在其他设备上使用空操作。
    synchronize()
    # 此时已完成末尾同步，dt 才能覆盖 GPU 的实际执行，而不只是 CPU 提交任务的时间。
    # 记录设备同步后的结束时间。
    t1 = time.time()
    # 得到本次完整训练更新耗时，单位为秒。
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 【11. 记录速度与损失】这一段只读取训练结果，不再修改模型参数。
    # logging (CPU action only)
    # EMA 只平滑日志中的损失曲线，不参与反向传播，不改变模型更新。
    # 设置 EMA 历史权重为 0.9，当前损失权重为 0.1。
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    # 用旧 EMA 的 90% 加当前损失的 10%，平滑日志曲线。
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # EMA the training loss
    # EMA 初始为 0，早期结果会偏小；除以 1-beta^(step+1) 做初始偏差修正。
    # 修正 EMA 从零初始化造成的早期偏小，得到用于显示的损失。
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    # 这里尚未 step += 1，所以进度显示沿用本轮开始时的 step；首轮更新后仍显示 step=0。
    # 用递增前的 step 计算训练完成百分比。
    pct_done = 100 * step / num_iterations
    # 吞吐量使用所有卡本步处理的 token 数除以耗时；FLOPs/s 同样是全局估算。
    # 用全局 batch token 数除以本步耗时，取整得到每秒 token 吞吐。
    tok_per_sec = int(total_batch_size / dt)
    # 用本步估计 FLOPs 除以耗时，得到全局估计计算速度。
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    # MFU 用估计计算速度除以所有 GPU 的理论峰值；它不是显存占用率，也不是监控工具的 GPU 利用率。
    # 非 CUDA 设备前面将峰值设为无穷，此数值没有实际性能比较意义。
    # 将估计计算速度除以所有卡的理论峰值并乘 100，得到 MFU 百分比。
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    # 按实际条件排除 step=0..10（共 11 次）的耗时，减轻首次编译等启动开销的影响。
    # 所以 total_training_time 并不是程序启动以来的总墙钟时间，也不含评估与保存。
    # 只在索引大于 10 时开始累计训练时间，跳过 step=0 到 10。
    if step > 10:
        # 将本次耗时加入累计值，不计评估、保存和日志输出的时间。
        total_training_time += dt # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    # 当 step=11 时首次累计耗时，steps_done=1，与上述计时条件对应。
    # 计算当前已经纳入有效计时的步数。
    steps_done = step - 10
    # 至少有一个有效计时样本时，才估算剩余时间。
    if steps_done > 0:
        # 用累计计时除以计时步数，得到平均每步秒数。
        avg_time_per_step = total_training_time / steps_done
        # ETA 按已计时训练步的平均速度估算，不含未来评估/保存开销。
        # 此处仍用递增前的 step，remaining_steps 比本轮更新后实际剩余迭代数多 1。
        # 用总步数减递增前的 step 估算剩余步数，较本轮完成后的实际剩余多一。
        remaining_steps = num_iterations - step
        # 用剩余步数乘平均每步耗时，估计剩余训练秒数。
        eta_seconds = remaining_steps * avg_time_per_step
        # 将 ETA 换算成分钟，保留一位小数用于日志拼接。
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    # 尚无有效计时步，暂不估算和显示 ETA。
    else:
        # 计时样本不足时不显示 ETA。
        eta_str = ""
    # pq_idx 是数据分片位置，rg_idx 是 Parquet 行组位置；这里展示已预取数据的加载状态。
    # 把数据轮次、Parquet 文件索引和行组索引拼接为进度字符串。
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    # 主进程打印本轮损失、学习率乘数、毫秒耗时、吞吐、MFU 和数据进度等。
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    # 控制台每步输出；wandb 每 100 步记录一次，未启用日志的进程使用 DummyWandb。
    # 每 100 个步索引提交一次训练日志，包括 step=0。
    if step % 100 == 0:
        # 构造本次训练指标字典，随后统一提交给日志对象。
        log_data = {
            # 记录迭代索引，用于对齐训练进度
            "step": step,
            # 记录累计训练计算量估计
            "total_training_flops": flops_so_far,
            # 记录累计有效训练秒数
            "total_training_time": total_training_time,
            # 记录偏差修正后的平滑训练损失
            "train/loss": debiased_smooth_loss,
            # 记录当前学习率乘数
            "train/lrm": lrm,
            # 记录本步训练耗时，单位秒
            "train/dt": dt,
            # 记录全局 token 吞吐
            "train/tok_per_sec": tok_per_sec,
            # 记录估计 MFU 百分比
            "train/mfu": mfu,
            # 记录数据轮次及文件、行组进度
            "train/epoch": epoch,
        # 结束训练日志指标字典。
        }
        # 提交训练指标；未启用 wandb 的进程使用空实现。
        wandb_run.log(log_data)

    # 【12. 推进进度】先判断是否是本次进程启动后的第一步，再递增 step。
    # 续训时 first_step_of_run 也会成立一次，便于清理重新加载产生的临时对象。
    # state update
    # 判断当前迭代是否为本次启动后的第一步，包含从 checkpoint 恢复的首步。
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    # 迭代计数加一；即使 FP16 溢出跳过参数更新，此计数仍前进。
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    # 【13. 减少 Python GC 扫描开销】先收集初始化遗留对象，再冻结存活对象并关闭自动循环垃圾回收。
    # 这不是释放所有 GPU 显存；普通引用计数仍然工作，后续每 5000 步手动做一次循环垃圾回收。
    # 本次运行首次训练结束时，执行一次初始化对象清理与 GC 设置。
    if first_step_of_run:
        # 手动运行 Python 垃圾回收，清理不可达的循环引用对象。
        gc.collect() # manually collect a lot of garbage from setup
        # 将当前 GC 跟踪的存活对象移到永久代，避免后续循环垃圾回收反复扫描。
        gc.freeze() # immediately freeze all currently surviving objects and exclude them from GC
        # 关闭自动循环垃圾回收；普通引用计数释放仍然有效。
        gc.disable() # nuclear intervention here: disable GC entirely except:
    # 非首次迭代时，每完成 5000 步手动触发一次垃圾回收。
    elif step % 5000 == 0: # every 5000 steps...
        # 手动运行 Python 垃圾回收，清理不可达的循环引用对象。
        gc.collect() # manually collect, just to be safe for very, very long runs

# 【14. 循环结束】输出显存峰值和累计计时；只有做过验证才输出最低 bpb。
# 下面关闭日志会话并清理分布式运行环境。
# print a few more stats
# 输出峰值张量显存分配量，以 MiB 表示；非 CUDA 时本脚本返回零。
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
# 将累计有效训练秒数换算为分钟并输出。
print0(f"Total training time: {total_training_time/60:.2f}m")
# 只有至少取得过一次验证结果，才打印最低验证 bpb。
if val_bpb is not None:
    # 输出运行中记录的最低验证 bpb，保留六位小数。
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# cleanup
# 结束日志会话，让日志后端完成收尾。
wandb_run.finish() # wandb run finish
# 清理计算环境，包括已初始化的分布式进程组。
compute_cleanup()
