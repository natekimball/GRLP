"""
Generalized RLP prototype (discounted-return advantage estimation)
- Model: Qwen/Qwen3-1.7B-Base (or any causal LM)
- Dataset: HuggingFaceFW/fineweb (use a small subset for testing)
"""

import copy
import os
import atexit
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset, Dataset
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import contextmanager
import bitsandbytes as bnb


# -----------------------------
# Config
# -----------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description="Generalized RLP prototype training")
parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-1.7B-Base", help="Model name or path to saved model")
parser.add_argument("--plot-path", type=str, default="grlp-plots.png", help="Path to save metric plots")
parser.add_argument("--mini-batch-size", type=int, default=1, help="PPO mini-batch size (number of contexts per update step)")
args = parser.parse_args()

MODEL_NAME = args.model_name
DATASET = "HuggingFaceFW/fineweb"
SPLIT = "train"
N_DATA = 10_000
DATA_CACHE_DIR = Path("data/fineweb-10k-tokenized")

# MAX_SEQ_LEN = 256
# HORIZON = 8                             # reward horizon T (small for debug; paper uses long)
# THOUGHT_MAX_TOKENS = 256
# G = 2                                   # number of rollouts per context
# GAMMA = 0.7                             # discount factor
# BATCH_SIZE = 2                          # token-level RLP is expensive; tune for your memory
# MINI_BATCH_SIZE = args.mini_batch_size
# LR = 1e-6
# TAU = 0.999                             # EMA decay
# EPS_CLIP_LOW, EPS_CLIP_HIGH = 0.1, 0.1  # PPO clipping
# PPO_EPOCHS = 2                          # number of policy optimization iterations per batch
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.bfloat16
# COMPILE_MODEL = True
# USE_FLASH_ATTN = True
# TEMPERATURE = 0.7

MAX_SEQ_LEN = 2048
HORIZON = 8                             # reward horizon T (small for debug; paper uses long)
THOUGHT_MAX_TOKENS = 2048
G = 16                                  # number of rollouts per context
GAMMA = 0.7                             # discount factor
BATCH_SIZE = 16                         # token-level RLP is expensive; tune for your memory
MINI_BATCH_SIZE = args.mini_batch_size
LR = 1e-6
TAU = 0.999                             # EMA decay
EPS_CLIP_LOW, EPS_CLIP_HIGH = 0.1, 0.1  # PPO clipping
PPO_EPOCHS = 3                          # number of policy optimization iterations per batch

# DDP Setup
local_rank = int(os.environ.get("LOCAL_RANK", -1))
if local_rank != -1:
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    DEVICE = f"cuda:{local_rank}"
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    IS_DDP = True
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RANK = 0
    WORLD_SIZE = 1
    IS_DDP = False

DTYPE = torch.bfloat16
COMPILE_MODEL = True
USE_FLASH_ATTN = True
TEMPERATURE = 0.7

DEBUG_LOG_PATH = Path("debug.txt")
PLOT_SAVE_INTERVAL = 5
MODEL_SAVE_INTERVAL = 100
METRIC_FIG_PATH = Path(args.plot_path)

#os.environ["TORCH_LOGS"] = "recompiles"
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True  # noqa
torch._dynamo.config.allow_unspec_int_on_nn_module = True

_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_BACKGROUND_FUTURES: List[Future] = []
_EXECUTOR_SHUT_DOWN = False


def _clear_completed_futures() -> None:
    _BACKGROUND_FUTURES[:] = [f for f in _BACKGROUND_FUTURES if not f.done()]


def submit_background_task(fn, *args, **kwargs) -> Future:
    if _EXECUTOR_SHUT_DOWN:
        raise RuntimeError("Background executor already shut down")
    _clear_completed_futures()
    future = _BACKGROUND_EXECUTOR.submit(fn, *args, **kwargs)
    _BACKGROUND_FUTURES.append(future)
    return future


def wait_for_background_tasks() -> None:
    global _EXECUTOR_SHUT_DOWN
    while _BACKGROUND_FUTURES:
        _BACKGROUND_FUTURES.pop(0).result()
    if not _EXECUTOR_SHUT_DOWN:
        _BACKGROUND_EXECUTOR.shutdown(wait=True)
        _EXECUTOR_SHUT_DOWN = True


atexit.register(wait_for_background_tasks)

def print_gpu_memory(prefix: str) -> None:
    """Log CUDA memory stats if a GPU is available."""
    if RANK != 0: return
    if not torch.cuda.is_available():
        print(f"{prefix} GPU memory - CUDA not available")
        return
    torch.cuda.synchronize()
    device_index = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device_index) / 1e6
    reserved = torch.cuda.memory_reserved(device_index) / 1e6
    total = torch.cuda.get_device_properties(device_index).total_memory / 1e6
    print(
        f"{prefix} GPU memory - allocated: {allocated:.1f} MB, "
        f"reserved: {reserved:.1f} MB, total: {total:.1f} MB"
    )


def release_cuda_cache() -> None:
    """Release cached CUDA memory if available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_cpu_pinned(tensor: torch.Tensor) -> torch.Tensor:
    """Detach and move tensor to pinned CPU memory for async transfers."""
    return tensor.detach().to("cpu", non_blocking=True).pin_memory()


class PackedRolloutBuffer:
    __slots__ = ("tokens", "logps", "lengths", "num_rollouts", "max_seq_len")

    def __init__(self, tokens: torch.Tensor, logps: torch.Tensor, lengths: torch.Tensor) -> None:
        self.tokens = tokens
        self.logps = logps
        self.lengths = lengths
        self.num_rollouts = 0
        self.max_seq_len = 0

    def views(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.tokens[: self.num_rollouts, : self.max_seq_len],
            self.logps[: self.num_rollouts, : self.max_seq_len],
            self.lengths[: self.num_rollouts],
        )


_ROLLOUT_BUFFER_POOL: List[PackedRolloutBuffer] = []


def acquire_rollout_buffer(num_rollouts: int, max_seq_len: int, logp_dtype: torch.dtype) -> PackedRolloutBuffer:
    target_seq_len = max(1, max_seq_len)
    for idx, buffer in enumerate(_ROLLOUT_BUFFER_POOL):
        if (
            buffer.tokens.size(0) >= num_rollouts
            and buffer.tokens.size(1) >= target_seq_len
            and buffer.logps.dtype == logp_dtype
        ):
            return _ROLLOUT_BUFFER_POOL.pop(idx)

    tokens = torch.empty((num_rollouts, target_seq_len), dtype=torch.long, pin_memory=True)
    logps = torch.empty((num_rollouts, target_seq_len), dtype=logp_dtype, pin_memory=True)
    lengths = torch.empty((num_rollouts,), dtype=torch.int32, pin_memory=True)
    return PackedRolloutBuffer(tokens, logps, lengths)


def release_rollout_buffer(buffer: PackedRolloutBuffer) -> None:
    buffer.num_rollouts = 0
    buffer.max_seq_len = 0
    _ROLLOUT_BUFFER_POOL.append(buffer)


def pack_rollout_tensors(
    cot_tokens: List[torch.Tensor],
    logprobs: List[torch.Tensor],
) -> PackedRolloutBuffer:
    """Pack variable-length rollout tensors into reusable pinned CPU buffers."""
    if not cot_tokens:
        return acquire_rollout_buffer(0, 1, torch.float32)

    num_rollouts = len(cot_tokens)
    lengths_list = [ct.size(1) for ct in cot_tokens]
    max_seq_len = max(lengths_list, default=0)
    logp_dtype = logprobs[0].dtype if logprobs else torch.float32
    buffer = acquire_rollout_buffer(num_rollouts, max_seq_len, logp_dtype)

    pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    active_seq_len = max(1, max_seq_len)
    buffer.tokens[:num_rollouts, :active_seq_len].fill_(pad_value)
    buffer.logps[:num_rollouts, :active_seq_len].zero_()

    for idx, (ct, lp) in enumerate(zip(cot_tokens, logprobs)):
        seq_cpu = ct.squeeze(0).detach().to("cpu", non_blocking=True)
        lp_cpu = lp.detach().to("cpu", non_blocking=True).to(logp_dtype)
        seq_len = seq_cpu.size(0)
        if seq_len > 0:
            buffer.tokens[idx, :seq_len].copy_(seq_cpu)
            buffer.logps[idx, :seq_len].copy_(lp_cpu)
        buffer.lengths[idx] = seq_len

    if num_rollouts < buffer.lengths.size(0):
        buffer.lengths[num_rollouts:] = 0

    buffer.num_rollouts = num_rollouts
    buffer.max_seq_len = max_seq_len
    return buffer


# -----------------------------
# Helpers
# -----------------------------

def _save_metric_plot(metrics: Dict[str, List[float]]) -> None:
    reward_history = metrics["reward"]
    reward_std_history = metrics["reward_std"]
    cot_length_history = metrics["cot_length"]
    loss_history = metrics["loss"]
    step_history = range(1, len(reward_history) + 1)
    if not reward_history:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(step_history, reward_history, color="tab:blue", label="Avg Reward")
    axes[0].fill_between(step_history,
                         [r - s for r, s in zip(reward_history, reward_std_history)], 
                         [r + s for r, s in zip(reward_history, reward_std_history)], 
                         color="tab:blue", alpha=0.2, label="Reward Std")
    axes[0].legend()
    axes[0].set_ylabel("Avg Reward")
    axes[0].grid(alpha=0.3)
    axes[1].plot(step_history, cot_length_history, color="tab:orange")
    axes[1].set_ylabel("Avg CoT Tokens")
    axes[1].grid(alpha=0.3)
    axes[2].plot(step_history, loss_history, color="tab:green")
    axes[2].set_ylabel("Loss")
    axes[2].set_xlabel("Global Step")
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(METRIC_FIG_PATH, bbox_inches="tight")
    plt.close(fig)


def save_metric_plot(metrics: Dict[str, List[float]]) -> None:
    if RANK != 0: return
    reward_history = metrics.get("reward")
    if not reward_history:
        return
    metrics_snapshot = {key: list(values) for key, values in metrics.items()}
    submit_background_task(_save_metric_plot, metrics_snapshot)


@torch.no_grad()
def _save_model_checkpoint(
    save_dir: Path,
    state_dict: Dict[str, torch.Tensor],
    config,
    generation_config,
    tokenizer,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, save_dir / "pytorch_model.bin")
    config.save_pretrained(save_dir)
    if generation_config is not None:
        generation_config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def schedule_model_save(model, tokenizer, save_dir: str | Path) -> None:
    if RANK != 0: return
    save_path = Path(save_dir)
    raw_model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        state_dict = raw_model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = value.detach().to("cpu", copy=True)
    config_copy = copy.deepcopy(raw_model.config)
    generation_config_copy = copy.deepcopy(getattr(raw_model, "generation_config", None))
    submit_background_task(
        _save_model_checkpoint,
        save_path,
        state_dict,
        config_copy,
        generation_config_copy,
        tokenizer,
    )

@contextmanager
def inference_context(model, with_grad: bool = False, inference_mode: bool = False):
    was_training = model.training
    if with_grad:
        model.train()
    else:
        model.eval()
    try:
        if with_grad:
            with torch.enable_grad():
                yield
        elif inference_mode:
            with torch.inference_mode():
                yield
        else:
            with torch.no_grad():
                yield
    finally:
        model.train(was_training)


def gather_token_logprobs_from_logits(logits, target_ids):
    """
    logits: (B, S, V)
    target_ids: (B, S) tokens that the logits are predicting (i.e. next-token targets)
    returns: per-token log probability shape (B, S)
    """
    # Use cross_entropy for memory efficiency (avoids materializing full log_softmax)
    # F.cross_entropy returns negative log likelihood, so we negate it.
    return -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), 
        target_ids.reshape(-1), 
        reduction='none'
    ).view(target_ids.size())


def compute_teacher_forced_logprobs(model, input_ids, gold_targets, attention_mask=None, *, temperature=1.0, keep_grad=True, inference_mode=False):
    """
    Compute per-token log-probs under teacher forcing for the sequence:
      model(input_ids) where input_ids = prefix + optional cot + gold_seq
    We want the log-probabilities for gold_seq positions (the final len(gold_targets) tokens).
    Returns: tensor shape (B, len(gold_targets)) of log-probs (float)
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
        
    with inference_context(model, with_grad=keep_grad, inference_mode=inference_mode):
        # model returns logits for each position predicting the *next* token
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False
        ).logits  # (B, seq_len, vocab)
    # the predicted distribution at position j corresponds to next token at j+1.
    # Suppose input_ids = [prefix_tokens, cot_tokens, gold0, gold1, ...,
    #                       gold_{H-1}]
    # We want the model predictions for gold0..gold_{H-1}.
    seq_len = input_ids.size(1)
    target_len = gold_targets.size(1)
    # number of tokens before gold tokens:
    prefix_len = seq_len - target_len
    # logits indices that predict gold tokens are at positions: prefix_len - 1 ... prefix_len + target_len - 2
    # but ensure prefix_len >= 1 (if prefix is empty, typical causal LM has BOS; this script assumes prefix_len >=1)
    start_idx = prefix_len - 1
    end_idx = prefix_len + target_len - 1  # exclusive index in python slicing
    # select logits corresponding to predictions for gold tokens:
    # Memory optimization: Clone the slice we need and drop the full outputs tensor
    # This allows the large (B, seq_len, V) tensor to be freed if not needed for backward of the slice source
    logits_for_gold = outputs[:, start_idx:end_idx, :].clone()   # (B, H, V)
    del outputs
    
    # targets for those positions are exactly gold_targets (shape (B, H))
    per_token_logp = gather_token_logprobs_from_logits(logits_for_gold / temperature, gold_targets)
    # shape (B, H)
    return per_token_logp if keep_grad else per_token_logp.detach()


def make_prefix_and_gold_from_full_input(input_ids, t):
    prefix = input_ids[:, : t]
    gold = input_ids[:, t : t + HORIZON]
    return prefix, gold

@torch.no_grad()
def compute_returns(s_pred_list, s_ema_per_token, horizon=HORIZON):
    """Compute discounted returns for each rollout individually."""

    device = s_pred_list[0].device
    s_ema = s_ema_per_token.to(device)
    steps = torch.arange(horizon, device=device, dtype=s_ema.dtype)
    discounts = torch.pow(torch.tensor(GAMMA, device=device, dtype=s_ema.dtype), steps)

    returns = []

    for s_pred in s_pred_list:
        r_per_token = s_pred - s_ema
        R = torch.sum(r_per_token * discounts)
        returns.append(R)

    if not returns:
        return torch.empty(0, device=device, dtype=s_ema.dtype)

    return torch.stack(returns)

def compute_surrogate_loss(logp_new, logp_old, advantage, mask_cot, lengths_tensor):
    # compute per-thought clipped surrogate loss (Eq.8)
    # importance ratios per token
    log_rhos = logp_new - logp_old.to(logp_new.device)  # (B, C)
    rhos = torch.exp(log_rhos)
    # clip
    clip_rhos = torch.clamp(rhos, 1.0 - EPS_CLIP_LOW, 1.0 + EPS_CLIP_HIGH)
    
    # surrogate per token: min(rho * sg(A), clip(rho) * sg(A))
    # negative sign because we maximize reward -> minimize negative surrogate
    surrogate = torch.min(rhos * advantage, clip_rhos * advantage)
    
    # mask out padding
    surrogate = surrogate * mask_cot
    
    # also divide by |c| as in Eq.8
    row_sums = surrogate.sum(dim=1)
    row_counts = lengths_tensor.squeeze(1).float().clamp(min=1.0)
    row_means = row_sums / row_counts
    
    # average over batch (G)
    return - row_means.mean()

@torch.no_grad()
def rollout(model, prefix, gold_window, num_rollouts=G, temperature=TEMPERATURE, max_new_tokens=THOUGHT_MAX_TOKENS, prefix_cache=None):
    """
    Optimized rollout generation:
      - Computes prefix cache once
      - Samples G rollouts in parallel (temperature 0.7)
      - Collects both sampled CoT tokens and logprobs
        Returns:
            rollouts_ct: List[Tensor]  (G, 1, C_i)
            per_rollout_thought_logprobs_old: List[Tensor]  (G, C_i)
            rollout_caches: List[DynamicCache]
    """
    model.eval()
    eos_token_id = tokenizer.eos_token_id

    base_prefix = prefix[:, :-1]
    if prefix_cache is None:
        prefix_out = model(
            input_ids=base_prefix,
            attention_mask=torch.ones_like(base_prefix),
            past_key_values=None,
            use_cache=True
        )
        cache_copy = DynamicCache.from_legacy_cache(prefix_out.past_key_values)
    else:
        legacy_cache = prefix_cache if isinstance(prefix_cache, tuple) else prefix_cache.to_legacy_cache()
        cache_copy = DynamicCache.from_legacy_cache(legacy_cache)

    # 3️⃣ Initialize rollout state
    past_key_values = cache_copy.batch_repeat_interleave(num_rollouts)
    # We only need the last token of the prefix to start generation if using cache
    current_input_ids = prefix[:, -1:].repeat(num_rollouts, 1)
    
    # Attention mask must cover the full sequence (prefix + generated so far)
    attention_mask = torch.ones((num_rollouts, prefix.size(1)), dtype=torch.long, device=prefix.device)
    
    eos_reached = torch.zeros(num_rollouts, dtype=torch.bool, device=prefix.device)

    rollouts_ct = torch.zeros((num_rollouts, max_new_tokens), dtype=torch.long, device=prefix.device)
    per_token_logp = torch.zeros((num_rollouts, max_new_tokens), dtype=torch.float32, device=prefix.device)
    per_token_end_think_logp = torch.zeros((num_rollouts, max_new_tokens), dtype=torch.float32, device=prefix.device)

    # Prealloc temp tensors to avoid reallocations
    token_buffer = torch.empty(num_rollouts, 1, dtype=torch.long, device=prefix.device)

    # 4️⃣ Step-by-step autoregressive sampling
    step = 0
    for _ in range(max_new_tokens):
        out = model(
            input_ids=current_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )

        logits = out.logits[:, -1, :] / temperature  # [G, vocab]
        past_key_values = out.past_key_values

        # Temperature sampling
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1, out=token_buffer)

        # Record logprobs for chosen tokens
        logprobs = torch.log(probs)
        chosen_logp = logprobs.gather(-1, next_tokens)  # [G, 1]
        per_token_logp[:, step] = chosen_logp.squeeze(-1)
        per_token_end_think_logp[:, step] = logprobs[:, end_think_id]

        # Append new tokens to rollout continuation
        rollouts_ct[:, step] = next_tokens.squeeze(-1)
        
        # Update state for next step
        current_input_ids = next_tokens
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=1)

        # Handle early stopping <eos>
        eos_reached |= (next_tokens.squeeze(-1) == eos_token_id) | (next_tokens.squeeze(-1) == end_think_id)
        step += 1
        if eos_reached.all():
            break
    

    # 5️⃣ Post-process: split per-rollout and prepare for batched s_pred
    rollouts_ct_list = []
    per_token_logp_list = []
    cutoffs = []
    prefix_len = prefix.size(1)

    for i in range(num_rollouts):
        tokens = rollouts_ct[i, :step]
        logps = per_token_logp[i, :step]

        eos_positions = ((tokens == eos_token_id) | (tokens == end_think_id)).nonzero(as_tuple=False)
        is_terminated = eos_positions.numel() > 0
        
        if is_terminated:
            cutoff = eos_positions[0].item() # exclude eos or /think
            tokens_ret = tokens[:cutoff].clone()
            logps_ret = logps[:cutoff].clone()
            # Get logprob of the /think token at termination
            end_think_logp = per_token_end_think_logp[i, cutoff]
            
            # Append </think>
            tokens_ret = torch.cat([tokens_ret, end_think.squeeze(0)], dim=0)
            logps_ret = torch.cat([logps_ret, end_think_logp.unsqueeze(0)], dim=0)
            
            cutoffs.append(cutoff)
        else:
            cutoff = step
            tokens_ret = tokens.clone()
            logps_ret = logps.clone()
            cutoffs.append(cutoff)

        rollouts_ct_list.append(tokens_ret.unsqueeze(0))
        per_token_logp_list.append(logps_ret)

    # 6️⃣ Batched s_pred computation
    # We want to compute logprobs of gold_window given prefix + thought + </think>
    # We reuse the past_key_values from generation, masking out the "future" garbage for shorter rollouts.
    
    # Determine cache length from the model output
    # past_key_values is a DynamicCache or tuple.
    if isinstance(past_key_values, DynamicCache):
        cache_len = past_key_values.get_seq_length()
    else:
        cache_len = past_key_values[0][0].size(2)

    H = gold_window.size(1)
    # Input: [</think>, gold_window]
    input_ids = torch.cat([end_think, gold_window], dim=1).expand(num_rollouts, -1) # (G, 1+H)
    
    # Position IDs: start after the valid thought
    cutoffs_tensor = torch.tensor(cutoffs, device=prefix.device, dtype=torch.long)
    starts = prefix_len + cutoffs_tensor # (G,)
    offsets = torch.arange(1 + H, device=prefix.device, dtype=torch.long).unsqueeze(0) # (1, 1+H)
    position_ids = starts.unsqueeze(1) + offsets # (G, 1+H)
    
    # Attention Mask
    # We need to mask out the garbage in the cache (indices > prefix_len + cutoff)
    # cache_mask: 1 if valid, 0 if garbage
    cache_range = torch.arange(cache_len, device=prefix.device).unsqueeze(0) # (1, cache_len)
    valid_len = (prefix_len + cutoffs_tensor).unsqueeze(1) # (G, 1)
    cache_mask = cache_range < valid_len # (G, cache_len)
    
    # New tokens are always valid
    new_mask = torch.ones((num_rollouts, 1 + H), dtype=torch.bool, device=prefix.device)
    attention_mask = torch.cat([cache_mask, new_mask], dim=1).long() # (G, cache_len + 1 + H)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True
    )
    
    logits = outputs.logits # (G, 1+H, V)
    # We want logprobs for gold_window.
    # The logits at index 0 (corresponding to </think>) predict gold_0.
    # The logits at index H-1 (corresponding to gold_{H-1}) predict gold_H (which we don't have/need).
    # Wait, we want P(gold | context).
    # Input: [</think>, gold_0, ... gold_{H-1}]
    # Logits at [0] -> predict gold_0
    # Logits at [1] -> predict gold_1
    # ...
    # Logits at [H-1] -> predict gold_{H-1}
    # So we need logits[:, :-1, :]
    
    logits_gold = logits[:, :-1, :] # (G, H, V)
    targets = gold_window.expand(num_rollouts, -1) # (G, H)
    
    s_pred_batch = gather_token_logprobs_from_logits(logits_gold, targets) # (G, H)
    s_pred_list = list(s_pred_batch.unbind(0))

    return rollouts_ct_list, per_token_logp_list, s_pred_list


def slice_cache_to_length(full_cache, length: int) -> DynamicCache:
    """Create a DynamicCache truncated to the first ``length`` tokens."""
    if length <= 0:
        return DynamicCache()

    if isinstance(full_cache, DynamicCache):
        base_layers = full_cache.to_legacy_cache()
    else:
        base_layers = full_cache

    truncated_layers = []
    for keys, values in base_layers:
        if keys is None or keys.numel() == 0:
            truncated_layers.append((keys, values))
            continue
        limit = min(keys.size(-2), length)
        # Clone tensors to avoid CUDA graph issues
        truncated_layers.append((
            keys[:, :, :limit, :].clone().contiguous(), 
            values[:, :, :limit, :].clone().contiguous()
        ))

    return DynamicCache.from_legacy_cache(tuple(truncated_layers))


def select_cache_for_batch(full_cache, batch_idx: int, length: int | None = None) -> DynamicCache:
    """Extract (and optionally truncate) a single batch entry from a cache."""
    if isinstance(full_cache, DynamicCache):
        base_layers = full_cache.to_legacy_cache()
    else:
        base_layers = full_cache

    selected_layers = []
    for keys, values in base_layers:
        if keys is None or keys.numel() == 0:
            selected_layers.append((keys, values))
            continue
        # Clone tensors to avoid CUDA graph issues
        keys_sel = keys[batch_idx : batch_idx + 1].clone()
        values_sel = values[batch_idx : batch_idx + 1].clone()
        if length is not None:
            limit = min(keys_sel.size(-2), length)
            keys_sel = keys_sel[:, :, :limit, :].contiguous()
            values_sel = values_sel[:, :, :limit, :].contiguous()
        selected_layers.append((keys_sel, values_sel))

    return DynamicCache.from_legacy_cache(tuple(selected_layers))


def collect_prefix_caches(model, input_ids, positions):
    """Build KV caches for specified prefix lengths via a single forward pass."""
    if not positions:
        return {}

    unique_positions = sorted({pos for pos in positions if pos > 0})
    if not unique_positions:
        return {}

    max_required = unique_positions[-1]
    with inference_context(model, with_grad=False):
        attn_mask = torch.ones((input_ids.size(0), max_required), dtype=torch.long, device=input_ids.device)
        outputs = model(
            input_ids=input_ids[:, :max_required],
            attention_mask=attn_mask,
            past_key_values=None,
            use_cache=True,
        )

    cache_full = outputs.past_key_values
    caches: Dict[int, DynamicCache] = {}
    for length in unique_positions:
        caches[length] = slice_cache_to_length(cache_full, length)

    return caches


def initialize_think_embeddings(model, tokenizer, embeddings_path="think_embeddings.pt"):
    if not os.path.exists(embeddings_path):
        print(f"Warning: Embeddings file '{embeddings_path}' not found. Skipping initialization.")
        return

    print(f"Loading think embeddings from {embeddings_path}...")
    embeddings_data = torch.load(embeddings_path, map_location="cpu")
    
    input_embeddings = model.get_input_embeddings()
    weight = input_embeddings.weight
    
    for token_str, embedding_tensor in embeddings_data.items():
        # Try to find the token ID
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1:
            token_id = ids[0]
        else:
            # Fallback: check if it's a known special token
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id == tokenizer.unk_token_id and token_str != tokenizer.unk_token:
                 print(f"Warning: Token '{token_str}' not found in tokenizer. Skipping.")
                 continue

        if token_id >= weight.shape[0]:
            # This might happen if tokenizer has more tokens than model embeddings
            print(f"Resizing token embeddings to accommodate token ID {token_id}...")
            model.resize_token_embeddings(len(tokenizer))
            input_embeddings = model.get_input_embeddings()
            weight = input_embeddings.weight
            
        with torch.no_grad():
            if embedding_tensor.shape[-1] != weight.shape[-1]:
                print(f"Error: Dimension mismatch for '{token_str}'. File: {embedding_tensor.shape[-1]}, Model: {weight.shape[-1]}")
                continue
                
            weight[token_id].copy_(embedding_tensor.to(weight.device).to(weight.dtype))
            print(f"Initialized embedding for '{token_str}' (ID: {token_id})")


# -----------------------------
# Load model, tokenizer, dataset
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Set pad_token to eos_token to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
start_think = tokenizer("<think>", return_tensors="pt").input_ids.to(DEVICE)
end_think = tokenizer("</think>", return_tensors="pt").input_ids.to(DEVICE)
start_think_id = start_think.item()
end_think_id = end_think.item()

assert start_think.shape == (1, 1)
assert end_think.shape == (1, 1)


def log_sampled_thought(step_idx: int, position: int, rollout_idx: int, cot_tokens: torch.Tensor, prefix: str, target: str) -> None:
    """Append the decoded chain-of-thought sample to the debug log."""
    if RANK != 0: return
    token_ids = cot_tokens.reshape(-1).tolist()
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[step={step_idx} t={position} rollout={rollout_idx} len={len(token_ids)}]\n")
        log_file.write(f"Prefix: {prefix}\n")
        log_file.write(f"Target: {target}\n")
        log_file.write(repr(decoded) + "\n\n")

print("Loading model...")
attn_implementation = "flash_attention_2" if USE_FLASH_ATTN else "eager"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation=attn_implementation, trust_remote_code=True, dtype=DTYPE).to(DEVICE)
model.config.use_flash_attention = USE_FLASH_ATTN
model.gradient_checkpointing_enable()

initialize_think_embeddings(model, tokenizer)

ema_model = copy.deepcopy(model).to(DEVICE)
for p in ema_model.parameters():
    p.requires_grad_(False)

if IS_DDP:
    model = DDP(model, device_ids=[local_rank])
    inference_model = model.module
else:
    inference_model = model

compile_mode = "default" # set to "max-autotune-no-cudagraphs" for performance
if COMPILE_MODEL:
    print("Compiling model...")
    compiled_model = torch.compile(model, mode=compile_mode, dynamic=True)
else:
    compiled_model = model
ema_model = torch.compile(ema_model, mode=compile_mode, dynamic=True)

optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=LR)

print_gpu_memory("Post-model-load")

if os.path.exists(DATA_CACHE_DIR):
    ds = Dataset.load_from_disk(DATA_CACHE_DIR)
else:
    # Load a small subset for debugging. For full experiments, remove slicing.
    ds = load_dataset(DATASET, split=SPLIT, streaming=True).take(N_DATA)
    ds = ds.map(lambda ex: {"input_ids": tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]})
    ds = Dataset.from_list(list(ds))
    ds.save_to_disk(DATA_CACHE_DIR)
    
ds = ds.shuffle(seed=42)

if IS_DDP:
    ds = ds.shard(num_shards=WORLD_SIZE, index=RANK)

# reset log for new training run
if RANK == 0:
    DEBUG_LOG_PATH.write_text("", encoding="utf-8")

# Metric tracking for visualization
metric_history = {
    "reward": [],
    "reward_std": [],
    "cot_length": [],
    "loss": [],
}

# -----------------------------
# Training loop
# -----------------------------
model.train()
global_step = 0
progress = tqdm(ds, desc="GRLP Training", disable=RANK!=0)
for item in progress:
    full_input_ids = torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, L)
    L = full_input_ids.size(1)
    # choose some positions t to apply RLP on in this example: for fast debug pick a few
    # In large runs you'd iterate t across the sequence

    if L <= HORIZON + 3:
        continue
    num_candidate_positions = min(BATCH_SIZE, L - HORIZON - 1)
    candidate_positions = torch.randint(2, L - HORIZON - 1, (num_candidate_positions,)).tolist()

    ema_token_logprobs_full = compute_teacher_forced_logprobs(ema_model, full_input_ids, full_input_ids[:, 1:], keep_grad=False, inference_mode=True).squeeze(0)  # (L-1,)
    prefix_cache_map = collect_prefix_caches(inference_model, full_input_ids, candidate_positions)

    batch_advantages = []
    batch_rewards = []
    batch_cot_lengths = []
    batch_prefixes = []
    batch_targets = []
    batch_rollout_buffers = []
    
    # For each selected position t, compute G rollouts and log probs under old_theta
    for t in candidate_positions:
        prefix, gold_window = make_prefix_and_gold_from_full_input(full_input_ids, t)  # shapes (1, P-1), (1, H)
        # TODO: in the future could handle end positions with shorter horizons

        # prefix_str = tokenizer.decode(prefix[:, -2*HORIZON:].squeeze(0).tolist())
        # target_str = tokenizer.decode(gold_window.squeeze(0).tolist())

        ema_start = t - 1  # logits index predicting token at position t
        ema_end = ema_start + HORIZON
        s_ema_per_token = ema_token_logprobs_full[ema_start:ema_end]

        reasoning_prefix = torch.cat([prefix, start_think], dim=1)  # (1, P+1)
        cache_for_prefix = prefix_cache_map.get(t)
        rollouts_ct, per_rollout_thought_logprobs_old, s_pred_list = rollout(
            inference_model,
            reasoning_prefix,
            gold_window,
            num_rollouts=G,
            prefix_cache=cache_for_prefix,
        )

        # Now for each rollout evaluate the reasoned per-token log-probs under the current model (p_theta)
        returns = compute_returns(s_pred_list, s_ema_per_token)  # shape (G,)

        # Drop rollout caches to avoid holding GPU references beyond this point
        del s_ema_per_token

        # Move rollout artifacts to pinned CPU buffers to free VRAM until PPO phase
        rollout_buffer = pack_rollout_tensors(
            rollouts_ct,
            per_rollout_thought_logprobs_old,
        )
        reasoning_prefix_cpu = to_cpu_pinned(reasoning_prefix)
        gold_window_cpu = to_cpu_pinned(gold_window)

        # Compute group-relative advantages (Eq.7)
        r_mean = returns.mean()
        # advantage scaling factor G/(G-1)
        advantages = (G / (G - 1)) * (returns - r_mean)  # shape (G,)

        batch_advantages.append(to_cpu_pinned(advantages))
        batch_rewards.append(to_cpu_pinned(returns))
        batch_prefixes.append(reasoning_prefix_cpu)
        batch_targets.append(gold_window_cpu)
        batch_rollout_buffers.append(rollout_buffer)
        lengths_view = rollout_buffer.lengths[: rollout_buffer.num_rollouts]
        if lengths_view.numel():
            batch_cot_lengths.append(float(lengths_view.to(torch.float32).mean().item()))
        else:
            batch_cot_lengths.append(0.0)

        # Remove GPU tensors from scope
        del prefix
        del rollouts_ct
        del per_rollout_thought_logprobs_old
        del reasoning_prefix
        del gold_window
        del returns
        del advantages

    prefix_cache_map.clear()
    del prefix_cache_map
    del ema_token_logprobs_full
    release_cuda_cache()

    if not batch_rewards:
        for buffer in batch_rollout_buffers:
            release_rollout_buffer(buffer)
        batch_rollout_buffers.clear()
        print("No valid rollouts in batch, skipping...")
        continue

    batch_rewards = torch.stack(batch_rewards).float()
    avg_reward = float(batch_rewards.mean())
    metric_history["reward"].append(avg_reward)
    metric_history["reward_std"].append(float(batch_rewards.std()))
    avg_cot_length_value = sum(batch_cot_lengths) / len(batch_cot_lengths)
    metric_history["cot_length"].append(avg_cot_length_value)

    global_step += 1

    avg_loss = 0.0
    # For each selected position t, compute G rollouts and returns; then accumulate surrogate losses
    num_contexts = len(batch_prefixes)
    indices = list(range(num_contexts))
    
    for epoch in range(PPO_EPOCHS):
        batch_surrogate_losses = []
        optimizer.zero_grad(set_to_none=True)

        for i in range(0, num_contexts, MINI_BATCH_SIZE):
            chunk_indices = indices[i : i + MINI_BATCH_SIZE]
            
            chunk_prefixes = [batch_prefixes[j] for j in chunk_indices]
            chunk_targets = [batch_targets[j] for j in chunk_indices]
            chunk_buffers = [batch_rollout_buffers[j] for j in chunk_indices]
            chunk_advantages = [batch_advantages[j] for j in chunk_indices]
            
            chunk_data = []
            max_prefix_len = 0
            max_cot_len = 0
            
            for prefix_cpu, gold_window_cpu, buffer, adv_cpu in zip(chunk_prefixes, chunk_targets, chunk_buffers, chunk_advantages):
                tokens_cpu, logps_cpu, lengths_cpu = buffer.views()
                if tokens_cpu.size(0) == 0:
                    continue
                
                P = prefix_cpu.size(1)
                max_prefix_len = max(max_prefix_len, P)
                L = tokens_cpu.size(1)
                max_cot_len = max(max_cot_len, L)
                
                chunk_data.append({
                    'prefix': prefix_cpu,
                    'tokens': tokens_cpu,
                    'logps': logps_cpu,
                    'lengths': lengths_cpu,
                    'adv': adv_cpu
                })
            
            if not chunk_data:
                continue
                
            batched_input_ids = []
            batched_targets = []
            batched_attention_mask = []
            batched_old_logps = []
            batched_advantages = []
            batched_mask_cot = []
            batched_lengths = []
            
            for item in chunk_data:
                prefix = item['prefix']
                tokens = item['tokens']
                logps = item['logps']
                lengths = item['lengths']
                adv = item['adv']
                
                G_curr = tokens.size(0)
                P = prefix.size(1)
                L = tokens.size(1)
                
                # Left-pad prefix
                pad_p = max_prefix_len - P
                prefix_expanded = prefix.repeat(G_curr, 1)
                if pad_p > 0:
                    prefix_padded = F.pad(prefix_expanded, (pad_p, 0), value=tokenizer.pad_token_id)
                    prefix_mask = torch.cat([
                        torch.zeros((G_curr, pad_p), dtype=torch.bool),
                        torch.ones((G_curr, P), dtype=torch.bool)
                    ], dim=1)
                else:
                    prefix_padded = prefix_expanded
                    prefix_mask = torch.ones((G_curr, P), dtype=torch.bool)
                
                # Right-pad tokens (CoT)
                pad_l = max_cot_len - L
                if pad_l > 0:
                    tokens_padded = F.pad(tokens, (0, pad_l), value=tokenizer.pad_token_id)
                    logps_padded = F.pad(logps, (0, pad_l), value=0.0)
                else:
                    tokens_padded = tokens
                    logps_padded = logps
                
                cot_range = torch.arange(max_cot_len).unsqueeze(0)
                cot_mask = cot_range < lengths.unsqueeze(1)
                
                input_ids = torch.cat([prefix_padded, tokens_padded], dim=1)
                att_mask = torch.cat([prefix_mask, cot_mask], dim=1)
                
                batched_input_ids.append(input_ids)
                batched_targets.append(tokens_padded)
                batched_attention_mask.append(att_mask)
                batched_old_logps.append(logps_padded)
                batched_advantages.append(adv)
                batched_mask_cot.append(cot_mask)
                batched_lengths.append(lengths)

            input_ids_tensor = torch.cat(batched_input_ids, dim=0).to(DEVICE, non_blocking=True)
            targets_tensor = torch.cat(batched_targets, dim=0).to(DEVICE, non_blocking=True)
            attention_mask_tensor = torch.cat(batched_attention_mask, dim=0).to(DEVICE, non_blocking=True).long()
            old_logps_tensor = torch.cat(batched_old_logps, dim=0).to(DEVICE, non_blocking=True)
            advantages_tensor = torch.cat(batched_advantages, dim=0).to(DEVICE, non_blocking=True).unsqueeze(1)
            mask_cot_tensor = torch.cat(batched_mask_cot, dim=0).to(DEVICE, non_blocking=True)
            lengths_tensor = torch.cat(batched_lengths, dim=0).to(DEVICE, non_blocking=True).unsqueeze(1)
            
            per_token_logp_new = compute_teacher_forced_logprobs(
                compiled_model, 
                input_ids_tensor, 
                targets_tensor, 
                attention_mask=attention_mask_tensor, 
                temperature=TEMPERATURE
            )
            
            loss = compute_surrogate_loss(
                logp_new=per_token_logp_new, 
                logp_old=old_logps_tensor, 
                advantage=advantages_tensor, 
                mask_cot=mask_cot_tensor, 
                lengths_tensor=lengths_tensor
            )
            
            loss.backward()
            batch_surrogate_losses.append(loss.detach().cpu().item())
            
            del input_ids_tensor, targets_tensor, attention_mask_tensor, old_logps_tensor, advantages_tensor, mask_cot_tensor, lengths_tensor

        torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss += sum(batch_surrogate_losses) / len(batch_surrogate_losses) / PPO_EPOCHS

    metric_history["loss"].append(avg_loss)
    postfix = {
        "loss": avg_loss,
        "reward": avg_reward,
        "cot_len": avg_cot_length_value,
    }
    progress.set_postfix(postfix)

    for buffer in batch_rollout_buffers:
        release_rollout_buffer(buffer)
    batch_rollout_buffers.clear()

    # EMA update
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(TAU).add_(p.data, alpha=1.0 - TAU)

    if global_step and global_step % PLOT_SAVE_INTERVAL == 0:
        save_metric_plot(metric_history)
    
    if global_step and global_step % MODEL_SAVE_INTERVAL == 0:
        save_dir = f"{MODEL_NAME.split('/')[-1]}-grlp-step{global_step}"
        schedule_model_save(model, tokenizer, save_dir)
    
    release_cuda_cache()

# optional: save checkpoint each epoch
save_dir = f"{MODEL_NAME.split('/')[-1]}-grlp"
schedule_model_save(model, tokenizer, save_dir)

save_metric_plot(metric_history)

wait_for_background_tasks()

print_gpu_memory("Post-training")

print("done")