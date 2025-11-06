"""
Generalized RLP prototype (discounted-return advantage estimation)
- Model: Qwen/Qwen3-0.6B-Base (or any causal LM)
- Dataset: HuggingFaceFW/fineweb (use a small subset for testing)
"""

import copy
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset, Dataset
from tqdm import tqdm
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from contextlib import contextmanager


# -----------------------------
# Config
# -----------------------------
# Parse command line arguments
parser = argparse.ArgumentParser(description="Generalized RLP prototype training")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Base", help="Model name or path to saved model")
args = parser.parse_args()

MODEL_NAME = args.model_name
DATASET = "HuggingFaceFW/fineweb"
SPLIT = "train"
N_DATA = 10_000
DATA_CACHE_DIR = Path("data/fineweb-10k-tokenized")

MAX_SEQ_LEN = 2048
HORIZON = 8                             # reward horizon T (small for debug; paper uses long)
THOUGHT_MAX_TOKENS = 2048
G = 16                                  # number of rollouts per context
GAMMA = 0.7                             # discount factor
BATCH_SIZE = 16                         # token-level RLP is expensive; tune for your memory
LR = 1e-6
TAU = 0.999                             # EMA decay
EPS_CLIP_LOW, EPS_CLIP_HIGH = 0.1, 0.1  # PPO clipping
PPO_EPOCHS = 3                          # number of policy optimization iterations per batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
COMPILE_MODEL = True
USE_FLASH_ATTN = False

DEBUG_LOG_PATH = Path("debug.txt")
PLOT_SAVE_INTERVAL = 5
MODEL_SAVE_INTERVAL = 100
METRIC_FIG_PATH = Path("simple_grlp_training_metrics.png")

torch.set_float32_matmul_precision('high')

def print_gpu_memory(prefix: str) -> None:
    """Log CUDA memory stats if a GPU is available."""
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

def save_metric_plot(metrics: Dict[str, List[float]]) -> None:
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
    logprobs = F.log_softmax(logits, dim=-1)
    # gather the logprob of each target token
    # target_ids must be long dtype
    return logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def compute_teacher_forced_logprobs(model, input_ids, gold_targets, *, keep_grad=True, inference_mode=False):
    """
    Compute per-token log-probs under teacher forcing for the sequence:
      model(input_ids) where input_ids = prefix + optional cot + gold_seq
    We want the log-probabilities for gold_seq positions (the final len(gold_targets) tokens).
    Returns: tensor shape (len(gold_targets),) of log-probs (float)
    """
    with inference_context(model, with_grad=keep_grad, inference_mode=inference_mode):
        # model returns logits for each position predicting the *next* token
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=None,
            use_cache=False
        ).logits  # (1, seq_len, vocab)
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
    logits_for_gold = outputs[:, start_idx:end_idx, :]   # (1, H, V)
    # targets for those positions are exactly gold_targets (shape (1, H))
    per_token_logp = gather_token_logprobs_from_logits(logits_for_gold, gold_targets)
    # shape (1, H) -> squeeze
    result = per_token_logp.squeeze(0)  # (H,)
    return result if keep_grad else result.detach()


def make_prefix_and_gold_from_full_input(input_ids, t):
    prefix = input_ids[:, : t]
    gold = input_ids[:, t : t + HORIZON]
    return prefix, gold

@torch.no_grad()
def compute_returns(rollouts_ct, rollout_caches, prefix, gold_window, model, s_ema_per_token):
    """Compute discounted returns for each rollout individually."""

    device = prefix.device
    s_ema = s_ema_per_token.to(device)
    gold_len = gold_window.size(1)
    steps = torch.arange(gold_len, device=device, dtype=s_ema.dtype)
    discounts = torch.pow(torch.tensor(GAMMA, device=device, dtype=s_ema.dtype), steps)

    appended_tokens = torch.cat([end_thought_id.to(device), gold_window], dim=1)
    gold_window = gold_window.to(device)

    returns = []
    prefix_len = prefix.size(1)

    for ct, cache in zip(rollouts_ct, rollout_caches):
        ct_len = ct.size(1)
        eos_positions = (ct == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            ct_len = eos_positions[0, 1].item()

        context_len = prefix_len + ct_len
        cache_trimmed = slice_cache_to_length(cache, context_len)

        attn_mask = torch.ones(1, context_len + appended_tokens.size(1), dtype=torch.long, device=device)

        outputs = model(
            input_ids=appended_tokens,
            attention_mask=attn_mask,
            past_key_values=cache_trimmed,
            use_cache=True,
        )

        logits_gold = outputs.logits[:, 1:, :]
        s_pred = gather_token_logprobs_from_logits(logits_gold, gold_window)
        r_per_token = s_pred.squeeze(0) - s_ema
        R = torch.sum(r_per_token * discounts)
        returns.append(R)

    if not returns:
        return torch.empty(0, device=device, dtype=s_ema.dtype)

    return torch.stack(returns)

def compute_surrogate_loss(per_rollout_thought_logprobs_new, per_rollout_thought_logprobs_old, advantages):
    surrogate_loss = 0
    # For each rollout, compute per-thought clipped surrogate loss (Eq.8)
    for i in range(G):
        # per-token log probs under new/current (we have per_token_logp_new)
        logp_new = per_rollout_thought_logprobs_new[i]  # shape (C,)
        # per-token log probs under old/behavior (we computed earlier in rollouts_logprob_old_tokens)
        logp_old = per_rollout_thought_logprobs_old[i]      # shape (C,)
        # importance ratios per token
        log_rhos = logp_new - logp_old.to(logp_new.device)  # (C,)
        rhos = torch.exp(log_rhos)                           # (C,)
        # clip
        clip_rhos = torch.clamp(rhos, 1.0 - EPS_CLIP_LOW, 1.0 + EPS_CLIP_HIGH)
        A = advantages[i]  # scalar
        # surrogate per token: min(rho * sg(A), clip(rho) * sg(A))
        # negative sign because we maximize reward -> minimize negative surrogate
        # Also divide by |c| as in Eq.8
        surrogate_per_token = torch.min(rhos * A.detach(), clip_rhos * A.detach())
        loss_i = - surrogate_per_token.mean()  # scalar
        surrogate_loss += loss_i

    return surrogate_loss / G


@torch.no_grad()
def rollout(model, prefix, num_rollouts=G, temperature=0.7, max_new_tokens=THOUGHT_MAX_TOKENS, prefix_cache=None):
    """
    Optimized rollout generation:
      - Computes prefix cache once
      - Samples G rollouts in parallel (temperature 0.7)
      - Collects both sampled CoT tokens and untempered logprobs
        Returns:
            rollouts_ct: List[Tensor]  (G, 1, C_i)
            per_rollout_thought_logprobs_old: List[Tensor]  (G, C_i)
            rollout_caches: List[DynamicCache]
    """
    model.eval()
    eos_token_id = tokenizer.eos_token_id

    if prefix_cache is not None:
        if isinstance(prefix_cache, tuple):
            past_key_values = DynamicCache.from_legacy_cache(prefix_cache)
        else:
            past_key_values = DynamicCache.from_legacy_cache(prefix_cache.to_legacy_cache())
        past_key_values.batch_repeat_interleave(num_rollouts)

        # Split prefix into actual document prefix + deterministic think token
        base_prefix = prefix[:, :-1]
        think_token = prefix[:, -1:].repeat(num_rollouts, 1)

        generated = base_prefix.repeat(num_rollouts, 1)
        attention_mask = torch.ones_like(generated)
        eos_reached = torch.zeros(num_rollouts, dtype=torch.bool, device=generated.device)

        rollouts_ct = torch.full((num_rollouts, 0), fill_value=0, dtype=torch.long, device=generated.device)
        per_token_logp = torch.empty(num_rollouts, 0, dtype=torch.bfloat16, device=generated.device)
        token_buffer = torch.empty(num_rollouts, 1, dtype=torch.long, device=generated.device)

        attention_mask = torch.cat((attention_mask, torch.ones_like(think_token)), dim=1)
        out = model(
            input_ids=think_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        past_key_values = out.past_key_values
        generated = torch.cat((generated, think_token), dim=1)
    else:
        # 1️⃣ Compute prefix KV-cache once
        prefix_out = model(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            past_key_values=None,
            use_cache=True
        )
        prefix_pkv = prefix_out.past_key_values
        if isinstance(prefix_pkv, tuple):
            prefix_pkv = DynamicCache.from_legacy_cache(prefix_pkv)
        prefix_pkv.batch_repeat_interleave(num_rollouts)
        past_key_values = prefix_pkv

        # 3️⃣ Initialize rollout state
        generated = prefix.repeat(num_rollouts, 1)
        attention_mask = torch.ones_like(generated)
        eos_reached = torch.zeros(num_rollouts, dtype=torch.bool, device=generated.device)

        rollouts_ct = torch.full((num_rollouts, 0), fill_value=0, dtype=torch.long, device=generated.device)
        per_token_logp = torch.empty(num_rollouts, 0, dtype=torch.bfloat16, device=generated.device)

        # Prealloc temp tensors to avoid reallocations
        token_buffer = torch.empty(num_rollouts, 1, dtype=torch.long, device=generated.device)

    # 4️⃣ Step-by-step autoregressive sampling
    for _ in range(max_new_tokens):
        out = model(
            input_ids=generated[:, -1:],  # last token only
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )

        logits = out.logits[:, -1, :]  # [G, vocab]
        past_key_values = out.past_key_values

        # Store untempered log probs for PPO baseline (not divided by temperature)
        logp_untempered = torch.log_softmax(logits, dim=-1)

        # Temperature sampling
        probs = torch.softmax(logits / temperature, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1, out=token_buffer)

        # Record logprobs for chosen tokens
        chosen_logp = logp_untempered.gather(-1, next_tokens)
        per_token_logp = torch.cat((per_token_logp, chosen_logp), dim=1)

        # Append new tokens to rollout continuation
        rollouts_ct = torch.cat((rollouts_ct, next_tokens), dim=1)
        generated = torch.cat((generated, next_tokens), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=1)

        # Handle early stopping <eos>
        eos_reached |= (next_tokens.squeeze(-1) == eos_token_id) 
        # eos_reached |= (next_tokens.squeeze(-1) == end_thought)
        if eos_reached.all():
            break

    # 5️⃣ Post-process: split per-rollout
    rollouts_ct_list = []
    per_token_logp_list = []
    for i in range(num_rollouts):
        tokens = rollouts_ct[i, :]
        logps = per_token_logp[i, :]

        eos_positions = (tokens == eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            cutoff = eos_positions[0].item() + 1  # include eos token
            tokens = tokens[:cutoff].clone()
            logps = logps[:cutoff].clone()
        else:
            tokens = tokens.clone()
            logps = logps.clone()

        rollouts_ct_list.append(tokens.unsqueeze(0))
        per_token_logp_list.append(logps)

    if not isinstance(past_key_values, DynamicCache):
        final_cache = DynamicCache.from_legacy_cache(past_key_values)
    else:
        final_cache = past_key_values

    prefix_len = prefix.size(1)
    rollout_cache_list = []
    for i, ct_tokens in enumerate(rollouts_ct_list):
        total_len = prefix_len + ct_tokens.size(1)
        cache_i = select_cache_for_batch(final_cache, i, total_len)
        rollout_cache_list.append(cache_i)

    return rollouts_ct_list, per_token_logp_list, rollout_cache_list


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


# -----------------------------
# Load model, tokenizer, dataset
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Set pad_token to eos_token to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
start_thought_id = tokenizer("<think>", return_tensors="pt").input_ids.to(DEVICE)
end_thought_id = tokenizer("</think>", return_tensors="pt").input_ids.to(DEVICE)

assert start_thought_id.shape == (1, 1)
assert end_thought_id.shape == (1, 1)


def log_sampled_thought(step_idx: int, position: int, rollout_idx: int, cot_tokens: torch.Tensor, prefix: str, target: str) -> None:
    """Append the decoded chain-of-thought sample to the debug log."""
    decoded = tokenizer.decode(cot_tokens.tolist(), skip_special_tokens=False)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[step={step_idx} t={position} rollout={rollout_idx} len={cot_tokens.size(0)}]\n")
        log_file.write(f"Prefix: {prefix}\n")
        log_file.write(f"Target: {target}\n")
        log_file.write(repr(decoded) + "\n\n")

print("Loading model...")
attn_implementation = "flash_attention_2" if USE_FLASH_ATTN else "eager"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation=attn_implementation, trust_remote_code=True).to(DEVICE)
model = model.to(dtype=DTYPE)
model.config.use_flash_attention = USE_FLASH_ATTN
model.gradient_checkpointing_enable()

ema_model = copy.deepcopy(model).to(DEVICE)
for p in ema_model.parameters():
    p.requires_grad_(False)

if COMPILE_MODEL:
    print("Compiling model...")
    model = torch.compile(model, dynamic=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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

# reset log for new training run
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
progress = tqdm(ds, desc="GRLP Training")
for item in progress:
    full_input_ids = torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, L)
    L = full_input_ids.size(1)
    # choose some positions t to apply RLP on in this example: for fast debug pick a few
    # In large runs you'd iterate t across the sequence

    if L <= HORIZON + 3:
        continue
    num_candidate_positions = min(BATCH_SIZE, L - HORIZON - 1)
    candidate_positions = torch.randint(2, L - HORIZON - 1, (num_candidate_positions,)).tolist()

    ema_token_logprobs_full = compute_teacher_forced_logprobs(ema_model, full_input_ids, full_input_ids[:, 1:], keep_grad=False, inference_mode=True)  # (L-1,)
    prefix_cache_map = collect_prefix_caches(model, full_input_ids, candidate_positions)

    batch_advantages = []
    batch_rewards = []
    batch_cot_lengths = []
    batch_prefixes = []
    batch_targets = []
    batch_rollout_buffers = []
    batch_s_ema_per_token = []
    
    # For each selected position t, compute G rollouts and log probs under old_theta
    for t in candidate_positions:
        prefix, gold_window = make_prefix_and_gold_from_full_input(full_input_ids, t)  # shapes (1, P-1), (1, H)
        P = prefix.size(1)
        # TODO: in the future could handle end positions with shorter horizons
        H_current = gold_window.size(1)

        # prefix_str = tokenizer.decode(prefix[:, -2*H_current:].squeeze(0).tolist())
        # target_str = tokenizer.decode(gold_window.squeeze(0).tolist())

        ema_start = t - 1  # logits index predicting token at position t
        ema_end = ema_start + H_current
        s_ema_per_token = ema_token_logprobs_full[ema_start:ema_end]

        reasoning_prefix = torch.cat([prefix, start_thought_id], dim=1)  # (1, P)
        cache_for_prefix = prefix_cache_map.get(P)
        rollouts_ct, per_rollout_thought_logprobs_old, rollout_caches = rollout(
            model,
            reasoning_prefix,
            prefix_cache=cache_for_prefix,
        )

        # Now for each rollout evaluate the reasoned per-token log-probs under the current model (p_theta)
        returns = compute_returns(rollouts_ct, rollout_caches, reasoning_prefix, gold_window, model, s_ema_per_token)

        # Drop rollout caches to avoid holding GPU references beyond this point
        del rollout_caches

        # Move rollout artifacts to pinned CPU buffers to free VRAM until PPO phase
        rollout_buffer = pack_rollout_tensors(
            rollouts_ct,
            per_rollout_thought_logprobs_old,
        )
        s_ema_cpu = to_cpu_pinned(s_ema_per_token)
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
        batch_s_ema_per_token.append(s_ema_cpu)
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
        del s_ema_per_token
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
    for epoch in range(PPO_EPOCHS):
        batch_surrogate_losses = []
        optimizer.zero_grad(set_to_none=True)

        for prefix_cpu, gold_window_cpu, rollout_buffer, _, rollout_advantages_cpu in zip(
            batch_prefixes, batch_targets, batch_rollout_buffers, batch_s_ema_per_token, batch_advantages
        ):

            prefix = prefix_cpu.to(DEVICE, non_blocking=True)
            gold_window = gold_window_cpu.to(DEVICE, non_blocking=True)
            tokens_cpu_view, logp_cpu_view, lengths_cpu_view = rollout_buffer.views()
            tokens_buffer = tokens_cpu_view.to(DEVICE, non_blocking=True)
            logp_buffer = logp_cpu_view.to(DEVICE, non_blocking=True)
            lengths_list = lengths_cpu_view.tolist()
            rollout_advantages = rollout_advantages_cpu.to(DEVICE, non_blocking=True)
            del prefix_cpu
            del gold_window_cpu
            del rollout_advantages_cpu

            # - compute reasoned per-token logprobs s_pred^i for horizon H under current model conditioned on prefix + c_t + gold_window
            per_rollout_thought_logprobs_new = []  # per-rollout per-token logprobs under current model for thought tokens
            per_rollout_thought_logprobs_old = []
            for i, length in enumerate(lengths_list):
                if length == 0:
                    per_rollout_thought_logprobs_new.append(logp_buffer.new_empty((0,), device=logp_buffer.device))
                    per_rollout_thought_logprobs_old.append(logp_buffer[i, :0])
                    continue
                ct_tokens = tokens_buffer[i, :length].unsqueeze(0)
                # Build input: prefix + cot_tokens + gold_window
                inp_thought = torch.cat([prefix, ct_tokens], dim=1)  # (1, P+C)
                # compute per-token log-probs under current model for cot_tokens
                per_token_logp_new = compute_teacher_forced_logprobs(model, inp_thought, ct_tokens)  # (C,)
                per_rollout_thought_logprobs_new.append(per_token_logp_new)  # (C,)
                per_rollout_thought_logprobs_old.append(logp_buffer[i, :length])

            surrogate_loss = compute_surrogate_loss(
                per_rollout_thought_logprobs_new,
                per_rollout_thought_logprobs_old,
                rollout_advantages
            )
            surrogate_loss.backward()
            batch_surrogate_losses.append(float(surrogate_loss.detach().cpu()))

            del prefix
            del gold_window
            del tokens_buffer
            del logp_buffer
            del tokens_cpu_view
            del logp_cpu_view
            del lengths_cpu_view
            del rollout_advantages
            del per_rollout_thought_logprobs_new
            del per_rollout_thought_logprobs_old

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    
    release_cuda_cache()

# optional: save checkpoint each epoch
save_dir = f"{MODEL_NAME.split('/')[-1]}-grlp"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

save_metric_plot(metric_history)

print_gpu_memory("Post-training")

print("done")
