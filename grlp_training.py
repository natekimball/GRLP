"""
Generalized RLP prototype (discounted-return advantage estimation)
- Model: Qwen/Qwen3-0.6B-Base (or any causal LM)
- Dataset: allenai/dolma (use a small subset for testing)
"""

import copy
import os
from pathlib import Path
from typing import Dict, List

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

MAX_SEQ_LEN = 1024
HORIZON = 8                             # reward horizon T (small for debug; paper uses long)
THOUGHT_MAX_TOKENS = 512
G = 4                                   # number of rollouts per context
GAMMA = 0.7                             # discount factor
BATCH_SIZE = 4                          # token-level RLP is expensive; tune for your memory
LR = 1e-6
TAU = 0.999                             # EMA decay
EPS_CLIP_LOW, EPS_CLIP_HIGH = 0.1, 0.1  # PPO clipping
PPO_EPOCHS = 3                          # number of policy optimization iterations per batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
COMPILE_MODEL = False
USE_FLASH_ATTN = False

DEBUG_LOG_PATH = Path("debug.txt")
PLOT_SAVE_INTERVAL = 5
MODEL_SAVE_INTERVAL = 100
METRIC_FIG_PATH = Path("simple_grlp_training_metrics.png")


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
def inference_context(model, with_grad: bool = False):
    was_training = model.training
    if with_grad:
        model.train()
    else:
        model.eval()
    try:
        if with_grad:
            with torch.enable_grad():
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


def compute_teacher_forced_logprobs(model, input_ids, gold_targets, *, keep_grad=True):
    """
    Compute per-token log-probs under teacher forcing for the sequence:
      model(input_ids) where input_ids = prefix + optional cot + gold_seq
    We want the log-probabilities for gold_seq positions (the final len(gold_targets) tokens).
    Returns: tensor shape (len(gold_targets),) of log-probs (float)
    """
    with inference_context(model, with_grad=keep_grad):
        # model returns logits for each position predicting the *next* token
        outputs = model(input_ids=input_ids).logits  # (1, seq_len, vocab)
    # the predicted distribution at position j corresponds to next token at j+1.
    # Suppose input_ids = [prefix_tokens, cot_tokens, gold0, gold1, ...,
    #                       gold_{H-1}]
    # We want the model predictions for gold0..gold_{H-1}.
    seq_len = input_ids.size(1)
    H = gold_targets.size(1)
    # number of tokens before gold tokens:
    prefix_len = seq_len - H
    # logits indices that predict gold tokens are at positions: prefix_len - 1 ... prefix_len + H - 2
    # but ensure prefix_len >= 1 (if prefix is empty, typical causal LM has BOS; this script assumes prefix_len >=1)
    start_idx = prefix_len - 1
    end_idx = prefix_len + H - 1  # exclusive index in python slicing
    # select logits corresponding to predictions for gold tokens:
    logits_for_gold = outputs[:, start_idx:end_idx, :]   # (1, H, V)
    # targets for those positions are exactly gold_targets (shape (1, H))
    per_token_logp = gather_token_logprobs_from_logits(logits_for_gold, gold_targets)
    # shape (1, H) -> squeeze
    result = per_token_logp.squeeze(0)  # (H,)
    return result if keep_grad else result.detach()


def make_prefix_and_gold_from_full_input(input_ids, t):
    """
    Given full document token sequence (2D tensor), build:
      prefix = input_ids[:t]
      gold_window = input_ids[t : t + HORIZON]
      combined = input_ids[:t+HORIZON]  (for EMA input)
    Returns three 2D tensors with batch dim: (1, P), (1, H), (1, P+H)
    """
    prefix = input_ids[:, : t]
    gold = input_ids[:, t : t + HORIZON]
    combined = input_ids[:, : t + HORIZON]
    return prefix, gold, combined

def compute_returns(rollouts_ct, prefix, gold_window, model, s_ema_per_token):
    returns = []  # discounted returns R(c_t) for each rollout

    for i in range(len(rollouts_ct)):
        # Build input: prefix + cot_tokens + gold_window
        inp_reasoned = torch.cat([prefix, rollouts_ct[i], gold_window], dim=1)  # (1, P+C+H)
        # compute per-token log-probs under current model for gold_window
        s_pred_per_token = compute_teacher_forced_logprobs(model, inp_reasoned, gold_window, keep_grad=False)  # (H,)
        # compute r_i per token = s_pred^i - s_ema^i
        # ensure ema_per_token_logp has same length H_current
        r_per_token = s_pred_per_token - s_ema_per_token.to(s_pred_per_token.device)
        # discounted sum on the same device to avoid host syncs
        steps = torch.arange(r_per_token.size(0), device=r_per_token.device, dtype=r_per_token.dtype)
        discounts = GAMMA ** steps
        R = torch.sum(r_per_token * discounts)
        returns.append(R)  # R should be a constant (no grad)

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
    """
    model.eval()
    eos_token_id = tokenizer.eos_token_id
    end_thought = end_thought_id.item()

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

        # Handle early stopping (<eos> or </think>)
        eos_reached |= (next_tokens.squeeze(-1) == eos_token_id) | (next_tokens.squeeze(-1) == end_thought)
        if eos_reached.all():
            break

    # 5️⃣ Post-process: split per-rollout
    rollouts_ct_list = [rollouts_ct[i, :].unsqueeze(0) for i in range(num_rollouts)]
    per_token_logp_list = [per_token_logp[i, :] for i in range(num_rollouts)]

    return rollouts_ct_list, per_token_logp_list


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
        truncated_layers.append((keys[:, :, :limit, :].contiguous(), values[:, :, :limit, :].contiguous()))

    return DynamicCache.from_legacy_cache(tuple(truncated_layers))


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
model.config.use_flash_attention = USE_FLASH_ATTN

if COMPILE_MODEL:
    print("Compiling model...")
    model = torch.compile(model)

ema_model = copy.deepcopy(model).to(DEVICE)
for p in ema_model.parameters():
    p.requires_grad_(False)

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
    
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x) # no shuffle for streaming dataset

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
loop = tqdm(loader, desc="GRLP Training", total=len(ds) // BATCH_SIZE)
for batch_raw in loop:
    # collate: batch_raw is a list of dataset items; we handle batch size 1 primarily
    item = batch_raw[0]
    full_input_ids = torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, L)
    L = full_input_ids.size(1)
    # choose some positions t to apply RLP on in this example: for fast debug pick a few
    # In large runs you'd iterate t across the sequence

    if L <= HORIZON + 3:
        continue
    num_candidate_positions = min(BATCH_SIZE, L - HORIZON - 1)
    candidate_positions = torch.randint(2, L - HORIZON - 1, (num_candidate_positions,)).tolist()

    prefix_cache_map = collect_prefix_caches(model, full_input_ids, candidate_positions)

    batch_advantages = []
    batch_rewards = []
    batch_cot_lengths = []
    batch_prefixes = []
    batch_targets = []
    batch_cts = []
    batch_logp_old = []
    batch_s_ema_per_token = []
    
    # For each selected position t, compute G rollouts and log probs under old_theta
    for t in candidate_positions:
        prefix, gold_window, full_sequence = make_prefix_and_gold_from_full_input(full_input_ids, t)  # shapes (1, P-1), (1, H), (1, P-1+H)
        P = prefix.size(1)
        H_current = gold_window.size(1)

        if P == 0 or H_current == 0:
            print(f'skipping position {t} with P={P}, H_current={H_current}, document_len={full_input_ids.size(1)}')
            print(candidate_positions)
            continue

        # prefix_str = tokenizer.decode(prefix[:, -2*H_current:].squeeze(0).tolist())
        # target_str = tokenizer.decode(gold_window.squeeze(0).tolist())

        # compute s_ema per token under EMA model (no-think baseline)
        s_ema_per_token = compute_teacher_forced_logprobs(ema_model, full_sequence, gold_window, keep_grad=False)  # (H,)
        # shape already (H,) representing log p(x_{t+k} | x_{<t+k}) under EMA

        reasoning_prefix = torch.cat([prefix, start_thought_id], dim=1)  # (1, P)
        cache_for_prefix = prefix_cache_map.get(P)
        rollouts_ct, per_rollout_thought_logprobs_old = rollout(
            model,
            reasoning_prefix,
            prefix_cache=cache_for_prefix,
        )

        # Now for each rollout evaluate the reasoned per-token log-probs under the current model (p_theta)
        returns = compute_returns(rollouts_ct, reasoning_prefix, gold_window, model, s_ema_per_token)

        # Compute group-relative advantages (Eq.7)
        r_mean = returns.mean()
        # advantage scaling factor G/(G-1)
        advantages = (G / (G - 1)) * (returns - r_mean)  # shape (G,)

        batch_advantages.append(advantages)
        batch_rewards.append(returns.detach().cpu())

        batch_prefixes.append(reasoning_prefix)
        batch_targets.append(gold_window)
        batch_cts.append(rollouts_ct)
        batch_logp_old.append(per_rollout_thought_logprobs_old)
        batch_s_ema_per_token.append(s_ema_per_token)
        batch_cot_lengths.append(float(sum([ct.size(1) for ct in rollouts_ct]) / G))

    if not batch_rewards:
        print("No valid rollouts in batch, skipping...")
        continue

    batch_rewards = torch.stack(batch_rewards)
    avg_reward = float(batch_rewards.mean())
    metric_history["reward"].append(avg_reward)
    metric_history["reward_std"].append(float(batch_rewards.std()))
    avg_cot_length_value = sum(batch_cot_lengths) / len(batch_cot_lengths)
    metric_history["cot_length"].append(avg_cot_length_value)

    global_step += 1

    avg_loss = 0.0
    # For each selected position t, compute G rollouts and returns; then accumulate surrogate losses
    for epoch in range(PPO_EPOCHS):
        batch_rewards = []
        batch_surrogate_losses = []
        optimizer.zero_grad()

        for prefix, gold_window, rollouts_ct, per_rollout_thought_logprobs_old, s_ema_per_token, rollout_advantages in zip(
            batch_prefixes, batch_targets, batch_cts, batch_logp_old, batch_s_ema_per_token, batch_advantages
        ):

            # - compute reasoned per-token logprobs s_pred^i for horizon H under current model conditioned on prefix + c_t + gold_window
            per_rollout_thought_logprobs_new = []  # per-rollout per-token logprobs under current model for thought tokens
            for i in range(len(rollouts_ct)):
                ct_tokens = rollouts_ct[i]         # shape (1, C)
                # Build input: prefix + cot_tokens + gold_window
                inp_thought = torch.cat([prefix, ct_tokens], dim=1)  # (1, P+C)
                # compute per-token log-probs under current model for cot_tokens
                per_token_logp_new = compute_teacher_forced_logprobs(model, inp_thought, ct_tokens)  # (C,)
                per_rollout_thought_logprobs_new.append(per_token_logp_new)  # (C,)

            surrogate_loss = compute_surrogate_loss(
                per_rollout_thought_logprobs_new,
                per_rollout_thought_logprobs_old,
                rollout_advantages
            )
            surrogate_loss.backward()
            batch_surrogate_losses.append(float(surrogate_loss.detach().cpu()))

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        avg_loss += sum(batch_surrogate_losses) / len(batch_surrogate_losses) / PPO_EPOCHS

    metric_history["loss"].append(avg_loss)
    postfix = {
        "loss": avg_loss,
        "reward": avg_reward,
        "cot_len": avg_cot_length_value,
    }
    loop.set_postfix(postfix)

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
    
    torch.cuda.empty_cache()

# optional: save checkpoint each epoch
save_dir = f"{MODEL_NAME.split('/')[-1]}-grlp"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

save_metric_plot(metric_history)

print_gpu_memory("Post-training")

print("done")
