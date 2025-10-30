"""
Generalized RLP prototype (discounted-return advantage estimation)
- Model: Qwen/Qwen3-0.6B-Base (or any causal LM)
- Dataset: allenai/dolma (use a small subset for testing)
"""

import copy
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
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
# DATASET = "allenai/dolma"
DATASET = "HuggingFaceFW/fineweb"
SPLIT = "train"
DATA_CACHE_DIR = Path("data/fineweb-10k-tokenized")
MAX_SEQ_LEN = 2048
HORIZON = 8                             # reward horizon T (small for debug; paper uses long)
THOUGHT_MAX_TOKENS = 200
G = 4                                   # number of rollouts per context
GAMMA = 0.7                             # discount factor
BATCH_SIZE = 8                          # token-level RLP is expensive; tune for your memory
LR = 1e-6
TAU = 0.999                             # EMA decay
EPS_CLIP_LOW, EPS_CLIP_HIGH = 0.1, 0.1  # PPO clipping
PPO_EPOCHS = 3                          # number of policy optimization iterations per batch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_data = 10_000
# N_data = 1000
dataset_dir = None
DEBUG_LOG_PATH = Path("debug.txt")

PLOT_SAVE_INTERVAL = 5
MODEL_SAVE_INTERVAL = 200
METRIC_FIG_PATH = Path("simple_grpl_training_metrics.png")


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

def save_metric_plot(
        reward_history,
        reward_std_history,
        cot_length_history,
        loss_history,
    ):
    if not reward_history:
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    step_history = range(1, len(reward_history) + 1)
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
    Returns two 2D tensors with batch dim: (1, prefix_len), (1, H)
    """
    prefix = input_ids[:, t-MAX_SEQ_LEN:t]
    gold = input_ids[:, t : t + HORIZON]
    return prefix, gold

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


def rollout(model, prefix, P):
    # For rollouts we need:
    # - sample G thoughts c_t^{(i)} ~ pi_{theta_old}( . | x_{<t})
    # - compute likelihood under theta_old for importance sampling
    with inference_context(model, with_grad=False):
        rollouts_ct = []
        per_rollout_thought_logprobs_old = []  # per-rollout per-token logprobs under theta_old for the thought tokens (for importance)
        # Sample G thoughts using theta_old_model. Use generate for clarity; for production sample step-wise while recording logprobs.
        # Simple generation: supply prefix, generate short CoT (max_new_tokens ~ 32)
        # We generate only the CoT tokens, not the gold next tokens.
        in_ids = torch.cat([prefix, start_thought_id], dim=1)  # (1, P+1)
        for gidx in range(G):
            # To call generate with prefix as tensors we need to decode + generate, or use generate with input_ids directly:
            # use do_sample True and some sampling config
            generated = model.generate(
                in_ids, 
                attention_mask=torch.ones_like(in_ids),
                max_new_tokens=THOUGHT_MAX_TOKENS,
                do_sample=True, 
                top_p=0.95, 
                eos_token_id=[tokenizer.eos_token_id, end_thought_id.item()],
                pad_token_id=tokenizer.pad_token_id,
            )
            if generated[0, -1] == tokenizer.eos_token_id:
                generated[0, -1] = end_thought_id.item()
            elif generated[0, -1] != end_thought_id.item():
                generated = torch.cat([generated, end_thought_id], dim=1)

            # generated contains prefix + cot; remove the prefix part to get only cot tokens
            cot_tokens = generated[:, P:]  # shape (1, C)
            # log_sampled_thought(global_step, t, gidx, cot_tokens.squeeze(0), prefix_str, target_str)

            # (keep as-is; the logprob computation will handle exact tokens)
            rollouts_ct.append(cot_tokens)

            # compute per-token logprob of cot_tokens under theta_old (behavior). We'll need these to form log pi_old per token.
            # we want the logprob of each token in cot_tokens (teacher forcing style)
            # logits where position j predicts next token j+1; probabilities for cot token u are at logits indices P+u-1
            per_token_logp_old = compute_teacher_forced_logprobs(model, generated, cot_tokens, keep_grad=False)  # (C,)
            per_rollout_thought_logprobs_old.append(per_token_logp_old)

    return rollouts_ct, per_rollout_thought_logprobs_old


# -----------------------------
# Load model, tokenizer, dataset
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Set pad_token to eos_token to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
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


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
ema_model = copy.deepcopy(model).to(DEVICE)
for p in ema_model.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print_gpu_memory("Post-model-load")

if os.path.exists(DATA_CACHE_DIR):
    ds = Dataset.load_from_disk(DATA_CACHE_DIR)
else:
    # Load a small subset for debugging. For full experiments, remove slicing.
    ds = load_dataset(DATASET, split=SPLIT, streaming=True).take(N_data)
    def tokenize_batch(examples):
        # assume examples["text"] exists; truncate to MAX_SEQ_LEN
        out = tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LEN, return_tensors=None)
        return out
    ds = ds.map(lambda ex: {"input_ids": tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]})
    ds = Dataset.from_list(list(ds))
    ds.save_to_disk(DATA_CACHE_DIR)
    
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x) # no shuffle for streaming dataset

# reset log for new training run
DEBUG_LOG_PATH.write_text("", encoding="utf-8")

# Metric tracking for visualization
reward_history = []
reward_std_history = []
cot_length_history = []
loss_history = []

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
    candidate_positions = torch.randint(1, max(2, L - HORIZON - 1), (min(BATCH_SIZE, max(1, L - HORIZON - 1)),)).tolist()
    if len(candidate_positions) == 0:
        continue

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
        prefix, gold_window = make_prefix_and_gold_from_full_input(full_input_ids, t)  # shapes (1, P), (1, H)
        P = prefix.size(1)
        H_current = gold_window.size(1)

        # prefix_str = tokenizer.decode(prefix[:, -2*H_current:].squeeze(0).tolist())
        # target_str = tokenizer.decode(gold_window.squeeze(0).tolist())

        # compute s_ema per token under EMA model (no-think baseline)
        # EMA input: prefix + gold_window
        ema_input = full_input_ids[:, :P+H_current] # (1, P+H)
        # compute per-token logprobs for the gold_window under EMA
        s_ema_per_token = compute_teacher_forced_logprobs(ema_model, ema_input, gold_window, keep_grad=False)  # (H,)
        # shape already (H,) representing log p(x_{t+k} | x_{<t+k}) under EMA

        rollouts_ct, per_rollout_thought_logprobs_old = rollout(model, prefix, P)

        # Now for each rollout evaluate the reasoned per-token log-probs under the current model (p_theta)
        returns = compute_returns(rollouts_ct, prefix, gold_window, model, s_ema_per_token)

        # Compute group-relative advantages (Eq.7)
        r_mean = returns.mean()
        # advantage scaling factor G/(G-1)
        advantages = (G / (G - 1)) * (returns - r_mean)  # shape (G,)

        batch_advantages.append(advantages)
        batch_rewards.append(returns.detach().cpu())

        batch_prefixes.append(prefix)
        batch_targets.append(gold_window)
        batch_cts.append(rollouts_ct)
        batch_logp_old.append(per_rollout_thought_logprobs_old)
        batch_s_ema_per_token.append(s_ema_per_token)
        batch_cot_lengths.extend([ct.size(1) for ct in rollouts_ct])

    batch_rewards = torch.stack(batch_rewards)
    avg_reward = float(batch_rewards.mean())
    reward_history.append(avg_reward)
    reward_std_history.append(float(batch_rewards.std()))
    avg_cot_length_value = float(sum(batch_cot_lengths) / len(batch_cot_lengths)) if batch_cot_lengths else None
    cot_length_history.append(avg_cot_length_value)

    global_step += 1

    epoch_loss = 0.0
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

        optimizer.step()

        epoch_loss += sum(batch_surrogate_losses)

    loss_value = epoch_loss / (PPO_EPOCHS * len(candidate_positions))
    loss_history.append(loss_value)

    postfix = {"loss": loss_value, "step": global_step}
    postfix["avg_reward"] = avg_reward
    postfix["avg_cot_len"] = avg_cot_length_value
    loop.set_postfix(postfix)

    # TODO: remove this debug print after verifying
    print(f"Global step {global_step}: clipped tokens so far = {clipped}")

    # EMA update
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(TAU).add_(p.data, alpha=1.0 - TAU)

    if global_step and global_step % PLOT_SAVE_INTERVAL == 0:
        save_metric_plot(reward_history, reward_std_history, cot_length_history, loss_history)
    
    if global_step and global_step % MODEL_SAVE_INTERVAL == 0:
        save_dir = f"{MODEL_NAME.split('/')[-1]}-grlp-step{global_step}"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    
    torch.cuda.empty_cache()

# optional: save checkpoint each epoch
save_dir = f"{MODEL_NAME.split('/')[-1]}-grlp"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

save_metric_plot(reward_history, reward_std_history, cot_length_history, loss_history)

print_gpu_memory("Post-training")

print("done")
