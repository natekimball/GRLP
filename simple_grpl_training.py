"""
Generalized RLP prototype (discounted-return advantage estimation)
- Model: Qwen/Qwen3-0.6B-Base (or any causal LM)
- Dataset: allenai/dolma (use a small subset for testing)
"""

import copy
import os
from contextlib import nullcontext
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
HORIZON = 8                          # reward horizon T (small for debug; paper uses long)
THOUGHT_MAX_TOKENS = 128
G = 8                                  # number of rollouts per context
GAMMA = 0.7                            # discount factor
BATCH_SIZE = 1                         # token-level RLP is expensive; tune for your memory
LR = 1e-6
TAU = 0.999                            # EMA decay
EPS_CLIP_LOW, EPS_CLIP_HIGH = 0.1, 0.1 # PPO clipping
NUM_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_data = 10_000
# N_data = 1000
dataset_dir = None
DEBUG_LOG_PATH = Path("debug.txt")

# Metric tracking for visualization
reward_history = []
cot_length_history = []
loss_history = []
step_history = []
PLOT_SAVE_INTERVAL = 20
MODEL_SAVE_INTERVAL = 200
METRIC_FIG_PATH = Path("simple_grpl_training_metrics.png")


def _bf16_supported() -> bool:
    if DEVICE != "cuda":
        return False
    if hasattr(torch.cuda, "is_bf16_supported"):
        return torch.cuda.is_bf16_supported()
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


MODEL_DTYPE = (
    torch.bfloat16
    if DEVICE == "cuda" and _bf16_supported()
    else torch.float16 if DEVICE == "cuda" else torch.float32
)
USE_AUTOCAST = DEVICE == "cuda" and MODEL_DTYPE in (torch.float16, torch.bfloat16)


def get_autocast_context():
    return torch.autocast(device_type="cuda", dtype=MODEL_DTYPE) if USE_AUTOCAST else nullcontext()


def save_metric_plot():
    if not reward_history:
        return
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(step_history, reward_history, color="tab:blue")
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

# -----------------------------
# Helpers
# -----------------------------
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


def compute_teacher_forced_logprobs(model, input_ids, gold_targets):
    """
    Compute per-token log-probs under teacher forcing for the sequence:
      model(input_ids) where input_ids = prefix + optional cot + gold_seq
    We want the log-probabilities for gold_seq positions (the final len(gold_targets) tokens).
    Returns: tensor shape (len(gold_targets),) of log-probs (float)
    """
    # model returns logits for each position predicting the *next* token
    with get_autocast_context():
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
    return per_token_logp.squeeze(0)  # (H,)


def make_prefix_and_gold_from_full_input(input_ids, t):
    """
    Given full document token sequence (2D tensor), build:
      prefix = input_ids[:t]
      gold_window = input_ids[t : t + HORIZON]
    Returns two 2D tensors with batch dim: (1, prefix_len), (1, H)
    """
    prefix = input_ids[:, :t]
    gold = input_ids[:, t : t + HORIZON]
    return prefix, gold


def compute_prefix_state(model, prefix_ids):
    """Compute past key values and attention mask for a fixed prefix."""
    if prefix_ids.size(1) == 0:
        attention_mask = torch.zeros_like(prefix_ids, dtype=torch.long)
        return None, attention_mask
    attention_mask = torch.ones(prefix_ids.size(0), prefix_ids.size(1), dtype=torch.long, device=prefix_ids.device)
    was_checkpointing = getattr(model, "is_gradient_checkpointing", False) and model.training
    if was_checkpointing:
        model.gradient_checkpointing_disable()
    prev_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    try:
        with get_autocast_context():
            outputs = model(input_ids=prefix_ids, attention_mask=attention_mask, use_cache=True)
    finally:
        model.config.use_cache = prev_use_cache
        if was_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = prev_use_cache
    return outputs.past_key_values, attention_mask


def compute_sequence_logprobs_with_cache(model, token_ids, past_key_values, attention_mask):
    """Compute per-token log probabilities given cached prefix state."""
    if token_ids.size(1) == 0:
        empty = torch.empty(token_ids.size(1), device=token_ids.device, dtype=torch.float32)
        return empty, past_key_values, attention_mask
    if attention_mask is None:
        raise ValueError("attention_mask must be provided when using cached computation")
    new_mask = torch.ones(token_ids.size(0), token_ids.size(1), dtype=attention_mask.dtype, device=token_ids.device)
    extended_attention = torch.cat([attention_mask, new_mask], dim=1)
    was_checkpointing = getattr(model, "is_gradient_checkpointing", False) and model.training
    if was_checkpointing:
        model.gradient_checkpointing_disable()
    prev_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = True
    try:
        with get_autocast_context():
            outputs = model(
                input_ids=token_ids,
                attention_mask=extended_attention,
                past_key_values=past_key_values,
                use_cache=True,
            )
    finally:
        model.config.use_cache = prev_use_cache
        if was_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model.config.use_cache = prev_use_cache
    logits = outputs.logits
    logprobs = gather_token_logprobs_from_logits(logits, token_ids).to(torch.float32)
    return logprobs.squeeze(0), outputs.past_key_values, extended_attention


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


def log_sampled_thought(step_idx: int, epoch_idx: int, position: int, rollout_idx: int, cot_tokens: torch.Tensor, prefix: str, target: str) -> None:
    """Append the decoded chain-of-thought sample to the debug log."""
    decoded = tokenizer.decode(cot_tokens.tolist(), skip_special_tokens=False)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[step={step_idx} epoch={epoch_idx} t={position} rollout={rollout_idx} len={cot_tokens.size(1)}]\n")
        log_file.write(f"Prefix: {prefix}\n")
        log_file.write(f"Target: {target}\n")
        log_file.write(repr(decoded) + "\n\n")


model_kwargs = {"torch_dtype": MODEL_DTYPE}
if DEVICE == "cuda":
    model_kwargs["attn_implementation"] = "flash_attention_2"
try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, **model_kwargs)
except (TypeError, ValueError):
    model_kwargs.pop("attn_implementation", None)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, **model_kwargs)
model.to(DEVICE)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

ema_model = copy.deepcopy(model).to(DEVICE)
ema_model.eval()
ema_model.config.use_cache = True
for p in ema_model.parameters():
    p.requires_grad_(False)

# Reusable snapshot for behavior policy; avoid reallocating every step
theta_old_model = copy.deepcopy(model).to(DEVICE)
theta_old_model.eval()
theta_old_model.config.use_cache = True
for p in theta_old_model.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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

# -----------------------------
# Training loop
# -----------------------------
model.train()
global_step = 0
for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_raw in loop:
        # collate: batch_raw is a list of dataset items; we handle batch size 1 primarily
        item = batch_raw[0]
        full_input_ids = torch.tensor(item["input_ids"], dtype=torch.long).unsqueeze(0).to(DEVICE)  # (1, L)
        L = full_input_ids.size(1)
        # choose some positions t to apply RLP on in this example: for fast debug pick a few
        # In large runs you'd iterate t across the sequence (possibly sampled)
        candidate_positions = list(range(1, max(2, min(L - HORIZON - 1, 8))))  # up to 8 pos or less
        if len(candidate_positions) == 0:
            continue

        # Snapshot theta_old for sampling (behavior policy)
        with torch.no_grad():
            theta_old_model.load_state_dict(model.state_dict())
        theta_old_model.eval()

        # For each selected position t, compute G rollouts and returns; then accumulate surrogate losses
        surrogate_losses = []
        batch_rewards = []
        batch_cot_lengths = []
        for t in candidate_positions:
            prefix, gold_window = make_prefix_and_gold_from_full_input(full_input_ids, t)  # shapes (1, P), (1, H)
            P = prefix.size(1)
            H_current = gold_window.size(1)

            prefix_str = tokenizer.decode(prefix[:, -2 * H_current :].squeeze(0).tolist())
            target_str = tokenizer.decode(gold_window.squeeze(0).tolist())

            with torch.no_grad():
                ema_prefix_past, ema_prefix_mask = compute_prefix_state(ema_model, prefix)
                s_ema_per_token, _, _ = compute_sequence_logprobs_with_cache(
                    ema_model, gold_window, ema_prefix_past, ema_prefix_mask
                )
                theta_prefix_past, theta_prefix_mask = compute_prefix_state(theta_old_model, prefix)
                model_prefix_past, model_prefix_mask = compute_prefix_state(model, prefix)

            rollout_infos = []
            returns = []
            in_ids = torch.cat([prefix, start_thought_id], dim=1)
            for gidx in range(G):
                with torch.no_grad():
                    generated = theta_old_model.generate(
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

                cot_tokens = generated[:, P:]
                if cot_tokens.size(1) == 0:
                    continue

                log_sampled_thought(global_step, epoch, t, gidx, cot_tokens.squeeze(0), prefix_str, target_str)
                batch_cot_lengths.append(int(cot_tokens.size(1)))

                with torch.no_grad():
                    logp_old, _, _ = compute_sequence_logprobs_with_cache(
                        theta_old_model, cot_tokens, theta_prefix_past, theta_prefix_mask
                    )

                rollout_infos.append(
                    {
                        "cot_tokens": cot_tokens.clone().detach(),
                        "logp_old": logp_old.detach().cpu(),
                    }
                )

                with torch.no_grad():
                    _, cot_past, cot_mask = compute_sequence_logprobs_with_cache(
                        model, cot_tokens, model_prefix_past, model_prefix_mask
                    )
                    s_pred_per_token, _, _ = compute_sequence_logprobs_with_cache(
                        model, gold_window, cot_past, cot_mask
                    )

                r_per_token = s_pred_per_token - s_ema_per_token.to(s_pred_per_token.device)
                discount_base = torch.full((H_current,), GAMMA, device=r_per_token.device, dtype=r_per_token.dtype)
                exponents = torch.arange(H_current, device=r_per_token.device, dtype=r_per_token.dtype)
                discounts = torch.pow(discount_base, exponents)
                R = torch.dot(r_per_token, discounts)
                returns.append(R.detach())
                batch_rewards.append(float(R.detach().cpu()))

            effective_rollouts = len(rollout_infos)
            if effective_rollouts == 0:
                continue

            returns_tensor = torch.stack(returns).to(torch.float32)
            r_mean = returns_tensor.mean()
            if effective_rollouts > 1:
                advantages = (effective_rollouts / (effective_rollouts - 1)) * (returns_tensor - r_mean)
            else:
                advantages = returns_tensor - r_mean

            for adv, rollout_info in zip(advantages, rollout_infos):
                cot_tokens = rollout_info["cot_tokens"].to(DEVICE)
                logp_old = rollout_info["logp_old"].to(device=cot_tokens.device, dtype=torch.float32)
                logp_new, _, _ = compute_sequence_logprobs_with_cache(
                    model, cot_tokens, model_prefix_past, model_prefix_mask
                )
                logp_new = logp_new.to(torch.float32)
                log_rhos = logp_new - logp_old
                rhos = torch.exp(log_rhos)
                clip_rhos = torch.clamp(rhos, 1.0 - EPS_CLIP_LOW, 1.0 + EPS_CLIP_HIGH)
                adv_val = adv.to(logp_new.device).detach()
                surrogate_per_token = torch.min(rhos * adv_val, clip_rhos * adv_val)
                loss_i = -surrogate_per_token.mean()
                surrogate_losses.append(loss_i)

        if len(surrogate_losses) == 0:
            continue

        total_loss = torch.stack(surrogate_losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        avg_reward_value = float(sum(batch_rewards) / len(batch_rewards)) if batch_rewards else None
        avg_cot_length_value = float(sum(batch_cot_lengths) / len(batch_cot_lengths)) if batch_cot_lengths else None
        loss_value = float(total_loss.detach().cpu())

        # EMA update
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(TAU).add_(p.data, alpha=1.0 - TAU)

        global_step += 1
        postfix = {"loss": loss_value, "step": global_step}
        if avg_reward_value is not None:
            postfix["avg_reward"] = avg_reward_value
        if avg_cot_length_value is not None:
            postfix["avg_cot_len"] = avg_cot_length_value
        loop.set_postfix(postfix)

        if avg_reward_value is not None and avg_cot_length_value is not None:
            reward_history.append(avg_reward_value)
            cot_length_history.append(avg_cot_length_value)
            loss_history.append(loss_value)
            step_history.append(global_step)
            if global_step % PLOT_SAVE_INTERVAL == 0:
                save_metric_plot()
        
        if global_step and global_step % MODEL_SAVE_INTERVAL == 0:
            save_dir = f"{MODEL_NAME.split('/')[-1]}-grlp-epoch{epoch}-step{global_step}"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    # optional: save checkpoint each epoch
    save_dir = f"{MODEL_NAME.split('/')[-1]}-grlp-epoch{epoch}"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

save_metric_plot()

print("done")
