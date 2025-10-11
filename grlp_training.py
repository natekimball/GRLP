"""
Generalized RLP prototype (discounted-return advantage estimation)
- Model: Qwen/Qwen3-0.6B-Base (or any causal LM)
- Dataset: allenai/dolma (use a small subset for testing)
"""

import copy
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"   # change to local checkpoint if needed
# DATASET = "allenai/dolma"
DATASET = "HuggingFaceFW/fineweb"
SPLIT = "train"       # small subset for debug; remove for real runs
MAX_SEQ_LEN = 2048
HORIZON = 32                          # reward horizon T (small for debug; paper uses long)
G = 4                                  # number of rollouts per context
GAMMA = 0.9                            # discount factor
BATCH_SIZE = 1                         # token-level RLP is expensive; tune for your memory
LR = 1e-5
TAU = 0.999                            # EMA decay
EPS_CLIP_LOW, EPS_CLIP_HIGH = 0.1, 0.1 # PPO clipping
NUM_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    Given full document token sequence (1D tensor), build:
      prefix = input_ids[:t]
      gold_window = input_ids[t : t + HORIZON]
    Returns two 2D tensors with batch dim: (1, prefix_len), (1, H)
    """
    full = input_ids.squeeze(0)
    prefix = full[:t].unsqueeze(0)
    gold = full[t : t + HORIZON].unsqueeze(0)
    return prefix, gold


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


model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
ema_model = copy.deepcopy(model).to(DEVICE)
for p in ema_model.parameters():
    p.requires_grad_(False)

# Reusable snapshot for behavior policy; avoid reallocating every step
theta_old_model = copy.deepcopy(model).to(DEVICE)
theta_old_model.eval()
for p in theta_old_model.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Load a small subset for debugging. For full experiments, remove slicing.
ds = load_dataset(DATASET, split=SPLIT, streaming=True)
def tokenize_batch(examples):
    # assume examples["text"] exists; truncate to MAX_SEQ_LEN
    out = tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LEN, return_tensors=None)
    return out
ds = ds.map(lambda ex: {"input_ids": tokenizer(ex["text"], truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]})
# loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)
loader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=lambda x: x)

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
        for t in candidate_positions:
            prefix, gold_window = make_prefix_and_gold_from_full_input(full_input_ids, t)  # shapes (1, P), (1, H)
            P = prefix.size(1)
            H_current = gold_window.size(1)

            # compute s_ema per token under EMA model (no-think baseline)
            with torch.no_grad():
                # EMA input: prefix + gold_window
                ema_input = torch.cat([prefix, gold_window], dim=1)  # (1, P+H)
                # ema_input = full_input_ids[:, P+H_current]
                # compute per-token logprobs for the gold_window under EMA
                ema_per_token_logp = compute_teacher_forced_logprobs(ema_model, ema_input, gold_window)  # (H,)
                # shape already (H,) representing log p(x_{t+k} | x_{<t+k}) under EMA

            # For rollouts we need:
            # - sample G thoughts c_t^{(i)} ~ pi_{theta_old}( . | x_{<t})
            # - compute reasoned per-token logprobs s_pred^i for horizon H under current model conditioned on prefix + c_t + gold_window
            rollouts_ct = []
            rollouts_logprob_old_tokens = []  # per-rollout per-token logprobs under theta_old for the thought tokens (for importance)
            # Sample G thoughts using theta_old_model. Use generate for clarity; for production sample step-wise while recording logprobs.
            for gidx in range(G):
                # Simple generation: supply prefix, generate short CoT (max_new_tokens ~ 32)
                # We generate only the CoT tokens, not the gold next tokens.
                in_ids = torch.cat([prefix, start_thought_id], dim=1)  # (1, P+1)
                # To call generate with prefix as tensors we need to decode + generate, or use generate with input_ids directly:
                # use do_sample True and some sampling config
                generated = theta_old_model.generate(
                    in_ids, 
                    attention_mask=torch.ones_like(in_ids),
                    max_new_tokens=32, 
                    do_sample=True, 
                    top_p=0.95, 
                    eos_token_id=[tokenizer.eos_token_id, end_thought_id.item()],
                    pad_token_id=tokenizer.pad_token_id,
                )
                # generated contains prefix + cot + maybe EOS; remove the prefix part to get only cot tokens
                cot_tokens = generated[:, P:]  # shape (1, C)
                assert generated[0,0] == start_thought_id.item(), "Generation should start with <think> token"
                if cot_tokens.size(1) <= 1:
                    continue
                # remove trailing eos if present
                # (keep as-is; the logprob computation will handle exact tokens)
                rollouts_ct.append(cot_tokens)

                # compute per-token logprob of cot_tokens under theta_old (behavior). We'll need these to form log pi_old per token.
                # For that, compute logits of theta_old over cot_tokens autoregressively conditioned on prefix.
                with torch.no_grad():
                    # we want the logprob of each token in cot_tokens (teacher forcing style)
                    # logits where position j predicts next token j+1; probabilities for cot token u are at logits indices P+u-1
                    per_token_logp_old = compute_teacher_forced_logprobs(theta_old_model, generated, cot_tokens)  # (C,)
                    rollouts_logprob_old_tokens.append(per_token_logp_old.detach())

            # Now for each rollout evaluate the reasoned per-token log-probs under the current model (p_theta)
            returns = []  # discounted returns R(c_t) for each rollout
            per_rollout_thought_logprobs_new = []  # list of per-token logp for thought tokens under current model
            for i in range(len(rollouts_ct)):
                c_tokens = rollouts_ct[i]         # shape (1, C)
                C = c_tokens.size(1)
                # Build input: prefix + cot_tokens + gold_window
                inp_reasoned = torch.cat([prefix, c_tokens, gold_window], dim=1)  # (1, P+C+H)
                # The gold targets corresponding to the horizon are the last H tokens of inp_reasoned
                gold_targets_in_reasoned = inp_reasoned[:, -(H_current):]  # (1, H)
                # compute per-token log-probs under current model for gold_window
                with torch.no_grad():
                    s_pred_per_token = compute_teacher_forced_logprobs(model, inp_reasoned, gold_targets_in_reasoned)  # (H,)
                # compute r_i per token = s_pred^i - s_ema
                # ensure ema_per_token_logp has same length H_current
                r_per_token = s_pred_per_token - ema_per_token_logp[:H_current].to(s_pred_per_token.device)
                # discounted sum
                discounts = torch.tensor([GAMMA ** k for k in range(H_current)], device=r_per_token.device)
                R = (discounts * r_per_token).sum().detach()  # treat R as constant (no grad)
                returns.append(R)

                # compute per-token logprobs of the thought tokens under current model (for policy)
                # we'll compute the new-model per-token logprobs on the thought tokens conditioned on prefix
                inp_for_cot_new = torch.cat([prefix, c_tokens], dim=1)  # (1, P+C)
                # Remove torch.no_grad() here since we need gradients for the policy loss
                targets_cot = inp_for_cot_new[:, P:]  # (1, C)
                per_token_logp_new = compute_teacher_forced_logprobs(model, inp_for_cot_new, targets_cot)  # (C,)
                per_rollout_thought_logprobs_new.append(per_token_logp_new)  # (C,)

            # Convert returns to tensor and compute group-relative advantages (Eq.7)
            returns_tensor = torch.stack(returns)  # (G,)
            r_mean = returns_tensor.mean().detach()
            # advantage scaling factor G/(G-1)
            advantages = (G / (G - 1)) * (returns_tensor - r_mean)  # shape (G,)
            # stop gradients on advantages
            advantages = advantages.detach()

            # For each rollout, compute per-thought clipped surrogate loss (Eq.8)
            for i in range(G):
                # per-token log probs under new/current (we have per_token_logp_new)
                logp_new = per_rollout_thought_logprobs_new[i]  # shape (C,)
                # per-token log probs under old/behavior (we computed earlier in rollouts_logprob_old_tokens)
                logp_old = rollouts_logprob_old_tokens[i]      # shape (C,)
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
                surrogate_losses.append(loss_i)

        if len(surrogate_losses) == 0:
            continue

        total_loss = torch.stack(surrogate_losses).mean()
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(TAU).add_(p.data, alpha=1.0 - TAU)

        global_step += 1
        loop.set_postfix({"loss": float(total_loss.detach().cpu()), "step": global_step})

    # optional: save checkpoint each epoch
    model.save_pretrained(f"qwen3-0.6B-rlp-epoch{epoch}")
    tokenizer.save_pretrained(f"qwen3-0.6B-rlp-epoch{epoch}")

print("done")
