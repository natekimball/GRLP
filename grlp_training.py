"""
Generalized RLP prototype (discounted-return advantage estimation)
- Model: Qwen/Qwen3-0.6B-Base (or any causal LM)
- Dataset: allenai/dolma (use a small subset for testing)
"""

import copy
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from tqdm import tqdm
import os

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
BATCH_SIZE = 4                         # leverage batching for better GPU utilization
MAX_DATASET_SAMPLES = 10_000           # cap for cached subset
DATA_CACHE_DIR = Path("data/fineweb-10k-tokenized")
THOUGHT_MAX_TOKENS = 128
THOUGHT_TOP_P = 0.95
THOUGHT_TEMPERATURE = 0.7
LR = 1e-5
TAU = 0.999                            # EMA decay
EPS_CLIP_LOW, EPS_CLIP_HIGH = 0.1, 0.1 # PPO clipping
NUM_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGPROB_CHUNK_SIZE = 2                 # split large log-prob batches to save memory
# N_data = 10_000
N_data = 100
dataset_dir = None

AMP_ENABLED = DEVICE.startswith("cuda")
AMP_DTYPE = torch.float16


def autocast_context():
    return torch.cuda.amp.autocast(dtype=AMP_DTYPE) if AMP_ENABLED else nullcontext()

# -----------------------------
# Helpers
# -----------------------------
def pad_sequences_1d(sequences, pad_token_id):
    if not sequences:
        raise ValueError("pad_sequences_1d received an empty sequence list")
    device = sequences[0].device
    lengths = torch.tensor([seq.size(0) for seq in sequences], device=device)
    max_len = lengths.max().item()
    padded = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long, device=device)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.size(0)] = seq
    attention_mask = (padded != pad_token_id).long()
    return padded, attention_mask, lengths


def batched_token_logprobs(
    model,
    sequences,
    prefix_lengths,
    target_lengths,
    pad_token_id,
    requires_grad=False,
):
    if not sequences:
        return []

    results = []
    for start_idx in range(0, len(sequences), LOGPROB_CHUNK_SIZE):
        end_idx = start_idx + LOGPROB_CHUNK_SIZE
        seq_chunk = sequences[start_idx:end_idx]
        prefix_chunk = prefix_lengths[start_idx:end_idx]
        target_chunk = target_lengths[start_idx:end_idx]

        padded, attention_mask, _ = pad_sequences_1d(seq_chunk, pad_token_id)
        with torch.set_grad_enabled(requires_grad):
            with autocast_context():
                outputs = model(input_ids=padded, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                targets = padded[:, 1:]
                gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        for local_idx, (prefix_len, target_len) in enumerate(zip(prefix_chunk, target_chunk)):
            if target_len == 0:
                empty = torch.zeros(0, device=gathered.device, dtype=torch.float32, requires_grad=requires_grad)
                results.append(empty)
                continue
            start_pos = prefix_len - 1
            end_pos = start_pos + target_len
            results.append(gathered[local_idx, start_pos:end_pos].to(dtype=torch.float32))
    return results


# -----------------------------
# Load model, tokenizer, dataset
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# Set pad_token to eos_token to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
added_tokens = tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
if added_tokens > 0:
    model.resize_token_embeddings(len(tokenizer))

if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
model.config.use_cache = False

ema_model = copy.deepcopy(model).to(DEVICE)
if hasattr(ema_model, "gradient_checkpointing_disable"):
    ema_model.gradient_checkpointing_disable()
ema_model.config.use_cache = True
for p in ema_model.parameters():
    p.requires_grad_(False)

# Reusable snapshot for behavior policy; avoid reallocating every step
theta_old_model = copy.deepcopy(model).to(DEVICE)
theta_old_model.eval()
if hasattr(theta_old_model, "gradient_checkpointing_disable"):
    theta_old_model.gradient_checkpointing_disable()
theta_old_model.config.use_cache = True
for p in theta_old_model.parameters():
    p.requires_grad_(False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

start_thought_id = torch.tensor([
    tokenizer.convert_tokens_to_ids("<think>")
], device=DEVICE, dtype=torch.long)
end_thought_id = torch.tensor([
    tokenizer.convert_tokens_to_ids("</think>")
], device=DEVICE, dtype=torch.long)


def build_dataset() -> Dataset:
    if DATA_CACHE_DIR.exists():
        return Dataset.load_from_disk(str(DATA_CACHE_DIR))

    streaming_ds = load_dataset(DATASET, split=SPLIT, streaming=True)
    subset = []
    for idx, example in enumerate(streaming_ds):
        if idx >= MAX_DATASET_SAMPLES:
            break
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            add_special_tokens=False,
        )
        subset.append({"input_ids": tokenized["input_ids"]})
    dataset = Dataset.from_list(subset)
    DATA_CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(DATA_CACHE_DIR))
    return dataset


def collate_fn(batch):
    tensors = [torch.tensor(sample["input_ids"], dtype=torch.long) for sample in batch]
    padded = pad_sequence(tensors, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (padded != tokenizer.pad_token_id).long()
    return {"input_ids": padded, "attention_mask": attention_mask}


dataset = build_dataset()
loader = DataLoader(dataset.shuffle(seed=42), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# -----------------------------
# Training loop
# -----------------------------
model.train()
global_step = 0
for epoch in range(NUM_EPOCHS):
    loop = tqdm(loader, desc=f"Epoch {epoch}")
    for batch_raw in loop:
        input_ids = batch_raw["input_ids"].to(DEVICE)
        attention_mask = batch_raw["attention_mask"].to(DEVICE)
        lengths = attention_mask.sum(dim=1)

        valid_mask = lengths > 2
        if valid_mask.sum() == 0:
            continue
        input_ids = input_ids[valid_mask]
        attention_mask = attention_mask[valid_mask]
        lengths = lengths[valid_mask]

        horizons = torch.clamp(lengths - 1, min=1)
        horizons = torch.minimum(horizons, torch.full_like(horizons, HORIZON))
        max_prefix = lengths - horizons
        prefix_valid = max_prefix > 1
        if prefix_valid.sum() == 0:
            continue

        input_ids = input_ids[prefix_valid]
        lengths = lengths[prefix_valid]
        horizons = horizons[prefix_valid]
        max_prefix = max_prefix[prefix_valid]

        rand = torch.rand_like(lengths, dtype=torch.float)
        prefix_lengths = 1 + torch.floor(rand * (max_prefix.float() - 1)).long()

        with torch.no_grad():
            theta_old_model.load_state_dict(model.state_dict())
        theta_old_model.config.use_cache = True
        theta_old_model.eval()

        baseline_sequences = []
        baseline_prefix_lengths = []
        gold_lengths = []
        discounts = []
        thought_sequences = []
        thought_prefix_lengths = []
        thought_token_lengths = []
        reasoned_sequences = []
        reasoned_prefix_lengths = []
        gold_lengths_expanded = []
        thoughts_per_sample = []

        start_marker = start_thought_id
        end_marker = end_thought_id

        global_idx = 0
        for sample_idx in range(input_ids.size(0)):
            seq_len = lengths[sample_idx].item()
            horizon_len = horizons[sample_idx].item()
            prefix_len = prefix_lengths[sample_idx].item()
            seq = input_ids[sample_idx, :seq_len]

            gold_tokens = seq[prefix_len : prefix_len + horizon_len]
            if gold_tokens.size(0) == 0:
                continue

            prefix_tokens = seq[:prefix_len]
            baseline_sequences.append(torch.cat([prefix_tokens, gold_tokens], dim=0))
            baseline_prefix_lengths.append(prefix_len)
            gold_lengths.append(gold_tokens.size(0))
            exponents = torch.arange(gold_tokens.size(0), device=DEVICE, dtype=torch.float)
            base = torch.full_like(exponents, GAMMA)
            discounts.append(torch.pow(base, exponents))

            prefix_with_marker = torch.cat([prefix_tokens, start_marker], dim=0).unsqueeze(0)
            prefix_repeat = prefix_with_marker.repeat(G, 1)
            attention_repeat = torch.ones_like(prefix_repeat)
            with torch.no_grad():
                generated = theta_old_model.generate(
                    prefix_repeat,
                    attention_mask=attention_repeat,
                    max_new_tokens=THOUGHT_MAX_TOKENS,
                    do_sample=True,
                    top_p=THOUGHT_TOP_P,
                    temperature=THOUGHT_TEMPERATURE,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=[tokenizer.eos_token_id, end_marker.item()],
                )

            indices = []
            for g in range(generated.size(0)):
                thought_tokens = generated[g, prefix_with_marker.size(1) :]
                if thought_tokens.size(0) == 0:
                    thought_tokens = end_marker.clone()
                else:
                    end_positions = (thought_tokens == end_marker.item()).nonzero(as_tuple=False)
                    if end_positions.numel() > 0:
                        first_end = end_positions[0, 0]
                        thought_tokens = thought_tokens[: first_end + 1]
                    else:
                        thought_tokens = torch.cat([thought_tokens, end_marker], dim=0)

                thought_tokens = thought_tokens.long().to(DEVICE)
                prefix_and_thought = torch.cat([prefix_tokens, thought_tokens], dim=0)
                reasoned_seq = torch.cat([prefix_and_thought, gold_tokens], dim=0)

                thought_sequences.append(prefix_and_thought)
                thought_prefix_lengths.append(prefix_len)
                thought_token_lengths.append(thought_tokens.size(0))
                reasoned_sequences.append(reasoned_seq)
                reasoned_prefix_lengths.append(prefix_len + thought_tokens.size(0))
                gold_lengths_expanded.append(gold_tokens.size(0))
                indices.append(global_idx)
                global_idx += 1

            thoughts_per_sample.append(indices)

        if not baseline_sequences or not thought_sequences:
            continue

        baseline_logps = batched_token_logprobs(
            ema_model,
            baseline_sequences,
            baseline_prefix_lengths,
            gold_lengths,
            tokenizer.pad_token_id,
            requires_grad=False,
        )

        reasoned_logps = batched_token_logprobs(
            model,
            reasoned_sequences,
            reasoned_prefix_lengths,
            gold_lengths_expanded,
            tokenizer.pad_token_id,
            requires_grad=False,
        )

        old_thought_logps = batched_token_logprobs(
            theta_old_model,
            thought_sequences,
            thought_prefix_lengths,
            thought_token_lengths,
            tokenizer.pad_token_id,
            requires_grad=False,
        )

        new_thought_logps = batched_token_logprobs(
            model,
            thought_sequences,
            thought_prefix_lengths,
            thought_token_lengths,
            tokenizer.pad_token_id,
            requires_grad=True,
        )

        advantages = [None] * len(thought_sequences)
        for sample_idx, indices in enumerate(thoughts_per_sample):
            if not indices:
                continue
            baseline_lp = baseline_logps[sample_idx].float()
            discount = discounts[sample_idx]
            rewards = []
            for idx in indices:
                delta = reasoned_logps[idx].float() - baseline_lp
                reward = torch.dot(discount, delta.float())
                rewards.append(reward)
            rewards_tensor = torch.stack(rewards)
            if rewards_tensor.size(0) > 1:
                scale = rewards_tensor.size(0) / (rewards_tensor.size(0) - 1)
                mean_reward = rewards_tensor.mean()
                adv_values = scale * (rewards_tensor - mean_reward)
            else:
                adv_values = rewards_tensor - rewards_tensor.mean()
            for local, global_index in enumerate(indices):
                advantages[global_index] = adv_values[local]

        loss_terms = []
        for idx, advantage in enumerate(advantages):
            if advantage is None:
                continue
            logp_new = new_thought_logps[idx].float()
            logp_old = old_thought_logps[idx].float().to(logp_new.device)
            log_rhos = logp_new - logp_old
            rhos = torch.exp(log_rhos)
            clipped = torch.clamp(rhos, 1.0 - EPS_CLIP_LOW, 1.0 + EPS_CLIP_HIGH)
            adv_detached = advantage.detach().float()
            surrogate = torch.min(rhos * adv_detached, clipped * adv_detached)
            loss_terms.append(-surrogate.mean())

        if not loss_terms:
            continue

        total_loss = torch.stack(loss_terms).mean()
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
