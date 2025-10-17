import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


model_name = "Qwen/Qwen3-0.6B-Base"
dataset_name = "HuggingFaceFW/fineweb"
split="train"
max_length = 2048
batch_size = 1
lr = 1e-5
tau = 0.999
eps_clip = (0.1, 0.1)
num_epochs = 1
G = 4
thought_max_tokens = 128
temperature = 0.7
top_p = 0.9
device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
special_tokens = {"additional_special_tokens": ["<think>", "</think>"]}
added = tokenizer.add_special_tokens(special_tokens)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
if added > 0:
    model.resize_token_embeddings(len(tokenizer))
ema_model = copy.deepcopy(model).to(device)
for p in ema_model.parameters():
    p.requires_grad = False
model.config.pad_token_id = tokenizer.pad_token_id
ema_model.config.pad_token_id = tokenizer.pad_token_id
ema_model.eval()

start_thought_id = tokenizer("<think>", return_tensors="pt").input_ids.to(device)
end_thought_id = tokenizer("</think>", return_tensors="pt").input_ids.to(device)

def tokenize_fn(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
    )
    return tokens


ds = load_dataset(dataset_name, split=split, streaming=True)
ds = ds.map(tokenize_fn, batched=True, remove_columns=ds.column_names)
loader = DataLoader(ds, batch_size=batch_size)


def ema_update(ema_module, src_module, decay):
    for p_ema, p in zip(ema_module.parameters(), src_module.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def gather_token_logprobs(model_ref, prefix, suffix):
    combined = torch.cat([prefix, suffix], dim=1)
    outputs = model_ref(input_ids=combined)
    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    next_tokens = combined[:, 1:].unsqueeze(-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=next_tokens).squeeze(-1)
    prefix_len = prefix.size(1)
    suffix_len = suffix.size(1)
    start_idx = prefix_len - 1 if prefix_len > 0 else 0
    return token_log_probs[:, start_idx : start_idx + suffix_len]


def next_token_logprob(model_ref, prefix, target_token):
    logp = gather_token_logprobs(model_ref, prefix, target_token)
    return logp.squeeze(0).squeeze(0)


def old_token_logprobs(model_ref, prefix, suffix):
    was_training = model_ref.training
    model_ref.eval()
    with torch.no_grad():
        logp = gather_token_logprobs(model_ref, prefix, suffix)
    if was_training:
        model_ref.train()
    return logp


def old_next_token_logprob(model_ref, prefix, target_token):
    was_training = model_ref.training
    model_ref.eval()
    with torch.no_grad():
        logp = next_token_logprob(model_ref, prefix, target_token)
    if was_training:
        model_ref.train()
    return logp


def sample_thought(model_ref, prefix_tokens):
    was_training = model_ref.training
    model_ref.eval()
    with torch.no_grad():
        generated = model_ref.generate(
            prefix_tokens,
            max_new_tokens=thought_max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if was_training:
        model_ref.train()
    continuation = generated[:, prefix_tokens.size(1) :]
    if continuation.size(1) == 0:
        continuation = torch.tensor([[tokenizer.eos_token_id]], device=device)
    row = continuation[0]
    valid = row[row != tokenizer.pad_token_id]
    if valid.size(0) == 0:
        valid = torch.tensor([tokenizer.eos_token_id], device=device)
    if valid[-1].item() == tokenizer.eos_token_id:
        valid = valid[:-1] if valid.size(0) > 1 else valid
    valid = valid[:thought_max_tokens]
    if valid.size(0) == 0:
        valid = torch.tensor([tokenizer.eos_token_id], device=device)
    return valid.unsqueeze(0)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model.train()


for epoch in range(num_epochs):
    progress = tqdm(loader, desc=f"epoch {epoch}")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        valid_len = attention_mask.sum(dim=1).long()
        if valid_len.min().item() <= 2:
            continue
        prefix_lengths = []
        for length in valid_len:
            prefix_lengths.append(torch.randint(1, length.item() - 1, (1,), device=device).item())
        total_rewards = []
        thought_contexts = []
        thought_tokens = []
        old_logps = []

        for b_idx in range(input_ids.size(0)):
            prefix_end = prefix_lengths[b_idx]
            prefix = input_ids[b_idx : b_idx + 1, :prefix_end]
            target_token = input_ids[b_idx : b_idx + 1, prefix_end : prefix_end + 1]
            thought_prefix = torch.cat([prefix, start_thought_id], dim=1)

            baseline_logp = old_next_token_logprob(ema_model, prefix, target_token)

            rewards = []
            collected_tokens = []
            collected_prefixes = []
            collected_old_logps = []

            for _ in range(G):
                sampled = sample_thought(model, thought_prefix)
                thought = torch.cat([sampled, end_thought_id], dim=1)
                prefix_for_gradient = thought_prefix
                collected_tokens.append(thought)
                collected_prefixes.append(prefix_for_gradient)
                old_logp = old_token_logprobs(model, prefix_for_gradient, thought)
                reasoned_prefix = torch.cat([prefix_for_gradient, thought], dim=1)
                reasoned_logp = old_next_token_logprob(model, reasoned_prefix, target_token)
                reward = (reasoned_logp - baseline_logp).detach()
                collected_old_logps.append(old_logp.detach())
                rewards.append(reward)

            rewards_tensor = torch.stack(rewards)
            mean_reward = rewards_tensor.mean()
            advantages = (G / (G - 1)) * (rewards_tensor - mean_reward)

            for idx in range(G):
                thought_tokens.append(collected_tokens[idx])
                thought_contexts.append(collected_prefixes[idx])
                old_logps.append(collected_old_logps[idx])
                total_rewards.append(advantages[idx].detach())

        if not thought_tokens:
            continue

        policy_losses = []
        for idx, thought in enumerate(thought_tokens):
            prefix_ctx = thought_contexts[idx]
            new_logp = gather_token_logprobs(model, prefix_ctx, thought)
            old_logp = old_logps[idx]
            ratio = (new_logp - old_logp).exp()
            clipped_ratio = torch.clamp(ratio, 1 - eps_clip[0], 1 + eps_clip[1])
            advantage = total_rewards[idx]
            policy_losses.append(-torch.min(ratio * advantage, clipped_ratio * advantage).mean())

        loss = torch.stack(policy_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_update(ema_model, model, tau)

        progress.set_postfix({"loss": loss.item()})


model.save_pretrained("qwen3-0.6B-rlp")
tokenizer.save_pretrained("qwen3-0.6B-rlp")
