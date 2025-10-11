import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import copy
from tqdm import tqdm
import math

# --------------------------
# Config
# --------------------------
model_name = "Qwen/Qwen3-0.6B-Base"
dataset_name = "allenai/dolma"
max_length = 2048
batch_size = 1            # token-level RLP acts per token; keep small
lr = 1e-5
tau = 0.999               # EMA decay
eps_clip = (0.1, 0.1)     # (ε_l, ε_h)
num_epochs = 1
G = 4                     # number of CoT rollouts per sample
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Load model and EMA teacher
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
ema_model = copy.deepcopy(model).to(device)
for p in ema_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# --------------------------
# Dataset
# --------------------------
ds = load_dataset(dataset_name, split="train[:1%]")  # use subset for demo
def tokenize_fn(ex):
    return tokenizer(ex["text"], truncation=True, max_length=max_length, return_tensors="pt")
ds = ds.map(tokenize_fn, batched=True)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# --------------------------
# Utility functions
# --------------------------
@torch.no_grad()
def compute_logp(model, input_ids, target_ids):
    """Compute log-probability of target_ids given input_ids (teacher forcing)."""
    out = model(input_ids=input_ids, labels=target_ids)
    logp = -out.loss * target_ids.numel()
    return logp / target_ids.numel()

def ema_update(ema_model, model, tau=0.999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(tau).add_(p.data, alpha=1 - tau)

# --------------------------
# RLP Training Loop
# --------------------------
model.train()
for epoch in range(num_epochs):
    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        input_ids = batch["input_ids"][0].to(device)
        # teacher forcing target: shift right
        target_ids = input_ids.clone()

        # --- Step 1: sample G thoughts for each prefix
        # for simplicity, we prepend "<think>" and let model generate a short reasoning string
        thoughts = []
        for _ in range(G):
            prefix = torch.cat([tokenizer("<think>", return_tensors="pt").input_ids.to(device), input_ids], dim=-1)
            thought = model.generate(prefix, max_new_tokens=32, do_sample=True, top_p=0.95)
            thoughts.append(thought[:, :-1])  # exclude EOS

        # --- Step 2: compute log-likelihoods
        with torch.no_grad():
            s_ema = compute_logp(ema_model, input_ids, target_ids)
        s_preds, rewards = [], []
        for c in thoughts:
            s_pred = compute_logp(model, torch.cat([input_ids, c], dim=-1), target_ids)
            s_preds.append(s_pred)
            rewards.append((s_pred - s_ema).detach())

        # --- Step 3: group-relative advantages (Eq. 7)
        r_mean = torch.stack(rewards).mean()
        advantages = [G / (G - 1) * (r - r_mean) for r in rewards]

        # --- Step 4: PPO-style clipped loss on thought tokens
        losses = []
        for i, c in enumerate(thoughts):
            logp_new = model(input_ids=c).logits.log_softmax(-1)
            with torch.no_grad():
                logp_old = model(input_ids=c).logits.log_softmax(-1)
            rho = (logp_new - logp_old).exp()
            clip_rho = torch.clamp(rho, 1 - eps_clip[0], 1 + eps_clip[1])
            losses.append(-torch.min(rho * advantages[i], clip_rho * advantages[i]).mean())
        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_update(ema_model, model, tau)

        tqdm.write(f"Loss {loss.item():.4f}")

# --------------------------
# Save final model
# --------------------------
model.save_pretrained("qwen3-0.6B-rlp")
tokenizer.save_pretrained("qwen3-0.6B-rlp")
