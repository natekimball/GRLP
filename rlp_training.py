from __future__ import annotations
 
import random
from copy import deepcopy
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pickle

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
 
 
MODEL_NAME = "Qwen/Qwen3-0.6B"
 
 
DTYPE = "bfloat16"
 
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
LR = 1e-6
BATCH_SIZE_PROMPTS = 8
MAX_CONTEXT_LEN = 2048
 
NUM_ROLLOUTS = 4
TEMPERATURE = 0.7
TOP_P = 0.95
THOUGHT_MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 1
 
CLIP_EPS_LOW = 0.2
CLIP_EPS_HIGH = 0.2
GRAD_CLIP_NORM = 1.0
 
MAX_STEPS = 1
GRAD_ACCUM_STEPS = 1
LOG_EVERY = 10
PPO_EPOCHS = 4
 
EMA_TAU = 0.999
EMA_LAZY_INIT = True
EMA_TEACHER: Optional[torch.nn.Module] = None

DEBUG_LOG_PATH = Path("debug.txt")
GLOBAL_STEP = 0
CLIPPED_TOKENS = 0
reward_history: List[float] = []
reward_std_history: List[float] = []
cot_length_history: List[float] = []
loss_history: List[float] = []
step_history: List[int] = []
METRICS_PICKLE_PATH = Path("rlp_metrics.pkl")


DATASET_PATH = "HuggingFaceFW/fineweb"
DATASET_NAME = None
SPLIT = "train"
STREAMING = True
N_SAMPLES = 1024
TEXT_COLUMN = "text"
 
 
K_POSITIONS = 4
POSITION_STRATEGY = "random"
POSITION_STRIDE = 64
 
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True  # noqa
    torch.backends.cudnn.allow_tf32 = True  # noqa


def log_sampled_thought(
    tokenizer,
    step_idx: int,
    position: int,
    rollout_idx: int,
    thought_ids: List[int],
    prefix_ids: List[int],
    target_id: int,
) -> None:
    """Append decoded thought samples to a debug log for inspection."""
    decoded_thought = tokenizer.decode(thought_ids, skip_special_tokens=False)
    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    target_text = tokenizer.decode([target_id], skip_special_tokens=False)
    with DEBUG_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[step={step_idx} t={position} rollout={rollout_idx} len={len(thought_ids)}]\n"
        )
        log_file.write(f"Prefix: {prefix_text}\n")
        log_file.write(f"Target: {target_text}\n")
        log_file.write(repr(decoded_thought) + "\n\n")


def save_metric_snapshot() -> None:
    """Persist metric histories so partial runs can be analyzed later."""
    data = {
        "step_history": list(step_history),
        "reward_history": list(reward_history),
        "reward_std_history": list(reward_std_history),
        "cot_length_history": list(cot_length_history),
        "loss_history": list(loss_history),
        "clipped_tokens": CLIPPED_TOKENS,
        "global_step": GLOBAL_STEP,
    }
    with METRICS_PICKLE_PATH.open("wb") as fh:
        pickle.dump(data, fh)

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
 
    dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
 
    # Load to CPU, then move whole model to DEVICE.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(DEVICE)
    model.train()
    return model, tok
 
 
@torch.no_grad()
def ema_maybe_init(student: torch.nn.Module):
    global EMA_TEACHER
    if EMA_TEACHER is None:
        EMA_TEACHER = deepcopy(student).eval()
        for p in EMA_TEACHER.parameters():
            p.requires_grad_(False)
 
 
@torch.no_grad()
def ema_update(student: torch.nn.Module, tau: float = EMA_TAU):
    global EMA_TEACHER
    if EMA_TEACHER is None:
        ema_maybe_init(student)
    for p_t, p_s in zip(EMA_TEACHER.parameters(), student.parameters()):
        p_t.data.mul_(tau).add_(p_s.data, alpha=1.0 - tau)
    for b_t, b_s in zip(EMA_TEACHER.buffers(), student.buffers()):
        b_t.copy_(b_s)
 
 
def iter_fineweb_first_n(
    n: int,
    tokenizer,
    batch_size: int = BATCH_SIZE_PROMPTS,
    max_len: int = MAX_CONTEXT_LEN,
):
    ds = load_dataset(
        DATASET_PATH, name=DATASET_NAME, split=SPLIT, streaming=STREAMING
    ).take(n)
 
    def get_text(row: Dict[str, Any]):
        if TEXT_COLUMN is not None and TEXT_COLUMN in row:
            return row[TEXT_COLUMN]
        for key in ("text", "content", "raw_content", "document", "body"):
            if key in row:
                return row[key]
        strings = [v for v in row.values() if isinstance(v, str)]
        return max(strings, key=len) if strings else None
 
    def encode(ex: Dict[str, Any]):
        txt = get_text(ex)
        if not txt:
            return {"skip": True}
        enc = tokenizer(
            txt,
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        if tokenizer.eos_token_id is not None and len(ids) == max_len:
            ids[-1] = tokenizer.eos_token_id
        return {"input_ids": ids, "attention_mask": mask}
 
    tokenized = ds.map(encode)
 
    def collate(batch):
        batch = [b for b in batch if "input_ids" in b]
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        attention_mask = torch.tensor(
            [b["attention_mask"] for b in batch], dtype=torch.long
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}
 
    loader = DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return loader
 
def _positions_from_mask_random(mask: torch.Tensor, k: int) -> List[int]:
    valid_len = int(mask.sum().item())
    if valid_len < 2:
        return []
    cand = list(range(1, valid_len))
    return random.sample(cand, min(k, len(cand)))
 
 
def _positions_from_mask_stride(mask: torch.Tensor, stride: int) -> List[int]:
    valid_len = int(mask.sum().item())
    if valid_len < 2:
        return []
    return list(range(1, valid_len, max(1, stride)))
 
 
@torch.no_grad()
@torch.no_grad()
def _ema_next_token_logprobs_all(
    ema_model, ids: torch.Tensor, mask: torch.Tensor
) -> List[float]:
    device = next(ema_model.parameters()).device
    valid_len = int(mask.sum().item())
    if valid_len < 2:
        return []
    seq = ids[:valid_len].unsqueeze(0).to(device)
    was_training = ema_model.training
    ema_model.eval()
    out = ema_model(input_ids=seq)
    if was_training:
        ema_model.train()
    logits = out.logits[:, : valid_len - 1, :]
    targets = seq[:, 1:valid_len]
    logp = torch.log_softmax(logits, dim=-1)
    lp = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return lp[0].tolist()
 
 
def _positions_from_mask_surprising(
    ema_model, ids: torch.Tensor, mask: torch.Tensor, k: int
) -> List[int]:
    lp = _ema_next_token_logprobs_all(ema_model, ids, mask)
    if not lp:
        return []
    lpt = torch.tensor(lp)
    k = min(k, lpt.numel())
    idx = torch.topk(lpt, k=k, largest=False).indices.tolist()
    return [int(u) + 1 for u in idx]
 
@torch.no_grad()
@torch.no_grad()
def _logprob_next_token(
    model, input_ids: List[int], next_token_id: int
) -> float:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(input_ids=x)
    logp = torch.log_softmax(out.logits[:, -1, :], dim=-1)
    if was_training:
        model.train()
    return float(logp[0, next_token_id].item())
 
 
def _sample_thought_ids_for_prefix(
    model,
    tokenizer,
    prefix_ids: List[int],
    max_new_tokens: int = THOUGHT_MAX_NEW_TOKENS,
) -> Tuple[List[int], torch.Tensor]:
    device = next(model.parameters()).device
    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    eos_token_id = tokenizer.eos_token_id

    def _run_generate(
        max_new: int,
        min_new: int,
        eos_override: Optional[int],
    ):
        was_training = model.training
        if was_training:
            model.eval()
        with torch.no_grad():
            output = model.generate(
                inp,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=max_new,
                min_new_tokens=min_new,
                pad_token_id=eos_token_id,
                eos_token_id=eos_override,
                return_dict_in_generate=True,
                output_scores=True,
            )
        if was_training:
            model.train()
        return output

    def _tokens_and_logps_from_output(generate_output):
        seq = generate_output.sequences[0, inp.shape[1] :]
        tokens = seq.tolist()
        logps: List[torch.Tensor] = []
        if generate_output.scores:
            for step, tok_id in enumerate(tokens):
                logits_step = generate_output.scores[step][0]
                log_prob_step = torch.log_softmax(logits_step, dim=-1)[tok_id]
                logps.append(log_prob_step)
        return tokens, logps

    output = _run_generate(
        max_new=max_new_tokens,
        min_new=MIN_NEW_TOKENS,
        eos_override=eos_token_id,
    )
    thought_ids, logp_values = _tokens_and_logps_from_output(output)

    if eos_token_id is not None and eos_token_id in thought_ids:
        eos_index = thought_ids.index(eos_token_id)
        thought_ids = thought_ids[:eos_index]
        logp_values = logp_values[:eos_index]

    if not thought_ids:
        output = _run_generate(max_new=1, min_new=1, eos_override=None)
        thought_ids, logp_values = _tokens_and_logps_from_output(output)
        if eos_token_id is not None and eos_token_id in thought_ids:
            eos_index = thought_ids.index(eos_token_id)
            thought_ids = thought_ids[:eos_index]
            logp_values = logp_values[:eos_index]
        if not thought_ids:
            fallback_ids = [int(inp[0, -1].item())]
            was_training = model.training
            if was_training:
                model.eval()
            logp_tensor = _thought_token_logprobs(
                model, prefix_ids, fallback_ids, require_grad=False
            ).detach()
            if was_training:
                model.train()
            return fallback_ids, logp_tensor

    if logp_values:
        logp_tensor = torch.stack(logp_values).detach()
    else:
        logp_tensor = torch.zeros(
            len(thought_ids), dtype=next(model.parameters()).dtype, device=device
        )

    return thought_ids, logp_tensor
 
 
def _thought_token_logprobs(
    model, prefix_ids: List[int], thought_ids: List[int], require_grad: bool
) -> torch.Tensor:
    device = next(model.parameters()).device
    seq = torch.tensor([prefix_ids + thought_ids], dtype=torch.long, device=device)
    prefix_len = len(prefix_ids)
 
    if require_grad:
        out = model(input_ids=seq)
        logits = out.logits[:, prefix_len - 1 : -1, :]
        logp = torch.log_softmax(logits, dim=-1)[0]
        idx = torch.tensor(thought_ids, dtype=torch.long, device=device)
        return logp.gather(1, idx.view(-1, 1)).squeeze(1)
    else:
        with torch.no_grad():
            out = model(input_ids=seq)
            logits = out.logits[:, prefix_len - 1 : -1, :]
            logp = torch.log_softmax(logits, dim=-1)[0]
            idx = torch.tensor(thought_ids, dtype=torch.long, device=device)
            return logp.gather(1, idx.view(-1, 1)).squeeze(1).detach()
 
 
def _clipped_surrogate_term(
    logp_cur: torch.Tensor,
    logp_old: torch.Tensor,
    advantage: torch.Tensor,
    eps_low: float,
    eps_high: float,
) -> torch.Tensor:
    global CLIPPED_TOKENS
    A = advantage.detach()
    rho = torch.exp(logp_cur - logp_old)
    rho_clip = torch.clamp(rho, 1.0 - eps_low, 1.0 + eps_high)
    with torch.no_grad():
        rho_det = rho.detach()
        clipped_mask = (rho_det < 1.0 - eps_low) | (rho_det > 1.0 + eps_high)
        CLIPPED_TOKENS += int(clipped_mask.sum().item())
    left = rho * A
    right = rho_clip * A
    token_terms = -torch.minimum(left, right)
    return token_terms.mean()
 
def train_loop(model, tokenizer):
    optimizer = AdamW(model.parameters(), lr=LR)

    DEBUG_LOG_PATH.write_text("", encoding="utf-8")

    global EMA_TEACHER, GLOBAL_STEP
    loader = iter_fineweb_first_n(
            N_SAMPLES, tokenizer, BATCH_SIZE_PROMPTS, MAX_CONTEXT_LEN
        )
    for step in range(1, MAX_STEPS + 1):
 
        for batch_idx, batch in enumerate(loader):
            if EMA_LAZY_INIT and EMA_TEACHER is None:
                ema_maybe_init(model)

            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            model.train()
            # Collect rollouts and statistics once so we can reuse them over multiple PPO epochs.
            ppo_terms: List[Dict[str, Any]] = []

            B = input_ids.shape[0]
            G = NUM_ROLLOUTS
            batch_reward_means: List[float] = []
            batch_reward_stds: List[float] = []
            batch_cot_lengths: List[float] = []
 
            for b in range(B):
                ids_b = input_ids[b]
                msk_b = attention_mask[b]
                valid_len = int(msk_b.sum().item())
                if valid_len < 2:
                    continue
 
                # Choose positions
                if POSITION_STRATEGY == "random":
                    pos_list = _positions_from_mask_random(msk_b, K_POSITIONS)
                    ema_lp_all = None
                elif POSITION_STRATEGY == "stride":
                    pos_list = _positions_from_mask_stride(msk_b, POSITION_STRIDE)[
                        :K_POSITIONS
                    ]
                    ema_lp_all = None
                else:
                    pos_list = _positions_from_mask_surprising(
                        EMA_TEACHER, ids_b, msk_b, K_POSITIONS
                    )
                    ema_lp_all = (
                        _ema_next_token_logprobs_all(EMA_TEACHER, ids_b, msk_b)
                        if pos_list
                        else None
                    )
 
                if not pos_list:
                    continue
 
                max_prefix = max(1, MAX_CONTEXT_LEN - THOUGHT_MAX_NEW_TOKENS - 1)
 
                S_ema_cache: Dict[int, float] = {}
                if ema_lp_all is not None:
                    for t in pos_list:
                        S_ema_cache[t] = float(ema_lp_all[t - 1])
 
                for t in pos_list:
                    full_prefix_ids = ids_b[:t].tolist()
                    next_tok = int(ids_b[t].item())
 
                    prefix_ids = (
                        full_prefix_ids[-max_prefix:]
                        if len(full_prefix_ids) > max_prefix
                        else full_prefix_ids
                    )
                    
                    if (len(full_prefix_ids) <= max_prefix) and (t in S_ema_cache):
                        S_ema = S_ema_cache[t]
                    else:
                        S_ema = _logprob_next_token(EMA_TEACHER, prefix_ids, next_tok)
 
                    r_list: List[float] = []
                    thought_records: List[Dict[str, Any]] = []

                    for rollout_idx in range(G):
                        thought_ids, logp_old = _sample_thought_ids_for_prefix(
                            model, tokenizer, prefix_ids, THOUGHT_MAX_NEW_TOKENS
                        )
                        S_pred = _logprob_next_token(
                            model, prefix_ids + thought_ids, next_tok
                        )
                        r = S_pred - S_ema
                        r_list.append(r)
                        thought_records.append(
                            {"thought_ids": thought_ids, "logp_old": logp_old}
                        )
                        batch_cot_lengths.append(float(len(thought_ids)))
                        log_sampled_thought(
                            tokenizer,
                            step_idx=step,
                            position=t,
                            rollout_idx=rollout_idx,
                            thought_ids=thought_ids,
                            prefix_ids=prefix_ids,
                            target_id=next_tok,
                        )
 
                    r_tensor = torch.tensor(
                        r_list, dtype=torch.float32, device=device
                    )
                    A_vec = (G / (G - 1.0)) * (r_tensor - r_tensor.mean())
                    batch_reward_means.append(float(r_tensor.mean().item()))
                    batch_reward_stds.append(
                        float(r_tensor.std(unbiased=False).item())
                        if r_tensor.numel() > 1
                        else 0.0
                    )

                    for i, record in enumerate(thought_records):
                        thought_ids = record["thought_ids"]
                        logp_old = record["logp_old"]
                        advantage = A_vec[i].detach()
                        ppo_terms.append(
                            {
                                "prefix_ids": list(prefix_ids),
                                "thought_ids": list(thought_ids),
                                "logp_old": logp_old,
                                "advantage": advantage,
                            }
                        )

            num_terms = len(ppo_terms)
            if num_terms == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            if batch_reward_means:
                avg_reward_mean = float(
                    sum(batch_reward_means) / len(batch_reward_means)
                )
                avg_reward_std = float(
                    sum(batch_reward_stds) / len(batch_reward_stds)
                )
            else:
                avg_reward_mean = 0.0
                avg_reward_std = 0.0
            if batch_cot_lengths:
                avg_cot_len = float(
                    sum(batch_cot_lengths) / len(batch_cot_lengths)
                )
            else:
                avg_cot_len = 0.0
 
            for epoch_idx in range(PPO_EPOCHS):
                optimizer.zero_grad(set_to_none=True)
                loss_values: List[torch.Tensor] = []
                for term in ppo_terms:
                    logp_cur = _thought_token_logprobs(
                        model,
                        term["prefix_ids"],
                        term["thought_ids"],
                        require_grad=True,
                    )
                    L_clip_i = _clipped_surrogate_term(
                        logp_cur,
                        term["logp_old"],
                        term["advantage"],
                        eps_low=CLIP_EPS_LOW,
                        eps_high=CLIP_EPS_HIGH,
                    )
                    loss_values.append(L_clip_i.detach())
                    L_clip_i.backward()

                grad_scale = num_terms * max(1, GRAD_ACCUM_STEPS)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(grad_scale)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                ema_update(model, tau=EMA_TAU)

                if loss_values:
                    avg_loss = float(
                        torch.stack(loss_values).mean().detach().cpu().item()
                    )
                else:
                    avg_loss = 0.0

                GLOBAL_STEP += 1
                step_history.append(GLOBAL_STEP)
                reward_history.append(avg_reward_mean)
                reward_std_history.append(avg_reward_std)
                cot_length_history.append(avg_cot_len)
                loss_history.append(avg_loss)
                print(
                    f"Global step {GLOBAL_STEP}: clipped tokens so far = {CLIPPED_TOKENS}"
                )
                print(
                    len(reward_history),
                    len(reward_std_history),
                    len(cot_length_history),
                    len(loss_history),
                )
                save_metric_snapshot()
 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
 
        if step % LOG_EVERY == 0:
            print(
                f"step={step} N={N_SAMPLES} G={NUM_ROLLOUTS} "
                f"K={K_POSITIONS} strategy={POSITION_STRATEGY} device={DEVICE}"
            )
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    print(f"Loaded {MODEL_NAME} on {DEVICE} (dtype={DTYPE})")
    train_loop(model, tokenizer)
