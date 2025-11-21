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
import torch.nn.functional as F  # NEW

THINK_BOS_TOKEN = "<think>"
THINK_EOS_TOKEN = "</think>"
THINK_BOS_ID: Optional[int] = None
THINK_EOS_ID: Optional[int] = None


PPO_POLICY_EVAL = "temp_only" 

EPS_MIN_PROB = 1e-12
 
 
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
 
 
DTYPE = "bfloat16"
 
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
LR = 2e-5
BATCH_SIZE_PROMPTS = 8
MAX_CONTEXT_LEN = 2048

CLIPPED_TOKENS = 0
NUM_ROLLOUTS = 4
TEMPERATURE = 0.7
TOP_P = 0.95
THOUGHT_MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 2
 
CLIP_EPS_LOW = 0.2
CLIP_EPS_HIGH = 0.2
GRAD_CLIP_NORM = 1.0
 
MAX_STEPS = 1
GRAD_ACCUM_STEPS = 1
LOG_EVERY = 10
PPO_EPOCHS = 2
 
EMA_TAU = 0.999
EMA_LAZY_INIT = True
EMA_TEACHER: Optional[torch.nn.Module] = None

DEBUG_LOG_PATH = Path("debug.txt")
GLOBAL_STEP = 0
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
POSITION_STRATEGY = "surprising"
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
    data = {
        "step_history": list(step_history),
        "reward_history": list(reward_history),
        "reward_std_history": list(reward_std_history),
        "cot_length_history": list(cot_length_history),
        "loss_history": list(loss_history),
        "global_step": GLOBAL_STEP,
    }
    with METRICS_PICKLE_PATH.open("wb") as fh:
        pickle.dump(data, fh)

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    global THINK_BOS_ID, THINK_EOS_ID
    vocab = tok.get_vocab()

    if THINK_BOS_TOKEN not in vocab or THINK_EOS_TOKEN not in vocab:
        raise ValueError("Thinking tokens not present in tokenizer vocab")

    THINK_BOS_ID = vocab[THINK_BOS_TOKEN]
    THINK_EOS_ID = vocab[THINK_EOS_TOKEN]
    ids_bos = tok.encode(THINK_BOS_TOKEN, add_special_tokens=False)
    ids_eos = tok.encode(THINK_EOS_TOKEN, add_special_tokens=False)
    assert len(ids_bos) == 1 and ids_bos[0] == THINK_BOS_ID, "THINK_BOS_TOKEN not atomic"
    assert len(ids_eos) == 1 and ids_eos[0] == THINK_EOS_ID, "THINK_EOS_TOKEN not atomic"

    dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=dtype,
        attn_implementation="flash_attention_2"
    )
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
 

def _policy_distribution_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    eval_mode: str = "strict",
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    temp = max(1e-6, float(temperature))
    probs = F.softmax(logits / temp, dim=-1)

    if eval_mode == "temp_only" or top_p >= 1.0:
        mask = torch.ones_like(probs, dtype=torch.bool)
        return probs, mask

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(
        cumsum, torch.tensor(top_p, device=probs.device, dtype=probs.dtype), right=False
    )
    cutoff = int(min(cutoff.item(), probs.numel() - 1))

    keep_sorted = torch.zeros_like(sorted_probs, dtype=torch.bool)
    keep_sorted[: cutoff + 1] = True
    mask = torch.zeros_like(probs, dtype=torch.bool)
    mask.scatter_(0, sorted_idx, keep_sorted)

    masked = probs * mask.to(probs.dtype)
    denom = masked.sum()
    probs_norm = masked / (denom + 1e-12)
    return probs_norm, mask

 
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
    logits = out.logits[:, : valid_len - 1, :].float()
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
def _logprob_next_token(
    model, input_ids: List[int], next_token_id: int
) -> float:
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    out = model(input_ids=x)
    logp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    if was_training:
        model.train()
    return float(logp[0, next_token_id].item())
 
 
@torch.no_grad()
def _sample_thought_ids_for_prefix(
    model,
    tokenizer,
    prefix_ids: List[int],
    max_new_tokens: int = THOUGHT_MAX_NEW_TOKENS,
) -> Tuple[List[int], torch.Tensor]:
    assert THINK_BOS_ID is not None and THINK_EOS_ID is not None, "Think tokens not initialized"

    device = next(model.parameters()).device
    was_training = model.training
    if was_training:
        model.eval()

    eval_mode = "temp_only" if PPO_POLICY_EVAL == "temp_only" else "strict"
    top_p_eval = 1.0 if eval_mode == "temp_only" else TOP_P

    ctx = torch.tensor([prefix_ids + [THINK_BOS_ID]], dtype=torch.long, device=device)

    thought_ids: List[int] = []
    logp_list: List[torch.Tensor] = []

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    for step in range(max_new_tokens):
        out = model(input_ids=ctx)
        logits_step = out.logits[0, -1, :]  # [V]

        probs_norm, _ = _policy_distribution_from_logits(
            logits_step, TEMPERATURE, top_p_eval, eval_mode=eval_mode
        )

        changed = False
        if eos_id is not None and eos_id != THINK_EOS_ID and eos_id < probs_norm.numel():
            probs_norm = probs_norm.clone(); probs_norm[eos_id] = 0.0; changed = True
        if pad_id is not None and pad_id != THINK_EOS_ID and pad_id < probs_norm.numel():
            probs_norm = probs_norm.clone(); probs_norm[pad_id] = 0.0; changed = True
        if step < max(MIN_NEW_TOKENS - 1, 0) and THINK_EOS_ID < probs_norm.numel():
            probs_norm = probs_norm.clone(); probs_norm[THINK_EOS_ID] = 0.0; changed = True
        if THINK_BOS_ID < probs_norm.numel():
            probs_norm = probs_norm.clone(); probs_norm[THINK_BOS_ID] = 0.0; changed = True

        if changed:
            s = probs_norm.sum()
            if (not torch.isfinite(s)) or s.item() <= 0:
                probs_norm = torch.softmax(logits_step.float() / max(1e-6, TEMPERATURE), dim=-1)
                if eos_id is not None and eos_id != THINK_EOS_ID and eos_id < probs_norm.numel():
                    probs_norm[eos_id] = 0.0
                if pad_id is not None and pad_id != THINK_EOS_ID and pad_id < probs_norm.numel():
                    probs_norm[pad_id] = 0.0
                if step < max(MIN_NEW_TOKENS - 1, 0) and THINK_EOS_ID < probs_norm.numel():
                    probs_norm[THINK_EOS_ID] = 0.0
                if THINK_BOS_ID < probs_norm.numel():
                    probs_norm[THINK_BOS_ID] = 0.0
            probs_norm = probs_norm / (probs_norm.sum() + 1e-12)

        next_id = int(torch.multinomial(probs_norm, 1).item())
        logp_list.append(torch.log(probs_norm[next_id].clamp_min(EPS_MIN_PROB)))
        thought_ids.append(next_id)

        ctx = torch.cat([ctx, torch.tensor([[next_id]], device=device)], dim=1)

        if next_id == THINK_EOS_ID and step + 1 >= MIN_NEW_TOKENS:
            break

    if len(thought_ids) == 0:
        thought_ids = [THINK_EOS_ID]
        logp_list = [torch.log(torch.tensor(EPS_MIN_PROB, device=device))]

    if was_training:
        model.train()

    logp_old = torch.stack(logp_list).to(torch.float32).detach()
    return thought_ids, logp_old

 
 
def _thought_token_logprobs(
    model, tokenizer, prefix_ids: List[int], thought_ids: List[int], require_grad: bool
) -> torch.Tensor:
    assert THINK_BOS_ID is not None and THINK_EOS_ID is not None
    device = next(model.parameters()).device

    def forward_once(x_ids: List[int]) -> torch.Tensor:
        x = torch.tensor([x_ids], dtype=torch.long, device=device)
        out = model(input_ids=x)
        return out.logits[0, -1, :]

    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if len(eos_id) > 0 else None
    pad_id = tokenizer.pad_token_id
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0] if len(pad_id) > 0 else None

    eval_mode = "temp_only" if PPO_POLICY_EVAL == "temp_only" else "strict"
    top_p_eval = 1.0 if PPO_POLICY_EVAL == "temp_only" else TOP_P

    ctx_ids = list(prefix_ids) + [THINK_BOS_ID]
    logps: List[torch.Tensor] = []

    was_training = model.training
    if was_training:
        model.eval()

    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(require_grad)
    try:
        for tok in thought_ids:
            logits_step = forward_once(ctx_ids)
            probs_norm, _ = _policy_distribution_from_logits(
                logits_step, TEMPERATURE, top_p_eval, eval_mode=eval_mode
            )


            changed = False
            if eos_id is not None and eos_id != THINK_EOS_ID and eos_id < probs_norm.numel():
                probs_norm = probs_norm.clone(); probs_norm[eos_id] = 0.0; changed = True
            if pad_id is not None and pad_id != THINK_EOS_ID and pad_id < probs_norm.numel():
                probs_norm = probs_norm.clone(); probs_norm[pad_id] = 0.0; changed = True

            step_idx = len(ctx_ids) - (len(prefix_ids) + 1)  # tokens already generated in thought
            if step_idx < max(MIN_NEW_TOKENS - 1, 0) and THINK_EOS_ID < probs_norm.numel():
                probs_norm = probs_norm.clone(); probs_norm[THINK_EOS_ID] = 0.0; changed = True

            if THINK_BOS_ID < probs_norm.numel():
                probs_norm = probs_norm.clone(); probs_norm[THINK_BOS_ID] = 0.0; changed = True

            if changed:
                s = probs_norm.sum()
                if not torch.isfinite(s) or s.item() <= 0:
                    probs_norm = F.softmax(logits_step.float() / max(1e-6, TEMPERATURE), dim=-1)
                    if eos_id is not None and eos_id != THINK_EOS_ID and eos_id < probs_norm.numel():
                        probs_norm[eos_id] = 0.0
                    if pad_id is not None and pad_id != THINK_EOS_ID and pad_id < probs_norm.numel():
                        probs_norm[pad_id] = 0.0
                    if step_idx < max(MIN_NEW_TOKENS - 1, 0) and THINK_EOS_ID < probs_norm.numel():
                        probs_norm[THINK_EOS_ID] = 0.0
                    if THINK_BOS_ID < probs_norm.numel():
                        probs_norm[THINK_BOS_ID] = 0.0
                probs_norm = probs_norm / (probs_norm.sum() + 1e-12)

            p_tok = probs_norm[tok] if tok < probs_norm.numel() else torch.tensor(0.0, device=device)
            logps.append(torch.log(p_tok.clamp_min(EPS_MIN_PROB)))
            ctx_ids.append(tok)
    finally:
        torch.set_grad_enabled(prev_grad_state)
        if was_training:
            model.train()

    return torch.stack(logps).to(torch.float32)

def init_think_tokens(model, tok, ref_token="."):
    emb = model.get_input_embeddings().weight
    out = model.get_output_embeddings().weight
    bos_id = tok.convert_tokens_to_ids("<think>")
    eos_id = tok.convert_tokens_to_ids("</think>")
    ref_id = tok.encode(ref_token, add_special_tokens=False)[0]
    with torch.no_grad():
        emb[bos_id].copy_(emb[ref_id])
        emb[eos_id].copy_(emb[ref_id])
        if out is not emb:
            out[bos_id].copy_(out[ref_id])
            out[eos_id].copy_(out[ref_id])
 
def _clipped_surrogate_term(
    logp_cur: torch.Tensor,
    logp_old: torch.Tensor,
    advantage: torch.Tensor,
    eps_low: float,
    eps_high: float,
) -> torch.Tensor:
    global CLIPPED_TOKENS
    A = advantage.detach()
    assert logp_cur.shape == logp_old.shape, "Per-token logp length mismatch"

    rho = torch.exp(logp_cur - logp_old)  # float32
    rho_clip = torch.clamp(rho, 1.0 - eps_low, 1.0 + eps_high)

    with torch.no_grad():
        clipped_mask = (rho < 1.0 - eps_low) | (rho > 1.0 + eps_high)
        CLIPPED_TOKENS += int(clipped_mask.sum().item())

    token_terms = -torch.minimum(rho * A, rho_clip * A)
    return token_terms.mean()
 
def train_loop(model, tokenizer):
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay = 0.0, betas = (0.9, 0.95))

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
 
                max_prefix = max(1, MAX_CONTEXT_LEN - THOUGHT_MAX_NEW_TOKENS - 2)
 
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

                        thought_for_pred = [THINK_BOS_ID] + thought_ids
                        if thought_ids[-1] != THINK_EOS_ID:
                            thought_for_pred = thought_for_pred + [THINK_EOS_ID]

                        
                        S_pred = _logprob_next_token(
                            model, prefix_ids + thought_for_pred, next_tok
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
                            thought_ids=thought_for_pred,
                            prefix_ids=prefix_ids,
                            target_id=next_tok,
                        )
 
                    r_tensor = torch.tensor(
                        r_list, dtype=torch.float32, device=device
                    )

                    mean, std = r_tensor.mean(), r_tensor.std(unbiased=False)
                    A_vec = (r_tensor - mean) / (std + 1e-6)
                    A_vec = (G / (G - 1.0)) * A_vec
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
                loss_values: List[torch.Tensor] = []
                for term in ppo_terms:
                    logp_cur = _thought_token_logprobs(
                        model, tokenizer, term["prefix_ids"], term["thought_ids"], require_grad=True
                    )
                    L_clip_i = _clipped_surrogate_term(
                    logp_cur,
                    term["logp_old"],
                    term["advantage"],
                    eps_low=CLIP_EPS_LOW,
                    eps_high=CLIP_EPS_HIGH,
                    )
                    loss_values.append(L_clip_i)
                optimizer.zero_grad(set_to_none=True)
                loss = torch.stack(loss_values).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

                ema_update(model, tau=EMA_TAU)

                avg_loss = float(loss.detach().cpu())
                GLOBAL_STEP += 1
                step_history.append(GLOBAL_STEP)
                reward_history.append(avg_reward_mean)
                reward_std_history.append(avg_reward_std)
                cot_length_history.append(avg_cot_len)
                loss_history.append(avg_loss)
                save_metric_snapshot()
        if step % LOG_EVERY == 0:
            print(
                f"step={step} N={N_SAMPLES} G={NUM_ROLLOUTS} "
                f"K={K_POSITIONS} strategy={POSITION_STRATEGY} device={DEVICE}"
            )
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    init_think_tokens(model, tokenizer, ref_token=".")

    print(f"Loaded {MODEL_NAME} on {DEVICE} (dtype={DTYPE})")
    train_loop(model, tokenizer)
