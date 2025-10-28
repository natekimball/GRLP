from __future__ import annotations

import random
import statistics
from copy import deepcopy
from typing import Optional, Dict, Any, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"


DTYPE = "bfloat16"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LR = 1e-6
BATCH_SIZE_PROMPTS = 8
MAX_CONTEXT_LEN = 1024

NUM_ROLLOUTS = 16
TEMPERATURE = 0.7
TOP_P = 0.95
THOUGHT_MAX_NEW_TOKENS = 64
MIN_NEW_TOKENS = 1

CLIP_EPS_LOW = 0.2
CLIP_EPS_HIGH = 0.2
GRAD_CLIP_NORM = 1.0

MAX_STEPS = 100
GRAD_ACCUM_STEPS = 1
LOG_EVERY = 10

EMA_TAU = 0.999
EMA_LAZY_INIT = True
EMA_TEACHER: Optional[torch.nn.Module] = None


DATASET_PATH = "HuggingFaceFW/fineweb"
DATASET_NAME = None
SPLIT = "train"
STREAMING = True
N_SAMPLES = 1024
TEXT_COLUMN = "text"


K_POSITIONS = 4
POSITION_STRATEGY = "surprising"
POSITION_STRIDE = 64

REWARD_PLOT_PATH = "reward_metrics.png"
LOSS_PLOT_PATH = "loss_metrics.png"
THOUGHT_PLOT_PATH = "thought_length_metrics.png"

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from contextlib import contextmanager

@contextmanager
def set_use_cache(m, flag: bool):
    prev = getattr(m.config, "use_cache", False)
    m.config.use_cache = flag
    try:
        yield
    finally:
        m.config.use_cache = prev


def _save_reward_plot(steps: List[int], means: List[float], stds: List[float]) -> None:
    if not steps:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        if not getattr(_save_reward_plot, "_warned", False):
            print("matplotlib is required to plot reward metrics; skipping plot.")
            _save_reward_plot._warned = True  # type: ignore[attr-defined]
        return

    lower = [m - s for m, s in zip(means, stds)]
    upper = [m + s for m, s in zip(means, stds)]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, marker="o", color="tab:blue", label="Mean Reward")
    plt.fill_between(steps, lower, upper, color="tab:blue", alpha=0.2, label="±1 Std Dev")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.title("Reward Statistics Over Training")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REWARD_PLOT_PATH)
    plt.close()


def _save_loss_plot(steps: List[int], losses: List[float]) -> None:
    if not steps:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        if not getattr(_save_loss_plot, "_warned", False):
            print("matplotlib is required to plot loss metrics; skipping plot.")
            _save_loss_plot._warned = True  # type: ignore[attr-defined]
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker="o", color="tab:red")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Average Loss Over Training")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()


def _save_thought_plot(steps: List[int], means: List[float]) -> None:
    if not steps:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError:
        if not getattr(_save_thought_plot, "_warned", False):
            print("matplotlib is required to plot thought length metrics; skipping plot.")
            _save_thought_plot._warned = True  # type: ignore[attr-defined]
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps, means, marker="o", color="tab:orange")
    plt.xlabel("Training Step")
    plt.ylabel("Mean Thought Length")
    plt.title("Mean Chain-of-Thought Length Over Training")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(THOUGHT_PLOT_PATH)
    plt.close()

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if DTYPE == "bfloat16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.to(DEVICE)
    model.train()
    return model, tok


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


def make_behavior_snapshot(student: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        snap = deepcopy(student)
        snap.eval()
        if hasattr(snap, "config"):
            snap.config.use_cache = True
        for p in snap.parameters():
            p.requires_grad_(False)
        return snap



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
def _ema_next_token_logprobs_all(ema_model, ids: torch.Tensor, mask: torch.Tensor) -> List[float]:
    device = next(ema_model.parameters()).device
    valid_len = int(mask.sum().item())
    if valid_len < 2:
        return []
    seq = ids[:valid_len].unsqueeze(0).to(device)
    was_training = ema_model.training
    ema_model.eval()
    with set_use_cache(ema_model, False):
        out = ema_model(input_ids=seq)
        logits = out.logits[:, : valid_len - 1, :]
        targets = seq[:, 1:valid_len]
        logp = torch.log_softmax(logits, dim=-1)
        lp = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    if was_training:
        ema_model.train()
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
    model, input_ids: List[int], next_token_id: int, stable_eval: bool = True
) -> float:
    device = next(model.parameters()).device
    was_training = model.training
    if stable_eval:
        model.eval()
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    # enable cache briefly; tiny sequence here
    with set_use_cache(model, True):
        out = model(input_ids=x)
        logp = torch.log_softmax(out.logits[:, -1, :], dim=-1)
    if stable_eval and was_training:
        model.train()
    return float(logp[0, next_token_id].item())

@torch.no_grad()
def _sample_thought_ids_for_prefix(
    model, tokenizer, prefix_ids: List[int], max_new_tokens: int = THOUGHT_MAX_NEW_TOKENS
) -> List[int]:
    device = next(model.parameters()).device
    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with set_use_cache(model, True):
        was_training = model.training
        model.eval()
        gen = model.generate(
            inp,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=max_new_tokens,
            min_new_tokens=MIN_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if was_training:
            model.train()
    thought = gen[0, inp.shape[1]:].tolist()
    eos = tokenizer.eos_token_id
    if eos in thought:
        thought = thought[: thought.index(eos)]
    if len(thought) == 0:
        with set_use_cache(model, True):
            was_training = model.training
            model.eval()
            gen2 = model.generate(
                inp,
                do_sample=True,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                max_new_tokens=1,
                pad_token_id=eos,
                eos_token_id=None,
            )
            if was_training:
                model.train()
        thought = gen2[0, inp.shape[1]:].tolist() or [inp[0, -1].item()]
    return thought

@torch.no_grad()
def _sample_thought_and_logprobs(
    model,
    tokenizer,
    prefix_ids: List[int],
    max_new_tokens: int = THOUGHT_MAX_NEW_TOKENS,
):
    device = next(model.parameters()).device
    inp = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    with set_use_cache(model, True):
        was_training = model.training
        model.eval()
        out = model.generate(
            inp,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=max_new_tokens,
            min_new_tokens=MIN_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        if was_training:
            model.train()

    full_seq = out.sequences[0]
    gen_ids = full_seq[inp.shape[1]:].tolist()
    scores = out.scores  # list[time] of (batch, vocab)

    old_logprobs: List[float] = []
    for t, logits in enumerate(scores):
        tok_id = int(full_seq[inp.shape[1] + t].item())
        lp = torch.log_softmax(logits, dim=-1)[0, tok_id]
        old_logprobs.append(float(lp.item()))

    eos = tokenizer.eos_token_id
    if eos in gen_ids:
        stop = gen_ids.index(eos)
        gen_ids = gen_ids[:stop]
        old_logprobs = old_logprobs[:stop]

    if len(gen_ids) == 0:
        with set_use_cache(model, True):
            was_training = model.training
            model.eval()
            out2 = model.generate(
                inp,
                do_sample=True,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                max_new_tokens=1,
                pad_token_id=eos,
                eos_token_id=None,
                return_dict_in_generate=True,
                output_scores=True,
            )
            if was_training:
                model.train()

        full_seq2 = out2.sequences[0]
        gen_ids = full_seq2[inp.shape[1]:].tolist()
        logits = out2.scores[0]
        tok_id = int(full_seq2[inp.shape[1]].item())
        lp = torch.log_softmax(logits, dim=-1)[0, tok_id]
        old_logprobs = [float(lp.item())]

    return gen_ids, old_logprobs



def _thought_token_logprobs(model, prefix_ids: List[int], thought_ids: List[int], require_grad: bool) -> torch.Tensor:
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
    A = advantage.detach()
    rho = torch.exp(logp_cur - logp_old)
    rho_clip = torch.clamp(rho, 1.0 - eps_low, 1.0 + eps_high)
    left = rho * A
    right = rho_clip * A
    token_terms = -torch.minimum(left, right)
    return token_terms.mean()
def train_loop(model, tokenizer):
    optimizer = AdamW(model.parameters(), lr=LR)

    global EMA_TEACHER
    reward_steps: List[int] = []
    mean_rewards: List[float] = []
    std_rewards: List[float] = []
    mean_losses: List[float] = []
    mean_thought_lengths: List[float] = []

    for step in tqdm(range(1, MAX_STEPS + 1), desc="Steps"):
        loader = iter_fineweb_first_n(
            N_SAMPLES, tokenizer, BATCH_SIZE_PROMPTS, MAX_CONTEXT_LEN
        )

        step_rewards: List[float] = []
        step_losses: List[float] = []
        step_thought_lengths: List[int] = []

        for batch_idx, batch in enumerate(tqdm(loader, desc="Batches", leave=False)):
            if EMA_LAZY_INIT and EMA_TEACHER is None:
                ema_maybe_init(model)

            device = next(model.parameters()).device
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            num_terms = 0  # how many L_clip_i we actually backprop

            B = input_ids.shape[0]
            G = NUM_ROLLOUTS

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
                    pos_list = _positions_from_mask_stride(msk_b, POSITION_STRIDE)[:K_POSITIONS]
                    ema_lp_all = None
                else:  # "surprising"
                    pos_list = _positions_from_mask_surprising(EMA_TEACHER, ids_b, msk_b, K_POSITIONS)
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
                        if len(full_prefix_ids) > max_prefix else full_prefix_ids
                    )

                    S_ema = S_ema_cache.get(
                        t, _logprob_next_token(EMA_TEACHER, prefix_ids, next_tok)
                    )

                    r_list: List[float] = []
                    thought_list: List[List[int]] = []
                    old_lp_list: List[List[float]] = []

                    for _ in range(G):
                        thought_ids, old_logprobs = _sample_thought_and_logprobs(
                            model,
                            tokenizer,
                            prefix_ids,
                        )
                        S_pred = _logprob_next_token(model, prefix_ids + thought_ids, next_tok)
                        r = S_pred - S_ema
                        r_list.append(r)
                        thought_list.append(thought_ids)
                        old_lp_list.append(old_logprobs)

                    if not r_list:
                        continue

                    step_rewards.extend(r_list)

                    r_tensor = torch.tensor(r_list, dtype=torch.float32, device=device)
                    A_vec = (G / (G - 1.0)) * (r_tensor - r_tensor.mean())

                    for i in range(G):
                        thought_ids = thought_list[i]
                        step_thought_lengths.append(len(thought_ids))
                        logp_cur = _thought_token_logprobs(
                            model, prefix_ids, thought_ids, require_grad=True
                        )
                        logp_old = torch.tensor(
                            old_lp_list[i],
                            dtype=logp_cur.dtype,
                            device=logp_cur.device,
                        )
                        L_clip_i = _clipped_surrogate_term(
                            logp_cur,
                            logp_old,
                            A_vec[i],
                            eps_low=CLIP_EPS_LOW,
                            eps_high=CLIP_EPS_HIGH,
                        )

                        step_losses.append(float(L_clip_i.detach().item()))
                        (L_clip_i / max(1, GRAD_ACCUM_STEPS)).backward()
                        num_terms += 1

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if num_terms > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(max(1, num_terms))

                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                ema_update(model, tau=EMA_TAU)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if step_rewards:
            mean_reward = statistics.mean(step_rewards)
            std_reward = statistics.stdev(step_rewards) if len(step_rewards) > 1 else 0.0
        else:
            mean_reward = float("nan")
            std_reward = float("nan")

        if step_losses:
            mean_loss = statistics.mean(step_losses)
        else:
            mean_loss = float("nan")

        if step_thought_lengths:
            mean_thought_length = statistics.mean(step_thought_lengths)
        else:
            mean_thought_length = float("nan")

        reward_steps.append(step)
        mean_rewards.append(mean_reward)
        std_rewards.append(std_reward)
        mean_losses.append(mean_loss)
        mean_thought_lengths.append(mean_thought_length)

        _save_reward_plot(reward_steps, mean_rewards, std_rewards)
        _save_loss_plot(reward_steps, mean_losses)
        _save_thought_plot(reward_steps, mean_thought_lengths)

        if step % LOG_EVERY == 0:
            tqdm.write(
                f"step={step} N={N_SAMPLES} G={NUM_ROLLOUTS} "
                f"K={K_POSITIONS} strategy={POSITION_STRATEGY} device={DEVICE} "
                f"mean_reward={mean_reward:.6f} std_reward={std_reward:.6f} "
                f"mean_loss={mean_loss:.6f} mean_thought_len={mean_thought_length:.2f}"
            )
if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    print(f"Loaded {MODEL_NAME} on {DEVICE} (dtype={DTYPE})")
    train_loop(model, tokenizer)