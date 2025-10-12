"""
Utility script to evaluate a GRLP-trained causal language model on a suite of
reasoning benchmarks using the EleutherAI lm-eval-harness.

Example usage (requires lm_eval to be installed):

    python evaluate_benchmarks.py \
        --model-path qwen3-0.6B-rlp-epoch0 \
        --tokenizer-path qwen3-0.6B-rlp-epoch0 \
        --output results.json

By default the script evaluates the following tasks:
    - AIME25
    - MATH500
    - GSM8K
    - AMC23
    - Minerva
    - MMLU
    - MMLU-Pro
    - GPQA

For each task we request Pass@1 style metrics from the harness.  The script
prints a compact summary to stdout and writes the full result dictionary to
`--output` (if provided).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator

try:
    from lm_eval.models.base import LM  # Older harness releases
except ModuleNotFoundError:  # lm-eval-harness >= 0.4.0
    from lm_eval.api.model import LM

# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------
# Mapping from display name to lm-eval task identifiers.
# Adjust identifiers if you are using a custom fork of the harness.
DEFAULT_TASKS = {
    # "AIME25": "aime-2025",
    # "MATH500": "math500",
    # "GSM8K": "gsm8k",
    # "AMC23": "amc23",
    # "Minerva": "minerva_math",
    "MMLU": "mmlu",
    # "MMLU-Pro": "mmlu_pro",
    # "GPQA": "gpqa_diamond",
}


@dataclass
class RLPConfig:
    thought_max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    max_generation_tokens: int = 256


class RLPReasoningLM(LM):
    """Custom lm-eval adapter that injects an internal thought before scoring tokens.

    The model samples a chain-of-thought for each next-token prediction using the
    provided fine-tuned checkpoint.  Log-likelihoods and generation operate by
    alternating between thought sampling and token prediction, matching the RLP
    inference procedure.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str | None = None,
        batch_size: int = 1,
        device: str | None = None,
        config: RLPConfig | None = None,
    ) -> None:
        super().__init__()

        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._batch_size = batch_size
        self._cfg = config or RLPConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        added = self.tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self._device)
        if added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        self._start_think_id = self.tokenizer.convert_tokens_to_ids("<think>")
        self._end_think_id = self.tokenizer.convert_tokens_to_ids("</think>")
        if self._start_think_id is None or self._end_think_id is None:
            raise ValueError("Tokenizer is missing <think> or </think> tokens.")

        self._start_think_tensor = torch.tensor(
            [[self._start_think_id]], device=self._device, dtype=torch.long
        )
        self._end_think_tensor = torch.tensor(
            [[self._end_think_id]], device=self._device, dtype=torch.long
        )

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_batch_size(self) -> int:
        return self._batch_size

    @property
    def max_seq_length(self) -> int:
        return getattr(self.model.config, "max_position_embeddings", 2048)

    @property
    def max_gen_toks(self) -> int:
        return self._cfg.max_generation_tokens

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self._device)
        if encoded.size(1) == 0:
            return torch.tensor([[self.eot_token_id]], device=self._device)
        return encoded

    def _sample_thought(self, prefix: torch.Tensor) -> torch.Tensor:
        # Prefix shape: (1, L)
        input_with_marker = torch.cat([prefix, self._start_think_tensor], dim=1)
        with torch.no_grad():
            generated = self.model.generate(
                input_with_marker,
                max_new_tokens=self._cfg.thought_max_tokens,
                do_sample=True,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=[self._end_think_id, self.tokenizer.eos_token_id],
            )
        thought_tokens = generated[:, input_with_marker.size(1) :]
        if thought_tokens.size(1) == 0:
            thought_tokens = self._end_think_tensor
        else:
            # Trim after </think> if present, otherwise append it to terminate.
            end_positions = (thought_tokens == self._end_think_id).nonzero(as_tuple=False)
            if end_positions.numel() > 0:
                first_end = end_positions[0, 1]
                thought_tokens = thought_tokens[:, : first_end + 1]
            else:
                thought_tokens = torch.cat([thought_tokens, self._end_think_tensor], dim=1)
        return thought_tokens

    def _next_token_distribution(self, prefix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        thought = self._sample_thought(prefix)
        combined = torch.cat([prefix, thought], dim=1)
        with torch.no_grad():
            logits = self.model(input_ids=combined).logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        return logprobs, thought

    # ------------------------------------------------------------------
    # LM API implementations
    # ------------------------------------------------------------------
    def _unpack_request(self, request: Any) -> Tuple[str, str]:
        if isinstance(request, tuple):
            return request
        if hasattr(request, "args"):
            return tuple(request.args)  # type: ignore[misc]
        if hasattr(request, "arguments"):
            return tuple(request.arguments)  # type: ignore[misc]
        raise TypeError(f"Unsupported request type: {type(request)!r}")

    def loglikelihood(self, requests: Sequence[Any]):
        outputs = []
        for req in requests:
            context, continuation = self._unpack_request(req)
            ctx_tokens = self._tokenize(context)
            cont_tokens = self._tokenize(continuation)

            prefix = ctx_tokens
            total_logprob = 0.0
            greedy = True
            for token in cont_tokens[0].tolist():
                token_tensor = torch.tensor([[int(token)]], device=self._device, dtype=torch.long)
                thought = self._sample_thought(prefix)
                seq = torch.cat([prefix, thought, token_tensor], dim=1)
                with torch.no_grad():
                    logits = self.model(input_ids=seq).logits
                next_logits = logits[:, -2, :]  # position predicting the final token
                logprob = torch.log_softmax(next_logits, dim=-1)[0, int(token)]
                total_logprob += logprob.item()
                if greedy:
                    greedy = int(token) == torch.argmax(next_logits, dim=-1).item()
                prefix = torch.cat([prefix, token_tensor], dim=1)

            outputs.append((total_logprob, greedy))
        return outputs

    def loglikelihood_rolling(self, requests):
        # Fallback implementation using point-wise loglikelihood.
        outputs = []
        for req in requests:
            if isinstance(req, tuple):
                (text,) = req
            elif hasattr(req, "args"):
                (text,) = req.args
            elif hasattr(req, "arguments"):
                (text,) = req.arguments
            else:
                raise TypeError(f"Unsupported request type: {type(req)!r}")
            tokens = self._tokenize(text)
            prefix = tokens[:, :1]
            rolling = []
            for token in tokens[0, 1:]:
                token_tensor = token.view(1, 1).to(self._device).long()
                thought = self._sample_thought(prefix)
                seq = torch.cat([prefix, thought, token_tensor], dim=1)
                with torch.no_grad():
                    logits = self.model(input_ids=seq).logits
                token_index = int(token_tensor.item())
                logprob = torch.log_softmax(logits[:, -2, :], dim=-1)[0, token_index]
                rolling.append(logprob.item())
                prefix = torch.cat([prefix, token_tensor], dim=1)
            outputs.append(rolling)
        return outputs

    def generate_until(self, requests):
        generations = []
        for req in requests:
            context, until = self._unpack_request(req)
            prefix = self._tokenize(context)
            generated: List[int] = []
            decoded = ""
            for _ in range(self._cfg.max_generation_tokens):
                logprobs, _ = self._next_token_distribution(prefix)
                next_token = torch.argmax(logprobs, dim=-1).item()
                generated.append(int(next_token))
                prefix = torch.cat(
                    [prefix, torch.tensor([[int(next_token)]], device=self._device, dtype=torch.long)], dim=1
                )
                decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
                if any(decoded.endswith(s) for s in until):
                    break
            generations.append(decoded)
        return generations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GRLP model on reasoning benchmarks")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path or identifier of the fine-tuned model weights (passed to lm-eval's hf backend).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer path/identifier. Defaults to --model-path when omitted.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(DEFAULT_TASKS.keys()),
        help="Subset of benchmark names to evaluate. Choices: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Evaluation batch size for lm-eval harness (number of prompts per forward pass).",
    )
    parser.add_argument(
        "--model-type",
        choices=("rlp", "hf"),
        default="rlp",
        help="Selects the evaluation adapter: 'rlp' uses the custom RLPReasoningLM, 'hf' uses the lm-eval Hugging Face backend.",
    )
    parser.add_argument(
        "--fewshot",
        type=Optional[int],
        default=None,
        help="Number of few-shot examples to use. Set to 0 for zero-shot evaluation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path to store the detailed metrics emitted by lm-eval.",
    )
    return parser.parse_args()


def _resolve_tasks(selected: Iterable[str]) -> List[str]:
    resolved = []
    for name in selected:
        if name not in DEFAULT_TASKS:
            raise ValueError(f"Unknown task '{name}'. Valid choices: {sorted(DEFAULT_TASKS)}")
        resolved.append(DEFAULT_TASKS[name])
    return resolved


def main() -> None:
    args = parse_args()

    tokenizer_path = args.tokenizer_path or args.model_path
    lm_eval_tasks = _resolve_tasks(args.tasks)

    eval_kwargs = {
        "tasks": lm_eval_tasks,
        "num_fewshot": args.fewshot,
    }

    if args.model_type == "rlp":
        eval_kwargs["model"] = RLPReasoningLM(
            model_path=args.model_path,
            tokenizer_path=tokenizer_path,
            batch_size=args.batch_size,
        )
    else:
        model_args = [f"pretrained={args.model_path}", f"batch_size={args.batch_size}", "trust_remote_code=True"]
        if tokenizer_path != args.model_path:
            model_args.append(f"tokenizer={tokenizer_path}")
        eval_kwargs["model"] = "hf"
        eval_kwargs["model_args"] = ",".join(model_args)

    results = evaluator.simple_evaluate(**eval_kwargs)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print("\n=== Aggregate Metrics ===")
    for display_name in args.tasks:
        task_id = DEFAULT_TASKS[display_name]
        task_result = results["results"].get(task_id, {})
        if not task_result:
            print(f"{display_name}: no results emitted by lm-eval")
            continue

        # Prefer pass@1 when available, otherwise fall back to accuracy or the first metric.
        metric_name = None
        if "pass@1" in task_result:
            metric_name = "pass@1"
        elif "acc" in task_result:
            metric_name = "acc"
        else:
            metric_name = next(iter(task_result))

        metric_value = task_result[metric_name]
        print(f"{display_name:<12} {metric_value * 100:.2f}% ({metric_name})")

    if "aggregate" in results:
        agg = results["aggregate"]
        macro = agg.get("macro_avg", None)
        if macro is not None:
            print(f"\nMacro average: {macro * 100:.2f}%")


if __name__ == "__main__":
    main()
