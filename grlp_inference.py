import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THINK_START = "<think>"
THINK_END = "</think>"


def load_model(model_path: str):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def extract_reasoning_and_answer(new_tokens, tokenizer):
    start_id = tokenizer.convert_tokens_to_ids(THINK_START)
    end_id = tokenizer.convert_tokens_to_ids(THINK_END)

    try:
        start_idx = new_tokens.index(start_id)
    except ValueError:
        start_idx = None

    try:
        end_idx = new_tokens.index(end_id)
    except ValueError:
        end_idx = None

    reasoning_text = None
    answer_tokens = []

    if start_idx is not None and end_idx is not None and end_idx > start_idx:
        reasoning_ids = new_tokens[start_idx : end_idx + 1]
        reasoning_text = tokenizer.decode(reasoning_ids, skip_special_tokens=False)
        answer_tokens = new_tokens[end_idx + 1 : end_idx + 1 + 16]
    else:
        answer_tokens = new_tokens[:16]

    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return reasoning_text, answer_text


def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=[tokenizer.convert_tokens_to_ids(THINK_END), tokenizer.eos_token_id],
            pad_token_id=tokenizer.pad_token_id,
        )

    new_token_ids = output[0, inputs["input_ids"].size(1) :].tolist()
    return extract_reasoning_and_answer(new_token_ids, tokenizer)


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRLP model inference with reasoning output")
    parser.add_argument("prompt", help="Input prompt for the model")
    parser.add_argument(
        "--model-path",
        default="qwen3-0.6B-rlp-epoch0",
        help="Path to the fine-tuned model checkpoint directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model, tokenizer = load_model(args.model_path)
    model.to(device)
    model.eval()

    reasoning, answer = generate(
        model,
        tokenizer,
        args.prompt,
        device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    if reasoning:
        print("Reasoning:")
        print(reasoning)
    else:
        print("Reasoning: <none>")

    print("\nAnswer (16 tokens):")
    print(answer.strip())


if __name__ == "__main__":
    main()
