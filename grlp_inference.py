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

def rlp_generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    inputs = tokenizer(prompt + THINK_START, return_tensors="pt").to(device)
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

    if output[0, -1].item() == tokenizer.eos_token_id:
        output = output[:, :-1]
    if output[0, -1].item() != tokenizer.convert_tokens_to_ids(THINK_END):
        output = torch.cat(
            [output, torch.tensor([[tokenizer.convert_tokens_to_ids(THINK_END)]], device=device)], dim=1
        )
    
    with torch.no_grad():
        attention_mask = torch.ones_like(output)
        full_output = model.generate(
            input_ids=output,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
    return tokenizer.decode(full_output[0], skip_special_tokens=False)



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

    sequence = rlp_generate(
        model,
        tokenizer,
        args.prompt,
        device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
    )

    print(sequence)


if __name__ == "__main__":
    main()
