#!/usr/bin/env python
"""Fine-tune Qwen3-0.6B on HuggingFaceH4/MATH-500 for structured reasoning outputs."""

import argparse
from dataclasses import dataclass
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class SampleKeys:
    problem: str
    solution: str
    answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for reasoning prompts.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B-Base",
        help="Base model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceH4/MATH-500",
        help="Dataset to load from the Hugging Face hub.",
    )
    parser.add_argument(
        "--problem_key",
        type=str,
        default="problem",
        help="Column containing the math problem.",
    )
    parser.add_argument(
        "--solution_key",
        type=str,
        default="solution",
        help="Column containing the worked solution.",
    )
    parser.add_argument(
        "--answer_key",
        type=str,
        default="answer",
        help="Column containing the final short answer.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--validation_split",
        type=str,
        default="validation",
        help="Optional validation split. Ignored if not present.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./qwen3-0.6B-math-sft",
        help="Directory to stash checkpoints and logs.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation to emulate larger batches.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for AdamW.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for the scheduler.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log frequency for the Trainer.",
    )
    parser.add_argument(
        "--eval_strategy",
        type=str,
        default="no",
        help="Evaluation strategy passed to TrainingArguments.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        help="Checkpoint save strategy.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 mixed precision.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    return parser.parse_args()


def build_prompt(problem: str) -> str:
    return f"Q: {problem}\nA: "


def build_target(solution: str, answer: str) -> str:
    return f"<think>{solution}</think>{answer}"


def prepare_keys(dataset_features: Dict[str, str], args: argparse.Namespace) -> SampleKeys:
    if args.problem_key not in dataset_features:
        raise KeyError(f"Problem key '{args.problem_key}' not in dataset: {sorted(dataset_features)}")
    if args.solution_key not in dataset_features:
        raise KeyError(f"Solution key '{args.solution_key}' not in dataset: {sorted(dataset_features)}")
    if args.answer_key not in dataset_features:
        raise KeyError(f"Answer key '{args.answer_key}' not in dataset: {sorted(dataset_features)}")
    return SampleKeys(
        problem=args.problem_key,
        solution=args.solution_key,
        answer=args.answer_key,
    )


def tokenize_function(
    tokenizer: AutoTokenizer,
    sample_keys: SampleKeys,
    max_length: int,
):
    def _tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        input_ids: List[List[int]] = []
        attention_masks: List[List[int]] = []
        labels: List[List[int]] = []
        for problem, solution, answer in zip(
            batch[sample_keys.problem],
            batch[sample_keys.solution],
            batch[sample_keys.answer],
        ):
            prompt = build_prompt(problem)
            target = build_target(solution, answer)
            full_text = prompt + target
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="do_not_pad",
            )
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens["input_ids"])
            full_input_ids = tokenized["input_ids"]
            label_ids = full_input_ids.copy()
            # Mask the prompt portion so loss only applies to the target.
            label_ids[:prompt_len] = [-100] * prompt_len
            input_ids.append(full_input_ids)
            attention_masks.append(tokenized["attention_mask"])
            labels.append(label_ids)
        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

    return _tokenize


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    raw_dataset = load_dataset(args.dataset_name)
    print(raw_dataset)

    sample_keys = prepare_keys(raw_dataset[args.split].column_names, args)

    tokenized_dataset = raw_dataset.map(
        tokenize_function(tokenizer, sample_keys, args.max_seq_length),
        batched=True,
        remove_columns=raw_dataset[args.split].column_names,
    )

    evaluation_dataset = None
    if args.eval_strategy != "no" and args.validation_split in tokenized_dataset:
        evaluation_dataset = tokenized_dataset[args.validation_split]

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset[args.split],
        eval_dataset=evaluation_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
