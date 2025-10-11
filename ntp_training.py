import os
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)
from transformers import default_data_collator
from tqdm import tqdm

def main():
    model_name = "Qwen/Qwen3-0.6B-Base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Move model to GPU (or model parallel / distributed setup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optionally enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Load the Dolma dataset (train split)
    ds = load_dataset("allenai/dolma", split="train", streaming=False)  # or streaming if needed

    # Preprocessing / tokenization
    max_length = 2048  # you may choose another context length
    def tokenize_fn(ex):
        # you may want to join text blocks, ensure proper slicing
        return tokenizer(ex["text"], truncation=True, max_length=max_length, return_attention_mask=False)

    tokenized = ds.map(tokenize_fn, remove_columns=ds.column_names, batched=True)

    # Group into blocks (concatenate and chunk)
    block_size = 2048
    def group_texts(examples):
        # Concatenate all texts and chunk into block_size
        concatenated = {k: sum(examples[k], []) for k in examples}
        total_len = len(concatenated["input_ids"])
        # drop the remainder
        total_len = (total_len // block_size) * block_size
        result = {
            "input_ids": [concatenated["input_ids"][i : i + block_size] for i in range(0, total_len, block_size)]
        }
        return result

    lm_dataset = tokenized.map(group_texts, batched=True, batch_size=1000, remove_columns=["input_ids"])

    # DataLoader
    train_loader = DataLoader(lm_dataset, batch_size=4, shuffle=True, collate_fn=default_data_collator)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    global_step = 0
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    for epoch in range(num_epochs):
        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in epoch_iter:
            input_ids = batch["input_ids"].to(device)
            # For causal LM, labels = input_ids
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                loss = loss / 1  # if using gradient accumulation set divisor

            scaler.scale(loss).backward()

            # Gradient step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            global_step += 1
            epoch_iter.set_postfix(loss=loss.item())

            # Save checkpoint periodically
            if global_step % 1000 == 0:
                ckpt_path = os.path.join(save_dir, f"checkpoint-step{global_step}")
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

    # Final save
    model.save_pretrained(os.path.join(save_dir, "final"))
    tokenizer.save_pretrained(os.path.join(save_dir, "final"))


if __name__ == "__main__":
    main()
