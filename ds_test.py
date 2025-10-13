from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MAX_DATASET_SAMPLES = 10_000           # cap for cached subset
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
