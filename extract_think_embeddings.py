import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os

def extract_embeddings(model_name, output_file):
    print(f"Loading tokenizer and model from: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # We only need the embedding layer, but loading the full model is the standard way to ensure we get the correct weights.
        # Using device_map="cpu" to avoid OOM if user doesn't have GPU or if model is large, 
        # though 1.7B should fit on most GPUs. Let's stick to auto or cpu.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype="auto", 
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    tokens_to_extract = ["<think>", "</think>"]
    embeddings_dict = {}

    input_embeddings = model.get_input_embeddings()
    vocab_size = input_embeddings.weight.shape[0]
    print(f"Model vocab size: {vocab_size}")

    for token in tokens_to_extract:
        # Try to find the token ID
        token_id = tokenizer.convert_tokens_to_ids(token)
        
        # Check if it mapped to unk_token (usually means it doesn't exist, unless unk is the token)
        if token_id == tokenizer.unk_token_id and token != tokenizer.unk_token:
            print(f"Warning: Token '{token}' not found in tokenizer (mapped to unk_token_id {tokenizer.unk_token_id}).")
            # Check if it's in added_tokens_decoder
            found = False
            if tokenizer.added_tokens_decoder:
                for id, added_token in tokenizer.added_tokens_decoder.items():
                    if added_token.content == token:
                        token_id = id
                        found = True
                        break
            
            if not found:
                print(f"Skipping {token}")
                continue

        print(f"Found token '{token}' with ID: {token_id}")
        
        if token_id >= vocab_size:
            print(f"Error: Token ID {token_id} is out of bounds for embedding layer with size {vocab_size}")
            continue

        # Extract embedding
        with torch.no_grad():
            # Clone to detach and copy
            token_embedding = input_embeddings.weight[token_id].cpu().clone()
        
        embeddings_dict[token] = token_embedding
        print(f"Extracted embedding for '{token}' with shape {token_embedding.shape}")

    if embeddings_dict:
        print(f"Saving embeddings to {output_file}")
        torch.save(embeddings_dict, output_file)
        print("Success.")
    else:
        print("No embeddings were extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract embeddings for specific tokens from a model.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B", help="Model identifier on HuggingFace")
    parser.add_argument("--output_file", type=str, default="think_embeddings.pt", help="Output file path for the embeddings")
    
    args = parser.parse_args()
    
    extract_embeddings(args.model_name, args.output_file)
