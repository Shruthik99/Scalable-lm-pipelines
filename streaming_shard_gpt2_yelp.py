# streaming_shard_gpt2_yelp.py
"""
Multi-Process Streaming Data Pipeline for Language Modeling
Dataset: Yelp Reviews (650K reviews)
Model: GPT-2

Demonstrates:
1. Streaming dataset loading (memory efficient)
2. Manual sharding across multiple processes (parallel processing)
3. Rolling buffer for fixed-length blocks (LM preprocessing)

Run with: python streaming_shard_gpt2_yelp.py
"""

import os
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing as mp
import sys, time
from datetime import datetime

# ============================================================
# Rolling buffer generator for causal LM fixed-length blocks
# ============================================================
def rolling_token_blocks(token_iter, block_size, pad_token_id):
    """
    Takes an iterator of tokenized sequences and yields fixed-length token blocks.
    Uses a rolling buffer to concatenate sequences until a full block is formed.
    
    Args:
        token_iter: Iterator yielding lists of token IDs
        block_size: Target length for each block
        pad_token_id: Token ID to use for padding
    
    Yields:
        Dict with 'input_ids' and 'attention_mask' tensors of length block_size
    """
    buffer = []
    for tokens in token_iter:
        buffer.extend(tokens)
        
        # Yield complete blocks
        while len(buffer) >= block_size:
            chunk = buffer[:block_size]
            buffer = buffer[block_size:]
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.ones(block_size, dtype=torch.long)
            }
    
    # Pad leftover tokens if any remain
    if buffer:
        padded = buffer + [pad_token_id] * (block_size - len(buffer))
        yield {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(
                [1] * len(buffer) + [0] * (block_size - len(buffer)), 
                dtype=torch.long
            )
        }

# ============================================================
# Manual sharding function
# ============================================================
def manual_shard(dataset_iter, num_shards, process_index):
    """
    Manually shard a streaming dataset across multiple processes.
    
    Each process gets every num_shards-th example:
    - Process 0: examples 0, 4, 8, 12...
    - Process 1: examples 1, 5, 9, 13...
    - Process 2: examples 2, 6, 10, 14...
    - Process 3: examples 3, 7, 11, 15...
    
    Args:
        dataset_iter: Streaming dataset iterator
        num_shards: Total number of processes
        process_index: Current process rank (0 to num_shards-1)
    
    Yields:
        Examples belonging to this process's shard
    """
    for idx, example in enumerate(dataset_iter):
        if idx % num_shards == process_index:
            yield example

# ============================================================
# IterableDataset wrapper for LM
# ============================================================
class LMStreamingDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming language modeling data.
    
    Tokenizes text on-the-fly and groups into fixed-length blocks.
    """
    def __init__(self, dataset_iter, tokenizer, block_size):
        self.dataset_iter = dataset_iter
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        # Create token stream by tokenizing each text example
        token_stream = (
            self.tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
            for ex in self.dataset_iter
        )
        # Yield fixed-length blocks from the token stream
        yield from rolling_token_blocks(
            token_stream, 
            self.block_size, 
            self.tokenizer.pad_token_id
        )

# ============================================================
# Collate function
# ============================================================
def collate_fn(batch):
    """
    Collate function for batching.
    Simply stacks tensors since all are already the same length.
    """
    return {
        "input_ids": torch.stack([ex["input_ids"] for ex in batch]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in batch])
    }

# ============================================================
# Worker entry function
# ============================================================
def worker_entry(rank, world_size, model_name, block_size, batch_size, batches_to_show):
    """
    Entry point for each worker process.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        model_name: HuggingFace model name for tokenizer
        block_size: Sequence length for LM training
        batch_size: Number of sequences per batch
        batches_to_show: Number of batches to process before stopping
    """
    # Load streaming dataset
    stream_ds = load_dataset(
        "yelp_review_full",
        split="train", 
        streaming=True
    )
    
    # Shard the dataset for this process
    sharded_iter = manual_shard(stream_ds, world_size, rank)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Create LM dataset
    lm_dataset = LMStreamingDataset(sharded_iter, tokenizer, block_size)
    
    # Create DataLoader
    loader = DataLoader(
        lm_dataset, 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        num_workers=0  # Important: no additional workers in streaming
    )

    print(f"{datetime.now()} [PID {os.getpid()} | rank {rank}] Starting...", flush=True)
    
    # Process batches
    for i, batch in enumerate(loader):
        print(
            f"{datetime.now()} [rank {rank}] batch {i} → {batch['input_ids'].shape}", 
            flush=True
        )
        time.sleep(0.5)  # Slow down so parallel execution is visible
        
        if i + 1 >= batches_to_show:
            break
    
    print(f"{datetime.now()} [rank {rank}] Done.", flush=True)


# ============================================================
# Launcher
# ============================================================
def launch_multi_proc(num_procs, model_name, block_size, batch_size, batches_to_show):
    """
    Launch multiple processes for parallel data processing.
    
    Args:
        num_procs: Number of processes to spawn
        model_name: HuggingFace model name
        block_size: Sequence length
        batch_size: Batch size per process
        batches_to_show: Number of batches each process should show
    """
    ctx = mp.get_context("spawn")  # Safe for all platforms
    procs = []
    
    print(f"\n{'='*60}")
    print(f"Launching {num_procs} processes for parallel data loading")
    print(f"Dataset: Yelp Reviews (650K reviews, streaming)")
    print(f"Model: {model_name}")
    print(f"Block size: {block_size}")
    print(f"Batch size per process: {batch_size}")
    print(f"{'='*60}\n")
    
    # Start all processes
    for rank in range(num_procs):
        p = ctx.Process(
            target=worker_entry, 
            args=(rank, num_procs, model_name, block_size, batch_size, batches_to_show)
        )
        p.start()
        procs.append(p)
    
    # Wait for all processes to complete
    for p in procs:
        p.join()
    
    print(f"\n{'='*60}")
    print(f"All processes completed!")
    print(f"{'='*60}\n")

# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    launch_multi_proc(
        num_procs=4,        # Number of CPU cores to use
        model_name="gpt2",  # GPT-2 tokenizer
        block_size=128,     # Sequence length per sample
        batch_size=4,       # Batch size per process
        batches_to_show=3   # Number of batches to show per process
    )
    
    print("\n💡 Key Observations:")
    print("  1. Each process handles 1/4 of the data (sharding)")
    print("  2. All processes run in parallel (look at timestamps)")
    print("  3. Memory usage is LOW (streaming + sharding)")
    print("  4. This scales to any dataset size!\n")