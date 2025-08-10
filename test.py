from dpo_train import DPOTrainer, DPODataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np

import random


SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device="cuda" if torch.cuda.is_available() else "cpu"

def prepare_data(max_examples=500):
    dataset_ts = load_dataset("Anthropic/hh-rlhf", split="test")
    subset = dataset_ts.select(range(min(max_examples, len(dataset_ts))))
    data = subset.select(range(max_examples))
    return data

def main(ckpt_dir="checkpoints/checkpoint_beta0.3"):
    model_name = "gpt2"
    max_length = 256
    test_batch_size = 16
    beta = 0.3
    max_examples = 500

    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data
    test_data = prepare_data(max_examples)
    test_dataset = DPODataset(test_data, tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

    # define 2 models
    model = GPT2LMHeadModel.from_pretrained(ckpt_dir)
    ref_model = GPT2LMHeadModel.from_pretrained(model_name)

    trainer = DPOTrainer(model, ref_model, tokenizer, beta=beta, max_length=max_length)

    eval_metrics = trainer.evaluate(test_loader)
    print(f"Test metrics: {eval_metrics}")
