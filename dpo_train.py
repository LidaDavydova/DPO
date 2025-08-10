import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
import pandas as pd

import os
import random
import gc
from tqdm import tqdm


SEED = 111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset for DPO:
# choosen/rejected ids + mask
class DPODataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_text = item['chosen']
        rejected_text = item['rejected']

        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # remove batch dim 
        return {
            'chosen_input_ids': chosen_tokens['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_tokens['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_tokens['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_tokens['attention_mask'].squeeze(0)
        }


class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, beta=0.5, lr=1e-6, max_length=256):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.beta = beta
        self.max_length = max_length

        # freeze ref model
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = None

    # setup scheduler 
    def setup_scheduler(self, total_steps):
        warmup_steps = int(0.1 * total_steps)  # 10% warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )


    def get_batch_logps(self, model, input_ids, attention_mask, compute_gradients=False):
        """
        Returns avg_logps average token log-prob per example (next-token prediction).
        """
        if not compute_gradients:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous().float()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        picked = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1) 
        picked = picked * shift_mask    

        token_sums = picked.sum(dim=-1)                        
        token_counts = shift_mask.sum(dim=-1).clamp_min(1.0)   
        avg_logps = token_sums / token_counts                  
        return avg_logps

    def compute_dpo_loss(self, pi_chosen, pi_rejected, ref_chosen, ref_rejected):
        pi_diff = pi_chosen - pi_rejected
        ref_diff = ref_chosen - ref_rejected

        logits = self.beta * (pi_diff - ref_diff)
        losses = -F.logsigmoid(logits)
        loss = losses.mean()

        # rewards for logging (detached for safety)
        chosen_reward = (self.beta * (pi_chosen - ref_chosen)).detach().mean()
        rejected_reward = (self.beta * (pi_rejected - ref_rejected)).detach().mean()

        return loss, chosen_reward, rejected_reward

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        chosen_input_ids = batch['chosen_input_ids'].to(device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(device)

        # compute policy logprobs (with grad)
        pi_chosen = self.get_batch_logps(self.model, chosen_input_ids, chosen_attention_mask, compute_gradients=True)
        pi_rejected = self.get_batch_logps(self.model, rejected_input_ids, rejected_attention_mask, compute_gradients=True)

        # compute ref logprobs (no grad)
        ref_chosen = self.get_batch_logps(self.ref_model, chosen_input_ids, chosen_attention_mask, compute_gradients=False)
        ref_rejected = self.get_batch_logps(self.ref_model, rejected_input_ids, rejected_attention_mask, compute_gradients=False)

        # DPO loss
        loss, chosen_reward, rejected_reward = self.compute_dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected)

        # backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            'loss': loss.item(),
            'chosen_reward': chosen_reward.item(),
            'rejected_reward': rejected_reward.item(),
            'reward_margin': (chosen_reward - rejected_reward).item()
        }

    def evaluate(self, eval_dataloader, num_batches=None):
        self.model.eval()
        metrics = {
            'loss': 0.0,
            'chosen_reward': 0.0,
            'rejected_reward': 0.0,
            'reward_margin': 0.0
        }
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if num_batches is not None and i >= num_batches:
                    break

                chosen_input_ids = batch['chosen_input_ids'].to(device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(device)
                rejected_input_ids = batch['rejected_input_ids'].to(device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(device)

                # compute logprobs 
                pi_chosen = self.get_batch_logps(self.model, chosen_input_ids, chosen_attention_mask, compute_gradients=False)
                pi_rejected = self.get_batch_logps(self.model, rejected_input_ids, rejected_attention_mask, compute_gradients=False)
                ref_chosen = self.get_batch_logps(self.ref_model, chosen_input_ids, chosen_attention_mask, compute_gradients=False)
                ref_rejected = self.get_batch_logps(self.ref_model, rejected_input_ids, rejected_attention_mask, compute_gradients=False)

                loss, chosen_reward, rejected_reward = self.compute_dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected)
              
                metrics['loss'] += loss.item()
                metrics['chosen_reward'] += chosen_reward.item()
                metrics['rejected_reward'] += rejected_reward.item()
                metrics['reward_margin'] += (chosen_reward - rejected_reward).item()

                count += 1

        if count > 0:
            for i in metrics:
                metrics[i] /= count
        self.model.train()
        return metrics

def prepare_data(max_examples=3000):
    dataset_tr = load_dataset("Anthropic/hh-rlhf", split="train")
    subset = dataset_tr.select(range(min(max_examples, len(dataset_tr))))
    train_size = int(0.9 * len(subset))
    train_data = subset.select(range(train_size))
    val_data = subset.select(range(train_size, len(subset)))
    return train_data, val_data

def main():
    model_name = "gpt2"
    max_length = 256
    train_batch_size = 16
    eval_batch_size = 16
    lr = 1e-5
    beta = 0.7
    num_epochs = 10
    max_examples = 3000

    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data
    train_data, val_data = prepare_data(max_examples)
    train_dataset = DPODataset(train_data, tokenizer, max_length=max_length)
    val_dataset = DPODataset(val_data, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    # define 2 models
    model = GPT2LMHeadModel.from_pretrained(model_name)
    ref_model = GPT2LMHeadModel.from_pretrained(model_name)

    trainer = DPOTrainer(model, ref_model, tokenizer, beta=beta, lr=lr, max_length=max_length)

    total_steps = len(train_loader) * num_epochs
    trainer.setup_scheduler(total_steps)
    print(f"Train for {num_epochs} epochs, {total_steps} total steps")

    out_dir = f"./dpo_gpt2_model_beta{beta}"
    os.makedirs(f"checkpoints_beta{beta}", exist_ok=True)

    statistic =  {
            'loss': [],
            'chosen_reward': [],
            'rejected_reward': [],
            'reward_margin': []
        }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        stat_epoch = {
            'loss': [],
            'chosen_reward': [],
            'rejected_reward': [],
            'reward_margin': []
        }
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for step, batch in pbar:
            metrics = trainer.train_step(batch)
            for k, v in metrics.items():
                stat_epoch[k] += [v]

            if step % 20 == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'reward_margin': f"{metrics['reward_margin']:.4f}"
                })


            if step % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        avg_loss = float(np.mean(stat_epoch['loss']))
        for k in statistic.keys():
            statistic[k] += [float(np.mean(stat_epoch[k]))]

        print(f"Epoch {epoch+1} avg train loss: {avg_loss:.4f}")

        eval_metrics = trainer.evaluate(val_loader, num_batches=50)
        print(f"Validation metrics: {eval_metrics}")
        
        # save checkpoint
        ckpt = f"checkpoints_beta{beta}/dpo_epoch{epoch+1}"
        os.makedirs(ckpt, exist_ok=True)
        trainer.model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print(f"Saved checkpoint to {ckpt}")

        # final save
        trainer.model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
    print(f"Final model saved to {out_dir}")

    # save statistics
    df = pd.DataFrame(statistic)
    df.to_csv(f'{out_dir}/train_stat.csv', index=False)

