#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fine-tuning script for SSL Augmentation experiments.
Refactored to use centralized configuration system.
"""

import os, time, json
from datetime import datetime, timezone
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import torch
from datasets import Dataset

# Import configuration system
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import get_config
from src.utils.paths import ensure_dir

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except Exception:
    HAS_CODECARBON = False
ENABLE_CARBON = True
CARBON_MEASURE_SECS = 1.0

# ---- Unsloth / HF / TRL ----
os.environ.setdefault("UNSLOTH_DISABLE_FUSED_LOSS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback

# ---------------- CLI ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--model", type=str, default="qwen", choices=["qwen","llama"])
parser.add_argument("--run_id", type=str, default=None)
parser.add_argument("-N", "--num_samples", type=int, default=3625)
parser.add_argument("--cfg", type=str,
                    choices=["baseline", "semantic", "syntactic", "lexical", "all"],
                    default="baseline")
args = parser.parse_args()

RUN_ID = os.environ.get("RUN_ID") or args.run_id or time.strftime("%Y%m%d_%H%M%S")

# ---------------- Configuration ----------------
# Load centralized configuration
config = get_config()

# Get dataset paths from config
DS_TRAIN_SQUAD = config.get_dataset_path("squad", "train")
DS_DEV_SQUAD   = config.get_dataset_path("squad", "dev")
AUG_CSV_SQUAD  = config.get_augmented_path("squad", "train.csv")
VALID_CSV_SQUAD = config.get_augmented_path("squad", "validation_annotated.csv")

DS_TRAIN_PUBMED  = config.get_dataset_path("pubmed", "train")
DS_DEV_PUBMED    = config.get_dataset_path("pubmed", "dev")
AUG_CSV_PUBMED   = config.get_augmented_path("pubmed", "train.csv")
VALID_CSV_PUBMED = config.get_augmented_path("pubmed", "validation_annotated.csv")

# Get model paths from config
MODELS = {
    "qwen": config.get_model_path("qwen"),
    "llama": config.get_model_path("llama"),
}

# ---------------- Hyperparameters ----------------
# Load from config
MAX_SEQ_LEN = config.training["max_seq_length"]
EPOCHS = config.training["epochs"]
PER_DEVICE_TRAIN_BS = config.training["per_device_batch_size"]
GRAD_ACCUM = config.training["gradient_accumulation_steps"]
LR = config.training["learning_rate"]
WARMUP_RATIO = config.training["warmup_ratio"]
LOG_STEPS = config.training["logging_steps"]

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

def is_main_process(trainer) -> bool:
    try:  # accelerate-backed inside HF Trainer
        return trainer.accelerator.is_main_process
    except Exception:
        return getattr(trainer, "is_world_process_zero", False)

class EpochTimerCallback(TrainerCallback):
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._epoch_t0 = None
        self.rows = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_t0 = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_t0 is None:
            return
        dur = time.time() - self._epoch_t0
        ep_float = state.epoch if state.epoch is not None else float("nan")
        ep_int = int(round(ep_float)) if isinstance(ep_float, (int, float)) else None
        self.rows.append({"epoch": ep_int, "duration_sec": dur})

    def on_train_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        # DO NOT call trainer.accelerator.wait_for_everyone() here
        if trainer is None or not (getattr(trainer, "is_world_process_zero", False) 
                                or getattr(getattr(trainer, "accelerator", None), "is_main_process", False)):
            return
        if self.rows:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.rows).to_csv(self.run_dir / "epoch_times.csv", index=False)

# Unsloth fused-loss protection
class SafeLossSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        if isinstance(loss, torch.Tensor):
            loss = loss.clone()
        return (loss, outputs) if return_outputs else loss

def to_chat_text(tokenizer, ctx, q, a):
    a = "unanswerable" if (a is None or (isinstance(a, float) and np.isnan(a))) else str(a)
    messages = [
        {"role": "system", "content": "You are a helpful assistant for extractive QA."},
        {"role": "user", "content": f"Answer strictly from the context. If not present, reply exactly: unanswerable.\n\nContext:\n{ctx}\n\nQuestion:\n{q}"},
        {"role": "assistant", "content": a},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

def df_to_sft_dataset(df, tokenizer):
    def _s(x): 
        return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
    texts = [to_chat_text(tokenizer, _s(r["context"]), _s(r["question"]), r.get("answer"))
             for _, r in df.iterrows()]
    return Dataset.from_dict({"text": texts})

# ---------------- Model load ----------------
def load_model_tokenizer(path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=path,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=False,
        dtype=None,
        trust_remote_code=True,
        use_gradient_checkpointing="unsloth",
        device_map=None,          
    )
    # Load LoRA config
    lora_config = config.training["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        use_rslora=False,
        loftq_config=None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = MAX_SEQ_LEN
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    return model, tokenizer

import pandas as pd
import numpy as np

def process_and_sample(
    df, 
    kind, 
    n_samples=None, 
    label_1_ratio=None
): 
    df = df[df['type'] == kind]
    
    required_cols = ['id', 'context', 'question', 'answer']
    if kind in ["SYNTACTIC", "LEXICAL"]:
        required_cols.append('original_answer')
    processed_df = df[required_cols].copy()

    if kind in ["SYNTACTIC", "LEXICAL"]:
        processed_df['answer'] = processed_df['original_answer']
    
    condition = (processed_df['answer'].str.lower() == "unanswerable").fillna(False)

    processed_df['label'] = np.where(condition, 0, 1)
    
    final_cols = ['id', 'context', 'question', 'answer', 'label']
    processed_df = processed_df[final_cols]
    
    if n_samples is None:
        return processed_df
    df_1 = processed_df[processed_df['label'] == 1]
    df_0 = processed_df[processed_df['label'] == 0]
    
    n_available_1 = len(df_1)
    n_available_0 = len(df_0)
    n_available_total = n_available_1 + n_available_0
    
    # Handle edge case of empty data
    if n_available_total == 0:
        return processed_df.iloc[0:0] # Return empty df with correct columns

    # Cap n_samples at the maximum available data
    if n_samples > n_available_total:
        print(f"Warning: Requested n_samples ({n_samples}) is more than available data ({n_available_total}). Returning all {n_available_total} samples.")
        n_samples = n_available_total
        
    # --- Step 3: Determine Sample Counts ---
    
    # Determine the target ratio
    target_ratio_1 = label_1_ratio
    if target_ratio_1 is None:
        # Case 1: Proportional sampling (use the original ratio)
        target_ratio_1 = n_available_1 / n_available_total
    elif not (0.0 <= target_ratio_1 <= 1.0):
        raise ValueError("label_1_ratio must be between 0.0 and 1.0")

    # Case 2: Fixed-ratio sampling (or proportional)
    
    # 1. Calculate desired counts
    n_desired_1 = round(n_samples * target_ratio_1)
    n_desired_0 = n_samples - n_desired_1

    # 2. Check for shortages (how many we're missing from each class)
    shortage_1 = max(0, n_desired_1 - n_available_1)
    shortage_0 = max(0, n_desired_0 - n_available_0)
    
    # 3. Adjust initial counts to what's available
    n_to_sample_1 = n_desired_1 - shortage_1 # This is min(n_desired_1, n_available_1)
    n_to_sample_0 = n_desired_0 - shortage_0 # This is min(n_desired_0, n_available_0)

    # 4. Re-allocate the shortfall to meet n_samples
    #    (If we ran out of 1s, fill the gap with 0s, and vice-versa)
    if shortage_1 > 0: 
        n_to_sample_0 = min(n_available_0, n_to_sample_0 + shortage_1)
    elif shortage_0 > 0:
        n_to_sample_1 = min(n_available_1, n_to_sample_1 + shortage_0)
            
    # --- Step 4: Perform Sampling and Combine ---
    
    # Sample from each group without replacement
    samples_1 = df_1.sample(n=n_to_sample_1, replace=False)
    samples_0 = df_0.sample(n=n_to_sample_0, replace=False)
    
    # Combine the samples
    final_df = pd.concat([samples_1, samples_0])
    
    # Shuffle the final dataframe and reset the index
    final_df = final_df.sample(frac=1).reset_index(drop=True)
    
    return final_df

def main():
    N = args.num_samples  
    n_dev = int(N/10)
    if args.dataset == 'squad':
        base_data = DS_TRAIN_SQUAD
        dev_data = DS_DEV_SQUAD   
        aug_data = AUG_CSV_SQUAD     
        valid_data = VALID_CSV_SQUAD 
    else:
        base_data = DS_TRAIN_PUBMED
        dev_data = DS_DEV_PUBMED   
        aug_data = AUG_CSV_PUBMED     
        valid_data = VALID_CSV_PUBMED 
    
    base_init = pd.read_csv(aug_data)
    valid = pd.read_csv(valid_data)
    valid = valid[valid['valid'] == 1]
    valid_ids = valid["item_id"].unique()
    base_aug = base_init[base_init["aug_id"].isin(valid_ids)].copy()
    dev  = pd.read_csv(dev_data).iloc[:min(len(pd.read_csv(dev_data)), n_dev)]
    base = pd.read_csv(base_data).iloc[:N]

    TRAIN_BUILDERS = {
        "baseline":  lambda: base,
        "semantic":  lambda: process_and_sample(base_aug, "SEMANTIC", N, 0.8),
        "syntactic": lambda: process_and_sample(base_aug, "SYNTACTIC", N, 0.8),
        "lexical":   lambda: process_and_sample(base_aug, "LEXICAL", N, 0.8),
        "all":       lambda: pd.concat([
                            base.iloc[:int(N/4)],
                            process_and_sample( base_aug, "SEMANTIC", N, 0.8).iloc[:int(N/4)],
                            process_and_sample(base_aug, "SYNTACTIC", N, 0.8).iloc[:int(N/4)],
                            process_and_sample(base_aug, "LEXICAL", N, 0.8).iloc[:int(N/4)],
                        ], ignore_index=True),
    } 
   
    train_df = TRAIN_BUILDERS[args.cfg]()
    dev_df   = dev   
    
    model_key = args.model
    run_name = f"{RUN_ID}_{model_key}_{args.cfg}_N{N}_gpus{WORLD_SIZE}"

    # Use config to get output directory
    run_dir = config.get_training_run_dir(
        dataset=args.dataset,
        model=model_key,
        config=args.cfg,
        num_samples=N,
        run_id=RUN_ID
    )
    if LOCAL_RANK == 0:
        ensure_dir(run_dir)

    model_path = MODELS[model_key]
    model, tokenizer = load_model_tokenizer(model_path)

    train_ds = df_to_sft_dataset(train_df, tokenizer)
    dev_ds   = df_to_sft_dataset(dev_df, tokenizer)

    def _tok(ex):
        return tokenizer(ex["text"], padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    train_tok = train_ds.map(_tok, batched=True, remove_columns=["text"])
    dev_tok   = dev_ds.map(_tok,   batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # CodeCarbon on rank 0 only
    tracker = None
    if ENABLE_CARBON and HAS_CODECARBON and (LOCAL_RANK == 0):
        tracker = EmissionsTracker(
            project_name=f"{Path(model_path).name}",
            output_dir=str(run_dir),
            output_file=f"emissions_{run_name}.csv",
            measure_power_secs=CARBON_MEASURE_SECS,
            log_level="error",
            save_to_file=True,
            gpu_ids=None,
            experiment_id=str(RUN_ID), 
        )
        tracker.start()

    t0 = time.time()
    started_at = datetime.now(timezone.utc).isoformat()

    args_tr = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        per_device_eval_batch_size=max(1, PER_DEVICE_TRAIN_BS * 2),
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=0.8,
        optim="adamw_torch_fused",

        # Precision
        bf16=torch.cuda.is_available(),
        tf32=True,

        # Logging/eval/save
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_on_each_node=False,

        # DDP
        ddp_backend="nccl",
        ddp_find_unused_parameters=False,
        ddp_timeout=3600,
        
        dataloader_num_workers=2,

        # Misc
        remove_unused_columns=False,
        group_by_length=True,
        seed=42,
        run_name=run_name,
        save_safetensors=True,
    )

    trainer = SafeLossSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_tok,
        eval_dataset=dev_tok,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        args=args_tr,
        data_collator=collator,
    )

    trainer.add_callback(EpochTimerCallback(run_dir))
    trainer.train()

    ended_at = datetime.now(timezone.utc).isoformat()
    runtime_sec = time.time() - t0

    # --------- ARTIFACTS (rank-0 only) ---------
    if is_main_process(trainer):
        emissions_kg = None
        if tracker is not None:
            try:
                emissions_kg = tracker.stop()
            except Exception:
                emissions_kg = None

        # Save weights & tokenizer
        trainer.save_model()
        tokenizer.save_pretrained(run_dir)

        # Logs
        history = pd.DataFrame(trainer.state.log_history)
        history.to_csv(run_dir / "log_history.csv", index=False)

        compact = []
        cur_epoch = None
        last_train_loss = None
        for row in trainer.state.log_history:
            if "epoch" in row:
                cur_epoch = int(round(row["epoch"]))
            if "loss" in row:
                last_train_loss = row["loss"]
            if "eval_loss" in row:
                compact.append({
                    "epoch": cur_epoch,
                    "train_loss": last_train_loss,
                    "eval_loss": row["eval_loss"],
                })
        if compact:
            pd.DataFrame(compact).to_csv(run_dir / "epoch_losses.csv", index=False)

        meta = {
            "run_dir": str(run_dir),
            "model_path": model_path,
            "model_key": model_key,
            "world_size": WORLD_SIZE,
            "local_rank": LOCAL_RANK,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "runtime_sec": runtime_sec,
            "epochs": EPOCHS,
            "batch_per_device": PER_DEVICE_TRAIN_BS,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": PER_DEVICE_TRAIN_BS * WORLD_SIZE * GRAD_ACCUM,
            "learning_rate": LR,
            "max_seq_len": MAX_SEQ_LEN,
            "enable_carbon": ENABLE_CARBON and HAS_CODECARBON,
            "carbon_measure_secs": CARBON_MEASURE_SECS,
            "emissions_kg": emissions_kg,
        }
        with open(run_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        pd.DataFrame([meta]).to_csv(run_dir / "run_meta.csv", index=False)

if __name__ == "__main__":
    main()
