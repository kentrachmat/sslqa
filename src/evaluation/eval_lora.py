#!/usr/bin/env python3
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import re
import time
import pandas as pd
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

DATASET = args.dataset

CSV_PATH = f"/home/brachmat/phd/datasets/{DATASET}_final/test.csv" 
OUT_DIR  = f"results/{DATASET}_eval"
BATCH_SIZE = 64
MAX_NEW_TOKENS = 64
MAX_INPUT_TOKENS = 3500
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False

# Base models
BASE_MODELS = {
    "qwen":  "/home/brachmat/phd/models/Qwen2.5-7B-Instruct",
    "llama": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline",
}

if DATASET == 'squad_v2':
    LORA_ROOTS = [
        Path("runs_unsloth_ddp_clean_squad"),
    ]
    
    SYS_PROMPT = (
        "You are a careful assistant for extractive question answering. Answer in English only. "
        "Answer using only the given context. If the answer is not present, reply exactly: 'unanswerable'."
    )
else:
    LORA_ROOTS = [
        Path("runs_unsloth_ddp_clean_pubmed"),
    ]

    SYS_PROMPT = (
        "You are a careful assistant for question answering. Answer in English only. "
        "You must answer using only the provided context. "
        "state your answer clearly as 'yes', 'no', or 'unanswerable'. "
        # "If your answer is 'yes' or 'no', you must then provide a brief reason for your answer, based *strictly* on the context. "
        # "If the answer cannot be derived from the context, reply *exactly* 'unanswerable' and do not provide a reason."
    )

USER_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

os.makedirs(OUT_DIR, exist_ok=True)

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", name)

def chunk(it, size):
    for i in range(0, len(it), size):
        yield it[i:i+size]

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed_cols = {"id", "question", "context", "answer", "split"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df

def make_messages(sys_prompt: str, context: str, question: str):
    user = USER_TEMPLATE.format(context=context, question=question)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]

def ensure_padding(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

def run_model(exp_name: str, base_key: str, base_path: str, lora_path: Path, df: pd.DataFrame):
    print(f"\n=== Loading {exp_name} ({base_key}) ===")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    ensure_padding(tokenizer)

    # Load base model + LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    messages_list = [make_messages(SYS_PROMPT, c, q) for c, q in zip(df["context"], df["question"])]

    rendered_inputs = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,
    )

    preds = []
    ids = df["id"].tolist()
    golds = df["answer"].tolist()

    with torch.no_grad():
        for idxs in tqdm(list(chunk(list(range(len(rendered_inputs))), BATCH_SIZE)), desc=f"Inferring {exp_name}"):
            texts = [rendered_inputs[i] for i in idxs]
            tok = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )
            tok = {k: v.to(model.device) for k, v in tok.items()}

            gen = model.generate(
                **tok,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

            out_ids = []
            for i, g in enumerate(gen):
                prompt_len = tok["input_ids"][i].size(0)
                out_ids.append(g[prompt_len:])

            batch_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            preds.extend([t.strip() for t in batch_text])

    dt = time.time() - t0
    print(f"{exp_name} done in {dt/60:.1f} min")

    # Build result DF
    out_df = pd.DataFrame({
        "id": ids,
        "pred_answer": preds,
        "gold_answer": golds,
        "model": exp_name,
    })

    # Free CUDA memory
    del model
    del base_model
    torch.cuda.empty_cache()

    return out_df

def main():
    df = load_csv(CSV_PATH)
    all_results = []

    for root in LORA_ROOTS:
        print(f"Scanning {root} ...")
        for run_dir in sorted(root.iterdir()):
            if "13187" not in run_dir.name and DATASET == "pubmed":
                continue
            if run_dir.is_dir():
                exp_name = run_dir.name

                if "qwen" in exp_name.lower():
                    base_key = "qwen"
                elif "llama" in exp_name.lower():
                    base_key = "llama"
                else:
                    print(f"Skipping {exp_name} (unknown base model)")
                    continue
                
                res_df = run_model(exp_name, base_key, BASE_MODELS[base_key], run_dir, df)
                
                out_path_ind = Path(OUT_DIR) / f"{sanitize(exp_name)}_results.csv"
                res_df.to_csv(out_path_ind, index=False)
                print(f"Saved partial results for {exp_name}: {out_path_ind} ({len(res_df)} rows)")

                all_results.append(res_df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        out_path = Path(OUT_DIR) / f"results.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\nSaved combined results: {out_path} ({len(final_df)} rows)")

if __name__ == "__main__":
    main()
