#!/usr/bin/env python3

import os
import re
import time
import pandas as pd
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
args = parser.parse_args()

DATASET = args.dataset

CSV_PATH = f"/home/brachmat/phd/datasets/{DATASET}_final/test.csv" 
OUT_DIR  = f"results/{DATASET}_eval"
BATCH_SIZE = 18
MAX_NEW_TOKENS = 64         
MAX_INPUT_TOKENS = 3500        
TEMPERATURE = 0.0
TOP_P = 1.0
DO_SAMPLE = False

MODELS = {
    "llama-3.1-70b-instruct": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline",
    # "qwen2.5-7b-instruct": "/home/brachmat/phd/models/Qwen2.5-7B-Instruct",
    # "llama-3.1-8b-instruct": "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct-offline",
}

if DATASET == 'pubmed':
    SYS_PROMPT = (
    "You are a careful assistant for question answering. Answer in English only. "
    "You must answer using only the provided context. "
    "First, state your answer clearly as 'yes', 'no', or 'unanswerable'. "
    "If your answer is 'yes' or 'no', you must then provide a brief reason for your answer, based *strictly* on the context. "
    "If the answer cannot be derived from the context, reply *exactly* 'unanswerable' and do not provide a reason."
)
else:
    SYS_PROMPT = (
        "You are a careful assistant for extractive question answering. Answer in English only. "
        "Answer using only the given context. If the answer is not present, reply exactly: 'unanswerable'."
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
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

def run_model(model_key: str, model_path: str, df: pd.DataFrame):
    print(f"\n=== Loading {model_key} from {model_path} ===")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    ensure_padding(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    messages_list = [make_messages(SYS_PROMPT, c, q) for c, q in zip(df["context"], df["question"])]

    # Pre-render chat to text with assistant preamble
    rendered_inputs = tokenizer.apply_chat_template(
        messages_list,
        tokenize=False,
        add_generation_prompt=True,   # adds assistant turn
    )

    preds = []
    ids = df["id"].tolist()
    golds = df["answer"].tolist()
    questions = df["question"].tolist()
    contexts = df["context"].tolist()

    with torch.no_grad():
        for idxs in tqdm(list(chunk(list(range(len(rendered_inputs))), BATCH_SIZE)), desc=f"Inferring {model_key}"):
            texts = [rendered_inputs[i] for i in idxs]

            # Tokenize with truncation to keep prompt within context window
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

            # Strip the prompt part to get only the newly generated text
            # (works because we passed the full prompt in `tok`)
            out_ids = []
            for i, g in enumerate(gen):
                prompt_len = tok["input_ids"][i].size(0)
                out_ids.append(g[prompt_len:])

            batch_text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
            preds.extend([t.strip() for t in batch_text])

    dt = time.time() - t0
    print(f"{model_key} done in {dt/60:.1f} min")

    # Save results
    out_df = pd.DataFrame({
        "id": ids,
        # "question": questions,
        # "context": contexts,
        "pred_answer": preds,
        "gold_answer": golds,
        "model": model_key,
    })

    out_path = Path(OUT_DIR) / f"{DATASET}_{sanitize(model_key)}70b.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(out_df)} rows)")

def main():
    df = load_csv(CSV_PATH)
    for key, path in MODELS.items():
        run_model(key, path, df)

if __name__ == "__main__":
    main()
