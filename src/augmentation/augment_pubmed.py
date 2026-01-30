#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import argparse
import csv
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# -----------------------
# Model / generation config
# -----------------------
MODEL_PATH = "/export/home/cache/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct-offline"

MAX_NEW_TOKENS = 256
MAX_INPUT_TOKENS = 3072
MAX_CONTEXT_TOKENS = 1024

TEMPERATURE = 0.6
TOP_P = 0.9
SEED = 42

BATCH_SIZE = 16                
USE_BNB8 = True
LOAD_IN_4BIT = False
GPU_MEMORY_UTILIZATION = 0.75

# -----------------------
# Prompts
# -----------------------
JSON_SYS_PROMPT = (
    "You are a careful data augmentation engine. "
    "Always return ONLY valid, strict JSON. Do not add explanations, code fences, or extra text. "
)

SEMANTIC_USER_TMPL = """You are given a context and an original question-answer. Create exactly 1 new question-short answer pair that asks about a different facet of the context (not a paraphrase). You need to be creative.

Return the result strictly in JSON with the following structure:
{{
  "questions": ["..."],
  "answers":   ["..."]
}}

----
Context:
{context}

Question:
{question}

Answer:
{answer}
"""

SYNTACTIC_USER_TMPL = """You are given a context and an original question-answer. Rewrite the question so that it asks for the same information but with a different sentence structure (e.g., active/passive, simple/complex, alternate grammar). Keep the meaning identical.

Generate exactly 1 syntactically diverse alternative and its type of structure in 1 word.
Return the result strictly in JSON with the following structure:
{{
  "alternatives": ["..."],
  "type": ["..."]
}}

----
Context:
{context}

Question:
{question}

Answer:
{answer}
"""

LEXICAL_USER_TMPL = """You are given a context and an original question-answer. Rewrite the question so that it asks for the same information but uses different vocabulary and phrasing. Keep the meaning identical.

Generate exactly 1 alternative.
Return the result strictly in JSON with the following structure:
{{
  "alternatives": ["..."]
}}

----
Context:
{context}

Question:
{question}

Answer:
{answer}
"""

# -----------------------
# Helpers: chat wrap, parsing, token truncation
# -----------------------
def apply_chat_template(tokenizer, user_content: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": JSON_SYS_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{JSON_SYS_PROMPT}\n\n{user_content}\n"

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Empty generation.")
    m = JSON_BLOCK_RE.search(text)
    if not m:
        raise ValueError("No JSON object detected.")
    block = m.group(0).strip()
    try:
        return json.loads(block)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}\nBlock: {block[:3000]}")

def ensure_list_len(d: Dict[str, Any], key: str, n: int) -> List[str]:
    val = d.get(key, [])
    if not isinstance(val, list):
        val = [val]
    val = [str(x).strip() for x in val]
    if len(val) < n:
        val += [""] * (n - len(val))
    return val[:n]

def truncate_by_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

# -----------------------
# Data loading
# -----------------------
REQUIRED_COLS = {"id", "context", "question", "answer", "split"}  

def load_df(data_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV from {data_path}: {e}")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    for col in ["id", "context", "question", "answer"]:
        df[col] = df[col].astype(str)
    return df

def to_examples(df: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        out.append({
            "qid": getattr(row, "id"),
            "question": getattr(row, "question"),
            "answer": getattr(row, "answer"),
            "context": getattr(row, "context"),
        })
    return out

# -----------------------
# Build prompts for one example
# -----------------------
def build_user_prompts(tokenizer, context: str, question: str, answer: str) -> Dict[str, str]:
    context_trunc = truncate_by_tokens(tokenizer, context, MAX_CONTEXT_TOKENS)
    sem_user = SEMANTIC_USER_TMPL.format(context=context_trunc, question=question, answer=answer or "")
    syn_user = SYNTACTIC_USER_TMPL.format(context=context_trunc, question=question, answer=answer or "")
    lex_user = LEXICAL_USER_TMPL.format(context=context_trunc, question=question, answer=answer or "")
    return {"semantic": sem_user, "syntactic": syn_user, "lexical": lex_user}

def build_chat_prompts(tokenizer, context: str, question: str, answer: str) -> Dict[str, str]:
    up = build_user_prompts(tokenizer, context, question, answer)
    return {k: apply_chat_template(tokenizer, v) for k, v in up.items()}

# -----------------------
# vLLM batched generate (with micro-batching)
# -----------------------
def vllm_generate_many_in_chunks(
    llm: LLM,
    prompts: List[str],
    seeds: List[Optional[int]],
    *,  # force kwargs
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: Optional[List[str]] = None,
    chunk_size: int = 12,
) -> List[str]:
    """
    Generate one sequence per prompt, in chunks, returning a flat list[str]
    aligned with 'prompts'. Each prompt gets its own SamplingParams (for its seed).
    """
    assert len(prompts) == len(seeds)
    out_texts: List[str] = []
    for i in range(0, len(prompts), chunk_size):
        p_chunk = prompts[i:i+chunk_size]
        s_chunk = seeds[i:i+chunk_size]
        sp_list = [
            SamplingParams(
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or ["<|eot_id|>", "<|end_of_turn|>"],
                seed=s,
            )
            for s in s_chunk
        ]
        outs = llm.generate(p_chunk, sp_list)
        for o in outs:
            if not o.outputs:
                out_texts.append("")
            else:
                out_texts.append(o.outputs[0].text.lstrip())
    return out_texts

# -----------------------
# Flatteners → requested schema
# -----------------------
CSV_FIELDS = [
    "aug_id", "id", "type", "context", "question", "answer", "structure",
    "original_answer", "original_question", "is_valid", "reason"
]

def flatten_semantic(orig_id: str, orig_q: str, orig_a: str, ctx: str,
                     qs: List[str], ans: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for i, (q, a) in enumerate(zip(qs, ans)):
        rows.append({
            "aug_id": f"{orig_id}::SEM::{i}",
            "id": orig_id,
            "type": "SEMANTIC",
            "context": ctx,
            "question": q,
            "answer": a,
            "structure": "",
            "original_answer": orig_a,
            "original_question": orig_q,
            "is_valid": "",
            "reason": "",
        })
    return rows

def flatten_syntactic(orig_id: str, orig_q: str, orig_a: str, ctx: str,
                      alts: List[str], typs: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for i, (q, t) in enumerate(zip(alts, typs)):
        rows.append({
            "aug_id": f"{orig_id}::SYN::{i}",
            "id": orig_id,
            "type": "SYNTACTIC",
            "context": ctx,
            "question": q,
            "answer": "",
            "structure": t,
            "original_answer": orig_a,
            "original_question": orig_q,
            "is_valid": "",
            "reason": "",
        })
    return rows

def flatten_lexical(orig_id: str, orig_q: str, orig_a: str, ctx: str,
                    alts: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for i, q in enumerate(alts):
        rows.append({
            "aug_id": f"{orig_id}::LEX::{i}",
            "id": orig_id,
            "type": "LEXICAL",
            "context": ctx,
            "question": q,
            "answer": "",
            "structure": "",
            "original_answer": orig_a,
            "original_question": orig_q,
            "is_valid": "",
            "reason": "",
        })
    return rows

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pubmed", "squadqa"], required=True,
                        help="Dataset kind (columns must exist but split is ignored).")
    parser.add_argument("--data_path", required=True,
                        help="Path to train CSV file.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, limit number of examples.")
    parser.add_argument("--start", type=int, default=0, help="Starting index for selection.")
    parser.add_argument("--n_calls_per_type", type=int, default=3, help="Separate calls per augmentation type.")
    parser.add_argument("--gen_chunk", type=int, default=12, help="Max prompts per vLLM.generate call (micro-batch).")
    parser.add_argument("--flush_every", type=int, default=50, help="Flush files to disk every N written rows (>=1).")
    parser.add_argument("--fsync", action="store_true", help="Also call os.fsync at each flush (slower, safer).")
    parser.add_argument("--out_jsonl", default="augmented.jsonl")
    parser.add_argument("--out_csv", default="augmented.csv")
    args = parser.parse_args()

    print("[BOOT] starting augmentation run")
    torch.manual_seed(SEED if SEED is not None else 0)

    # Load data
    df = load_df(args.data_path)
    if args.start > 0:
        df = df.iloc[args.start:].reset_index(drop=True)
    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].reset_index(drop=True)
        print(f"[Info] Limiting to {len(df)} examples (start={args.start}).")
    else:
        print(f"[Info] Using {len(df)} examples (start={args.start}).")

    examples = to_examples(df)

    # Tokenizer / vLLM init
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=MODEL_PATH,
        trust_remote_code=True,
        dtype="float16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_INPUT_TOKENS,
        tensor_parallel_size=1,
    )
    if LOAD_IN_4BIT or USE_BNB8:
        llm_kwargs["quantization"] = "bitsandbytes"

    print(f"[BOOT] starting vLLM on CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')} …")
    llm = LLM(**llm_kwargs)

    # Writers & incremental flush
    with open(args.out_jsonl, "w", encoding="utf-8", buffering=1) as f_jsonl, \
         open(args.out_csv,  "w", encoding="utf-8", newline="", buffering=1) as f_csv:

        csv_writer = csv.DictWriter(f_csv, fieldnames=CSV_FIELDS)
        csv_writer.writeheader()

        state = {"written": 0, "flush_every": max(1, args.flush_every)}
        def write_row(row: Dict[str, Any]):
            f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            csv_writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
            state["written"] += 1
            if state["written"] % state["flush_every"] == 0:
                f_jsonl.flush()
                f_csv.flush()
                if args.fsync:
                    os.fsync(f_jsonl.fileno())
                    os.fsync(f_csv.fileno())

        n = len(examples)
        bs = max(1, BATCH_SIZE)
        k = args.n_calls_per_type

        for i0 in range(0, n, bs):
            batch = examples[i0:i0+bs]
            m = len(batch)
            if m == 0:
                continue

            # Pre-build chat prompts per example (once)
            chat_sem = []
            chat_syn = []
            chat_lex = []
            for ex in batch:
                chat = build_chat_prompts(tokenizer, ex["context"], ex["question"], ex.get("answer", ""))
                chat_sem.append(chat["semantic"])
                chat_syn.append(chat["syntactic"])
                chat_lex.append(chat["lexical"])

            # Allocate per-example aggregators
            sem_qs = [[""] * k for _ in range(m)]
            sem_as = [[""] * k for _ in range(m)]
            syn_alts = [[""] * k for _ in range(m)]
            syn_types = [[""] * k for _ in range(m)]
            lex_alts = [[""] * k for _ in range(m)]

            # ---------- BATCh CATEGORY: SEMANTIC ----------
            sem_prompts: List[str] = []
            sem_meta: List[Tuple[int, int]] = []   # (example_idx, call_idx)
            sem_seeds: List[Optional[int]] = []
            for idx in range(m):
                for j in range(k):
                    sem_prompts.append(chat_sem[idx])
                    sem_meta.append((idx, j))
                    # unique-ish seed per ex/call
                    sem_seeds.append((SEED + 10_000*idx + j) if SEED is not None else None)

            sem_texts = vllm_generate_many_in_chunks(
                llm, sem_prompts, sem_seeds,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=None,
                chunk_size=args.gen_chunk,
            )
            for txt, (idx, j) in zip(sem_texts, sem_meta):
                try:
                    parsed = extract_first_json(txt)
                    q = ensure_list_len(parsed, "questions", 1)[0]
                    a = ensure_list_len(parsed, "answers", 1)[0]
                except Exception as e:
                    print(f"[WARN] semantic parse failed (ex={i0+idx} call={j}): {e}", file=sys.stderr)
                    q, a = "", ""
                sem_qs[idx][j] = q
                sem_as[idx][j] = a

            # ---------- BATCh CATEGORY: SYNTACTIC ----------
            syn_prompts: List[str] = []
            syn_meta: List[Tuple[int, int]] = []
            syn_seeds: List[Optional[int]] = []
            for idx in range(m):
                for j in range(k):
                    syn_prompts.append(chat_syn[idx])
                    syn_meta.append((idx, j))
                    syn_seeds.append((SEED + 20_000*idx + j) if SEED is not None else None)

            syn_texts = vllm_generate_many_in_chunks(
                llm, syn_prompts, syn_seeds,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=None,
                chunk_size=args.gen_chunk,
            )
            for txt, (idx, j) in zip(syn_texts, syn_meta):
                try:
                    parsed = extract_first_json(txt)
                    alt = ensure_list_len(parsed, "alternatives", 1)[0]
                    typ = ensure_list_len(parsed, "type", 1)[0]
                except Exception as e:
                    print(f"[WARN] syntactic parse failed (ex={i0+idx} call={j}): {e}", file=sys.stderr)
                    alt, typ = "", ""
                syn_alts[idx][j] = alt
                syn_types[idx][j] = typ

            # ---------- BATCh CATEGORY: LEXICAL ----------
            lex_prompts: List[str] = []
            lex_meta: List[Tuple[int, int]] = []
            lex_seeds: List[Optional[int]] = []
            for idx in range(m):
                for j in range(k):
                    lex_prompts.append(chat_lex[idx])
                    lex_meta.append((idx, j))
                    lex_seeds.append((SEED + 30_000*idx + j) if SEED is not None else None)

            lex_texts = vllm_generate_many_in_chunks(
                llm, lex_prompts, lex_seeds,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stop=None,
                chunk_size=args.gen_chunk,
            )
            for txt, (idx, j) in zip(lex_texts, lex_meta):
                try:
                    parsed = extract_first_json(txt)
                    alt = ensure_list_len(parsed, "alternatives", 1)[0]
                except Exception as e:
                    print(f"[WARN] lexical parse failed (ex={i0+idx} call={j}): {e}", file=sys.stderr)
                    alt = ""
                lex_alts[idx][j] = alt

            # ---------- WRITE RESULTS FOR THIS BATCH ----------
            for b_idx, ex in enumerate(batch):
                pid = ex["qid"]
                q = ex["question"]
                a = ex.get("answer", "")
                ctx = ex["context"]

                # semantic rows
                for r in flatten_semantic(pid, q, a, ctx, sem_qs[b_idx], sem_as[b_idx]):
                    write_row(r)
                # syntactic rows
                for r in flatten_syntactic(pid, q, a, ctx, syn_alts[b_idx], syn_types[b_idx]):
                    write_row(r)
                # lexical rows
                for r in flatten_lexical(pid, q, a, ctx, lex_alts[b_idx]):
                    write_row(r)

            done = min(i0 + bs, n)
            if (i0 // bs) % 5 == 0 or done == n:
                print(f"[Progress] {done}/{n}", file=sys.stderr, flush=True)

    print("✅ Augmentation finished.")
    print(f"JSONL: {os.path.abspath(args.out_jsonl)}")
    print(f"CSV:   {os.path.abspath(args.out_csv)}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
