#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from openai import OpenAI

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")     
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o")

TEMPERATURE = 0.5
TOP_P       = 1.0
MAX_TOKENS  = 1024
STOP_TOKENS = ["<|eot_id|>", "<|end_of_turn|>"]

# SYS_PROMPT = (
#     "You are a careful assistant for extractive question answering. "
#     "Answer using only the given context. If the answer is not present, reply exactly: 'unanswerable'."
# )


SYS_PROMPT = (
    "You are a careful assistant for question answering. Answer in English only. "
    "You must answer using only the provided context. "
    "First, state your answer clearly as 'yes', 'no', or 'unanswerable'. "
    "If your answer is 'yes' or 'no', you must then provide a brief reason for your answer, based *strictly* on the context. "
    "If the answer cannot be derived from the context, reply *exactly* 'unanswerable' and do not provide a reason."
)

USER_TEMPLATE = (
    "Answer the question strictly based on the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)



def ask_chat(client: OpenAI, question: str, context: str) -> str:
    resp = client.chat.completions.create(
        model=DEPLOYMENT_NAME,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(context=context, question=question)},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop=STOP_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()

def load_data(dataset):
    if dataset == 'squad_v2':
        df = pd.read_csv("../datasets/squad_v2_final/test.csv")
    else:
        df = pd.read_csv("../datasets/pubmed_final/test.csv")
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pubmed", "squad_v2"],
                        help="Choose which dataset loader + output format to use.")
    parser.add_argument("--limit", type=int, default=0, help="Optional quick test limit.")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[ERROR] Set OPENAI_API_KEY in your environment or .env", file=sys.stderr)
        sys.exit(1)

    df = load_data(args.dataset)
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    out_rows = []
    
    for index, ex in df.iterrows():
        try:
            pred = ask_chat(client, question=ex["question"], context=ex["context"])
        except Exception as e:
            print(f"[WARN] API error on {ex['id']}: {e}", file=sys.stderr)
            pred = ""

        out = {
            "id": ex["id"],
            "pred_answer": pred,
            "gold_answer": ex["answer"],
            "model": "gpt4o",
        } 
        out_rows.append(out)

        if index % 25 == 0 or index == len(df):
            print(f"[Progress] {index}/{len(df)}", file=sys.stderr, flush=True)

    out_path = f"results/{args.dataset}_eval/gpt4o.csv"
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()