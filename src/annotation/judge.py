#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dotenv import load_dotenv
load_dotenv()

import os, sys, json, time, argparse, re
import time
import json, re, time, argparse, sys
from statistics import mean


import pandas as pd

from openai import OpenAI

DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "gpt-4o")
BASE_URL   = os.getenv("OPENAI_BASE_URL", "")
API_KEY    = os.getenv("OPENAI_API_KEY", "")

if not API_KEY:
    print("[Error] OPENAI_API_KEY is empty.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=BASE_URL if BASE_URL else None, api_key=API_KEY)

 
PROMPT_SHARED_HEADER = """You are an expert judge evaluating the validity of augmented QA pairs. Decide if a generated QA is VALID (1), NOT SURE (0), or INVALID (-1).

### Rules:
- Answerability: The AUGMENTED_QUESTION must be answerable *only* from the given context OR explicitly unanswerable.
- Ambiguity: If multiple conflicting answers are possible = INVALID (AMBIGUOUS).
- If the given PROVIDED_ANSWER misses or only partially matches required info = INVALID (SPAN_MISSING).
- Coverage: If the AUGMENTED_QUESTION asks about info not in the context = INVALID (OUT_OF_SCOPE).
- Other: Use OTHER for malformed or nonsensical cases.
"""

PROMPT_SEMANTIC = (
    PROMPT_SHARED_HEADER
    + """
### Type-specific notes (Semantic):
- The AUGMENTED_QUESTION should probe a different facet or piece of information in the same context (not merely a paraphrase of ORIGINAL_QUESTION).
- If the AUGMENTED_QUESTION targets content not present in the context, mark INVALID (OUT_OF_SCOPE).

Output strict JSON only:
{
  "valid": "-1/0/1",
  "reason": "SPAN_MISSING|HALLUCINATION|AMBIGUOUS|OUT_OF_SCOPE|DUPLICATE|OTHER|''",
  "notes": "short justification (2 sentences)"
}
Return nothing except this JSON.
""".strip()
)

PROMPT_SYNTACTIC = (
    PROMPT_SHARED_HEADER
    + """
### Type-specific notes (Syntactic):
- The AUGMENTED_QUESTION should preserve the exact meaning of ORIGINAL_QUESTION but vary grammatical structure (e.g., active-passive, clause reordering, simple-complex).
- If the rewording changes the meaning or target, mark INVALID (OUT_OF_SCOPE or AMBIGUOUS as appropriate).

Output strict JSON only:
{
  "valid": "-1/0/1",
  "reason": "SPAN_MISSING|HALLUCINATION|AMBIGUOUS|OUT_OF_SCOPE|DUPLICATE|OTHER|''",
  "notes": "short justification (2 sentences)"
}
Return nothing except this JSON.
""".strip()
)

PROMPT_LEXICAL = (
    PROMPT_SHARED_HEADER
    + """
### Type-specific notes (Lexical):
- The AUGMENTED_QUESTION should preserve the exact meaning but change surface wording (synonyms, phrasing) without altering the information requested.
- If the rephrasing introduces or removes meaning, mark INVALID (AMBIGUOUS or OUT_OF_SCOPE).

Output strict JSON only:
{
  "valid": "-1/0/1",
  "reason": "SPAN_MISSING|HALLUCINATION|AMBIGUOUS|OUT_OF_SCOPE|DUPLICATE|OTHER|''",
  "notes": "short justification (2 sentences)"
}
Return nothing except this JSON.
""".strip()
)

def get_system_prompt(t):
    t_norm = (t or "").strip().lower()
    if t_norm == "semantic":
        return PROMPT_SEMANTIC
    if t_norm == "syntactic":
        return PROMPT_SYNTACTIC
    if t_norm == "LEXICAL":
        return PROMPT_LEXICAL 
    return PROMPT_SEMANTIC

USER_TMPL = """
===================================
TYPE: {type}
CONTEXT:
{context}

AUGMENTED_QUESTION:
{question}

PROVIDED_ANSWER:
{answer}

ORIGINAL_QUESTION:
{question_ori}
""".strip()

VALID_REASONS = {"", "SPAN_MISSING", "HALLUCINATION", "AMBIGUOUS", "OUT_OF_SCOPE", "DUPLICATE", "OTHER"}


def extract_first_json(text):
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    start = s.find("{"); end = s.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in response.")
    return json.loads(s[start:end + 1])

def normalize_schema(obj):
    if "valid" not in obj or obj["valid"] not in [0, 1, -1, "0", "1", "-1"]:
        raise ValueError("Missing/invalid 'valid' field.")
    valid = int(obj["valid"])
    reason = str(obj.get("reason", "")).strip()
    if reason not in VALID_REASONS:
        reason = "OTHER" if valid == 0 else ""
    notes = str(obj.get("notes", "")).strip()
    return {"valid": valid, "reason": reason, "notes": notes[:600]}

def force_json(response_text):
    data = extract_first_json(response_text)
    return normalize_schema(data)

def _extract_logprob_payload(choice): 
    lp = getattr(choice, "logprobs", None)
    tokens = []
    if lp and getattr(lp, "content", None):
        for step in lp.content: 
            top = []
            if getattr(step, "top_logprobs", None):
                for alt in step.top_logprobs:
                    top.append({
                        "token": getattr(alt, "token", None),
                        "logprob": float(getattr(alt, "logprob", None)),
                    })
            tokens.append({
                "token": getattr(step, "token", None),
                "logprob": float(getattr(step, "logprob", None)),
                "top": top,
            })

    vals = [t["logprob"] for t in tokens if t.get("logprob") is not None]
    return {
        "tokens": tokens,
        "avg_logprob": (mean(vals) if vals else None),
        "sum_logprob": (sum(vals) if vals else None),
        "num_tokens": len(tokens),
    }

def judge_once(system_prompt, user_msg):
    result = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT,
                temperature=0.4,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                logprobs=True,         
                top_logprobs=5,         
            )
            
            choice = resp.choices[0]
            result = force_json(choice.message.content)
            break
        except Exception as e:
            print("******************************")
            print(choice.message.content)
            print("******************************")
            print(system_prompt)
            print()
            print()
            print(user_msg)
            print("******************************")
            print("******************************")
            
            if attempt == 1:  
                raise e 
            
    lp_payload = _extract_logprob_payload(choice)
    result["avg_logprob"] = lp_payload["avg_logprob"]
    result["sum_logprob"] = lp_payload["sum_logprob"]
    result["num_tokens"] = lp_payload["num_tokens"] 
    
    result["logprobs_json"] = json.dumps(lp_payload["tokens"], ensure_ascii=False, separators=(",", ":"))
    return result

def call_judge(row_type, context, question, answer, question_ori, max_retries=3):
    system_prompt = get_system_prompt(row_type)
    user_msg = USER_TMPL.format(type=row_type or "", context=context or "", question=question or "", answer=answer or "", question_ori=question_ori or "")
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return judge_once(system_prompt, user_msg)
        except Exception as e:
            last_err = e
            time.sleep(0.5 * attempt)
    return {
        "valid": 0, "reason": "OTHER",
        "notes": f"Parsing/LLM error: {type(last_err).__name__}: {last_err}",
        "avg_logprob": None, "sum_logprob": None, "num_tokens": 0, "logprobs_json": "[]"
    }

def main():
    ap = argparse.ArgumentParser(description="LLM-as-Judge for augmented QA validity (Semantic/Syntactic/Lexical).")
    ap.add_argument("--input", required=True, help="Input CSV with columns: type, context, question, answer")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of rows processed")
    ap.add_argument("--name", type=str)
    ap.add_argument("--dataset", type=str)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    required = ["type", "context", "question", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[Error] Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.limit and args.limit > 0:
        df = df.iloc[25000:args.limit].copy()

    val_list, reason_list, notes_list = [], [], []
    avg_lp_list, sum_lp_list, n_tok_list, lp_json_list = [], [], [], []

    for i, row in df.iterrows():
        t = str(row.get("type", "")).strip()
        context = "" if pd.isna(row.get("context")) else str(row.get("context"))
        question = "" if pd.isna(row.get("question")) else str(row.get("question"))
        question_ori = "" if pd.isna(row.get("original_question")) else str(row.get("original_question"))

        if pd.isna(row.get("answer")) or str(row.get("answer")).strip() == "":
            answer = "" if pd.isna(row.get("original_answer")) else str(row.get("original_answer"))
        else:
            answer = str(row.get("answer"))

        out = call_judge(t, context, question, answer, question_ori)
        val_list.append(int(out["valid"]))
        reason_list.append(out["reason"])
        notes_list.append(out["notes"])
        avg_lp_list.append(out.get("avg_logprob"))
        sum_lp_list.append(out.get("sum_logprob"))
        n_tok_list.append(out.get("num_tokens"))
        lp_json_list.append(out.get("logprobs_json", "[]"))

        if (i + 1) % 25 == 0:
            print(f"[Info] Judged {i + 1}/{len(df)} items...", file=sys.stderr)

    df["valid"] = val_list
    df["reason"] = reason_list
    df["notes"] = notes_list
    df["avg_logprob"] = avg_lp_list
    df["sum_logprob"] = sum_lp_list
    df["num_tokens"] = n_tok_list
    df["logprobs_json"] = lp_json_list  

    out_df = pd.DataFrame({
        "item_id": df["aug_id"],
        "annotator": args.name,
        "valid": df["valid"],
        "reason": df["reason"],
        "notes": df["notes"],
        "avg_logprob": df["avg_logprob"],
        "sum_logprob": df["sum_logprob"],
        "num_tokens": df["num_tokens"],
        "logprobs_json": df["logprobs_json"],
    })

    out_df.to_csv(f"results/annotator_{args.dataset}_{args.name}.csv", index=False)

if __name__ == "__main__":
    start = time.time()
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")

    main()

    end = time.time()
    print(f"End time:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"Duration:   {end - start:.2f} seconds")