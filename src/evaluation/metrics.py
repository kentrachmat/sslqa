#!/usr/bin/env python3
import string
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ---------------- Normalization ----------------
_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_NO_ANSWER_CANON = {"", "no answer", "unanswerable", "n/a", "none", "null", "na", "not available"}

def normalize_answer(s):
    if s is None:
        return ""
    s = s.lower().translate(_PUNCT_TABLE)
    s = " ".join(w for w in s.split() if w not in _ARTICLES)
    return " ".join(s.split())

def canon_no_answer(s):
    return "" if normalize_answer(s) in _NO_ANSWER_CANON else normalize_answer(s)

def f1_score(pred, gold):
    pred_tokens = canon_no_answer(pred).split()
    gold_tokens = canon_no_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in gold_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in pred_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(pred, gold):
    return 1.0 if canon_no_answer(pred) == canon_no_answer(gold) else 0.0

def exact_match_binary(pred, gold):
    pred_norm = canon_no_answer(pred)
    gold_norm = canon_no_answer(gold)

    if gold_norm in pred_norm:
        return 1.0
    return 0.0

EMBED_MODEL_NAME = "/home/brachmat/phd/models/models--qwen3-embedding-0.6B"
model = SentenceTransformer(EMBED_MODEL_NAME)

def semantic_similarity(pred, gold):
    pred_c = canon_no_answer(pred)
    gold_c = canon_no_answer(gold)
    if pred_c == "" and gold_c == "":
        return 1.0
    if pred_c == "" or gold_c == "":
        return 0.0
    emb_pred = model.encode([pred_c], convert_to_numpy=True, normalize_embeddings=True)[0]
    emb_gold = model.encode([gold_c], convert_to_numpy=True, normalize_embeddings=True)[0]
    return float(cos_sim([emb_pred], [emb_gold])[0][0])

# ---------------- Evaluation ----------------
def evaluate_pubmed(results_csv, out_csv):
    df = pd.read_csv(results_csv)
    all_out = []
    
    binary_df = pd.read_csv("../datasets/pubmed_final/binary_pubmed.csv")
    for m, group in df.groupby("model"):
        ems, f1s, sims, is_na, binary = [], [], [], [], []
        for i, (index, row) in enumerate(tqdm(group.iterrows(), total=len(group), desc=f"Evaluating {m}")):
            pred = str(row["pred_answer"]) if not pd.isna(row["pred_answer"]) else ""
            gold = str(row["gold_answer"]) if not pd.isna(row["gold_answer"]) else ""
            bina = str(binary_df.iloc[i]["binary_answer"]) if not pd.isna(binary_df.iloc[i]["binary_answer"]) else ""
            
            binary.append(exact_match_binary(pred, bina))
            ems.append(exact_match(pred, gold))
            f1s.append(f1_score(pred, gold))
            sims.append(semantic_similarity(pred, gold))
            is_na.append(1.0 if canon_no_answer(gold) == "" else 0.0)

        all_out.append({
            "model": m,
            "em_mean_binary": float(np.mean(binary)),
            "em_mean": float(np.mean(ems)),
            "f1_mean": float(np.mean(f1s)),
            "sim_mean": float(np.mean(sims)),
            "sim_std": float(np.std(sims)),
            "unanswerable_rate": float(np.mean(is_na)),
        })
        print(all_out)

    out_df = pd.DataFrame(all_out)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved evaluation: {out_csv}")
    print(out_df.to_string(index=False))
    
def evaluate_squad(results_csv, out_csv):
    df = pd.read_csv(results_csv)
    all_out = []

    for m, group in df.groupby("model"):
        ems, f1s, sims, is_na = [], [], [], []
        for _, row in tqdm(group.iterrows(), total=len(group), desc=f"Evaluating {m}"):
            pred = str(row["pred_answer"]) if not pd.isna(row["pred_answer"]) else ""
            gold = str(row["gold_answer"]) if not pd.isna(row["gold_answer"]) else ""
            ems.append(exact_match(pred, gold))
            f1s.append(f1_score(pred, gold))
            sims.append(semantic_similarity(pred, gold))
            is_na.append(1.0 if canon_no_answer(gold) == "" else 0.0)

        all_out.append({
            "model": m,
            "em_mean": float(np.mean(ems)),
            "f1_mean": float(np.mean(f1s)),
            "sim_mean": float(np.mean(sims)),
            "sim_std": float(np.std(sims)),
            "unanswerable_rate": float(np.mean(is_na)),
        })
        print(all_out)
        

    out_df = pd.DataFrame(all_out)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved evaluation: {out_csv}")
    print(out_df.to_string(index=False))

# ---------------- Run ----------------
if __name__ == "__main__":
    for folder in ["results/pubmed_eval"]:
    # for folder in ["results/squad_v2_eval"]:
        results_csv = Path(folder) / "all_result.csv"
        out_csv = Path(folder) / "all_evaluation_2.csv"

        if not results_csv.exists():
            print(f"⚠️ Skipping {folder}, no all_result.csv found")
            continue

        if "pubmed" in folder:
            print(f"▶ Evaluating PubMed in {folder}")
            evaluate_pubmed(results_csv, out_csv)
        elif "squad" in folder:
            print(f"▶ Evaluating SQuAD in {folder}")
            evaluate_squad(results_csv, out_csv)
        else:
            print(f"⚠️ Unknown folder type: {folder}, skipping")