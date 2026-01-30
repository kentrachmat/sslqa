#!/bin/bash
set -e

# Paths
SQUAD_INPUT="../augmented_squad_train.csv"
PUBMED_INPUT="../augmented_pubmed_train.csv"

# Run for SQuADv2
for i in 1
do
    echo "[Info] Running judge for annotator chatgpt$i on SQuADv2 ..."
    python llm_as_judge.py \
        --input "$SQUAD_INPUT" \
        --name "gpt4o_$i" \
        --dataset "squadv2" \
        --limit 35000
done 

# Run for PUBMED
for i in 1
do
    echo "[Info] Running judge for annotator chatgpt$i on pubmed ..."
    python llm_as_judge.py \
        --input "$PUBMED_INPUT" \
        --name "gpt4o_$i" \
        --dataset "pubmed" \
        --limit 35000
done 

echo "[Done] All runs completed."