# SSL Augmentation for Extractive QA

Research code for data augmentation strategies (semantic, syntactic, lexical) for extractive question answering on SQuAD v2 and PubMedQA.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your model paths and API keys

# 3. Run training
torchrun --nproc_per_node=2 src/training/finetune.py \
  --dataset squad --model qwen -N 13187 --cfg baseline
```

## Repository Structure

```
configs/          # YAML configurations (models, training, paths)
src/              # Source code
  ├── training/   # LoRA fine-tuning
  ├── augmentation/ # Data augmentation
  ├── evaluation/ # Metrics and evaluation
  └── utils/      # Config loader
data/             # Datasets (processed/augmented)
experiments/      # Training outputs and results
```

## Experiment Types

- **baseline** - Original data only
- **semantic** - Different question aspects
- **syntactic** - Different grammar/structure
- **lexical** - Different vocabulary
- **all** - Combined (25% each)

## Models Required

Download to `MODELS_DIR` (set in `.env`):
- Qwen2.5-7B-Instruct
- Meta-Llama-3.1-8B-Instruct
- Meta-Llama-3.1-70B-Instruct (for augmentation)

## Configuration

Edit YAML files in `configs/` to customize:
- `training.yaml` - Hyperparameters (epochs, batch size, LoRA params)
- `models.yaml` - Model paths
- `paths.yaml` - Data locations

## Citation

```bibtex
@article{XXX,
  title={SSL Augmentation Strategies for Extractive QA},
  author={Your Name},
  year={2026}
}
```
