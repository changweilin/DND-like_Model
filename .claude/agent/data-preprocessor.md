# Data Preprocessor Skill

## Description
This skill handles the transformation of raw RPG and literature datasets into the ShareGPT format required for LoRA fine-tuning. It involves data conversion, training/validation splitting, and token-level validation to prevent Out-Of-Memory (OOM) errors.
Trigger on: "prepare data", "convert dataset", "check tokens", "split training data", "validate records", "dataset preparation".

## Category
**Data Fetching & Analysis** (Standardizing data formats and performing statistical validation).

## Core Responsibilities
- **Format Conversion**: Convert instructions/input/output into ShareGPT-style `conversations` via `prepare.py`.
- **Dataset Splitting**: Enforce a 90/10 train/validation split with fixed `SEED=1234`.
- **Validation**: Identify records exceeding the `LONG_TOKEN_THRESHOLD` (1024 tokens) which may cause memory issues on 12GB GPUs.

## Project Nuances & Gotchas
- **Token Estimation**: The script uses a fast estimation formula (`char_length // 4`). If precision is needed, use actual tokenizers, but for pre-screening, this is the project's standard.
- **Source Mapping**: `rpg_dataset` and `literature_dataset` maps to `fantasy` and `litrpg` system prompts respectively. Ensure new sources are added to `FANTASY_NARRATOR_SOURCES` or `LITRPG_SOURCES` in `prepare.py`.
- **Directory Structure**: Outputs are saved in `dataset/lora_{task}/`. Do NOT change this structure as `train_lora.py` expects it.

## Commands
- **Run pipeline**: `python prepare.py`
- **Output analysis**: Check stdout for "[OK]" or "[WARN]" status per task.

## Persistent Data
- Log conversion issues and token length distributions in `${CLAUDE_PLUGIN_DATA}/preprocessor.log` for trend analysis across data additions.
- Also note: RLVR cleaned datasets are stored in `dataset/finetune/sharegpt/cleaned/` — these are the post-filtering outputs used for Phase 2 RL training, not raw SFT data.
