# Model Trainer Skill

## Description
This skill automates the LoRA fine-tuning process for the DND-like model using the Unsloth library. It focuses on 4-bit quantization, memory efficiency on 12GB GPUs, and multi-task experimentation.
Trigger on: "train model", "fine-tune lora", "run training", "optimize hyperparameters", "train analyst", "train storyteller", "train translator".

## Category
**Business Process & Team Automation** (Automating complex model training and hyperparameter tuning).

## Core Responsibilities
- **Task-Specific presets**: Manage pre-defined hyperparameter configurations for `storyteller`, `storyteller_extra`, `translator`, and `analyst` tasks.
- **Resource Oversight**: Monitor VRAM usage (targeting < 11GB on RTX 3060) and handle OOM (Out-of-Memory) by adjusting `--batch-size`, `--grad-accum`, or `--max-seq-len`.
- **Experiment Logging**: Ensure training losses, peaks in memory, and final statuses are recorded in `results.tsv`.

## Project Nuances & Gotchas
- **Quantization Required**: The framework is built for 4-bit (`bnb-4bit`). Do NOT attempt full fine-tuning or 8-bit without explicit permission.
- **Batch Size Optimization**: Default is `batch_size=2` with `grad_accum=4`. If OOM persists, use `batch_size=1` and `grad_accum=8` for gradient accumulation.
- **Task Presets**: Different tasks imply different rank/alpha needs (e.g., `translator` needs higher rank `64`, `analyst` needs lower rank `16`). Always mention the task name.
- **LoRA Targets**: Target modules are fixed to `q, k, v, o, gate, up, down` projectors.

## Commands
- **Standard Training**: `python train_lora.py --task <task_name>`
- **Quick Experiment**: `python train_lora.py --task <task_name> --max-steps 10 --rank 32 --alpha 64`
- **Memory-Safe**: `python train_lora.py --task analyst --batch-size 1 --grad-accum 8`

## Output
- LoRA adapters are saved to `outputs/lora_{task}/`.
- Training results are appended to `results.tsv`.
