# Model Trainer Skill

## Description
This skill automates LoRA fine-tuning (Phase 1 SFT) and RL alignment (Phase 2 GRPO/DPO) for the DND-like model using Unsloth. It focuses on 4-bit quantization, memory efficiency on 12GB GPUs, and multi-task experimentation.
Trigger on: "train model", "fine-tune lora", "run training", "run grpo", "run dpo", "optimize hyperparameters", "train analyst", "train storyteller", "train translator", "train reasoning", "phase 2", "rl alignment".

## Category
**Business Process & Team Automation** (Automating complex model training and hyperparameter tuning).

## Core Responsibilities
- **Task-Specific presets**: Manage pre-defined hyperparameter configurations for `analyst`, `reasoning`, `translator`, and `storyteller` tasks across both SFT and RL phases.
- **Resource Oversight**: Monitor VRAM usage (targeting < 11GB on RTX 3060) and handle OOM by adjusting `--batch-size`, `--grad-accum`, or `--max-seq-len`.
- **Experiment Logging**: Ensure training losses and final statuses are recorded in `results.tsv` (SFT) and `results_rl.tsv` (RL).

## Hyperparameter Presets

| Task     | SFT rank/alpha/lr  | RL method | RL lr  |
|----------|--------------------|-----------|--------|
| analyst  | 16 / 32 / 3e-4     | GRPO      | 5e-6   |
| reasoning| 32 / 64 / 1e-4     | GRPO      | 5e-6   |
| translator| 64 / 128 / 1e-4   | GRPO      | 5e-6   |
| storyteller| 32 / 64 / 2e-5   | DPO       | —      |

## Phase 1 — SFT Commands
- **Standard**: `python train_lora.py --task <task_name>`
- **Quick Experiment**: `python train_lora.py --task <task_name> --max-steps 10 --rank 32 --alpha 64`
- **Memory-Safe**: `python train_lora.py --task analyst --batch-size 1 --grad-accum 8`

## Phase 2 — RL Alignment Commands
- **GRPO** (analyst / reasoning / translator):
  `python train_grpo.py --task <task_name> --sft-adapter outputs/lora_<task_name>`
- **DPO** (storyteller only):
  `python train_dpo.py --task storyteller --sft-adapter outputs/lora_storyteller`
- **RL execution order**: analyst → reasoning → translator → storyteller (run sequentially, each depends on its own SFT adapter)

## Output
- SFT adapters: `outputs/lora_{task}/` — results appended to `results.tsv`
- RL adapters: `outputs/lora_{task}_grpo/` or `outputs/lora_{task}_dpo/` — results appended to `results_rl.tsv`

## Phase 3 — Deploy
After all RL/DPO training and evaluation:
```bash
python deploy_lora.py --all --update-config
```
Merges all LoRA adapters and updates Ollama config so the RPG engine picks up the fine-tuned models.

## AutoResearch (`autoresearch.py`)
Automated hyperparameter research loop using an LLM advisor. Supports three backends (`--advisor`):
- `gemini` (default) — requires `GEMINI_API_KEY`
- `claude-api` — requires `ANTHROPIC_API_KEY`
- `subagent` — calls `claude --print` CLI, no API key needed

```bash
python autoresearch.py --task analyst     --mode sft
python autoresearch.py --task reasoning   --mode grpo --max-iterations 3
python autoresearch.py --list
python experiment_tracker.py --ar         # view autoresearch history
```

## OOM Handling Order
1. Lower `--max-seq-len 512`
2. Lower `--batch-size 1`
3. Lower `--rank`

## Project Nuances & Gotchas
- **Quantization Required**: The framework is built for 4-bit (`bnb-4bit`). Do NOT attempt full fine-tuning or 8-bit without explicit permission.
- **Batch Size Optimization**: Default is `batch_size=2` with `grad_accum=4`. If OOM persists, use `batch_size=1` and `grad_accum=8`.
- **LoRA Targets**: Target modules are fixed to `q, k, v, o, gate, up, down` projectors.
- **Analyst SFT only ran 10 steps**: The `outputs/lora_analyst/` adapter is not fully converged. If GRPO metrics are poor after Phase 2, re-run full SFT (3 epochs) before re-running GRPO.
- **Unsloth Patches Required** (unsloth 2026.3.8 + transformers 5.3.0): Several manual patches must be in place before training:
  - `unsloth/models/llama.py:515` — rotary embedding broadcast fix
  - `unsloth/kernels/utils.py:1043` — `matmul_lora` backward dtype fix
  - `unsloth/kernels/fast_lora.py:172–213` — MLP backward float32 fix
  - `unsloth/kernels/fast_lora.py:624–645` — Attention LoRA backward float32 fix
  - `trl/train_grpo.py` — `kl_coeff` → `beta` (GRPOConfig API rename)
  If training crashes with dtype or broadcast errors, verify these patches are still in place (package updates may overwrite them).
- **GRPO checkpoint delay**: `outputs/lora_{task}_grpo/` will be empty until the first checkpoint save. An empty directory does NOT mean training failed — check the live process.
