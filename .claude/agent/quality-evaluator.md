# Quality Evaluator Skill

## Description
This skill performs comprehensive evaluation of fine-tuned LoRA models in the DND-like model pipeline. It covers both Phase 1 SFT evaluation and Phase 2 RL/DPO post-alignment evaluation, handling perplexity, average loss, JSON success rate, and qualitative generation checks.
Trigger on: "evaluate model", "check metrics", "run perplexity", "verify generation quality", "check analyst JSON", "test storyteller style", "compare translation", "evaluate after grpo", "evaluate after dpo", "phase 3 evaluation", "rl eval".

## Category
**Product Verification** (Programmatic and qualitative verification of model performance).

## Core Responsibilities
- **Quantitative Metrics**: Calculate **Perplexity** and **Average Loss** on the validation (`val`) datasets.
- **Qualitative Checks**:
    - `analyst`: Calculate JSON parsing success rate over 10 samples (default).
    - `translator`: Generate bilingual side-by-side comparison samples.
    - `storyteller`: Sample story continuations to check narrative style.
- **Reporting**: Generate and store `eval_report.json` and `eval_report.txt` in the adapter's directory.

## Phase 1 — SFT Evaluation
```bash
python evaluate.py --task <task_name>
```
Reads adapter from `outputs/lora_{task}/`. Results recorded in `results.tsv`.

## Phase 2 — RL/DPO Evaluation (Phase 3)
```bash
# After GRPO training:
python evaluate.py --task analyst     --rl
python evaluate.py --task reasoning   --rl
python evaluate.py --task translator  --rl

# After DPO training:
python evaluate.py --task storyteller --dpo
```
`--rl` reads adapter from `outputs/lora_{task}_grpo/`.
`--dpo` reads adapter from `outputs/lora_{task}_dpo/`.
Results recorded in `results_rl.tsv`.

## Decision Tree After Evaluation
```
analyst GRPO eval → metrics good?
  Yes → proceed to reasoning GRPO → translator GRPO → storyteller DPO
  No  → re-run full analyst SFT (3 epochs, rank 16/alpha 32/lr 3e-4), then GRPO again
```

## Project Nuances & Gotchas
- **Validation Dependency**: Requires `dataset/lora_{task}/{task}_val.jsonl` to exist (produced by `prepare.py`).
- **Adapter Pathing**: Defaults differ by flag — `--rl` uses `_grpo/`, `--dpo` uses `_dpo/`. If using a custom checkpoint, pass `--adapter-path` explicitly.
- **Inference Mode**: Automatically switches model to inference mode via `FastLanguageModel.for_inference`.
- **Sample Count**: Use `--num-samples` to adjust qualitative check depth. High sample counts may be slow on single GPUs.
- **use_cache=False**: `generate_response()` requires `use_cache=False` to avoid Unsloth rotary embedding broadcast errors during inference. If you see shape mismatch errors, verify this flag is set.

## Commands Reference
- **Standard SFT Eval**: `python evaluate.py --task <task_name>`
- **RL Eval**: `python evaluate.py --task <task_name> --rl`
- **DPO Eval**: `python evaluate.py --task storyteller --dpo`
- **Custom Path**: `python evaluate.py --task analyst --adapter-path outputs/lora_analyst_grpo/checkpoint-100`
- **More Samples**: `python evaluate.py --task storyteller --num-samples 5`

## Output
- `eval_report.json`: Machine-readable metrics and samples (inside adapter directory).
- `eval_report.txt`: Human-readable summary for quick review.
- JSON summary log printed to stdout under `[EVAL_JSON]`.
- Aggregate log: `results.tsv` (SFT), `results_rl.tsv` (RL/DPO).
