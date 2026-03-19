# Quality Evaluator Skill

## Description
This skill performs comprehensive evaluation of fine-tuned LoRA models in the DND-like model pipeline. It handles perplexity calculation, average loss analysis, and qualitative generation testing for specific tasks like JSON success rate and translation quality.
Trigger on: "evaluate model", "check metrics", "run perplexity", "verify generation quality", "check analyst JSON", "test storyteller style", "compare translation".

## Category
**Product Verification** (Programmatic and qualitative verification of model performance).

## Core Responsibilities
- **Quantitative Metrics**: Calculate **Perplexity** and **Average Loss** on the validation (`val`) datasets.
- **Qualitative Checks**:
    - `analyst`: Calculate JSON parsing success rate over 10 samples (default).
    - `translator`: Generate bilingual side-by-side comparison samples.
    - `storyteller`: Sample story continuations to check narrative style.
- **Reporting**: Generate and store `eval_report.json` and `eval_report.txt` in the adapter's directory.

## Project Nuances & Gotchas
- **Validation Dependency**: Requires `dataset/lora_{task}/{task}_val.jsonl` to exist (produced by `prepare.py`).
- **Adapter Pathing**: Defaults to `outputs/lora_{task}/`. If using a custom checkpoint, pass the `--adapter-path` explicitly.
- **Inference Mode**: Automatically switches model to inference mode via `FastLanguageModel.for_inference`.
- **Sample Count**: Use `--num-samples` to adjust qualitative check depth. High sample counts may be slow on single GPUs.

## Commands
- **Standard Evaluation**: `python evaluate.py --task <task_name>`
- **Custom Path**: `python evaluate.py --task analyst --adapter-path outputs/lora_analyst/checkpoint-100`
- **More Samples**: `python evaluate.py --task storyteller --num-samples 5`

## Output
- `eval_report.json`: Machine-readable metrics and samples.
- `eval_report.txt`: Human-readable summary for quick review.
- JSON summary log printed to stdout under `[EVAL_JSON]`.
