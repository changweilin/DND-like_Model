"""
train_lora.py — 多任務可配置 QLoRA 微調腳本

使用方式：
  python train_lora.py --task analyst
  python train_lora.py --task storyteller --rank 64 --alpha 128
  python train_lora.py --task analyst --max-steps 10   # 快速實驗模式

輸出：
  - LoRA adapter 存至 outputs/lora_{task}/
  - 結構化 JSON 摘要行列印至 stdout（供自動化迴圈解析）
"""

import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

# ── 任務預設參數 ───────────────────────────────────────────────────────────────

TASK_PRESETS = {
    "storyteller": {
        "lora_rank":    32,
        "lora_alpha":   64,
        "learning_rate": 2e-5,
        "num_epochs":   3,
        "dataset_dir":  "dataset/lora_storyteller",
        "output_dir":   "outputs/lora_storyteller",
    },
    "storyteller_extra": {
        "lora_rank":    64,
        "lora_alpha":   64,
        "learning_rate": 2e-5,
        "num_epochs":   3,
        "dataset_dir":  "dataset/lora_storyteller_extra",
        "output_dir":   "outputs/lora_storyteller_extra",
    },
    "translator": {
        "lora_rank":    64,
        "lora_alpha":   128,
        "learning_rate": 1e-4,
        "num_epochs":   2,
        "dataset_dir":  "dataset/lora_translator",
        "output_dir":   "outputs/lora_translator",
    },
    "analyst": {
        "lora_rank":    16,
        "lora_alpha":   32,
        "learning_rate": 3e-4,
        "num_epochs":   2,
        "dataset_dir":  "dataset/lora_analyst",
        "output_dir":   "outputs/lora_analyst",
    },
    "reasoning": {
        "lora_rank":    32,
        "lora_alpha":   64,
        "learning_rate": 1e-4,
        "num_epochs":   3,
        "dataset_dir":  "dataset/lora_reasoning",
        "output_dir":   "outputs/lora_reasoning",
    },
}

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="多任務 QLoRA 微調腳本")
    parser.add_argument(
        "--task", required=True,
        choices=list(TASK_PRESETS.keys()),
        help="訓練任務（對應 TASK_PRESETS 的 key）",
    )
    parser.add_argument("--rank",       type=int,   default=None, help="LoRA rank（覆蓋預設值）")
    parser.add_argument("--alpha",      type=int,   default=None, help="LoRA alpha（覆蓋預設值）")
    parser.add_argument("--lr",         type=float, default=None, help="Learning rate（覆蓋預設值）")
    parser.add_argument("--epochs",     type=int,   default=None, help="訓練 epoch 數（覆蓋預設值）")
    parser.add_argument("--batch-size", type=int,   default=2,    help="per_device_train_batch_size（預設 2）")
    parser.add_argument("--grad-accum", type=int,   default=4,    help="gradient_accumulation_steps（預設 4）")
    parser.add_argument("--max-seq-len",type=int,   default=1024, help="最大序列長度（預設 1024）")
    parser.add_argument("--max-steps",  type=int,   default=-1,   help="最大訓練步數，-1 表示跑完全部（預設 -1）")
    parser.add_argument("--seed",       type=int,   default=1234, help="隨機種子（預設 1234）")
    return parser.parse_args()


# ── 資料格式化 ────────────────────────────────────────────────────────────────

def make_formatting_func(tokenizer):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}
    return formatting_prompts_func


# ── 主訓練流程 ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    preset = dict(TASK_PRESETS[args.task])

    # 命令列覆蓋
    if args.rank    is not None: preset["lora_rank"]    = args.rank
    if args.alpha   is not None: preset["lora_alpha"]   = args.alpha
    if args.lr      is not None: preset["learning_rate"] = args.lr
    if args.epochs  is not None: preset["num_epochs"]   = args.epochs

    rank    = preset["lora_rank"]
    alpha   = preset["lora_alpha"]
    lr      = preset["learning_rate"]
    epochs  = preset["num_epochs"]
    ds_dir  = preset["dataset_dir"]
    out_dir = preset["output_dir"]

    train_path = os.path.join(ds_dir, f"{os.path.basename(ds_dir)}_train.jsonl")
    val_path   = os.path.join(ds_dir, f"{os.path.basename(ds_dir)}_val.jsonl")

    # 檢查資料集
    if not os.path.exists(train_path):
        print(f"[ERROR] 找不到訓練資料：{train_path}")
        print("請先執行 python prepare.py")
        sys.exit(1)

    print(f"[Task]  {args.task}")
    print(f"[Model] {MODEL_NAME}")
    print(f"[Rank]  {rank}  Alpha={alpha}  LR={lr}  Epochs={epochs}")
    print(f"[Data]  {train_path}")

    # ── 載入模型 ──────────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    fmt_fn = make_formatting_func(tokenizer)

    # ── 載入資料集 ────────────────────────────────────────────────────────────
    train_dataset = load_dataset("json", data_files=train_path, split="train")
    train_dataset = train_dataset.map(fmt_fn, batched=True)

    eval_dataset = None
    if os.path.exists(val_path):
        eval_dataset = load_dataset("json", data_files=val_path, split="train")
        eval_dataset = eval_dataset.map(fmt_fn, batched=True)
        print(f"[Eval]  {val_path}  ({len(eval_dataset)} 筆)")
    else:
        print(f"[WARN] 找不到驗證集：{val_path}，跳過 eval")

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        learning_rate=lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=out_dir,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # ── 訓練 ──────────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    trainer_stats = trainer.train()

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    train_loss = trainer_stats.training_loss
    eval_loss  = None
    if eval_dataset is not None and trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
                break

    # ── 儲存 ──────────────────────────────────────────────────────────────────
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n[Done] LoRA adapter 已儲存至 {out_dir}")

    # ── 結構化輸出 ────────────────────────────────────────────────────────────
    result = {
        "task":           args.task,
        "train_loss":     round(train_loss, 4) if train_loss is not None else None,
        "eval_loss":      round(eval_loss, 4)  if eval_loss  is not None else None,
        "epochs":         epochs,
        "max_steps":      args.max_steps,
        "peak_memory_gb": round(peak_mem_gb, 2),
        "rank":           rank,
        "alpha":          alpha,
        "lr":             lr,
        "batch_size":     args.batch_size,
        "grad_accum":     args.grad_accum,
        "max_seq_len":    args.max_seq_len,
        "seed":           args.seed,
        "status":         "ok",
    }
    print("\n[RESULT_JSON]")
    print(json.dumps(result, ensure_ascii=False))

    # ── 自動記錄到 results.tsv ────────────────────────────────────────────────
    tsv_path = "results.tsv"
    write_header = not os.path.exists(tsv_path)
    with open(tsv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join([
                "timestamp", "task", "rank", "alpha", "lr", "epochs",
                "max_steps", "train_loss", "eval_loss", "peak_memory_gb",
                "status", "description"
            ]) + "\n")
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        row = [
            ts, args.task, str(rank), str(alpha), str(lr), str(epochs),
            str(args.max_steps),
            str(result["train_loss"] or ""),
            str(result["eval_loss"]  or ""),
            str(result["peak_memory_gb"]),
            "ok", "",
        ]
        f.write("\t".join(row) + "\n")
    print(f"[TSV] 結果已附加至 {tsv_path}")


if __name__ == "__main__":
    main()
