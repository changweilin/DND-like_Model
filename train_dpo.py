"""
train_dpo.py — DPO 直接偏好優化訓練腳本

DPO (Direct Preference Optimization) 讓模型從 chosen/rejected 偏好對中學習，
無需顯式 Reward Model，直接從偏好資料的對數機率差優化 policy。
適合無法設計規則式獎勵的任務（如故事接龍的文風品質）。

使用方式：
  python train_dpo.py --task storyteller
  python train_dpo.py --task storyteller --sft-adapter outputs/lora_storyteller
  python train_dpo.py --task storyteller --max-steps 50   # 快速測試

前置需求：
  python prepare_dpo.py   # 生成 DPO 偏好對資料集

輸出：
  - DPO LoRA adapter 存至 outputs/lora_{task}_dpo/
  - 結構化 JSON 摘要行列印至 stdout（供自動化迴圈解析）
  - 訓練記錄附加至 results_rl.tsv
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
from trl import DPOTrainer, DPOConfig

# ── 任務預設參數 ───────────────────────────────────────────────────────────────

TASK_PRESETS = {
    "storyteller": {
        "lora_rank":        32,
        "lora_alpha":       64,
        "learning_rate":    1e-5,
        "num_epochs":       2,
        "beta":             0.1,    # DPO KL 懲罰係數（0.1 是 DPO 標準值）
        "max_length":       1024,   # prompt + response 總長度上限
        "max_prompt_length": 512,
        "dataset_dir":      "dataset/lora_storyteller_dpo",
        "output_dir":       "outputs/lora_storyteller_dpo",
    },
}

MODEL_NAME     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024


# ── 資料格式轉換 ──────────────────────────────────────────────────────────────

def convert_to_dpo_dataset(dataset, tokenizer):
    """
    把 prepare_dpo.py 輸出的偏好對格式轉換為 TRL DPOTrainer 所需格式。

    輸入格式（每筆 JSONL）：
      {
        "conversations": [{"from": "system", ...}, {"from": "human", ...}],
        "chosen":   {"from": "gpt", "value": "（高品質回應）"},
        "rejected": {"from": "gpt", "value": "（低品質回應）"}
      }

    輸出格式（TRL DPOTrainer）：
      {
        "prompt":   "<|im_start|>system\n...<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
        "chosen":   "（高品質回應）",
        "rejected": "（低品質回應）"
      }
    """
    def process(examples):
        prompts, chosens, rejecteds = [], [], []
        for convos, chosen_obj, rejected_obj in zip(
            examples["conversations"],
            examples["chosen"],
            examples["rejected"],
        ):
            prompt = tokenizer.apply_chat_template(
                convos, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            chosens.append(chosen_obj["value"])
            rejecteds.append(rejected_obj["value"])
        return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}

    return dataset.map(process, batched=True, remove_columns=dataset.column_names)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DPO 直接偏好優化訓練腳本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True,
        choices=list(TASK_PRESETS.keys()),
        help="訓練任務",
    )
    parser.add_argument("--rank",        type=int,   default=None, help="LoRA rank（覆蓋預設值）")
    parser.add_argument("--alpha",       type=int,   default=None, help="LoRA alpha（覆蓋預設值）")
    parser.add_argument("--lr",          type=float, default=None, help="Learning rate（覆蓋預設值）")
    parser.add_argument("--epochs",      type=int,   default=None, help="訓練 epoch 數（覆蓋預設值）")
    parser.add_argument("--beta",        type=float, default=None, help="DPO KL beta 係數（覆蓋預設值）")
    parser.add_argument("--grad-accum",  type=int,   default=8,    help="gradient_accumulation_steps（預設 8）")
    parser.add_argument("--max-steps",   type=int,   default=-1,   help="最大訓練步數，-1 表示跑完全部（預設 -1）")
    parser.add_argument("--seed",        type=int,   default=1234, help="隨機種子（預設 1234）")
    parser.add_argument("--sft-adapter", type=str,   default=None,
                        help="從 SFT adapter 繼續 DPO 訓練（路徑，預設從 base model 開始）")
    return parser.parse_args()


# ── 主訓練流程 ────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    preset = dict(TASK_PRESETS[args.task])

    # CLI 覆蓋預設值
    if args.rank   is not None: preset["lora_rank"]    = args.rank
    if args.alpha  is not None: preset["lora_alpha"]   = args.alpha
    if args.lr     is not None: preset["learning_rate"] = args.lr
    if args.epochs is not None: preset["num_epochs"]   = args.epochs
    if args.beta   is not None: preset["beta"]         = args.beta

    sft_adapter  = args.sft_adapter
    model_source = sft_adapter if sft_adapter else MODEL_NAME
    ds_dir       = preset["dataset_dir"]
    out_dir      = preset["output_dir"]
    train_path   = os.path.join(ds_dir, f"{os.path.basename(ds_dir)}_train.jsonl")

    if not os.path.exists(train_path):
        print(f"[ERROR] 找不到訓練資料：{train_path}")
        print("請先執行 python prepare_dpo.py")
        sys.exit(1)

    print(f"[Task]     {args.task}  (mode=DPO)")
    print(f"[Model]    {model_source}")
    print(f"[LoRA]     rank={preset['lora_rank']}  alpha={preset['lora_alpha']}")
    print(f"[DPO]      beta={preset['beta']}  max_length={preset['max_length']}  "
          f"max_prompt={preset['max_prompt_length']}")
    print(f"[Optim]    lr={preset['learning_rate']}  "
          f"epochs={preset['num_epochs']}  grad_accum={args.grad_accum}")
    print(f"[Data]     {train_path}")
    if sft_adapter:
        print(f"[SFT]      從 {sft_adapter} 繼續訓練")
    print()

    # ── 載入模型 ──────────────────────────────────────────────────────────────
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_source,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # 若從 SFT adapter 載入，已含 LoRA 權重，不需再 get_peft_model
    if not sft_adapter:
        model = FastLanguageModel.get_peft_model(
            model,
            r=preset["lora_rank"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=preset["lora_alpha"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # ── 載入並轉換資料集 ──────────────────────────────────────────────────────
    raw_train     = load_dataset("json", data_files=train_path, split="train")
    train_dataset = convert_to_dpo_dataset(raw_train, tokenizer)
    print(f"[Dataset]  {len(train_dataset)} 筆已轉換為 DPO 格式")

    eval_dataset = None
    val_path = os.path.join(ds_dir, f"{os.path.basename(ds_dir)}_val.jsonl")
    if os.path.exists(val_path):
        raw_val      = load_dataset("json", data_files=val_path, split="train")
        eval_dataset = convert_to_dpo_dataset(raw_val, tokenizer)
        print(f"[Eval]     {len(eval_dataset)} 筆")
    else:
        print(f"[WARN] 找不到驗證集：{val_path}，跳過 eval")

    # ── DPOConfig ─────────────────────────────────────────────────────────────
    dpo_config = DPOConfig(
        output_dir=out_dir,
        # DPO 核心設定
        beta=preset["beta"],
        max_length=preset["max_length"],
        max_prompt_length=preset["max_prompt_length"],
        # 優化設定
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=preset["num_epochs"],
        max_steps=args.max_steps,
        learning_rate=preset["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        # 記錄與儲存
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
        seed=args.seed,
        # 記憶體優化
        gradient_checkpointing=True,
    )

    # ref_model=None：讓 Unsloth 處理 reference model（節省 VRAM）
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tokenizer,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # ── 訓練 ──────────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    t0            = time.time()
    trainer_stats = trainer.train()
    elapsed       = time.time() - t0
    peak_mem_gb   = torch.cuda.max_memory_allocated() / 1e9
    train_loss    = trainer_stats.training_loss

    eval_loss = None
    if eval_dataset is not None and trainer.state.log_history:
        for entry in reversed(trainer.state.log_history):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
                break

    # ── 儲存 adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n[Done] DPO LoRA adapter 已儲存至 {out_dir}")

    # ── 結構化 JSON 輸出 ──────────────────────────────────────────────────────
    result = {
        "task":           args.task,
        "mode":           "dpo",
        "train_loss":     round(train_loss, 4) if train_loss is not None else None,
        "eval_loss":      round(eval_loss, 4)  if eval_loss  is not None else None,
        "epochs":         preset["num_epochs"],
        "max_steps":      args.max_steps,
        "elapsed_sec":    round(elapsed, 1),
        "peak_memory_gb": round(peak_mem_gb, 2),
        "rank":           preset["lora_rank"],
        "alpha":          preset["lora_alpha"],
        "lr":             preset["learning_rate"],
        "beta":           preset["beta"],
        "grad_accum":     args.grad_accum,
        "sft_adapter":    sft_adapter or "none",
        "status":         "ok",
    }
    print("\n[RESULT_JSON]")
    print(json.dumps(result, ensure_ascii=False))

    # ── 自動記錄至 results_rl.tsv ─────────────────────────────────────────────
    tsv_path     = "results_rl.tsv"
    write_header = not os.path.exists(tsv_path)
    with open(tsv_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join([
                "timestamp", "task", "mode", "rank", "alpha", "lr",
                "num_generations", "max_completion", "kl_coeff",
                "epochs", "max_steps", "train_loss",
                "peak_memory_gb", "elapsed_sec", "sft_adapter", "status",
            ]) + "\n")
        ts  = time.strftime("%Y-%m-%dT%H:%M:%S")
        row = [
            ts, args.task, "dpo",
            str(preset["lora_rank"]), str(preset["lora_alpha"]),
            str(preset["learning_rate"]),
            "N/A", "N/A", str(preset["beta"]),   # num_gen / max_completion / kl_coeff
            str(preset["num_epochs"]), str(args.max_steps),
            str(result["train_loss"] or ""),
            str(result["peak_memory_gb"]), str(result["elapsed_sec"]),
            sft_adapter or "none", "ok",
        ]
        f.write("\t".join(row) + "\n")
    print(f"[TSV] 結果已附加至 {tsv_path}")


if __name__ == "__main__":
    main()
