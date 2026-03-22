"""
train_grpo.py — 多任務 GRPO 強化學習訓練腳本

GRPO (Group Relative Policy Optimization) 讓模型從自己的輸出中學習：
對同一個 prompt 生成多個候選答案，以獎勵函數評分後優化 policy。
無需 Reward Model，完全基於規則式獎勵，適合 RTX 3060 12GB 環境。

使用方式：
  python train_grpo.py --task reasoning
  python train_grpo.py --task analyst --num-generations 4
  python train_grpo.py --task reasoning --sft-adapter outputs/lora_reasoning --max-steps 50
  python train_grpo.py --task analyst --max-steps 10   # 快速實驗

輸出：
  - RL LoRA adapter 存至 outputs/lora_{task}_grpo/
  - 結構化 JSON 摘要行列印至 stdout（供自動化迴圈解析）
  - 訓練記錄附加至 results_rl.tsv
"""

import argparse
import json
import os
import re
import sys
import time

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import GRPOTrainer, GRPOConfig

# ── Unsloth GRPO 整合補丁 ──────────────────────────────────────────────────────
# 若安裝的 Unsloth 版本支援 PatchFastRL，啟用以取得額外記憶體優化
try:
    from unsloth import PatchFastRL
    PatchFastRL("GRPO", FastLanguageModel)
    print("[Info] Unsloth PatchFastRL(GRPO) 已套用")
except (ImportError, AttributeError):
    print("[Info] 使用標準 TRL GRPOTrainer（不含 Unsloth GRPO patch）")

# ── 任務預設參數 ───────────────────────────────────────────────────────────────
#
# RTX 3060 12GB VRAM 估算：
#   SFT QLoRA 4-bit  ≈ 8-9 GB
#   GRPO num_gen=4   ≈ 10-11 GB（policy + reference 同時持有）
#   GRPO num_gen=6   ≈ 11-12 GB（接近上限，需小心 max_completion_length）
#
TASK_PRESETS = {
    "reasoning": {
        "lora_rank":             16,
        "lora_alpha":            32,
        "learning_rate":         5e-6,   # RL 需比 SFT 更小的 LR
        "num_epochs":            2,
        "num_generations":       4,      # 每個 prompt 生成 4 個候選比較
        "max_completion_length": 384,    # 推理輸出需要足夠空間
        "kl_coeff":              0.1,    # KL 懲罰：防止偏離 reference 太遠
        "dataset_dir":           "dataset/lora_reasoning",
        "output_dir":            "outputs/lora_reasoning_grpo",
    },
    "analyst": {
        "lora_rank":             8,
        "lora_alpha":            16,
        "learning_rate":         1e-5,
        "num_epochs":            3,
        "num_generations":       6,      # JSON 輸出短，可多生成幾個
        "max_completion_length": 128,    # JSON 輸出很短
        "kl_coeff":              0.05,   # 格式任務偏離少，KL 可小
        "dataset_dir":           "dataset/lora_analyst",
        "output_dir":            "outputs/lora_analyst_grpo",
    },
    "translator": {
        "lora_rank":             32,
        "lora_alpha":            64,
        "learning_rate":         2e-6,   # BLEU 梯度噪音大，LR 要更小
        "num_epochs":            2,
        "num_generations":       4,
        "max_completion_length": 256,    # 截斷問題：盡量小以節省 VRAM
        "kl_coeff":              0.1,
        "dataset_dir":           "dataset/lora_translator",
        "output_dir":            "outputs/lora_translator_grpo",
    },
}

MODEL_NAME     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024


# ── 資料格式轉換 ──────────────────────────────────────────────────────────────

def convert_to_grpo_dataset(dataset, tokenizer):
    """
    把 ShareGPT 格式的資料集轉換成 GRPO 所需格式：
      - prompt     : 格式化後的聊天模板（不含 gpt 回應）
      - answer     : 原始 gpt 回應（reward function 可用於參考）
      - input_text : human 欄位原始文字（analyst 任務用於實體定位）
    """
    def process(examples):
        prompts, answers, input_texts = [], [], []
        for convos in examples["conversations"]:
            human_val = next(
                (c["value"] for c in convos if c["from"] == "human"), ""
            )
            gpt_val = next(
                (c["value"] for c in convos if c["from"] == "gpt"), ""
            )
            prompt_convos = [c for c in convos if c["from"] != "gpt"]
            prompt = tokenizer.apply_chat_template(
                prompt_convos, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            answers.append(gpt_val)
            input_texts.append(human_val)
        return {
            "prompt":     prompts,
            "answer":     answers,
            "input_text": input_texts,
        }

    return dataset.map(process, batched=True, remove_columns=dataset.column_names)


# ── 獎勵函數 ──────────────────────────────────────────────────────────────────
#
# TRL GRPOTrainer 呼叫獎勵函數的簽名：
#   reward_fn(completions: list[str], prompts: list[str], **dataset_columns)
# dataset 中除 "prompt" 外的欄位（answer, input_text）會透過 **kwargs 傳入

def _extract_json_from_conclusion(text: str) -> dict | None:
    """從推理任務的【結論】區塊提取 JSON 物件。"""
    match = re.search(r"【結論】\s*(\{.*?\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_init_affinity(input_text: str) -> int | None:
    """
    從 human 輸入的遊戲狀態 JSON 中提取初始好感度數值。
    例如：{"「德拉克」好感度": 30, ...} → 30
    """
    match = re.search(r'"[^"]*好感度"\s*:\s*(\d+)', input_text)
    if match:
        return int(match.group(1))
    return None


def reasoning_reward_fn(completions, prompts, answer, input_text, **kwargs):
    """
    lora_reasoning 的 GRPO 獎勵函數（最高分 4.0）

    R1 (+1.0)：格式完整——含【推理步驟】與【結論】兩個區塊
    R2 (+1.0)：【結論】內的 JSON 可被 json.loads() 解析
    R3 (+1.0)：JSON 含「好感度增量」與「新好感度」兩個關鍵欄位
    R4 (+1.0)：數值計算正確（初始好感度 + 增量 ≈ 新好感度）
    """
    rewards = []
    for completion, inp_text in zip(completions, input_text):
        score = 0.0

        # R1: 格式完整性
        has_steps      = "【推理步驟】" in completion
        has_conclusion = "【結論】"     in completion
        if has_steps and has_conclusion:
            score += 1.0
        elif has_steps or has_conclusion:
            score += 0.3   # 部分格式給予少量分數

        # R2 / R3 / R4: JSON 驗證
        result_json = _extract_json_from_conclusion(completion)
        if result_json is not None:
            score += 1.0   # R2: JSON 可解析

            delta   = result_json.get("好感度增量")
            new_aff = result_json.get("新好感度")

            if delta is not None and new_aff is not None:
                score += 1.0   # R3: 關鍵欄位存在

                # R4: 數值計算正確
                init_aff = _extract_init_affinity(inp_text)
                if init_aff is not None:
                    try:
                        if abs((int(init_aff) + int(delta)) - int(new_aff)) <= 1:
                            score += 1.0
                    except (TypeError, ValueError):
                        pass

        rewards.append(score)
    return rewards


def analyst_reward_fn(completions, prompts, answer, input_text, **kwargs):
    """
    lora_analyst 的 GRPO 獎勵函數（最高分約 3.5，無上限懲罰）

    R1 (+1.0)：輸出是合法 JSON
    R2 (+0.5)：JSON 含「角色」與「組織」兩個 key
    R3 (+0.25×n, max +2.0)：每個實體出現在輸入原文中（定位獎勵）
    R4 (-0.5×n)：每個實體未出現在輸入原文（幻覺懲罰）
    最低分 clamp 到 0.0
    """
    rewards = []
    for completion, inp_text in zip(completions, input_text):
        score     = 0.0
        inp_lower = inp_text.lower()

        try:
            result = json.loads(completion.strip())
            score += 1.0   # R1
        except (json.JSONDecodeError, ValueError):
            rewards.append(0.0)
            continue

        # R2: key 結構
        has_roles = "角色" in result
        has_orgs  = "組織" in result
        if has_roles and has_orgs:
            score += 0.5
        elif has_roles or has_orgs:
            score += 0.2

        # R3 + R4: 實體定位
        entities: list = []
        if has_roles and isinstance(result["角色"], list):
            entities += result["角色"]
        if has_orgs and isinstance(result["組織"], list):
            entities += result["組織"]

        ground_score = 0.0
        for entity in entities:
            if isinstance(entity, str) and entity.strip():
                if entity.lower() in inp_lower:
                    ground_score += 0.25           # R3
                else:
                    score -= 0.5                   # R4

        score += min(ground_score, 2.0)            # R3 上限 2.0
        score  = max(score, 0.0)
        rewards.append(score)
    return rewards


def translator_reward_fn(completions, prompts, answer, input_text, **kwargs):
    """
    lora_translator 的 GRPO 獎勵函數（基於 BLEU 分數）

    使用 sacrebleu 計算生成譯文與參考答案的句子級 BLEU 分數。
    分數範圍：0.0 ~ 1.0（歸一化後）

    依賴：pip install sacrebleu
    若未安裝 sacrebleu 則 fallback 至 NLTK sentence_bleu。
    """
    try:
        import sacrebleu as sb
        _use_sacrebleu = True
    except ImportError:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            _use_sacrebleu = False
        except ImportError:
            # 無任何 BLEU 套件時，用長度比例粗估
            _use_sacrebleu = None

    rewards = []
    for completion, ref in zip(completions, answer):
        completion = completion.strip()
        ref        = ref.strip()

        if not completion or not ref:
            rewards.append(0.0)
            continue

        if _use_sacrebleu is True:
            bleu  = sb.sentence_bleu(completion, [ref])
            score = bleu.score / 100.0        # sacrebleu 回傳 0~100
        elif _use_sacrebleu is False:
            smoother = SmoothingFunction().method1
            score    = sentence_bleu(
                [ref.split()], completion.split(), smoothing_function=smoother
            )
        else:
            # 粗估：輸出長度與參考長度之比（≤1.0）
            score = min(len(completion.split()) / max(len(ref.split()), 1), 1.0)

        rewards.append(float(score))
    return rewards


REWARD_FUNCTIONS = {
    "reasoning":  reasoning_reward_fn,
    "analyst":    analyst_reward_fn,
    "translator": translator_reward_fn,
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="多任務 GRPO 強化學習訓練腳本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task", required=True,
        choices=list(TASK_PRESETS.keys()),
        help="訓練任務",
    )
    parser.add_argument("--rank",            type=int,   default=None, help="LoRA rank（覆蓋預設值）")
    parser.add_argument("--alpha",           type=int,   default=None, help="LoRA alpha（覆蓋預設值）")
    parser.add_argument("--lr",              type=float, default=None, help="Learning rate（覆蓋預設值）")
    parser.add_argument("--epochs",          type=int,   default=None, help="訓練 epoch 數（覆蓋預設值）")
    parser.add_argument("--num-generations", type=int,   default=None,
                        help="每個 prompt 生成的候選數（GRPO 群組大小）")
    parser.add_argument("--max-completion",  type=int,   default=None,
                        help="最大生成長度（token），影響 VRAM 用量")
    parser.add_argument("--kl-coeff",        type=float, default=None,
                        help="KL 散度懲罰係數，防止 reward hacking")
    parser.add_argument("--grad-accum",      type=int,   default=4,
                        help="gradient_accumulation_steps")
    parser.add_argument("--max-steps",       type=int,   default=-1,
                        help="最大訓練步數，-1 表示跑完全部 epoch")
    parser.add_argument("--seed",            type=int,   default=1234)
    parser.add_argument("--sft-adapter",     type=str,   default=None,
                        help="從 SFT adapter 繼續 GRPO 訓練（路徑，預設從 base model 開始）")
    return parser.parse_args()


# ── 主訓練流程 ────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    preset = dict(TASK_PRESETS[args.task])

    # CLI 覆蓋預設值
    if args.rank            is not None: preset["lora_rank"]             = args.rank
    if args.alpha           is not None: preset["lora_alpha"]            = args.alpha
    if args.lr              is not None: preset["learning_rate"]         = args.lr
    if args.epochs          is not None: preset["num_epochs"]            = args.epochs
    if args.num_generations is not None: preset["num_generations"]       = args.num_generations
    if args.max_completion  is not None: preset["max_completion_length"] = args.max_completion
    if args.kl_coeff        is not None: preset["kl_coeff"]              = args.kl_coeff

    sft_adapter  = args.sft_adapter
    model_source = sft_adapter if sft_adapter else MODEL_NAME
    ds_dir       = preset["dataset_dir"]
    out_dir      = preset["output_dir"]
    train_path   = os.path.join(ds_dir, f"{os.path.basename(ds_dir)}_train.jsonl")

    if not os.path.exists(train_path):
        print(f"[ERROR] 找不到訓練資料：{train_path}")
        print("請先執行 python prepare.py")
        sys.exit(1)

    print(f"[Task]     {args.task}  (mode=GRPO)")
    print(f"[Model]    {model_source}")
    print(f"[LoRA]     rank={preset['lora_rank']}  alpha={preset['lora_alpha']}")
    print(f"[GRPO]     num_gen={preset['num_generations']}  "
          f"max_completion={preset['max_completion_length']}  "
          f"kl={preset['kl_coeff']}")
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
    raw_dataset   = load_dataset("json", data_files=train_path, split="train")
    train_dataset = convert_to_grpo_dataset(raw_dataset, tokenizer)
    print(f"[Dataset]  {len(train_dataset)} 筆已轉換為 GRPO 格式")

    # ── GRPOConfig ────────────────────────────────────────────────────────────
    max_prompt_len = MAX_SEQ_LENGTH - preset["max_completion_length"]

    grpo_config = GRPOConfig(
        output_dir=out_dir,
        # 生成設定
        num_generations=preset["num_generations"],
        max_completion_length=preset["max_completion_length"],
        max_prompt_length=max_prompt_len,
        # 優化設定
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=preset["num_epochs"],
        max_steps=args.max_steps,
        learning_rate=preset["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        # KL 懲罰（防止 reward hacking）
        beta=preset["kl_coeff"],
        # 記錄與儲存
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        # 記憶體優化
        gradient_checkpointing=True,
    )

    reward_fn = REWARD_FUNCTIONS[args.task]

    # TRL >= 0.12 使用 processing_class；舊版用 tokenizer
    # 若出現 TypeError 請嘗試將 processing_class 改為 tokenizer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
    )

    # ── 訓練 ──────────────────────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats()
    t0           = time.time()
    trainer_stats = trainer.train()
    elapsed       = time.time() - t0
    peak_mem_gb   = torch.cuda.max_memory_allocated() / 1e9
    train_loss    = trainer_stats.training_loss

    # ── 儲存 adapter ──────────────────────────────────────────────────────────
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n[Done] RL LoRA adapter 已儲存至 {out_dir}")

    # ── 結構化 JSON 輸出（供自動化迴圈解析）────────────────────────────────────
    result = {
        "task":               args.task,
        "mode":               "grpo",
        "train_loss":         round(train_loss, 4) if train_loss is not None else None,
        "epochs":             preset["num_epochs"],
        "max_steps":          args.max_steps,
        "elapsed_sec":        round(elapsed, 1),
        "peak_memory_gb":     round(peak_mem_gb, 2),
        "rank":               preset["lora_rank"],
        "alpha":              preset["lora_alpha"],
        "lr":                 preset["learning_rate"],
        "num_generations":    preset["num_generations"],
        "max_completion":     preset["max_completion_length"],
        "kl_coeff":           preset["kl_coeff"],
        "grad_accum":         args.grad_accum,
        "sft_adapter":        sft_adapter or "none",
        "status":             "ok",
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
            ts, args.task, "grpo",
            str(preset["lora_rank"]),
            str(preset["lora_alpha"]),
            str(preset["learning_rate"]),
            str(preset["num_generations"]),
            str(preset["max_completion_length"]),
            str(preset["kl_coeff"]),
            str(preset["num_epochs"]),
            str(args.max_steps),
            str(result["train_loss"] or ""),
            str(result["peak_memory_gb"]),
            str(result["elapsed_sec"]),
            sft_adapter or "none",
            "ok",
        ]
        f.write("\t".join(row) + "\n")
    print(f"[TSV] 結果已附加至 {tsv_path}")


if __name__ == "__main__":
    main()
