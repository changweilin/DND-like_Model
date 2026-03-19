"""
evaluate.py — 訓練後評估腳本

載入已訓練的 LoRA adapter，在驗證集上計算 perplexity，
並依任務類型輸出次要品質檢查樣本。

使用方式：
  python evaluate.py --task analyst
  python evaluate.py --task storyteller --adapter-path outputs/lora_storyteller
  python evaluate.py --task translator --num-samples 5
"""

import argparse
import json
import math
import os
import sys

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ── 任務設定 ──────────────────────────────────────────────────────────────────

TASK_CONFIG = {
    "storyteller": {
        "adapter_path": "outputs/lora_storyteller",
        "val_path":     "dataset/lora_storyteller/lora_storyteller_val.jsonl",
    },
    "storyteller_extra": {
        "adapter_path": "outputs/lora_storyteller_extra",
        "val_path":     "dataset/lora_storyteller_extra/lora_storyteller_extra_val.jsonl",
    },
    "translator": {
        "adapter_path": "outputs/lora_translator",
        "val_path":     "dataset/lora_translator/lora_translator_val.jsonl",
    },
    "analyst": {
        "adapter_path": "outputs/lora_analyst",
        "val_path":     "dataset/lora_analyst/lora_analyst_val.jsonl",
    },
}

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 1024

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA adapter 評估腳本")
    parser.add_argument(
        "--task", required=True,
        choices=list(TASK_CONFIG.keys()),
        help="評估任務",
    )
    parser.add_argument(
        "--adapter-path", default=None,
        help="LoRA adapter 路徑（預設用任務對應的 outputs/lora_{task}）",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="次要品質檢查的生成筆數（預設：analyst=10, translator=10, storyteller=3）",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="生成時最大新 token 數（預設 256）",
    )
    return parser.parse_args()


# ── Perplexity 計算 ───────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, val_path: str) -> float:
    """在驗證集上計算平均 cross-entropy loss，回傳 perplexity。"""
    items = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"  驗證集筆數：{len(items)}")

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for item in items:
            convos = item.get("conversations", [])
            text = tokenizer.apply_chat_template(
                convos, tokenize=False, add_generation_prompt=False
            )
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LENGTH
            ).to(model.device)

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss.item()
            n_tokens = (labels != -100).sum().item()

            total_loss   += loss * n_tokens
            total_tokens += n_tokens

    avg_loss   = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# ── 次要品質檢查 ───────────────────────────────────────────────────────────────

def sample_items(val_path: str, n: int) -> list[dict]:
    items = []
    with open(val_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:n]


def get_human_prompt(convos: list[dict]) -> str:
    for c in convos:
        if c.get("from") == "human":
            return c.get("value", "")
    return ""


def get_system_prompt(convos: list[dict]) -> str:
    for c in convos:
        if c.get("from") == "system":
            return c.get("value", "")
    return ""


def generate_response(model, tokenizer, convos: list[dict], max_new_tokens: int) -> str:
    FastLanguageModel.for_inference(model)
    # 構建不含 gpt 回應的 prompt
    prompt_convos = [c for c in convos if c.get("from") != "gpt"]
    text = tokenizer.apply_chat_template(
        prompt_convos, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=MAX_SEQ_LENGTH).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def check_analyst(model, tokenizer, val_path: str, n: int, max_new_tokens: int) -> dict:
    """檢查 analyst 任務：JSON 解析成功率。"""
    items = sample_items(val_path, n)
    parsed_ok = 0
    samples = []

    for item in items:
        convos = item["conversations"]
        generated = generate_response(model, tokenizer, convos, max_new_tokens)
        try:
            json.loads(generated.strip())
            parsed_ok += 1
            parse_status = "ok"
        except json.JSONDecodeError:
            parse_status = "fail"
        samples.append({
            "prompt": get_human_prompt(convos)[:100],
            "generated_preview": generated[:200],
            "json_parse": parse_status,
        })

    return {
        "json_parse_success_rate": parsed_ok / len(items) if items else 0,
        "samples": samples,
    }


def check_translator(model, tokenizer, val_path: str, n: int, max_new_tokens: int) -> dict:
    """生成翻譯對照樣本。"""
    items = sample_items(val_path, n)
    samples = []

    for item in items:
        convos = item["conversations"]
        generated = generate_response(model, tokenizer, convos, max_new_tokens)
        ref = next((c["value"] for c in convos if c.get("from") == "gpt"), "")
        samples.append({
            "prompt":    get_human_prompt(convos)[:120],
            "reference": ref[:200],
            "generated": generated[:200],
        })

    return {"samples": samples}


def check_storyteller(model, tokenizer, val_path: str, n: int, max_new_tokens: int) -> dict:
    """生成故事接龍範例。"""
    items = sample_items(val_path, n)
    samples = []

    for item in items:
        convos = item["conversations"]
        generated = generate_response(model, tokenizer, convos, max_new_tokens)
        samples.append({
            "prompt":    get_human_prompt(convos)[:120],
            "generated": generated[:400],
        })

    return {"samples": samples}


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = TASK_CONFIG[args.task]

    adapter_path = args.adapter_path or cfg["adapter_path"]
    val_path     = cfg["val_path"]

    if not os.path.exists(adapter_path):
        print(f"[ERROR] 找不到 adapter：{adapter_path}")
        print("請先執行 python train_lora.py --task {args.task}")
        sys.exit(1)

    if not os.path.exists(val_path):
        print(f"[ERROR] 找不到驗證集：{val_path}")
        print("請先執行 python prepare.py")
        sys.exit(1)

    print(f"[Task]    {args.task}")
    print(f"[Adapter] {adapter_path}")
    print(f"[Val]     {val_path}")

    # ── 載入模型 + adapter ────────────────────────────────────────────────────
    print("\n載入基底模型與 LoRA adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # ── Perplexity ────────────────────────────────────────────────────────────
    print("\n計算 Perplexity...")
    avg_loss, perplexity = compute_perplexity(model, tokenizer, val_path)
    print(f"  avg_loss={avg_loss:.4f}  perplexity={perplexity:.2f}")

    # ── 次要品質檢查 ───────────────────────────────────────────────────────────
    quality_check = {}
    default_samples = {"analyst": 10, "translator": 10,
                       "storyteller": 3, "storyteller_extra": 3}
    n = args.num_samples or default_samples.get(args.task, 5)

    print(f"\n次要品質檢查（{n} 筆）...")
    if args.task == "analyst":
        quality_check = check_analyst(model, tokenizer, val_path, n, args.max_new_tokens)
        print(f"  JSON 解析成功率：{quality_check['json_parse_success_rate']*100:.1f}%")
    elif args.task == "translator":
        quality_check = check_translator(model, tokenizer, val_path, n, args.max_new_tokens)
        print(f"  已生成 {len(quality_check['samples'])} 筆翻譯對照")
    elif args.task in ("storyteller", "storyteller_extra"):
        quality_check = check_storyteller(model, tokenizer, val_path, n, args.max_new_tokens)
        print(f"  已生成 {len(quality_check['samples'])} 筆接龍範例")

    # ── 輸出報告 ──────────────────────────────────────────────────────────────
    report = {
        "task":        args.task,
        "adapter":     adapter_path,
        "val_path":    val_path,
        "avg_loss":    round(avg_loss, 4),
        "perplexity":  round(perplexity, 2),
        "quality":     quality_check,
    }

    os.makedirs(adapter_path, exist_ok=True)
    report_path = os.path.join(adapter_path, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Report] 已儲存至 {report_path}")

    # 純文字摘要
    txt_path = os.path.join(adapter_path, "eval_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Task:       {args.task}\n")
        f.write(f"Adapter:    {adapter_path}\n")
        f.write(f"Val set:    {val_path}\n")
        f.write(f"avg_loss:   {avg_loss:.4f}\n")
        f.write(f"Perplexity: {perplexity:.2f}\n")
        if args.task == "analyst" and "json_parse_success_rate" in quality_check:
            rate = quality_check["json_parse_success_rate"] * 100
            f.write(f"JSON parse: {rate:.1f}%\n")
        if "samples" in quality_check:
            f.write(f"\n--- Samples ({n}) ---\n")
            for i, s in enumerate(quality_check["samples"], 1):
                f.write(f"\n[{i}] Prompt: {s.get('prompt','')}\n")
                if "reference" in s:
                    f.write(f"    Ref:     {s['reference']}\n")
                f.write(f"    Generated: {s.get('generated','')}\n")
    print(f"[Report] 純文字摘要已儲存至 {txt_path}")

    # 結構化摘要行
    print("\n[EVAL_JSON]")
    print(json.dumps({
        "task": args.task,
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
        "status": "ok",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
