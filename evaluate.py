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
import re
import sys

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ── 任務設定 ──────────────────────────────────────────────────────────────────
# adapter_path_rl: GRPO 訓練後的 adapter 路徑（可選）

TASK_CONFIG = {
    "storyteller": {
        "adapter_path":     "outputs/lora_storyteller",
        "adapter_path_dpo": "outputs/lora_storyteller_dpo",
        "val_path":         "dataset/lora_storyteller/lora_storyteller_val.jsonl",
    },
    "storyteller_extra": {
        "adapter_path":    "outputs/lora_storyteller_extra",
        "val_path":        "dataset/lora_storyteller_extra/lora_storyteller_extra_val.jsonl",
    },
    "translator": {
        "adapter_path":    "outputs/lora_translator",
        "adapter_path_rl": "outputs/lora_translator_grpo",
        "val_path":        "dataset/lora_translator/lora_translator_val.jsonl",
    },
    "analyst": {
        "adapter_path":    "outputs/lora_analyst",
        "adapter_path_rl": "outputs/lora_analyst_grpo",
        "val_path":        "dataset/lora_analyst/lora_analyst_val.jsonl",
    },
    "reasoning": {
        "adapter_path":    "outputs/lora_reasoning",
        "adapter_path_rl": "outputs/lora_reasoning_grpo",
        "val_path":        "dataset/lora_reasoning/lora_reasoning_val.jsonl",
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
        "--rl", action="store_true",
        help="評估 GRPO 訓練後的 RL adapter（outputs/lora_{task}_grpo）",
    )
    parser.add_argument(
        "--dpo", action="store_true",
        help="評估 DPO 訓練後的 adapter（outputs/lora_{task}_dpo，僅支援 storyteller）",
    )
    parser.add_argument(
        "--base-model", action="store_true",
        help="直接評估 base model（不載入任何 adapter），用於測試 zero-shot 基準",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="次要品質檢查的生成筆數（預設：analyst=10, translator=10, storyteller=3）",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="生成時最大新 token 數（預設 256）",
    )
    parser.add_argument(
        "--llm-judge", action="store_true",
        help="使用 Gemini 作為 LLM Judge 進行自動化文風與翻譯品質評估",
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
    """
    檢查 analyst 任務：JSON 解析成功率 + 實體定位率。
    實體定位率：提取出的實體中，出現在原始輸入文本的比率（越高越好，代表越少幻覺）。
    """
    items     = sample_items(val_path, n)
    parsed_ok = 0
    grounded_total   = 0
    entity_total     = 0
    samples   = []

    for item in items:
        convos     = item["conversations"]
        inp_text   = get_human_prompt(convos).lower()
        generated  = generate_response(model, tokenizer, convos, max_new_tokens)

        parse_status = "fail"
        entity_grounding = None

        try:
            result = json.loads(generated.strip())
            parsed_ok   += 1
            parse_status = "ok"

            # 計算實體定位率
            entities: list = []
            if isinstance(result.get("角色"), list):
                entities += result["角色"]
            if isinstance(result.get("組織"), list):
                entities += result["組織"]

            if entities:
                grounded = sum(
                    1 for e in entities
                    if isinstance(e, str) and e.lower() in inp_text
                )
                entity_grounding = grounded / len(entities)
                grounded_total  += grounded
                entity_total    += len(entities)

        except (json.JSONDecodeError, ValueError):
            pass

        samples.append({
            "prompt":            get_human_prompt(convos)[:100],
            "generated_preview": generated[:200],
            "json_parse":        parse_status,
            "entity_grounding":  entity_grounding,
        })

    overall_grounding = grounded_total / entity_total if entity_total > 0 else None

    return {
        "json_parse_success_rate": parsed_ok / len(items) if items else 0,
        "entity_grounding_rate":   overall_grounding,
        "samples":                 samples,
    }


def _extract_json_from_conclusion(text: str) -> dict | None:
    """從推理任務的【結論】區塊提取 JSON 物件。"""
    match = re.search(r"【結論】\s*(\{.*?\})", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_init_affinity(text: str) -> int | None:
    """從 human 輸入中提取初始好感度數值。"""
    match = re.search(r'"[^"]*好感度"\s*:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def check_reasoning(model, tokenizer, val_path: str, n: int, max_new_tokens: int) -> dict:
    """
    檢查 reasoning 任務的 RL 關鍵指標：
    - format_compliance_rate  : 含【推理步驟】與【結論】兩個區塊的比率
    - json_validity_rate      : 【結論】內 JSON 可解析的比率
    - arithmetic_accuracy     : 數值計算正確的比率（含好感度欄位的樣本中）
    """
    items = sample_items(val_path, n)

    format_ok    = 0
    json_ok      = 0
    arith_total  = 0
    arith_ok     = 0
    samples      = []

    for item in items:
        convos    = item["conversations"]
        inp_text  = get_human_prompt(convos)
        generated = generate_response(model, tokenizer, convos, max_new_tokens)

        has_steps      = "【推理步驟】" in generated
        has_conclusion = "【結論】"     in generated
        fmt_ok = has_steps and has_conclusion
        if fmt_ok:
            format_ok += 1

        result_json  = _extract_json_from_conclusion(generated)
        json_valid   = result_json is not None
        if json_valid:
            json_ok += 1

        arith_correct = None
        if json_valid:
            delta   = result_json.get("好感度增量")
            new_aff = result_json.get("新好感度")
            if delta is not None and new_aff is not None:
                init_aff = _extract_init_affinity(inp_text)
                if init_aff is not None:
                    arith_total += 1
                    try:
                        if abs((int(init_aff) + int(delta)) - int(new_aff)) <= 1:
                            arith_ok     += 1
                            arith_correct = True
                        else:
                            arith_correct = False
                    except (TypeError, ValueError):
                        arith_correct = False

        samples.append({
            "prompt":         inp_text[:120],
            "generated":      generated[:400],
            "format_ok":      fmt_ok,
            "json_valid":     json_valid,
            "arith_correct":  arith_correct,
        })

    return {
        "format_compliance_rate": format_ok   / len(items) if items else 0,
        "json_validity_rate":     json_ok     / len(items) if items else 0,
        "arithmetic_accuracy":    arith_ok    / arith_total if arith_total > 0 else None,
        "samples":                samples,
    }


def check_translator(model, tokenizer, val_path: str, n: int, max_new_tokens: int) -> dict:
    """生成翻譯對照樣本，並計算句子級 BLEU 分數。"""
    try:
        import sacrebleu as sb
        _bleu_mode = "sacrebleu"
    except ImportError:
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            _bleu_mode = "nltk"
        except ImportError:
            _bleu_mode = None

    items   = sample_items(val_path, n)
    samples = []
    bleu_scores = []

    for item in items:
        convos    = item["conversations"]
        generated = generate_response(model, tokenizer, convos, max_new_tokens)
        ref       = next((c["value"] for c in convos if c.get("from") == "gpt"), "")

        bleu = None
        if _bleu_mode == "sacrebleu" and ref.strip():
            bleu = sb.sentence_bleu(generated.strip(), [ref.strip()]).score
            bleu_scores.append(bleu)
        elif _bleu_mode == "nltk" and ref.strip():
            smoother = SmoothingFunction().method1
            bleu = sentence_bleu(
                [ref.split()], generated.split(), smoothing_function=smoother
            ) * 100
            bleu_scores.append(bleu)

        samples.append({
            "prompt":    get_human_prompt(convos)[:120],
            "reference": ref[:200],
            "generated": generated[:200],
            "bleu":      round(bleu, 2) if bleu is not None else None,
        })

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None
    return {
        "samples":  samples,
        "avg_bleu": round(avg_bleu, 2) if avg_bleu is not None else None,
    }


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


# ── LLM Judge ─────────────────────────────────────────────────────────────────

def run_llm_judge(samples: list[dict], task: str) -> dict:
    if genai is None:
        print("[WARN] 未安裝 google-generativeai，略過 LLM Judge。請執行 pip install google-generativeai")
        return {"error": "Module not installed"}

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[WARN] 未設定 GEMINI_API_KEY，略過 LLM Judge。")
        return {"error": "No API Key"}

    genai.configure(api_key=api_key)
    # 使用 2.5 flash 作為高CP值的 judge
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = ""
    if task in ("storyteller", "storyteller_extra"):
        prompt = "你是一位專業的 TRPG 遊戲文字敘事評審。\n"
        prompt += "請根據以下三個面向為每個故事接龍樣本評分，每個面向 0~5 分，總分滿分 15 分：文學性(生動描寫與氛圍)、連貫性(與前文脈絡相符)、角色聲音(對話符合個性)。\n"
        prompt += "請在 'comment' 中給予一句短評，最後在 'overall' 給予整體建議。\n"
        prompt += "請務必回傳 JSON 格式如下：\n"
        prompt += '{"samples": [{"score": 12, "comment": "氛圍佳但結尾倉促"}], "overall": "整體建議..."}\n\n'
        for i, s in enumerate(samples):
            prompt += f"【樣本 {i+1}】\nPrompt: {s['prompt']}\nGenerated: {s['generated']}\n\n"
    elif task == "translator":
        prompt = "你是一位專業的 TRPG 遊戲本地化翻譯評審。\n"
        prompt += "請根據以下兩個面向為每個翻譯樣本評分，每個面向 0~5 分，總分滿分 10 分：忠實度(語意正確傳達)、流暢度(目標語言自然)。\n"
        prompt += "請在 'comment' 中給予一句短評，最後在 'overall' 給予整體建議。\n"
        prompt += "請務必回傳 JSON 格式如下：\n"
        prompt += '{"samples": [{"score": 9, "comment": "翻譯流暢且精確"}], "overall": "整體建議..."}\n\n'
        for i, s in enumerate(samples):
            prompt += f"【樣本 {i+1}】\nRef(參考): {s.get('reference', '')}\nGenerated(生成): {s['generated']}\n\n"
    else:
        return {}

    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"[WARN] LLM Judge 呼叫或解析失敗: {e}")
        return {"error": str(e)}


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = TASK_CONFIG[args.task]

    # 決定載入來源與模式標籤
    if args.base_model:
        adapter_path = MODEL_NAME
        mode_tag     = "base-model(zero-shot)"
    elif args.adapter_path:
        adapter_path = args.adapter_path
        mode_tag     = "custom"
    elif args.dpo and cfg.get("adapter_path_dpo"):
        adapter_path = cfg["adapter_path_dpo"]
        mode_tag     = "RL(DPO)"
    elif args.rl and cfg.get("adapter_path_rl"):
        adapter_path = cfg["adapter_path_rl"]
        mode_tag     = "RL(GRPO)"
    else:
        adapter_path = cfg["adapter_path"]
        mode_tag     = "SFT"

    val_path = cfg["val_path"]

    if not args.base_model and not os.path.exists(adapter_path):
        print(f"[ERROR] 找不到 adapter：{adapter_path}")
        if args.dpo:
            print(f"請先執行 python train_dpo.py --task {args.task}")
        elif args.rl:
            print(f"請先執行 python train_grpo.py --task {args.task}")
        else:
            print(f"請先執行 python train_lora.py --task {args.task}")
        sys.exit(1)

    if not os.path.exists(val_path):
        print(f"[ERROR] 找不到驗證集：{val_path}")
        print("請先執行 python prepare.py")
        sys.exit(1)

    print(f"[Task]    {args.task}  ({mode_tag})")
    print(f"[Model]   {adapter_path}")
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
    default_samples = {
        "analyst":          10,
        "reasoning":        10,
        "translator":       10,
        "storyteller":       3,
        "storyteller_extra": 3,
    }
    n = args.num_samples or default_samples.get(args.task, 5)

    print(f"\n次要品質檢查（{n} 筆）...")
    if args.task == "analyst":
        quality_check = check_analyst(model, tokenizer, val_path, n, args.max_new_tokens)
        print(f"  JSON 解析成功率：{quality_check['json_parse_success_rate']*100:.1f}%")
        if quality_check.get("entity_grounding_rate") is not None:
            print(f"  實體定位率：    {quality_check['entity_grounding_rate']*100:.1f}%")
    elif args.task == "reasoning":
        quality_check = check_reasoning(model, tokenizer, val_path, n, args.max_new_tokens)
        print(f"  格式合規率：    {quality_check['format_compliance_rate']*100:.1f}%")
        print(f"  JSON 有效率：   {quality_check['json_validity_rate']*100:.1f}%")
        if quality_check.get("arithmetic_accuracy") is not None:
            print(f"  數值計算正確率：{quality_check['arithmetic_accuracy']*100:.1f}%")
    elif args.task == "translator":
        quality_check = check_translator(model, tokenizer, val_path, n, args.max_new_tokens)
        print(f"  已生成 {len(quality_check['samples'])} 筆翻譯對照")
        if quality_check.get("avg_bleu") is not None:
            print(f"  平均 BLEU：       {quality_check['avg_bleu']:.2f}")
    elif args.task in ("storyteller", "storyteller_extra"):
        quality_check = check_storyteller(model, tokenizer, val_path, n, args.max_new_tokens)
        print(f"  已生成 {len(quality_check['samples'])} 筆接龍範例")

    if args.llm_judge and args.task in ("storyteller", "storyteller_extra", "translator"):
        print("\n執行 LLM Judge (Gemini) 評估...")
        judge_result = run_llm_judge(quality_check.get("samples", []), args.task)
        quality_check["llm_judge"] = judge_result
        if "overall" in judge_result:
            print(f"  [LLM Judge] 整體建議：{judge_result['overall']}")

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

        if "llm_judge" in quality_check and "samples" in quality_check["llm_judge"]:
            f.write(f"\n--- LLM Judge (Gemini) ---\n")
            f.write(f"整體建議: {quality_check['llm_judge'].get('overall', '')}\n")
            for i, res in enumerate(quality_check["llm_judge"]["samples"], 1):
                f.write(f"[{i}] Score: {res.get('score')} | Comment: {res.get('comment')}\n")

    print(f"[Report] 純文字摘要已儲存至 {txt_path}")

    # 結構化摘要行
    eval_summary = {
        "task":       args.task,
        "mode":       mode_tag,
        "adapter":    adapter_path,
        "avg_loss":   round(avg_loss, 4),
        "perplexity": round(perplexity, 2),
        "status":     "ok",
    }
    # 附加 RL 指標（若有）
    if args.task == "analyst":
        eval_summary["json_parse_success_rate"] = round(
            quality_check.get("json_parse_success_rate", 0), 4)
        if quality_check.get("entity_grounding_rate") is not None:
            eval_summary["entity_grounding_rate"] = round(
                quality_check["entity_grounding_rate"], 4)
    elif args.task == "reasoning":
        eval_summary["format_compliance_rate"] = round(
            quality_check.get("format_compliance_rate", 0), 4)
        eval_summary["json_validity_rate"] = round(
            quality_check.get("json_validity_rate", 0), 4)
        if quality_check.get("arithmetic_accuracy") is not None:
            eval_summary["arithmetic_accuracy"] = round(
                quality_check["arithmetic_accuracy"], 4)
    elif args.task == "translator":
        if quality_check.get("avg_bleu") is not None:
            eval_summary["avg_bleu"] = quality_check["avg_bleu"]

    if "llm_judge" in quality_check and "samples" in quality_check["llm_judge"]:
        scores = [s.get("score", 0) for s in quality_check["llm_judge"].get("samples", []) if isinstance(s.get("score"), (int, float))]
        if scores:
            eval_summary["llm_judge_avg_score"] = round(sum(scores) / len(scores), 2)

    print("\n[EVAL_JSON]")
    print(json.dumps(eval_summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
