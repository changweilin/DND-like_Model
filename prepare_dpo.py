"""
prepare_dpo.py — DPO 偏好對資料集生成

從現有 SFT 資料集自動構造 DPO 偏好對：
  chosen  = 原始高品質 GPT 回應（對應正確上下文）
  rejected = 來自不同 prompt 的 GPT 回應（脈絡錯位）

策略說明（脈絡錯位法）：
  對每個 prompt，從資料集的其他條目中隨機抽取一個回應作為 rejected。
  雖然不如人工標記精確，但足以教導模型：「回應必須與 prompt 內容相關」，
  並且可以完全自動化生成，適合冷啟動 DPO 訓練。

使用方式：
  python prepare_dpo.py

輸出：
  dataset/lora_storyteller_dpo/lora_storyteller_dpo.jsonl
  dataset/lora_storyteller_dpo/lora_storyteller_dpo_train.jsonl
  dataset/lora_storyteller_dpo/lora_storyteller_dpo_val.jsonl
"""

import json
import random
from pathlib import Path

# ── 常數 ──────────────────────────────────────────────────────────────────────

SEED        = 1234
SPLIT_RATIO = 0.9

# 合併 lora_storyteller 與 lora_storyteller_extra 兩個資料來源
SOURCES = {
    "storyteller":       "dataset/lora_storyteller/lora_storyteller.jsonl",
    "storyteller_extra": "dataset/lora_storyteller_extra/lora_storyteller_extra.jsonl",
}

OUTPUT_DIR  = "dataset/lora_storyteller_dpo"
OUTPUT_FILE = f"{OUTPUT_DIR}/lora_storyteller_dpo.jsonl"


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] {path}:{lineno} JSON 解析失敗：{e}")
    return items


def write_jsonl(path: str, items: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_dataset(items: list[dict], ratio: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * ratio)
    return shuffled[:cut], shuffled[cut:]


# ── DPO 偏好對構造 ────────────────────────────────────────────────────────────

def build_dpo_pairs(items: list[dict], seed: int) -> list[dict]:
    """
    對每筆資料構造一個 DPO 偏好對：
      - chosen  = 本筆的 gpt 回應（原本正確的故事接龍）
      - rejected = 隨機從另一筆抽取的 gpt 回應（脈絡不符）

    格式與 train_dpo.py 的 convert_to_dpo_dataset() 相容：
      {
        "conversations": [{"from": "system", ...}, {"from": "human", ...}],
        "chosen":   {"from": "gpt", "value": "..."},
        "rejected": {"from": "gpt", "value": "..."}
      }
    """
    rng = random.Random(seed)
    n   = len(items)

    # 預先提取所有 gpt 回應
    responses = []
    for item in items:
        convos = item.get("conversations", [])
        gpt_val = next((c["value"] for c in convos if c["from"] == "gpt"), "")
        responses.append(gpt_val)

    pairs = []
    skipped = 0
    for i, item in enumerate(items):
        convos  = item.get("conversations", [])
        chosen  = responses[i]

        # 去掉 gpt 回應，保留 system + human（作為 prompt）
        prompt_convos = [c for c in convos if c["from"] != "gpt"]

        if not chosen.strip() or not prompt_convos:
            skipped += 1
            continue

        # 從不同索引中隨機選 rejected
        candidates   = [j for j in range(n) if j != i and responses[j].strip()]
        if not candidates:
            skipped += 1
            continue
        rejected_idx = rng.choice(candidates)
        rejected     = responses[rejected_idx]

        pairs.append({
            "conversations": prompt_convos,
            "chosen":        {"from": "gpt", "value": chosen},
            "rejected":      {"from": "gpt", "value": rejected},
        })

    if skipped:
        print(f"  [WARN] 跳過 {skipped} 筆（缺少 gpt 回應或 prompt）")

    return pairs


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("prepare_dpo.py — DPO 偏好對資料集生成")
    print("=" * 60)

    # 讀取所有 storyteller 資料
    all_items = []
    for name, path in SOURCES.items():
        if not Path(path).exists():
            print(f"  [SKIP] {name}：找不到 {path}")
            continue
        items = read_jsonl(path)
        print(f"  讀取 {name}：{len(items)} 筆")
        all_items.extend(items)

    if not all_items:
        print("[ERROR] 找不到任何 storyteller 資料，請先執行 python prepare.py")
        return

    print(f"\n  合計：{len(all_items)} 筆原始資料")

    # 構造 DPO 偏好對
    pairs = build_dpo_pairs(all_items, seed=SEED)
    print(f"  生成 DPO 偏好對：{len(pairs)} 筆")

    # 儲存完整資料集
    write_jsonl(OUTPUT_FILE, pairs)
    print(f"  輸出：{OUTPUT_FILE}")

    # Train / Val 分割
    train, val = split_dataset(pairs, SPLIT_RATIO, SEED)
    train_path = f"{OUTPUT_DIR}/lora_storyteller_dpo_train.jsonl"
    val_path   = f"{OUTPUT_DIR}/lora_storyteller_dpo_val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    print(f"  分割：train={len(train)}, val={len(val)}")

    print("\n" + "=" * 60)
    print("DPO 資料準備完成！")
    print(f"  下一步：python train_dpo.py --task storyteller --sft-adapter outputs/lora_storyteller")
    print("=" * 60)


if __name__ == "__main__":
    main()
