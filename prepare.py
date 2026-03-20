"""
prepare.py — 資料準備與驗證

步驟：
  1a. 將 rpg_dataset + literature_dataset 從 instruction/output 轉為 ShareGPT 格式
      輸出至 dataset/lora_storyteller_extra/
  1b. 所有資料集 90/10 train/val 分割（seed=1234）
  1c. 驗證每筆格式、報告 token 長度分佈

使用方式：
  python prepare.py
"""

import json
import os
import random
from pathlib import Path

# ── 常數 ──────────────────────────────────────────────────────────────────────

SEED = 1234
SPLIT_RATIO = 0.9          # 90% train, 10% val
LONG_TOKEN_THRESHOLD = 1024  # 超過此數標記為長條目（估算：字元數 / 4）

FANTASY_NARRATOR_SOURCES = {
    "call_of_cthulhu", "critical_role", "dnd",
    "forgotten_realms", "dragonlance", "eberron",
    "iron_kingdoms", "l5r", "pathfinder",
    "warhammer_fantasy", "wh40k", "world_of_darkness",
    "elder_scrolls", "witcher", "dark_souls",
    "the_alexandrian", "starfinder", "dragonball",
    "shadowrun", "scp_foundation", "gurps",
    "bilibili_wiki_genshin", "harry_potter_es",
    "dragon_ball_es", "star_wars_fr", "one_piece_cn",
    "cyberpunk", "hameln_fanfic",
}
LITRPG_SOURCES = {"royalroad_litrpg"}

SYSTEM_PROMPTS = {
    "fantasy": (
        "You are a creative narrative writer specializing in fantasy RPG lore and worldbuilding. "
        "Given a prompt, topic, or story excerpt, write vivid, immersive prose that fits the setting and tone."
    ),
    "litrpg": (
        "You are a LitRPG author. Given an opening line or scene setup, "
        "continue the story with engaging prose that includes game-like elements and character voice."
    ),
}

DATASET = {
    "lora_storyteller":       "dataset/lora_storyteller/lora_storyteller.jsonl",
    "lora_storyteller_extra": None,   # generated in step 1a
    "lora_analyst":           "dataset/lora_analyst/lora_analyst.jsonl",
    "lora_translator":        "dataset/lora_translator/lora_translator.jsonl",
    "lora_reasoning":         "dataset/lora_reasoning/lora_reasoning.jsonl",
}

RAW_SOURCES = [
    "dataset/rpg_dataset/rpg_dataset.jsonl",
    "dataset/literature_dataset/literature_dataset.jsonl",
]
EXTRA_OUTPUT = "dataset/lora_storyteller_extra/lora_storyteller_extra.jsonl"

# ── 工具函式 ──────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """粗估 token 數（字元數 / 4）。"""
    return len(text) // 4


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


# ── Step 1a：格式轉換 ──────────────────────────────────────────────────────────

def get_system_prompt(source_id: str) -> str:
    if source_id in LITRPG_SOURCES:
        return SYSTEM_PROMPTS["litrpg"]
    return SYSTEM_PROMPTS["fantasy"]


def convert_to_sharegpt(item: dict) -> dict:
    """將 instruction/input/output 格式轉為 ShareGPT conversations 格式。"""
    instruction = item.get("instruction", "").strip()
    inp = item.get("input", "").strip()
    output = item.get("output", "").strip()
    source_id = item.get("metadata", {}).get("source_id", "unknown")

    human_value = instruction
    if inp:
        human_value = f"{instruction}\n\n{inp}"

    return {
        "conversations": [
            {"from": "system", "value": get_system_prompt(source_id)},
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": output},
        ]
    }


def step_1a_convert():
    print("\n=== Step 1a：格式轉換 ===")
    converted = []
    for src_path in RAW_SOURCES:
        items = read_jsonl(src_path)
        print(f"  讀取 {src_path}：{len(items)} 筆")
        for item in items:
            converted.append(convert_to_sharegpt(item))

    write_jsonl(EXTRA_OUTPUT, converted)
    print(f"  輸出 {EXTRA_OUTPUT}：{len(converted)} 筆")
    return converted


# ── Step 1b：Train/Val 分割 ────────────────────────────────────────────────────

def step_1b_split():
    print("\n=== Step 1b：Train/Val 分割（90/10, seed=1234）===")

    # 更新 DATASET 中的 extra 路徑
    dataset = dict(DATASET)
    dataset["lora_storyteller_extra"] = EXTRA_OUTPUT

    for name, src_path in dataset.items():
        if src_path is None or not Path(src_path).exists():
            print(f"  [SKIP] {name}：找不到 {src_path}")
            continue

        items = read_jsonl(src_path)
        train, val = split_dataset(items, SPLIT_RATIO, SEED)

        base_dir = f"dataset/{name}"
        train_path = f"{base_dir}/{name}_train.jsonl"
        val_path   = f"{base_dir}/{name}_val.jsonl"

        write_jsonl(train_path, train)
        write_jsonl(val_path, val)
        print(f"  {name}：{len(items)} 筆 → train={len(train)}, val={len(val)}")


# ── Step 1c：資料驗證 ──────────────────────────────────────────────────────────

def validate_dataset(name: str, path: str) -> dict:
    """驗證資料集格式，回傳統計資料。"""
    items = read_jsonl(path)
    total = len(items)
    errors = []
    long_count = 0
    token_lengths = []

    for i, item in enumerate(items):
        # 檢查 conversations 欄位
        if "conversations" not in item:
            errors.append(f"  筆 {i}: 缺少 conversations 欄位")
            continue

        convos = item["conversations"]
        has_human = any(c.get("from") == "human" for c in convos)
        has_gpt   = any(c.get("from") == "gpt"   for c in convos)

        if not has_human:
            errors.append(f"  筆 {i}: 缺少 human 對話")
        if not has_gpt:
            errors.append(f"  筆 {i}: 缺少 gpt 對話")

        # 估算 token 長度
        full_text = " ".join(c.get("value", "") for c in convos)
        tokens = estimate_tokens(full_text)
        token_lengths.append(tokens)
        if tokens > LONG_TOKEN_THRESHOLD:
            long_count += 1

    # 統計
    if token_lengths:
        sorted_tl = sorted(token_lengths)
        n = len(sorted_tl)
        stats = {
            "total": total,
            "valid": total - len(errors),
            "errors": len(errors),
            "long_entries": long_count,
            "token_min": sorted_tl[0],
            "token_p50": sorted_tl[n // 2],
            "token_p90": sorted_tl[int(n * 0.9)],
            "token_p99": sorted_tl[int(n * 0.99)],
            "token_max": sorted_tl[-1],
        }
    else:
        stats = {"total": 0, "valid": 0, "errors": 0}

    return stats, errors[:5]   # 只回傳前 5 個錯誤


def step_1c_validate():
    print("\n=== Step 1c：資料驗證 ===")

    dataset = dict(DATASET)
    dataset["lora_storyteller_extra"] = EXTRA_OUTPUT

    all_ok = True
    for name, src_path in dataset.items():
        # 優先驗證 train 分割
        train_path = f"dataset/{name}/{name}_train.jsonl"
        val_path   = f"dataset/{name}/{name}_val.jsonl"

        for label, path in [("train", train_path), ("val", val_path)]:
            if not Path(path).exists():
                print(f"  [SKIP] {name}/{label}：找不到 {path}")
                continue

            stats, sample_errors = validate_dataset(f"{name}/{label}", path)
            status = "OK" if stats["errors"] == 0 else "WARN"
            if stats["errors"] > 0:
                all_ok = False

            print(
                f"  [{status}] {name}/{label}: "
                f"{stats['total']} 筆, 錯誤={stats['errors']}, "
                f"超長={stats['long_entries']}, "
                f"tokens: p50={stats.get('token_p50','?')} "
                f"p90={stats.get('token_p90','?')} "
                f"max={stats.get('token_max','?')}"
            )
            for e in sample_errors:
                print(f"    {e}")

    return all_ok


# ── 主程式 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("prepare.py — LoRA 資料前處理")
    print("=" * 60)

    step_1a_convert()
    step_1b_split()
    ok = step_1c_validate()

    print("\n" + "=" * 60)
    if ok:
        print("資料準備完成，所有驗證通過。")
    else:
        print("資料準備完成，但有部分驗證警告，請檢查上方輸出。")
    print("=" * 60)
