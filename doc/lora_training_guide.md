# 文本遊戲多重 LoRA 訓練指南 (基於 RTX 3060 12GB 環境)

採用方案 B「單一強大 Base Model + 動態掛載多重 LoRA」是目前在消費級顯示卡上最有效率的做法。以下針對指定模型與任務，為您整理詳細的訓練超參數與資料集準備規格說明。

## 1. 模型與參數規模適配性分析 (RTX 3060 12GB)

要在 12GB VRAM 的環境中進行 **QLoRA (4-bit 量化微調)**，顯存的管理非常嚴格：

| 參數規模 | 推薦模型選項 | 3060 12GB 訓練可行性分析 |
| :--- | :--- | :--- |
| **8B (7B-9B)** | • **Qwen 2.5 (7B)**<br>• **Llama 3.1 (8B)**<br>• **Breeze 3 (8B)**<br>• **DeepSeek-R1-Distill (7B/8B)** | **🏆 最佳甜點區。**<br>使用 Unsloth 等框架，佔用約 7~9GB VRAM，可順暢設定 4096 甚至 8192 Token 的上下文訓練。 |
| **14B** | • **Qwen 2.5 (14B)**<br>• **DeepSeek-R1-Distill (14B)** | **⚠️ 極限挑戰。**<br>模型檔佔用近 9GB，需強烈依靠 Gradient Checkpointing 技術，訓練輸入文本極度受限 (需小於 512-1024 Token)。 |
| **32B** | • **Qwen 2.5 (32B)**<br>• **DeepSeek-R1-Distill (32B)** | **❌ 無法在單卡 12GB 訓練。**<br>建議租用雲端 (如 RunPod) 單卡 RTX 4090 或 A6000 訓練完 LoRA 後，再將小型的 LoRA 檔與 4-bit Base Model 載入 3060 進行「推理」。 |

> **開發建議**：對於文本遊戲的地端核心，**強烈推薦以 8B 級別 (如 Qwen 2.5 7B 或 Breeze-3-8B) 作為 Base Model**。DeepSeek-R1 系列特色為「自帶思維鏈 (CoT)」，如果遊戲不需看見 AI 長篇大論的思考過程，建議直接使用標準版的 Qwen 2.5 或 Llama 3.1。

---

## 2. 訓練資料集格式與需求

無論是哪一種 LoRA 任務，都建議採用標準的 **ShareGPT (JSONL)** 格式。
一包資料（單筆資料）的長度建議控制在 **512 ~ 1024 Tokens** 內，以保證 3060 不會 OOM (Out of Memory)。

### 標準對話格式範例 (ShareGPT Format)
```json
{"conversations": [
  {"from": "system", "value": "你是一個遊戲後台的實體抓取器。請分析文本中的角色名與組織。"},
  {"from": "human", "value": "艾莉絲拔出長劍，衝向了黑龍騎士團的陣型中。"},
  {"from": "gpt", "value": "{\"角色\": [\"艾莉絲\"], \"組織\": [\"黑龍騎士團\"]}"}
]}
```
* **一包資料大小**：通常一份 LoRA 訓練集對應一個 `dataset.jsonl`，該檔案大小約在 **1MB 到 10MB** 之間。

---

## 3. 各任務的 LoRA 超參數與資料需求設定大全

針對您文本遊戲內的五個不同需求，在訓練 LoRA 時，設定的走向完全不同。

*註解：以下參數基於 Unsloth / PEFT 架構。為適應 3060 12GB，所有任務一律設定 `Per-device Train Batch Size = 1 或 2`，並透過 `Gradient Accumulation Steps = 4 到 8` 來達到等效的全局批次大小。*

### A. 關鍵字故事接龍生成 (Story Continuation)
**🎯 目標：** 對齊特定的奇幻/科幻文風，學習角色的說話口吻。
* **需求資料量**：2,000 ~ 5,000 筆 (高品質的遊戲文本劇本)
* **超參數設定**：
  * **LoRA Rank (r)**：`32` 或 `64` (需要較大的空間記憶文風和寫作習慣)
  * **LoRA Alpha**：`r 的一倍或兩倍 (32 或 64)`
  * **Learning Rate (LR)**：`2e-5` (較小的學習率，避免破壞原模型優秀的文法基礎，只微調「風格」)
  * **Epochs**：`3 ~ 5` (風格對齊需要多次迭代)

### B. 翻譯 (Translation)
**🎯 目標：** 針對遊戲世界觀的「專有名詞」與語氣進行精準翻譯。
* **需求資料量**：3,000 ~ 10,000 筆 (對齊的雙語劇本/詞彙對照)
* **超參數設定**：
  * **LoRA Rank (r)**：`64` (需要高 Rank 來記憶大量字彙跨語言的映射關係)
  * **LoRA Alpha**：`128`
  * **Learning Rate (LR)**：`1e-4` 或 `2e-4`
  * **Epochs**：`2 ~ 3` (訓練過多次容易引發過擬合，導致只會照抄翻譯表)

### C. 語意推理 (Semantic Reasoning)
**🎯 目標：** 根據玩家的行為，推斷出隱藏任務是否觸發，或是 NPC 的好感度變化。
* **需求資料量**：1,000 ~ 3,000 筆 (提供完整的「輸入條件 -> 思考過程 -> 結論」標準答案)
* **超參數設定**：
  * **LoRA Rank (r)**：`64` 或 `128` (邏輯轉折最複雜，需要最多的專家權重)
  * **LoRA Alpha**：`r 的兩倍`
  * **Learning Rate (LR)**：`1e-4` 
  * **Epochs**：`3 ~ 4`

### D. 人名與組織抓取 (NER / Entity Extraction)
**🎯 目標：** 閱讀玩家輸入，精準提取文字並強制輸出標準 JSON。
* **需求資料量**：500 ~ 1,500 筆 (任務類型單純，主要訓練 JSON 格式穩定性)
* **超參數設定**：
  * **LoRA Rank (r)**：`8` 或 `16` (任務極度單純，使用小 Rank 即可，模型切換速度極快)
  * **LoRA Alpha**：`16` 或 `32`
  * **Learning Rate (LR)**：`3e-4` (學習率可稍高，因為目標很明確)
  * **Epochs**：`2 ~ 3` 

### E. 情緒判讀 (Sentiment Classification)
**🎯 目標：** 將文本分類為特定幾種標籤（如：憤怒、悲傷、喜悅）。
* **需求資料量**：300 ~ 800 筆 (分類任務是 LLM 最簡單的任務)
* **超參數設定**：
  * **LoRA Rank (r)**：`8` (最小的體積)
  * **LoRA Alpha**：`16`
  * **Learning Rate (LR)**：`3e-4` 或 `5e-4`
  * **Epochs**：`2` 

---

## 4. 防坑建議與實作流程

1. **合併或獨立？**
   * 「接龍」與「翻譯」因為牽涉底層語感，建議**獨立**成不同的 LoRA。
   * 「抓取」與「情緒判讀」因為任務重疊度極高（都屬於後台數據結構化任務），強烈建議將這兩者的訓練資料**合併訓練成同一個 `後台分析_LoRA`**。這樣不僅省事，也能提高 VRAM 掛載效率。
2. **LoRA Target Modules**：在設定訓練腳本時，請確保掛載 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` 全線性層部位，這樣訓練出來的 LoRA 效果最好（稍增 VRAM 佔用但十分值得）。
3. **動態載入**：在伺服器端（推薦使用 `vLLM` 的 Multi-LoRA 功能），預先載入 Base Model (如 Qwen2.5-7B)，並向 API Request 的 `model` 欄位送出想啟用的 LoRA 名稱（如 `lora-storyteller` 或 `lora-analyst`），即可實現零延遲並發執行。

---

## 5. LoRA 強化學習訓練計畫 (針對 DND-like RPG)

> **背景**：SFT（監督式微調）教模型「怎麼回答」，但強化學習讓模型學習「什麼回答更好」。
> 針對 DND-like RPG 的特性，可驗證的遊戲邏輯（數值計算、JSON 格式、實體定位）非常適合無獎勵模型的規則式 RL。

### 5.1 各任務適合的 RL 方法一覽

| LoRA 任務 | 推薦 RL 方法 | 原因 | 優先級 |
|---|---|---|---|
| **lora_reasoning** (語意推理) | **GRPO** (規則式獎勵) | 輸出可驗證（數值計算、JSON格式），無需 Reward Model | ★★★ 最高 |
| **lora_analyst** (NER 抓取) | **GRPO** (規則式獎勵) | JSON 格式與實體定位可程式化驗證 | ★★★ 最高 |
| **lora_storyteller** (故事接龍) | **DPO** (直接偏好優化) | 需要品質判斷，DPO 可從現有資料構造偏好對 | ★★ 中 |
| **lora_translator** (翻譯) | **GRPO** (BLEU 獎勵) | 可用 BLEU/ROUGE 對照參考答案計分，但截斷問題需先解決 | ★ 低（待資料修正） |

---

### 5.2 Phase 1：GRPO × lora_reasoning（最高優先）

**核心思想**：推理任務的輸出完全可驗證。模型必須：
1. 按格式輸出 `【推理步驟】` 和 `【結論】` 兩個區塊
2. `【結論】` 中的 JSON 數值必須與步驟計算一致
3. 數字計算必須正確（如好感度加減）

#### 獎勵函數設計（Rule-based Reward，無需 Reward Model）

```python
def reasoning_reward_fn(completions, prompts, **kwargs):
    """
    針對 lora_reasoning 的 GRPO 獎勵函數
    最高分: 4.0，最低分: 0.0
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        score = 0.0

        # R1: 格式完整性 (+1.0)
        has_steps = "【推理步驟】" in completion
        has_conclusion = "【結論】" in completion
        if has_steps and has_conclusion:
            score += 1.0
        elif has_steps or has_conclusion:
            score += 0.3  # 部分格式

        # R2: JSON 可解析性 (+1.0)
        conclusion_match = re.search(r"【結論】\s*(\{.*?\})", completion, re.DOTALL)
        if conclusion_match:
            try:
                result_json = json.loads(conclusion_match.group(1))
                score += 1.0

                # R3: JSON 欄位完整性 (+1.0)
                expected_keys = {"好感度增量", "新好感度"}
                if expected_keys.issubset(result_json.keys()):
                    score += 1.0

                    # R4: 數值計算正確性 (+1.0)
                    # 從 prompt 中提取初始好感度，驗算
                    init_affinity = extract_initial_affinity(prompt)
                    delta = result_json.get("好感度增量", 0)
                    new_affinity = result_json.get("新好感度", 0)
                    if init_affinity is not None:
                        if abs((init_affinity + delta) - new_affinity) <= 1:
                            score += 1.0

            except json.JSONDecodeError:
                pass  # JSON 無法解析，不加分

        rewards.append(score)
    return rewards
```

#### GRPO 訓練超參數（RTX 3060 12GB）

```python
# train_grpo_reasoning.py
from trl import GRPOTrainer, GRPOConfig

grpo_config = GRPOConfig(
    output_dir="outputs/lora_reasoning_grpo",
    # --- 核心 GRPO 設定 ---
    num_generations=4,          # 每個 prompt 生成幾個候選答案（群組大小）
    max_completion_length=512,  # 推理輸出長度上限
    # --- 優化設定 ---
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 等效 batch_size=8
    learning_rate=5e-6,             # RL 需要比 SFT 更小的 LR
    num_train_epochs=2,
    # --- 記憶體優化 ---
    gradient_checkpointing=True,
    bf16=True,
    # --- KL 散度懲罰（防止 reward hacking）---
    kl_coeff=0.1,               # 不能偏離 SFT 基準太遠
    # --- LoRA 設定（從已訓練的 SFT LoRA 繼續）---
    # 載入 lora_reasoning SFT checkpoint 再進行 GRPO
)
```

**訓練流程**：
```
SFT lora_reasoning  →  GRPO lora_reasoning  →  評估（格式正確率 + 數值準確率）
    (已有資料)          (加入獎勵訊號)          (compare vs SFT baseline)
```

---

### 5.3 Phase 2：GRPO × lora_analyst（JSON 定位獎勵）

NER 任務的核心問題是：**模型容易幻覺出不在原文中的實體名稱**（現有資料已知有品質問題）。
GRPO 可以透過「定位獎勵」強制模型只提取原文有出現的名稱。

#### 獎勵函數設計

```python
def analyst_reward_fn(completions, prompts, **kwargs):
    """
    針對 lora_analyst 的 GRPO 獎勵函數
    重點：實體必須在原文中找到（防止幻覺）
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        score = 0.0
        input_text = extract_human_turn(prompt).lower()

        # R1: 是合法 JSON (+1.0)
        try:
            result = json.loads(completion.strip())
            score += 1.0
        except:
            rewards.append(0.0)
            continue

        # R2: 有正確的 key 結構 (+0.5)
        if "角色" in result and "組織" in result:
            score += 0.5

        # R3: 實體定位獎勵 — 每個實體出現在原文 (+0.25 each, max +2.0)
        all_entities = result.get("角色", []) + result.get("組織", [])
        grounded = sum(1 for e in all_entities if e.lower() in input_text)
        score += min(grounded * 0.25, 2.0)

        # R4: 幻覺懲罰 — 實體不在原文 (-0.5 each)
        hallucinated = sum(1 for e in all_entities if e.lower() not in input_text)
        score -= hallucinated * 0.5
        score = max(score, 0.0)  # 不低於 0

        rewards.append(score)
    return rewards
```

#### GRPO 超參數（lora_analyst）

```python
grpo_config = GRPOConfig(
    output_dir="outputs/lora_analyst_grpo",
    num_generations=6,          # NER 任務較簡單，可多生成幾個比較
    max_completion_length=128,  # JSON 輸出很短
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    num_train_epochs=3,
    kl_coeff=0.05,              # 格式任務偏離較少，KL 可以小一點
)
```

---

### 5.4 Phase 3：DPO × lora_storyteller（偏好對齊）

故事接龍的品質難以用規則驗證，適合用 DPO（Direct Preference Optimization）從偏好對中學習。

#### 偏好資料構造策略

從現有的 `lora_storyteller` 資料中自動構造 chosen/rejected 對：

```
策略 A（自動化）：
  chosen  = 原資料集中的 GPT 回應（視為高品質）
  rejected = 截斷版、隨機拼接版、或換任務生成的降質版本

策略 B（人工標記，最優）：
  讓 GPT-4o 對同一 prompt 生成 2 個版本，人工選較好的當 chosen
```

#### DPO 資料格式（ShareGPT Preference Format）

```json
{
  "conversations": [
    {"from": "system", "value": "You are a creative narrative writer..."},
    {"from": "human", "value": "Continue the story from where it left off:\n..."}
  ],
  "chosen": {"from": "gpt", "value": "（高品質的故事延續）"},
  "rejected": {"from": "gpt", "value": "（低品質的故事延續）"}
}
```

#### DPO 超參數

```python
# 使用 TRL DPOTrainer
dpo_config = DPOConfig(
    output_dir="outputs/lora_storyteller_dpo",
    beta=0.1,                   # KL 懲罰係數，0.1 是 DPO 標準值
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=2,
    max_length=1024,
    max_prompt_length=512,
)
```

---

### 5.5 整體 RL 訓練流水線

```
┌─────────────────────────────────────────────────────┐
│                   Base Model                        │
│           (Qwen2.5-7B-Instruct-bnb-4bit)            │
└─────────────────┬───────────────────────────────────┘
                  │  Step 1: SFT（已完成）
         ┌────────┼────────┬────────┐
         ▼        ▼        ▼        ▼
    SFT_analyst  SFT_reasoning  SFT_storyteller  SFT_translator
         │        │        │        │
         │  Step 2: RL 微調（新增）
         ▼        ▼        ▼
    GRPO_analyst  GRPO_reasoning  DPO_storyteller
         │        │        │
         └────────┴────────┘
                  │  Step 3: 評估比較（evaluate.py 擴充）
                  ▼
         RL vs SFT 性能對比報告
```

---

### 5.6 evaluate.py 需新增的 RL 評估指標

| 指標 | 計算方式 | 適用任務 |
|---|---|---|
| `json_validity_rate` | 輸出可被 `json.loads()` 解析的比率 | analyst, reasoning |
| `entity_grounding_rate` | 實體出現在輸入文本的比率 | analyst |
| `format_compliance_rate` | 包含【推理步驟】+【結論】的比率 | reasoning |
| `arithmetic_accuracy` | 數值計算正確的比率 | reasoning |
| `bleu_score` | 對照參考答案的 BLEU-4 分數 | translator |

---

### 5.7 RTX 3060 12GB 記憶體使用估算

| 訓練模式 | VRAM 估算 | 說明 |
|---|---|---|
| SFT (QLoRA 4-bit) | ~8-9 GB | 目前已驗證可行 |
| GRPO (QLoRA 4-bit, num_gen=4) | ~10-11 GB | 需同時持有 policy + reference，接近上限 |
| GRPO (QLoRA 4-bit, num_gen=6) | ~11-12 GB | 極限，建議設 `max_completion_length ≤ 256` |
| DPO (QLoRA 4-bit) | ~9-10 GB | 需同時計算 policy 和 reference log probs |

> **建議**：GRPO 訓練時將 `per_device_train_batch_size=1`、`num_generations=4`、`max_completion_length=256` 作為起點，確認不 OOM 後再逐步提升。

---

## 6. RL 訓練完整資料格式規格

本節統一列出所有 RL 訓練方法所需的資料格式，供資料集維護與新增時參考。

---

### 6.1 通用 SFT 格式（GRPO / DPO 訓練前的基礎）

所有任務的 SFT 資料與 GRPO 輸入資料均採用標準 **ShareGPT JSONL** 格式：

```jsonc
// 每行一個 JSON 物件（lora_*_train.jsonl）
{
  "conversations": [
    {"from": "system",  "value": "（系統提示，定義角色與任務）"},
    {"from": "human",   "value": "（玩家輸入或遊戲狀態）"},
    {"from": "gpt",     "value": "（模型預期輸出）"}
  ]
}
```

**欄位說明：**
- `from`: `"system"` / `"human"` / `"gpt"` 三選一
- `value`: 純文字字串，不可為 null
- `system` 欄位可省略（部分任務無系統提示）
- 每筆建議長度：**512 ~ 1024 tokens**（超過會被截斷）

---

### 6.2 lora_reasoning：GRPO 資料格式

**來源檔案：** `dataset/lora_reasoning/lora_reasoning_train.jsonl`

```jsonc
{
  "conversations": [
    {
      "from": "system",
      "value": "你是一個 DND 遊戲的後台推理引擎。根據玩家行動與當前遊戲狀態，推理 NPC 好感度的變化..."
    },
    {
      "from": "human",
      "value": "遊戲狀態：{\"「德拉克」好感度\": 30, \"場景\": \"酒館\"}\n玩家行動：幫德拉克付了一輪酒錢"
    },
    {
      "from": "gpt",
      "value": "【推理步驟】\n玩家主動幫NPC付酒錢，屬於善意舉動，預計好感度上升...\n\n【結論】\n{\"好感度增量\": 5, \"新好感度\": 35}"
    }
  ]
}
```

**GRPO 獎勵函數驗證的欄位：**
- `gpt.value` 中必須含 `【推理步驟】` 與 `【結論】` 兩個區塊
- `【結論】` 中必須是合法 JSON，含 `"好感度增量"` 和 `"新好感度"`
- 數值關係：`human` 中的初始好感度 + 增量 ≈ 新好感度（誤差 ≤ 1）

**生成指令：**
```bash
python prepare.py   # 自動分割 train/val
```

---

### 6.3 lora_analyst：GRPO 資料格式

**來源檔案：** `dataset/lora_analyst/lora_analyst_train.jsonl`

```jsonc
{
  "conversations": [
    {
      "from": "system",
      "value": "你是遊戲後台的實體抓取器。請分析文本中的角色名與組織名稱，輸出標準 JSON。"
    },
    {
      "from": "human",
      "value": "艾莉絲拔出長劍，衝向了黑龍騎士團的陣型，德拉克在後方支援。"
    },
    {
      "from": "gpt",
      "value": "{\"角色\": [\"艾莉絲\", \"德拉克\"], \"組織\": [\"黑龍騎士團\"]}"
    }
  ]
}
```

**GRPO 獎勵函數驗證的欄位：**
- `gpt.value` 必須是合法 JSON
- JSON 必須含 `"角色"` 和 `"組織"` 兩個 key，值為陣列
- **所有實體名稱必須出現在 `human.value` 的原文中**（anti-hallucination）

**已知品質問題：** 部分資料含假實體（如 `"Age"`、`"Campaign Four"`），建議搭配 GRPO 訓練以獎勵修正。

---

### 6.4 lora_translator：GRPO+BLEU 資料格式

**來源檔案：** `dataset/lora_translator/lora_translator_train.jsonl`

```jsonc
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a professional translator specializing in fantasy RPG terminology..."
    },
    {
      "from": "human",
      "value": "Translate the following text to Traditional Chinese:\n\nThe dragon swooped down..."
    },
    {
      "from": "gpt",
      "value": "巨龍俯衝而下..."
    }
  ]
}
```

**GRPO 獎勵函數（BLEU）：**
- `gpt.value` 作為 reference translation 計算 BLEU 分數
- 模型生成的譯文與 reference 比較，分數 0.0 ~ 1.0
- 需安裝：`pip install sacrebleu`（或 `pip install nltk`）

**已知問題：** 資料集長度過長（p50 ≈ 1366 tokens），在 1024 max_seq_len 下約 89% 的驗證集被截斷。
建議在 GRPO 訓練前先解決此問題（可過濾超長條目或增加 max_seq_len）。

---

### 6.5 lora_storyteller：DPO 偏好對資料格式

**來源檔案（生成後）：** `dataset/lora_storyteller_dpo/lora_storyteller_dpo_train.jsonl`

```jsonc
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a creative narrative writer specializing in fantasy RPG lore..."
    },
    {
      "from": "human",
      "value": "Continue the story:\nThe knight stepped into the dark forest, his torch barely cutting the shadows..."
    }
  ],
  "chosen": {
    "from": "gpt",
    "value": "The ancient trees groaned overhead, their gnarled branches reaching like skeletal fingers..."
  },
  "rejected": {
    "from": "gpt",
    "value": "（來自不同 prompt 的錯誤上下文回應，用於負樣本對比學習）"
  }
}
```

**DPO 偏好對構造策略（脈絡錯位法）：**
- `chosen` = 原始 SFT 資料中的正確 gpt 回應
- `rejected` = 從資料集中隨機抽取的另一筆的 gpt 回應（上下文不符）
- 此策略完全自動化，適合冷啟動 DPO 訓練

**生成指令：**
```bash
python prepare_dpo.py   # 自動生成並分割 train/val
```

---

### 6.6 完整 RL 訓練指令速查表

| 任務 | 步驟 | 指令 |
|------|------|------|
| **所有任務** | 資料準備（SFT） | `python prepare.py` |
| **storyteller DPO** | 資料準備（DPO） | `python prepare_dpo.py` |
| reasoning | SFT 訓練 | `python train_lora.py --task reasoning` |
| reasoning | GRPO 訓練 | `python train_grpo.py --task reasoning --sft-adapter outputs/lora_reasoning` |
| reasoning | 評估（三段對比） | `python evaluate.py --task reasoning --base-model` |
|  | | `python evaluate.py --task reasoning` |
|  | | `python evaluate.py --task reasoning --rl` |
| analyst | SFT 訓練 | `python train_lora.py --task analyst` |
| analyst | GRPO 訓練 | `python train_grpo.py --task analyst --sft-adapter outputs/lora_analyst` |
| analyst | 評估 | `python evaluate.py --task analyst --rl` |
| storyteller | SFT 訓練 | `python train_lora.py --task storyteller` |
| storyteller | DPO 訓練 | `python train_dpo.py --task storyteller --sft-adapter outputs/lora_storyteller` |
| storyteller | 評估 | `python evaluate.py --task storyteller --dpo` |
| translator | SFT 訓練 | `python train_lora.py --task translator` |
| translator | GRPO+BLEU 訓練 | `python train_grpo.py --task translator --sft-adapter outputs/lora_translator` |
| translator | 評估（含 BLEU） | `python evaluate.py --task translator --rl` |

**快速實驗（OOM 測試）：**
```bash
python train_grpo.py --task reasoning --max-steps 10
python train_grpo.py --task analyst   --max-steps 10
python train_grpo.py --task translator --max-steps 10
python train_dpo.py  --task storyteller --max-steps 10
```

---

### 6.7 RL 相關 Python 套件需求

```
# 必要（已含於 unsloth 環境）
unsloth
trl>=0.12
transformers
datasets
torch

# translator GRPO+BLEU 評估所需
pip install sacrebleu   # 優先使用

# DPO 偏好對訓練（trl 已內含 DPOTrainer）
# 無需額外安裝
```
