# DND-like Model: 強化學習與 LoRA 微調框架

本專案提供一套針對大型語言模型 (LLM) 進行 **多任務 QLoRA 微調** 與 **強化學習 (RLVR)** 的自動化實驗框架，專為文本遊戲 (TRPG / LitRPG) 的角色扮演、邏輯推理與翻譯對齊所設計。

本框架基於 **Unsloth** 構建，支援在消費級顯示卡 (如 RTX 3060 12GB) 上進行高效訓練，並整合了 DPO (直接偏好優化) 與 GRPO (群組相對策略優化) 等先進的 RL 演算法。

---

## 🏗️ 系統架構與流程圖

本專案將開發流程分為三個階段：**SFT (監督式微調)**、**RL (強化學習優化)** 以及 **Eval (多維度評估)**。

```mermaid
graph TD
    subgraph 數據準備
        Data[ShareGPT 資料集] --> Prep[prepare.py / prepare_dpo.py]
    end

    subgraph 階段一：SFT (基礎能力)
        Prep --> SFT(train_lora.py)
        SFT -->|輸出 Adapter| SFT_Adap[SFT LoRA Weights]
    end

    subgraph 階段二：RL (風格與邏輯對齊)
        SFT_Adap --> DPO(train_dpo.py <br/> 偏好優化)
        SFT_Adap --> GRPO(train_grpo.py <br/> 規則獎勵)
        DPO -->|輸出 Adapter| RL_Adap[RL LoRA Weights]
        GRPO -->|輸出 Adapter| RL_Adap
    end

    subgraph 階段三：評估 (Quality Assurance)
        SFT_Adap --> Eval(evaluate.py)
        RL_Adap --> Eval
        Eval -->|LLM Judge| Judge[Gemini 2.5 Flash]
        Eval -->|指標報告| Results[[results.tsv / eval_report.json]]
    end

    style SFT fill:#bbf,stroke:#333
    style DPO fill:#f9f,stroke:#333
    style GRPO fill:#f9f,stroke:#333
    style Judge fill:#dfd,stroke:#333
```

---

## 🛠️ 自動化工具矩陣

### 1. 基礎開發 (SFT)
- **`prepare.py`**: 將資料集標準化為 ShareGPT 格式，自動進行 9:1 的 Train/Val 切分與 Token 長度驗證。
- **`train_lora.py`**: 執行標準的 QLoRA 訓練。支援 `analyst`, `translator`, `storyteller` 等任務預設值。

### 2. 強化學習 (RL / Alignment)
本階段旨在解決 SFT 無法輕易處理的「風格偏好」與「硬性邏輯限制」。
- **`prepare_dpo.py`**: 自動構造 DPO 偏好對（Chosen vs Rejected）。支援使用 **Gemini** 輔助生成低品質樣本。
- **`train_dpo.py`**: 針對 `storyteller` 任務進行風格對齊，讓模型學習「什麼是更好的敘事」。
- **`train_grpo.py`**: 針對具備明確規則的任務進行獎勵優化。
    - `reasoning`: 優化思維鏈 (CoT) 格式與數值計算正確性。
    - `analyst`: 強化 JSON 結構正確性與實體定位（避免幻覺）。
    - `translator`: 透過 BLEU 分數獎勵來優化翻譯忠實度。

### 3. 多維評估 (Evaluation)
- **`evaluate.py`**: 提供自動化評估管線。
    - **指標**: 計算 Perplexity, BLEU (翻譯), JSON 解析率 (分析), 數值精確度 (推理)。
    - **--llm-judge**: 調用 **Gemini 2.5 Flash** 作為智慧評審，對生成的文學性、流暢度給予 0~5 分評分與短評。
    - **--rl / --dpo**: 自動載入對應的強化學習 Adapter 進行比較。

---

## 🚀 腳本使用指南

### SFT 訓練與測試
```bash
# 正式訓練分析任務
python train_lora.py --task analyst

# 快速測試推理任務 (75步)
python train_lora.py --task reasoning --max-steps 75
```

### RL 強化學習
```bash
# 準備 DPO 資料 (可選項 --use-llm 調用 Gemini 生成被拒絕樣本)
python prepare_dpo.py --use-llm

# 執行 DPO 訓練 (從現有 SFT adapter 開始)
python train_dpo.py --task storyteller --sft-adapter outputs/lora_storyteller

# 執行 GRPO 訓練 (優化邏輯推理)
python train_grpo.py --task reasoning --sft-adapter outputs/lora_reasoning
```

### 多維度評估
```bash
# 基礎評估
python evaluate.py --task analyst

# 評估 RL 後的模型並開啟 LLM Judge
python evaluate.py --task storyteller --dpo --llm-judge

# 評估推理任務的 GRPO 效果
python evaluate.py --task reasoning --rl
```

---

## 📊 任務參數參考 (Presets)

| 任務 (Task) | 模式 | 核心指標 | Rank / Alpha | 學習率 (LR) | 說明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `analyst` | GRPO | JSON Valid / Grounding | 8 / 16 | 1e-5 | 強化實體提取與格式規範 |
| `reasoning` | GRPO | Math Accuracy / Format | 16 / 32 | 5e-6 | 二階段推理：【推理步驟】➔【結論】 |
| `translator` | SFT | BLEU Score | 64 / 128 | 1e-4 | 跨語言映射需要較大參數空間 |
| `storyteller` | DPO | 文學性 (LLM Judge) | 32 / 64 | 1e-5 | 透過偏好對微調敘事文風 |

---

## 💻 資源管理與優化 (RTX 3060 12GB)

1. **Unsloth 4-bit 量化**: 預設載入 4-bit 量化基座模型，節省約 60% VRAM。
2. **GRPO 群組限制**: 在 12GB 環境下，`num_generations` 建議設為 4~6，並限制 `max_completion_length` 在 384 token 以內避免 OOM。
3. **梯度累積**: 為了在單卡達成有效 Batch Size，預設使用 `gradient_accumulation_steps=4~8`。

如果您在訓練中遇到 `Out of Memory`，請優先嘗試：
- 降低 `--max-steps` (僅用於測試) 或縮短 `--max-seq-len`。
- 關閉所有背景佔用 VRAM 的程式。
- 對於 GRPO，降低 `--num-generations`。
