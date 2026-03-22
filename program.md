# LoRA 自動實驗框架 — AI Agent 操作指引

本文件描述如何以 AI Agent 自動執行完整訓練流程，涵蓋 SFT → GRPO/DPO → 評估三個階段。

---

## 檔案角色

| 檔案 | 角色 | 可修改？ |
|------|------|----------|
| `prepare.py` | SFT 資料前處理（格式轉換、分割、驗證） | 唯讀 |
| `prepare_dpo.py` | 生成 DPO 偏好對資料集（storyteller） | 唯讀 |
| `train_lora.py` | SFT 核心訓練腳本（5 任務） | **可調整** |
| `train_grpo.py` | GRPO 強化學習訓練（reasoning/analyst/translator） | **可調整** |
| `train_dpo.py` | DPO 偏好優化訓練（storyteller） | **可調整** |
| `evaluate.py` | 統一評估（支援 SFT/GRPO/DPO/base-model） | 唯讀 |
| `results.tsv` | SFT 實驗紀錄（自動附加） | 唯讀 |
| `results_rl.tsv` | GRPO/DPO 實驗紀錄（自動附加） | 唯讀 |
| `dataset/` | 分割好的 train/val JSONL | 唯讀 |
| `outputs/lora_{task}/` | SFT adapter 輸出 | 讀/寫 |
| `outputs/lora_{task}_grpo/` | GRPO adapter 輸出 | 讀/寫 |
| `outputs/lora_{task}_dpo/` | DPO adapter 輸出 | 讀/寫 |
| `deploy_lora.py` | LoRA → GGUF → Ollama 部署工具 | 唯讀 |
| `outputs/gguf_{task}/` | GGUF 匯出目錄（含 Modelfile） | 讀/寫 |
| `outputs/deploy_state.json` | 部署狀態追蹤 | 讀/寫 |

---

## 完整訓練流程總覽

```
Phase 0：資料準備
  prepare.py ──────────────────────────→ dataset/lora_{task}/
  prepare_dpo.py ──────────────────────→ dataset/lora_storyteller_dpo/

Phase 1：SFT（監督式微調）
  train_lora.py --task analyst  ────────→ outputs/lora_analyst/
  train_lora.py --task translator ──────→ outputs/lora_translator/
  train_lora.py --task storyteller ─────→ outputs/lora_storyteller/
  train_lora.py --task storyteller_extra→ outputs/lora_storyteller_extra/
  train_lora.py --task reasoning ───────→ outputs/lora_reasoning/

Phase 2：RL 對齊（SFT → RL）
  train_grpo.py --task reasoning ───────→ outputs/lora_reasoning_grpo/
  train_grpo.py --task analyst ─────────→ outputs/lora_analyst_grpo/
  train_grpo.py --task translator ──────→ outputs/lora_translator_grpo/
  train_dpo.py  --task storyteller ─────→ outputs/lora_storyteller_dpo/

Phase 3：評估與比較
  evaluate.py --task {T}          (SFT 基準)
  evaluate.py --task {T} --rl     (GRPO 版本)
  evaluate.py --task {T} --dpo    (DPO 版本，僅 storyteller)

Phase 4：部署至 DND-like_RPG
  deploy_lora.py --task {T}                    (單一任務，預設 q4_k_m)
  deploy_lora.py --all --update-config         (所有任務，更新 RPG config.py)
  deploy_lora.py --status                      (查看部署狀態)
```

---

## 建議訓練順序與原因

### 1. 先跑 analyst（最快、最穩定）
analyst 任務資料量最小、JSON 格式明確、評估指標客觀（JSON 解析率），
是調整超參數流程的最佳起點。確認流程跑通後再進行其他任務。

```bash
python train_lora.py --task analyst --max-steps 10   # 確認流程可跑通
python evaluate.py --task analyst

python train_lora.py --task analyst                  # 正式 SFT
python evaluate.py --task analyst

python train_grpo.py --task analyst --sft-adapter outputs/lora_analyst --max-steps 50  # 快速 GRPO 測試
python train_grpo.py --task analyst --sft-adapter outputs/lora_analyst                 # 正式 GRPO
python evaluate.py --task analyst --rl
```

### 2. 接著跑 reasoning（結構化輸出 + 數值計算）
reasoning 有最完整的 GRPO 獎勵函數（格式 + JSON + 數值正確性），
SFT 先建立格式基礎，GRPO 再強化數值計算準確率。

```bash
python train_lora.py --task reasoning
python evaluate.py --task reasoning

python train_grpo.py --task reasoning --sft-adapter outputs/lora_reasoning
python evaluate.py --task reasoning --rl
```

### 3. 再跑 storyteller（文風訓練）
storyteller 資料量較大，文本較長，建議先用 512 token 快速測試，
再用 1024 做正式訓練。SFT 後用 DPO 對齊文風偏好。

```bash
# 先合併補充語料測試
python train_lora.py --task storyteller_extra --max-seq-len 512 --max-steps 75
python evaluate.py --task storyteller_extra

python train_lora.py --task storyteller
python evaluate.py --task storyteller

# DPO 對齊（從 SFT adapter 繼續）
python train_dpo.py --task storyteller --sft-adapter outputs/lora_storyteller
python evaluate.py --task storyteller --dpo
```

### 4. 最後跑 translator（高 rank，最耗 VRAM）
translator 需要 rank=64，VRAM 壓力最大，排到最後避免干擾其他任務的測試。

```bash
python train_lora.py --task translator
python evaluate.py --task translator

python train_grpo.py --task translator --sft-adapter outputs/lora_translator
python evaluate.py --task translator --rl
```

---

## 完整實驗指令（可直接複製執行）

```bash
# ── Phase 0：資料準備（只需執行一次）────────────────────────────────
python prepare.py
python prepare_dpo.py

# ── Phase 1：SFT ─────────────────────────────────────────────────────
python train_lora.py --task analyst
python evaluate.py --task analyst

python train_lora.py --task reasoning
python evaluate.py --task reasoning

python train_lora.py --task storyteller
python train_lora.py --task storyteller_extra
python evaluate.py --task storyteller
python evaluate.py --task storyteller_extra

python train_lora.py --task translator
python evaluate.py --task translator

# ── Phase 2：RL 對齊 ─────────────────────────────────────────────────
python train_grpo.py --task analyst --sft-adapter outputs/lora_analyst
python evaluate.py --task analyst --rl

python train_grpo.py --task reasoning --sft-adapter outputs/lora_reasoning
python evaluate.py --task reasoning --rl

python train_grpo.py --task translator --sft-adapter outputs/lora_translator
python evaluate.py --task translator --rl

python train_dpo.py --task storyteller --sft-adapter outputs/lora_storyteller
python evaluate.py --task storyteller --dpo

# ── Phase 3：比較基準（零樣本 base model）────────────────────────────
python evaluate.py --task analyst --base-model
python evaluate.py --task reasoning --base-model

# ── Phase 4：部署至 Ollama / DND-like_RPG ────────────────────────────
python deploy_lora.py --status                          # 確認哪些 adapter 可部署
python deploy_lora.py --task analyst                    # 單一任務部署
python deploy_lora.py --task storyteller --update-config # 部署並更新 RPG config.py
python deploy_lora.py --all --update-config             # 部署全部並更新 config
```

---

## 評估指令對照表

| 任務 | SFT 基準 | GRPO 版本 | DPO 版本 |
|------|----------|-----------|----------|
| analyst | `evaluate.py --task analyst` | `evaluate.py --task analyst --rl` | — |
| reasoning | `evaluate.py --task reasoning` | `evaluate.py --task reasoning --rl` | — |
| translator | `evaluate.py --task translator` | `evaluate.py --task translator --rl` | — |
| storyteller | `evaluate.py --task storyteller` | — | `evaluate.py --task storyteller --dpo` |

---

## 各 RL 方法說明

| 方法 | 腳本 | 適合任務 | 關鍵設定 |
|------|------|----------|----------|
| **GRPO** | `train_grpo.py` | analyst / reasoning / translator | `--num-generations`（候選數）, `--kl-coeff`（KL 懲罰）|
| **DPO** | `train_dpo.py` | storyteller | `--beta`（KL 懲罰）, `--sft-adapter`（從 SFT 繼續）|

- **GRPO**：規則式獎勵，適合有客觀評估標準的任務（JSON 格式、數值計算、BLEU）
- **DPO**：偏好對資料，適合無法設計規則獎勵的任務（文風、創意）

---

## 快速實驗模式

所有訓練腳本均支援 `--max-steps` 進行 5 分鐘內的快速驗證：

```bash
python train_lora.py --task analyst --max-steps 75
python train_grpo.py --task reasoning --max-steps 50
python train_dpo.py --task storyteller --max-steps 50
```

---

## 可實驗方向

### SFT 超參數（train_lora.py）

| 參數 | CLI flag | 建議範圍 |
|------|----------|----------|
| LoRA rank | `--rank` | 8 / 16 / 32 / 64 |
| LoRA alpha | `--alpha` | = rank 或 2× rank |
| Learning rate | `--lr` | 1e-5 ~ 5e-4 |
| Epochs | `--epochs` | 1 ~ 5 |
| Batch size | `--batch-size` | 1 / 2 |
| Max seq len | `--max-seq-len` | 512 / 1024 |

### GRPO 超參數（train_grpo.py）

| 參數 | CLI flag | 建議範圍 |
|------|----------|----------|
| 候選數 | `--num-generations` | 4 / 6 |
| KL 係數 | `--kl-coeff` | 0.05 ~ 0.2 |
| 生成長度 | `--max-completion` | 128 ~ 512 |

### DPO 超參數（train_dpo.py）

| 參數 | CLI flag | 建議範圍 |
|------|----------|----------|
| Beta | `--beta` | 0.05 ~ 0.3（預設 0.1）|
| SFT 起點 | `--sft-adapter` | 建議從 SFT adapter 繼續 |

---

## 硬體約束（RTX 3060 12GB）

- **OOM 處理順序**：先降 `--max-seq-len 512`，再降 `--batch-size 1`，最後降 `--rank`
- **GRPO 特別注意**：`num_generations × max_completion` 影響 VRAM，超過 11.5GB 需降低其中一個
- 峰值 VRAM 目標：< 11GB（留 1GB 給 OS）

---

## results.tsv 格式（SFT）

```
timestamp	task	rank	alpha	lr	epochs	max_steps	train_loss	eval_loss	peak_memory_gb	status	description
```

## results_rl.tsv 格式（GRPO / DPO）

```
timestamp	task	mode	rank	alpha	lr	num_generations	max_completion	kl_coeff	epochs	max_steps	train_loss	peak_memory_gb	elapsed_sec	sft_adapter	status
```

---

## 各任務預設超參數

### analyst（NER / 資訊抽取）
- SFT：rank=16, alpha=32, lr=3e-4, epochs=2
- GRPO：rank=8, alpha=16, lr=1e-5, num_gen=6, max_completion=128, kl=0.05
- 評估指標：JSON 解析率、實體定位率

### reasoning（推理 + 數值計算）
- SFT：rank=32, alpha=64, lr=1e-4, epochs=3
- GRPO：rank=16, alpha=32, lr=5e-6, num_gen=4, max_completion=384, kl=0.1
- 評估指標：格式合規率、JSON 有效率、數值計算正確率

### translator（翻譯）
- SFT：rank=64, alpha=128, lr=1e-4, epochs=2
- GRPO：rank=32, alpha=64, lr=2e-6, num_gen=4, max_completion=256, kl=0.1
- 評估指標：平均 BLEU 分數

### storyteller（故事接龍）
- SFT：rank=32, alpha=64, lr=2e-5, epochs=3
- DPO：rank=32, alpha=64, lr=1e-5, beta=0.1, epochs=2
- 評估指標：文風樣本（人工審查）

### storyteller_extra（補充語料）
- SFT：rank=64, alpha=64, lr=2e-5, epochs=3
- 可與 storyteller 對比 eval_loss，判斷補充語料是否有幫助

---

## AutoResearch 自動化工具

### autoresearch.py — 自動超參數實驗迴圈

仿照 Karpathy's autoresearch，以 Claude API 自動分析歷史結果並建議下一組超參數。

```bash
# 環境設定
set ANTHROPIC_API_KEY=your_key_here

# SFT 自動實驗（每次 ~5 分鐘，共 5 次）
python autoresearch.py --task analyst --mode sft

# GRPO 自動實驗
python autoresearch.py --task reasoning --mode grpo --max-iterations 3

# 完整訓練模式（每次數小時）
python autoresearch.py --task analyst --mode sft --full

# 查看歷史迭代紀錄
python autoresearch.py --list
```

記錄檔：`autoresearch_log.jsonl`

---

### experiment_tracker.py — 實驗視覺化分析器

依賴：`pip install rich`

```bash
python experiment_tracker.py                     # 全部任務 SFT + RL 表格
python experiment_tracker.py --task analyst      # 只看 analyst
python experiment_tracker.py --best              # 各任務最佳結果總覽
python experiment_tracker.py --compare --task analyst  # SFT vs RL 比較
python experiment_tracker.py --ar                # AutoResearch 迭代紀錄
python experiment_tracker.py --html              # 輸出 HTML 趨勢報告
```

---

## 尚未涵蓋

- **多 adapter 合併推理**：可透過 vLLM LoRA serving 動態載入多個 adapter
- **超過 1024 token 的長文本**：考慮升級至 24GB VRAM 或使用 Flash Attention + context chunking
- **storyteller GRPO**：目前文風無法設計規則式獎勵，使用 DPO 替代
