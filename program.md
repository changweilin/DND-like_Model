# LoRA 自動實驗框架 — AI Agent 操作指引

本文件描述如何以 AI Agent 自動執行 LoRA 微調實驗迴圈，參考 karpathy/autoresearch 模式。

---

## 檔案角色

| 檔案 | 角色 | 可修改？ |
|------|------|----------|
| `prepare.py` | 資料前處理與驗證（一次性執行） | 唯讀（資料邏輯已固定） |
| `train_lora.py` | 核心訓練腳本，帶 CLI 參數 | **可調整**（實驗主體） |
| `evaluate.py` | 載入 adapter、計算 perplexity、生成樣本 | 唯讀（評估指標固定） |
| `results.tsv` | 實驗紀錄（自動附加） | 唯讀（只能讀取） |
| `dataset/` | 分割好的 train/val JSONL | 唯讀 |
| `outputs/lora_{task}/` | 訓練輸出、eval_report.json | 讀/寫 |

---

## 實驗迴圈

```
┌─────────────────────────────────────────────────────┐
│  1. 選擇任務與初始超參數                               │
│  2. python train_lora.py --task <T> [--max-steps 75] │
│  3. 解析 [RESULT_JSON] 行                             │
│  4. python evaluate.py --task <T>                    │
│  5. 解析 [EVAL_JSON] 行                               │
│  6. 附加一行至 results.tsv                            │
│  7. 比較 eval_loss / perplexity                      │
│  8a. 若改善 → 保留 adapter，繼續下一個實驗            │
│  8b. 若退步 → 記錄失敗，回退超參數或嘗試其他方向       │
└─────────────────────────────────────────────────────┘
```

---

## 快速實驗模式

加上 `--max-steps 75` 可在約 5 分鐘內完成一輪（適合超參數搜尋）：

```bash
python train_lora.py --task analyst --max-steps 75
python evaluate.py --task analyst
```

完整訓練（移除 `--max-steps`）：

```bash
python train_lora.py --task analyst
python evaluate.py --task analyst
```

---

## 可實驗方向

以下參數可透過 CLI 覆蓋，無需修改程式碼：

| 參數 | CLI flag | 建議範圍 | 說明 |
|------|----------|----------|------|
| LoRA rank | `--rank` | 8 / 16 / 32 / 64 | 越高容量越大，VRAM 也越多 |
| LoRA alpha | `--alpha` | = rank 或 2× rank | alpha/rank 比例影響學習強度 |
| Learning rate | `--lr` | 1e-5 ~ 5e-4 | 過大易震盪，過小收斂慢 |
| Epochs | `--epochs` | 1 ~ 5 | 小資料集易過擬合 |
| Batch size | `--batch-size` | 1 / 2 | 12GB VRAM 建議 2 |
| Grad accum | `--grad-accum` | 4 / 8 | 等效 global batch = batch × grad_accum |
| Max seq len | `--max-seq-len` | 512 / 1024 | 降低可大幅減少 VRAM |

---

## 硬體約束（RTX 3060 12GB）

- **OOM 處理順序**：先降 `--max-seq-len 512`，再降 `--batch-size 1`，最後降 `--rank`
- 峰值 VRAM 目標：< 11GB（留 1GB 給 OS）
- `train_lora.py` 會輸出 `peak_memory_gb`，超過 11.5GB 視為高風險

---

## results.tsv 格式

```
timestamp	task	rank	alpha	lr	epochs	max_steps	train_loss	eval_loss	peak_memory_gb	status	description
2025-01-01T12:00:00	analyst	16	32	0.0003	2	75	1.234	1.456	9.8	ok	quick test
```

- `max_steps=-1` 表示跑完全部步數
- `description` 欄位可手動填寫實驗說明
- `status`：`ok` / `oom` / `failed`

---

## 各任務預設超參數與建議

### analyst（NER/資訊抽取）
- 預設：rank=16, alpha=32, lr=3e-4, epochs=2
- 注意：資料集有部分 NER 標註錯誤（如將 "Age"、"Campaign Four" 誤標為角色名），eval perplexity 僅作相對比較用
- 次要指標：JSON 解析成功率（`eval_report.json` 的 `quality.json_parse_success_rate`）

### translator（翻譯）
- 預設：rank=64, alpha=128, lr=1e-4, epochs=2
- 高 rank/alpha 對翻譯風格遷移效果較佳
- 次要指標：生成 10 筆翻譯對照，人工審查

### storyteller（故事接龍）
- 預設：rank=32, alpha=64, lr=2e-5, epochs=3
- 文本較長（400-500 字/筆），1024 token 下可能截斷，但對學習文風仍有效
- 可嘗試 `--max-seq-len 512` 快速測試，再用 1024 做正式訓練

### storyteller_extra（補充語料）
- 預設：rank=64, alpha=64, lr=2e-5, epochs=3
- 包含 rpg_dataset（TRPG 世界觀）+ literature_dataset（LitRPG 小說）
- 可與 storyteller 對比 eval_loss，判斷補充語料是否有幫助

---

## 完整實驗流程範例

```bash
# 1. 資料準備（只需執行一次）
python prepare.py

# 2. 快速測試（確認流程可跑通）
python train_lora.py --task analyst --max-steps 10
python evaluate.py --task analyst

# 3. 正式訓練 analyst
python train_lora.py --task analyst
python evaluate.py --task analyst

# 4. 嘗試更高 LR
python train_lora.py --task analyst --lr 5e-4 --max-steps 75
python evaluate.py --task analyst

# 5. 比較 results.tsv，選出最佳配置
# 6. 正式訓練 storyteller
python train_lora.py --task storyteller
python evaluate.py --task storyteller
```

---

## 尚未涵蓋

- **Task C（語意推理）**：目前無專用資料集，需另行製備後加入 `TASK_PRESETS`
- **多 adapter 合併推理**：可透過 vLLM LoRA serving 動態載入多個 adapter
- **超過 1024 token 的長文本**：若要訓練完整長文本，考慮升級至 24GB VRAM 或使用 Flash Attention + context chunking
