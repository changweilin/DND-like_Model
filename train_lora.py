import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

# =====================================================================
# 1. 訓練參數設定區 (可依據 lora_training_guide.md 調整)
# =====================================================================
# 模型名稱 (以 7B/8B 最適合 12GB VRAM，此處使用 Qwen2.5-7B 為例)
# 如果是 Llama-3.1-8B-Instruct，請改為 "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

# 任務超參數 (此處預設為「故事接龍」，如為「實體抓取」，請將 RANK 降為 8，LR 改為 3e-4)
LORA_RANK = 32
LORA_ALPHA = 64
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3

# 資料集與輸出設定
DATASET_PATH = "dataset.jsonl"             # 您的 ShareGPT 格式對話訓練資料
OUTPUT_DIR = "outputs/lora_storyteller"      # 儲存 LoRA 結果的資料夾
MAX_SEQ_LENGTH = 1024                      # 12GB VRAM 建議設為 512 ~ 1024

# =====================================================================
# 2. 載入模型與分詞器 (4-bit 強制量化以適配單卡 12GB)
# =====================================================================
print(f"Loading Base Model: {MODEL_NAME} in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,           # 自動偵測 fp16 / bf16
    load_in_4bit = True,    # 核心：強制 4-bit 載入
)

# 套用 LoRA 權重設定
print("Applying LoRA adaptors...")
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    # 將所有線性層都掛上 LoRA 能得到最全面的微調效果
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_ALPHA,
    lora_dropout = 0,       # Unsloth 架構下建議設為 0，效能最高
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 極大節省 VRAM 的關鍵技術
    random_state = 1234,
)

# =====================================================================
# 3. 準備與格式化資料集
# =====================================================================
# 指定對話模板 (Chat Template)。若使用 Llama3 請改為 "llama-3.1"
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5", 
)

# 格式化函數：將 ShareGPT 的系統/用戶/助理對話轉為模型接受的一長串文本
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text" : texts }

if not os.path.exists(DATASET_PATH):
    print(f"警告: 找不到訓練資料 {DATASET_PATH}")
    print("請先建立 ShareGPT 格式的 jsonl 檔案，再執行此腳本。")
    exit(1)

print(f"Loading dataset from {DATASET_PATH}...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# =====================================================================
# 4. 開始訓練
# =====================================================================
# SFT (Supervised Fine-Tuning) 訓練器設定
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        # 12GB VRAM 批次設定
        per_device_train_batch_size = 2,    # 每次塞入記憶體的筆數
        gradient_accumulation_steps = 4,    # 等效全局 Batch Size = 2 * 4 = 8
        warmup_steps = 10,
        num_train_epochs = NUM_EPOCHS,      # 訓練回合數
        learning_rate = LEARNING_RATE,      # 學習率
        fp16 = not is_bfloat16_supported(), # 支援舊卡(例如 3060 支援 fp16)的相容設定
        bf16 = is_bfloat16_supported(),     # 如果是 40 系列以上會自動用 bf16
        logging_steps = 5,
        optim = "adamw_8bit",               # 使用 8-bit 優化器，進一步減少 VRAM 佔用
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 1234,
        output_dir = OUTPUT_DIR,
        save_strategy = "epoch",            # 每個 Epoch 存檔一次
    ),
)

print("\n🚀 開始訓練 LoRA...")
trainer_stats = trainer.train()

# =====================================================================
# 5. 儲存 LoRA 輕量權重
# =====================================================================
print(f"\n✅ 訓練完成！正在儲存 LoRA Adapter 至: {OUTPUT_DIR}...")
# Unsloth 只會儲存 LoRA 權重 (約幾十 MB 到幾百 MB)，不會複製整個基地模型
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 儲存完畢。")
print("後續可在 vLLM 或 llama.cpp 中，動態掛載這個 LoRA 進行推論。")
