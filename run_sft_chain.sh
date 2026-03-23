#!/bin/bash
set -e
export UNSLOTH_CE_LOSS_TARGET_GB=128

echo "=== [$(date)] 等待 analyst 訓練完成 ==="
while wmic process where "Name='python.exe' and CommandLine like '%train_lora.py --task analyst%'" get ProcessId 2>&1 | grep -q "[0-9]"; do
    sleep 30
done
echo "=== [$(date)] analyst 訓練完成 ==="
sleep 5

echo "=== [$(date)] 評估 analyst ==="
python evaluate.py --task analyst 2>&1 | tee logs/eval_analyst.log

echo "=== [$(date)] 訓練 reasoning ==="
python train_lora.py --task reasoning 2>&1 | tee logs/sft_reasoning_rerun.log
echo "=== [$(date)] 評估 reasoning ==="
python evaluate.py --task reasoning 2>&1 | tee logs/eval_reasoning.log

echo "=== [$(date)] 訓練 storyteller ==="
python train_lora.py --task storyteller 2>&1 | tee logs/sft_storyteller_rerun.log
echo "=== [$(date)] 評估 storyteller ==="
python evaluate.py --task storyteller 2>&1 | tee logs/eval_storyteller.log

echo "=== [$(date)] 訓練 storyteller_extra ==="
python train_lora.py --task storyteller_extra 2>&1 | tee logs/sft_storyteller_extra_rerun.log
echo "=== [$(date)] 評估 storyteller_extra ==="
python evaluate.py --task storyteller_extra 2>&1 | tee logs/eval_storyteller_extra.log

echo "=== [$(date)] 訓練 translator ==="
python train_lora.py --task translator 2>&1 | tee logs/sft_translator_rerun.log
echo "=== [$(date)] 評估 translator ==="
python evaluate.py --task translator 2>&1 | tee logs/eval_translator.log

echo "=== [$(date)] 全部 Phase 1 SFT 完成 ==="
