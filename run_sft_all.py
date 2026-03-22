"""
run_sft_all.py — Phase 1 SFT 批次訓練（reasoning / storyteller / storyteller_extra / translator）

使用方式：
  python run_sft_all.py
  python run_sft_all.py --tasks reasoning storyteller   # 只跑指定任務
  python run_sft_all.py --skip-existing                 # 跳過已有 adapter 的任務

OOM 自動降參順序：
  1. --max-seq-len 512
  2. --batch-size 1
  3. --rank 降半
"""

import argparse
import datetime
import os
import subprocess
import sys
import time

# ── 任務順序 ──────────────────────────────────────────────────────────────────

TASKS = ["reasoning", "storyteller", "storyteller_extra", "translator"]

# OOM 降參策略：依序嘗試
OOM_FALLBACKS = [
    {"max_seq_len": 512},
    {"max_seq_len": 512, "batch_size": 1},
    {"max_seq_len": 512, "batch_size": 1, "reduce_rank": True},
]

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ── 工具函數 ──────────────────────────────────────────────────────────────────

def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str):
    print(f"[{timestamp()}] {msg}", flush=True)


def adapter_exists(task: str) -> bool:
    path = os.path.join("outputs", f"lora_{task}", "adapter_config.json")
    return os.path.exists(path)


def is_oom(returncode: int, output: str) -> bool:
    if returncode == 0:
        return False
    oom_signals = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "out of memory",
        "OOM",
        "CUBLAS_STATUS_ALLOC_FAILED",
    ]
    return any(s.lower() in output.lower() for s in oom_signals)


def build_cmd(task: str, extra: dict) -> list[str]:
    cmd = [sys.executable, "train_lora.py", "--task", task]
    if "max_seq_len" in extra:
        cmd += ["--max-seq-len", str(extra["max_seq_len"])]
    if "batch_size" in extra:
        cmd += ["--batch-size", str(extra["batch_size"])]
    if extra.get("reduce_rank"):
        # 從 TASK_PRESETS 讀出預設 rank，降半
        presets = {
            "reasoning": 32, "storyteller": 32,
            "storyteller_extra": 64, "translator": 64,
        }
        new_rank = max(8, presets.get(task, 32) // 2)
        cmd += ["--rank", str(new_rank), "--alpha", str(new_rank * 2)]
    return cmd


def run_task(task: str, log_path: str, extra: dict = None) -> tuple[int, str]:
    """執行單次訓練，回傳 (returncode, combined_output)。"""
    cmd = build_cmd(task, extra or {})
    label = " ".join(cmd[2:])   # 去掉 python train_lora.py
    log(f"  執行：python train_lora.py {label}")

    buf: list[str] = []
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*60}\n[{timestamp()}] CMD: {' '.join(cmd)}\n{'='*60}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            buf.append(line)
            lf.write(line)
        proc.wait()

    return proc.returncode, "".join(buf)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳過已有 adapter_config.json 的任務")
    args = parser.parse_args()

    results: dict[str, str] = {}
    start_total = time.time()

    for task in args.tasks:
        log(f"{'='*60}")
        log(f"任務：{task}")

        if args.skip_existing and adapter_exists(task):
            log(f"  已存在 outputs/lora_{task}/adapter_config.json，跳過。")
            results[task] = "skipped"
            continue

        log_path = os.path.join(LOG_DIR, f"sft_{task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        log(f"  Log → {log_path}")

        start = time.time()
        success = False

        # 第一次嘗試（預設參數）
        rc, output = run_task(task, log_path)
        if rc == 0:
            success = True
        elif is_oom(rc, output):
            log(f"  OOM 偵測到，嘗試降參…")
            for i, fallback in enumerate(OOM_FALLBACKS):
                log(f"  降參策略 {i+1}/{len(OOM_FALLBACKS)}: {fallback}")
                rc, output = run_task(task, log_path, extra=fallback)
                if rc == 0:
                    success = True
                    break
                if not is_oom(rc, output):
                    break   # 非 OOM 錯誤，不再嘗試

        elapsed = time.time() - start
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))

        if success:
            log(f"  完成 ({elapsed_str})")
            results[task] = f"ok ({elapsed_str})"
        else:
            log(f"  失敗 (returncode={rc}, elapsed={elapsed_str})")
            results[task] = f"FAILED (rc={rc})"

    # ── 總結 ──────────────────────────────────────────────────────────────────
    total_elapsed = str(datetime.timedelta(seconds=int(time.time() - start_total)))
    log(f"{'='*60}")
    log(f"Phase 1 SFT 批次完成（總耗時 {total_elapsed}）")
    log("")
    for task, status in results.items():
        mark = "✓" if status.startswith("ok") else ("－" if status == "skipped" else "✗")
        log(f"  {mark}  {task:<20} {status}")
    log("")

    all_ok = all(s.startswith("ok") or s == "skipped" for s in results.values())
    if all_ok:
        log("全部成功。下一步：")
        log("  python evaluate.py --task analyst")
        log("  python evaluate.py --task reasoning")
        log("  python evaluate.py --task storyteller")
        log("  python evaluate.py --task translator")
    else:
        log("部分任務失敗，請檢查 logs/ 目錄下的對應 log 檔。")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
