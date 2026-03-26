"""
deploy_lora.py — 將訓練完成的 LoRA adapter 部署至 Ollama

功能：
  1. 掃描 outputs/lora_{task}/ 確認 adapter 存在
  2. 以 Unsloth 載入 base model + LoRA，合併後匯出 GGUF
  3. 產生 Ollama Modelfile（Qwen 2.5 chat template）
  4. 執行 ollama create 註冊模型至本機 Ollama
  5. 可選：更新 DND-like_RPG/engine/config.py 的 LLM_MODEL_NAME
  6. 狀態追蹤：outputs/deploy_state.json 記錄已部署版本

使用方式：
  python deploy_lora.py --task analyst
  python deploy_lora.py --task storyteller --quant q5_k_m
  python deploy_lora.py --task analyst --update-config
  python deploy_lora.py --all
  python deploy_lora.py --status
  python deploy_lora.py --dry-run --task analyst
"""

import argparse
import datetime
import json
import os
import sys
import pathlib
import re
import subprocess
import sys

# ── 常數 ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = pathlib.Path(__file__).parent.resolve()
OUTPUTS_DIR  = SCRIPT_DIR / "outputs"
STATE_FILE   = OUTPUTS_DIR / "deploy_state.json"
RPG_CONFIG   = SCRIPT_DIR.parent / "DND-like_RPG" / "engine" / "config.py"

BASE_MODEL   = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LEN  = 4096

TASKS = ["analyst", "reasoning", "translator", "storyteller", "storyteller_extra"]

QUANT_METHODS = ["q4_k_m", "q5_k_m", "q8_0", "f16"]
DEFAULT_QUANT = "q4_k_m"

# RL adapter 目錄對照表（task → RL adapter dir suffix）
RL_ADAPTER_SUFFIX = {
    "analyst":    "lora_analyst_grpo",
    "reasoning":  "lora_reasoning_grpo",
    "translator": "lora_translator_grpo",
    "storyteller": "lora_storyteller_dpo",
}

# Ollama 模型命名規則：dnd-{task}
def ollama_model_name(task):
    return f"dnd-{task.replace('_', '-')}"

# Unsloth 匯出 GGUF 後的檔名格式
def gguf_filename(quant):
    return f"unsloth.{quant.upper()}.gguf"

# Qwen 2.5 instruct chat template for Ollama Modelfile
_QWEN25_TEMPLATE = """\
<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

# ── ANSI 顏色 ─────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"


def _log(level, msg, *args):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color = {
        "INFO":  "",
        "OK":    GREEN,
        "WARN":  YELLOW,
        "ERROR": RED,
        "STEP":  CYAN + BOLD,
    }.get(level, "")
    text = msg % args if args else msg
    print(f"{ts} [{color}{level:<5}{RESET}] {text}")


# ── 狀態管理 ──────────────────────────────────────────────────────────────────

class DeployState:
    """
    outputs/deploy_state.json 記錄每個 task 的部署資訊。

    Schema per entry:
    {
      "task":        str,
      "ollama_name": str,
      "quant":       str,
      "gguf_path":   str,
      "deployed_at": str,   # ISO-8601
      "adapter_mtime": float
    }
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self._data = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get(self, task):
        return self._data.get(task)

    def record(self, task, ollama_name, quant, gguf_path, adapter_mtime):
        self._data[task] = {
            "task":          task,
            "ollama_name":   ollama_name,
            "quant":         quant,
            "gguf_path":     str(gguf_path),
            "deployed_at":   datetime.datetime.now().isoformat(timespec="seconds"),
            "adapter_mtime": adapter_mtime,
        }

    def is_up_to_date(self, task, adapter_dir, quant):
        entry = self._data.get(task)
        if not entry:
            return False
        if entry.get("quant") != quant:
            return False
        adapter_file = pathlib.Path(adapter_dir) / "adapter_model.safetensors"
        if not adapter_file.exists():
            return False
        current_mtime = adapter_file.stat().st_mtime
        return abs(current_mtime - entry.get("adapter_mtime", 0)) < 1.0

    def reset(self, task):
        if task in self._data:
            del self._data[task]
            return True
        return False


# ── Adapter 驗證 ───────────────────────────────────────────────────────────────

def adapter_dir(task, prefer_rl=False):
    if prefer_rl and task in RL_ADAPTER_SUFFIX:
        rl_dir = OUTPUTS_DIR / RL_ADAPTER_SUFFIX[task]
        if rl_dir.exists():
            return rl_dir
    return OUTPUTS_DIR / f"lora_{task}"


def check_adapter(task, prefer_rl=False):
    """
    回傳 (ok: bool, reason: str)。
    ok=True 表示 adapter 完整可用。
    """
    d = adapter_dir(task, prefer_rl)
    if not d.exists():
        return False, f"目錄不存在：{d}"
    required = ["adapter_config.json", "adapter_model.safetensors"]
    for fname in required:
        if not (d / fname).exists():
            return False, f"缺少必要檔案：{fname}"
    return True, "OK"


def adapter_mtime(task, prefer_rl=False):
    f = adapter_dir(task, prefer_rl) / "adapter_model.safetensors"
    return f.stat().st_mtime if f.exists() else 0.0


# ── GGUF 匯出 ─────────────────────────────────────────────────────────────────

def gguf_dir(task):
    return OUTPUTS_DIR / f"gguf_{task}"


LLAMA_CPP_DIR = pathlib.Path("C:/Users/user/llama.cpp")
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"


def export_gguf(task, quant, dry_run, prefer_rl=False):
    """
    Merge LoRA → float16 safetensors via Unsloth, then convert to GGUF
    using llama.cpp's convert_hf_to_gguf.py (pure Python, no compiled binary).
    Quantization is done in the same step (q8_0 supported natively; q4_k_m
    falls back to q8_0 since llama-quantize is not built).
    回傳 gguf_path (Path) 或在失敗時 raise RuntimeError。
    """
    # q4_k_m requires compiled llama-quantize; fall back to q8_0 (Python-only)
    gguf_quant = quant if quant in ("f16", "bf16", "f32", "q8_0") else "q8_0"
    if gguf_quant != quant:
        _log("WARN", "llama-quantize 未編譯，改用 %s（原請求：%s）", gguf_quant, quant)

    out_dir = gguf_dir(task)
    merge_dir = out_dir / "merged_f16"
    gguf_path = out_dir / f"model-{gguf_quant.upper()}.gguf"

    if dry_run:
        _log("INFO", "[DRY-RUN] 跳過 GGUF 匯出：%s", gguf_path)
        return gguf_path

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    _log("STEP", "載入 base model + LoRA adapter（task=%s）…", task)
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise RuntimeError("找不到 unsloth 套件。請先安裝：pip install unsloth")

    src_adapter = adapter_dir(task, prefer_rl)
    _log("INFO", "使用 adapter：%s", src_adapter)

    # Force all 4-bit layers onto GPU (device_map="auto" causes CPU offload which
    # bitsandbytes 4-bit does not support)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(src_adapter),
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )

    # Step 1: merge LoRA into float16 safetensors locally
    _log("STEP", "合併 LoRA 至 float16 safetensors → %s …", merge_dir)
    merge_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(str(merge_dir), tokenizer, save_method="merged_16bit")

    # Verify all shards were written
    import json as _json
    idx_file = merge_dir / "model.safetensors.index.json"
    if idx_file.exists():
        with idx_file.open() as _f:
            expected = set(_json.load(_f)["weight_map"].values())
        missing = [s for s in expected if not (merge_dir / s).exists()]
        if missing:
            raise RuntimeError(f"Merge 不完整，缺少 shards：{missing}")
    elif not (merge_dir / "model.safetensors").exists():
        raise RuntimeError(f"Merge 失敗：找不到 safetensors 於 {merge_dir}")

    # Step 2: convert to GGUF using llama.cpp Python script
    _log("STEP", "轉換 GGUF（%s）via convert_hf_to_gguf.py …", gguf_quant)
    import subprocess as _sub
    result = _sub.run(
        [sys.executable, str(CONVERT_SCRIPT),
         str(merge_dir),
         "--outtype", gguf_quant,
         "--outfile", str(gguf_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"convert_hf_to_gguf.py 失敗（exit {result.returncode}）:\n{result.stderr[-500:]}"
        )

    # Step 3: remove float16 shards to reclaim ~14 GB
    import shutil as _shutil
    _shutil.rmtree(merge_dir, ignore_errors=True)
    _log("INFO", "已刪除 float16 中間檔（~14 GB 回收）")

    size_gb = gguf_path.stat().st_size / 1_073_741_824
    _log("OK", "GGUF 匯出完成：%s  (%.2f GB)", gguf_path, size_gb)
    return gguf_path


# ── Ollama Modelfile ──────────────────────────────────────────────────────────

def write_modelfile(task, gguf_path, dry_run):
    """產生 Ollama Modelfile，回傳 modelfile_path。"""
    mf_path = gguf_dir(task) / "Modelfile"
    content = (
        f"FROM {gguf_path}\n\n"
        f'TEMPLATE """{_QWEN25_TEMPLATE}"""\n\n'
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|im_start|>"\n'
        "PARAMETER num_ctx 8192\n"
    )

    if dry_run:
        _log("INFO", "[DRY-RUN] 跳過 Modelfile 寫入：%s", mf_path)
        return mf_path

    gguf_dir(task).mkdir(parents=True, exist_ok=True)
    mf_path.write_text(content, encoding="utf-8")
    _log("INFO", "Modelfile 已寫入：%s", mf_path)
    return mf_path


# ── Ollama 註冊 ───────────────────────────────────────────────────────────────

def register_ollama(task, modelfile_path, dry_run):
    """執行 ollama create <name> -f <Modelfile>。"""
    name = ollama_model_name(task)

    if dry_run:
        _log("INFO", "[DRY-RUN] 跳過 ollama create %s -f %s", name, modelfile_path)
        return name

    _log("STEP", "執行 ollama create %s …", name)
    result = subprocess.run(
        ["ollama", "create", name, "-f", str(modelfile_path)],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama create 失敗（exit code {result.returncode}）")

    _log("OK", "Ollama 模型已註冊：%s", name)
    return name


# ── 更新 DND-like_RPG config.py ───────────────────────────────────────────────

def update_rpg_config(ollama_name, dry_run):
    """將 DND-like_RPG/engine/config.py 的 LLM_MODEL_NAME 更新為 ollama_name。"""
    if not RPG_CONFIG.exists():
        _log("WARN", "找不到 RPG config.py：%s — 跳過更新", RPG_CONFIG)
        return

    original = RPG_CONFIG.read_text(encoding="utf-8")
    updated = re.sub(
        r'(LLM_MODEL_NAME\s*=\s*)["\'][^"\']*["\']',
        rf'\1"{ollama_name}"',
        original,
    )

    if updated == original:
        _log("WARN", "config.py 中找不到 LLM_MODEL_NAME = ... 的樣式，跳過更新")
        return

    if dry_run:
        _log("INFO", "[DRY-RUN] 跳過 config.py 更新（LLM_MODEL_NAME → %s）", ollama_name)
        return

    RPG_CONFIG.write_text(updated, encoding="utf-8")
    _log("OK", "config.py 已更新：LLM_MODEL_NAME = \"%s\"", ollama_name)


# ── 單一任務部署流程 ───────────────────────────────────────────────────────────

def _cleanup_gguf_dir(d: pathlib.Path) -> None:
    """刪除 GGUF 匯出目錄（Ollama 已複製 GGUF，來源不再需要）。"""
    import shutil
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
        _log("INFO", "已清理 GGUF 目錄（磁碟空間回收）：%s", d)


def deploy_task(task, quant, update_config, dry_run, state, force, prefer_rl=False):
    _log("STEP", "═══ 開始部署 task=%s  quant=%s  rl=%s ═══", task, quant, prefer_rl)

    # 1. 驗證 adapter
    ok, reason = check_adapter(task, prefer_rl)
    if not ok:
        _log("ERROR", "Adapter 不完整，跳過（%s）", reason)
        return False

    # 2. 檢查是否已部署且為最新（除非 --force）
    if not force and state.is_up_to_date(task, adapter_dir(task, prefer_rl), quant):
        _log("INFO", "已部署且 adapter 未變更，跳過（--force 可強制重新部署）")
        entry = state.get(task)
        if entry:
            _log("INFO", "當前模型：%s  部署於：%s", entry["ollama_name"], entry["deployed_at"])
        return True

    # 3. 匯出 GGUF
    try:
        gguf_path = export_gguf(task, quant, dry_run, prefer_rl)
    except RuntimeError as e:
        _log("ERROR", "GGUF 匯出失敗：%s", e)
        return False

    # 4. 寫入 Modelfile
    mf_path = write_modelfile(task, gguf_path, dry_run)

    # 5. 註冊至 Ollama
    try:
        ollama_name = register_ollama(task, mf_path, dry_run)
    except RuntimeError as e:
        _log("ERROR", "Ollama 註冊失敗：%s", e)
        return False

    # 6. 更新 RPG config（可選）
    if update_config:
        update_rpg_config(ollama_name, dry_run)

    # 7. 記錄狀態
    if not dry_run:
        state.record(task, ollama_name, quant, gguf_path, adapter_mtime(task, prefer_rl))
        state.save()

    # 8. 清理 GGUF 目錄（Ollama 已複製至自身 blob 存儲，來源可刪除）
    if not dry_run:
        _cleanup_gguf_dir(gguf_dir(task))

    _log("OK", "task=%s 部署完成 → ollama 模型名稱：%s", task, ollama_model_name(task))
    return True


# ── 狀態報告 ──────────────────────────────────────────────────────────────────

def print_status(state):
    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}  LoRA 部署狀態報告  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}")

    deployed = 0
    pending = 0
    missing = 0

    for task in TASKS:
        ok, reason = check_adapter(task)
        entry = state.get(task)

        print(f"\n  {BOLD}{task}{RESET}")

        if not ok:
            print(f"    Adapter:   {RED}不存在 ({reason}){RESET}")
            missing += 1
            continue

        mtime = datetime.datetime.fromtimestamp(adapter_mtime(task)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"    Adapter:   {GREEN}存在{RESET}  (修改時間: {mtime})")

        if entry:
            up_to_date = state.is_up_to_date(task, adapter_dir(task), entry["quant"])
            status = f"{GREEN}已部署 ✓{RESET}" if up_to_date else f"{YELLOW}已部署但 adapter 已更新{RESET}"
            print(f"    狀態:      {status}")
            print(f"    Ollama:    {CYAN}{entry['ollama_name']}{RESET}")
            print(f"    Quant:     {entry['quant']}")
            print(f"    部署時間:  {entry['deployed_at']}")
            gguf = pathlib.Path(entry["gguf_path"])
            gguf_size = f"{gguf.stat().st_size / 1_073_741_824:.2f} GB" if gguf.exists() else "（檔案已移除）"
            print(f"    GGUF:      {gguf_size}")
            deployed += 1
        else:
            print(f"    狀態:      {YELLOW}尚未部署{RESET}")
            pending += 1

    print(f"\n{BOLD}{'─' * 72}{RESET}")
    print(
        f"  合計：{GREEN}{deployed} 已部署{RESET}  "
        f"{YELLOW}{pending} 待部署{RESET}  "
        f"{RED}{missing} adapter 不存在{RESET}"
    )
    print(f"{BOLD}{'=' * 72}{RESET}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="將訓練完成的 LoRA adapter 部署至 Ollama"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--task", choices=TASKS,
        help="指定要部署的任務",
    )
    group.add_argument(
        "--all", action="store_true",
        help="部署所有有 adapter 的任務",
    )
    group.add_argument(
        "--status", action="store_true",
        help="顯示所有任務的部署狀態",
    )
    parser.add_argument(
        "--quant", choices=QUANT_METHODS, default=DEFAULT_QUANT,
        help=f"GGUF 量化方式（預設：{DEFAULT_QUANT}）",
    )
    parser.add_argument(
        "--update-config", action="store_true",
        help="部署後更新 DND-like_RPG/engine/config.py 的 LLM_MODEL_NAME",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="強制重新匯出與部署，即使 adapter 未變更",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="模擬執行，顯示操作但不實際匯出或部署",
    )
    parser.add_argument(
        "--rl", action="store_true",
        help="優先使用 RL adapter（GRPO/DPO），若不存在則回退至 SFT adapter",
    )
    parser.add_argument(
        "--reset", metavar="TASK", choices=TASKS,
        help="重置指定任務的部署記錄（不刪除 GGUF）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    state = DeployState(STATE_FILE)

    # ── --reset ────────────────────────────────────────────────────────────
    if args.reset:
        if state.reset(args.reset):
            state.save()
            _log("OK", "已重置 task=%s 的部署記錄", args.reset)
        else:
            _log("WARN", "task=%s 沒有部署記錄", args.reset)
        return

    # ── --status ───────────────────────────────────────────────────────────
    if args.status:
        print_status(state)
        return

    if args.dry_run:
        _log("WARN", "*** DRY-RUN 模式：不會實際匯出 GGUF 或呼叫 ollama ***")

    # ── --all ──────────────────────────────────────────────────────────────
    if args.all:
        results = {}
        for task in TASKS:
            ok, _ = check_adapter(task, args.rl)
            if not ok:
                _log("INFO", "task=%s 沒有 adapter，跳過", task)
                results[task] = "skip"
                continue
            success = deploy_task(
                task=task,
                quant=args.quant,
                update_config=False,   # --all 時不自動更新 config，避免最後一個 task 覆蓋
                dry_run=args.dry_run,
                state=state,
                force=args.force,
                prefer_rl=args.rl,
            )
            results[task] = "ok" if success else "fail"

        print(f"\n{BOLD}── 部署摘要 ──{RESET}")
        for task, result in results.items():
            color = GREEN if result == "ok" else (YELLOW if result == "skip" else RED)
            print(f"  {task:<20} {color}{result}{RESET}")

        if args.update_config:
            # 優先用 storyteller，若無則用第一個成功的
            preferred = "storyteller"
            target_task = preferred if results.get(preferred) == "ok" else next(
                (t for t, r in results.items() if r == "ok"), None
            )
            if target_task:
                update_rpg_config(ollama_model_name(target_task), args.dry_run)
            else:
                _log("WARN", "沒有成功部署的任務，跳過 config.py 更新")
        return

    # ── --task ─────────────────────────────────────────────────────────────
    if not args.task:
        print("請指定 --task <task>、--all 或 --status")
        sys.exit(1)

    success = deploy_task(
        task=args.task,
        quant=args.quant,
        update_config=args.update_config,
        dry_run=args.dry_run,
        state=state,
        force=args.force,
        prefer_rl=args.rl,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
