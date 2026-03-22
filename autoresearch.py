"""
autoresearch.py — AutoResearch 自動化超參數實驗迴圈

仿照 Karpathy's autoresearch 模式：
  program.md 定義研究目標 → LLM 顧問讀取歷史結果 → 決定下一組超參數
  → 執行快速訓練（--max-steps 75 ≈ 5 分鐘）→ 評估 → 記錄 → 迴圈

使用方式：
  python autoresearch.py --task analyst --mode sft
  python autoresearch.py --task reasoning --mode grpo --max-iterations 5
  python autoresearch.py --task analyst --mode sft --full      # 完整訓練，每次數小時
  python autoresearch.py --list                                # 列出歷史迭代紀錄

LLM 顧問選項（--advisor）：
  gemini      使用 Gemini API（預設，需 GEMINI_API_KEY）
  claude-api  使用 Claude Anthropic API（需 ANTHROPIC_API_KEY + pip install anthropic）
  subagent    透過 Claude Code CLI 呼叫 Claude sub-agent（需 claude 指令在 PATH）

環境需求（依顧問而異）：
  set GEMINI_API_KEY=your_key_here      # gemini（預設）
  set ANTHROPIC_API_KEY=your_key_here  # claude-api
  # subagent 不需要 API key，直接調用 claude CLI
"""

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time

# Windows cp950 終端機相容：強制 stdout 使用 UTF-8
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 常數 ────────────────────────────────────────────────────────────────────────

QUICK_MAX_STEPS = 75     # 快速實驗：每次約 5 分鐘（RTX 3060）
VRAM_LIMIT_GB   = 11.0   # RTX 3060 12GB，留 1GB 給 OS
LOG_PATH        = "autoresearch_log.jsonl"

# 各任務的主要 / 次要評估指標說明（送給 Claude 參考）
TASK_METRICS = {
    "analyst":           "primary=eval_loss（越低越好）; secondary=json_parse_success_rate, entity_grounding_rate（越高越好）",
    "reasoning":         "primary=eval_loss（越低越好）; secondary=format_compliance_rate, json_validity_rate, arithmetic_accuracy（越高越好）",
    "translator":        "primary=eval_loss（越低越好）; secondary=avg_bleu（越高越好）",
    "storyteller":       "primary=eval_loss（越低越好）; secondary=文風樣本（人工審查）",
    "storyteller_extra": "primary=eval_loss（越低越好）; secondary=文風樣本（人工審查）",
}


# ── TSV / JSON 讀取 ──────────────────────────────────────────────────────────────

def _load_tsv(path: str, task: str) -> list[dict]:
    """通用 TSV 讀取器，篩選指定任務的歷史記錄。"""
    if not os.path.exists(path):
        return []
    results = []
    with open(path, encoding="utf-8") as f:
        header = None
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                continue
            if len(fields) < len(header):
                fields += [""] * (len(header) - len(fields))
            row = dict(zip(header, fields))
            if row.get("task") == task:
                results.append(row)
    return results


def load_sft_results(task: str) -> list[dict]:
    return _load_tsv("results.tsv", task)


def load_rl_results(task: str) -> list[dict]:
    return _load_tsv("results_rl.tsv", task)


def load_eval_report(adapter_path: str) -> dict | None:
    """讀取 evaluate.py 輸出的 eval_report.json。"""
    report_path = os.path.join(adapter_path, "eval_report.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path, encoding="utf-8") as f:
        return json.load(f)


def load_autoresearch_log(task: str, mode: str) -> list[dict]:
    """讀取本次 AutoResearch 的歷史迭代紀錄（同 task + mode）。"""
    if not os.path.exists(LOG_PATH):
        return []
    records = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("task") == task and rec.get("mode") == mode:
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    return records


def append_log(record: dict):
    """附加一條迭代紀錄到 autoresearch_log.jsonl。"""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── 子程序執行 ────────────────────────────────────────────────────────────────────

def run_subprocess(cmd: list[str], timeout: int = 7200) -> tuple[int, str]:
    """
    執行子程序，即時輸出到 terminal，同時捕獲完整 stdout（含 stderr）。
    返回 (returncode, stdout_text)。
    """
    print(f"\n[RUN] {' '.join(cmd)}\n{'─' * 60}")
    full_stdout: list[str] = []
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # 合併 stderr，方便即時顯示
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    for line in process.stdout:
        print(line, end="", flush=True)
        full_stdout.append(line)
    process.wait(timeout=timeout)
    print(f"{'─' * 60}")
    return process.returncode, "".join(full_stdout)


def parse_result_json(stdout: str) -> dict | None:
    """從 train_lora.py / train_grpo.py 的 stdout 解析 [RESULT_JSON] 區塊。"""
    lines = stdout.splitlines()
    capture = False
    json_buf: list[str] = []
    for line in lines:
        if "[RESULT_JSON]" in line:
            capture = True
            continue
        if capture:
            json_buf.append(line)
            try:
                return json.loads("\n".join(json_buf))
            except json.JSONDecodeError:
                pass    # 繼續收集更多行
    return None


# ── LLM 顧問（Gemini / Claude API / Sub-agent） ──────────────────────────────────

def _parse_advisor_json(text: str, advisor_label: str) -> dict | None:
    """從 LLM 回應中提取 JSON 超參數建議（三種解析策略）。"""
    # 優先從 ```json ... ``` 提取
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 嘗試直接解析整個回應
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # 找第一個 {...} 區塊
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    print(f"[WARN] 無法從 {advisor_label} 回應中解析 JSON，回應片段：\n{text[:400]}")
    return None


def ask_gemini(prompt: str, model: str = "gemini-2.5-flash") -> dict | None:
    """
    呼叫 Gemini API，解析回傳的 JSON 超參數建議。
    需要環境變數 GEMINI_API_KEY 與套件 google-generativeai。
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("[ERROR] 未安裝 google-generativeai，請執行：pip install google-generativeai")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] 未設定 GEMINI_API_KEY 環境變數。")
        return None

    genai.configure(api_key=api_key)
    try:
        model_obj = genai.GenerativeModel(model)
        response = model_obj.generate_content(prompt)
        text = response.text
        return _parse_advisor_json(text, "Gemini")
    except Exception as e:
        print(f"[ERROR] Gemini API 呼叫失敗: {e}")
        return None


def ask_claude_api(prompt: str, model: str = "claude-sonnet-4-6") -> dict | None:
    """
    呼叫 Claude Anthropic API，解析回傳的 JSON 超參數建議。
    需要環境變數 ANTHROPIC_API_KEY 與套件 anthropic。
    """
    try:
        import anthropic
    except ImportError:
        print("[ERROR] 未安裝 anthropic，請執行：pip install anthropic")
        return None

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[ERROR] 未設定 ANTHROPIC_API_KEY 環境變數。")
        return None

    client = anthropic.Anthropic()
    try:
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text
        return _parse_advisor_json(text, "Claude API")
    except Exception as e:
        print(f"[ERROR] Claude API 呼叫失敗: {e}")
        return None


def ask_claude_subagent(prompt: str) -> dict | None:
    """
    透過 Claude Code CLI（`claude -p`）呼叫 Claude sub-agent 分析超參數。
    不需要 API key，直接調用本機已登入的 claude 指令。
    Prompt 寫入暫存檔以繞過命令列長度限制。
    """
    # 寫入暫存檔，避免 Windows 命令列長度限制 (~32767 chars)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(prompt)
            tmp_path = f.name

        # 使用 --print 模式（等同 -p），讀取暫存檔內容後送出
        result = subprocess.run(
            ["claude", "--print", f"以下是分析任務，請直接輸出 JSON 回應，不要其他文字：\n\n{prompt}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )

        if result.returncode != 0:
            print(f"[ERROR] Claude sub-agent 呼叫失敗（returncode={result.returncode}）")
            if result.stderr:
                print(f"[ERROR] stderr: {result.stderr[:300]}")
            return None

        text = result.stdout
        return _parse_advisor_json(text, "Claude sub-agent")

    except FileNotFoundError:
        print("[ERROR] 找不到 `claude` 指令，請確認 Claude Code CLI 已安裝並在 PATH 中。")
        return None
    except subprocess.TimeoutExpired:
        print("[ERROR] Claude sub-agent 超時（>180s）。")
        return None
    except Exception as e:
        print(f"[ERROR] Claude sub-agent 呼叫失敗: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def ask_advisor(prompt: str, advisor: str, model: str) -> dict | None:
    """
    統一入口：依據 --advisor 參數路由到對應的 LLM 顧問。
    advisor: 'gemini' | 'claude-api' | 'subagent'
    """
    if advisor == "gemini":
        return ask_gemini(prompt, model=model)
    elif advisor == "claude-api":
        return ask_claude_api(prompt, model=model)
    elif advisor == "subagent":
        return ask_claude_subagent(prompt)
    else:
        print(f"[ERROR] 未知的顧問類型：{advisor}")
        return None


# ── 超參數建議提示建構 ────────────────────────────────────────────────────────────

def build_advisor_prompt(
    task: str,
    mode: str,
    sft_history: list[dict],
    rl_history: list[dict],
    ar_log: list[dict],
    quick: bool,
) -> str:
    """構建送給 Claude 的分析提示，要求輸出下一組超參數建議。"""
    L: list[str] = []

    L.append(f"# AutoResearch 超參數顧問")
    L.append(f"任務: **{task}**　模式: **{mode}**　快速實驗: **{'是 (max-steps=' + str(QUICK_MAX_STEPS) + ')' if quick else '否（完整訓練）'}**")
    L.append("")
    L.append("## 硬體約束")
    L.append(f"- GPU: RTX 3060 12GB VRAM")
    L.append(f"- VRAM 峰值需 < {VRAM_LIMIT_GB} GB（留 1 GB 給 OS）")
    L.append(f"- GRPO 注意：`num_generations × max_completion` 決定額外 VRAM 用量")
    L.append("")
    L.append("## 評估指標")
    L.append(f"- {TASK_METRICS.get(task, 'eval_loss 越低越好')}")
    L.append("")

    # ── SFT 歷史 ──
    if sft_history:
        L.append("## 歷史 SFT 實驗（results.tsv，最近 10 筆）")
        L.append("| timestamp | rank | alpha | lr | epochs | train_loss | eval_loss | peak_mem_gb |")
        L.append("|-----------|------|-------|----|--------|------------|-----------|-------------|")
        for r in sft_history[-10:]:
            L.append(
                f"| {r.get('timestamp','')} | {r.get('rank','')} | {r.get('alpha','')} "
                f"| {r.get('lr','')} | {r.get('epochs','')} | {r.get('train_loss','')} "
                f"| {r.get('eval_loss','')} | {r.get('peak_memory_gb','')} |"
            )
        L.append("")

    # ── RL 歷史 ──
    if rl_history:
        L.append("## 歷史 RL 實驗（results_rl.tsv，最近 10 筆）")
        L.append("| timestamp | mode | rank | lr | num_gen | max_comp | kl | train_loss | peak_mem_gb |")
        L.append("|-----------|------|------|----|---------|----------|----|------------|-------------|")
        for r in rl_history[-10:]:
            L.append(
                f"| {r.get('timestamp','')} | {r.get('mode','')} | {r.get('rank','')} "
                f"| {r.get('lr','')} | {r.get('num_generations','')} | {r.get('max_completion','')} "
                f"| {r.get('kl_coeff','')} | {r.get('train_loss','')} | {r.get('peak_memory_gb','')} |"
            )
        L.append("")

    # ── AutoResearch 迭代紀錄 ──
    if ar_log:
        L.append("## 本輪 AutoResearch 迭代紀錄")
        for i, rec in enumerate(ar_log):
            L.append(f"\n### 迭代 {i + 1}（{rec.get('timestamp', '')}）")
            L.append(f"- 理由：{rec.get('claude_reason', '')}")
            L.append(f"- 參數：`{json.dumps(rec.get('params', {}), ensure_ascii=False)}`")
            L.append(f"- train_loss={rec.get('train_loss')}  eval_loss={rec.get('eval_loss')}  peak_mem={rec.get('peak_memory_gb')} GB")
            er = rec.get("eval_report") or {}
            if er:
                quality_str = json.dumps(er.get("quality", {}), ensure_ascii=False)
                quality_str = quality_str[:300] + ("..." if len(quality_str) > 300 else "")
                L.append(f"- perplexity={er.get('perplexity')}  quality={quality_str}")
        L.append("")

    # ── 指令 ──
    L.append("## 你的任務")
    L.append("根據以上歷史數據，選出下一組最有潛力的超參數組合。")
    L.append("若最近 3 次迭代 eval_loss 下降幅度均 < 0.005，或已達到良好指標，請回報 `action=converged`。")
    L.append("避免重複嘗試已測試過且效果不佳的參數組合。")
    L.append("")

    if mode == "sft":
        L.append("### SFT 可調整參數（train_lora.py）")
        L.append("| 參數 | CLI flag | 建議範圍 |")
        L.append("|------|----------|----------|")
        L.append("| LoRA rank | `--rank` | 8 / 16 / 32 / 64 |")
        L.append("| LoRA alpha | `--alpha` | = rank 或 2×rank |")
        L.append("| Learning rate | `--lr` | 1e-5 ~ 5e-4 |")
        L.append("| Epochs | `--epochs` | 1 ~ 5 |")
        L.append("| Max seq len | `--max-seq-len` | 512 / 1024 |")
        L.append("| Batch size | `--batch-size` | 1 / 2 |")
        L.append("")
        L.append("請以 JSON 格式回應（僅輸出 JSON，不要其他文字）：")
        L.append('```json')
        L.append('{"action": "continue", "reason": "簡短分析說明", "params": {"rank": 16, "alpha": 32, "lr": 3e-4, "epochs": 2, "max_seq_len": 1024, "batch_size": 2}}')
        L.append('```')
        L.append('或收斂時：')
        L.append('```json')
        L.append('{"action": "converged", "reason": "已連續 3 次未改善，eval_loss 已穩定在 X", "params": {}}')
        L.append('```')

    elif mode == "grpo":
        L.append("### GRPO 可調整參數（train_grpo.py）")
        L.append("| 參數 | CLI flag | 建議範圍 |")
        L.append("|------|----------|----------|")
        L.append("| LoRA rank | `--rank` | 8 / 16 / 32 |")
        L.append("| LoRA alpha | `--alpha` | = rank 或 2×rank |")
        L.append("| Learning rate | `--lr` | 1e-6 ~ 2e-5 |")
        L.append("| 候選生成數 | `--num-generations` | 4 / 6 |")
        L.append("| 最大生成長度 | `--max-completion` | 128 / 256 / 384 |")
        L.append("| KL 係數 | `--kl-coeff` | 0.05 / 0.1 / 0.2 |")
        L.append("| SFT adapter 起點 | `--sft-adapter` | outputs/lora_{task}（建議填入）|")
        L.append("")
        L.append("VRAM 估算參考：num_generations=6 & max_completion=128 ≈ 10-11 GB；如超過請降低其中一個。")
        L.append("")
        L.append("請以 JSON 格式回應（僅輸出 JSON，不要其他文字）：")
        L.append('```json')
        L.append('{"action": "continue", "reason": "簡短分析說明", "params": {"rank": 8, "alpha": 16, "lr": 1e-5, "num_generations": 6, "max_completion": 128, "kl_coeff": 0.05, "sft_adapter": "outputs/lora_analyst"}}')
        L.append('```')

    return "\n".join(L)




# ── 命令組合 ──────────────────────────────────────────────────────────────────────

def build_train_cmd(task: str, mode: str, params: dict, quick: bool) -> list[str]:
    """根據模式和參數組合訓練命令。"""
    p = params

    if mode == "sft":
        cmd = [sys.executable, "train_lora.py", "--task", task]
        if "rank"        in p: cmd += ["--rank",        str(p["rank"])]
        if "alpha"       in p: cmd += ["--alpha",       str(p["alpha"])]
        if "lr"          in p: cmd += ["--lr",          str(p["lr"])]
        if "epochs"      in p: cmd += ["--epochs",      str(p["epochs"])]
        if "max_seq_len" in p: cmd += ["--max-seq-len", str(p["max_seq_len"])]
        if "batch_size"  in p: cmd += ["--batch-size",  str(p["batch_size"])]
        if quick:              cmd += ["--max-steps",   str(QUICK_MAX_STEPS)]

    elif mode == "grpo":
        cmd = [sys.executable, "train_grpo.py", "--task", task]
        if "rank"            in p: cmd += ["--rank",            str(p["rank"])]
        if "alpha"           in p: cmd += ["--alpha",           str(p["alpha"])]
        if "lr"              in p: cmd += ["--lr",              str(p["lr"])]
        if "num_generations" in p: cmd += ["--num-generations", str(p["num_generations"])]
        if "max_completion"  in p: cmd += ["--max-completion",  str(p["max_completion"])]
        if "kl_coeff"        in p: cmd += ["--kl-coeff",        str(p["kl_coeff"])]
        if "sft_adapter"     in p: cmd += ["--sft-adapter",     str(p["sft_adapter"])]
        if quick:                  cmd += ["--max-steps",       str(QUICK_MAX_STEPS)]

    else:
        raise ValueError(f"不支援的模式：{mode}")

    return cmd


def build_eval_cmd(task: str, mode: str) -> list[str]:
    """組合 evaluate.py 命令。"""
    cmd = [sys.executable, "evaluate.py", "--task", task]
    if mode == "grpo":
        cmd.append("--rl")
    elif mode == "dpo":
        cmd.append("--dpo")
    return cmd


def get_adapter_path(task: str, mode: str) -> str:
    """返回對應模式的 adapter 輸出路徑。"""
    if mode == "grpo":
        return f"outputs/lora_{task}_grpo"
    if mode == "dpo":
        return f"outputs/lora_{task}_dpo"
    return f"outputs/lora_{task}"


# ── 輔助顯示 ──────────────────────────────────────────────────────────────────────

def print_banner(task: str, mode: str, max_iter: int, quick: bool):
    w = 62
    print("=" * w)
    print(f"{'AutoResearch — LoRA 自動超參數實驗迴圈':^{w}}")
    print("=" * w)
    print(f"  任務：{task}")
    print(f"  模式：{mode.upper()}")
    print(f"  最大迭代：{max_iter}")
    print(f"  訓練模式：{'快速 (max-steps=' + str(QUICK_MAX_STEPS) + ')' if quick else '完整（耗時數小時）'}")
    print("=" * w)
    print()


def cmd_list():
    """列出所有歷史 AutoResearch 紀錄。"""
    if not os.path.exists(LOG_PATH):
        print("尚無 AutoResearch 實驗紀錄。")
        return
    records: list[dict] = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not records:
        print("尚無 AutoResearch 實驗紀錄。")
        return

    header = f"{'#':<4} {'task':<18} {'mode':<6} {'iter':<6} {'eval_loss':<10} {'ppl':<8} {'status':<12} {'timestamp':<20} params"
    print(header)
    print("-" * 110)
    for i, rec in enumerate(records):
        er = rec.get("eval_report") or {}
        ppl = er.get("perplexity", "")
        params_str = json.dumps(rec.get("params", {}), ensure_ascii=False)
        params_str = params_str[:50] + ("…" if len(params_str) > 50 else "")
        print(
            f"{i+1:<4} {rec.get('task',''):<18} {rec.get('mode',''):<6} "
            f"{rec.get('iteration',''):<6} {str(rec.get('eval_loss','')):<10} "
            f"{str(ppl):<8} {rec.get('status',''):<12} {rec.get('timestamp',''):<20} {params_str}"
        )


# ── 主迴圈 ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoResearch — LoRA 自動超參數實驗迴圈",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task", choices=["analyst", "reasoning", "translator", "storyteller", "storyteller_extra"],
        help="訓練任務",
    )
    parser.add_argument(
        "--mode", default="sft", choices=["sft", "grpo"],
        help="訓練模式（預設 sft）",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="最大迭代次數（預設 5）",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="完整訓練模式（不限制 max-steps，每次訓練數小時）",
    )
    parser.add_argument(
        "--no-eval", action="store_true",
        help="跳過 evaluate.py（加快迭代速度，但缺少品質指標）",
    )
    parser.add_argument(
        "--advisor", default="gemini",
        choices=["gemini", "claude-api", "subagent"],
        help=(
            "LLM 顧問後端（預設 gemini）：\n"
            "  gemini     使用 Gemini API（需 GEMINI_API_KEY）\n"
            "  claude-api 使用 Claude Anthropic API（需 ANTHROPIC_API_KEY）\n"
            "  subagent   透過 Claude Code CLI 呼叫 Claude sub-agent"
        ),
    )
    parser.add_argument(
        "--model", default="",
        help=(
            "LLM 模型名稱（留空使用各顧問預設值）：\n"
            "  gemini 預設：gemini-2.5-flash\n"
            "  claude-api 預設：claude-sonnet-4-6\n"
            "  subagent：忽略此參數"
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出歷史 AutoResearch 迭代紀錄後退出",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── --list 模式 ──────────────────────────────────────────────────────────────
    if args.list:
        cmd_list()
        return

    if not args.task:
        print("[ERROR] 必須指定 --task 參數。使用 --help 查看說明。")
        sys.exit(1)

    # ── 環境檢查（依顧問類型） ───────────────────────────────────────────────────
    if args.advisor == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        print("[ERROR] 未設定 GEMINI_API_KEY 環境變數。")
        print("  Windows: set GEMINI_API_KEY=your_key_here")
        print("  Linux:   export GEMINI_API_KEY=your_key_here")
        sys.exit(1)
    elif args.advisor == "claude-api" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("[ERROR] 未設定 ANTHROPIC_API_KEY 環境變數。")
        print("  Windows: set ANTHROPIC_API_KEY=your_key_here")
        print("  Linux:   export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    # 各顧問預設模型名稱
    advisor_model = args.model or {
        "gemini": "gemini-2.5-flash",
        "claude-api": "claude-sonnet-4-6",
        "subagent": "",
    }.get(args.advisor, "")

    quick = not args.full
    print_banner(args.task, args.mode, args.max_iterations, quick)
    print(f"  顧問後端：{args.advisor}" + (f"（{advisor_model}）" if advisor_model else ""))

    total_start = time.time()
    converged   = False

    for iteration in range(1, args.max_iterations + 1):
        iter_start = time.time()
        print(f"\n{'─' * 62}")
        print(f"  迭代 {iteration} / {args.max_iterations}  ({time.strftime('%H:%M:%S')})")
        print(f"{'─' * 62}")

        # ── Step 1：讀取歷史資料 ────────────────────────────────────────────────
        sft_history = load_sft_results(args.task)
        rl_history  = load_rl_results(args.task)
        ar_log      = load_autoresearch_log(args.task, args.mode)

        print(f"[State] SFT 歷史={len(sft_history)} 筆  RL 歷史={len(rl_history)} 筆  本輪迭代={len(ar_log)} 次")

        # ── Step 2：詢問 LLM 顧問 ──────────────────────────────────────────────
        advisor_label = {
            "gemini": "Gemini", "claude-api": "Claude API", "subagent": "Claude sub-agent"
        }.get(args.advisor, args.advisor)
        print(f"\n[{advisor_label}] 分析歷史結果，建議下一組超參數...")
        prompt = build_advisor_prompt(
            args.task, args.mode,
            sft_history, rl_history, ar_log,
            quick=quick,
        )
        advice = ask_advisor(prompt, advisor=args.advisor, model=advisor_model)

        if advice is None:
            print(f"[WARN] {advisor_label} 未能提供建議，本次迭代使用空參數（預設值）繼續。")
            advice = {"action": "continue", "reason": "API 失敗，使用訓練腳本預設值", "params": {}}

        action = advice.get("action", "continue")
        reason = advice.get("reason", "")
        params = advice.get("params", {})

        print(f"[{advisor_label}] 決策: {action}")
        print(f"[{advisor_label}] 理由: {reason}")
        print(f"[{advisor_label}] 建議參數: {json.dumps(params, ensure_ascii=False)}")

        if action == "converged":
            print(f"\n[AutoResearch] Claude 判斷已收斂，提前終止實驗。")
            converged = True
            break

        # ── Step 3：執行訓練 ────────────────────────────────────────────────────
        train_cmd  = build_train_cmd(args.task, args.mode, params, quick=quick)
        returncode, stdout = run_subprocess(train_cmd)

        if returncode != 0:
            print(f"[ERROR] 訓練失敗（returncode={returncode}），跳過本次迭代。")
            append_log({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "task": args.task, "mode": args.mode,
                "iteration": iteration, "params": params,
                "advisor": args.advisor, "claude_reason": reason, "status": "train_failed",
            })
            continue

        # 解析訓練結果
        train_result   = parse_result_json(stdout) or {}
        train_loss     = train_result.get("train_loss")
        eval_loss      = train_result.get("eval_loss")
        peak_memory_gb = train_result.get("peak_memory_gb")

        print(f"\n[Train] train_loss={train_loss}  eval_loss={eval_loss}  peak_mem={peak_memory_gb} GB")

        if peak_memory_gb and float(peak_memory_gb) > VRAM_LIMIT_GB:
            print(f"[WARN] VRAM 峰值 {peak_memory_gb} GB 超過限制 {VRAM_LIMIT_GB} GB！"
                  "下次迭代請降低 rank 或 batch_size。")

        # ── Step 4：評估 ────────────────────────────────────────────────────────
        eval_report = None
        if not args.no_eval:
            eval_cmd = build_eval_cmd(args.task, args.mode)
            eval_rc, _ = run_subprocess(eval_cmd)
            if eval_rc == 0:
                adapter_path = get_adapter_path(args.task, args.mode)
                eval_report  = load_eval_report(adapter_path)
                if eval_report:
                    print(f"[Eval] perplexity={eval_report.get('perplexity')}  avg_loss={eval_report.get('avg_loss')}")
            else:
                print(f"[WARN] evaluate.py 失敗（returncode={eval_rc}），略過評估。")

        # ── Step 5：記錄 ────────────────────────────────────────────────────────
        iter_elapsed = time.time() - iter_start
        log_record = {
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "task":           args.task,
            "mode":           args.mode,
            "iteration":      iteration,
            "params":         params,
            "advisor":        args.advisor,
            "train_loss":     train_loss,
            "eval_loss":      eval_loss,
            "peak_memory_gb": peak_memory_gb,
            "eval_report":    eval_report,
            "elapsed_sec":    round(iter_elapsed),
            "status":         "ok",
            "claude_reason":  reason,
        }
        append_log(log_record)
        print(f"\n[Log] 迭代 {iteration} 完成，耗時 {iter_elapsed / 60:.1f} 分鐘。紀錄已附加至 {LOG_PATH}")

    # ── 結束摘要 ──────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    w = 62
    print(f"\n{'=' * w}")
    print(f"{'AutoResearch 完成':^{w}}")
    print(f"{'=' * w}")
    print(f"  總耗時：{total_elapsed / 60:.1f} 分鐘")
    print(f"  狀態：{'已收斂（提前終止）' if converged else f'完成 {args.max_iterations} 次迭代'}")

    # 找最佳結果
    ar_log = load_autoresearch_log(args.task, args.mode)
    best: dict | None = None
    for rec in ar_log:
        if rec.get("eval_loss") is not None and rec.get("status") == "ok":
            if best is None or float(rec["eval_loss"]) < float(best["eval_loss"]):
                best = rec

    if best:
        print(f"\n  最佳迭代：第 {best['iteration']} 次")
        print(f"  最佳參數：{json.dumps(best['params'], ensure_ascii=False)}")
        print(f"  最佳 eval_loss：{best['eval_loss']}")
        er = best.get("eval_report") or {}
        if er.get("perplexity"):
            print(f"  最佳 perplexity：{er['perplexity']}")

    print(f"{'=' * w}")
    print(f"\n  使用 `python autoresearch.py --list` 查看完整歷史紀錄")
    print(f"  使用 `python deploy_lora.py --task {args.task}` 部署最佳 adapter")


if __name__ == "__main__":
    main()
