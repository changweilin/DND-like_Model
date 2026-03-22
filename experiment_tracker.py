"""
experiment_tracker.py — 實驗結果視覺化分析器

讀取 results.tsv、results_rl.tsv、autoresearch_log.jsonl，
以 Rich 表格呈現各任務訓練歷史，並輸出 HTML 趨勢報告。

使用方式：
    python experiment_tracker.py                  # 顯示全部任務總覽
    python experiment_tracker.py --task analyst   # 只看 analyst
    python experiment_tracker.py --html           # 輸出 HTML 報告到 outputs/
    python experiment_tracker.py --best           # 各任務最佳結果
    python experiment_tracker.py --compare        # SFT vs RL 指標比較

依賴：
    pip install rich        # 終端機彩色表格（必要）
    pip install matplotlib  # HTML 趨勢圖（--html 選項，可選）
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Windows cp950 終端機相容：強制 stdout 使用 UTF-8
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box
    _RICH = True
except ImportError:
    _RICH = False

# ── 路徑 ─────────────────────────────────────────────────────────────────────────

SCRIPT_DIR    = Path(__file__).parent.resolve()
SFT_TSV       = SCRIPT_DIR / "results.tsv"
RL_TSV        = SCRIPT_DIR / "results_rl.tsv"
AR_LOG        = SCRIPT_DIR / "autoresearch_log.jsonl"
OUTPUTS_DIR   = SCRIPT_DIR / "outputs"

ALL_TASKS = ["analyst", "reasoning", "translator", "storyteller", "storyteller_extra"]


# ── 資料讀取 ──────────────────────────────────────────────────────────────────────

def _load_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
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
            rows.append(dict(zip(header, fields)))
    return rows


def load_sft(task: str | None = None) -> list[dict]:
    rows = _load_tsv(SFT_TSV)
    return [r for r in rows if task is None or r.get("task") == task]


def load_rl(task: str | None = None) -> list[dict]:
    rows = _load_tsv(RL_TSV)
    return [r for r in rows if task is None or r.get("task") == task]


def load_ar_log(task: str | None = None, mode: str | None = None) -> list[dict]:
    if not AR_LOG.exists():
        return []
    records: list[dict] = []
    with open(AR_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if (task is None or rec.get("task") == task) and \
                   (mode is None or rec.get("mode") == mode):
                    records.append(rec)
            except json.JSONDecodeError:
                pass
    return records


def load_eval_report(task: str, mode: str = "sft") -> dict | None:
    """讀取 evaluate.py 輸出的 eval_report.json。"""
    if mode == "grpo":
        path = OUTPUTS_DIR / f"lora_{task}_grpo" / "eval_report.json"
    elif mode == "dpo":
        path = OUTPUTS_DIR / f"lora_{task}_dpo" / "eval_report.json"
    else:
        path = OUTPUTS_DIR / f"lora_{task}" / "eval_report.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Rich 顯示工具 ─────────────────────────────────────────────────────────────────

def _loss_color(val_str: str, lower_threshold: float = 0.5) -> str:
    """根據 loss 值決定顏色標記（Rich markup）。"""
    try:
        v = float(val_str)
        if v < lower_threshold:
            return f"[green]{val_str}[/green]"
        if v < lower_threshold * 2:
            return f"[yellow]{val_str}[/yellow]"
        return f"[red]{val_str}[/red]"
    except (ValueError, TypeError):
        return val_str or "—"


def _fmt(val, decimals: int = 4) -> str:
    if val is None or val == "":
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


# ── 顯示模式 ──────────────────────────────────────────────────────────────────────

def show_sft_table(task: str | None, console):
    rows = load_sft(task)
    if not rows:
        console.print(f"[yellow]  尚無 SFT 實驗紀錄（{SFT_TSV.name}）[/yellow]")
        return

    t = Table(
        title=f"SFT 實驗紀錄 {'— ' + task if task else '（全部任務）'}",
        box=box.ROUNDED,
        show_lines=True,
    )
    t.add_column("#",           style="dim",    width=4)
    t.add_column("task",        style="cyan",   width=18)
    t.add_column("rank",        justify="right", width=6)
    t.add_column("alpha",       justify="right", width=6)
    t.add_column("lr",          width=10)
    t.add_column("ep",          justify="right", width=4)
    t.add_column("steps",       justify="right", width=7)
    t.add_column("train_loss",  justify="right", width=11)
    t.add_column("eval_loss",   justify="right", width=10)
    t.add_column("peak_GB",     justify="right", width=9)
    t.add_column("timestamp",   width=20)

    for i, r in enumerate(rows, 1):
        t.add_row(
            str(i),
            r.get("task", ""),
            r.get("rank", ""),
            r.get("alpha", ""),
            r.get("lr", ""),
            r.get("epochs", ""),
            r.get("max_steps", ""),
            Text.from_markup(_loss_color(r.get("train_loss", ""), 1.5)),
            Text.from_markup(_loss_color(r.get("eval_loss", ""), 0.5)),
            r.get("peak_memory_gb", ""),
            r.get("timestamp", ""),
        )

    console.print(t)


def show_rl_table(task: str | None, console):
    rows = load_rl(task)
    if not rows:
        console.print(f"[yellow]  尚無 RL 實驗紀錄（{RL_TSV.name}）[/yellow]")
        return

    t = Table(
        title=f"RL 實驗紀錄（GRPO/DPO）{'— ' + task if task else '（全部任務）'}",
        box=box.ROUNDED,
        show_lines=True,
    )
    t.add_column("#",          style="dim",   width=4)
    t.add_column("task",       style="cyan",  width=18)
    t.add_column("mode",       width=6)
    t.add_column("rank",       justify="right", width=6)
    t.add_column("lr",         width=10)
    t.add_column("num_gen",    justify="right", width=8)
    t.add_column("max_comp",   justify="right", width=9)
    t.add_column("kl",         justify="right", width=6)
    t.add_column("train_loss", justify="right", width=11)
    t.add_column("peak_GB",    justify="right", width=9)
    t.add_column("timestamp",  width=20)

    for i, r in enumerate(rows, 1):
        t.add_row(
            str(i),
            r.get("task", ""),
            r.get("mode", ""),
            r.get("rank", ""),
            r.get("lr", ""),
            r.get("num_generations", ""),
            r.get("max_completion", ""),
            r.get("kl_coeff", ""),
            Text.from_markup(_loss_color(r.get("train_loss", ""), 1.5)),
            r.get("peak_memory_gb", ""),
            r.get("timestamp", ""),
        )

    console.print(t)


def show_best(console):
    """各任務最佳實驗結果。"""
    t = Table(
        title="各任務最佳結果",
        box=box.HEAVY_HEAD,
        show_lines=True,
    )
    t.add_column("task",          style="bold cyan", width=20)
    t.add_column("SFT eval_loss", justify="right",   width=14)
    t.add_column("RL train_loss", justify="right",   width=14)
    t.add_column("perplexity",    justify="right",   width=12)
    t.add_column("quality",       width=40)

    for task in ALL_TASKS:
        sft_rows = load_sft(task)
        rl_rows  = load_rl(task)

        # 最佳 SFT eval_loss
        best_sft = None
        for r in sft_rows:
            try:
                v = float(r.get("eval_loss", "inf") or "inf")
                if best_sft is None or v < float(best_sft.get("eval_loss", "inf") or "inf"):
                    best_sft = r
            except ValueError:
                pass

        # 最新 RL train_loss
        best_rl = rl_rows[-1] if rl_rows else None

        # eval_report
        for mode in ("grpo", "dpo", "sft"):
            report = load_eval_report(task, mode)
            if report:
                break
        else:
            report = None

        sft_loss_str = _fmt(best_sft.get("eval_loss") if best_sft else None)
        rl_loss_str  = _fmt(best_rl.get("train_loss") if best_rl else None)
        ppl_str      = _fmt(report.get("perplexity") if report else None, 2)

        # 品質指標
        quality_str = "—"
        if report:
            q = report.get("quality", {})
            parts = []
            if "json_parse_success_rate" in q:
                parts.append(f"JSON={q['json_parse_success_rate']*100:.0f}%")
            if "format_compliance_rate" in q:
                parts.append(f"格式={q['format_compliance_rate']*100:.0f}%")
            if "avg_bleu" in q and q["avg_bleu"] is not None:
                parts.append(f"BLEU={q['avg_bleu']:.1f}")
            if "arithmetic_accuracy" in q and q["arithmetic_accuracy"] is not None:
                parts.append(f"數值={q['arithmetic_accuracy']*100:.0f}%")
            quality_str = "  ".join(parts) or "—"

        t.add_row(task, sft_loss_str, rl_loss_str, ppl_str, quality_str)

    console.print(t)


def show_compare(task: str, console):
    """SFT vs RL eval_loss 比較（同任務）。"""
    sft_report = load_eval_report(task, "sft")
    rl_report  = load_eval_report(task, "grpo") or load_eval_report(task, "dpo")

    t = Table(
        title=f"SFT vs RL 指標比較 — {task}",
        box=box.ROUNDED,
    )
    t.add_column("指標",   style="bold", width=25)
    t.add_column("SFT",    justify="right", width=15)
    t.add_column("RL",     justify="right", width=15)
    t.add_column("改善",   justify="center", width=10)

    def compare_row(label: str, sft_val, rl_val, lower_better: bool = True):
        s = _fmt(sft_val, 4)
        r = _fmt(rl_val, 4)
        if sft_val is not None and rl_val is not None:
            try:
                diff = float(rl_val) - float(sft_val)
                improved = diff < 0 if lower_better else diff > 0
                arrow = "↓" if diff < 0 else "↑"
                mark  = "✓" if improved else "✗"
                delta = f"[{'green' if improved else 'red'}]{mark}{arrow}{abs(diff):.4f}[/]"
            except ValueError:
                delta = "—"
        else:
            delta = "—"
        t.add_row(label, s, r, Text.from_markup(delta))

    def _get(report, *keys):
        if report is None:
            return None
        val = report
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return None
        return val

    compare_row("avg_loss",            _get(sft_report, "avg_loss"),  _get(rl_report, "avg_loss"))
    compare_row("perplexity",          _get(sft_report, "perplexity"), _get(rl_report, "perplexity"))
    compare_row("json_parse_rate",     _get(sft_report, "quality", "json_parse_success_rate"),
                                       _get(rl_report,  "quality", "json_parse_success_rate"), False)
    compare_row("format_compliance",   _get(sft_report, "quality", "format_compliance_rate"),
                                       _get(rl_report,  "quality", "format_compliance_rate"), False)
    compare_row("arithmetic_accuracy", _get(sft_report, "quality", "arithmetic_accuracy"),
                                       _get(rl_report,  "quality", "arithmetic_accuracy"), False)
    compare_row("avg_bleu",            _get(sft_report, "quality", "avg_bleu"),
                                       _get(rl_report,  "quality", "avg_bleu"), False)

    console.print(t)
    if sft_report is None:
        console.print(f"[yellow]  找不到 SFT eval_report.json（outputs/lora_{task}/eval_report.json）[/yellow]")
    if rl_report is None:
        console.print(f"[yellow]  找不到 RL eval_report.json（GRPO 或 DPO）[/yellow]")


def show_autoresearch(task: str | None, console):
    """顯示 AutoResearch 迭代紀錄。"""
    records = load_ar_log(task)
    if not records:
        console.print(f"[yellow]  尚無 AutoResearch 紀錄（{AR_LOG.name}）[/yellow]")
        return

    t = Table(
        title=f"AutoResearch 迭代紀錄 {'— ' + task if task else '（全部）'}",
        box=box.ROUNDED,
        show_lines=True,
    )
    t.add_column("#",          style="dim",  width=4)
    t.add_column("task",       style="cyan", width=14)
    t.add_column("mode",       width=5)
    t.add_column("iter",       justify="right", width=5)
    t.add_column("eval_loss",  justify="right", width=10)
    t.add_column("ppl",        justify="right", width=8)
    t.add_column("peak_GB",    justify="right", width=9)
    t.add_column("elapsed",    justify="right", width=9)
    t.add_column("status",     width=14)
    t.add_column("reason",     width=35)

    for i, rec in enumerate(records, 1):
        er      = rec.get("eval_report") or {}
        elapsed = rec.get("elapsed_sec")
        elapsed_str = f"{elapsed//60}m{elapsed%60:02d}s" if elapsed else "—"
        t.add_row(
            str(i),
            rec.get("task", ""),
            rec.get("mode", ""),
            str(rec.get("iteration", "")),
            Text.from_markup(_loss_color(rec.get("eval_loss", ""), 0.5)),
            _fmt(er.get("perplexity"), 2),
            _fmt(rec.get("peak_memory_gb"), 2),
            elapsed_str,
            rec.get("status", ""),
            (rec.get("claude_reason") or "")[:35],
        )

    console.print(t)


# ── HTML 報告 ─────────────────────────────────────────────────────────────────────

def generate_html(task: str | None) -> str:
    """產生 HTML 趨勢報告（使用內嵌 SVG 折線圖，不依賴 matplotlib）。"""
    sft_rows = load_sft(task)
    rl_rows  = load_rl(task)
    ar_log   = load_ar_log(task)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    def make_svg_chart(points: list[tuple], title: str, ylabel: str) -> str:
        """產生簡單 SVG 折線圖。"""
        if not points:
            return f"<p><em>（無資料）</em></p>"
        W, H, PAD = 600, 200, 40
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = 1, max(len(points), 1)
        y_min, y_max = min(ys) * 0.9, max(ys) * 1.1
        if y_max == y_min:
            y_max = y_min + 1

        def tx(i): return PAD + (i - 1) / max(x_max - 1, 1) * (W - 2*PAD)
        def ty(v): return H - PAD - (v - y_min) / (y_max - y_min) * (H - 2*PAD)

        pts_str = " ".join(f"{tx(i+1):.1f},{ty(v):.1f}" for i, v in enumerate(ys))
        circles = "".join(
            f'<circle cx="{tx(i+1):.1f}" cy="{ty(v):.1f}" r="4" fill="#3b82f6"/>'
            for i, v in enumerate(ys)
        )
        labels = "".join(
            f'<text x="{tx(i+1):.1f}" y="{ty(v)-8:.1f}" text-anchor="middle" font-size="10" fill="#374151">{v:.4f}</text>'
            for i, v in enumerate(ys)
        )

        return (
            f'<h3>{title}</h3>'
            f'<svg width="{W}" height="{H}" style="border:1px solid #e5e7eb;border-radius:8px;background:#f9fafb">'
            f'<polyline points="{pts_str}" fill="none" stroke="#3b82f6" stroke-width="2"/>'
            f'{circles}{labels}'
            f'<text x="{W//2}" y="{H-5}" text-anchor="middle" font-size="11" fill="#6b7280">迭代 #</text>'
            f'<text x="12" y="{H//2}" transform="rotate(-90,12,{H//2})" text-anchor="middle" font-size="11" fill="#6b7280">{ylabel}</text>'
            f'</svg>'
        )

    # SFT eval_loss 趨勢
    sft_points = []
    for i, r in enumerate(sft_rows):
        try:
            sft_points.append((i+1, float(r["eval_loss"])))
        except (KeyError, ValueError, TypeError):
            pass
    sft_chart = make_svg_chart(sft_points, "SFT eval_loss 趨勢", "eval_loss")

    # AutoResearch eval_loss 趨勢
    ar_points = []
    for rec in ar_log:
        try:
            ar_points.append((rec.get("iteration", 0), float(rec["eval_loss"])))
        except (KeyError, ValueError, TypeError):
            pass
    ar_chart = make_svg_chart(ar_points, "AutoResearch eval_loss 趨勢", "eval_loss")

    # SFT 表格 HTML
    def dict_to_html_table(rows: list[dict], cols: list[str]) -> str:
        if not rows:
            return "<p><em>（無資料）</em></p>"
        ths = "".join(f"<th>{c}</th>" for c in cols)
        trs = ""
        for r in rows:
            tds = "".join(f"<td>{r.get(c,'')}</td>" for c in cols)
            trs += f"<tr>{tds}</tr>"
        return f"<table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>"

    sft_cols = ["timestamp", "task", "rank", "alpha", "lr", "epochs", "max_steps",
                "train_loss", "eval_loss", "peak_memory_gb"]
    rl_cols  = ["timestamp", "task", "mode", "rank", "lr", "num_generations",
                "max_completion", "kl_coeff", "train_loss", "peak_memory_gb"]

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="UTF-8">
<title>Experiment Tracker — {task or '全部任務'}</title>
<style>
body {{ font-family: 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; background: #fff; }}
h1   {{ color: #1d4ed8; }}
h2   {{ color: #374151; border-bottom: 2px solid #e5e7eb; padding-bottom: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
th   {{ background: #1d4ed8; color: white; padding: 8px; text-align: left; }}
td   {{ padding: 6px 8px; border-bottom: 1px solid #e5e7eb; }}
tr:hover {{ background: #f0f9ff; }}
.meta {{ color: #6b7280; font-size: 13px; margin-bottom: 24px; }}
.charts {{ display: flex; gap: 32px; flex-wrap: wrap; margin-bottom: 32px; }}
</style>
</head>
<body>
<h1>Experiment Tracker</h1>
<p class="meta">任務：{task or '全部'}　　產生時間：{ts}</p>

<h2>趨勢圖</h2>
<div class="charts">
{sft_chart}
{ar_chart}
</div>

<h2>SFT 實驗紀錄</h2>
{dict_to_html_table(sft_rows, sft_cols)}

<h2>RL 實驗紀錄（GRPO / DPO）</h2>
{dict_to_html_table(rl_rows, rl_cols)}
</body>
</html>"""

    return html


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="實驗結果視覺化分析器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task", choices=ALL_TASKS, default=None,
        help="篩選特定任務（預設顯示全部）",
    )
    parser.add_argument("--best",    action="store_true", help="顯示各任務最佳結果")
    parser.add_argument("--compare", action="store_true", help="SFT vs RL 指標比較（需指定 --task）")
    parser.add_argument("--ar",      action="store_true", help="顯示 AutoResearch 迭代紀錄")
    parser.add_argument("--html",    action="store_true", help="輸出 HTML 報告到 outputs/")
    parser.add_argument("--all",     action="store_true", help="顯示全部：SFT + RL + AutoResearch")
    return parser.parse_args()


def main():
    args = parse_args()

    if not _RICH:
        print("[ERROR] 請先安裝 Rich：pip install rich")
        sys.exit(1)

    console = Console()

    # ── HTML 報告 ──────────────────────────────────────────────────────────────
    if args.html:
        html = generate_html(args.task)
        ts_str    = time.strftime("%Y-%m-%d_%H%M%S")
        task_tag  = f"_{args.task}" if args.task else "_all"
        html_path = OUTPUTS_DIR / f"experiment_report{task_tag}_{ts_str}.html"
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        console.print(f"[green]✓ HTML 報告已儲存至：{html_path}[/green]")
        return

    # ── 最佳結果 ───────────────────────────────────────────────────────────────
    if args.best:
        show_best(console)
        return

    # ── SFT vs RL 比較 ─────────────────────────────────────────────────────────
    if args.compare:
        if not args.task:
            console.print("[red]--compare 需要指定 --task[/red]")
            sys.exit(1)
        show_compare(args.task, console)
        return

    # ── AutoResearch 紀錄 ──────────────────────────────────────────────────────
    if args.ar:
        show_autoresearch(args.task, console)
        return

    # ── 預設 / --all：顯示 SFT + RL + AR ─────────────────────────────────────
    show_sft_table(args.task, console)
    show_rl_table(args.task, console)
    if args.all:
        show_autoresearch(args.task, console)


if __name__ == "__main__":
    main()
