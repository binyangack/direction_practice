"""题目3 主程序：用 AutoGen 编排五个智能体，完成从自然语言需求到报告的闭环。

用法
----
    python -m q3_autogen.main                      # 交互模式：循环输入研究目标，输入 exit 退出
    python -m q3_autogen.main --task "..."         # 先跑指定任务，再进入交互模式
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from .agents import build_agents
from .tools import WORK_DIR

DEFAULT_TASK = "比较钢、铝、铜的导热性能，并画出对比柱状图。"


def build_team():
    """构建 RoundRobinGroupChat 团队（Planner→Research→Coder→Executor→Analyst）。"""
    participants, _ = build_agents()

    # 正常结束：Analyst 输出结论后写出 TERMINATE。
    # 兜底：消息数达到上限则强制结束，避免模型偶发漏写导致死循环。
    termination = TextMentionTermination("TERMINATE", sources=["Analyst"]) | MaxMessageTermination(
        max_messages=30
    )

    team = RoundRobinGroupChat(
        participants=participants,
        termination_condition=termination,
        max_turns=10,
    )
    return team


def message_to_text(message) -> str:
    """把 AutoGen 消息转换为可读文本（用于日志/报告）。"""
    mtype = type(message).__name__
    source = getattr(message, "source", "?")
    content = getattr(message, "content", "")
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    return f"[{mtype}][{source}] {content}"


async def run_task(task: str) -> dict:
    """执行任务，返回结构化结果（供报告生成与调试）。"""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    team = build_team()

    print("=" * 70)
    print("研究目标：", task)
    print("多智能体团队：Planner -> Research -> Coder -> Executor -> Analyst")
    print("=" * 70)

    result = await team.run(task=task)

    # 收集各角色发言（按出现顺序）
    transcript = []
    for m in result.messages:
        if type(m).__name__ == "TaskResult":
            continue
        transcript.append(message_to_text(m))
        print(message_to_text(m))
        print("-" * 30)

    print("=" * 70)
    print("停止原因：", getattr(result, "stop_reason", type(result).__name__))
    print("=" * 70)

    return {"task": task, "transcript": transcript, "result": result}


def collect_artifacts() -> dict:
    """从 work 目录收集本次运行产生的交付物。"""
    artifacts = {}
    for name in ("materials.csv", "analysis.py", "comparison.png"):
        p = WORK_DIR / name
        artifacts[name] = bool(p.exists()) and ("file" if p.is_file() else "dir")
    return artifacts


def write_report(task: str, transcript: list[str]) -> Path:
    """把本次运行的关键结论整理为 report.md。"""
    # 从 transcript 中抽取每个角色的发言内容
    # 只收 TextMessage（正式发言）；模型内部思考(ThoughtEvent)不应进报告。
    by_agent = {}
    for line in transcript:
        # 形如 [TextMessage][Planner] ... / [ThoughtEvent][Analyst] ...
        tag = line.split("]", 2)
        if len(tag) >= 3:
            mtype = tag[0].lstrip("[").strip()
            if mtype != "TextMessage":
                continue
            try:
                src = tag[1].split("[")[1]
            except IndexError:
                src = "?"
            body = tag[2].lstrip(" ]")
            by_agent.setdefault(src, []).append(body)

    report_lines = [
        "# 多智能体实验方案生成与数据分析 —— 实验报告",
        "",
        f"**研究目标：** {task}",
        "",
        "## 一、多智能体协作流程",
        "",
        "本实验使用 AutoGen 编排五个智能体（Planner → Research → Coder → Executor → Analyst），"
        "完成“自然语言需求 → 方案 → 数据 → 代码 → 执行 → 结论”的全流程。",
        "",
        "| 角色 | 职责 |",
        "| --- | --- |",
        "| Planner | 实验规划员：拆解任务、给出步骤与数据需求 |",
        "| Research | 资料收集员：生成并保存标准化材料数据表 |",
        "| Coder | 程序生成员：自动生成分析/绘图代码 |",
        "| Executor | 实验执行员：运行代码并返回结果 |",
        "| Analyst | 结果解释员：解释结果，输出报告结论 |",
        "",
        "## 二、任务方案（Planner）",
        "",
    ]
    report_lines += [f"> {b}" for b in by_agent.get("Planner", ["（未生成）"])]

    report_lines += ["", "## 三、材料数据（Research）", ""]
    src_path = WORK_DIR / "data_source.md"
    if src_path.exists():
        # data_source.md 内容为 Research 生成的“数据来源：…”说明（成功注明 URL，失败标注模拟值），
        # 直接以 blockquote 呈现即可，无需再加 “数据来源” 标签，避免重复。
        for ln in src_path.read_text(encoding="utf-8").strip().splitlines():
            report_lines.append(f"> {ln}")
        report_lines.append("")
    report_lines += ["```csv"]
    csv_path = WORK_DIR / "materials.csv"
    if csv_path.exists():
        report_lines.append(csv_path.read_text(encoding="utf-8").strip())
    else:
        report_lines.append("（未生成数据表）")
    report_lines += ["```", ""]

    report_lines += ["", "## 四、分析代码（Coder）", "", "```python"]
    code_path = WORK_DIR / "analysis.py"
    if code_path.exists():
        report_lines.append(code_path.read_text(encoding="utf-8").strip())
    else:
        report_lines += ["# 说明：Coder 生成的代码位于 work/analysis.py", "# 若此处为空，说明本次运行未能把代码落盘，可从对话记录中查 Coder 的输出。"]
    report_lines += ["```", ""]

    report_lines += [
        "",
        "## 五、可视化结果",
        "",
        "![](comparison.png)",
        "",
        "## 六、结论（Analyst）",
        "",
    ]
    analyst_lines = by_agent.get("Analyst", ["（未生成）"])
    # 去掉用于触发终止的 TERMINATE 标记，避免其出现在报告里
    cleaned = [b.replace("TERMINATE", "").rstrip() for b in analyst_lines if b.strip()]
    cleaned = [ln for ln in cleaned if ln.strip()]
    report_lines += [f"> {b}" for b in cleaned]

    report_lines += ["", "---", "", "由 AutoGen 多智能体协作自动生成。", ""]

    report_path = WORK_DIR / "report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def clear_work_artifacts() -> None:
    """清空上一次任务留下的中间产物，避免本任务未生成某文件时误用旧文件。"""
    for name in ("materials.csv", "analysis.py", "comparison.png", "report.md", "data_source.md"):
        p = WORK_DIR / name
        if p.exists():
            p.unlink()


async def async_main(task: str) -> int:
    clear_work_artifacts()  # 每个新任务前清掉上一次的产物，避免旧 CSV/图/报告混入本任务。
    result = await run_task(task)

    # 先把 Coder 生成的代码落盘，写报告时才能引用到真实代码（而非占位符）。
    # Coder 可能先发思考(ThoughtEvent)再发代码(TextMessage)，须跳过思考、只命中含代码块的消息。
    if not (WORK_DIR / "analysis.py").exists():
        for m in result["result"].messages:
            if getattr(m, "source", "") == "Coder" and _save_code_from_message(m):
                break

    report_path = write_report(result["task"], result["transcript"])
    print("\n交付物：")
    for name, has in collect_artifacts().items():
        print(f"  - work/{name}: {'存在' if has else '缺失'}")
    print(f"  - {report_path}")
    return 0


def _save_code_from_message(message) -> bool:
    """从 Coder 的消息里提取 ```python ... ``` 代码块并保存到 analysis.py。

    返回 True 表示成功提取并落盘；False 表示该消息里没有可用的代码块（如思考文本）。
    """
    import re

    content = getattr(message, "content", "") or ""
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    code = None
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.DOTALL)
    if blocks:
        code = blocks[-1]
    elif "```" in content:
        code = content.split("```", 1)[1].split("```", 1)[0]
    if code:
        (WORK_DIR / "analysis.py").write_text(code.strip() + "\n", encoding="utf-8")
        print("已从对话记录提取 Coder 代码并保存到 work/analysis.py")
        return True
    return False


async def repl() -> int:
    """交互式循环：反复读入新的材料分析研究目标并运行，直到用户输入退出指令。"""
    exit_keys = {"exit", "quit", "q", "退出", "end", "eof"}
    print(
        "（交互模式）输入一个材料分析研究目标，例如："
        "比较纯铜与黄铜的导热/导电性能差异，分析合金化对性能的影响。输入 exit 退出。"
    )
    while True:
        print()
        try:
            raw = await asyncio.to_thread(input, "研究目标> ")
        except EOFError:
            print()
            break
        task = raw.strip()
        if not task:
            continue
        if task.lower() in exit_keys:
            break
        try:
            await async_main(task)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[错误] 本任务运行失败：{exc}")
            print("可重新输入一个新的研究目标。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="题目3：AutoGen 多智能体实验方案生成与数据分析（可交互）"
    )
    parser.add_argument("--task", default=None,
                        help="材料分析研究目标（自然语言）。不传则直接进入交互模式。")
    parser.add_argument("--strip-train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    async def _run_interface() -> int:
        if args.task:
            code = await async_main(args.task)
            if code != 0:
                return code
        return await repl()

    try:
        code = asyncio.run(_run_interface())
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    return code


if __name__ == "__main__":
    sys.exit(main())
