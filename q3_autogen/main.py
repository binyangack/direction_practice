"""题目3 主程序：用 AutoGen 编排五个智能体，完成从自然语言需求到报告的闭环。

用法
----
    python -m q3_autogen.main                      # 交互模式：循环输入研究目标，输入 exit 退出
    python -m q3_autogen.main --task "..."         # 先跑指定任务，再进入交互模式
    python -m q3_autogen.main --task "..." --hitl  # HITL 模式：方案/数据/图表三处暂停征询
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
from .validation import validate_materials_csv
from .metrics import analyze_materials, render_quant
from . import config

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

    result, transcript = await _run_team_with_progress(team, task)

    print("\n✓ 协作完成")
    return {"task": task, "transcript": transcript, "result": result}


def _extract_text_messages(result) -> tuple[list[str], str]:
    """从一次运行结果里抽取 agent 自己的 TextMessage（跳过 source="user" 的任务消息）。

    返回 (格式化行列表, 纯文本拼接)。注意：``agent.run(task=...)`` 会把 task 转成
    source="user" 的 TextMessage 并放进结果里，若不跳过会把"任务原文"误当成 agent 输出。
    """
    lines: list[str] = []
    texts: list[str] = []
    for m in result.messages:
        if type(m).__name__ == "TextMessage" and getattr(m, "source", "") != "user":
            lines.append(message_to_text(m))
            c = getattr(m, "content", "")
            if isinstance(c, list):
                c = " ".join(str(x) for x in c)
            texts.append(str(c))
    return lines, "\n".join(t for t in texts if t.strip())


async def _ask(prompt: str) -> str:
    """阻塞式向用户征询输入（放进线程池，避免卡住事件循环）。"""
    return (await asyncio.to_thread(input, prompt)).strip()


_SPINNER_CHARS = ("|", "/", "-", "\\")


async def _run_with_spinner(awaitable, label: str = "运行中"):
    """等待一个异步操作时显示旋转加载指示（提升 HITL 交互体验）。"""
    stop = asyncio.Event()

    async def _spin():
        i = 0
        while not stop.is_set():
            print(f"\r{_SPINNER_CHARS[i % 4]} {label}...", end="", flush=True)
            i += 1
            await asyncio.sleep(0.12)
        print("\r" + " " * (len(label) + 16) + "\r", end="", flush=True)  # 清除该行

    spin_task = asyncio.create_task(_spin())
    try:
        return await awaitable
    finally:
        stop.set()
        await spin_task


_AGENT_LABELS = {
    "Planner": "Planner 规划中",
    "Research": "Research 检索数据中",
    "Coder": "Coder 生成代码中",
    "Executor": "Executor 执行代码中",
    "Analyst": "Analyst 生成结论中",
}


async def _run_team_with_progress(team, task: str):
    """流式运行团队，实时显示当前正在工作的 agent，返回 (result, transcript)。"""
    result = None
    state = {"agent": "准备中", "done": False}

    async def _spin():
        i = 0
        while not state["done"]:
            print(f"\r{_SPINNER_CHARS[i % 4]} {state['agent']}...", end="", flush=True)
            i += 1
            await asyncio.sleep(0.12)
        print("\r" + " " * 50 + "\r", end="", flush=True)

    spin_task = asyncio.create_task(_spin())
    try:
        async for event in team.run_stream(task=task):
            if type(event).__name__ == "TaskResult":
                result = event
                continue
            source = getattr(event, "source", "")
            if source in _AGENT_LABELS:
                state["agent"] = _AGENT_LABELS[source]
    finally:
        state["done"] = True
        await spin_task

    transcript = []
    for m in result.messages:
        if type(m).__name__ == "TaskResult":
            continue
        transcript.append(message_to_text(m))
    return result, transcript


def _show_data() -> None:
    """打印当前数据表及其校验结果，供用户审查。"""
    csv_path = WORK_DIR / "materials.csv"
    if not csv_path.exists():
        print("\n（尚未生成数据表）")
        return
    print("\n--- 当前数据表 work/materials.csv ---")
    print(csv_path.read_text(encoding="utf-8").strip())
    src_path = WORK_DIR / "data_source.md"
    q = validate_materials_csv(
        csv_path.read_text(encoding="utf-8"),
        src_path.read_text(encoding="utf-8") if src_path.exists() else None,
    )
    if q["ok"] and not q["warnings"]:
        print("  [数据校验] ✓ 通过")
    else:
        for p in q["problems"]:
            print(f"  [数据校验] ⚠️ {p}")
        for w in q["warnings"]:
            print(f"  [数据校验] 提示 {w}")


def _show_chart() -> None:
    """提示图表是否已生成。"""
    p = WORK_DIR / "comparison.png"
    if p.exists():
        print(f"\n图已生成：{p}")
    else:
        print("\n⚠️ 未生成图（可能执行失败），可查看上方 Coder/Executor 输出。")


_STAGES = ("plan", "data", "chart", "analyze")


def _next_stage(stage: str) -> str:
    return _STAGES[_STAGES.index(stage) + 1]


def _clamp_target(target: str | None, current: str) -> str:
    """把反馈分类限制到当前及更早的阶段；分类失败/指向更晚阶段时视为改当前步。"""
    if target not in _STAGES:
        return current
    if _STAGES.index(target) > _STAGES.index(current):
        return current
    return target


_CLASS_TO_STAGE = {"plan": "plan", "scope": "plan", "value": "data", "chart": "chart"}


async def _classify_feedback(feedback: str) -> str | None:
    """用 LLM 把用户反馈归类到 plan / scope / value / chart 之一；失败返回 None。

    把「数据」再拆成 scope（增删材料/属性，会影响方案的数据需求）与 value（改某个数值，
    不影响方案范围）——这样增删材料时会回跳到方案重新规划，保证报告的方案与最终数据一致。
    """
    from autogen_core.models import UserMessage

    prompt = (
        "你是多智能体流水线里的分类器。把用户反馈归类到下面四类之一：\n"
        "- plan：关于实验方案/研究步骤/要分析什么方向\n"
        "- scope：增加或删除要分析的材料或属性（例：\"加上纯铁\"\"去掉密度\"\"也分析导电率\"）\n"
        "- value：修改某个已有材料或属性的具体数值（例：\"铝的导热率改成 237\"）\n"
        "- chart：关于图表/画法/图型/可视化\n\n"
        f"用户反馈：{feedback}\n\n"
        "只回答 plan / scope / value / chart 一个词，不要任何解释。"
    )
    try:
        result = await config.get_model_client().create(
            messages=[UserMessage(content=prompt, source="user")]
        )
        text = (result.content or "").strip().lower()
    except Exception:
        return None
    for word in ("plan", "scope", "value", "chart"):
        if word in text:
            return word
    return None


def _extract_source_text(result, source: str) -> str:
    """抽取指定 source 的 TextMessage 内容（用于取 Executor 的 stdout）。"""
    return "\n".join(
        str(getattr(m, "content", "")) for m in result.messages
        if type(m).__name__ == "TextMessage" and getattr(m, "source", "") == source
    )


_MAX_CODE_RETRIES = 3


def _exec_failed(output: str) -> bool:
    """判断 Executor 输出是否表示代码运行失败。"""
    return ("exited with an error" in output) or ("Traceback (most recent call last)" in output)


def _chart_ok(output: str) -> bool:
    """判断本次 Coder+Executor 是否成功：无报错且生成了图。"""
    return (not _exec_failed(output)) and (WORK_DIR / "comparison.png").exists()


async def run_task_hitl(task: str) -> dict:
    """HITL 模式：状态机编排，方案/数据/图表三处暂停，反馈经 LLM 分类后可回跳重做。

    与全自动不同，这里手动编排各 agent（``agent.run()`` 逐段驱动），并把流程建模为
    一个可在 plan / data / chart 之间回跳的状态机：用户对任一步的反馈经分类后路由到
    目标阶段，重跑该阶段并连带下游重新确认。
    """
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    planner, research, coder, executor, analyst = build_agents()[0]

    plan_text = ""
    exec_text = ""
    analyst_text = ""
    feedbacks = {"plan": [], "data": [], "chart": []}

    print("=" * 70)
    print("研究目标：", task)
    print("模式：HITL（人在回路）—— 方案/数据/图表三处暂停，反馈会自动识别要改哪一步，可回跳重做")
    print("=" * 70)

    state = "plan"
    while state != "done":
        if state == "plan":
            reqs = [task] + feedbacks["plan"]
            ctx = "\n".join(f"{i}. {r}" for i, r in enumerate(reqs, 1))
            ctx += "\n\n请据此输出实验方案（研究目标/实验步骤/数据需求），不要复述这些要求本身。"
            _, plan_text = _extract_text_messages(await _run_with_spinner(planner.run(task=ctx), "Planner 规划中"))
            print("\n[Planner]\n" + plan_text)
            fb = await _ask("\n👤 方案 OK？回车=确认，或输入修改意见（系统会判断要改哪一步）：")
            if not fb:
                state = _next_stage("plan")
            else:
                cls = await _run_with_spinner(_classify_feedback(fb), "识别反馈中")
                target = _clamp_target(_CLASS_TO_STAGE.get(cls, state), state)
                feedbacks[target].append(fb)
                state = target

        elif state == "data":
            ctx = f"研究目标：{task}\n\n实验方案：\n{plan_text}"
            for f in feedbacks["data"]:
                ctx += f"\n用户对数据的修正意见：{f}"
            ctx += "\n\n请据此联网检索并生成标准化的材料数据表。"
            _, rtext = _extract_text_messages(await _run_with_spinner(research.run(task=ctx), "Research 检索数据中"))
            print("\n[Research]\n" + rtext)
            _show_data()
            fb = await _ask("\n👤 数据 OK？回车=确认，或输入修改意见（可要求改方案/数据）：")
            if not fb:
                state = _next_stage("data")
            else:
                cls = await _run_with_spinner(_classify_feedback(fb), "识别反馈中")
                target = _clamp_target(_CLASS_TO_STAGE.get(cls, state), state)
                feedbacks[target].append(fb)
                state = target

        elif state == "chart":
            base_ctx = f"研究目标：{task}\n\n实验方案：\n{plan_text}"
            for f in feedbacks["chart"]:
                base_ctx += f"\n用户对图的修改意见：{f}"
            base_ctx += "\n\n请据此生成分析绘图代码并执行。"

            # 内层自纠错重试：代码在 Executor 跑挂就把报错反馈给 Coder 重写，
            # 区分「画图挂了」还是「分析挂了」给出精确提示，直到两者都能运行。
            error_hint = ""
            exec_text = ""
            for attempt in range(1, _MAX_CODE_RETRIES + 1):
                png = WORK_DIR / "comparison.png"
                if png.exists():
                    png.unlink()  # 清掉旧图，便于判断本次是否真正生成
                coder_team = RoundRobinGroupChat(
                    [coder, executor],
                    termination_condition=MaxMessageTermination(max_messages=20),
                    max_turns=2,
                )
                ce_result = await _run_with_spinner(
                    coder_team.run(task=base_ctx + error_hint), "Coder 生成代码 · Executor 执行中"
                )
                ce_lines, ce_text = _extract_text_messages(ce_result)
                print("\n[Coder + Executor]\n" + ce_text)
                exec_text = _extract_source_text(ce_result, "Executor")
                for m in ce_result.messages:  # 每次覆盖保存最新代码
                    if type(m).__name__ == "TextMessage" and getattr(m, "source", "") == "Coder" and _save_code_from_message(m):
                        break

                if _chart_ok(exec_text):
                    break
                if _exec_failed(exec_text):
                    if (WORK_DIR / "comparison.png").exists():
                        error_hint = ("\n\n[注意] 你的画图代码已成功生成图，但**分析代码**运行报错，请只修复分析部分、画图部分保持不动：\n" + exec_text)
                    else:
                        error_hint = "\n\n[注意] 你的代码在**画图阶段**运行报错，请修复后重新生成完整代码：\n" + exec_text
                else:
                    error_hint = "\n\n[注意] 你的代码运行无报错，但没有生成 work/comparison.png（可能忘了 plt.savefig），请补上保存图的代码。"
                print(f"\n[Coder 代码未通过，自动重试 {attempt}/{_MAX_CODE_RETRIES}]\n{error_hint}")

            _show_chart()
            if not _chart_ok(exec_text):
                print("\n⚠️ 代码多次重试仍未能通过，以下结果仅供参考。")

            fb = await _ask("\n👤 图 OK？回车=确认，或输入修改意见（可要求改方案/数据/图）：")
            if not fb:
                state = _next_stage("chart")
            else:
                cls = await _run_with_spinner(_classify_feedback(fb), "识别反馈中")
                target = _clamp_target(_CLASS_TO_STAGE.get(cls, state), state)
                feedbacks[target].append(fb)
                state = target

        elif state == "analyze":
            analyst_ctx = (
                f"研究目标：{task}\n\n"
                f"实验方案（仅作背景参考，不要复述）：\n{plan_text}\n\n"
                f"材料数据已由 Research 生成并校验（见 work/materials.csv）。\n"
                f"执行结果（Executor 输出）：\n{exec_text}\n\n"
                "请据此解释结果。不要重复实验方案步骤，直接输出 结论 / 机理解释 / 工程建议，并以 TERMINATE 收尾。"
            )
            _, analyst_text = _extract_text_messages(await _run_with_spinner(analyst.run(task=analyst_ctx), "Analyst 生成结论中"))
            print("\n[Analyst]\n" + analyst_text)
            state = "done"

    print("=" * 70)
    print("HITL 流程结束。")
    print("=" * 70)

    transcript = [
        f"[TextMessage][Planner] {plan_text}",
        f"[TextMessage][Analyst] {analyst_text}",
    ]
    return {"task": task, "transcript": transcript, "result": None}


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

    # 数据质量校验（机制化）：用确定性校验器复核最终落盘的 CSV，而非仅依赖模型自述来源。
    if csv_path.exists():
        q = validate_materials_csv(
            csv_path.read_text(encoding="utf-8"),
            src_path.read_text(encoding="utf-8") if src_path.exists() else None,
        )
        if q["ok"] and not q["warnings"]:
            report_lines.append("> **[数据质量校验]** ✓ 通过：结构完整，且已标注数据来源。")
        else:
            if q["problems"]:
                report_lines.append("> **[数据质量校验]** ⚠️ 未通过：" + "；".join(q["problems"]))
            if q["warnings"]:
                report_lines.append("> **[数据质量校验]** 提示：" + "；".join(q["warnings"]))
        report_lines.append("")

    report_lines += ["", "## 四、分析代码（Coder）", "", "```python"]
    code_path = WORK_DIR / "analysis.py"
    if code_path.exists():
        report_lines.append(code_path.read_text(encoding="utf-8").strip())
    else:
        report_lines += ["# 说明：Coder 生成的代码位于 work/analysis.py", "# 若此处为空，说明本次运行未能把代码落盘，可从对话记录中查 Coder 的输出。"]
    report_lines += ["```", ""]

    report_lines += ["", "## 五、定量分析（机器计算）", ""]
    if csv_path.exists():
        try:
            report_lines += render_quant(analyze_materials(csv_path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            report_lines.append(f"（定量分析失败：{exc}）")
    else:
        report_lines.append("（未生成数据表，无法定量分析）")

    report_lines += [
        "",
        "## 六、可视化结果",
        "",
        "![](comparison.png)",
        "",
        "## 七、结论（Analyst）",
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


async def async_main(task: str, hitl: bool = False) -> int:
    clear_work_artifacts()  # 每个新任务前清掉上一次的产物，避免旧 CSV/图/报告混入本任务。
    if hitl:
        result = await run_task_hitl(task)
    else:
        result = await run_task(task)
        # 先把 Coder 生成的代码落盘，写报告时才能引用到真实代码（而非占位符）。
        # Coder 可能先发思考(ThoughtEvent)再发代码(TextMessage)，须跳过思考、只命中含代码块的消息。
        if not (WORK_DIR / "analysis.py").exists():
            for m in result["result"].messages:
                if type(m).__name__ == "TextMessage" and getattr(m, "source", "") == "Coder" and _save_code_from_message(m):
                    break

    report_path = write_report(result["task"], result["transcript"])
    print(f"✓ 报告已生成：{report_path}")
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
        # 取最长的代码块：LLM 偶发会在真正代码之后追加一段"占位/摘要"代码块
        # （如 "python代码块，包含完整可运行代码…根据以上，编写代码"），最长者才是真正可运行的代码。
        code = max(blocks, key=len)
    elif "```" in content:
        code = content.split("```", 1)[1].split("```", 1)[0]
    if code:
        (WORK_DIR / "analysis.py").write_text(code.strip() + "\n", encoding="utf-8")
        print("已从对话记录提取 Coder 代码并保存到 work/analysis.py")
        return True
    return False


async def repl(hitl: bool = False) -> int:
    """交互式循环：反复读入新的材料分析研究目标并运行，直到用户输入退出指令。"""
    exit_keys = {"exit", "quit", "q", "退出", "end", "eof"}
    mode = "HITL（人在回路）" if hitl else "全自动"
    print(
        f"（交互模式[{mode}]）输入一个材料分析研究目标，例如："
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
            await async_main(task, hitl=hitl)
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
    parser.add_argument("--hitl", action="store_true",
                        help="HITL 模式（人在回路）：在 方案/数据/图表 三处暂停征询你的意见；默认全自动。")
    parser.add_argument("--strip-train", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    async def _run_interface() -> int:
        if args.task:
            code = await async_main(args.task, hitl=args.hitl)
            if code != 0:
                return code
        return await repl(hitl=args.hitl)

    try:
        code = asyncio.run(_run_interface())
    except KeyboardInterrupt:
        print("\n已中断。")
        return 130
    return code


if __name__ == "__main__":
    sys.exit(main())
