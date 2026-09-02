# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching project, 题目3 in the OS/environments *实践项目任务书*: a **5-agent AutoGen system** that turns a natural-language materials-analysis request into a report. It is NOT a general app — it's a fixed pipeline for prompt→plan→data→code→execute→visualize→report.

The whole codebase is one Python package: `q3_autogen/`. There is no test suite, no linter config, no build step. Everything runs against two live APIs (DeepSeek chat-completions + Tavily search), so **full runs cost money and take ~1–2 min**.

## Environment

- Python 3.10+ in conda env **`pytorch`**, at `D:\conda\envs\pytorch\python.exe`.
- Run **from the project root** (`E:\claude_code_workspace\direction_practice`) as a module, not from inside the package:
  ```bash
  PYTHONIOENCODING=utf-8 D:/conda/envs/pytorch/python.exe -m q3_autogen.main
  PYTHONIOENCODING=utf-8 D:/conda/envs/pytorch/python.exe -m q3_autogen.main --task "<研究目标>"
  ```
- `PYTHONIOENCODING=utf-8` is **required** on this Windows box. The pipeline prints Chinese to stdout; the console is GBK, so without it you get `UnicodeEncodeError` (the first run mangled output and only surfaced as a print crash). Same reason any inline `search_web` test script needs `sys.stdout.reconfigure(encoding="utf-8")`.
- `-m q3_autogen.main` must be run from the repo root so the relative imports (`.tools`, `.agents`, `.config`) and the `work/` path resolve.

## Run modes

- No `--task` → interactive REPL (`repl()`), loop of prompts until `exit`/`quit`/`q`/`退出`.
- `--task "<...>"` → run that one task, then enter the REPL.
- Each task starts with `clear_work_artifacts()` (deletes `materials.csv`, `analysis.py`, `comparison.png`, `report.md`, `data_source.md`) so stale files from the previous task don't leak in.

## Verify without burning tokens

There is no test suite. The cheap checks:
```bash
D:/conda/envs/pytorch/python.exe -m py_compile q3_autogen/main.py q3_autogen/agents.py q3_autogen/tools.py q3_autogen/config.py
```
Plus a structural-only `build_agents()` (constructs all 5 agents + tools, no API call). Do this for any change to `agents.py`/`tools.py` before triggering a real run.

## Architecture

Five agents, one `RoundRobinGroupChat` (fixed order, deterministic): **Planner → Research → Coder → Executor → Analyst**. Termination: `TextMentionTermination("TERMINATE", sources=["Analyst"]) | MaxMessageTermination(30)`.

**The single most important thing to understand:** these agents do **not** have independent contexts. AutoGen's group chat gives all participants **one shared, ever-growing message thread**; only each agent's `system_message` is independent. So each agent sees a *prefix* of the same transcript (Analyst sees all), and the token cost is a prefix-sum, not a multiplier. Don't claim "per-agent isolated context / 5× tokens" — it's wrong and it was already verified against the source.

File roles:
- `config.py` — DeepSeek client (`OpenAIChatCompletionClient`), **registering a `deepseek` message-transformer family** (DeepSeek isn't in the built-in families; without it you get `Unknown message type`). Also sets env vars for the Executor subprocess: `MPLCONFIGDIR` (project `.mpldir/` → Chinese matplotlib font) and `PYTHONIOENCODING=utf-8`. API keys are **not** hardcoded — a tiny `.env` loader reads `DEEPSEEK_API_KEY` / `TAVILY_API_KEY` from real env vars (taken as authoritative) or the gitignored project-root `.env` (see `.env.example`). Missing `DEEPSEEK_API_KEY` → real calls fail auth; missing `TAVILY_API_KEY` → Research falls back to memory values.
- `agents.py` — the five agents, their system prompts, and `build_agents()` (the only place agents are constructed). Executor is a non-LLM `CodeExecutorAgent` (`sources=["Coder"]`, `work_dir=WORK_DIR.parent`, i.e. the `q3_autogen/` dir so generated `work/...` paths resolve).
- `tools.py` — the two Research tools: `write_materials_csv(csv_text, source=None)` (writes `work/materials.csv`, and `work/data_source.md` if `source` is given) and `search_web(query, max_results)` (Tavily).
- `main.py` — entry point: `build_team()` (wraps agents + termination), `run_task()` (runs + prints transcript), `write_report()` (assembly), REPL, `clear_work_artifacts()`.

### Tool-calling loop — the trap that broke the pipeline
`AssistantAgent`'s **`max_tool_iterations` defaults to `1`** (`_assistant_agent.py:739`). With a tool agent that must *search then write* (Research: `search_web` → look at results → `write_materials_csv`), the happy path has 2+ model turns, so the default made Research run `search_web` once and then **stop without ever calling `write_materials_csv`** → missing `materials.csv` → Executor `FileNotFoundError`. **Research has `max_tool_iterations=5`.** Keep that — any new tool agent doing multi-step tool use needs it raised above 1.

### Search is Tavily, and it must annotate its source
Research reaches the web via the **Tavily** search API (server-side, returns clean JSON) — NOT via an LLM's built-in knowledge and NOT via a local scrape. `search_web` uses `Authorization: Bearer <key>` at `https://api.tavily.com/search`. DuckDuckGo/Startpage are blocked and Bing scraping degrades Chinese multi-word queries into single-character dictionary pages — that's why the project went with a keyed API.

The system is expected to **state where the data came from**: `RESEARCH_SYSTEM` requires Research to pass a `source` to `write_materials_csv` — "联网检索（Tavily）+ URLs" when search worked, or "⚠️ 检索失败，本表数据为【模拟值/估算值】" when it fell back to memory. That text lands in `work/data_source.md` and is rendered in the report's section 三 as a blockquote. Preserve this honesty contract. If search fails it's fine — but the output must say the values are simulated, not pretend they're measured.

### Report assembly
`write_report` builds `work/report.md` from the transcript: it splits on `]`, keeps **only `TextMessage`** (drops `ThoughtEvent` and all tool request/execution events so model "thinking" never appears), strips the `[Type][Source]` prefix, and groups by agent (Planner + Analyst only). CSV/code are read **directly from `work/` files**, not from the transcript. `TERMINATE` is stripped from the Analyst's text. Rendering `data_source.md` as a blockquote sits above the CSV in section 三.

## Conventions to respect in generated code (Coder system prompt)

- Reads `work/materials.csv` defensively: material column = `df.columns[0]`, numeric columns via `is_numeric_dtype`, **never hardcodes column names** (they vary per research goal, can be Chinese or English with `属性_单位` naming).
- Uses `matplotlib.use("Agg")` + a `font.sans-serif` fallback list, no interactive window, saves `work/comparison.png`.
- One subplot per numeric column in a grid.
- The Coder system prompt already encodes these; editing `agents.py` system messages changes agent behavior — that's the main knob for the project.

## Don't be tripped up by

- `builtins` note: importing the package triggers `config.py`, which imports `autogen_ext.models.openai` and registers the DeepSeek transformer — heavy but required; ~seconds per import.
- API keys are plaintext in `config.py` (teaching demo). Real runs bill the user's DeepSeek account and Tavily credits.
- Output randomness: LLM sampling means values/wording vary run to run; the pipeline structure is stable, values are not.
