"""供 Agent 调用的工具函数。

Research Agent 需要把“标准化数据表”落盘，供后续 Coder / Executor 消费。
用一个可被 AutoGen FunctionTool 包装的纯函数即可，避免交叉传递导致的脆弱性。
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config

# 项目工作目录：CSV、代码、图表、报告都放在这里。
WORK_DIR = Path(__file__).resolve().parent / "work"


def write_materials_csv(csv_text: str, source: str | None = None) -> str:
    """把一段 CSV 文本写入 ``work/materials.csv``，并可选地把数据来源说明写入 work/data_source.md。

    参数
    ----
    csv_text :
        形如 ``"material,thermal_conductivity_W_per_mK\\n铝,205\\n..."`` 的文本，
        第一行为列名（至少包含 material 与若干材料属性列）。
    source :
        数据来源说明。联网检索成功时写明检索到的来源（URL 等）；检索失败/未命中时，
        必须明确标注“本表数据为模拟值/估算值”。该说明会写入 work/data_source.md 供报告引用。

    返回值
    ------
    保存路径，用于让模型感知文件已生成。
    """
    path = WORK_DIR / "materials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text.strip() + "\n", encoding="utf-8")
    if source:
        (WORK_DIR / "data_source.md").write_text(source.strip() + "\n", encoding="utf-8")
    return f"数据表已保存到 {path}"


# ---------------------------------------------------------------------------
# 联网检索（方案B：给 Research 加一个搜索工具）
# ---------------------------------------------------------------------------
# 用 Tavily 搜索 API（服务端检索、返回干净 JSON），替代本地刮取 Bing 的脆落实现。
# 需要 TAVILY_API_KEY：优先读环境变量，未设置时回退到 config.py 从 .env/环境变量读到的值。
# 检索失败或未配置 key 时返回提示，模型应据此回退到已知常识值（最坏情况 = 原有基于记忆的方案）。
_TAVILY_URL = "https://api.tavily.com/search"


def _tavily_key() -> str:
    """读取 Tavily API key：优先环境变量，回退 config.py 读取到的值。"""
    return os.environ.get("TAVILY_API_KEY") or getattr(config, "TAVILY_API_KEY", "")


def search_web(query: str, max_results: int = 6) -> str:
    """用 Tavily 联网检索给定查询，返回若干条相关的标题+摘要+链接。

    参数
    ----
    query :
        搜索词，例如 “黄铜 导热系数 典型值”。建议用“材料 + 属性 + 衡量词”搭配。
    max_results :
        最多返回几条（默认 6）。

    返回值
    ------
    拼接好的检索文本。若未配置 key、检索失败或无有效结果，返回提示，模型应据此
    换词重试，或回退到已知常识值（并把“数据来源”标注为常识/估算）。
    """
    key = _tavily_key()
    if not key:
        return (
            "[检索失败] 未配置 TAVILY_API_KEY：请在项目根目录 .env 或环境变量中设置后重试；"
            "也可回退到已知常识值并在数据来源中注明“估算/常识值”。"
        )
    try:
        import requests  # noqa: 延迟导入，避免顶层硬依赖

        resp = requests.post(
            _TAVILY_URL,
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"[检索失败] {exc}"

    results = data.get("results") or []
    if not results:
        return (
            "（未检索到相关结果，请换关键词重试；若仍无结果，可依据已知常识补充并"
            "在数据来源中注明“估算/常识值”。）"
        )
    answers = []
    for r in results:
        title = r.get("title", "")
        content = (r.get("content") or "").strip()
        url = r.get("url", "")
        # 截断过长内容，避免某条结果（尤其带网站页脚的）把上下文撑爆
        if content and len(content) > 400:
            content = content[:400] + " …"
        block = f"- {title}"
        if content:
            block += f"\n  {content}"
        if url:
            block += f"\n  {url}"
        answers.append(block)
    return "\n".join(answers)
