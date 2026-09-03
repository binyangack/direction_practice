"""数据校验：把「数据可信度」从一句提示词劝告，变成一段确定性机制。

Research 用 write_materials_csv 落盘后，这里对 CSV 做**领域无关**的结构与统计校验，
返回明确的「问题 / 警告」清单：

- **问题（problems）**：结构缺陷，会导致下游 Coder 读表失败，或违反来源诚实契约。
- **警告（warnings）**：物理/统计层面的可疑值，需模型或人工判断，**不阻断**。

刻意**不做领域知识硬编码**（不写死「铜导热必须≈400」之类），避免重蹈 CODER_SYSTEM
当初把柱状图写死、一换任务就失效的覆辙——领域合理性仍交给 Research / Analyst 的模型知识。
"""

from __future__ import annotations

import csv
import io
from statistics import median


def validate_materials_csv(csv_text: str, source: str | None = None) -> dict:
    """校验一段材料数据 CSV 的结构与数值合理性。

    参数
    ----
    csv_text :
        待校验的 CSV 文本（含表头）。第一列视为对象/材料列，其后为数值属性列。
    source :
        数据来源说明（与 write_materials_csv 的 source 一致）。为空则视为未满足来源诚实契约。

    返回
    ----
    dict，键：``ok``(bool)、``problems``(list[str])、``warnings``(list[str])、
    ``rows``(int, 数据行数)、``numeric_cols``(int, 数值列数)。
    """
    problems: list[str] = []
    warnings: list[str] = []

    # 1) 来源诚实契约：必须标注数据来源
    if not (source and source.strip()):
        problems.append("未标注数据来源（source 为空，无法区分检索值/估算值）")

    text = (csv_text or "").strip()
    if not text:
        return {"ok": False, "problems": ["CSV 内容为空"], "warnings": warnings, "rows": 0, "numeric_cols": 0}

    try:
        rows = list(csv.reader(io.StringIO(text)))
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "problems": [f"CSV 解析失败：{exc}"], "warnings": warnings, "rows": 0, "numeric_cols": 0}

    if len(rows) < 2:
        return {"ok": False, "problems": ["至少需要表头 + 1 行数据"], "warnings": warnings, "rows": 0, "numeric_cols": 0}

    header = [str(h).strip() for h in rows[0]]
    ncols = len(header)
    data_rows = rows[1:]

    # 2) 结构：每行列数一致
    for i, row in enumerate(data_rows, start=2):
        if len(row) != ncols:
            problems.append(f"第 {i} 行列数 {len(row)} 与表头 {ncols} 不一致")

    # 3) 结构：空单元格
    for i, row in enumerate(data_rows, start=2):
        for j, cell in enumerate(row):
            if not str(cell).strip():
                problems.append(f"第 {i} 行第 {j + 1} 列为空")

    # 4) 数值列检测（从第 2 列起）+ 统计合理性
    numeric_cols = 0
    for j in range(1, ncols):
        vals: list[float] = []
        all_numeric = True
        for row in data_rows:
            cell = str(row[j]).strip() if j < len(row) else ""
            if cell == "":
                continue  # 空单元格已在上面单独报告，这里跳过
            try:
                vals.append(float(cell))
            except ValueError:
                all_numeric = False
                break
        if not all_numeric or not vals:
            continue
        numeric_cols += 1
        col_name = header[j]
        # 负值：多数物理量（导热/导电/密度/比热/电阻率/绝对温度）应为非负
        negs = [v for v in vals if v < 0]
        if negs:
            warnings.append(f"列「{col_name}」含负值 {negs[:3]}（若为摄氏温度等允许负值的量可忽略）")
        # 极端离群：与中位数相差 1000 倍以上，提示人工/模型确认
        med = median(vals)
        if med and med != 0:
            for v in vals:
                if v != 0 and abs(v) > 1000 * abs(med):
                    warnings.append(f"列「{col_name}」存在疑似异常值 {v}（中位数 {med:.3g}）")
                    break

    if numeric_cols == 0:
        problems.append("除第一列外没有数值列（Coder 将无法绘图）")

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "warnings": warnings,
        "rows": len(data_rows),
        "numeric_cols": numeric_cols,
    }
