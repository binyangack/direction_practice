"""定量分析：把「分析」从“画图”升级为“定量结论”。

对 work/materials.csv 做确定性、领域无关的定量计算，作为 LLM 结论之外的“机器地面真相”：

- 判断数据形态：多对象对比（每对象一行） vs 单一对象随自变量变化的序列（同一对象多行）。
- 序列：对每个数值 y 列，按 x 排序后拟合 线性 / 幂律 / 指数 三种模型，按 R² 选优，
  给出方程、R²、趋势与极值。
- 对比：对每个数值列给出降序排名、相对最大值的百分比、区间。

刻意只用 numpy（pandas / matplotlib 的既有传递依赖），不引入 scipy；也不写死领域知识。
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict

import numpy as np

_INDEP_KEYWORDS = ("temperature", "temp", "time", "pressure", "step", "concentration", "温度", "时间", "压力")


def _parse(csv_text: str):
    rows = list(csv.reader(io.StringIO((csv_text or "").strip())))
    if not rows:
        return [], []
    header = [str(h).strip() for h in rows[0]]
    return header, rows[1:]


def _to_float(cell):
    try:
        return float(str(cell).strip())
    except (ValueError, TypeError):
        return None


def _numeric_cols(header, data):
    """返回 [(col_idx, col_name, values)]，values 为该列所有可解析为 float 的值。"""
    out = []
    for j in range(1, len(header)):
        vals = []
        ok = True
        for row in data:
            v = _to_float(row[j]) if j < len(row) else None
            if v is None:
                ok = False
                break
            vals.append(v)
        if ok and vals:
            out.append((j, header[j], vals))
    return out


def _r2(y_true, y_pred) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def _fit_models(x, y):
    """拟合三种模型，返回按 R² 选优的 (模型名, 方程字符串, R²)。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    models = []
    if n >= 2:
        b, a = np.polyfit(x, y, 1)
        yp = a + b * x
        models.append(("线性", f"y = {a:.4g} + {b:.4g}·x", _r2(y, yp)))
    if n >= 2 and np.all(x > 0) and np.all(y > 0):
        b, la = np.polyfit(np.log(x), np.log(y), 1)
        a = float(np.exp(la))
        yp = a * (x ** b)
        models.append(("幂律", f"y = {a:.4g}·x^{b:.4g}", _r2(y, yp)))
    if n >= 2 and np.all(y > 0):
        b, la = np.polyfit(x, np.log(y), 1)
        a = float(np.exp(la))
        yp = a * np.exp(b * x)
        models.append(("指数", f"y = {a:.4g}·e^({b:.4g}·x)", _r2(y, yp)))
    if not models:
        return None
    return max(models, key=lambda m: m[2])


def analyze_materials(csv_text: str) -> dict:
    """对材料 CSV 做定量分析，返回可渲染的结构化 dict。"""
    header, data = _parse(csv_text)
    result: dict = {"mode": "comparison", "group_col": "", "groups": []}
    if not header or not data:
        return result

    group_name = header[0]
    groups = [str(r[0]).strip() for r in data]
    unique_groups = sorted(set(groups))
    is_series = len(unique_groups) < len(groups)  # 第一列有重复 → 序列
    result["group_col"] = group_name
    result["groups"] = unique_groups

    numcols = _numeric_cols(header, data)
    numeric_names = [name for _, name, _ in numcols]

    if not is_series:
        # 对比：每对象一行
        comp = []
        for j, name, vals in numcols:
            vmax = max(vals) or 0.0
            ranked = sorted(zip(groups, vals), key=lambda t: -t[1])
            rows = [
                {"group": g, "value": v, "rel_pct_of_max": (v / vmax * 100.0) if vmax else 0.0}
                for g, v in ranked
            ]
            comp.append({"col": name, "ranked": rows, "range": {"min": min(vals), "max": max(vals)}})
        result["mode"] = "comparison"
        result["comparison"] = comp
        return result

    # 序列：选出“自变量列”（名字匹配关键词；否则取第一个数值列），按组拟合
    x_idx = next((j for j, name, _ in numcols if any(k in name.lower() for k in _INDEP_KEYWORDS)), None)
    if x_idx is None and numcols:
        x_idx = numcols[0][0]
    x_name = header[x_idx] if x_idx is not None else ""

    grouped = defaultdict(list)
    for row in data:
        grouped[str(row[0]).strip()].append(row)

    per_group = []
    for g in unique_groups:
        rows_g = grouped[g]
        xvals = [_to_float(r[x_idx]) for r in rows_g] if x_idx is not None else []
        fits = []
        for j, name, _ in numcols:
            if j == x_idx:
                continue
            yvals = [_to_float(r[j]) for r in rows_g]
            pts = sorted(zip(xvals, yvals), key=lambda t: t[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            entry: dict = {"col": name, "min": min(ys), "max": max(ys)}
            best = _fit_models(xs, ys)
            if best:
                entry["model"], entry["eq"], entry["r2"] = best
            if len(ys) >= 2:
                entry["trend"] = (
                    "随自变量增大而增大" if ys[-1] > ys[0]
                    else "随自变量增大而减小" if ys[-1] < ys[0]
                    else "基本平稳"
                )
            imax = int(np.argmax(ys))
            entry["max_at_x"] = xs[imax]
            fits.append(entry)
        per_group.append({"group": g, "fits": fits})

    result["mode"] = "series"
    result["series"] = {"x_col": x_name, "numeric_cols": numeric_names, "per_group": per_group}
    return result


def render_quant(analysis: dict) -> list[str]:
    """把 analyze_materials 的结果渲染成报告用的 markdown 行（机器计算部分）。"""
    lines: list[str] = []
    if analysis["mode"] == "series":
        s = analysis["series"]
        lines.append(f"数据形态：同一对象随自变量「{s['x_col']}」变化的序列。")
        for pg in s["per_group"]:
            for f in pg["fits"]:
                parts = [f"**{pg['group']} · {f['col']}**"]
                if f.get("eq"):
                    parts.append(f"最优拟合：{f['model']} {f['eq']}（R² = {f['r2']:.3f}）")
                if f.get("trend"):
                    parts.append(f"趋势：{f['trend']}")
                parts.append(f"最大 {f['max']:.4g} @x={f['max_at_x']:.4g}；区间 [{f['min']:.4g}, {f['max']:.4g}]")
                lines.append("- " + "；".join(parts))
    else:
        c = analysis.get("comparison", [])
        lines.append(f"数据形态：{len(analysis['groups'])} 个对象的对比。")
        for col in c:
            ranking = " > ".join(f"{r['group']} {r['value']:.4g}（{r['rel_pct_of_max']:.0f}%）" for r in col["ranked"])
            lines.append(f"- **{col['col']}**：{ranking}；区间 [{col['range']['min']:.4g}, {col['range']['max']:.4g}]")
    return lines
