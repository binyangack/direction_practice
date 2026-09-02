# 题目3：基于 AutoGen 的多智能体实验方案生成与数据分析

> 对应任务书《实践项目任务书汇总.pdf》题目3。
> 在“材料分析”场景下，用 AutoGen 编排智能体，完成 **自然语言需求 → 实验方案 → 数据 → 代码 → 执行 → 可视化 → 报告** 的闭环。

---

## 一、项目目标

体验多智能体协作的全流程，训练用 AutoGen 编排 Agent：

1. 接收研究目标；
2. 生成实验 / 分析方案；
3. 自动生成代码；
4. 执行代码并返回结果；
5. 自动解释与可视化，输出实验报告。

用公开/模拟数据即可，保持低门槛。

## 二、多智能体架构

使用 AutoGen 0.4+ 的 `RoundRobinGroupChat`（固定顺序、确定性编排）驱动 5 个智能体：

```
用户研究目标
   │
   ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Planner     │──▶│ Research    │──▶│ Coder       │──▶│ Executor    │──▶│ Analyst     │
│ 实验规划员   │   │ 资料收集员   │   │ 程序生成员   │   │ 实验执行员   │   │ 结果解释员   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
   拆解任务           生成并保存        生成分析/绘图      运行代码并         解释结果，
   步骤+数据需求      材料数据表        代码              捕获结果          输出报告结论
```

| 角色 | 类型 | 职责 | 关键实现 |
| --- | --- | --- | --- |
| Planner | `AssistantAgent` | 拆解实验步骤与数据需求 | 无工具，纯文本规划 |
| Research | `AssistantAgent` + 工具 | 生成“模拟材料数据”并存为 `materials.csv` | 内置 `write_materials_csv` 工具落盘 |
| Coder | `AssistantAgent` | 自动生成分析/绘图 Python 代码 | 生成防御式脚本（自动识别材料列/目标列） |
| Executor | `CodeExecutorAgent` | 本地运行 Coder 代码 | `LocalCommandLineCodeExecutor` + `sources=["Coder"]` |
| Analyst | `AssistantAgent` | 解释结果、给出物理机理与工程建议 | 结论末尾写 `TERMINATE` 触发终止 |

**编排与终止：** `RoundRobinGroupChat` 按顺序轮流发言；`TextMentionTermination("TERMINATE", sources=["Analyst"])` 在结论生成后结束对话，`MaxMessageTermination(30)` 作为兜底。

## 三、运行结果

内置示例任务：`比较钢、铝、铜的导热性能，并画出对比柱状图。`

一次运行自动生成：

- `work/materials.csv` —— 材料数据表（Research 生成）
- `work/analysis.py` —— 分析/绘图代码（Coder 生成，Executor 已执行）
- `work/comparison.png` —— 导热性能对比柱状图
- `work/report.md` —— 汇总的完整实验报告

示例输出结论：**铜（385 W/(m·K)）＞ 铝（205 W/(m·K)）＞ 钢（50 W/(m·K)）**，
并从电子导热、晶格/杂质散射、密度与比热三方面解释机理，给出散热材料选型建议（铜高效但成本高、铝性价比最优、钢宜承重结构件）。

> 图中文字为中文（柱状图、坐标轴），已通过 `MPLCONFIGDIR` 配置项目级中文字体解决乱码问题。

## 四、目录结构

```
q3_autogen/
├── README.md          # 本说明
├── config.py          # LLM 客户端配置 + 中文字体/警告处理
├── agents.py          # 5 个智能体的定义与 system message
├── tools.py           # 供 Agent 调用的工具（写 CSV）
├── main.py            # 入口：编排团队、运行、生成报告
├── .mpldir/           # 项目级 matplotlib 配置（中文字体）
└── work/              # 运行产物：csv / py / png / report.md
```

## 五、快速开始

### 1. 依赖

```bash
pip install "autogen-agentchat>=0.4" "autogen-ext[openai]>=0.4" pandas matplotlib
```

### 2. 配置模型（DeepSeek OpenAI 兼容接口）

模型名与 API Key 已直接写死在 `config.py`（教学演示，明文 key 不适合生产）：

| 配置项 | 值 |
| --- | --- |
| 模型名 | `deepseek-chat` → 已改为 `v4-flash-vision` |
| API Key | 项目根目录 `.env` 中的 `DEEPSEEK_API_KEY` / `TAVILY_API_KEY`（参照 `.env.example`；已被 `.gitignore` 忽略，不会提交） |
| 基址 | `https://api.deepseek.com` |

仍可用环境变量 `Q3_MODEL` / `Q3_BASE_URL` 覆盖基址与模型（key 已不再从环境读取）。

### 3. 运行内置示例

```bash
python -m q3_autogen.main
```

### 4. 自定义研究目标

```bash
python -m q3_autogen.main --task "比较不同材料的导热性能并画出对比柱状图"
python -m q3_autogen.main --task "分析几种金属的密度差异并可视化"
```

## 六、设计要点

- **防御式代码生成**：Coder 生成的脚本**不写死列名**，而是用 `df.columns[0]` 取材料列、按列名关键词（thermal/导热/W_per_mK）选目标数值列，兜底取第一个数值列——这样 Coder 与 Research 解耦，且对中英文列名都稳健。
- **文件化数据交接**：Research 用工具把数据落盘为 `materials.csv`，Coder 读取它、Executor 在工作目录执行——避免跨 Agent 传文本导致的脆弱性。
- **Reliable execution**：`planner→research→coder→executor→analyst` 顺序固定；Executore 只执行 `sources=["Coder"]` 的代码块；中文字体由项目内 `.mpldir/matplotlibrc` 保证。
- **终止标记**：Analyst 结论以 `TERMINATE` 收尾，配合兜底 `MaxMessageTermination`，避免偶发漏写导致的死循环。

## 七、已知限制

- 生成代码/数据具一定随机性（LLM 采样）；多次运行数值与措辞会有差异，但流程稳定。
- 演示用 `LocalCommandLineCodeExecutor` 直接在本机执行 LLM 生成的代码，适合教学；生产环境建议改用 Docker 沙箱。
- 模型选用 `deepseek-chat`，具备函数调用能力；若换其它 OpenAI 兼容接口，可仅改 `config.py`。
