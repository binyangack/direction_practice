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

数据来源：**优先用 Tavily 联网检索真实属性数据**；检索失败或无有效结果时回退到已知常识/估算值（并如实标注），保持“无外网也能跑”的低门槛。

## 二、多智能体架构

使用 AutoGen 0.7.x 的 `RoundRobinGroupChat`（固定顺序、确定性编排）驱动 5 个智能体：

```
用户研究目标
   │
   ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Planner     │──▶│ Research    │──▶│ Coder       │──▶│ Executor    │──▶│ Analyst     │
│ 实验规划员   │   │ 资料收集员   │   │ 程序生成员   │   │ 实验执行员   │   │ 结果解释员   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
   拆解任务           联网检索并保存     生成分析/绘图      运行代码并         解释结果，
   步骤+数据需求      材料数据表         代码              捕获结果          输出报告结论
```

| 角色 | 类型 | 职责 | 关键实现 |
| --- | --- | --- | --- |
| Planner | `AssistantAgent` | 拆解实验步骤与数据需求 | 无工具，纯文本规划 |
| Research | `AssistantAgent` + 工具 | 联网检索并保存标准化的材料数据表 | `search_web`（Tavily）+ `write_materials_csv` 落盘；`max_tool_iterations=5` |
| Coder | `AssistantAgent` | 自动生成分析/绘图 Python 代码 | 防御式读取 + 按**数据形态**自适应图型（对比↔柱状图，序列↔折线图） |
| Executor | `CodeExecutorAgent` | 本地运行 Coder 代码 | `LocalCommandLineCodeExecutor` + `sources=["Coder"]` |
| Analyst | `AssistantAgent` | 解释结果、给出物理机理与工程建议 | 结论末尾写 `TERMINATE` 触发终止 |

**编排与终止：** `RoundRobinGroupChat` 按顺序轮流发言；`TextMentionTermination("TERMINATE", sources=["Analyst"]) | MaxMessageTermination(max_messages=30)` 作为正常结束 + 兜底；`RoundRobinGroupChat(max_turns=10)` 再兜一层。

## 三、运行产物

一次运行在 `work/` 下自动生成：

- `work/materials.csv` —— 标准化材料数据表（Research 生成）
- `work/data_source.md` —— 数据来源说明（Research 写入；检索成功注 URL，失败标注模拟值）
- `work/analysis.py` —— 分析/绘图代码（Coder 生成，Executor 已执行）
- `work/comparison.png` —— 可视化结果（对比柱状图 或 随自变量变化的折线图，视任务而定）
- `work/report.md` —— 汇总的完整实验报告

**示例一（多材料对比）：** `比较钢、铝、铜的导热性能，并画出对比柱状图。`
结论：**铜（385 W/(m·K)）＞ 铝（205 W/(m·K)）＞ 钢（50 W/(m·K)）**，从电子导热、晶格/杂质散射、密度与比热解释机理，给出散热材料选型建议。

**示例二（单一材料随温度变化）：** `收集纯铜在不同温度下的导热系数，绘制导热系数随温度变化的曲线，并分析趋势与机理。`
Research 生成 `材料, 温度_K, 导热率`（纯铜多行）；Coder 识别为序列数据，画出**折线图**（x=温度、y=导热率），而非柱状图。

> 图内文字为中文，已通过项目级 `.mpldir/matplotlibrc` 配置中文字体解决乱码。

## 四、目录结构

```
direction_practice/            # 仓库根（运行须在此目录用 -m 执行）
├── .env.example               # 密钥模板（提交）；复制为 .env 并填入真实 key
├── .gitignore                 # 忽略 .env、work/、.mpldir/、*.pdf 等
├── .gitattributes             # 统一 LF 换行，避免跨平台 diff
├── CLAUDE.md                  # 面向 Claude Code 的项目说明
└── q3_autogen/
    ├── README.md              # 本说明
    ├── config.py              # LLM 客户端 + DeepSeek 转换注册 + .env 加载 + 中文字体/警告
    ├── agents.py              # 5 个智能体的定义与 system message（角色提示词）
    ├── tools.py               # 供 Agent 调用的工具（Tavily 检索、写 CSV）
    ├── main.py                # 入口：编排团队、运行、生成报告、REPL
    ├── .mpldir/               # 项目级 matplotlib 配置（中文字体）
    └── work/                  # 运行产物：csv / md / py / png / report.md（已被忽略）
```

## 五、快速开始

### 1. 依赖

```bash
pip install "autogen-agentchat" "autogen-ext[openai]" pandas matplotlib requests
```
（当前仓库使用 AutoGen 0.7.x；Python 3.10+。）

### 2. 配置密钥（**.env**，不提交）

复制模板并填入自己的 key：

```bash
cp .env.example .env    # Windows: copy .env.example .env
```

| 配置项 | 值 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 必填。DeepSeek 开放平台领取。缺失时真实调用会鉴权失败 |
| `TAVILY_API_KEY` | 选填。Tavily 领取。缺失时 Research 回退到记忆/估算值。 |
| 模型名 | `deepseek-v4-flash-vision-exp`（默认） |
| 基址 | `https://api.deepseek.com` |

`config.py` 从**环境变量（优先）或项目根目录 `.env`** 读取密钥，**代码里没有明文 key**。可用环境变量 `Q3_MODEL` / `Q3_BASE_URL` 覆盖模型与基址。`.env` 已被 `.gitignore` 忽略，不会提交。

### 3. 运行

须在**仓库根目录**用模块方式执行（相对导入与 `work/` 路径依赖于此）：

```bash
# 交互模式：连续输入研究目标，exit 退出
python -m q3_autogen.main

# 先跑指定任务，再进入交互模式
python -m q3_autogen.main --task "比较钢、铝、铜的导热性能，并画出对比柱状图"
python -m q3_autogen.main --task "收集纯铜在不同温度下的导热系数并绘制随温度变化的曲线"
```

> **Windows 台注意：** 控制台默认 GBK，管道会打印中文，请加 `PYTHONIOENCODING=utf-8`（否则可能 `UnicodeEncodeError`）：
> ```bash
> PYTHONIOENCODING=utf-8 python -m q3_autogen.main
> ```

## 六、设计要点

- **联网检索 + 来源诚实**：Research 用 `search_web`（Tavily，服务端检索返回干净 JSON）拉真实数据；`write_materials_csv` 的 `source` 参数把来源写进 `work/data_source.md` —— 检索成功注 URL，失败写明“⚠️ 模拟值/估算值，未联网验证”。报告「三、材料数据」以 blockquote 呈现该说明，不做无来源的冒充实测。
- **`max_tool_iterations=5`**：AutoGen `AssistantAgent` 默认 `max_tool_iterations=1`，只够“执行一轮工具”即结束，会导致 Research 搜完就停、不写 CSV。放宽到 5 才能走 “search → 读结果 → write_csv → 总结” 的多步工具链。
- **防御式代码生成**：Coder 不写死列名——`group_col=df.columns[0]`，数值列用 `is_numeric_dtype` 判定，空则报错；并对中英文列名、任意列数稳健。
- **自适应图型**：Coder 先探查数据形态——第一列有重复值（同一对象多行）或存在自变量列（temperature/时间/压力/step 等）→ 序列数据 → **折线图**（x=自变量、y=各数值属性，多对象各画一条线）；否则 → 对比数据 → **柱状图**。
- **文件化数据交接**：Research 落盘 `materials.csv`，Coder 读取、Executor 在工作目录执行——避免跨 Agent 传文本的脆弱性；每任务开始先清掉上一次的中间产物。
- **可靠执行**：顺序固定；Executor 只执行 `sources=["Coder"]` 的代码块；中文字体由项目内 `.mpldir/matplotlibrc` 保证；Analyst 以 `TERMINATE` 收尾，配合兜底 `MaxMessageTermination` 避免死循环。

## 七、已知限制

- 生成代码/数据具一定随机性（LLM 采样）；多次运行数值与措辞会有差异，但流程稳定。
- 演示用 `LocalCommandLineCodeExecutor` 直接在**本机**执行 LLM 生成的代码，适合教学；生产环境建议改用 Docker 沙箱。
- 模型默认 `deepseek-v4-flash-vision-exp`；换其它 OpenAI 兼容接口时，仅改 `config.py`（并注意 DeepSeek 家族转换器）。
- 系统提示词用于教学、偏指令化；换任务形态（如“多材料 × 多自变量”）时依赖模型对提示词的理解，偶有需要重跑。
