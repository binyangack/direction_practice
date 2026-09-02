# 关键 AI 使用记录（AI Usage Log）

> 用途：配合《基于 AutoGen 的多智能体实验方案生成与数据分析》实验报告，记录在与 AI（Claude Code，底层由 `deepseek-v4-flash-vision-exp` 驱动）协作过程中提出的关键问题、AI 的回答与依据，特别是 AI 主动查阅资料、或先答错后经检索/被指正后纠正的部分。
>
> 分五组：**B. AutoGen 框架理解**、**C. 多智能体 vs 单一智能体**、**D. 联网检索（方案B）**、**E. 上下文缓存 / KV cache 原理（C3 延伸）**、**T. 时效性/事实错误与纠正**。

---

## B. AutoGen 框架理解

### B1. `autogen` 是一个库吗？`autogen_agentchat / autogen_ext / autogen_core` 分别是什么意思？各函数参数是什么？

**AI 回答：** 是本实验使用的 `autogen`（0.7.x，多智能体编排框架，Python 库）。采用三层架构：

| 模块 | 层次 | 内容 |
| --- | --- | --- |
| `autogen_core` | 基础层 | 智能体、工具、模型客户端等最底层抽象 |
| `autogen_agentchat` | 高层 | `AssistantAgent`、`CodeExecutorAgent`、`RoundRobinGroupChat`、终止条件等「组队聊天」组件 |
| `autogen_ext` | 扩展/插件层 | `LocalCommandLineCodeExecutor`（本地代码执行器）等由外部或环境提供的插件 |

常用构造参数举例：

- `AssistantAgent(name, system_message, model_client)`：LLM 驱动的对话智能体。
- `CodeExecutorAgent(name, code_executor, ...)`：执行智能体，运行由其他 agent 生成的代码。
- `LocalCommandLineCodeExecutor(timeout, work_dir)`：在本机起子进程执行代码。
- `FunctionTool(func, name, description, ...)`：把 Python 函数包装成可供 LLM 调用的工具。
- `OpenAIChatCompletionClient(model, base_url, api_key, model_info)`：具体 LLM 的网络客户端。

### B2. 「Agent 可调用的工具函数列表」是什么意思？

**AI 回答：** 工具在代码里是普通 Python 函数，但要让 LLM 能调用，需用 `FunctionTool` 包装并挂到 agent 上。挂上后：

- LLM **只能看到**工具的函数名、功能描述和参数 JSON schema，**并不能真正运行它**；
- 当模型决定该调某个工具时，会输出一个 `tool_call`（只带参数），**由 AutoGen 框架去真正执行**那个 Python 函数，再把执行结果回填给模型继续推理。

所以「工具列表」= 允许该 agent 调用的外部能力清单。模型是「提需求的那一方」，框架才是「干活执行的那一方」。

### B3. 工具 / 模型客户端 / Docker 沙箱是什么？AutoGen 是否不涉及神经网络、完全靠规则？

**AI 回答：**

- **工具**：给模型调用的函数包装，模型见描述+schema，AutoGen 代执行。
- **模型客户端**：某个具体 LLM 的网络驱动，负责鉴权、`base_url`、模型名、消息格式转换，AutoGen 靠它与远程模型 API 通信。
- **Docker 沙箱**：把 LLM 生成的代码放进隔离容器执行，比在宿主机直接跑更安全；`LocalCommandLineCodeExecutor` 则是直接在宿主机的子进程执行。
- **AutoGen 框架本身不包含神经网络**，是完全确定性的规则：它只看消息里有没有 `tool_call`、有没有触发终止的 `TERMINATE` 标记，有就执行对应工具/结束；没有则把当前状态交给 LLM 决定。分工是「**框架当指挥、LLM 当乐手**」——框架决定何时调用模型、调用哪个；模型决定输出什么内容。

---

## C. 多智能体 vs 单一智能体（实验核心主题）

### C1. 这种多智能体协作相比一个单一智能体有什么优点？

**AI 回答（第一版，详见 C2 修正）：**

- **角色与工具隔离**：各 agent 有不同 `system_message`；只有 Research 挂了写表工具、只有 Executor 能执行代码——「谁有什么权限」成为显式规则，而不是靠模型自律。
- **强制工件链条（护栏）**：数据必须写进 `materials.csv` → 代码必须真跑到 `comparison.png` → Analyst 只能解释**真跑出来**的结果。这规避了「单智能体跳过计算、直接编造结论」的典型失败。
- **显式终止契约**：`TextMentionTermination("TERMINATE", sources=["Analyst"]) | MaxMessageTermination(30)`，整个团队何时停止是硬约定。
- **逐角色可观测**：报告能精确归因到每个 agent（Planner/Research/Coder/Executor/Analyst）。

> ⚠️ 此版本中「每段独立上下文、防遗忘」的优点是**错误的**，已在 C2 查源码证伪并修正。

### C2. 现在的实现真的做到「每段独立上下文」了吗？还是其实是一份相同的超长上下文？token 调用是单一智能体的 5 倍吗？

**AI 回答（查 AutoGen 源码后修正，关键自我纠错）：**

- **不是独立上下文，是共享且不断增长的上下文。** 依据源码：
  - `autogen_agentchat/teams/_group_chat/_base_group_chat.py` 的 docstring 明确：「participants **share context by publishing their messages to all other participants**」。
  - `_base_group_chat_manager.py` 内部持有一份全局 `self._message_thread`，每次消息都 extend。
  - `_chat_agent_container.py` 的 `handle_request` 把**整个 buffer（到目前的完整共享记录）**传给 `on_messages_stream`；Agent 再把它加入自己的 `_model_context`，用**自己的 system prompt + 这份全部历史**调 LLM。
  - 所以真正独立的**只有 system prompt（角色指令+工具）**，消息记忆是共享且一路膨胀的。
- **线性流水线下各 agent 拿到的是「前缀」**：Planner 只见用户任务，Research 见 +Planner，Coder 见 +Research，Executor（无 model_client，纯执行不算模型调用），Analyst 见全量。
- **token 账**：是 **4 次模型调用（不是 5 次）**——Executor 是 `CodeExecutorAgent` 且未给 `model_client`，走纯执行分支，不碰模型。总 token ≈ 前缀和 `4U+3P+2R+C+E`，**不是「5×」**；按裸 token 量级约 1.5–2.5 倍，但把缓存折算进来看 C3 会更不划算。
- **结论**：C1 里「独立上下文/防遗忘」不成立；真正站得住的是「角色/工具隔离 + 强制工件链护栏」。

### C3. 多智能体的 kv-cache 命中率会比单一智能体低吗？

**AI 回答（WebSearch 核对后）：**

- **会，而且是断崖式：本配置下命中率几乎归零，单智能体则可达 90%+。**
- 原理：缓存只认「从 token 0 起相同的前缀」。vLLM 用「块 hash 链」从头匹配，一旦分叉整条链失效；DeepSeek 官方文档写明「仅在两个请求**从 token 0 起的前缀完全一致**时命中，中段的部分匹配不触发」（并以 64 token 为存储粒度）。
- 对照：
  - 单智能体：`Req1: S+U`、`Req2: S+U+A1`……每次新增请求几乎全部命中上一轮的缓存前缀。
  - 多智能体：`Req1: S_P+U`、`Req2: S_R+U+P`……每个请求**第 0 个 token 是不同的 system prompt**，在 token 0 即分叉，后面 U/P/R/C/E 再像也没用 → 全 miss。
- **讽刺点**：这套共享上下文的内容大量重复，本应是缓存福地；但都被不同 system prompt 挡在「前缀之外」，等于白共享。
- **影响**：反过来修正 C1/C2 的成本判断——按 DeepSeek v4 缓存命中与未命中约 **30 倍价差**（如 v4-flash off-peak $0.007/M vs $0.22/M），单智能体因大部分输入走命中价，实际成本可能**明显低于**多智能体。上一版「约 1.5–2.5 倍」是按裸 token 算的，未计缓存，需以本节为准。
- **可实测**：DeepSeek 的 `usage` 返回 `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`。
- **来源**：DeepSeek API 官方公告（Context Caching on Disk）、vLLM Automatic Prefix Caching 文档。

---

## D. 联网检索（方案B：给系统加真实搜索能力）

### D1. 为什么各模型的“网页端”能上网检索？Claude Code 是怎么实现的？能借鉴吗？

**AI 回答：**

- **核心：模型自己不会上网。** LLM 是纯文本进、纯文本出的神经网络，没有任何网络访问能力。所谓“联网检索”是外围应用替它做的三步循环：①模型判断“这要查一下”→ 输出 `tool_call`；②真正发 HTTP 请求、调搜索引擎的**不是模型，是应用**；③搜索结果被当作额外上下文回填进模型输入，模型据此写出带引用的答案。即“**大脑（模型）与手脚（harness）分离**”。
- 三种形态，差别只在“谁去执行”：
  - **各 AI 网页端**（DeepSeek / Claude / ChatGPT 网页）：`tools:[{"type":"web_search"}]`，由**厂商数据中心**自主决定何时搜、搜完把结果注入上下文并返回引用。
  - **Claude Code 的 WebSearch**：**服务端工具**——CLI 只把 query 打包成 `web_search_20250305` 经 Messages API 上传，真正的搜索在 Anthropic 侧完成。这也是为什么它在境内也能用：因为搜索根本不走你本地网络。只在使用实现该服务端工具的模型后端（Anthropic / 支持网关）时可用。
  - **Claude Code 的 WebFetch**：**客户端**——你本地发 HTTP 抓网页再转 Markdown（属于“本地抓取”这一系）。
- **能借鉴的是“循环”，借不到的是“后端”。** 那套“模型决定何时搜 + harness 执行 + 结果回填”的循环，方案 B 本来就是这么做的；唯独厂商“自家数据中心的私有搜索后端”借不到，只能从你的机器调一个第三方搜索 API（Tavily / Brave / Exa / SerpAPI），由服务商数据中心完成检索、返回干净 JSON，代价是需注册一个免费 API key。
- **顺带纠正**：DeepSeek 原生 `web_search` 只能经其 **Responses API（`/responses`，且仅 deepseek-v4-flash）** 或 Anthropic 兼容端点启用，**不是**给 agent 用的 `/chat/completions`，所以“直接让 DeepSeek 联网”在当前 `AssistantAgent` 上走不通，除非整体换 `model_client` / 端点。
- **来源**：Claude.ai 网络工具文档、[mikhailshilkov 对 Claude Code WebSearch/WebFetch 的实现分析](https://github.com/mikhailshilkov/mikhailio-hugo/issues/60)、[DeepSeek Anthropic 兼容 web_search 说明](https://cdn.jsdelivr.net/npm/@deepseek-ai/dsh-web-search-deepseek@0.0.1-rc.1/README.zh.md)、[pi-deepseek-responses-search](https://github.com/chasedream1129/pi-deepseek-responses-search)。

### D2. 方案B：为什么最终选 Tavily，而不是本地抓取 / Bing？

**AI 回答（工程权衡 + 实测）：**

- **候选一：本地刮 Bing（cn.bing.com）**——境内可达、无需 key，但中文多词查询偶发退化成“单个汉字/词典”页（如“纯铜 电导率 铜锌合金”被串成“纯”字的词典结果），抓取脆落、不可靠，已实测。
- **候选二（最终选用）：Tavily 搜索 API**——服务端检索、返回干净 JSON（title/content/url），需免费 key。
- **落地**：`search_web(query, max_results)` 用 `Authorization: Bearer <key>` POST 到 `https://api.tavily.com/search`，body `{"query","max_results","search_depth":"basic"}`；内容超 400 字符截断避免撑爆上下文；未配置 key / 检索失败时返回提示，模型回退常识值（最坏情况 = 原基于记忆的方案，不劣化）。
- **实测效果**：Research 拿到“黄铜导热≈109~121、纯铜≈386~401 W/(m·K)、黄铜电阻率≈0.071 Ω·mm²/m、纯铜≈0.018 Ω·mm²/m”等真实值，足以支撑“合金化使导热/导电随锌含量单调下降”的规律分析。
- **来源**：Tavily 官方 docs / agent-setup SKILL.md、[Claude.ai 网络工具文档](https://claude.com/docs/third-party/claude-desktop/web-tools)。

### D3. 踩坑：加了搜索工具，为什么 CSV 还是没写出来？（`max_tool_iterations` 默认=1）

**AI 回答（查 AutoGen 源码定位）：**

- **现象**：Research 正确调用了 `search_web`×3、拿到真实数据，但回合结束时只输出一串原始检索 dump，**从没调用 `write_materials_csv`** → `materials.csv` 缺失 → Executor 报 `FileNotFoundError`，`comparison.png` 也未生成。
- **根因**：AutoGen `AssistantAgent` 的 **`max_tool_iterations` 默认为 1**（`_assistant_agent.py:739`），即“执行完一轮工具”就强制 `break` 去总结，模型没有第二次机会再调 `write_materials_csv`。
- **修复**：给 Research 设 `max_tool_iterations=5`（AutoGen 自身多工具示例用的就是 5），让模型能走 “search → 读结果 → write_csv → 文本总结” 的多步工具链；顺带结尾输出的是**简洁总结**而非原始 dump，更省 token。
- **教训**：这类“默认参数只够一轮”的隐性约束，只能靠查源码确认，不能靠记忆推断；工具类 Agent 要支持多步工具链，必须显式放宽 `max_tool_iterations`。

---

## E. 上下文缓存 / KV cache 原理（C3 的延伸）

### E1. 怎么提高 kv-cache 命中率？（多智能体的结构性矛盾）

**AI 回答（WebSearch 核实 DeepSeek 缓存规则后）：**

- **本质**：提高命中率 = 让「共享内容」从第 0 个 token 起、把「差异内容」推到末尾。而多智能体把角色差异（`system_message`）顶在 token 0，恰好把共享的、不断膨胀的历史记录挡在缓存前缀之外——这是命中率归零的根因。
- DeepSeek 缓存规则（硬约束）：只认「从 token 0 起完全一致的前缀」，中段局部匹配不命中；64 token 一个存储单元；命中 vs 未命中差约 **30 倍**（命中折扣约 96.8%）。agent 型负载实测命中率可达 95%+，所以「能不能命中 ≈ 能不能省钱」。
- 杠杆（按收益排序）：
  1. **统一 system prompt、把角色降级为「历史之后的一句指令」**（收益最大）：四个 agent 共用一个 `system_message`，角色差异挪到历史末尾追加 → 历史从「白共享」变成逐级前缀链，每个 agent 命中上一个 agent 的整段输入。
  2. **坍缩成单智能体**（理论天花板 ~97%，违背题目3 的 5-agent 要求，只能当参照）。
  3. **稳定前缀跨 run 复用**（缓存 TTL 小时~天级，REPL 反复跑同任务命中 4 个 system 头）。
  4. **缩小必然 miss 的部分**：每个 agent 独有的 system 和工具 schema 永远 miss，尽量短。
  5. **确定性序列化 + 实测**：字节级一致才命中；打印 DeepSeek 返回的 `usage.prompt_cache_hit_tokens / miss_tokens` 验证，别靠推理。
- **结构性矛盾（关键）**：缓存最优 与「用 system_message 做角色/工具隔离」**不可兼得**——角色写进 system = token 0 分叉 = 缓存归零。要命中率就得把角色护栏降级为软指令。
- **来源**：[DeepSeek 上下文缓存文档](https://api-docs.deepseek.com/zh-cn/guides/kv_cache/)、[DeepSeek 定价](https://api-docs.deepseek.com/quick_start/pricing/)、[实测涨价与命中率（《财经》）](https://www.mycaijing.com/article/detail/578141?source_id=40)

### E2. KV cache 到底怎么实现？为什么能查历史、却命中不了上月缓存？

**AI 回答：**

- **先把三个被混在一起的东西拆开**：

| | 是什么 | 存哪 | 多久 |
|---|---|---|---|
| ① 对话记录 | 纯文本 | 数据库/对象存储（磁盘） | 永久 |
| ② KV cache | 每层 attention 的 K/V 中间激活 | 显存 | 请求进行时，结束即释放 |
| ③ 前缀缓存 | 可复用前缀的 KV 单元 | 磁盘 | 几小时~几天，淘汰 |

- 「随时能查历史」靠 ①；「缓存命中」算 ③；「占显存」是 ②，三者不是一回事。
- **KV cache 原理**：自回归解码每步都要读全部过去 token 的 K/V，缓存它们避免重算；解码是内存带宽瓶颈，所以必须放显存。
- **显存需求（结构公式）**：`KV = 2(K和V) × L层 × H_kv头 × d_head × T长度 × 2字节`。70B 实算（L=80,H_kv=8,d_head=128）≈ **320 KB/token**，满 128K ≈ **40 GB/请求**——这就是为什么一张 80GB 卡服务不了几条长上下文。
- **TTL 与上月对话**：缓存「几小时~几天」不用即淘汰，所以上个月对话**不能命中**（会全 miss 重新 prefill，正确性不受影响，只是这段输入多付点钱）；但**能查到历史**，因为 ① 文本永久存着，和 ③ 无关。
- **来源**：[DeepSeek 上下文缓存文档](https://api-docs.deepseek.com/zh-cn/guides/kv_cache/)、[磁盘缓存公告](https://github.com/thevibeworks/deepseek-docs/blob/main/content/en/news/news0802.md)

### E3. MLA 会让推理变慢吗？磁盘缓存降低显存要求吗？（纠正一条错误因果链）

**AI 回答（WebSearch 核实 MLA 出处后）：**

- **MLA 不慢，反而快**：解码是内存带宽瓶颈而非算力瓶颈，MLA 把 KV 压到约 1/4~1/8（V2 论文实测 **KV 减少约 93.3%**），每步读取量骤降 → 生成速度 **5.76×**。代价是每步一次「上投影」小矩阵乘，但解码阶段算力本就过剩，几乎免费。
- **纠正用户搭的因果链（错了一环）**：`MLA→KV小→落盘划算→容量大→命中率高` 基本对；但「**KV 在磁盘 → 降低显存要求**」**错误**——命中时 KV 照样搬回显存参与解码，活跃请求的显存峰值不变。**降低显存的是 MLA 本身**（每 token KV 变小），磁盘缓存降低的是「要不要重算 prefill」和计费，还让首 token 更快（128K 从 13s → ~500ms）。
- **MLA 是 DeepSeek 首创**（2024-05 DeepSeek-V2，arXiv:2405.04434），「**压缩而非共享**」，区别于 MQA/GQA 的「减少 KV 头数」。
- **其他厂商能到那么高命中率吗？** 能——命中率是「工作负载 + 缓存容量/保留策略」的函数，不是厂商固定值；各家 agent 负载实测都能到 80~96%。DeepSeek 独有的是「**磁盘缓存 + 长 TTL**」，靠 MLA 才有资格落盘。
- **来源**：[DeepSeek-V2 论文](https://arxiv.org/pdf/2405.04434)、[MLA 详解（Raschka）](https://sebastianraschka.com/llm-architecture-gallery/mla/)、[MLA 硬件分析](https://www.arxiv.org/pdf/2506.02523)、[各厂商缓存对比](https://www.edenai.co/post/prompt-caching-claude-vs-gpt-vs-gemini-cost-playbook)

### E4. 缓存怎么查找？为什么 Anthropic/Gemini 要手动标？vLLM 是什么？

**AI 回答：**

- **缓存是「按内容寻址」，不是「按用户/会话寻址」**：服务器不「检索」某个用户过去的缓存，只拿你这次输入的 token 序列算 hash，问「磁盘上有没有一段 KV 和我这次输入开头逐 token 一样」。**token 内容本身就是地址**，没有「会话 ID → 磁盘位置」的标记指路。
- **靠哈希，不是语义检索**：输入切成 64-token 块 → 每块算 hash → 在哈希表/前缀树里从第 0 块连续匹配，O(块数)、精确匹配。一个空格不同 hash 就不同——这就是「字节级一致」的由来。内部标记 = 内容哈希 + 账户作用域（文档明说「用户缓存相互隔离」）。
- **手动标 vs 自动的底层逻辑**：缓存越贵、TTL 越短 → 越要开发者手动标边界（**Anthropic** `cache_control` 有写费 1.25×/2×、TTL 5min/1hr，边界只有开发者自己最清楚；**Gemini** `CachedContent` 按小时计费、最少 1h，天生需显式生命周期）；缓存越便宜、容量越大 → 越可全自动（**DeepSeek/OpenAI**）。
- **纠正我此前的含糊说法**：DeepSeek 是「**权重开源（open-weight）、服务端（推理引擎/磁盘缓存）闭源**」——模型权重可下载，跑 API 的机房引擎不公开，所以缓存机制只能靠文档规则推断。
- **vLLM**：开源 LLM 推理/服务引擎（UC Berkeley 团队），贡献了 PagedAttention（KV 按块管理）与「块级前缀缓存」，是「hash 怎么当缓存钥匙」这套机制的**可读、公开实现**——DeepSeek 自己那套没得读，vLLM/SGLang 加 MLA 后也能跑 DeepSeek 模型。
- **来源**：[vLLM 文档](https://docs.vllm.ai/)、[各厂商缓存实现差异](https://dev.to/multigrid/prompt-caching-how-the-major-implementations-differ-1nbh)、[Claude vs GPT vs Gemini 对比](https://www.edenai.co/post/prompt-caching-claude-vs-gpt-vs-gemini-cost-playbook)

---

## T. 时效性 / 事实错误与纠正（AI 先答错，后检索或被指正）

### T1. 「你对 DeepSeek 模型型号和 DeepSeek Harness 理解有误，搜索后重新告诉我结果」

**AI 回答（检索纠错后）：**

- 当前 DeepSeek 模型为 **`deepseek-v4-flash`、`deepseek-v4-pro`、`deepseek-v4-flash-vision-exp`**；**`deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 退役**。
- **DeepSeek Harness（DSH / `dsh`）确实存在**：开源、基于 Node.js + Cordis（「一切皆插件」）的 **meta agent harness**，提供 web / TUI / headless / Python SDK 四种形态。它不是某个「写代码成品」，而是可用来搭建 agent 的**元框架**。
- **教训**：AI 之前的记忆停留在更早版本，凭记忆报模型名与产品形态 → 过时。此类可随时变化的硬事实必须「先检索、再下结论」。
- 来源：DeepSeek 发布公告、DeepSeek Harness 开源仓库。

### T2. DeepSeek Harness 是否只有网页版？Codex 默认是 CLI 还是交互式？豆包有没有工作模式？你所说的「泄露」到底指什么？

**AI 回答（检索 + 被指正后）：**

- DSH **不是只有网页**，尚有 TUI / headless 等形态。
- **Codex 默认是交互式桌面/工作台形态，不是 CLI**（我此前答成 CLI，错误）。豆包**除了聊天也有工作模式**。
- 「**泄露**」澄清：不是指用户数据泄露，而是指 **Claude Code（闭源、本地壳 + 远程模型架构）如何保护自家源代码不被反编译/蒸馏**——通过 minified bundle、混淆、native binary、防蒸馏的「fake tool」请求、attestation hash（`cch=`）、死代码消除、服务端逻辑、ToS 等；其于 2026-03-31 曾发生一次 source-map 泄露（约 512k 行 / 约 1900 文件）。框架本体运行在本地，模型逻辑在远程。
- **教训**：① 不能凭记忆断言产品「默认形态」，须查；② 先区分语义——「用户数据」泄露 vs「自家代码」防泄露，两者完全不同。

### T3. 你的知识截止到什么时候？为什么总是提供过时信息？

**AI 回答：**

- AI 由 `deepseek-v4-flash-vision-exp` 驱动，**无法读取自身训练截止点**；此前多次凭内部记忆回答，给出过时/错误的版本（模型名、产品形态、发布时间、价格等）。
- 承认这是缺陷，并确立规则：**凡可随时间变化的硬事实（模型名/版本、产品发布与形态、release 时间、benchmark、定价、API 格式、生态现状）→ 先 `WebSearch` 核实再下结论；凡结构性、不变的概念（框架=规则 vs 模型=神经网络；Agent=本地壳+远程脑）可凭理解直接讲**。
- 此教训已固化到全局 `CLAUDE.md`，口诀：「**越新越不能靠记忆。**」

---

> **记录要点**：本实验最真实体现 AI 作用的三段碰撞是 ——
> ① 我起初以为「多智能体 = 每段独立上下文 / 更省 token / 缓存友好」；
> ② 你追问后我**查 AutoGen 源码**，推翻「独立上下文」，改成「一份共享超长上下文 + 各自 system prompt」，并在 C2 纠正 C1；
> ③ 你进一步问缓存，我**检索确认前缀缓存机制**，指出多智能体命中率几乎归零、且折算后成本反而可能更高（C3 反向修正 C1/C2）。
> ④ 你沿缓存追问到底，我补齐了 KV cache 的完整机制链（三层存储、MLA 首创与磁盘缓存、内容寻址查找、手动标 vs 自动），并在你说「DeepSeek 不是开源的吗」时纠正了「闭源」的不严谨说法为「权重开源/服务端闭源」（E1–E4）。
> 同类的还有 T1/T2：我先凭记忆答错模型型号与产品形态，经检索与指正后纠正。这类「事实会被时间改变、须不断检索验证」正是 AI 使用的真实局限，也是可写进报告的对 AI 的观察。
