"""模型客户端配置。

统一封装 AutoGen 使用的 LLM 客户端（DeepSeek 的 OpenAI 兼容接口），供五个 Agent 复用。

说明
----
- AutoGen 0.4+ 的 OpenAI 模型客户端在把 ``TextMessage`` 等消息序列化为 OpenAI 格式之前，
  需要找到对应“模型家族(family)”的消息转换器。DeepSeek 不在内置家族列表中，因此这里
  手动为其注册一个基于 ``__BASE_TRANSFORMER_MAP`` 的转换映射，否则调用会抛
  ``Unknown message type``。
- ``model_info`` 用于携带模型能力描述（是否支持函数调用、结构化输出等）与家族名。
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

from autogen_ext.models.openai import OpenAIChatCompletionClient

# ---------------------------------------------------------------------------
# 为 DeepSeek 注册消息转换家族（必须先于客户端创建/使用）
# ---------------------------------------------------------------------------
from autogen_ext.models.openai._message_transform import __BASE_TRANSFORMER_MAP  # noqa: E402
from autogen_ext.models.openai._transformation.registry import (  # noqa: E402
    MESSAGE_TRANSFORMERS,
    register_transformer,
)

# 若家族尚未注册则注册一次（幂等，避免重复导入时覆盖）
if "deepseek" not in MESSAGE_TRANSFORMERS.get("openai", {}):
    register_transformer("openai", "deepseek", __BASE_TRANSFORMER_MAP)

# AutoGen 会把不认识的模型名做家族推断并给出“模型不匹配”的估算警告，这里忽略之。
warnings.filterwarnings(
    "ignore",
    message=r"Resolved model mismatch.*",
    category=UserWarning,
)

# 噪声警告：本地代码执行器 / 无审批函数的安全提示（本项目为教学演示，有意在本地运行）
for msg in (
    "Using LocalCommandLineCodeExecutor may execute code",
    "No approval function set for CodeExecutorAgent",
    "Using DockerCommandLineCodeExecutor",
):
    warnings.filterwarnings("ignore", message=f"{msg}.*", category=UserWarning)

# ---------------------------------------------------------------------------
# 为 Executor 子进程注入中文字体配置（项目级，不影响全局配置）
# ---------------------------------------------------------------------------
# Coder 生成的 matplotlib 脚本未必会自行设置中文字体（LLM 偶发遗漏）——若依赖默认字体，
# 中文标签会渲染成方框。这里通过 MPLCONFIGDIR 让 Executor 子进程在 import matplotlib 时
# 读取项目内 .mpldir/matplotlibrc，从而保证任何生成脚本都能正确显示中文。
_MPLDIR = Path(__file__).resolve().parent / ".mpldir"
_RCFILE = _MPLDIR / "matplotlibrc"
if _RCFILE.parent.exists() is False or _RCFILE.exists() is False:
    _MPLDIR.mkdir(parents=True, exist_ok=True)
    _RCFILE.write_text(
        "font.family: sans-serif\n"
        "font.sans-serif: Microsoft YaHei, SimHei, Noto Sans CJK SC, DejaVu Sans\n"
        "axes.unicode_minus: False\n",
        encoding="utf-8",
    )
# 在创建 Executor（及其子进程）之前写入环境变量，使子进程继承该配置目录。
os.environ.setdefault("MPLCONFIGDIR", str(_MPLDIR))
# Coder 生成的脚本会用 print 输出中文材料名/单位（如 “钢: 50 W/(m·K)”）。
# Windows 中文系统下子进程 stdout 默认为 GBK，而 AutoGen 的 LocalCommandLineCodeExecutor 拿到字节后
# 按 UTF-8 decode → 报 "UnicodeDecodeError: invalid continuation byte"。强制子进程用 UTF-8 输出即可修复。
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# ---------------------------------------------------------------------------
# 密钥加载（不写死进代码）
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    """轻量 .env 解析器（不引入第三方依赖）。

    语法：``KEY=VALUE``、空行、``#`` 注释、可选 ``export`` 前缀、可选双/单引号。
    已在 ``os.environ`` 中存在的变量**不会被覆盖**（真实环境变量优先于 .env 文件）。
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val


# 允许把密钥放在项目根目录的 .env（已被 .gitignore 忽略），避免明文写进源码。
# 加载后仍以真实环境变量为准（环境变量优先）。
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ---------------------------------------------------------------------------
# 可配置项
# ---------------------------------------------------------------------------
# 模型名：按需求改为 DeepSeek 的 v4-flash-vision（仍可通过 Q3_MODEL 环境变量覆盖）。
MODEL_NAME = os.environ.get("Q3_MODEL", "deepseek-v4-flash-vision-exp")
BASE_URL = os.environ.get("Q3_BASE_URL", "https://api.deepseek.com")

# DeepSeek API key：不再硬编码，从环境变量 / .env 读取（参照 .env.example）。
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Tavily 搜索 API key（方案B：让 Research 联网检索真实材料数据）。去 https://tavily.com 注册领取。
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

if not (API_KEY and TAVILY_API_KEY):
    warnings.warn(
        "未检测到完整的 API key（DEEPSEEK_API_KEY / TAVILY_API_KEY）。\n"
        "请复制项目根目录的 .env.example 为 .env 并填入你的 key（.env 已被 .gitignore 忽略，不会被提交）。\n"
        "缺少 DEEPSEEK_API_KEY 时真实调用会鉴权失败；缺少 TAVILY_API_KEY 时 Research 会回退到记忆/估算值。",
        stacklevel=2,
    )


def create_model_client(**overrides) -> OpenAIChatCompletionClient:
    """创建一个连接到 DeepSeek OpenAI 兼容接口的模型客户端。

    kwargs 可覆盖默认的 ``model`` / ``base_url`` / ``api_key``。
    """
    return OpenAIChatCompletionClient(
        model=overrides.get("model", MODEL_NAME),
        base_url=overrides.get("base_url", BASE_URL),
        api_key=overrides.get("api_key", API_KEY),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "deepseek",
        },
        # 值越小越保守；这里给出较宽松的 token 上限，避免长代码/长结论被截断
        model_info_usage={
            "prompt": 32768,
            "completion": 8192,
            "max": 65536,
        },
        **{k: v for k, v in overrides.items() if k in ("temperature",)},
    )


# 共享模型客户端：RoundRobinGroupChat 为顺序执行，不会并发，可安全复用。
_model_client = None


def get_model_client() -> OpenAIChatCompletionClient:
    """返回全局共享的模型客户端（懒加载单例）。"""
    global _model_client
    if _model_client is None:
        _model_client = create_model_client()
    return _model_client
