"""
原生 tool_call_loop Agent：不依赖 LangChain/LangGraph，纯 Python 实现 ReAct 循环。

流程：user → model → [有 tool_calls?] → 执行工具 → 追加结果 → 回到 model → 直到无 tool_calls → 返回
"""

# =============================================================================
# 导入模块
# =============================================================================
# from 模块 import 函数/类：从某模块中导入指定内容，导入后可直接用 load_dotenv() 而不写 dotenv.load_dotenv()
from dotenv import load_dotenv
# load_dotenv()：读取当前目录的 .env 文件，把其中的 KEY=value 注入到环境变量 os.environ
load_dotenv()

# import 模块：导入整个模块，使用时需加前缀，如 os.environ、json.loads
import os   # 操作系统相关，如环境变量 os.environ.get()
import json # 处理 JSON 字符串与 Python 对象的转换，如 json.loads()
# from typing import ...：类型注解用，仅用于提示类型，不参与运行
#   Any：任意类型，表示「可以是任何东西」，用于放宽约束
#   Callable：可调用对象，如函数、lambda，表示「可以加括号调用的东西」
from typing import Any, Callable

# 使用 openai 包（pip install openai），兼容 Qwen 等 OpenAI 接口
from openai import OpenAI


# =============================================================================
# 类型注解说明（Python 3.9+ 内置泛型）
# =============================================================================
# 参数后加 : 类型 = 值：表示该参数期望的类型
# 函数后加 -> 类型：表示返回值类型
#
# 常用写法：
#   str          - 字符串
#   int          - 整数
#   dict[K, V]   - 字典，键类型 K，值类型 V（如 dict[str, int] 表示键为 str、值为 int）
#   list[T]      - 列表，元素类型 T（如 list[str] 表示字符串列表）
#   tuple[A, B]  - 元组，第 1 个元素类型 A，第 2 个类型 B
#   Callable     - 可调用对象（函数）
#   Any          - 任意类型
#
# 类型注解不参与运行，主要用于 IDE 补全、静态检查（如 mypy），写错类型也不会报错
# =============================================================================


def build_tools_schema(tools: dict[str, tuple[Callable, dict]]) -> list[dict]:
    """
    将工具字典转为 OpenAI API 所需的 tools 格式。
    参数 tools: {工具名: (可调用函数, 描述字典)}
    返回: 符合 OpenAI 规范的列表
    """
    # 类型拆解：tools: dict[str, tuple[Callable, dict]]
    #   dict[str, ...]     - 字典，键是 str（工具名），值是 ...
    #   tuple[Callable, dict] - 值是一个元组：(函数, 描述字典)
    #   Callable           - 第一个元素是可调用的函数
    #   dict               - 第二个元素是普通字典（schema）
    # 返回 list[dict]      - 元素为 dict 的列表
    # result = []：创建空列表，[] 表示列表字面量
    result = []
    # for 变量 in 可迭代对象：遍历；.items() 返回 (键, 值) 对
    # (_, schema)：元组解包，_ 表示忽略第一个值（函数），只取 schema
    for name, (_, schema) in tools.items():
        # result.append(x)：在列表末尾追加元素 x
        result.append({
            # 字典：{键: 值}，键通常是字符串
            "type": "function",
            "function": {
                "name": name,
                # .get(键, 默认值)：取字典中的键，若不存在则返回默认值
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
            },
        })
    # return 返回值：函数结束并返回
    return result


def execute_tool(tools: dict[str, tuple[Callable, dict]], name: str, arguments: str) -> str:
    """
    按工具名查找并执行工具，返回字符串结果。
    参数 arguments：JSON 格式的参数字符串，如 '{"query": "xxx"}'
    """
    # 类型：name: str - 字符串；arguments: str - 字符串；-> str - 返回字符串
    # if 条件: ... 条件为真时执行
    # name not in tools：name 不在 tools 的键中
    if name not in tools:
        # f"..."：f-string，花括号 {} 内可写变量或表达式，会替换为实际值
        return f"Unknown tool: {name}"
    # fn, _ = tools[name]：从字典取值并解包为两个变量，_ 表示忽略第二个（schema）
    fn, _ = tools[name]
    # try: ... except 异常 as 变量: ... 捕获异常，避免程序崩溃
    try:
        # isinstance(变量, 类型)：判断变量是否属于某类型，返回 True/False
        # 条件 if 真 else 假：三元表达式，条件为真取 if 前，否则取 else 后
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        # json.loads(字符串)：把 JSON 字符串解析为 Python 字典
        # fn(**args)：** 表示把字典展开为关键字参数，如 fn(query="xxx")
        out = fn(**args)
        return str(out) if not isinstance(out, str) else out
    # except 异常类型 as 变量：捕获该类型异常，变量 e 保存异常对象
    except Exception as e:
        return f"Tool error: {e}"


def tool_call_loop(
    client: OpenAI,                              # OpenAI 客户端实例（类名作类型 = 该类的对象）
    model: str,                                  # 模型名，如 "qwen-plus"
    system_prompt: str,                          # 系统提示词
    user_message: str,                            # 用户输入
    tools: dict[str, tuple[Callable, dict]],     # 工具字典，同 build_tools_schema
    max_iterations: int = 10,                     # 最大循环次数，默认 10；调用时可省略
) -> str:
    """
    主循环：调用模型 → 若有 tool_calls 则执行并追加结果 → 重复直到模型返回纯文本。
    """
    # messages: list[dict[str, Any]]
    #   list[...]     - 列表
    #   dict[str, Any]- 列表元素是字典，键为 str（如 "role","content"），值为 Any（任意）
    #   Any 用于放宽约束：content 可能是 str，tool_calls 可能是 list 等
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    tools_schema = build_tools_schema(tools)

    # for _ in range(n)：循环 n 次，_ 表示循环变量用不到
    for _ in range(max_iterations):
        # kwargs: dict[str, Any] - 传给 API 的关键字参数字典，键如 "model"/"messages"/"tools"，值为任意类型
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        # if 条件: 条件为真时执行（空列表 [] 为 False，非空为 True）
        if tools_schema:
            kwargs["tools"] = tools_schema

        # client.chat.completions.create(...)：调用 OpenAI 聊天补全 API
        # **kwargs：把字典展开为关键字参数，如 create(model="x", messages=[...])
        resp = client.chat.completions.create(**kwargs)
        # resp.choices[0]：取响应中 choices 列表的第一个元素；.message 取其中的 message 属性
        msg = resp.choices[0].message

        # if not x：若 x 为假（None、空列表、空字符串等）则执行
        if not msg.tool_calls:
            # (msg.content or "")：若 content 为 None 则用 ""；.strip() 去掉首尾空白
            return (msg.content or "").strip()

        # messages.append({...})：在 messages 列表末尾追加一条消息
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            # 列表推导式：[表达式 for 变量 in 可迭代对象]，生成新列表
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        # for 变量 in 可迭代对象：遍历 msg.tool_calls 中的每个 tool_call
        for tc in msg.tool_calls:
            name = tc.function.name
            args_str = tc.function.arguments or "{}"
            result = execute_tool(tools, name, args_str)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # 若循环结束仍未 return，说明超过最大迭代次数，返回空字符串
    return ""


def main():
    """主函数：初始化客户端、定义工具、运行一次问答。"""
    # OpenAI(...)：创建客户端实例；关键字参数 名=值 的形式传参
    # os.environ.get("KEY", 默认值)：从环境变量取 KEY，不存在则返回默认值
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL") or None,  # or：前者为假时取后者
    )
    model = os.environ.get("OPENAI_MODEL") or "qwen-plus"

    # 示例：简单检索工具（内存存储）
    # store: dict[str, str] - 键为 str（主题名），值也为 str（内容），如 {"task_decomposition": "..."}
    store: dict[str, str] = {
        "task_decomposition": "Task decomposition breaks complex goals into smaller sub-tasks.",
        "reflection": "Reflection allows the agent to refine outputs through self-critique.",
    }

    # def 函数名(参数: 类型) -> 返回类型: 定义函数
    # query: str - 参数 query 是字符串；-> str - 返回值是字符串
    def retrieve(query: str) -> str:
        """Retrieve information to help answer a query."""
        # 列表推导式：[v for k, v in store.items() if 条件]
        # 遍历 store 的 (k,v)，满足条件时把 v 放入列表
        # query.lower()：转小写；in：子串/成员判断
        hits = [v for k, v in store.items() if query.lower() in k.lower() or query.lower() in v.lower()]
        return "\n\n".join(hits) if hits else "No relevant content found."
        # "sep".join(列表)：用 sep 连接列表元素为字符串

    # 工具定义：{名称: (函数, 描述字典)}
    tools = {
        "retrieve": (
            retrieve,
            {
                "description": "Retrieve information to help answer a query.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            },
        ),
    }

    system_prompt = "You have access to a retrieve tool. Use it to help answer user queries."
    query = "What is task decomposition?"
    print("Question:", query)
    print("=" * 60)  # 字符串 * 数字 = 重复该字符串 n 次

    result = tool_call_loop(client, model, system_prompt, query, tools)
    print("Answer:", result)


# __name__ 是 Python 的内置变量：直接运行本文件时为 "__main__"，被 import 时为模块名
# 这样写可以：直接运行则执行 main()，被导入时不会自动执行
if __name__ == "__main__":
    main()
