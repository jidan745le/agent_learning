"""
LangGraph 实现 SQL Agent：显式定义 StateGraph、节点、边，等价于 create_agent 的 tool-call loop。
"""
# 从 .env 加载环境变量（OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL 等）
from dotenv import load_dotenv
load_dotenv()

import os
import pathlib
import requests

# Literal：类型注解，表示返回值只能是 "tools" 或 "__end__" 之一
from typing import Literal

# init_chat_model：根据模型名创建 Chat 模型，支持 OpenAI / Qwen 等
from langchain.chat_models import init_chat_model
# SQLDatabase：封装数据库连接，提供 run()、get_usable_table_names() 等
from langchain_community.utilities import SQLDatabase
# SQLDatabaseToolkit：提供 sql_db_list_tables、sql_db_schema、sql_db_query 等工具
from langchain_community.agent_toolkits import SQLDatabaseToolkit
# HumanMessage：用户消息；SystemMessage：系统提示
from langchain_core.messages import HumanMessage, SystemMessage
# END：图结束节点；START：图入口；StateGraph：状态图构建器；MessagesState：消息列表状态
from langgraph.graph import END, START, StateGraph, MessagesState  # type: ignore[import-untyped]
# ToolNode：预置节点，自动执行 AIMessage 中的 tool_calls 并返回 ToolMessage
from langgraph.prebuilt import ToolNode  # type: ignore[import-untyped]


def main():
    # ========== 1) 初始化模型、数据库、工具 ==========
    # 从环境变量读模型名，空则默认 qwen-plus；openai: 前缀表示用 OpenAI 客户端（兼容 Qwen）
    model_name = os.environ.get("OPENAI_MODEL") or "qwen-plus"
    model = init_chat_model("openai:" + model_name)

    # 下载 Chinook 示例数据库（若不存在）
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    local_path = pathlib.Path("Chinook.db")
    if not local_path.exists():
        resp = requests.get(url)
        if resp.status_code == 200:
            local_path.write_bytes(resp.content)
            print(f"Downloaded {local_path}")
        else:
            print(f"Download failed: {resp.status_code}")
            return

    # 连接 SQLite，创建 toolkit 并获取工具列表
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    # ========== 2) 定义节点 ==========

    # agent_node：Agent 节点函数，接收 state，返回要合并进 state 的更新
    # state: MessagesState 即 {"messages": [...]}，类型注解帮助 IDE 补全
    def agent_node(state: MessagesState):
        # 系统提示：约束 Agent 行为（先查表、再查 schema、禁止 DML 等）
        system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples, always limit your query to at most {top_k} results.

Never query for all columns from a table, only ask for relevant columns.
You MUST double check your query before executing it.
DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.).

To start you should ALWAYS look at the tables in the database. Do NOT skip this step.
Then query the schema of the most relevant tables.
""".format(dialect=db.dialect, top_k=5).strip()

        # SystemMessage：系统角色消息，放在对话最前，指导模型行为
        system_msg = SystemMessage(content=system_prompt)
        # bind_tools(tools)：将工具 schema 绑定到模型，使模型能输出 tool_calls（结构化工具调用）
        # 模型会根据上下文决定：返回普通文本回答，或返回 tool_calls 让 ToolNode 执行
        llm_with_tools = model.bind_tools(tools)
        # invoke：同步调用模型；[system_msg] + list(state["messages"]) 即完整对话历史
        # 返回 AIMessage，可能含 content（文本）或 tool_calls（工具调用列表）
        print([system_msg] + list(state["messages"]))
        response = llm_with_tools.invoke([system_msg] + list(state["messages"]))
        # 返回 {"messages": [response]}：MessagesState 的 reducer add_messages 会将其追加到 state
        return {"messages": [response]}

    # ToolNode(tools)：预置节点，接收 state，取出最后一条 AIMessage 的 tool_calls，
    # 按 name 找到对应 tool 并 invoke，将结果封装为 ToolMessage 追加到 state
    tool_node = ToolNode(tools)

    # ========== 3) 条件边：根据 state 决定下一节点 ==========
    # should_continue：路由函数，返回值作为 path_map 的 key，决定走哪条边
    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        # hasattr(last, "tool_calls")：兼容非 AIMessage 类型；last.tool_calls：模型是否请求了工具调用
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"   # 有 tool_calls → 去 tools 节点执行
        return "__end__"     # 无 tool_calls → 模型已给出最终回答，结束图

    # ========== 4) 组装图 ==========
    # StateGraph(MessagesState)：创建图，状态类型为 MessagesState（含 messages 键）
    builder = StateGraph(MessagesState)
    # add_node(name, node)：添加节点；name 用于边的引用，node 为可调用对象（函数或 callable）
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    # add_edge(from, to)：固定边，from 执行完后无条件进入 to
    # START：虚拟入口节点，图启动时首先触发
    builder.add_edge(START, "agent")
    # add_conditional_edges(source, path_fn, path_map)：条件边
    # source 执行完后，调用 path_fn(state) 得到 key，用 path_map[key] 决定下一节点
    # "tools" → "tools" 节点；"__end__" → END 表示图结束
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    # tools 执行完后回到 agent，形成 agent → tools → agent 的 ReAct 循环
    builder.add_edge("tools", "agent")

    # compile()：将图编译为可执行的 CompiledGraph，可 stream/invoke
    graph = builder.compile()

    # ========== 4.5) 可选：查看图结构 ==========
    # 生态支持：LangGraph 内置 get_graph().draw_mermaid() / print_ascii() / draw_mermaid_png()
    # draw_mermaid()：输出 Mermaid 代码，粘贴到 https://mermaid.live 可渲染为流程图
    # print_ascii()：终端 ASCII 图（需 pip install grandalf）
    # draw_mermaid_png()：输出 PNG 图片（需 pip install pyppeteer）
    print("\n--- Mermaid（复制到 mermaid.live 查看）---")
    print(graph.get_graph().draw_mermaid())

    # ========== 5) 运行 ==========
    question = "Which genre on average has the longest tracks?"
    print("\nQuestion:", question)
    print("=" * 80)

    # graph.stream(input, stream_mode="values")：流式执行，每完成一个节点 yield 当前完整 state
    # input 格式须符合 MessagesState，即 {"messages": [HumanMessage(...)]}
    for step in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="values",
    ):
        # step["messages"][-1]：最新一条消息（AIMessage 或 ToolMessage）
        # pretty_print()：格式化打印，便于查看 tool_calls、content 等
        step["messages"][-1].pretty_print()


# 直接运行本文件时执行 main；被 import 时不执行
if __name__ == "__main__":
    main()
