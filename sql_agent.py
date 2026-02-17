# 从 python-dotenv 包导入 load_dotenv 函数，用于从 .env 文件加载环境变量
from dotenv import load_dotenv
# 调用 load_dotenv()：读取当前目录下的 .env 文件，把其中的 KEY=value 注入到 os.environ
# 这样后续 os.environ.get("OPENAI_API_KEY") 等就能读到值，无需在终端手动 set
load_dotenv()

# 标准库：os 用于读取环境变量（如 os.environ.get）
import os
# 标准库：pathlib 用于跨平台路径操作，Path 对象比字符串路径更易用
import pathlib
# 第三方库：requests 用于发起 HTTP GET 请求，下载 Chinook.db
import requests

# LangChain：init_chat_model 根据模型名/环境变量自动选择 provider（OpenAI、Qwen 等）
from langchain.chat_models import init_chat_model
# LangChain：create_agent 创建基于 ReAct 的 Agent，可调用 tools 并流式输出
from langchain.agents import create_agent
# LangChain Community：SQLDatabase 封装 SQL 数据库连接，提供 run()、get_usable_table_names() 等
from langchain_community.utilities import SQLDatabase
# LangChain Community：SQLDatabaseToolkit 提供 sql_db_list_tables、sql_db_schema、sql_db_query 等工具
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import BaseMessage, message_to_dict, messages_to_dict
import json
import logging
from datetime import datetime

# 日志文件：按日期命名，如 sql_agent_2025-02-17.log
LOG_FILE = pathlib.Path(__file__).parent / f"sql_agent_{datetime.now().strftime('%Y-%m-%d')}.log"


def setup_logging():
    """配置日志：同时输出到控制台和文件，文件记录完整 step。"""
    logger = logging.getLogger("sql_agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台：INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件：DEBUG，记录完整内容
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def main():
    log = setup_logging()
    log.info(f"Log file: {LOG_FILE}")

    # ========== 1) 初始化大模型 ==========
    # Qwen 等 OpenAI 兼容接口：模型名如 qwen3-235b-a22b 无法被 LangChain 自动推断 provider
    # 需显式加 "openai:" 前缀，让 LangChain 用 OpenAI 客户端（会读 OPENAI_BASE_URL、OPENAI_API_KEY）
    model_name = os.environ.get("OPENAI_MODEL") or "qwen-plus"
    model = init_chat_model("openai:" + model_name)

    # ========== 2) 下载示例数据库 Chinook.db ==========
    # 示例数据库 URL，Chinook 是数字音乐商店的示例 schema（表：Artist、Track、Genre 等）
    url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
    # Path("Chinook.db")：创建表示当前目录下 Chinook.db 的路径对象
    local_path = pathlib.Path("Chinook.db")
    # .exists()：判断文件是否已存在，避免重复下载
    if local_path.exists():
        msg = f"{local_path} already exists, skipping download."
        log.info(msg)
    else:
        # requests.get(url)：发起 HTTP GET 请求，返回 Response 对象
        response = requests.get(url)
        # status_code == 200：表示请求成功
        if response.status_code == 200:
            # response.content：响应体二进制内容（bytes）
            # .write_bytes()：将 bytes 写入文件，Path 对象的方法
            local_path.write_bytes(response.content)
            log.info(f"File downloaded and saved as {local_path}")
        else:
            # 下载失败时打印状态码并提前退出函数
            log.error(f"Failed to download the file. Status code: {response.status_code}")
            return

    # ========== 3) 连接数据库并创建工具集 ==========
    # SQLDatabase.from_uri("sqlite:///Chinook.db")：用 SQLAlchemy 风格的 URI 连接 SQLite
    # 三个斜杠表示相对路径，会连接当前目录下的 Chinook.db
    db = SQLDatabase.from_uri("sqlite:///Chinook.db")
    # SQLDatabaseToolkit：把 db 和 llm 绑定，生成可被 Agent 调用的工具（list_tables、schema、query 等）
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    # get_tools()：返回工具列表，每个工具是 LangChain 的 BaseTool 子类实例
    tools = toolkit.get_tools()

    # ========== 4) 构建系统提示词并创建 Agent ==========
    # 三引号字符串 """..."""：多行字符串，保留换行
    # {dialect}、{top_k}：占位符，由 .format(dialect=..., top_k=...) 填充
    # db.dialect：数据库方言，如 "sqlite"，用于生成正确语法的 SQL
    # .strip()：去掉字符串首尾空白（包括换行）
    system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(dialect=db.dialect, top_k=5).strip()

    # create_agent：创建 LangGraph Agent
    # model=model：大模型，负责理解问题、决定调用哪个工具、生成最终回答
    # tools=tools：工具列表，Agent 可调用 sql_db_list_tables、sql_db_schema、sql_db_query_checker、sql_db_query
    # system_prompt=system_prompt：系统提示，约束 Agent 行为（如先查表、再查 schema、禁止 DML）
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
    )

    # ========== 5) 运行一次问答 ==========
    # 示例问题：哪种音乐类型的平均曲目时长最长？
    question = "Which genre on average has the longest tracks?"
    log.info(f"Question: {question}")
    log.debug("=" * 80)

    # agent.stream(...)：流式执行 Agent
    # stream_mode="debug"：输出最详细信息，含 task（节点输入）、task_result（节点输出）、checkpoint（状态快照）
    # debug 模式下 yield 的格式为 (namespace, "debug", {"step", "timestamp", "type", "payload"})
    def _serialize_for_log(obj):
        """将可能含 LangChain 消息的对象转为可 JSON 序列化的 dict。"""
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {
                k: messages_to_dict(v) if k == "messages" else _serialize_for_log(v)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return [_serialize_for_log(x) for x in obj]
        if isinstance(obj, BaseMessage):
            return message_to_dict(obj)
        return obj

    for i, chunk in enumerate(agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="debug",
    )):
        # debug 模式 chunk 格式：{step, timestamp, type, payload} 或 (ns, mode, data)
        if isinstance(chunk, tuple):
            data = chunk[-1] if len(chunk) >= 2 else chunk
        else:
            data = chunk
        step_num = data.get("step", i + 1)
        event_type = data.get("type", "unknown")
        payload = data.get("payload", data)

        serialized = _serialize_for_log({"step": step_num, "type": event_type, "payload": payload})
        log.debug(f"--- Debug Event {i + 1} ({event_type}) ---\n{json.dumps(serialized, indent=2, ensure_ascii=False)}")

        # 简要信息：task=节点即将执行，task_result=节点执行完毕，checkpoint=状态快照
        if event_type == "task_result" and isinstance(payload, dict) and "name" in payload:
            log.info(f"Step {step_num}: {payload['name']} completed")
        elif event_type == "task" and isinstance(payload, dict) and "name" in payload:
            log.info(f"Step {step_num}: {payload['name']} started")


# __name__ == "__main__"：仅当本文件被直接运行（python sql_agent.py）时为 True
# 若被 import 则 __name__ 为模块名，不会执行 main()
# 这样既可作为脚本运行，也可被其他模块导入而不自动执行
if __name__ == "__main__":
    main()
