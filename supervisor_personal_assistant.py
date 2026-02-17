"""
Supervisor 模式个人助理：多 Agent 架构示例

=============================================================================
架构说明（三层结构）
=============================================================================
- 顶层：Supervisor Agent — 理解用户请求，路由到 schedule_event 或 manage_email
- 中层：Calendar Agent、Email Agent — 各自专注一个领域，将自然语言转为 API 调用
- 底层：API 工具 — create_calendar_event、send_email、get_available_time_slots（stub）

核心思想：将子 Agent 包装成工具（Agent-as-Tool），Supervisor 只看到高层能力，
无需理解 ISO 日期格式、邮箱格式等细节，从而简化路由、降低 prompt 复杂度。

=============================================================================
关键语法与概念
=============================================================================
1. @tool 装饰器
   - 将普通函数转为 LangChain Tool，模型可根据 docstring + 参数类型生成 JSON schema
   - 模型输出 tool_calls 时，LangGraph 的 ToolNode 会按 name 匹配并 invoke 对应工具

2. create_agent(model, tools, system_prompt)
   - 创建带 ReAct 循环的 Agent（内部为 StateGraph：agent → tools → agent → ...）
   - 模型可输出：纯文本回答 或 tool_calls（结构化工具调用）

3. agent.invoke(input) / agent.stream(input)
   - invoke：同步执行到底，返回最终 state
   - stream：流式 yield 每步 state，便于实时展示

4. 子 Agent 包装为工具
   - 定义 @tool def schedule_event(request: str)：内部 invoke 子 Agent，返回其最终文本
   - Supervisor 调用 schedule_event 时，实际执行的是整个子 Agent 的 ReAct 循环
"""
from dotenv import load_dotenv
load_dotenv()

import os

# =============================================================================
# 导入说明
# =============================================================================
# tool：装饰器，将普通函数转为 LangChain 工具，模型可根据 docstring 自动生成调用 schema
from langchain.tools import tool
# create_agent：创建带工具调用的 Agent（内部为 LangGraph ReAct 循环）
from langchain.agents import create_agent
# init_chat_model：根据模型名/环境变量创建 Chat 模型（支持 OpenAI、Qwen、Anthropic 等）
from langchain.chat_models import init_chat_model


# =============================================================================
# 第一步：定义底层 API 工具（本示例为 stub，实际应调用 Google Calendar、SendGrid 等）
# =============================================================================

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO 格式: "2024-01-15T14:00:00"
    end_time: str,          # ISO 格式: "2024-01-15T15:00:00"
    attendees: list[str],   # 邮箱列表
    location: str = ""
) -> str:
    """
    创建日历事件。需要严格的 ISO 日期时间格式。
    LangChain 会根据此 docstring 和参数类型，为模型生成工具 schema。
    """
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],      # 收件人邮箱
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """
    通过邮件 API 发送邮件。需要正确格式的邮箱地址。
    """
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,              # ISO 格式: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """
    检查指定日期、指定参与者的日历可用时段。
    """
    return ["09:00", "14:00", "16:00"]


# =============================================================================
# 第二步：创建 specialized 子 Agent
# =============================================================================
# 每个子 Agent 拥有：1) 领域专属的 tools  2) 领域专属的 system_prompt
# 子 Agent 负责：将自然语言（如「下周二下午 2 点」）解析为结构化 API 调用

CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)


# =============================================================================
# 第三步：将子 Agent 包装成 Supervisor 可调用的工具
# =============================================================================
# 关键设计：Supervisor 只看到 schedule_event、manage_email 两个高层工具，
# 而不是 create_calendar_event、send_email 等底层 API。
# 工具描述（docstring）帮助 Supervisor 决定何时调用哪个工具。

def _build_agents(model):
    """构建 calendar_agent、email_agent、supervisor_agent。需先有 model。"""
    calendar_agent = create_agent(
        model,
        tools=[create_calendar_event, get_available_time_slots],
        system_prompt=CALENDAR_AGENT_PROMPT,
    )

    email_agent = create_agent(
        model,
        tools=[send_email],
        system_prompt=EMAIL_AGENT_PROMPT,
    )

    # 包装子 Agent 为工具：接收自然语言请求，invoke 子 Agent，返回其最终文本回复
    @tool
    def schedule_event(request: str) -> str:
        """Schedule calendar events using natural language.

        Use this when the user wants to create, modify, or check calendar appointments.
        Handles date/time parsing, availability checking, and event creation.

        Input: Natural language scheduling request (e.g., 'meeting with design team
        next Tuesday at 2pm')
        """
        result = calendar_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        # 只返回子 Agent 最后一条消息的文本，Supervisor 不需要看到中间 tool_calls
        last_msg = result["messages"][-1]
        return getattr(last_msg, "content", "") or getattr(last_msg, "text", "") or ""

    @tool
    def manage_email(request: str) -> str:
        """Send emails using natural language.

        Use this when the user wants to send notifications, reminders, or any email
        communication. Handles recipient extraction, subject generation, and email
        composition.

        Input: Natural language email request (e.g., 'send them a reminder about
        the meeting')
        """
        result = email_agent.invoke({
            "messages": [{"role": "user", "content": request}]
        })
        last_msg = result["messages"][-1]
        return getattr(last_msg, "content", "") or getattr(last_msg, "text", "") or ""

    SUPERVISOR_PROMPT = (
        "You are a helpful personal assistant. "
        "You can schedule calendar events and send emails. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
    )

    supervisor_agent = create_agent(
        model,
        tools=[schedule_event, manage_email],
        system_prompt=SUPERVISOR_PROMPT,
    )

    return calendar_agent, email_agent, supervisor_agent


# =============================================================================
# 主流程：初始化、运行示例
# =============================================================================

def main():
    # 初始化模型：openai: 前缀表示使用 OpenAI 兼容 API（可接 Qwen、Azure 等）
    model_name = os.environ.get("OPENAI_MODEL") or "qwen-plus"
    model = init_chat_model("openai:" + model_name)

    calendar_agent, email_agent, supervisor_agent = _build_agents(model)

    # 示例 1：单领域请求（仅日历）
    print("\n" + "=" * 80)
    print("示例 1：单领域请求 - 安排团队站会")
    print("=" * 80)
    query1 = "Schedule a team standup for tomorrow at 9am"
    # stream(input, stream_mode="values")：流式执行，每完成一个节点 yield 当前完整 state
    # input 须为 MessagesState 格式：{"messages": [HumanMessage 或 {"role":"user","content":...}]}
    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query1}]},
        stream_mode="values",
    ):
        # step["messages"][-1]：最新一条消息（AIMessage 含 tool_calls，或 ToolMessage，或最终 AIMessage）
        step["messages"][-1].pretty_print()

    # 示例 2：多领域请求（日历 + 邮件）
    print("\n" + "=" * 80)
    print("示例 2：多领域请求 - 安排会议并发送邮件提醒")
    print("=" * 80)
    query2 = (
        "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
    )
    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": query2}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
