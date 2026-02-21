"""
LangGraph 实现经典 Plan-and-Execute Agent 流程。

=============================================================================
相关论文参考
=============================================================================

1. Plan-and-Solve Prompting (Wang et al., ACL 2023)
   - 标题：Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models
   - arXiv: https://arxiv.org/abs/2305.04091
   - 核心：先规划（拆分为子任务）→ 再逐步求解，缓解 Zero-shot-CoT 的计算/漏步/语义错误

2. ReAct: Synergizing Reasoning and Acting (Yao et al., ICLR 2023)
   - 标题：ReAct: Synergizing Reasoning and Acting in Language Models
   - arXiv: https://arxiv.org/abs/2210.03629
   - 对比：ReAct 是「边想边做」单步迭代；Plan-and-Execute 是「先规划再执行」多步显式规划

3. Baby-AGI (Nakajima, 2023)
   - GitHub: https://github.com/yoheinakajima/babyagi
   - 启发：任务分解 + 执行 + 优先级排序的自主 Agent 循环

4. Plan-and-Act (ICML 2025)
   - arXiv: https://arxiv.org/abs/2503.09572
   - 扩展：Planner 与 Executor 分离，针对长时程任务优化

5. LangGraph 官方教程
   - https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/

=============================================================================
流程说明
=============================================================================
  START → planner（生成计划）→ agent（执行第 1 步）→ replan（根据结果决定：继续执行 or 返回答案）
       ↑                                                                    |
       |                                                                    ↓
       +------------------------ agent ←─────────────────────────────── replan
                                 |                                        |
                                 ↓                                        ↓
                           执行下一步                               response → END

与 ReAct 区别：先整体规划，再逐步执行；可用小模型做执行、大模型做规划。
"""
from dotenv import load_dotenv
load_dotenv()

import os
import warnings

# 抑制 with_structured_output 触发的 Pydantic 序列化警告（LangChain 内部序列化与 Pydantic 类型推断不一致）
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*", category=UserWarning)
import operator
from typing import Annotated, Literal

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# =============================================================================
# State 定义
# =============================================================================
class PlanExecuteState(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[list[tuple[str, str]], operator.add]
    response: str


# =============================================================================
# 结构化输出（Planning / Replanning）
# =============================================================================
class Plan(BaseModel):
    """规划结果：有序步骤列表"""

    steps: list[str] = Field(description="ordered steps to follow")


class Response(BaseModel):
    """最终回复"""

    response: str


class Act(BaseModel):
    """Replan 输出：要么返回答案，要么更新计划"""

    action: Response | Plan = Field(
        description="If done, use Response. If more steps needed, use Plan."
    )


# =============================================================================
# 工具与 Executor
# =============================================================================
_store: dict[str, str] = {
    "australia open 2024": "Jannik Sinner won the men's singles.",
    "Jannik Sinner hometown": "Sexten, northern Italy.",
    "task decomposition": "Breaking complex goals into smaller sub-tasks.",
}


@tool
def search(query: str) -> str:
    """Search for information. Use for factual queries."""
    q = query.lower()
    # 全文匹配
    hits = [v for k, v in _store.items() if q in k.lower() or q in v.lower()]
    if not hits:
        # 关键词匹配：任意关键词命中即可（排查时发现 executor 常传长句，全文匹配易失败）
        words = [w for w in q.split() if len(w) > 2]
        hits = [v for k, v in _store.items() if any(w in k.lower() or w in v.lower() for w in words)]
    return "\n".join(hits) if hits else "No results found."


def main():
    model_name = os.environ.get("OPENAI_MODEL") or "qwen-plus"
    llm = init_chat_model("openai:" + model_name)
    tools = [search]

    executor = create_agent(llm, tools, system_prompt="You are a helpful assistant.")

    # Planner
    planner_prompt = ChatPromptTemplate.from_messages([
        ("system", """For the given objective, come up with a simple step-by-step plan.
Each step should be a single task. Do not add superfluous steps.
The final step's result should be the answer. Each step must be self-contained."""),
        ("placeholder", "{messages}"),
    ])
    planner = planner_prompt | llm.with_structured_output(Plan)

    # Replanner
    replanner_prompt = ChatPromptTemplate.from_template("""For the given objective, create or update the plan.

Objective: {input}

Original plan: {plan}

Completed steps:
{past_steps}

Update the plan. If you can answer the user now, use Response. Otherwise use Plan with remaining steps only.""")
    replanner = replanner_prompt | llm.with_structured_output(Act)

    # =============================================================================
    # 节点函数
    # =============================================================================
    def plan_step(state: PlanExecuteState) -> dict:
        out = planner.invoke({"messages": [HumanMessage(content=state["input"])]})
        return {"plan": out.steps}

    def execute_step(state: PlanExecuteState) -> dict:
        plan = state["plan"]
        if not plan:
            return {"past_steps": [], "response": "No plan generated."}
        task = plan[0]
        plan_str = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        past_steps = state.get("past_steps") or []
        step_num = len(past_steps) + 1
        past_str = "\n".join(f"- {t}: {r}" for t, r in past_steps) if past_steps else "(none)"
        task_msg = f"Plan:\n{plan_str}\n\nCompleted steps:\n{past_str}\n\nExecute step {step_num}: {task}"
        # 是否打印 executor 内部执行（工具调用、结果），设置 DEBUG_EXECUTOR=1 启用
        debug_executor = os.environ.get("DEBUG_EXECUTOR", "").lower() in ("1", "true", "yes")
        if debug_executor:
            print(f"\n  [Executor] task: {task[:60]}...")
            last_evt = None
            for evt in executor.stream(
                {"messages": [{"role": "user", "content": task_msg}]},
                stream_mode=["values", "messages"],
            ):
                if isinstance(evt, tuple) and len(evt) >= 2:
                    mode, chunk = evt[0], evt[1]
                    if mode == "messages":
                        msg, _ = chunk
                        if getattr(msg, "content", None):
                            print(msg.content, end="", flush=True)
                    elif mode == "values":
                        last_evt = chunk
                        for m in chunk.get("messages", [])[-1:]:
                            if getattr(m, "tool_calls", None):
                                print()
                                for tc in m.tool_calls:
                                    name = getattr(tc, "name", tc.get("name", ""))
                                    args = getattr(tc, "args", tc.get("args", ""))
                                    print(f"  [Executor] tool_call: {name} args={args}")
                            elif isinstance(m, ToolMessage):
                                preview = (m.content or "")[:120] + "..." if len(m.content or "") > 120 else (m.content or "")
                                print(f"  [Executor] tool_result: {preview}")
                else:
                    last_evt = evt
                    for m in (evt or {}).get("messages", [])[-1:]:
                        if getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                name = getattr(tc, "name", tc.get("name", ""))
                                args = getattr(tc, "args", tc.get("args", ""))
                                print(f"  [Executor] tool_call: {name} args={args}")
                        elif isinstance(m, ToolMessage):
                            preview = (m.content or "")[:120] + "..." if len(m.content or "") > 120 else (m.content or "")
                            print(f"  [Executor] tool_result: {preview}")
                        elif getattr(m, "content", None):
                            print(f"  [Executor] final_content: {(m.content or '')[:200]}...")
            if last_evt:
                print()
            result = last_evt or executor.invoke({"messages": [{"role": "user", "content": task_msg}]})
        else:
            result = executor.invoke({"messages": [{"role": "user", "content": task_msg}]})
        last_msg = result["messages"][-1]
        content = getattr(last_msg, "content", "") or getattr(last_msg, "text", "") or ""
        return {"past_steps": [(task, content)]}

    def replan_step(state: PlanExecuteState) -> dict:
        past_str = "\n".join(f"- {t}: {r}" for t, r in (state.get("past_steps") or []))
        out = replanner.invoke({
            "input": state["input"],
            "plan": state.get("plan") or [],
            "past_steps": past_str or "(none)",
        })
        if isinstance(out.action, Response):
            return {"response": out.action.response}
        return {"plan": out.action.steps}

    def should_continue(state: PlanExecuteState) -> Literal["agent", "__end__"]:
        if state.get("response"):
            return "__end__"
        return "agent"

    # =============================================================================
    # 组装图
    # =============================================================================
    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges("replan", should_continue, {"agent": "agent", "__end__": END})

    app = workflow.compile()

    # 输出流程图
    print("\n--- 流程图（Mermaid，复制到 mermaid.live 查看）---")
    print(app.get_graph().draw_mermaid())

    # 运行
    query = "What is the hometown of the men's 2024 Australia Open winner?"
    print("\nQuestion:", query)
    print("=" * 60)

    initial = {"input": query, "plan": [], "past_steps": [], "response": ""}
    for i, step in enumerate(app.stream(initial, stream_mode="values")):
        print(f"\n{'='*60}\n>>> Step {i+1}\n{'='*60}")
        print(step)


if __name__ == "__main__":
    main()
