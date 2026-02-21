/**
 * Human-in-the-Loop: interrupt + Command + approve/reject 模式
 *
 * ========== 核心概念 ==========
 *
 * 1. interrupt(value)
 *    - 在节点内调用，立即暂停图执行
 *    - value 会出现在 result.__interrupt__ 里，供 UI 展示（如弹窗）
 *    - 必须配合 checkpointer，状态会持久化，等待 resume
 *
 * 2. Command({ resume: x })
 *    - 作为 invoke 的输入传入，用于恢复被 interrupt 的图
 *    - x 会变成 interrupt() 的返回值，节点从 interrupt 之后继续执行
 *    - 注意：resume 时节点会从头重跑，但 interrupt 会直接返回 x，不会再次暂停
 *
 * 3. Command({ goto: "nodeName" })
 *    - 节点返回 Command 时可指定下一个节点，替代 addConditionalEdges
 *    - 需在 addNode 时用 ends: ["nodeA", "nodeB"] 声明合法目标
 *
 * 流程：START → approval (interrupt) → [proceed | cancel] → END
 */
import {
  Annotation,
  Command,
  END,
  interrupt,
  INTERRUPT,
  isInterrupted,
  MemorySaver,
  START,
  StateGraph,
} from "@langchain/langgraph";

// =============================================================================
// State 定义（用 Annotation 避免 Zod/StateSchema 兼容性问题）
// =============================================================================
const State = Annotation.Root({
  actionDetails: Annotation<string>(),
  status: Annotation<"pending" | "approved" | "rejected" | null>(),
});

// =============================================================================
// approval 节点：调用 interrupt 暂停，等待人工决策
// =============================================================================
const approvalNode = async (state: { actionDetails: string }) => {
  // interrupt 会：
  // 1. 暂停图执行
  // 2. 把 { question, details } 放到 result.__interrupt__ 里返回给调用方
  // 3. 保存 checkpoint，等待 resume
  //
  // 当调用方 invoke(Command({ resume: true/false })) 时，
  // decision 会收到 true 或 false，然后继续执行下面的 return
  const decision = interrupt({
    question: "Approve this action?",
    details: state.actionDetails,
  });

  // decision 就是 Command({ resume: x }) 里的 x
  // { approved: true } → proceed，{ approved: false } → cancel
  const approved = (decision as { approved?: boolean })?.approved ?? false;
  return new Command({ goto: approved ? "proceed" : "cancel" });
};

// =============================================================================
// 构建图
// =============================================================================
const graphBuilder = new StateGraph(State)
  .addNode("approval", approvalNode, {
    // ends 声明该节点可能 goto 到的目标，用于图校验
    ends: ["proceed", "cancel"],
  })
  .addNode("proceed", () => ({ status: "approved" }))
  .addNode("cancel", () => ({ status: "rejected" }))
  .addEdge(START, "approval")
  .addEdge("proceed", END)
  .addEdge("cancel", END);

const checkpointer = new MemorySaver();
const graph = graphBuilder.compile({ checkpointer });

// =============================================================================
// 运行示例
// =============================================================================
async function main() {
  const config = { configurable: { thread_id: "approval-123" } };

  // ---------- 第一次 invoke：触发 interrupt，图暂停 ----------
  console.log(">>> 第一次 invoke：传入初始 state，触发 approval 节点");
  const initial = await graph.invoke(
    { actionDetails: "Transfer $500", status: "pending" },
    config
  );

  console.log("initial:", JSON.stringify(initial, null, 2));

  // 检查是否被 interrupt
  if (isInterrupted(initial)) {
    const payload = initial[INTERRUPT][0];
    const v = payload?.value as { question?: string; details?: string } | undefined;
    console.log("\n[Interrupt 触发] 图已暂停，等待人工决策：");
    console.log("  question:", v?.question);
    console.log("  details:", v?.details);
    console.log("\n此时应渲染 UI，让用户点击 Approve 或 Reject");
  }

  // ---------- 第二次 invoke：用 Command({ resume }) 恢复 ----------
  console.log("\n>>> 第二次 invoke：传入 Command({ resume: { approved: true } })，模拟用户点击 Approve");
  const resumed = await graph.invoke(
    new Command({ resume: { approved: true } }),
    config
  );
  console.log("resumed.status:", resumed.status); // "approved"

  console.log("resumed:", JSON.stringify(resumed, null, 2));

  // ---------- 再跑一遍，这次 Reject ----------
  console.log("\n>>> 第三次 invoke：新 thread，再次触发 interrupt");
  const config2 = { configurable: { thread_id: "approval-456" } };
  await graph.invoke(
    { actionDetails: "Delete user data", status: "pending" },
    config2
  );


  console.log(">>> 第四次 invoke：Command({ resume: { approved: false } })，模拟用户点击 Reject");
  const rejected = await graph.invoke(
    new Command({ resume: { approved: false } }),
    config2
  );
  console.log("rejected.status:", rejected.status); // "rejected"
}

main().catch(console.error);
