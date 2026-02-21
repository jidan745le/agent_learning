/**
 * Plan-and-Execute Agent 带 checkpointer + thread + messages 压缩 (TypeScript)
 *
 * 流程：START → (summarize | planner) → agent → replan → (agent | END)
 * - checkpointer 持久化 state，同一 thread_id 下保持对话历史
 * - messages 超过阈值时 summarize 压缩历史，保留最近 keep 条
 */
import {
  AIMessage,
  AIMessageChunk,
  BaseMessage,
  HumanMessage,
  RemoveMessage,
  SystemMessage,
  getBufferString,
  isHumanMessage,
  isAIMessage,
} from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import {
  Annotation,
  END,
  REMOVE_ALL_MESSAGES,
  START,
  StateGraph,
  messagesStateReducer,
} from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
import { createAgent, tool } from "langchain";
import { resolve } from "path";
import { z } from "zod";
config({ path: resolve(process.cwd(), ".env") });
config({ path: resolve(process.cwd(), "../.env") });

// =============================================================================
// State 定义（含 history messages）
// =============================================================================
const PlanExecuteStateAnnotation = Annotation.Root({
  input: Annotation<string>(),
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  plan: Annotation<string[]>(),
  past_steps: Annotation<[string, string][]>({
    reducer: (left, right) =>
      Array.isArray(right) && right.length === 0
        ? []
        : [...(left ?? []), ...(Array.isArray(right) ? right : right != null ? [right] : [])],
    default: () => [],
  }),
  response: Annotation<string>(),
});

type PlanExecuteState = typeof PlanExecuteStateAnnotation.State;

// =============================================================================
// 结构化输出 Schema
// =============================================================================
const PlanSchema = z.object({
  steps: z.array(z.string()).describe("ordered steps to follow"),
});

const RouteSchema = z.object({
  route: z.enum(["response", "update_plan"]),
});

// =============================================================================
// 工具与 Store
// =============================================================================
const store: Record<string, string> = {
  "australia open 2024": "Jannik Sinner won the men's singles.",
  "Jannik Sinner hometown": "Sexten, northern Italy.",
  "task decomposition": "Breaking complex goals into smaller sub-tasks.",
};

const searchTool = tool(
  async ({ query }: { query: string }) => {
    const q = query.toLowerCase();
    let hits = Object.entries(store).filter(
      ([k, v]) => k.toLowerCase().includes(q) || v.toLowerCase().includes(q)
    );
    if (hits.length === 0) {
      const words = q.split(/\s+/).filter((w) => w.length > 2);
      hits = Object.entries(store).filter(([k, v]) =>
        words.some((w) => k.toLowerCase().includes(w) || v.toLowerCase().includes(w))
      );
    }
    return hits.length > 0 ? hits.map(([, v]) => v).join("\n") : "No results found.";
  },
  {
    name: "search",
    description: "Search for information. Use for factual queries.",
    schema: z.object({ query: z.string() }),
  }
);

// =============================================================================
// 上下文格式化（仅保留 Human/AI 对话消息作为上下文）
// =============================================================================
function formatHistoryContext(messages: BaseMessage[]): string {
  if (!messages?.length) return "(none)";
  const chatOnly = messages.filter((m) => isHumanMessage(m) || isAIMessage(m));
  if (chatOnly.length === 0) return "(none)";
  return getBufferString(chatOnly, "User", "Assistant");
}

// =============================================================================
// 构建带历史的 Plan-and-Execute Agent
// =============================================================================
export function createPlanExecuteWithHistory() {
  const baseConfig = {
    modelName: process.env.OPENAI_MODEL || "qwen-plus",
    openAIApiKey: process.env.OPENAI_API_KEY,
    configuration: process.env.OPENAI_BASE_URL
      ? { baseURL: process.env.OPENAI_BASE_URL }
      : undefined,
  };
  const model = new ChatOpenAI({ ...baseConfig });
  const executor = createAgent({
    model,
    tools: [searchTool],
    systemPrompt: "You are a helpful assistant.",
  });

  const plannerChain = model.withStructuredOutput(PlanSchema);

  const SUMMARIZE_THRESHOLD = 10;
  const SUMMARIZE_KEEP = 3;

  const summarizeStep = async (state: PlanExecuteState) => {
    const msgs = state.messages ?? [];
    if (msgs.length <= SUMMARIZE_THRESHOLD) return {};

    const toSummarize = msgs.slice(0, -SUMMARIZE_KEEP);
    const summary = await model.invoke([
      new SystemMessage("Summarize conversation history concisely, keeping key facts."),
      ...toSummarize,
    ]);
    const summaryContent = typeof summary.content === "string" ? summary.content : String(summary.content ?? "");
    return {
      messages: [
        new RemoveMessage({ id: REMOVE_ALL_MESSAGES }),
        new AIMessage(`Summary of prior conversation: ${summaryContent}`),
        ...msgs.slice(-SUMMARIZE_KEEP),
      ],
    };
  };

  const planStep = async (state: PlanExecuteState) => {
    const historyStr = formatHistoryContext(state.messages || []);
    const messages: BaseMessage[] = [
      new SystemMessage(`For the given objective, come up with a simple step-by-step plan.
Each step should be a single task. Do not add superfluous steps.
The final step's result should be the answer. Each step must be self-contained.
If there is conversation history, consider it when planning.`),
      new HumanMessage(
        historyStr !== "(none)"
          ? `[Conversation history]\n${historyStr}\n\n[Current objective]\n${state.input}`
          : state.input
      ),
    ];
    const out = await plannerChain.invoke(messages);
    const steps = out?.steps;
    if (!Array.isArray(steps) || steps.length === 0) {
      return { plan: [state.input] };
    }
    return { plan: steps };
  };

  const replanStep = async (state: PlanExecuteState) => {
    const pastStr =
      (state.past_steps || [])
        .map(([t, r]) => `- ${t}: ${r}`)
        .join("\n") || "(none)";
    const planStr = (state.plan || []).join(", ");
    const historyStr = formatHistoryContext(state.messages || []);

    const routePrompt = `For the given objective, create or update the plan.

Objective: ${state.input}

${historyStr !== "(none)" ? `Conversation history:\n${historyStr}\n\n` : ""}Original plan: ${planStr}

Completed steps:
${pastStr}

Decide: use "response" if you can answer the user now; use "update_plan" if more steps are needed.`;

    const { route } = await model.withStructuredOutput(RouteSchema).invoke([
      new HumanMessage(routePrompt),
    ]);

    if (route === "response") {
      const stream = await model.stream([
        new SystemMessage(
          "You synthesize execution results into a clear, direct answer. Use the full context: user objective, conversation history (if any), plan, and completed step results."
        ),
        new HumanMessage(`User question: ${state.input}

${historyStr !== "(none)" ? `Conversation history:\n${historyStr}\n\n` : ""}Original plan: ${planStr}

Completed steps and their results:
${pastStr}

Based on the above context, provide the final answer to the user's question.`),
      ]);
      let full: AIMessageChunk | null = null;
      for await (const chunk of stream) {
        full = full ? full.concat(chunk) : chunk;
        process.stdout.write(chunk.text);
      }
      const responseText = full?.text ?? "";
      return { response: responseText, messages: [new AIMessage(responseText)] };
    }

    const out = await model.withStructuredOutput(PlanSchema).invoke([
      new SystemMessage(
        "Update the plan. Return only remaining steps (do not repeat completed ones). Each step must be self-contained."
      ),
      new HumanMessage(`Objective: ${state.input}

${historyStr !== "(none)" ? `Conversation history:\n${historyStr}\n\n` : ""}Original plan: ${planStr}

Completed steps:
${pastStr}

Return remaining steps only.`),
    ]);
    const steps = out?.steps;
    return { plan: Array.isArray(steps) && steps.length > 0 ? steps : [] };
  };

  const executeStep = async (state: PlanExecuteState) => {
    const plan = state.plan || [];
    if (plan.length === 0) {
      return { past_steps: [] as [string, string][], response: "No plan generated." };
    }
    const task = plan[0];
    const pastSteps = state.past_steps || [];
    const stepNum = pastSteps.length + 1;
    const pastStr =
      pastSteps.map(([t, r]) => `- ${t}: ${r}`).join("\n") || "(none)";
    const planStr = plan.map((s, i) => `${i + 1}. ${s}`).join("\n");
    const historyStr = formatHistoryContext(state.messages || []);
    const taskMsg =
      historyStr !== "(none)"
        ? `[Conversation history]\n${historyStr}\n\n[Plan]\n${planStr}\n\n[Completed steps]\n${pastStr}\n\nExecute step ${stepNum}: ${task}`
        : `Plan:\n${planStr}\n\nCompleted steps:\n${pastStr}\n\nExecute step ${stepNum}: ${task}`;

    let lastEvt: { messages: unknown[] } | null = null;
    const stream = await executor.stream(
      { messages: [new HumanMessage(taskMsg)] },
      { streamMode: ["values", "messages"] as const }
    );
    for await (const evt of stream) {
      const [mode, chunk] = evt as [string, unknown];
      if (mode === "messages") {
        const [token] = chunk as [unknown, unknown];
        const t = token as { content?: string; contentBlocks?: { text?: string }[] };
        if (t?.content) process.stdout.write(t.content);
        else if (t?.contentBlocks?.length) {
          for (const b of t.contentBlocks) {
            if (b?.text) process.stdout.write(b.text);
          }
        }
      } else if (mode === "values") {
        lastEvt = chunk as { messages: unknown[] };
      }
    }
    if (lastEvt) process.stdout.write("\n");

    const result = lastEvt as { messages: { content?: string | unknown[] }[] };
    const lastMsg = result.messages[result.messages.length - 1];
    const content = typeof lastMsg.content === "string" ? lastMsg.content : String(lastMsg.content ?? "");
    return { past_steps: [[task, content]] as [string, string][] };
  };

  const shouldContinue = (state: PlanExecuteState): "agent" | "__end__" => {
    return state.response ? "__end__" : "agent";
  };

  const routeToSummarizeOrPlanner = (state: PlanExecuteState) =>
    (state.messages?.length ?? 0) > SUMMARIZE_THRESHOLD ? "summarize" : "planner";

  const checkpointer = new MemorySaver();
  const workflow = new StateGraph(PlanExecuteStateAnnotation)
    .addNode("summarize", summarizeStep)
    .addNode("planner", planStep)
    .addNode("agent", executeStep)
    .addNode("replan", replanStep)
    .addConditionalEdges(START, routeToSummarizeOrPlanner, { summarize: "summarize", planner: "planner" })
    .addEdge("summarize", "planner")
    .addEdge("planner", "agent")
    .addEdge("agent", "replan")
    .addConditionalEdges("replan", shouldContinue, {
      agent: "agent",
      __end__: END,
    });

  return workflow.compile({ checkpointer });
}

// =============================================================================
// 对话循环示例（checkpointer + thread_id）
// =============================================================================
async function main() {
  const app = createPlanExecuteWithHistory();
  const threadId = "plan-execute-thread-1";
  const config = { configurable: { thread_id: threadId } };

  const turns = [
    "Who won the men's 2024 Australia Open?",
    "What is his hometown?",
  ];

  for (let i = 0; i < turns.length; i++) {
    const user = turns[i];
    console.log(`\n${"=".repeat(60)}`);
    console.log(`\n>>> Turn ${i + 1}: ${user}`);
    console.log("=".repeat(60));

    const result = await app.invoke(
      { input: user, messages: [new HumanMessage(user)], plan: [], past_steps: [], response: "" },
      config
    );

    console.log(`\n[Response] ${result.response}`);
  }
}

main().catch(console.error);
