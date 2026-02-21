/**
 * LangGraph 实现经典 Plan-and-Execute Agent 流程 (TypeScript)
 *
 * 流程：START → planner → agent → replan → (agent | END)
 */
import { AIMessageChunk, HumanMessage, SystemMessage, ToolMessage } from "@langchain/core/messages";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
import { createAgent, tool } from "langchain";
import { resolve } from "path";
import { z } from "zod";
config({ path: resolve(process.cwd(), ".env") });
config({ path: resolve(process.cwd(), "../.env") });

// =============================================================================
// State 定义
// =============================================================================
const PlanExecuteStateAnnotation = Annotation.Root({
  input: Annotation<string>(),
  plan: Annotation<string[]>(),
  past_steps: Annotation<[string, string][]>({
    reducer: (left, right) => [...left, ...(Array.isArray(right) ? right : [right])],
    default: () => [],
  }),
  response: Annotation<string>(),
});

type PlanExecuteState = typeof PlanExecuteStateAnnotation.State;

// =============================================================================
// 结构化输出 Schema (zod)
// =============================================================================
const PlanSchema = z.object({
  steps: z.array(z.string()).describe("ordered steps to follow"),
});

// const ResponseSchema = z.object({
//   response: z.string(),
// });

const RouteSchema = z.object({
  route: z.enum(["response", "update_plan"]),
});

type Plan = z.infer<typeof PlanSchema>;

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
// Main
// =============================================================================
async function main() {
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
  const planStep = async (state: PlanExecuteState) => {
    const out = await plannerChain.invoke([
      new SystemMessage(`For the given objective, come up with a simple step-by-step plan.
Each step should be a single task. Do not add superfluous steps.
The final step's result should be the answer. Each step must be self-contained.`),
      new HumanMessage(state.input),
    ]);
    const steps = out?.steps;
    if (!Array.isArray(steps) || steps.length === 0) {
      return { plan: [state.input] };
    }
    return { plan: steps };
  };

  // Replanner: 先路由，再分别调用专属 LLM
  const replanStep = async (state: PlanExecuteState) => {
    const pastStr =
      (state.past_steps || [])
        .map(([t, r]) => `- ${t}: ${r}`)
        .join("\n") || "(none)";
    const planStr = (state.plan || []).join(", ");

    // 第一步：路由决定
    const { route } = await model.withStructuredOutput(RouteSchema).invoke([
      new HumanMessage(`For the given objective, create or update the plan.

Objective: ${state.input}

Original plan: ${planStr}

Completed steps:
${pastStr}

Decide: use "response" if you can answer the user now; use "update_plan" if more steps are needed.`),
    ]);

    // 第二步：路由到各自专属 LLM
    if (route === "response") {
      const stream = await model.stream([
        new SystemMessage("You synthesize execution results into a clear, direct answer. Use the full context: user objective, plan, and completed step results."),
        new HumanMessage(`User question: ${state.input}

Original plan: ${planStr}

Completed steps and their results:
${pastStr}

Based on the above context, provide the final answer to the user's question.`),
      ]);
      let full: AIMessageChunk | null = null;
      for await (const chunk of stream) {
        full = full ? full.concat(chunk) : chunk;
        process.stdout.write(chunk.text);
      }
      return { response: full?.text };
    }
    const out = await model.withStructuredOutput(PlanSchema).invoke([
      new SystemMessage("Update the plan. Return only remaining steps (do not repeat completed ones). Each step must be self-contained."),
      new HumanMessage(`Objective: ${state.input}

Original plan: ${planStr}

Completed steps:
${pastStr}

Return remaining steps only.`),
    ]);
    const steps = out?.steps;
    return { plan: Array.isArray(steps) && steps.length > 0 ? steps : [] };
  };

  // Execute step
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
    const taskMsg = `Plan:\n${planStr}\n\nCompleted steps:\n${pastStr}\n\nExecute step ${stepNum}: ${task}`;


    let result: { messages: { content?: string | unknown[] }[] };

    console.log(`\n  [Executor] task: ${task.slice(0, 60)}...`);
    let lastEvt: { messages: unknown[] } | null = null;
    const stream = await executor.stream(
      { messages: [new HumanMessage(taskMsg)] },
      { streamMode: ["values", "messages"] as const }
    );
    for await (const evt of stream) {
      const isTuple = Array.isArray(evt) && evt.length >= 2;

      const [mode, chunk] = evt as [string, unknown];
      if (mode === "messages") {
        const [token, _meta] = chunk as any
        if (token?.content) process.stdout.write(token.content);
        else if (token?.contentBlocks?.length) {
          for (const b of token.contentBlocks as { text?: string }[]) {
            if (b?.text) process.stdout.write(b.text);
          }
        }
      } else if (mode === "values") {
        lastEvt = chunk as { messages: unknown[] };
        const msgs = lastEvt.messages || [];
        const last = msgs[msgs.length - 1];
        console.log("last", last);
        if (last && "tool_calls" in (last as any)) {
          process.stdout.write("\n");
          for (const tc of (last as any).tool_calls) {
            console.log(`  [Executor] tool_call: ${tc.name} args=${JSON.stringify(tc.args)} id=${tc.id}`);
          }
        } else if (last instanceof ToolMessage) {
          const c = last.content;
          const preview = c;
          console.log(`  [Executor] tool_result: ${preview} tool_call_id=${last.tool_call_id}`);
        }
      }
    }
    if (lastEvt) process.stdout.write("\n");
    result = lastEvt as any;

    const lastMsg = result.messages[result.messages.length - 1];
    return { past_steps: [[task, lastMsg.content]] as [string, string][] };
  };

  // Conditional edge
  const shouldContinue = (state: PlanExecuteState): "agent" | "__end__" => {
    return state.response ? "__end__" : "agent";
  };

  // Build graph
  const workflow = new StateGraph(PlanExecuteStateAnnotation)
    .addNode("planner", planStep)
    .addNode("agent", executeStep)
    .addNode("replan", replanStep)
    .addEdge(START, "planner")
    .addEdge("planner", "agent")
    .addEdge("agent", "replan")
    .addConditionalEdges("replan", shouldContinue, {
      agent: "agent",
      __end__: END,
    });

  const app = workflow.compile();

  // Run
  const query = "What is the hometown of the men's 2024 Australia Open winner?";
  console.log("\nQuestion:", query);
  console.log("=".repeat(60));

  const initial: PlanExecuteState = {
    input: query,
    plan: [],
    past_steps: [],
    response: "",
  };

  let i = 0;
  // const stream = await app.stream(initial, { streamMode: "values" });
  // for await (const step of stream) {
  //   i++;
  //   console.log(`\n${"=".repeat(60)}\n>>> Step ${i}\n${"=".repeat(60)}`);
  //   console.log(JSON.stringify(step, null, 2));
  // }

  const stream = await app.stream(initial, { streamMode: "updates" });
  for await (const chunk of stream) {
    for (const [nodeName, stateUpdate] of Object.entries(chunk)) {
      i++;
      console.log(`\n${"=".repeat(60)}\n>>> Node: ${nodeName}\n${"=".repeat(60)}`);
      console.log(JSON.stringify(stateUpdate, null, 2));
    }
  }
}

main().catch(console.error);
