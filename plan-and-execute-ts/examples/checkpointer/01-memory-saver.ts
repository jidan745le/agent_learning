/**
 * 01 - MemorySaver: 基础 checkpointer，thread 级持久化
 *
 * 知识点：
 * - checkpointer 让 agent 在多次 invoke 间保持对话历史
 * - thread_id 区分不同会话
 * - 同一 thread_id 下，agent 能记住之前的消息
 */
import { createAgent } from "langchain";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
import { resolve } from "path";

config({ path: resolve(process.cwd(), ".env") });

async function main() {
  const checkpointer = new MemorySaver();
  const model = new ChatOpenAI({
    modelName: process.env.OPENAI_MODEL || "gpt-4o-mini",
    openAIApiKey: process.env.OPENAI_API_KEY,
    configuration: process.env.OPENAI_BASE_URL ? { baseURL: process.env.OPENAI_BASE_URL } : undefined,
  });

  const agent = createAgent({
    model,
    tools: [],
    checkpointer,
  });

  const configurable = { configurable: { thread_id: "thread-1" } };

  const printHistory = (label: string, messages: { getType?: () => string; content?: unknown }[]) => {
    console.log(`\n[${label}] messages 数量: ${messages.length}`);
    messages.forEach((m, i) => {
      const type = (m as { getType?: () => string }).getType?.() ?? "?";
      const content = typeof m.content === "string" ? m.content : String(m.content ?? "");
      const preview = content.length > 80 ? content.slice(0, 80) + "..." : content;
      console.log(`  ${i + 1}. [${type}] ${preview}`);
    });
  };

  console.log("--- 第一次对话 ---");
  const r1 = await agent.invoke(
    { messages: [{ role: "user", content: "hi! i am Bob" }] },
    configurable
  );
  printHistory("第一次 invoke 后", r1.messages);

  console.log("\n--- 第二次对话（同一 thread，agent 记得 Bob）---");
  const result = await agent.invoke(
    { messages: [{ role: "user", content: "what's my name?" }] },
    configurable
  );
  printHistory("第二次 invoke 后", result.messages);

  const lastMsg = result.messages.at(-1);
  console.log("\nAI 最终回复:", lastMsg?.content);
}

main().catch(console.error);
