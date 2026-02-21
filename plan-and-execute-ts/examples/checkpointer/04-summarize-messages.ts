/**
 * 04 - Summarize Messages: 用 summarizationMiddleware 压缩历史
 *
 * 知识点：
 * - 当 token/message 超过 trigger 时，用另一个 model 总结历史
 * - 用总结替换旧消息，保留最近 keep 条
 * - 适合长对话，避免超出 context window
 */
import { createAgent, summarizationMiddleware } from "langchain";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
import { resolve } from "path";

config({ path: resolve(process.cwd(), ".env") });

async function main() {
  const model = new ChatOpenAI({
    modelName: process.env.OPENAI_MODEL || "gpt-4o-mini",
    openAIApiKey: process.env.OPENAI_API_KEY,
    configuration: process.env.OPENAI_BASE_URL ? { baseURL: process.env.OPENAI_BASE_URL } : undefined,
  });

  const checkpointer = new MemorySaver();
  const agent = createAgent({
    model,
    tools: [],
    middleware: [
      summarizationMiddleware({
        model,
        trigger: { messages: 4 },
        keep: { messages: 6 },
      }),
    ],
    checkpointer,
  });

  const configurable = { configurable: { thread_id: "thread-4" } };

  await agent.invoke({ messages: [{ role: "user", content: "hi, my name is bob" }] }, configurable);
  await agent.invoke({ messages: [{ role: "user", content: "write a short poem about cats" }] }, configurable);
  await agent.invoke({ messages: [{ role: "user", content: "now do the same but for dogs" }] }, configurable);
  const final = await agent.invoke({ messages: [{ role: "user", content: "what's my name?" }] }, configurable);

  console.log("Final answer:", final.messages.at(-1)?.content);
  // 预期: "Your name is Bob!" 或类似
}

main().catch(console.error);
