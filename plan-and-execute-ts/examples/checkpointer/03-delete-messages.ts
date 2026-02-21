/**
 * 03 - Delete Messages: 用 RemoveMessage 删除历史消息
 *
 * 知识点：
 * - afterModel 在 model 输出后执行
 * - 当消息数 > 2 时，删除最早两条
 * - 用 RemoveMessage({ id: msg.id }) 删除指定消息
 */
import { RemoveMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
import { createAgent, createMiddleware } from "langchain";
import { resolve } from "path";

config({ path: resolve(process.cwd(), ".env") });

async function main() {
  const deleteOldMessages = createMiddleware({
    name: "DeleteOldMessages",
    afterModel: (state) => {
      const messages = state.messages;
      if (messages.length > 2) {
        return {
          messages: messages
            .slice(0, 2)
            .map((m) => new RemoveMessage({ id: m.id ?? "" })),
        };
      }
      return undefined;
    },
  });

  const checkpointer = new MemorySaver();
  const model = new ChatOpenAI({
    modelName: process.env.OPENAI_MODEL || "gpt-4o-mini",
    openAIApiKey: process.env.OPENAI_API_KEY,
    configuration: process.env.OPENAI_BASE_URL ? { baseURL: process.env.OPENAI_BASE_URL } : undefined,
  });

  const agent = createAgent({
    model,
    tools: [],
    systemPrompt: "Please be concise.",
    middleware: [deleteOldMessages],
    checkpointer,
  });

  const configurable = { configurable: { thread_id: "thread-3" } };

  console.log("--- 第一次: hi I'm bob ---");
  const stream1 = await agent.stream(
    { messages: [{ role: "user", content: "hi! I'm bob" }] },
    { ...configurable, streamMode: "values" }
  );
  for await (const event of stream1) {
    console.log("messages count:", event.messages.length);
  }

  console.log("\n--- 第二次: what's my name? ---");
  const stream2 = await agent.stream(
    { messages: [{ role: "user", content: "what's my name?" }] },
    { ...configurable, streamMode: "values" }
  );
  for await (const event of stream2) {
    const details = event.messages.map((m) => [m._getType(), (m as { content?: string }).content]);
    console.log(details);
  }
}

main().catch(console.error);
