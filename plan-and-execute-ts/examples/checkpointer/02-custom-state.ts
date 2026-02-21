/**
 * 02 - Custom State: 扩展 agent state
 *
 * 知识点：
 * - 用 createMiddleware + stateSchema 扩展 state
 * - invoke 时传入自定义字段（userId, preferences）
 * - middleware 和 tools 可访问这些 state
 */
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
import { createAgent, createMiddleware } from "langchain";
import { resolve } from "path";
import { z } from "zod";

config({ path: resolve(process.cwd(), ".env") });

async function main() {
  const CustomState = z.object({
    userId: z.string(),
    preferences: z.record(z.string(), z.any()),
  });

  const stateExtensionMiddleware = createMiddleware({
    name: "StateExtension",
    stateSchema: CustomState,
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
    middleware: [stateExtensionMiddleware],
    checkpointer,
  });

  const result = await agent.invoke(
    {
      messages: [{ role: "user", content: "Hello" }],
      userId: "user_123",
      preferences: { theme: "dark" },
    },
    { configurable: { thread_id: "thread-2" } }
  );
  console.log("result", result);
  console.log("Last message:", result.messages.at(-1)?.content);
}

main().catch(console.error);
