/**
 * 05 - Dynamic System Prompt: 根据 context 动态设置 system prompt
 *
 * 知识点：
 * - dynamicSystemPromptMiddleware 在每次 model 调用前设置 system prompt
 * - 可依赖 state 和 runtime.context
 * - 适合个性化（如用户名、区域等）
 */
import { createAgent, dynamicSystemPromptMiddleware, tool } from "langchain";
import { ChatOpenAI } from "@langchain/openai";
import { config } from "dotenv";
import { resolve } from "path";
import { z } from "zod";

config({ path: resolve(process.cwd(), ".env") });

async function main() {
  const contextSchema = z.object({
    userName: z.string(),
  });

  const getWeather = tool(
    async ({ city }: { city: string }) => `The weather in ${city} is sunny!`,
    {
      name: "get_weather",
      description: "Get weather for a city",
      schema: z.object({ city: z.string() }),
    }
  );

  const model = new ChatOpenAI({
    modelName: process.env.OPENAI_MODEL || "gpt-4o-mini",
    openAIApiKey: process.env.OPENAI_API_KEY,
    configuration: process.env.OPENAI_BASE_URL ? { baseURL: process.env.OPENAI_BASE_URL } : undefined,
  });

  const agent = createAgent({
    model,
    tools: [getWeather],
    contextSchema,
    middleware: [
      dynamicSystemPromptMiddleware<z.infer<typeof contextSchema>>((_, runtime) => {
        return `You are a helpful assistant. Address the user as ${runtime.context?.userName ?? "there"}.`;
      }),
    ],
  });

  const result = await agent.invoke(
    { messages: [{ role: "user", content: "What is the weather in SF?" }] },
    { context: { userName: "John Smith" } }
  );

  const last = result.messages.at(-1);
  console.log("AI:", last?.content);
  // 预期包含 "John Smith" 和天气信息
}

main().catch(console.error);
