# Checkpointer 示例

基于 [Short-term memory 文档](https://docs.langchain.com/oss/javascript/langchain/short-term-memory) 的可执行示例。

## 运行

从项目根目录：

```bash
npx tsx examples/checkpointer/01-memory-saver.ts
npx tsx examples/checkpointer/02-custom-state.ts
npx tsx examples/checkpointer/03-delete-messages.ts
npx tsx examples/checkpointer/04-summarize-messages.ts
npx tsx examples/checkpointer/05-dynamic-prompt.ts
```

## 示例说明

| 示例 | 知识点 |
|------|--------|
| **01-memory-saver** | MemorySaver + thread_id：同一 thread 下多次 invoke 保持对话历史 |
| **02-custom-state** | createMiddleware + stateSchema：扩展 agent state（userId, preferences） |
| **03-delete-messages** | RemoveMessage + afterModel：删除历史消息，控制 context 长度 |
| **04-summarize-messages** | summarizationMiddleware：超阈值时用 LLM 总结历史并替换 |
| **05-dynamic-prompt** | dynamicSystemPromptMiddleware：根据 context 动态设置 system prompt |

## 环境

需配置 `.env`：`OPENAI_API_KEY`、`OPENAI_BASE_URL`（可选）、`OPENAI_MODEL`（可选）。
