"""
LangChain RAG Agent：基于 Lilian Weng 博客的问答应用。
索引：WebBaseLoader 加载 → RecursiveCharacterTextSplitter 分块 → InMemoryVectorStore 存储
检索与生成：create_agent + retrieve_context 工具
"""
from dotenv import load_dotenv
load_dotenv()

import bs4
import os

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import Field
from langchain.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import ToolMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding


def main():
    # ========== 1) 初始化模型与 Embeddings ==========
    model_name = os.environ.get("OPENAI_MODEL") or "qwen-plus"
    model = init_chat_model("openai:" + model_name)

    # Fake Embeddings 无需 API；检索质量有限。可替换为 OpenAIEmbeddings(openai_api_base=os.environ.get("OPENAI_BASE_URL"))
    embeddings = DeterministicFakeEmbedding(size=1536)

    # ========== 2) 索引：加载、分块、存储 ==========
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=all_splits)
    print(f"Indexed {len(all_splits)} chunks.")

    # ========== 3) 定义检索工具并创建 Agent ==========
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str = Field(description="The user's question or search phrase to find relevant context from the indexed documents")):
        """Retrieve information to help answer a query."""
        print(f"Retrieving context for query: {query}")
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_context]
    prompt = (
        "You have access to a tool that retrieves context from a blog post. "
        "Use the tool to help answer user queries."
    )                                                                                       
    agent = create_agent(model, tools, system_prompt=prompt)

    # ========== 4) 运行问答（流式呈现） ==========
    query = "What is task decomposition?"
    print("\nQuestion:", query)
    print("=" * 80)

    async def run_stream():
        async for event in agent.astream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode=["messages", "updates"],
        ):
            if isinstance(event, tuple):
                mode, chunk = event
                if mode == "messages":
                    msg, _ = chunk
                    if msg.content:
                        print(msg.content, end="", flush=True)
                elif mode == "updates":
                    for update in chunk.values():
                        if isinstance(update, dict) and "messages" in update:
                            for m in update["messages"]:
                                if getattr(m, "tool_calls", None):
                                    for tc in m.tool_calls:
                                        name = tc["name"] if isinstance(tc, dict) else getattr(tc, "name", "")
                                        args = (tc.get("args") or (tc.get("function") or {}).get("arguments", "")) if isinstance(tc, dict) else getattr(tc, "args", "")
                                        print(f"\n调用工具: {name}  arguments: {args}")
                                elif isinstance(m, ToolMessage):
                                    content = m.content or ""
                                    preview = content[:200] + "..." if len(content) > 200 else content
                                    print("\n工具结果:", preview)

    asyncio.run(run_stream())
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
