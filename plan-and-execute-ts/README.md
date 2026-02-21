# Plan-and-Execute Agent (TypeScript)

TypeScript 复刻版 Plan-and-Execute Agent，基于 LangGraph。

## 运行

```bash
# 安装依赖
npm install

# 复制 .env.example 为 .env 并填写 API Key
cp .env.example .env

# 运行
npm start
```

## 环境变量

- `OPENAI_API_KEY` - 必填
- `OPENAI_BASE_URL` - 可选，兼容 Qwen 等
- `OPENAI_MODEL` - 默认 qwen-plus
