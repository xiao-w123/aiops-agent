"""FastAPI 后端：提供 Web 界面与多轮对话接口。"""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent_core import build_basic_agent, collect_called_tools

app = FastAPI(title="智能运维 Agent")

# 启动时构建一次 Agent 并复用（构建开销主要在 embedding 模型加载上）
agent = build_basic_agent()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    """对话请求。

    message 为本次用户输入；messages 为之前的对话历史（可选）。
    传入历史即可支持多轮追问，如先说"CPU 使用率是多少"再问"继续，定位根因"。
    """
    message: str
    messages: list[ChatMessage] = []


@app.post("/chat")
async def chat(req: ChatRequest):
    # 组装完整对话：历史 + 本轮输入
    history = [{"role": m.role, "content": m.content} for m in req.messages]
    history.append({"role": "user", "content": req.message})

    result = agent.invoke({"messages": history})

    called = collect_called_tools(result["messages"])
    return {
        "reply": result["messages"][-1].content,
        "tools": called,
        "tool_count": len(called),
    }


@app.get("/")
async def index():
    return HTMLResponse(content=open("index.html", encoding="utf-8").read())


@app.get("/health")
async def health():
    """供演示前快速确认服务与向量库是否就绪。"""
    return {"status": "ok", "vectorstore": os.path.isdir("./chroma_db")}
