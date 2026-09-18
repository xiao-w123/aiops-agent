"""agent_v2：完整 Agent（含 RAG 历史故障检索）。

真正的实现已收敛到 agent_core.py，本文件仅作演示入口。

用法：python agent_v2.py
"""

from agent_core import build_basic_agent

QUESTION = "帮我看看 order-service 昨晚为什么响应变慢了"

if __name__ == "__main__":
    agent = build_basic_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    print(result["messages"][-1].content)
