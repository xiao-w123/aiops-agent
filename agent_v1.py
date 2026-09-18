"""agent_v1：基础 Agent（无 RAG 检索）。

保留本文件的目的是作为"不使用历史故障知识库"的对照组。
真正的工具定义与 Agent 构造逻辑已统一收敛到 agent_core.py。

用法：python agent_v1.py
"""

from agent_core import (
    build_llm,
    query_prometheus_tool,
    search_logs_tool,
    get_deployment_record_tool,
)

from langchain.agents import create_agent

QUESTION = "帮我看看 order-service 昨晚为什么响应变慢了"

# 刻意不含 search_fault_reports，用于对比"有无历史故障检索"的差异
tools = [query_prometheus_tool, search_logs_tool, get_deployment_record_tool]

if __name__ == "__main__":
    agent = create_agent(build_llm(), tools)
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    print(result["messages"][-1].content)
