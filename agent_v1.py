from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from tools import query_prometheus, search_logs, get_deployment_record

@tool
def query_prometheus_tool(metric_name: str, service: str) -> str:
    """查询监控指标。metric_name 可以是 cpu_usage, memory_usage, response_time。"""
    return query_prometheus(metric_name, service)

@tool
def search_logs_tool(keyword: str, service: str) -> str:
    """搜索服务日志。keyword 是搜索关键词，如 error, timeout, exception。"""
    return search_logs(keyword, service)

@tool
def get_deployment_record_tool(service: str) -> str:
    """查询服务的发布记录。"""
    return get_deployment_record(service)

llm = ChatOpenAI(model="deepseek-chat")
tools = [query_prometheus_tool, search_logs_tool, get_deployment_record_tool]

agent = create_agent(llm, tools)

result = agent.invoke({
    "messages": [{"role": "user", "content": "帮我看看 order-service 昨晚为什么响应变慢了"}]
})
print(result["messages"][-1].content)