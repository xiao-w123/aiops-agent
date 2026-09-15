from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tools import query_prometheus, search_logs, get_deployment_record

# 加载向量库
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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

@tool
def search_fault_reports(query: str) -> str:
    """搜索历史故障报告，了解类似故障的原因和解决方案。当需要参考历史经验时使用。"""
    results = retriever.invoke(query)
    return "\n---\n".join([r.page_content for r in results])

llm = ChatOpenAI(model="deepseek-chat")
tools = [
    query_prometheus_tool,
    search_logs_tool,
    get_deployment_record_tool,
    search_fault_reports,
]

agent = create_agent(llm, tools)

result = agent.invoke({
    "messages": [{"role": "user", "content": "帮我看看 order-service 昨晚为什么响应变慢了"}]
})
print(result["messages"][-1].content)