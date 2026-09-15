import time
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tools import query_prometheus, search_logs, get_deployment_record

# ---------- 初始化 Agent（和 agent_v2.py 一致） ----------
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
tools = [query_prometheus_tool, search_logs_tool,
         get_deployment_record_tool, search_fault_reports]
agent = create_agent(llm, tools)

# ---------- 测试集 ----------
test_cases = [
    {
        "question": "帮我看看 order-service 昨晚为什么响应变慢了",
        "expected_tools": ["query_prometheus_tool", "search_logs_tool", "get_deployment_record_tool"],
        "expected_keywords": ["连接池", "CPU", "发布"]
    },
    {
        "question": "product-service 凌晨超时是什么原因",
        "expected_tools": ["search_logs_tool"],
        "expected_keywords": ["缓存", "穿透"]
    },
    {
        "question": "user-service 内存一直涨是怎么回事",
        "expected_tools": ["query_prometheus_tool"],
        "expected_keywords": ["内存", "泄漏"]
    },
    {
        "question": "历史上有没有类似的连接池故障",
        "expected_tools": ["search_fault_reports"],
        "expected_keywords": ["连接池", "故障报告"]
    },
    {
        "question": "order-service 昨晚发过版吗",
        "expected_tools": ["get_deployment_record_tool"],
        "expected_keywords": ["发布", "v2.3.1"]
    },
]

# ---------- 跑测试 ----------
results = []
for i, case in enumerate(test_cases, 1):
    print(f"\n[{i}/{len(test_cases)}] {case['question']}")

    start = time.time()
    result = agent.invoke({
        "messages": [{"role": "user", "content": case["question"]}]
    })
    elapsed = time.time() - start

    # 提取调用了哪些工具
    called_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                called_tools.append(tc["name"])

    final_answer = result["messages"][-1].content

    # 判断工具调用是否命中
    tool_hits = sum(1 for t in case["expected_tools"] if t in called_tools)
    tool_score = tool_hits / len(case["expected_tools"])

    # 判断关键词是否命中
    kw_hits = sum(1 for kw in case["expected_keywords"] if kw in final_answer)
    kw_score = kw_hits / len(case["expected_keywords"])

    print(f"  调用工具: {called_tools}")
    print(f"  期望工具: {case['expected_tools']}  命中率: {tool_score:.0%}")
    print(f"  期望关键词: {case['expected_keywords']}  命中率: {kw_score:.0%}")
    print(f"  耗时: {elapsed:.1f}s")

    results.append({
        "tool_score": tool_score,
        "kw_score": kw_score,
        "elapsed": elapsed,
    })

# ---------- 汇总 ----------
print("\n" + "=" * 50)
print("评测汇总")
print("=" * 50)
avg_tool = sum(r["tool_score"] for r in results) / len(results)
avg_kw = sum(r["kw_score"] for r in results) / len(results)
avg_time = sum(r["elapsed"] for r in results) / len(results)

print(f"工具调用平均准确率: {avg_tool:.0%}")
print(f"关键词平均命中率: {avg_kw:.0%}")
print(f"平均耗时: {avg_time:.1f}s")
print(f"测试用例数: {len(results)}")