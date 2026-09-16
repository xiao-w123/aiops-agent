import json
import time
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tools import query_prometheus, search_logs, get_deployment_record

# ---------- 初始化 Agent ----------
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

# ---------- 读取测试集 ----------
testset = json.load(open("testset.json", encoding="utf-8"))
print(f"加载了 {len(testset)} 条测试用例\n")

# ---------- 跑测试 ----------
results = []
level_stats = defaultdict(list)

for i, case in enumerate(testset, 1):
    print(f"[{i}/{len(testset)}] [{case['level']}] {case['question']}")

    start = time.time()
    result = agent.invoke({
        "messages": [{"role": "user", "content": case["question"]}]
    })
    elapsed = time.time() - start

    # 提取调用的工具
    called_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                called_tools.append(tc["name"])

    final_answer = result["messages"][-1].content

    # 工具调用命中率
    if case["expected_tools"]:
        tool_hits = sum(1 for t in case["expected_tools"] if t in called_tools)
        tool_score = tool_hits / len(case["expected_tools"])
    else:
        tool_score = 1.0

    # 关键词命中率
    if case["key_evidence"]:
        kw_hits = sum(1 for kw in case["key_evidence"] if kw in final_answer)
        kw_score = kw_hits / len(case["key_evidence"])
    else:
        kw_score = 1.0

    # 根因判断
    if case["level"] == "L5":
        # L5：无故障场景，回答里不应出现具体故障根因
        fake_roots = ["连接池耗尽", "缓存穿透", "内存泄漏", "连接耗尽", "OOM killed"]
        root_correct = not any(fr in final_answer for fr in fake_roots)
    else:
        # 其他层：ground_truth 的第一个关键词出现在回答里
        key = case["ground_truth"].split()[0]
        root_correct = key in final_answer

    print(f"  调用工具: {called_tools}")
    print(f"  工具命中率: {tool_score:.0%}  关键词命中率: {kw_score:.0%}  根因正确: {root_correct}")
    print(f"  耗时: {elapsed:.1f}s\n")

    results.append({
        "level": case["level"],
        "tool_score": tool_score,
        "kw_score": kw_score,
        "root_correct": root_correct,
        "elapsed": elapsed,
    })
    level_stats[case["level"]].append({
        "tool": tool_score, "kw": kw_score, "root": root_correct
    })

# ---------- 汇总 ----------
print("=" * 60)
print("评测汇总")
print("=" * 60)

avg_tool = sum(r["tool_score"] for r in results) / len(results)
avg_kw = sum(r["kw_score"] for r in results) / len(results)
avg_root = sum(1 for r in results if r["root_correct"]) / len(results)
avg_time = sum(r["elapsed"] for r in results) / len(results)

print(f"总体:")
print(f"  工具调用平均准确率: {avg_tool:.0%}")
print(f"  关键词平均命中率: {avg_kw:.0%}")
print(f"  根因定位准确率: {avg_root:.0%}")
print(f"  平均耗时: {avg_time:.1f}s")
print()

print("分层表现:")
for level in sorted(level_stats.keys()):
    stats = level_stats[level]
    t = sum(s["tool"] for s in stats) / len(stats)
    k = sum(s["kw"] for s in stats) / len(stats)
    r = sum(1 for s in stats if s["root"]) / len(stats)
    print(f"  {level}: 工具 {t:.0%} | 关键词 {k:.0%} | 根因 {r:.0%} | {len(stats)} 条")