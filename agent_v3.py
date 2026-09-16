import json
import time
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tools import query_prometheus, search_logs, get_deployment_record

# ---------- 初始化 ----------
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

# ---------- 证据校验 Prompt（优化版） ----------
VERIFY_PROMPT = """你是一个证据校验员。以下是 Agent 生成的诊断报告，以及工具返回的原始数据。

请**只检查一件事**：报告中是否给出了**具体的故障根因**，而这个根因在工具返回的数据中**找不到任何依据**？

判断规则：
- 报告说"连接池耗尽"，但日志里没有 `Connection pool exhausted` → 编造
- 报告说"内存泄漏"，但监控显示内存正常 → 编造
- 报告说"缓存穿透"，但日志里没有 Cache miss → 编造
- 报告说"无异常"、"数据不足"、"当前正常" → **这不算编造，不用管**
- 报告只是描述工具返回的数据（如"CPU 是 95%"） → **这不算编造**

工具返回的原始数据：
{tool_data}

Agent 生成的报告：
{answer}

请回答：
1. 报告中是否编造了**具体的故障根因**？（是/否）
2. 如果有，是哪个根因？
"""


def verify_and_refine(question, answer, tool_data):
    """只抓编造的故障根因，不抓保守回答"""
    prompt = VERIFY_PROMPT.format(tool_data=tool_data, answer=answer)
    verify_result = llm.invoke(prompt).content

    first_line = verify_result.strip().split("\n")[0]
    is_fabricated = first_line.startswith("1.") and "是" in first_line

    if is_fabricated:
        refined_prompt = f"""你之前的回答编造了一个具体的故障根因，但工具返回的数据里并没有支持这个根因的证据。

请重新回答，严格遵循：
1. 只能使用工具实际返回的数据作为证据
2. 如果工具返回的数据显示正常，就明确回答"当前无异常"
3. 如果数据不足以得出根因，就如实说明"证据不足"
4. 不要为了给出结论而编造证据

原始问题：{question}

工具返回的数据：
{tool_data}

请重新回答："""
        new_answer = llm.invoke(refined_prompt).content
        return new_answer, True

    return answer, False


# ---------- 读取测试集 ----------
testset = json.load(open("testset.json", encoding="utf-8"))
print(f"加载了 {len(testset)} 条测试用例（含证据校验）\n")

results = []
level_stats = defaultdict(list)

for i, case in enumerate(testset, 1):
    print(f"[{i}/{len(testset)}] [{case['level']}] {case['question']}")

    start = time.time()
    result = agent.invoke({
        "messages": [{"role": "user", "content": case["question"]}]
    })

    raw_answer = result["messages"][-1].content

    # 收集工具返回数据
    tool_data_list = []
    called_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                called_tools.append(tc["name"])
        if msg.__class__.__name__ == "ToolMessage":
            tool_data_list.append(msg.content)
    tool_data = "\n---\n".join(tool_data_list)

    # 证据校验
    final_answer, was_refined = verify_and_refine(case["question"], raw_answer, tool_data)
    elapsed = time.time() - start

    # 工具调用得分
    if case["expected_tools"]:
        tool_hits = sum(1 for t in case["expected_tools"] if t in called_tools)
        tool_score = tool_hits / len(case["expected_tools"])
    else:
        tool_score = 1.0

    # 关键词得分
    if case["key_evidence"]:
        kw_hits = sum(1 for kw in case["key_evidence"] if kw in final_answer)
        kw_score = kw_hits / len(case["key_evidence"])
    else:
        kw_score = 1.0

    # ---------- 根因判断（修正版） ----------
    if case["level"] == "L5":
        # L5：先看回答里有没有"拒答表达"，有就算对
        deny_phrases = ["无异常", "未发现", "未见异常", "当前正常", "一切正常",
                        "没有发现", "不存在", "无故障", "未检测到", "数据不足",
                        "正常", "没有连接池", "没有内存", "没有问题"]
        if any(dp in final_answer for dp in deny_phrases):
            root_correct = True
        else:
            fake_roots = ["连接池耗尽", "缓存穿透", "内存泄漏", "连接耗尽", "OOM killed"]
            root_correct = not any(fr in final_answer for fr in fake_roots)
    else:
        key = case["ground_truth"].split()[0]
        root_correct = key in final_answer

    refine_mark = " (已校验重答)" if was_refined else ""
    print(f"  工具: {tool_score:.0%}  关键词: {kw_score:.0%}  根因: {root_correct}{refine_mark}")
    print(f"  耗时: {elapsed:.1f}s\n")

    results.append({
        "level": case["level"], "tool_score": tool_score,
        "kw_score": kw_score, "root_correct": root_correct,
        "elapsed": elapsed,
    })
    level_stats[case["level"]].append({
        "tool": tool_score, "kw": kw_score, "root": root_correct
    })

# ---------- 汇总 ----------
print("=" * 60)
print("评测汇总（含证据校验·最终版）")
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