"""Agent 核心：工具封装、Agent 构建、证据校验。

此前 agent_v1/v2/v3 与 eval.py/eval_v2.py/eval_v3.py 六份文件大段复制粘贴，
改一处要改六处。现统一收敛到本模块，其余文件只做薄封装或调用。

提供两个层级的 Agent：

- ``build_basic_agent()``     : 仅 ReAct + RAG，无证据校验（消融实验的基线组）
- ``build_verified_agent()``  : 在基线之上叠加证据校验（本方案）

两者除"是否挂载证据校验"外完全一致，从而保证消融实验只改变一个变量。
"""

import os
import threading
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from tools import query_prometheus, search_logs, get_deployment_record

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_EMBEDDING = "BAAI/bge-small-zh-v1.5"
CHROMA_DIR = "./chroma_db"


# ----------------------------------------------------------------------------
# 工具层
# ----------------------------------------------------------------------------

@tool
def query_prometheus_tool(metric_name: str, service: str) -> str:
    """查询监控指标。metric_name 可以是 cpu_usage, memory_usage, response_time。"""
    return query_prometheus(metric_name, service)


@tool
def search_logs_tool(keyword: str, service: str) -> str:
    """搜索服务日志。keyword 是搜索关键词，如 error, timeout, exception, connection pool。"""
    return search_logs(keyword, service)


@tool
def get_deployment_record_tool(service: str) -> str:
    """查询服务的发布记录。"""
    return get_deployment_record(service)


_RETRIEVER_CACHE = {}
_RETRIEVER_LOCK = threading.Lock()


def get_retriever(k: int = 2):
    """惰性构建向量检索器。

    并发评测时多个线程会同时请求，需加锁避免重复加载 embedding 模型
    （每次加载都要读一遍模型权重，是本项目最大的单点启动开销）。
    """
    if k in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[k]
    with _RETRIEVER_LOCK:
        if k not in _RETRIEVER_CACHE:
            embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING)
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
            _RETRIEVER_CACHE[k] = vectorstore.as_retriever(search_kwargs={"k": k})
    return _RETRIEVER_CACHE[k]


def make_search_fault_reports(k: int = 2):
    """构造历史故障检索工具。k 可调，用于检索实验。"""

    @tool
    def search_fault_reports(query: str) -> str:
        """搜索历史故障报告，了解类似故障的原因和解决方案。当需要参考历史经验时使用。"""
        results = get_retriever(k).invoke(query)
        return "\n---\n".join(r.page_content for r in results)

    return search_fault_reports


def build_llm(model: str = DEFAULT_MODEL):
    return ChatOpenAI(model=model, temperature=0)


def default_tools(k: int = 2):
    """本方案的完整工具集（含 RAG 检索）。"""
    return [
        query_prometheus_tool,
        search_logs_tool,
        get_deployment_record_tool,
        make_search_fault_reports(k),
    ]


def build_basic_agent(model: str = DEFAULT_MODEL, k: int = 2, llm=None):
    """基线 Agent：ReAct + RAG，不含证据校验。"""
    llm = llm or build_llm(model)
    return create_agent(llm, default_tools(k))


# ----------------------------------------------------------------------------
# 证据校验
# ----------------------------------------------------------------------------

VERIFY_PROMPT = """你是运维诊断的证据审核员。下面是 Agent 生成的诊断报告，以及它调用工具拿到的原始数据。

请判断：报告中是否给出了一个**具体的故障根因**（如"连接池耗尽""缓存穿透""内存泄漏"），
而这个根因**完全无法**由工具返回的数据推出？

判断要点：
- **可以由数据合理推出 -> 不算编造**。例如"内存 3 天内从 40% 持续涨到 89% 并出现 OOM"
  推出"内存泄漏"，属于合理推断，**不算编造**。
- **完全无依据却给出根因 -> 算编造**。例如"日志和指标都正常"却断言"存在连接池耗尽"。
- 报告说"无异常""当前正常""证据不足" -> 这是保守回答，**不算编造**。
- 报告只是复述工具返回的数据（如"CPU 达到 95%"） -> **不算编造**。

【工具返回的原始数据】
{tool_data}

【Agent 生成的报告】
{answer}

请只回答一个词：是 或 否。"""

REFINE_PROMPT = """你的上一次回答给出了一个故障根因，但审核认为它缺乏数据支撑。

请**重新诊断**，并严格遵守：
1. 逐条列出工具返回的关键数据（指标趋势、日志原文、发布记录）；
2. 如果这些数据**能够指向**某个根因，就明确给出该根因，并写清"哪条数据支持它"；
3. 只有当数据**确实完全不支持任何根因**时，才回答"证据不足"，并说明缺什么数据；
4. **不要仅仅因为证据是间接的就放弃结论**。运维诊断本就依赖多项间接证据的交叉印证，
   例如"内存跨天持续爬升 + 出现 OOM"足以支撑"内存泄漏"这一结论。
5. 不要编造工具没有返回的数据。

原始问题：{question}

工具返回的数据：
{tool_data}

请重新回答："""


def normalize_verdict(text: str) -> bool:
    """把校验模型的回答解析成布尔值。

    只接受"是/否"这类明确表态；解析不出来时保守地判定为"未编造"，
    避免因解析失败而误触发重答、污染消融实验的结论。
    """
    if not text:
        return False
    first = text.strip().split("\n")[0].strip()
    # 去掉序号、标点等噪声（不动中文字符）
    for ch in "0123456789.、)．：:（(）)【】[] \t":
        first = first.replace(ch, "")
    first = first.strip()
    upper = first.upper()
    # 先判否，避免"不是""并非"被误判为"是"
    if first.startswith("否") or first.startswith("非") or upper.startswith("NO"):
        return False
    if first.startswith("是") or upper.startswith("YES"):
        return True
    # 首行无法判断时，回退到前 40 字内的首个明确表态
    head = text[:40]
    if "否" in head or "并非" in head or "不是" in head:
        return False
    if "是" in head:
        return True
    return False


def verify_and_refine(question, answer, tool_data, llm):
    """检查回答是否编造了根因；若编造则要求基于证据重答。

    返回 ``(最终回答, 是否触发了重答, 校验判定)``。
    """
    prompt = VERIFY_PROMPT.format(tool_data=tool_data or "（无工具调用）", answer=answer)
    verdict_text = llm.invoke(prompt).content
    is_fabricated = normalize_verdict(verdict_text)

    if not is_fabricated:
        return answer, False, False

    refined = llm.invoke(REFINE_PROMPT.format(
        question=question, tool_data=tool_data or "（无工具调用）"
    )).content
    return refined, True, True


class VerifiedAgent:
    """在基础 Agent 之上叠加证据校验的包装器。

    对外暴露与 ``create_agent`` 相同的 ``invoke`` 接口，因此评测框架可以
    无差别地对待两种配置。
    """

    def __init__(self, model: str = DEFAULT_MODEL, k: int = 2, agent=None):
        self.llm = build_llm(model)
        # agent 可注入，保证消融实验中基线组与本方案组的工具集、k 完全一致
        self.agent = agent if agent is not None else build_basic_agent(model=model, k=k)
        self.last_refined = False

    def invoke(self, inputs):
        result = self.agent.invoke(inputs)
        messages = result["messages"]
        raw_answer = messages[-1].content

        question = ""
        for msg in messages:
            if msg.__class__.__name__ == "HumanMessage":
                question = msg.content
                break

        tool_data = collect_tool_data(messages)

        t0 = time.time()
        final, refined, _ = verify_and_refine(question, raw_answer, tool_data, self.llm)
        verify_latency = time.time() - t0
        self.last_refined = refined

        result = dict(result)
        result["raw_answer"] = raw_answer
        result["final_answer"] = final
        result["was_refined"] = refined
        result["verify_latency"] = verify_latency
        return result


def build_verified_agent(model: str = DEFAULT_MODEL, k: int = 2):
    return VerifiedAgent(model=model, k=k)


# ----------------------------------------------------------------------------
# 结果解析辅助
# ----------------------------------------------------------------------------

def collect_called_tools(messages):
    """按调用顺序收集工具名（含重复，便于统计轮数）。"""
    called = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                called.append(tc["name"])
    return called


def collect_tool_data(messages):
    """把所有工具返回内容拼接成一段文本，供证据校验使用。"""
    chunks = []
    for msg in messages:
        if msg.__class__.__name__ == "ToolMessage":
            chunks.append(str(msg.content))
    return "\n---\n".join(chunks)


def final_answer_of(result):
    """兼容取最终回答。"""
    return result.get("final_answer") or result["messages"][-1].content
