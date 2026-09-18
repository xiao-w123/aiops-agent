"""可复现的评测框架。

相对旧版 eval.py / eval_v2.py / eval_v3.py 的修复点
--------------------------------------------------
1. **工具调用改为 P/R/F1**。旧版只算召回（``t in called_tools``），
   导致"每题都调用全部工具"的退化解也能拿 100%。现在同时惩罚多调。
2. **根因判定改为短语组**。旧版 ``ground_truth.split()[0]`` 在中文上会截断
   多字短语，把完全正确的回答判错；现按同义短语组（任一命中即算命中）匹配，
   并支持 ``forbidden_roots`` 反例断言，捕捉"结论正确但夹带幻觉根因"。
3. **拒答判定改为语义断言**。旧版只要回答里出现"正常"二字就算拒答正确，
   会把"CPU 正常，但日志有大量 ERROR"这种描述真实故障的回答误判为正确。
4. **重复实验**。旧版每组只跑一次，把单次噪声当成结论；现跑 N 次并报
   均值 ± 标准差，同时报告本次运行内的实测最小间隔（噪声下界）。
5. **真实耗时口径**。旧版把证据校验的耗时算进了平均延迟却未说明；现分开统计。
6. **结果落盘**。导出 JSON / CSV / Markdown，供论文直接引用，避免手工誊抄。
"""

import csv
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

CONFIG_BASIC = "basic"        # 基线：ReAct + RAG，无证据校验
CONFIG_VERIFIED = "verified"  # 本方案：基线 + 证据校验


@dataclass
class TestCase:
    id: str
    level: str
    question: str
    ground_truth: str
    expected_tools: List[str] = field(default_factory=list)
    evidence_keywords: List[str] = field(default_factory=list)
    must_roots: List[str] = field(default_factory=list)
    forbidden_roots: List[str] = field(default_factory=list)
    kind: Optional[str] = None       # L5 专用：deny / lookup / unknown
    review_note: str = ""
    # 等价工具集：某些问题可通过不同证据路径回答（如"有无内存泄漏"既看监控
    # 也看日志）。只要实际调用的工具去重后等于其中任一集合，即视为最优。
    valid_tool_sets: List[List[str]] = field(default_factory=list)

    @staticmethod
    def from_dict(d):
        return TestCase(
            id=d["id"],
            level=d["level"],
            question=d["question"],
            ground_truth=d.get("ground_truth", ""),
            expected_tools=d.get("expected_tools", []),
            evidence_keywords=d.get("evidence_keywords", []),
            must_roots=d.get("must_roots", []),
            forbidden_roots=d.get("forbidden_roots", []),
            kind=d.get("kind"),
            review_note=d.get("review_note", ""),
            valid_tool_sets=d.get("valid_tool_sets", []),
        )


@dataclass
class CaseResult:
    case_id: str
    level: str
    repeat: int
    config: str
    called_tools: List[str]
    tool_precision: float
    tool_recall: float
    tool_f1: float
    evidence_score: float
    root_correct: bool
    forbidden_hit: bool
    deny_correct: Optional[bool]
    was_refined: bool
    latency_total: float
    latency_agent: float
    latency_verify: float
    tool_calls: int
    answer_chars: int
    answer_snippet: str = ""

    def to_row(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def tool_scores(expected: List[str], called: List[str], valid_sets=None):
    """返回 (precision, recall, f1)。多调与漏调都会被惩罚。

    valid_sets 给出若干"等价最优工具集"。若实际调用（去重后）恰好等于其中
    任一个，则 P/R/F1 全记 1.0——因为存在多条等价证据路径时，
    用其中任意一条都算最优解，不该因未走另一条而扣分。
    """
    called_set = set(called)

    if valid_sets:
        for vs in valid_sets:
            if called_set == set(vs):
                return 1.0, 1.0, 1.0

    if not expected:
        return (1.0, 1.0, 1.0) if not called else (0.0, 1.0, 0.0)

    hits = sum(1 for t in expected if t in called_set)
    recall = hits / len(expected)
    precision = (sum(1 for t in called_set if t in expected) / len(called_set)) if called_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def match_groups(groups: List[str], text: str) -> float:
    """短语组匹配率。每个元素是一组同义表达，任一命中即算该组命中。"""
    if not groups:
        return 1.0
    hit = 0
    for group in groups:
        variants = [v.strip() for v in group.split("|") if v.strip()]
        if any(v in text for v in variants):
            hit += 1
    return hit / len(groups)


def any_hit(groups: List[str], text: str) -> bool:
    """must_roots 采用"全部组都要命中"的严格口径，避免单个宽泛词蒙混过关。"""
    if not groups:
        return True
    return match_groups(groups, text) >= 1.0


def forbidden_hit(groups: List[str], text: str) -> bool:
    """是否**断言**了反例根因（否定语气出现不算）。"""
    phrases = [v.strip() for group in groups for v in group.split("|") if v.strip()]
    return asserted_in_text(text, phrases)


DENY_MARKERS = ["无异常", "未见异常", "未发现", "没有发现", "当前正常", "运行正常",
                "一切正常", "不存在", "无故障", "未检测到", "没有", "正常",
                "不构成", "排除了", "未出现", "不具备", "尚未"]

# 用于判定"是否**断言**了具体根因"的词表（拒答题中不应出现肯定断言）。
# 注意只放完整根因名，不放"泄漏""连接池"这类部件名——
# 后者在正常回答里作为主语出现（如"连接池参数正常"），会制造大量误判。
CONCRETE_ROOT_MARKERS = ["内存泄漏", "缓存穿透", "连接池耗尽", "内存溢出",
                         "OOM killed", "磁盘 IO 瓶颈", "网络抖动", "死锁"]

# 否定词。若根因词之前出现否定词，说明该句是在"否认"这个根因而非断言它。
NEGATION_WORDS = ["未发现", "未出现", "未检测到", "未见", "没有", "没发现", "不存在",
                  "无", "非", "排除", "并非", "不属于", "算不上", "不具备", "未"]

# 转折词。出现转折时左右分句语气相反，必须拆开判断，
# 否则"CPU 指标正常，但日志有大量 ERROR"会被误判为"整体正常"。
CONTRAST_WORDS = ["但", "但是", "然而", "不过", "却", "可是"]

# 比较/指代上下文。根因词出现在这些词附近时，通常是在与**别的**故障做对照
# 或只起说明作用，而不是在断言当前服务的根因。例如：
#   "历史内存泄漏故障的典型特征是内存从 40% 涨到 85%"
#   "与 user-service 的内存泄漏相比，本服务不符合"
# 若不做这层过滤，正确的拒答会被误判为幻觉（本项目实测曾由此产生 3/6 的假阳性）。
# 比较/指代上下文。根因词出现在这些词构成的分句里时，通常是在与**别的**
# 故障做对照或只起说明作用，而不是在断言当前服务的根因。例如：
#   "历史内存泄漏故障的典型特征是从 40% 涨到 85%"
#   "与 user-service 的内存泄漏相比，本服务不符合"
# 若不做这层过滤，正确的拒答会被误判为幻觉（本项目实测曾由此产生 3/6 的假阳性）。
# 注意：词表必须保守。像"历史""经验"这类词会同时出现在
# "结合历史经验这类现象源于内存泄漏"这种**实质断言**里，不能列进来。
COMPARATIVE_CONTEXT_MARKERS = ["典型", "以往", "曾经", "上次", "相比",
                               "对比", "类似", "例如", "比如", "参考"]

# 弱化/推测语气标记。用于区分两种表述：
#   "没有发现内存泄漏"        -> 明确拒答，正确
#   "不排除存在内存泄漏的可能" -> 推测性断言，在拒答题中属于过度诊断
# 若不作区分，前者会被"不"字误判为否定词而放过后者。
UNCERTAINTY_MARKERS = ["不排除", "不能排除", "无法排除", "难以排除",
                       "大概率", "可能性很大", "很可能是", "极可能",
                       "或许", "也许", "怀疑", "倾向于认为"]

# 分句标点
CLAUSE_SEPS = "。；;！!？?\n"

# 子句长度下限。若去掉根因词后剩余内容过少，说明该子句不足以承载完整语义，
# 无法判断其语气，此时不应认定为断言。
# 该守卫专门用于防御**截断产生的碎片**：记录回答摘要时若恰好截断在根因词上，
# 会凭空产生一个"根因词单独成句"的碎片，使判分失去语境而误判。
# 取值不能过大，否则会漏掉"也存在的风险"（剩余 6 字）这类真实断言。
MIN_CLAUSE_LEN = 4


def _split_clauses(text: str):
    """先按转折词切分，再按句末标点切分，得到语气一致的子句列表。"""
    clauses = [text]
    for cw in CONTRAST_WORDS:
        expanded = []
        for c in clauses:
            expanded.extend(c.split(cw))
        clauses = expanded
    out = []
    for c in clauses:
        for sep in CLAUSE_SEPS:
            c = c.replace(sep, "\x00")
        out.extend([p.strip() for p in c.split("\x00") if p.strip()])
    return out


def _asserted_in_clause(clause: str, phrase: str) -> bool:
    """短语在该子句中是否以**肯定断言**语气出现。

    需同时满足：
    1. 未被前置否定词否认（"没有发现内存泄漏"不算断言）；
    2. 不处于比较/指代语境（"内存泄漏的典型特征是…"不算断言）；
    3. 该分句不含弱化/推测语气（"不排除存在内存泄漏的可能"仍算断言）。

    第 3 条是刻意选择的严格方向：只要给出了具体根因，即使带推测词也算，
    宁可多报幻觉也不少报，以免重蹈旧版"为让数字好看而放宽判定"的覆辙。
    """
    has_phrase = phrase in clause
    if not has_phrase:
        return False
    # 语义不完整的碎片不作断言认定（见 MIN_CLAUSE_LEN 说明）
    residue = clause.replace(phrase, "").strip(" \t*#-—:：，,、")
    if len(residue) < MIN_CLAUSE_LEN:
        return False

    if any(m in clause for m in UNCERTAINTY_MARKERS):
        return True

    start = 0
    while True:
        idx = clause.find(phrase, start)
        if idx < 0:
            return False
        prefix = clause[:idx]
        negated = any(neg in prefix for neg in NEGATION_WORDS)
        contextual = any(m in clause for m in COMPARATIVE_CONTEXT_MARKERS)
        if not negated and not contextual:
            return True
        start = idx + len(phrase)


def make_snippet(answer: str, limit: int = 400) -> str:
    """生成回答摘要，并保证截断落在句末标点上。

    若在句子中间截断，可能恰好切出一个只含根因词的碎片，
    使判分逻辑失去语境而误判（本项目实测曾由此产生假阳性）。
    """
    text = (answer or "").replace("\n", " ")
    if len(text) <= limit:
        return text
    window = text[:limit]
    for sep in CLAUSE_SEPS:
        pos = window.rfind(sep)
        if pos > limit * 0.5:
            return window[: pos + 1]
    return window


def asserted_in_text(text: str, phrases) -> bool:
    """短语集合中是否有任一短语在文中被肯定地断言（否定语气不算）。

    这是判分逻辑的核心修正点：旧版用 `phrase in answer` 判断，
    导致"没有发现内存泄漏"里的"内存泄漏"被当成幻觉证据，
    直接把正确答案判错，也让拒答率指标失真。
    """
    for clause in _split_clauses(text):
        for phrase in phrases:
            if _asserted_in_clause(clause, phrase):
                return True
    return False


def asserts_concrete_root(text: str) -> bool:
    """判断回答中是否**断言**了一个具体故障根因。"""
    return asserted_in_text(text, CONCRETE_ROOT_MARKERS)


def grade_l5(case: TestCase, answer: str):
    """L5 判定。返回 (root_correct, deny_correct)。

    与旧版的关键区别：旧版只要回答里出现"正常"二字就算拒答正确，
    会把"CPU 正常，但日志有大量 ERROR"这种描述真实故障的回答判为正确。
    现在必须同时满足：(1) 有明确的否认/说明表述；(2) 没有断言具体根因。
    """
    gave_root = asserts_concrete_root(answer)
    has_denial = any(m in answer for m in DENY_MARKERS)
    deny_correct = has_denial and not gave_root
    fh = forbidden_hit(case.forbidden_roots, answer)

    if case.kind == "unknown":
        # 不存在的服务：必须如实说明查不到，且不得编造指标或根因
        root_correct = any_hit(case.must_roots, answer) and not fh and not gave_root
        deny_correct = root_correct
    else:
        root_correct = deny_correct and not fh

    return root_correct, deny_correct


def grade_case(case: TestCase, answer: str):
    """返回通用指标字典。"""
    if case.level == "L5":
        root_correct, deny_correct = grade_l5(case, answer)
        return {
            "root_correct": root_correct,
            "deny_correct": deny_correct,
            "forbidden_hit": forbidden_hit(case.forbidden_roots, answer),
            "evidence_score": 1.0 if not case.evidence_keywords else match_groups(case.evidence_keywords, answer),
        }

    root_correct = any_hit(case.must_roots, answer)
    fh = forbidden_hit(case.forbidden_roots, answer)
    if fh:
        root_correct = False
    return {
        "root_correct": root_correct,
        "deny_correct": None,
        "forbidden_hit": fh,
        "evidence_score": match_groups(case.evidence_keywords, answer),
    }


# ---------------------------------------------------------------------------
# Agent 运行与计时
# ---------------------------------------------------------------------------

def load_testset(path="testset.json") -> List[TestCase]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [TestCase.from_dict(d) for d in raw]


def run_one(agent, case: TestCase, repeat_idx: int, config: str) -> CaseResult:
    from agent_core import (
        collect_called_tools, collect_tool_data, final_answer_of,
    )

    t0 = time.time()
    result = agent.invoke({"messages": [{"role": "user", "content": case.question}]})
    latency_total = time.time() - t0

    was_refined = bool(result.get("was_refined")) if isinstance(result, dict) else False
    called = collect_called_tools(result["messages"])
    answer = final_answer_of(result)

    # 证据校验耗时 = 总耗时 - 纯 Agent 耗时，无法直接拆分时置 0
    latency_verify = result.get("verify_latency", 0.0) if isinstance(result, dict) else 0.0
    latency_agent = latency_total - latency_verify

    precision, recall, f1 = tool_scores(case.expected_tools, called, case.valid_tool_sets)
    g = grade_case(case, answer)

    return CaseResult(
        case_id=case.id,
        level=case.level,
        repeat=repeat_idx,
        config=config,
        called_tools=called,
        tool_precision=precision,
        tool_recall=recall,
        tool_f1=f1,
        evidence_score=g["evidence_score"],
        root_correct=g["root_correct"],
        forbidden_hit=g["forbidden_hit"],
        deny_correct=g["deny_correct"],
        was_refined=was_refined,
        latency_total=latency_total,
        latency_agent=latency_agent,
        latency_verify=latency_verify,
        tool_calls=len(called),
        answer_chars=len(answer or ""),
        answer_snippet=make_snippet(answer),
    )


def evaluate(agent, cases: List[TestCase], config: str = CONFIG_VERIFIED,
             repeats: int = 1, warmup: int = 0, verbose: bool = True,
             sleep_between: float = 0.0, workers: int = 1) -> List[CaseResult]:
    """跑一轮完整评测。

    workers > 1 时并发执行用例。复杂用例（L3/L4）一次会话要 5-8 次 LLM 往返、
    耗时可达 40 秒，串行跑全量测试集会非常慢；并发可以显著压缩墙钟时间。
    """
    import concurrent.futures
    import threading

    for w in range(warmup):
        if verbose:
            print(f"[预热 {w + 1}/{warmup}] ...")
        agent.invoke({"messages": [{"role": "user", "content": cases[0].question}]})

    # 组装全部待跑任务（可复现：固定按 repeat -> case 的顺序编号）
    jobs = []
    for r in range(repeats):
        for case in cases:
            jobs.append((r, case))

    results: List[CaseResult] = []
    total = len(jobs)
    done = 0
    lock = threading.Lock()

    def _run(job):
        r, case = job
        res = run_one(agent, case, r, config)
        if sleep_between:
            time.sleep(sleep_between)
        return res

    if workers <= 1:
        for idx, job in enumerate(jobs, 1):
            res = _run(job)
            results.append(res)
            if verbose:
                _log_progress(idx, total, res)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            with lock:
                done += 1
                results.append(res)
                if verbose:
                    _log_progress(done, total, res)

    # 按 (repeat, case_id) 排序，保证输出稳定、便于逐次对比
    order = {c.id: i for i, c in enumerate(cases)}
    results.sort(key=lambda x: (x.repeat, order.get(x.case_id, 0)))
    return results


def _log_progress(idx, total, res):
    mark = "重答" if res.was_refined else "    "
    print(f"[{idx}/{total}] [{res.level}] {res.case_id} "
          f"F1={res.tool_f1:.2f} 证据={res.evidence_score:.2f} "
          f"根因={'OK' if res.root_correct else 'NG'} {mark} "
          f"{res.latency_total:.1f}s", flush=True)


# ---------------------------------------------------------------------------
# 汇总与统计
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    ("tool_precision", "工具精确率"),
    ("tool_recall", "工具召回率"),
    ("tool_f1", "工具F1"),
    ("evidence_score", "证据命中率"),
    ("root_rate", "根因准确率"),
    ("deny_rate", "拒答正确率"),
    ("forbidden_rate", "幻觉根因率"),
    ("refine_rate", "触发重答率"),
    ("latency_total", "平均耗时(s)"),
    ("latency_agent", "纯Agent耗时(s)"),
    ("latency_verify", "校验耗时(s)"),
]


def _agg(values):
    if not values:
        return None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def summarize(results: List[CaseResult]):
    """按 (config, level) 聚合；返回嵌套字典。"""
    by_config = defaultdict(list)
    for r in results:
        by_config[r.config].append(r)

    out = {}
    for config, rows in by_config.items():
        # 先按重复次数分组，算出每次重复的指标，再对重复取均值/标准差
        repeats = sorted({r.repeat for r in rows})
        per_repeat = defaultdict(list)
        for r in rows:
            per_repeat[r.repeat].append(r)

        def repeat_level(rows_sub):
            root = [1.0 if x.root_correct else 0.0 for x in rows_sub]
            deny = [1.0 if x.deny_correct else 0.0 for x in rows_sub if x.deny_correct is not None]
            forb = [1.0 if x.forbidden_hit else 0.0 for x in rows_sub]
            ref = [1.0 if x.was_refined else 0.0 for x in rows_sub]
            return {
                "tool_precision": statistics.mean([x.tool_precision for x in rows_sub]),
                "tool_recall": statistics.mean([x.tool_recall for x in rows_sub]),
                "tool_f1": statistics.mean([x.tool_f1 for x in rows_sub]),
                "evidence_score": statistics.mean([x.evidence_score for x in rows_sub]),
                "root_rate": statistics.mean(root),
                "deny_rate": statistics.mean(deny) if deny else None,
                "forbidden_rate": statistics.mean(forb),
                "refine_rate": statistics.mean(ref),
                "latency_total": statistics.mean([x.latency_total for x in rows_sub]),
                "latency_agent": statistics.mean([x.latency_agent for x in rows_sub]),
                "latency_verify": statistics.mean([x.latency_verify for x in rows_sub]),
                "n": len(rows_sub),
            }

        # 总体：每个重复先汇总，再跨重复统计
        overall_per_repeat = [repeat_level(per_repeat[rp]) for rp in repeats]
        overall = {}
        for key, _ in METRIC_KEYS:
            vals = [d[key] for d in overall_per_repeat if d[key] is not None]
            overall[key] = _agg(vals)

        levels = {}
        for level in sorted({r.level for r in rows}):
            lvl_per_repeat = []
            for rp in repeats:
                sub = [x for x in per_repeat[rp] if x.level == level]
                if sub:
                    lvl_per_repeat.append(repeat_level(sub))
            if not lvl_per_repeat:
                continue
            lv = {}
            for key, _ in METRIC_KEYS:
                vals = [d[key] for d in lvl_per_repeat if d[key] is not None]
                lv[key] = _agg(vals)
            lv["n"] = lvl_per_repeat[0]["n"]
            levels[level] = lv

        out[config] = {"overall": overall, "levels": levels, "n_repeats": len(repeats)}
    return out


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------

def fmt(metric_pair):
    """把 (mean, std) 格式化成 'xx.x ± y.y' 形式。"""
    if metric_pair is None:
        return "—"
    mean, std = metric_pair
    return f"{mean * 100:.1f} ± {std * 100:.1f}"


def fmt_sec(metric_pair):
    if metric_pair is None:
        return "—"
    mean, std = metric_pair
    return f"{mean:.2f} ± {std:.2f}"


def render_summary(summary) -> str:
    lines = []
    level_names = {
        "L1": "单工具事实查询", "L2": "多工具联合推理", "L3": "根因定位",
        "L4": "历史经验迁移", "L5": "边界与拒答",
    }
    for config, data in summary.items():
        title = "含证据校验（本方案）" if config == CONFIG_VERIFIED else "无证据校验（基线）"
        nrep = data["n_repeats"]
        lines.append(f"\n### 配置：{title}   （重复 {nrep} 次，指标为 均值 ± 标准差）\n")

        header = ("| 层级 | 能力 | 工具P | 工具R | 工具F1 | 证据命中 | 根因准确 | 拒答正确 | 幻觉根因 | 平均耗时 |")
        sep = "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        lines.append(header)
        lines.append(sep)

        for level in sorted(data["levels"].keys()):
            lv = data["levels"][level]
            lines.append(
                f"| {level} | {level_names.get(level, '')} | "
                f"{fmt(lv['tool_precision'])} | {fmt(lv['tool_recall'])} | {fmt(lv['tool_f1'])} | "
                f"{fmt(lv['evidence_score'])} | {fmt(lv['root_rate'])} | "
                f"{fmt(lv['deny_rate'])} | {fmt(lv['forbidden_rate'])} | "
                f"{fmt_sec(lv['latency_total'])} |"
            )

        ov = data["overall"]
        lines.append(
            f"| **总体** | — | {fmt(ov['tool_precision'])} | {fmt(ov['tool_recall'])} | "
            f"{fmt(ov['tool_f1'])} | {fmt(ov['evidence_score'])} | {fmt(ov['root_rate'])} | "
            f"{fmt(ov['deny_rate'])} | {fmt(ov['forbidden_rate'])} | {fmt_sec(ov['latency_total'])} |"
        )
    return "\n".join(lines)


def render_ablation(summary) -> str:
    if CONFIG_BASIC not in summary or CONFIG_VERIFIED not in summary:
        return "\n（只跑了一种配置，无法生成消融对比表）\n"

    b, v = summary[CONFIG_BASIC]["overall"], summary[CONFIG_VERIFIED]["overall"]
    rows = [
        ("工具调用 F1", b["tool_f1"], v["tool_f1"], True),
        ("证据命中率", b["evidence_score"], v["evidence_score"], True),
        ("根因定位准确率", b["root_rate"], v["root_rate"], True),
        ("幻觉根因率（越低越好）", b["forbidden_rate"], v["forbidden_rate"], True),
        ("平均耗时(s)", b["latency_total"], v["latency_total"], False),
    ]
    lines = ["\n### 消融实验：证据校验的增益\n",
             "| 指标 | 无证据校验 | 含证据校验 | 变化 |",
             "| :--- | :--- | :--- | :--- |"]
    for name, bm, vm, pct in rows:
        if pct:
            bm_s, vm_s = fmt(bm), fmt(vm)
            delta = (vm[0] - bm[0]) * 100
            lines.append(f"| {name} | {bm_s} | {vm_s} | {delta:+.1f} pp |")
        else:
            bm_s, vm_s = fmt_sec(bm), fmt_sec(vm)
            delta = (vm[0] - bm[0]) / bm[0] * 100 if bm[0] else 0
            lines.append(f"| {name} | {bm_s} | {vm_s} | {delta:+.1f}% |")

    lb = summary[CONFIG_BASIC]["levels"]
    lv = summary[CONFIG_VERIFIED]["levels"]
    if "L5" in lb and "L5" in lv:
        lines.append(
            f"\n**L5（边界与拒答）对比**：拒答正确率 "
            f"{fmt(lb['L5']['deny_rate'])} -> {fmt(lv['L5']['deny_rate'])}；"
            f"幻觉根因率 {fmt(lb['L5']['forbidden_rate'])} -> {fmt(lv['L5']['forbidden_rate'])}。"
        )
    return "\n".join(lines)


def save_diagnostics(results: List[CaseResult], path="results/diagnostics.md"):
    """导出逐条用例的工具调用序列，用于人工核查。

    指标异常时（例如工具精确率偏低、某层根因准确率骤降），
    必须先看清 Agent 实际调了哪些工具，再判断问题出在 Agent 还是期望标注上。
    """
    outdir = os.path.dirname(path)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    by_case = defaultdict(list)
    for r in results:
        by_case[(r.level, r.case_id)].append(r)

    lines = ["# 逐条诊断明细\n",
             "用于核查指标异常时，Agent 的实际行为与期望标注是否一致。\n"]
    for (level, cid) in sorted(by_case.keys()):
        rows = by_case[(level, cid)]
        lines.append(f"\n## {cid}（{level}，{len(rows)} 次）\n")
        for r in sorted(rows, key=lambda x: x.repeat):
            tools = " -> ".join(r.called_tools) if r.called_tools else "（未调用工具）"
            lines.append(
                f"- 第 {r.repeat + 1} 次：工具序列 `{tools}`；"
                f"P={r.tool_precision:.2f} R={r.tool_recall:.2f} F1={r.tool_f1:.2f}；"
                f"根因={'正确' if r.root_correct else '错误'}；"
                f"幻觉={'是' if r.forbidden_hit else '否'}；"
                f"{'触发重答；' if r.was_refined else ''}"
                f"{r.latency_total:.1f}s"
            )
            if r.answer_snippet:
                lines.append(f"  - 回答摘要：{r.answer_snippet}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def save_results(results: List[CaseResult], summary, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(outdir, f"raw_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_row() for r in results], f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(outdir, f"raw_{stamp}.csv")
    if results:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].to_row().keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_row())

    md_path = os.path.join(outdir, f"report_{stamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 评测报告 {stamp}\n")
        f.write(render_summary(summary))
        f.write("\n")
        f.write(render_ablation(summary))
        f.write("\n")

    # 便于 README 引用的固定文件名
    latest = os.path.join(outdir, "report_latest.md")
    with open(latest, "w", encoding="utf-8") as f:
        f.write(f"# 评测报告 {stamp}\n")
        f.write(render_summary(summary))
        f.write("\n")
        f.write(render_ablation(summary))
        f.write("\n")

    return json_path, csv_path, md_path, latest
