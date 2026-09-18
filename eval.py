"""eval.py：快速冒烟评测（6 条用例）。

用途仅限于"改动后确认链路没坏"，正式实验请使用 eval_v3.py。
覆盖四个工具、一次拒答、一次不存在服务。

用法：python eval.py
"""

import sys

from evalframework import (
    CONFIG_VERIFIED, TestCase, evaluate, summarize, render_summary,
)

# 覆盖四个工具与一次拒答的极小样例集
SMOKE_CASES = [
    TestCase(
        id="S-01", level="L1",
        question="order-service 昨晚的 CPU 使用率是多少？",
        ground_truth="95%",
        expected_tools=["query_prometheus_tool"],
        evidence_keywords=["95"], must_roots=["95"],
    ),
    TestCase(
        id="S-02", level="L1",
        question="order-service 昨晚发过版吗？",
        ground_truth="发过 v2.3.1",
        expected_tools=["get_deployment_record_tool"],
        evidence_keywords=["v2.3.1"], must_roots=["v2.3.1"],
    ),
    TestCase(
        id="S-03", level="L3",
        question="order-service 昨晚为什么响应变慢？",
        ground_truth="连接池耗尽",
        expected_tools=["query_prometheus_tool", "search_logs_tool", "get_deployment_record_tool"],
        evidence_keywords=["95", "Connection pool"],
        must_roots=["连接池"],
        forbidden_roots=["内存泄漏", "缓存穿透"],
    ),
    TestCase(
        id="S-04", level="L4",
        question="历史上有没有类似的连接池故障？当时是怎么解决的？",
        ground_truth="2024-01-10 连接池故障，回滚并调整 max_connections",
        expected_tools=["search_fault_reports"],
        evidence_keywords=["连接池"], must_roots=["max_connections"],
    ),
    TestCase(
        id="S-05", level="L5", kind="deny",
        question="gateway-service 有没有连接池问题？",
        ground_truth="没有，gateway-service 日志中无连接池相关记录",
        expected_tools=["search_logs_tool"],
        valid_tool_sets=[["search_logs_tool"]],
        forbidden_roots=["连接池耗尽", "内存泄漏", "缓存穿透"],
    ),
    TestCase(
        id="S-06", level="L5", kind="unknown",
        question="帮我查一下 payment-service 的监控",
        ground_truth="系统中不存在该服务",
        expected_tools=["query_prometheus_tool"],
        valid_tool_sets=[["query_prometheus_tool"]],
        must_roots=["未找到|不存在|没有找到|查不到|无法查询|没有这个服务|不包含"],
        forbidden_roots=["内存泄漏", "连接池耗尽", "缓存穿透"],
    ),
]


def main():
    from agent_core import build_verified_agent

    agent = build_verified_agent()
    results = evaluate(agent, SMOKE_CASES, config=CONFIG_VERIFIED,
                       repeats=1, warmup=0, verbose=True)
    summary = summarize(results)
    print(render_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
