"""自检：不调用任何 LLM API，验证仿真环境、测试集与判分逻辑的一致性。

这是论文实验可复现性的一部分——任何人 clone 后先跑本脚本，
即可确认"标准答案与仿真数据是自洽的"，不必先烧 API 额度。

用法：python selfcheck.py
"""

import json
import sys

from evalframework import TestCase, grade_case, load_testset, tool_scores
from tools import query_prometheus, search_logs, get_deployment_record

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}" + (f"  <- {detail}" if detail and not ok else ""))


# ---------------------------------------------------------------------------
# 1. 仿真环境自洽性
# ---------------------------------------------------------------------------

def check_simulator():
    print("\n--- 1. 仿真环境 ---")
    # order-service：连接池耗尽
    check("order-service CPU 峰值 95%", "95%" in query_prometheus("cpu_usage", "order-service"))
    check("order-service 响应时间 3.2s", "3.2s" in query_prometheus("response_time", "order-service"))
    check("order-service 内存平稳（无泄漏）",
          "58%" in query_prometheus("memory_usage", "order-service")
          and "OOM" not in query_prometheus("memory_usage", "order-service"))
    check("order-service 有连接池报错",
          "Connection pool exhausted" in search_logs("connection pool", "order-service"))
    check("order-service 无 OOM 日志", "OOM" not in search_logs("oom", "order-service"))
    check("order-service 无缓存穿透迹象", "Cache miss" not in search_logs("cache", "order-service"))
    check("order-service 发布记录含 v2.3.1",
          "v2.3.1" in get_deployment_record("order-service"))
    check("order-service 发布记录含干扰版本 v2.2.0",
          "v2.2.0" in get_deployment_record("order-service"))

    # product-service：缓存穿透（内存仍正常，用于考察交叉推理）
    check("product-service 有缓存穿透迹象",
          "Cache miss rate 95%" in search_logs("cache", "product-service"))
    check("product-service CPU 故障峰值 83%", "83%" in query_prometheus("cpu_usage", "product-service"))
    check("product-service 内存正常（非泄漏）",
          "55%" in query_prometheus("memory_usage", "product-service"))
    check("product-service 无连接池报错",
          "未找到" in search_logs("connection pool", "product-service"))

    # user-service：内存泄漏
    check("user-service 内存跨天爬升 40%->85%",
          all(x in query_prometheus("memory_usage", "user-service") for x in ["40%", "62%", "85%"]))
    check("user-service 有 OOM 日志", "OOM killed" in search_logs("oom", "user-service"))

    # gateway-service：完全健康的对照服务（拒答测试的前提）
    check("gateway-service CPU 全程健康",
          all(int(i["value"].rstrip("%")) < 50
              for i in json.loads(query_prometheus("cpu_usage", "gateway-service"))["series"]))
    check("gateway-service 内存平稳 44%-46%",
          "44%" in query_prometheus("memory_usage", "gateway-service"))
    check("gateway-service 无 ERROR 日志", "未找到" in search_logs("error", "gateway-service"))
    check("gateway-service 无连接池记录",
          "未找到" in search_logs("connection pool", "gateway-service"))
    check("gateway-service 无缓存 miss 记录",
          "未找到" in search_logs("cache", "gateway-service"))
    check("gateway-service 无 OOM 记录", "未找到" in search_logs("oom", "gateway-service"))
    check("gateway-service 无发布记录",
          "未找到" in get_deployment_record("gateway-service"))

    check("未知服务报错", "未找到服务" in query_prometheus("cpu_usage", "payment-service"))
    check("日志关键词同义词生效（'连接池' 可命中 connection pool）",
          "Connection pool" in search_logs("连接池", "order-service"))


# ---------------------------------------------------------------------------
# 2. 测试集结构合法性
# ---------------------------------------------------------------------------

def check_testset(cases):
    print("\n--- 2. 测试集结构 ---")
    ids = [c.id for c in cases]
    check("用例 ID 唯一", len(ids) == len(set(ids)))

    levels = {}
    for c in cases:
        levels[c.level] = levels.get(c.level, 0) + 1
    check(f"共 {len(cases)} 条用例，覆盖 L1-L5",
          set(levels) == {"L1", "L2", "L3", "L4", "L5"}, str(levels))

    bad_tools = [c.id for c in cases if not c.expected_tools]
    check("每条用例都标注了期望工具", not bad_tools, str(bad_tools))

    # L3/L4/L5 必须给出根因断言或拒答断言，否则无法判分
    no_assert = [c.id for c in cases
                 if c.level in ("L3", "L4", "L5") and not c.must_roots and not c.forbidden_roots
                 and c.kind != "deny"]
    check("L3/L4/L5 都有可判分的断言", not no_assert, str(no_assert))

    bad_l5 = [c.id for c in cases if c.level == "L5" and not c.kind]
    check("L5 用例都标注了 kind（deny/lookup/unknown）", not bad_l5, str(bad_l5))

    # 关键：拒答类用例的 expected_tools 必须能真正查到"无问题"的证据
    print("  拒答类用例（L5 deny）的 expected_tools 指向的工具，"
          "其返回值必须确实不含该故障证据：")
    for c in cases:
        if c.level == "L5" and c.kind == "deny":
            print(f"    {c.id}: {c.question}")


# ---------------------------------------------------------------------------
# 3. 判分逻辑的对抗性测试
# ---------------------------------------------------------------------------

def check_grading(cases):
    print("\n--- 3. 判分逻辑对抗性测试 ---")
    by_id = {c.id: c for c in cases}

    # 3.1 完美回答应当通过
    perfect = {
        "L3-001": "order-service 昨晚 23:00-01:00 CPU 冲到 95%、响应时间 3.2s，"
                  "日志出现 Connection pool exhausted，"
                  "根因是 v2.3.1 优化连接池配置导致连接池耗尽。",
        "L3-002": "product-service 出现缓存命中率骤降到 95% miss，属于缓存穿透导致，"
                  "而非连接池配置问题。",
        "L3-003": "user-service 内存从 40% 持续爬升到 89% 并触发 OOM，属于内存泄漏。",
        "L3-004": "根据历史故障报告，根因是异常处理分支未关闭数据库连接，导致连接对象未释放。",
        "L4-001": "历史上 2024-01-10 有过连接池故障，通过把 max_connections 从 50 调回 200 解决。",
    }
    for cid, ans in perfect.items():
        g = grade_case(by_id[cid], ans)
        check(f"{cid} 完美回答判为正确", g["root_correct"], str(g))

    # 3.1b 排除干扰项不等于断言干扰根因（曾因误设 forbidden_roots 而误判）
    g = grade_case(by_id["L3-002"],
                   "这是典型的缓存穿透，而不是连接池配置问题，也不是内存泄漏。")
    check("L3-002 排除连接池/内存泄漏干扰项仍判为正确", g["root_correct"], str(g))

    # 3.2 含幻觉根因的回答必须判错（即使主根因正确）
    hallucinated = {
        "L3-001": "根因是连接池耗尽，同时也存在内存泄漏和磁盘 IO 瓶颈。",
        "L3-003": "user-service 内存持续爬升属于内存泄漏，但也存在缓存穿透的风险。",
    }
    for cid, ans in hallucinated.items():
        g = grade_case(by_id[cid], ans)
        check(f"{cid} 夹带幻觉根因判为错误", not g["root_correct"], str(g))
        check(f"{cid} 幻觉根因被标记", g["forbidden_hit"], str(g))

    # 3.3 拒答类：正确描述"无异常"应通过
    deny_ok = {
        "L5-004": "gateway-service 日志中没有查到任何连接池相关记录，当前无连接池问题。",
        "L5-006": "gateway-service 内存指标平稳在 44%-46%，没有发现内存泄漏迹象。",
        "L5-003": "gateway-service 日志中未发现任何 Cache miss 记录，不存在缓存穿透问题。",
    }
    for cid, ans in deny_ok.items():
        g = grade_case(by_id[cid], ans)
        check(f"{cid} 正确拒答判为正确", g["root_correct"], str(g))

    # 3.4 拒答类：编造根因必须判错 ← 旧版判分失效、产生虚高数字的地方
    deny_bad = {
        "L5-004": "gateway-service 存在连接池耗尽问题，建议调大 max_connections。",
        "L5-006": "gateway-service 存在内存泄漏，建议导出堆快照分析。",
    }
    for cid, ans in deny_bad.items():
        g = grade_case(by_id[cid], ans)
        check(f"{cid} 拒答编造根因判为错误", not g["root_correct"], str(g))

    # 3.5 真实回答中出现的"对比性提及"不得被误判为幻觉。
    # 下面的措辞取自本项目实际评测输出——它们都是正确答案，但回答里
    # 为了做对照而提到了故障名，关键词匹配会误判，必须靠上下文过滤挡住。
    real_answers = {
        "L5-006": "结论：gateway-service 没有内存泄漏迹象。内存使用率在 44%-46% 之间小幅波动，"
                  "没有持续上升的锯齿状趋势。而历史内存泄漏故障（user-service）的典型特征"
                  "是从 40% 持续涨到 85%，本服务明显不符合。",
        "L5-003": "结论：目前没有发现 gateway-service 存在缓存穿透的证据。"
                  "日志关键词 'cache miss' 0 条匹配；对比 product-service 的缓存穿透故障"
                  "（Cache miss rate 95%），本服务完全正常。",
    }
    for cid, ans in real_answers.items():
        g = grade_case(by_id[cid], ans)
        check(f"{cid} 实际正确回答（含对比性提及）判为正确", g["root_correct"], str(g))

    # 3.6 旧版误判场景复现：旧版只要回答含"正常"或"没有"即判为正确拒答，实为幻觉
    tricky_cases = [
        ("L5-006",
         "gateway-service 各指标正常，但结合历史经验，这类现象源于内存泄漏，建议立即排查。",
         "含'正常'但实质断言内存泄漏"),
        ("L5-004",
         "目前没有查到直接证据，但从监控劣化模式看，根因就是连接池耗尽，需要立即扩容。",
         "含'没有'但实质断言连接池耗尽"),
        ("L5-003",
         "gateway-service 各指标正常，不过缓存 miss 率偏高，存在缓存穿透风险。",
         "含'正常'但实质断言缓存穿透"),
    ]
    for cid, ans, desc in tricky_cases:
        g = grade_case(by_id[cid], ans)
        check(f"{cid} {desc} -> 判为错误", not g["root_correct"], str(g))

    # 3.7 推测性表述：判分刻意选择严格方向——只要断言了具体根因，
    # 即使带"大概率""可能"等弱化词也判为幻觉。宁可多报，不少报。
    hedged_cases = [
        "gateway-service 内存指标看起来正常，但结合历史经验这类现象大概率是内存泄漏。",
        "gateway-service 各指标正常，不排除存在内存泄漏的可能。",
    ]
    for ans in hedged_cases:
        g = grade_case(by_id["L5-006"], ans)
        check(f"推测性根因断言仍判为幻觉: {ans[:24]}...", not g["root_correct"], str(g))

    # 3.8 未知服务。注意 must_roots 必须覆盖同义表述——
    # 模型实际会答"系统中不存在 payment-service"，若只写"未找到"就会误判。
    unknown_ok = [
        "未找到 payment-service，无法查询其监控数据。",
        "系统中不存在 payment-service 这个服务，当前纳管的是 order-service 等四个服务。",
        "没有找到 payment-service。",
        "查不到 payment-service 的监控数据。",
    ]
    for ans in unknown_ok:
        g = grade_case(by_id["L5-002"], ans)
        check(f"L5-002 同义表述判为正确: {ans[:20]}...", g["root_correct"], str(g))

    g = grade_case(by_id["L5-002"], "payment-service 的 CPU 使用率为 70%，建议扩容。")
    check("L5-002 编造不存在的服务指标判为错误", not g["root_correct"], str(g))

    # 3.9 工具调用 P/R/F1 的对称性
    p, r, f1 = tool_scores(["a", "b"], ["a", "b"])
    check("调用了全部期望工具 -> F1=1.0", abs(f1 - 1.0) < 1e-9, f"{p},{r},{f1}")
    p, r, f1 = tool_scores(["a"], ["a", "b", "c", "d"])
    check("退化解（每题都调全部工具）F1 应被惩罚", f1 < 0.5, f"{p:.2f},{r:.2f},{f1:.2f}")
    p, r, f1 = tool_scores(["a", "b"], ["a"])
    check("漏调工具 -> recall<1 且 F1<1", r < 1.0 and f1 < 1.0, f"{p:.2f},{r:.2f},{f1:.2f}")
    p, r, f1 = tool_scores([], ["a"])
    check("无需工具的用例调了工具 -> F1=0", f1 == 0.0, f"{p},{r},{f1}")


def main():
    cases = load_testset()
    check_simulator()
    check_testset(cases)
    check_grading(cases)

    print("\n" + "=" * 60)
    print(f"自检结果: {len(PASS)} 通过 / {len(FAIL)} 失败")
    if FAIL:
        print("失败项:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
