"""评测入口（可复现）。

用法
----
    # 单配置：本方案（含证据校验），重复 3 次
    python eval_v3.py --configs verified --repeats 3

    # 完整消融：基线与本方案各跑 3 次并生成对比表
    python eval_v3.py --configs both --repeats 3 --warmup 1

    # 先小样本验证链路是否正常，避免一上来就烧掉 60 次调用
    python eval_v3.py --configs both --limit 4 --repeats 1

参数
----
--configs   basic / verified / both
--repeats   每个用例重复次数（用于估计方差，建议 >= 3）
--warmup    预热次数，不计入统计（摊掉首次加载模型/建连的开销）
--limit     只跑前 N 条用例（调试用）
--level     只跑指定层级，如 --level L3,L5
--k         RAG 检索返回条数
--model     模型名
"""

import argparse
import sys

from evalframework import (
    CONFIG_BASIC, CONFIG_VERIFIED,
    evaluate, load_testset, render_ablation, render_summary,
    save_diagnostics, save_results, summarize,
)


def build_agent(config, model, k):
    from agent_core import build_basic_agent, build_verified_agent

    if config == CONFIG_BASIC:
        return build_basic_agent(model=model, k=k)
    # 关键：两组共用同一个基础 Agent 构造路径，唯一变量是证据校验
    return build_verified_agent(model=model, k=k)


def main():
    parser = argparse.ArgumentParser(description="智能运维 Agent 评测")
    parser.add_argument("--configs", default="both",
                        help="basic / verified / both，可用逗号分隔")
    parser.add_argument("--repeats", type=int, default=3, help="重复次数")
    parser.add_argument("--warmup", type=int, default=1, help="预热次数（不计入统计）")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条用例")
    parser.add_argument("--level", default="", help="只跑指定层级，如 L3,L5")
    parser.add_argument("--k", type=int, default=2, help="RAG 检索返回条数")
    parser.add_argument("--model", default="deepseek-chat", help="模型名")
    parser.add_argument("--outdir", default="results", help="结果输出目录")
    parser.add_argument("--workers", type=int, default=4,
                        help="并发线程数（复杂用例单次可达 40s，并发能显著压缩总耗时）")
    parser.add_argument("--quiet", action="store_true", help="不打印逐条日志")
    args = parser.parse_args()

    cases = load_testset()
    if args.level:
        wanted = {s.strip() for s in args.level.split(",") if s.strip()}
        cases = [c for c in cases if c.level in wanted]
    if args.limit:
        cases = cases[: args.limit]

    if not cases:
        print("没有匹配的测试用例，退出。")
        return 1

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if "both" in configs:
        configs = [CONFIG_BASIC, CONFIG_VERIFIED]

    print(f"模型: {args.model} | 检索 k={args.k} | 用例数: {len(cases)} | "
          f"重复: {args.repeats} | 预热: {args.warmup} | 并发: {args.workers}")
    print(f"配置: {', '.join(configs)}")
    sessions = len(cases) * args.repeats * len(configs)
    print(f"预计 Agent 会话数: {sessions}"
          f"（复杂用例单次可达 30-40s，并发 {args.workers} 时"
          f"预计墙钟时间约 {sessions * 12 / max(args.workers, 1) / 60:.0f} 分钟）\n")

    all_results = []
    for config in configs:
        print(f"\n{'=' * 60}\n开始评测配置: {config}\n{'=' * 60}")
        agent = build_agent(config, args.model, args.k)
        results = evaluate(
            agent, cases, config=config,
            repeats=args.repeats, warmup=args.warmup,
            verbose=not args.quiet, workers=args.workers,
        )
        all_results.extend(results)

    summary = summarize(all_results)
    print("\n" + "=" * 60)
    print("评测汇总（指标口径：均值 ± 标准差，跨重复次数统计）")
    print("=" * 60)
    print(render_summary(summary))
    print(render_ablation(summary))

    json_path, csv_path, md_path, latest = save_results(all_results, summary, args.outdir)
    diag_path = save_diagnostics(all_results, f"{args.outdir}/diagnostics.md")
    print(f"\n原始结果: {json_path}")
    print(f"CSV 明细: {csv_path}")
    print(f"报告: {md_path}")
    print(f"最新报告副本: {latest}")
    print(f"逐条诊断: {diag_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
