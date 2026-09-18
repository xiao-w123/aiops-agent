"""agent_v3：本方案的命令行演示入口（含证据校验）。

注意：本文件此前与 eval_v3.py 几乎逐行重复，且名为 agent 却在底部完整跑评测
并打印汇总，属于死代码。现已改为纯演示入口，评测请使用 eval_v3.py。

用法：python agent_v3.py
"""

from agent_core import build_verified_agent

QUESTION = "帮我看看 order-service 昨晚为什么响应变慢了"

if __name__ == "__main__":
    agent = build_verified_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    print("=" * 60)
    print("最终回答：")
    print(result["final_answer"])
    print("=" * 60)
    print(f"是否触发证据校验重答: {result['was_refined']}")
    print(f"校验耗时: {result.get('verify_latency', 0):.2f}s")
