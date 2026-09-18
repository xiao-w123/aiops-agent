"""确定性仿真运维环境。

设计原则
--------
1. **故障与健康是两套数据**：order-service / user-service 处于故障态，
   product-service 处于健康态。"某个服务现在正常吗"这类问题因此有唯一正确答案。
2. **指标是时间序列**：不是单值快照，Agent 必须自己读趋势、比对故障窗口
   （如"内存 3 天内 40% -> 85%" 才是泄漏特征，单看 85% 无法定性）。
3. **证据需要交叉推理**：order-service 的 CPU 与响应时间同步劣化、内存正常，
   要结合发布记录才能定位到连接池，单一工具读不出根因。
4. **确定性**：无随机数，同一输入永远得到同一输出，保证实验可复现。

时间轴设定（全部为历史数据）
--------------------------
- order-service : 2024-01-15 晚 22:50 发布 v2.3.1，23:00-01:00 故障窗口
- product-service: 2024-02-14 发布 v1.8.0，无异常
- user-service   : 2024-03-18 发布 v3.0.0，03-18 至 03-20 内存持续爬升
"""

import json

# ----------------------------------------------------------------------------
# 服务定义
# ----------------------------------------------------------------------------
#
# 四个服务，故障状态刻意各不相同：
#   order-service   连接池耗尽
#   product-service 缓存穿透
#   user-service    内存泄漏
#   gateway-service 完全健康  <- 用于"该说正常时能否说正常"的边界测试
#
# gateway-service 的存在是必要的：若所有服务都带故障，就无法测试拒答能力，
# 因为任何"正常吗"的问题都只能回答"有问题"。

# 健康基线：所有服务在无故障时的正常水位
BASELINE = {
    "cpu": 12,
    "memory": 45,
    "response_time": 0.2,
}

# 故障窗口（历史数据所在的日期与时段）
ORDER_WINDOW = {"date": "2024-01-15", "start": "23:00", "end": "01:00"}
PRODUCT_WINDOW = {"date": "2024-02-15", "start": "02:00", "end": "03:00"}
USER_WINDOW = {"date": "2024-03-18", "start": "14:00", "end": "16:00"}
USER_LEAK_START = "2024-03-18"
USER_LEAK_END = "2024-03-20"

# 各服务在各时段的 CPU 时间序列：[(时刻, 百分比), ...]
CPU_SERIES = {
    "order-service": [
        ("22:00", 14),
        ("22:50", 16),   # 发布 v2.3.1
        ("23:00", 78),   # 故障窗口开始
        ("23:30", 92),
        ("00:00", 95),
        ("00:30", 94),
        ("01:00", 88),
        ("02:00", 15),   # 故障窗口结束，回落
        ("09:00", 13),
    ],
    "product-service": [
        ("01:00", 20),
        ("02:00", 68),   # 故障窗口开始，数据库被无效查询打满
        ("02:30", 83),
        ("03:00", 71),
        ("04:00", 19),   # 恢复
        ("09:00", 14),
    ],
    "user-service": [
        ("14:00", 22),
        ("15:00", 28),
        ("16:00", 31),
        ("16:30", 12),   # OOM 重启后回落
        ("17:00", 11),
    ],
    "gateway-service": [
        ("09:00", 13),
        ("12:00", 15),
        ("15:00", 14),
        ("18:00", 12),
        ("21:00", 11),
    ],
}


# 各服务响应时间时间序列（秒）
RESPONSE_SERIES = {
    "order-service": [
        ("22:00", 0.21),
        ("22:50", 0.22),
        ("23:00", 1.4),
        ("23:30", 2.8),
        ("00:00", 3.2),
        ("00:30", 3.1),
        ("01:00", 2.2),
        ("02:00", 0.24),
        ("09:00", 0.20),
    ],
    "product-service": [
        ("01:00", 0.22),
        ("02:00", 1.6),
        ("02:30", 2.4),
        ("03:00", 2.1),
        ("04:00", 0.25),
        ("09:00", 0.21),
    ],
    "user-service": [
        ("14:00", 0.9),
        ("15:00", 1.1),
        ("16:00", 1.2),
        ("16:30", 0.25),
        ("17:00", 0.22),
    ],
    "gateway-service": [
        ("09:00", 0.20),
        ("12:00", 0.22),
        ("15:00", 0.21),
        ("18:00", 0.19),
        ("21:00", 0.20),
    ],
}

# user-service 内存泄漏：跨天持续爬升，这是"泄漏"而非"高负载"的关键证据
MEMORY_LEAK_SERIES = [
    ("2024-03-18 10:00", 40),
    ("2024-03-18 22:00", 48),
    ("2024-03-19 10:00", 62),
    ("2024-03-19 22:00", 74),
    ("2024-03-20 10:00", 85),
    ("2024-03-20 16:20", 89),   # 触发 OOM
    ("2024-03-20 16:30", 46),   # 重启后回落
]

# 其余服务的内存为平稳序列（注意 product-service 内存正常，其故障是缓存穿透）
MEMORY_FLAT_SERIES = {
    "order-service": [("23:00", 58), ("23:30", 60), ("00:00", 60), ("01:00", 59)],
    "product-service": [("02:00", 55), ("02:30", 56), ("03:00", 55), ("04:00", 54)],
    "gateway-service": [("09:00", 44), ("12:00", 46), ("15:00", 45), ("18:00", 45), ("21:00", 44)],
}

# 发布记录
DEPLOYMENTS = {
    "order-service": [
        {"time": "2024-01-15 22:50", "version": "v2.3.1", "change": "优化连接池配置"},
        {"time": "2024-01-10 15:00", "version": "v2.2.0", "change": "修复分页 bug"},
    ],
    "product-service": [
        {"time": "2024-02-14 20:00", "version": "v1.8.0", "change": "新增商品推荐"},
    ],
    "user-service": [
        {"time": "2024-03-18 10:00", "version": "v3.0.0", "change": "重构用户中心"},
        {"time": "2024-03-05 09:00", "version": "v2.9.1", "change": "升级日志组件"},
    ],
    # gateway-service 刻意没有发布记录，用于测试"无记录"场景
}

# 日志库：按服务组织，(时间戳, 级别, 正文)
LOGS = {
    "order-service": [
        ("2024-01-15 22:00:11", "INFO", "Request received path=/api/order"),
        ("2024-01-15 23:12:01", "ERROR", "Connection pool exhausted max=50 active=50 waiting=37"),
        ("2024-01-15 23:15:33", "ERROR", "Timeout after 30000ms while acquiring connection"),
        ("2024-01-15 23:20:02", "WARN", "Request queue depth=128 exceeds threshold"),
        ("2024-01-15 23:40:19", "INFO", "GC pause 41ms young generation"),
        ("2024-01-16 02:00:03", "INFO", "Request received path=/api/order"),
    ],
    "product-service": [
        ("2024-02-15 01:00:04", "INFO", "Cache warmed up keys=12000"),
        ("2024-02-15 02:00:07", "INFO", "Cache warmed up keys=12000"),
        ("2024-02-15 02:15:00", "WARN", "Cache miss rate 95% for key prefix product:"),
        ("2024-02-15 02:18:22", "ERROR", "Slow query 4200ms SELECT * FROM product WHERE id=?"),
        ("2024-02-15 02:30:11", "ERROR", "DB connection acquire timeout after 5000ms"),
        ("2024-02-15 02:45:33", "WARN", "Cache miss rate 91% suspicious key scan detected"),
        ("2024-02-15 03:20:41", "INFO", "Cache hit ratio recovered to 0.94"),
        ("2024-02-15 09:00:00", "INFO", "Health check passed"),
    ],
    "user-service": [
        ("2024-03-19 10:00:00", "INFO", "Service started version=v3.0.0"),
        ("2024-03-20 15:40:12", "WARN", "Heap usage 85% after full GC, old gen not reclaimed"),
        ("2024-03-20 16:20:55", "WARN", "Memory usage 89% connection pool active=180 idle=3"),
        ("2024-03-20 16:21:30", "ERROR", "OOM killed: Java heap space"),
        ("2024-03-20 16:30:02", "INFO", "Service restarted, heap usage 46%"),
    ],
    # 健康服务的日志只有 INFO，用于拒答测试
    "gateway-service": [
        ("2024-03-21 09:00:00", "INFO", "Health check passed rps=1200 p99=0.19s"),
        ("2024-03-21 12:00:00", "INFO", "Health check passed rps=1350 p99=0.21s"),
        ("2024-03-21 15:00:00", "INFO", "Health check passed rps=1280 p99=0.20s"),
        ("2024-03-21 21:00:00", "INFO", "Health check passed rps=1100 p99=0.20s"),
    ],
}

# 日志关键词同义词表：让 Agent 用不同措辞检索也能命中
LOG_ALIASES = {
    "error": ["ERROR"],
    "错误": ["ERROR"],
    "warn": ["WARN"],
    "warning": ["WARN"],
    "timeout": ["Timeout", "timeout"],
    "超时": ["Timeout"],
    "connection": ["connection", "Connection"],
    "连接": ["connection", "Connection"],
    "pool": ["pool", "Pool"],
    "连接池": ["Connection pool", "connection pool", "connection pool active"],
    "oom": ["OOM"],
    "内存": ["Memory", "memory", "Heap", "heap"],
    "memory": ["Memory", "memory", "Heap", "heap"],
    "heap": ["Heap", "heap"],
    "cache": ["Cache", "cache"],
    "缓存": ["Cache", "cache"],
    "miss": ["miss"],
    "gc": ["GC"],
    "restart": ["restart"],
    "重启": ["restart"],
}

SERVICES = ["order-service", "product-service", "user-service", "gateway-service"]


# ----------------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------------

def _fmt_series(series, unit=""):
    return " -> ".join(f"{t} {v}{unit}" for t, v in series)


def _series_json(series, unit=""):
    return [{"time": t, "value": f"{v}{unit}"} for t, v in series]


def _window_of(service, date, clock):
    """判断某条日志属于哪个故障窗口，用于按时间筛选。"""
    if service == "order-service" and date == ORDER_WINDOW["date"] and ("22:50" <= clock or clock <= "02:00"):
        return "order"
    if service == "product-service" and date == PRODUCT_WINDOW["date"] and (
            PRODUCT_WINDOW["start"] <= clock <= "04:00"):
        return "product"
    if service == "user-service" and USER_LEAK_START <= date <= USER_LEAK_END:
        return "user"
    return "normal"


def _match_log(entry, keywords):
    """关键词是否命中某条日志（正文 + 级别）。"""
    text = f"{entry[1]} {entry[2]}"
    text_lower = text.lower()
    for kw in keywords:
        if kw in text or kw.lower() in text_lower:
            return True
    return False


def _expand_keyword(keyword):
    """把用户给的关键词扩展成同义词列表。"""
    kw = keyword.strip()
    if not kw:
        return []
    expanded = list(LOG_ALIASES.get(kw.lower(), []))
    if not expanded:
        expanded = list(LOG_ALIASES.get(kw, []))
    # 保留原词，保证未登记的同义词也能做子串匹配
    if kw not in expanded:
        expanded.append(kw)
    return expanded


# ----------------------------------------------------------------------------
# 对外接口：三个运维工具（保持原有函数签名不变）
# ----------------------------------------------------------------------------

def query_prometheus(metric_name: str, service: str) -> str:
    """查询监控指标。metric_name 可以是 cpu_usage, memory_usage, response_time。"""
    if service not in SERVICES:
        return json.dumps(
            {"error": f"未找到服务 {service}", "available_services": SERVICES},
            ensure_ascii=False,
        )

    if metric_name == "cpu_usage":
        series = CPU_SERIES.get(service)
        if series is None:
            series = [("now", BASELINE["cpu"])]
        return json.dumps({
            "service": service,
            "metric": metric_name,
            "unit": "%",
            "series": _series_json(series, "%"),
            "note": "波动超过 3 倍基线值即判定为异常，请结合响应时间与日志交叉判断",
        }, ensure_ascii=False)

    if metric_name == "response_time":
        series = RESPONSE_SERIES.get(service)
        if series is None:
            series = [("now", BASELINE["response_time"])]
        return json.dumps({
            "service": service,
            "metric": metric_name,
            "unit": "s",
            "series": _series_json(series, "s"),
            "note": "基线约 0.2s，持续高于 1s 即为劣化",
        }, ensure_ascii=False)

    if metric_name == "memory_usage":
        if service == "user-service":
            series = MEMORY_LEAK_SERIES
            return json.dumps({
                "service": service,
                "metric": metric_name,
                "unit": "%",
                "series": _series_json(series, "%"),
                "note": "该服务指标跨天采集，可观察长期趋势",
            }, ensure_ascii=False)
        series = MEMORY_FLAT_SERIES.get(service, [("now", BASELINE["memory"])])
        return json.dumps({
            "service": service,
            "metric": metric_name,
            "unit": "%",
            "series": _series_json(series, "%"),
            "note": "该服务指标跨天采集，可观察长期趋势",
        }, ensure_ascii=False)

    return json.dumps(
        {"error": f"无此指标 {metric_name}",
         "available_metrics": ["cpu_usage", "memory_usage", "response_time"]},
        ensure_ascii=False,
    )


def search_logs(keyword: str, service: str, window: str = "all") -> str:
    """搜索服务日志。

    keyword 是搜索关键词，如 error, timeout, exception, connection pool。
    window 可选 all / recent，recent 只返回该服务故障窗口内的日志。
    """
    if service not in LOGS:
        return json.dumps(
            {"error": f"未找到服务 {service} 的日志", "available_services": SERVICES},
            ensure_ascii=False,
        )

    expanded = _expand_keyword(keyword)
    entries = LOGS[service]

    if window == "recent":
        target = {"order-service": "order", "product-service": "product",
                  "user-service": "user"}.get(service, "normal")
        entries = [e for e in entries if _window_of(service, e[0][:10], e[0][11:16]) == target]

    matched = [e for e in entries if _match_log(e, expanded)]

    if not matched:
        return json.dumps(
            {"service": service, "keyword": keyword,
             "matched": 0, "logs": [],
             "message": f"未找到含'{keyword}'的日志"},
            ensure_ascii=False,
        )

    return json.dumps({
        "service": service,
        "keyword": keyword,
        "matched": len(matched),
        "logs": [f"[{service}] {ts} {lvl} {msg}" for ts, lvl, msg in matched],
    }, ensure_ascii=False)


def get_deployment_record(service: str) -> str:
    """查询服务的发布记录。"""
    if service not in DEPLOYMENTS:
        return json.dumps(
            {"error": f"未找到服务 {service} 的发布记录",
             "available_services": SERVICES},
            ensure_ascii=False,
        )
    records = DEPLOYMENTS[service]
    return json.dumps({
        "service": service,
        "records": [
            f"[{service}] {r['time']} 发布 {r['version']}（{r['change']}）"
            for r in records
        ],
    }, ensure_ascii=False)


# ----------------------------------------------------------------------------
# 自检：验证测试集的标准答案与本环境一致
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    checks = [
        ("order-service CPU 峰值应为 95", "95%" in query_prometheus("cpu_usage", "order-service")),
        ("order-service 无内存泄漏迹象", "OOM" not in query_prometheus("memory_usage", "order-service")),
        ("order-service 有连接池报错", "Connection pool exhausted" in search_logs("connection pool", "order-service")),
        ("order-service 无 OOM 日志", "OOM" not in search_logs("oom", "order-service")),
        ("product-service 有缓存穿透迹象", "Cache miss rate 95%" in search_logs("cache", "product-service")),
        ("product-service CPU 故障峰值 >60", "83%" in query_prometheus("cpu_usage", "product-service")),
        ("product-service 内存正常（非泄漏）", "55%" in query_prometheus("memory_usage", "product-service")),
        ("user-service 内存应为泄漏爬升", "40%" in query_prometheus("memory_usage", "user-service")
         and "85%" in query_prometheus("memory_usage", "user-service")),
        ("user-service 有 OOM 日志", "OOM killed" in search_logs("oom", "user-service")),
        ("gateway-service 全程健康", all(
            int(item["value"].rstrip("%")) < 50
            for item in json.loads(query_prometheus("cpu_usage", "gateway-service"))["series"])),
        ("gateway-service 无 error 日志", "未找到" in search_logs("error", "gateway-service")),
        ("gateway-service 无连接池问题", "未找到" in search_logs("connection pool", "gateway-service")),
        ("gateway-service 无内存泄漏关键词", "OOM" not in search_logs("oom", "gateway-service")),
        ("gateway-service 无发布记录", "未找到" in get_deployment_record("gateway-service")),
        ("未知服务应报错", "未找到服务" in query_prometheus("cpu_usage", "payment-service")),
    ]
    failed = 0
    for name, ok in checks:
        print(f"[{'OK ' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1
    print(f"\n自检结果: {len(checks) - failed}/{len(checks)} 通过")
