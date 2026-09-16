import json
import random

random.seed(42)  # 固定种子，保证可复现

# 不同 service 的真实数据
SERVICE_METRICS = {
    "order-service": {"cpu": "95%", "memory": "60%", "response_time": "3.2s"},
    "product-service": {"cpu": "40%", "memory": "55%", "response_time": "0.8s"},
    "user-service": {"cpu": "50%", "memory": "85%", "response_time": "1.2s"},
}

SERVICE_LOGS = {
    "order-service": [
        "[order-service] 23:12:01 ERROR Connection pool exhausted",
        "[order-service] 23:15:33 ERROR Timeout after 30000ms",
        "[order-service] 23:10:00 INFO Request received",  # 干扰项
    ],
    "product-service": [
        "[product-service] 02:15:00 ERROR Cache miss rate 95%",
        "[product-service] 02:20:00 INFO Cache warmed up",  # 干扰项
    ],
    "user-service": [
        "[user-service] 10:30:00 WARN Memory usage 85%",
        "[user-service] 10:35:00 ERROR OOM killed",
    ],
}

def query_prometheus(metric_name: str, service: str) -> str:
    """查询监控指标。metric_name 可以是 cpu_usage, memory_usage, response_time。"""
    if service not in SERVICE_METRICS:
        return json.dumps({"error": f"未找到服务 {service}"})
    metrics = SERVICE_METRICS[service]
    key_map = {"cpu_usage": "cpu", "memory_usage": "memory", "response_time": "response_time"}
    if metric_name not in key_map:
        return json.dumps({"error": "无此指标"})
    return json.dumps({
        "service": service,
        "metric": metric_name,
        "value": metrics[key_map[metric_name]],
    })

def search_logs(keyword: str, service: str) -> str:
    """搜索服务日志。keyword 是搜索关键词，如 error, timeout, exception。"""
    if service not in SERVICE_LOGS:
        return f"未找到服务 {service} 的日志"
    logs = SERVICE_LOGS[service]
    matched = [log for log in logs if keyword.lower() in log.lower()]
    return "\n".join(matched) if matched else f"未找到含'{keyword}'的日志"

def get_deployment_record(service: str) -> str:
    """查询服务的发布记录。"""
    records = {
        "order-service": "[order-service] 2024-01-15 22:50 发布 v2.3.1（优化连接池配置）\n[order-service] 2024-01-10 15:00 发布 v2.2.0（修复分页 bug）",
        "product-service": "[product-service] 2024-02-14 20:00 发布 v1.8.0（新增商品推荐）",
        "user-service": "[user-service] 2024-03-18 10:00 发布 v3.0.0（重构用户中心）",
    }
    return records.get(service, f"未找到服务 {service} 的发布记录")