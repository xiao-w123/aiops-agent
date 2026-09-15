import json

def query_prometheus(metric_name: str, service: str) -> str:
    """查询监控指标。metric_name 可以是 cpu_usage, memory_usage, response_time。"""
    mock_data = {
        "cpu_usage": {"service": service, "value": "95%", "time": "昨晚 23:00-01:00"},
        "memory_usage": {"service": service, "value": "60%", "time": "昨晚 23:00-01:00"},
        "response_time": {"service": service, "value": "3.2s", "time": "昨晚 23:00-01:00"},
    }
    return json.dumps(mock_data.get(metric_name, {"error": "无此指标"}))

def search_logs(keyword: str, service: str) -> str:
    """搜索服务日志。keyword 是搜索关键词，如 error, timeout, exception。"""
    mock_logs = [
        f"[{service}] 2024-01-15 23:12:01 ERROR Connection pool exhausted",
        f"[{service}] 2024-01-15 23:15:33 ERROR Timeout after 30000ms",
    ]
    return "\n".join(mock_logs) if keyword == "error" else f"未找到含'{keyword}'的日志"

def get_deployment_record(service: str) -> str:
    """查询服务的发布记录。"""
    return f"[{service}] 2024-01-15 22:50:00 发布了 v2.3.1，变更内容：优化连接池配置"