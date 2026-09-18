"""运维工具集（薄封装）。

真实实现位于 simulator.py，本模块仅做转发，保持原有导入路径不变：
    from tools import query_prometheus, search_logs, get_deployment_record
"""

from simulator import (  # noqa: F401
    query_prometheus,
    search_logs,
    get_deployment_record,
    SERVICES,
)

__all__ = [
    "query_prometheus",
    "search_logs",
    "get_deployment_record",
    "SERVICES",
]
