# 智能运维 Agent

基于大模型和 RAG 的自动化故障排查系统。用户输入一句自然语言的运维问题，Agent 自动查监控、查日志、查发布记录、翻历史故障报告，输出结构化的根因分析。

## 项目背景

传统运维排查流程需要人工依次查看监控面板、搜索日志、核对发布记录、翻找历史故障报告，耗时长且容易遗漏。本项目将这套流程交给 AI Agent 自动完成，降低排查成本，缩短故障恢复时间。

## 核心功能

- 自动调用监控工具查询 CPU、内存、响应时间
- 自动搜索日志中的错误信息
- 自动查询服务的发布记录
- 基于 RAG 检索历史故障报告，引用历史经验
- 输出结构化根因分析报告（含证据链、根因、建议）
- 提供 Web 界面，支持浏览器交互演示
- 内置评测脚本，量化 Agent 的工具调用准确率和回答质量

## 技术栈

- 大模型：DeepSeek（deepseek-chat）
- Agent 框架：LangChain
- 向量数据库：Chroma
- Embedding 模型：BAAI/bge-small-zh-v1.5（本地运行，免费）
- 后端：FastAPI + Uvicorn
- 前端：原生 HTML + JavaScript

## 架构说明

Agent 采用 ReAct（Reasoning + Acting）模式，工作流程如下：

```
用户提问
   ↓
模型思考：需要哪些信息？
   ↓
调用工具（可多轮）：
  ├── query_prometheus      → 查监控
  ├── search_logs           → 查日志
  ├── get_deployment_record → 查发布记录
  └── search_fault_reports  → 查历史故障报告（RAG）
   ↓
观察结果 → 继续思考 → 直到信息足够
   ↓
输出结构化根因分析报告
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- 一个 DeepSeek API Key（[platform.deepseek.com](https://platform.deepseek.com/)）

### 2. 安装依赖

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langchain langchain-openai langgraph chromadb fastapi uvicorn python-dotenv langchain-community langchain-text-splitters sentence-transformers
```

### 3. 配置 API Key

在项目根目录创建 `.env` 文件，内容如下：

```
OPENAI_API_KEY=你的DeepSeekKey
OPENAI_API_BASE=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

### 4. 构建知识库

```bash
# 设置 HuggingFace 国内镜像（加速 Embedding 模型下载）
set HF_ENDPOINT=https://hf-mirror.com

python rag_setup.py
```

### 5. 启动 Web 服务

```bash
python -m uvicorn main:app --reload
```

打开浏览器访问 `http://127.0.0.1:8000`。

### 6. 命令行运行（可选）

```bash
python agent_v2.py
```

## 评测结果

使用 30 条分层测试用例进行评测（L1-L5，每层 6 条），覆盖从单工具查询到边界拒答的五个能力层级。

| 层级 | 能力 | 工具调用 | 关键词命中 | 根因定位 |
| :--- | :--- | :--- | :--- | :--- |
| L1 | 单工具事实查询 | 100% | 100% | 100% |
| L2 | 多工具联合推理 | 100% | 100% | 67% |
| L3 | 根因定位 | 100% | 94% | 67% |
| L4 | 历史经验迁移 | 83% | 100% | 83% |
| L5 | 边界与拒答 | 100% | 92% | 17% |
| **总体** | — | **97%** | **97%** | **67%** |

**关键发现**：L5（边界与拒答）层根因准确率仅为 17%，揭示 Agent 在无故障场景下存在**过度诊断倾向**——倾向于编造具体根因，而非承认"无异常"。这为后续引入"证据校验 Agent"提供了动机。

运行评测：

```bash
python eval_v2.py
```

## 项目结构

```
ai-agent/
├── .env                  # 环境变量（API Key）
├── knowledge/            # 历史故障报告
│   ├── fault_001.txt     # 连接池故障
│   ├── fault_002.txt     # 缓存穿透故障
│   └── fault_003.txt     # 内存泄漏故障
├── chroma_db/            # 向量数据库
├── tools.py              # 三个模拟运维工具
├── rag_setup.py          # 建知识库脚本
├── agent_v1.py           # 基础 Agent（无 RAG）
├── agent_v2.py           # 完整 Agent（含 RAG）
├── main.py               # FastAPI 后端
├── index.html            # 网页前端
├── eval.py               # 评测脚本（旧版）
├── eval_v2.py            # 评测脚本（分层测试集版）
├── testset.json          # 分层测试集（30 条）
├── test_llm.py           # 模型调用测试
└── README.md
```

## 示例输出

输入：

> 帮我看看 order-service 昨晚为什么响应变慢了

输出（节选）：

> **结论**：order-service 昨晚变慢的根本原因是 v2.3.1 的连接池配置变更导致数据库连接池耗尽。
>
> **证据链**：
> - 22:50 发布 v2.3.1，变更内容为"优化连接池配置"
> - 23:00-01:00 响应时间升至 3.2s，CPU 冲到 95%，内存正常
> - 23:12 日志报错 `Connection pool exhausted`
>
> **历史故障对照**：这与 2024-01-10 的故障报告几乎完全一致。上次同样是因为连接池最大连接数配置过小，当时将 max_connections 从 50 调整到 200 解决。本次是**重复踩坑**，建议纳入发布 checklist 强制卡点。

## 后续可扩展方向

- 引入证据校验 Agent，抑制无故障场景下的过度诊断
- 用 LangGraph 显式编排工作流，支持"检索不相关时重试"
- 引入多 Agent 协作（诊断 Agent + 报告 Agent）
- 对接真实的 Prometheus、ELK、发布系统 API
- 增加更多评测维度和测试用例