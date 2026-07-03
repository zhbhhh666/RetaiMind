# RetailMind 项目开发说明书

> 零售智能决策助手 — 完整开发技术文档

---

## 目录

- [一、项目概述](#一项目概述)
- [二、系统架构设计](#二系统架构设计)
- [三、技术选型说明](#三技术选型说明)
- [四、目录结构详解](#四目录结构详解)
- [五、核心模块开发详解](#五核心模块开发详解)
  - [5.1 配置模块 (config.py)](#51-配置模块-configpy)
  - [5.2 DeepSeek API 客户端 (deepseek_client.py)](#52-deepseek-api-客户端-deepseek_clientpy)
  - [5.3 数据加载模块 (data_loader.py)](#53-数据加载模块-data_loaderpy)
  - [5.4 会员分析引擎 (member_analyzer.py)](#54-会员分析引擎-member_analyzerpy)
  - [5.5 RAG 检索引擎 (rag_engine.py)](#55-rag-检索引擎-rag_enginepy)
  - [5.6 库存预警 Agent (agent_inventory.py)](#56-库存预警-agent-agent_inventorypy)
  - [5.7 运营日报生成器 (report_generator.py)](#57-运营日报生成器-report_generatorpy)
- [六、前端页面开发](#六前端页面开发)
- [七、Prompt 工程设计](#七prompt-工程设计)
- [八、数据模型设计](#八数据模型设计)
- [九、API 接口说明](#九api-接口说明)
- [十、开发时间表与里程碑](#十开发时间表与里程碑)
- [十一、测试策略](#十一测试策略)
- [十二、模型评估体系](#十二模型评估体系)
- [十三、扩展开发指南](#十三扩展开发指南)

---

## 一、项目概述

### 1.1 项目定位

RetailMind 是一个面向零售业务的智能决策助手，融合 **RAG（检索增强生成）、AI Agent（智能代理）、数据分析** 三大技术栈，同时覆盖会员运营和供应链两大核心业务场景。

### 1.2 核心功能

| 模块 | 功能描述 | 技术要点 |
|------|---------|---------|
| 数据管理中心 | 集中式数据上传、校验、预览与重置 | pandas、配置驱动校验、Streamlit 文件上传 |
| 会员数据洞察 | 自动分析会员消费行为，生成 RFM 分级与趋势洞察 | pandas、plotly、RFM 模型、手动规则分群 |
| 供应链知识库问答 | 基于 RAG 的供应链文档智能问答 | LangChain、ChromaDB、本地 Embedding |
| 库存预警 Agent | AI Agent 监控库存水位，执行预警并给出补货建议 | DeepSeek 原生 Tool Calling、多步推理 |
| 运营日报生成 | AIGC 自动生成运营日报与策略建议 | 结构化 Prompt、Markdown 输出 |

### 1.3 技术栈

| 组件 | 选型 | 版本要求 |
|------|------|---------|
| 编程语言 | Python | 3.10+ |
| Web 框架 | Streamlit | 1.30+ |
| LLM 框架 | LangChain（含 community / core / openai / text-splitters / classic 子包） | 0.1+ |
| 向量数据库 | ChromaDB | 0.4.22+ |
| Embedding 模型 | BAAI/bge-large-zh-v1.5 | sentence-transformers 2.2+ |
| 大模型 | DeepSeek API (deepseek-chat / deepseek-reasoner) | - |
| LLM 客户端 | OpenAI SDK（兼容 DeepSeek API） | 2.0+ |
| 数据处理 | pandas / numpy | 2.1+ / 1.26+ |
| 可视化 | Plotly | 5.18+ |
| 数据分析 | scikit-learn | 1.3+ |
| 环境变量管理 | python-dotenv | 1.0+ |
| HTTP 请求 | requests | 2.31+ |
| 测试框架 | pytest | 7.4+ |

---

## 二、系统架构设计

### 2.1 架构图

```
+------------------+     +------------------+     +------------------+
|   Streamlit UI   |<--->|   RetailMind     |<--->|  DeepSeek API    |
|  (前端交互层)     |     |   (核心业务层)    |     |  (大模型服务)     |
+------------------+     +------------------+     +------------------+
        |                       |                          |
        |                +------+------+                   |
        |                |             |                   |
        |          +-----v-----+ +-----v-----+             |
        |          |  Data     | |   RAG     |             |
        |          |  Analysis | |  Engine   |             |
        |          +-----------+ +-----------+             |
        |                |             |                   |
        |          +-----v-----+ +-----v-----+             |
        |          |  Agent    | |   AIGC    |             |
        |          |  Module   | |  Module   |             |
        |          +-----------+ +-----------+             |
        |                                                  |
        +------------------+-------------------------------+
                           |
                    +------v------+
                    |  ChromaDB   |
                    | (向量存储)   |
                    +-------------+
```

### 2.2 分层架构说明

| 层级 | 职责 | 对应文件 |
|------|------|---------|
| **表现层** | 用户交互、数据展示 | `app.py`、`pages/*.py` |
| **业务层** | 核心业务逻辑处理 | `core/*.py` |
| **数据层** | 数据加载与存储 | `data/`、`vector_db/` |
| **外部服务** | LLM 调用 | DeepSeek API |

### 2.3 数据流

```
用户输入 → Streamlit 页面 → 核心模块 → DeepSeek API / ChromaDB / pandas
         ← 结果展示 ← 核心模块 ← 响应数据
```

---

## 三、技术选型说明

### 3.1 为什么选择 DeepSeek？

| 对比维度 | DeepSeek | OpenAI |
|---------|----------|--------|
| API 格式 | 兼容 OpenAI SDK | 原生 |
| 中文效果 | 优异 | 良好 |
| 价格 | 显著更低（约 1/10） | 较高 |
| 推理能力 | deepseek-reasoner 强 | GPT-4 |
| Function Calling | 支持 | 支持 |
| Embedding | ❌ 不提供 | ✅ 提供 |

### 3.2 Embedding 替代方案

由于 DeepSeek 不提供 Embedding API，项目选用本地 Embedding 模型：

| 方案 | 模型 | 优势 | 劣势 |
|------|------|------|------|
| **当前方案** | BAAI/bge-large-zh-v1.5 | 免费、中文效果好、离线运行 | 首次下载 1.2GB、占用内存 |
| 备选方案 1 | 智谱 Embedding API | 云端无需下载 | 需额外 API Key、按量付费 |
| 备选方案 2 | 通义 Embedding API | 云端无需下载 | 需额外 API Key、按量付费 |

### 3.3 DeepSeek 模型选择

| 模型 | 适用场景 | 特点 |
|------|---------|------|
| deepseek-chat (V3) | 通用对话、问答、日报生成 | 速度快、成本低 |
| deepseek-reasoner (R1) | 复杂推理、Agent 决策 | 推理能力强、速度较慢 |

---

## 四、目录结构详解

```
retailmind/
├── app.py                      # Streamlit 主入口，首页展示
├── pages/
│   ├── 00_data_management.py   # 数据管理中心页面（上传/校验/重置）
│   ├── 01_member_analysis.py   # 会员分析页面
│   ├── 02_knowledge_qa.py      # 知识库问答页面
│   ├── 03_inventory_agent.py   # 库存预警 Agent 页面
│   └── 04_daily_report.py      # 运营日报页面
├── core/
│   ├── __init__.py             # 包初始化文件
│   ├── config.py               # 配置管理（API Key、模型参数）
│   ├── deepseek_client.py      # DeepSeek API 客户端封装（基于 OpenAI SDK）
│   ├── data_loader.py          # 数据加载、校验、上传与重置
│   ├── member_analyzer.py      # 会员分析引擎（RFM + 手动规则分群）
│   ├── rag_engine.py           # RAG 检索引擎
│   ├── agent_inventory.py      # 库存预警 Agent（DeepSeek 原生 Tool Calling）
│   └── report_generator.py     # 运营日报生成器
├── prompts/
│   ├── member_insight.txt      # 会员洞察 Prompt
│   ├── qa_system.txt           # 问答系统 Prompt
│   ├── agent_react.txt         # Agent Prompt 模板
│   └── daily_report.txt        # 日报生成 Prompt
├── data/
│   ├── raw/                    # 原始数据（CSV）
│   │   ├── inventory.csv
│   │   ├── member_transactions.csv
│   │   ├── sales_forecast.csv
│   │   └── operations_YYYY-MM-DD.csv  # 运营数据（按日期存储）
│   ├── processed/              # 处理后数据（运行时生成）
│   └── knowledge/              # 知识库文档（.txt）
│       ├── inventory_management.txt
│       ├── member_service.txt
│       ├── order_processing.txt
│       └── supply_chain_policy.txt
├── vector_db/                  # ChromaDB 存储目录（含 chroma.sqlite3）
├── .env                        # 环境变量（DEEPSEEK_API_KEY 等）
└── requirements.txt            # 依赖清单
```

---

## 五、核心模块开发详解

### 5.1 配置模块 (config.py)

**文件路径**：`core/config.py`

**职责**：集中管理所有配置项，包括 API Key、模型参数、路径配置。

**配置项说明**：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| DEEPSEEK_API_KEY | 从环境变量读取 | DeepSeek API 密钥 |
| DEEPSEEK_BASE_URL | https://api.deepseek.com/v1 | API 基础 URL |
| CHAT_MODEL | deepseek-chat | 通用对话模型 |
| REASONER_MODEL | deepseek-reasoner | 推理模型 |
| EMBEDDING_MODEL | BAAI/bge-large-zh-v1.5 | 本地 Embedding 模型 |
| VECTOR_DB_DIR | ./vector_db | 向量数据库存储路径 |
| TEMPERATURE_QA | 0.3 | 问答场景温度（偏低，保证准确性） |
| TEMPERATURE_AGENT | 0.1 | Agent 场景温度（最低，保证确定性） |
| TEMPERATURE_REPORT | 0.5 | 报告生成温度（适中，保证创造性） |
| MAX_TOKENS | 4096 | 最大生成 Token 数 |
| DATA_DIR | ./data | 数据根目录 |
| RAW_DATA_DIR | ./data/raw | 原始数据目录 |
| PROCESSED_DATA_DIR | ./data/processed | 处理后数据目录 |
| KNOWLEDGE_DIR | ./data/knowledge | 知识库文档目录 |

**完整代码**：

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

    CHAT_MODEL = "deepseek-chat"
    REASONER_MODEL = "deepseek-reasoner"

    EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"

    VECTOR_DB_DIR = "./vector_db"

    TEMPERATURE_QA = 0.3
    TEMPERATURE_AGENT = 0.1
    TEMPERATURE_REPORT = 0.5
    MAX_TOKENS = 4096

    DATA_DIR = "./data"
    RAW_DATA_DIR = "./data/raw"
    PROCESSED_DATA_DIR = "./data/processed"
    KNOWLEDGE_DIR = "./data/knowledge"
```

**开发说明**：
- 使用 `python-dotenv` 从 `.env` 文件加载环境变量，未配置时返回空字符串
- 配置项以类属性形式集中管理，便于全局引用
- 温度参数根据场景不同设置不同值（QA 0.3 / Agent 0.1 / Report 0.5）
- 数据路径配置支持灵活切换数据源目录

### 5.2 DeepSeek API 客户端 (deepseek_client.py)

**文件路径**：`core/deepseek_client.py`

**职责**：封装 DeepSeek API 调用，基于 OpenAI SDK 提供同步和流式两种调用方式，并支持原生 Tool Calling。

**类定义**：仅包含 `DeepSeekClient` 一个类。

#### 5.2.1 DeepSeekClient（基于 OpenAI SDK）

```python
from openai import OpenAI
from typing import Iterator, Any

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: list,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: list = None,
        tool_choice: str = None
    ) -> Any:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        return self.client.chat.completions.create(**kwargs)

    def chat_stream(self, messages: list, **kwargs) -> Iterator[str]:
        response = self.chat(messages, stream=True, **kwargs)
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

**特点**：
- 基于 OpenAI SDK，只需修改 `base_url` 和 `api_key` 即可对接 DeepSeek
- `chat()` 方法支持 `tools` 和 `tool_choice` 参数，用于原生 Function Calling
- `chat_stream()` 方法支持流式输出，适用于实时展示场景
- 被库存预警 Agent 模块复用，承担 Tool Calling 调用

**开发要点**：
- 仅在 `tools` 和 `tool_choice` 不为空时才加入请求参数，避免 API 报错
- 流式输出通过 `yield` 逐块返回 delta.content
- 可扩展支持 `deepseek-reasoner` 模型用于复杂推理场景

### 5.3 数据加载模块 (data_loader.py)

**文件路径**：`core/data_loader.py`

**职责**：数据加载、校验、上传、重置与概览，是所有业务模块的数据入口。

#### 5.3.1 核心架构：配置驱动的数据管理

采用 **DATA_TYPES 配置表** 驱动的设计模式，所有数据类型的定义、校验规则、文件路径都集中在一个字典中管理。新增数据类型只需修改配置，无需改动核心逻辑。

```python
DATA_TYPES = {
    "member_transactions": {
        "filename": "member_transactions.csv",
        "required_columns": ["customer_id", "last_purchase_date", "frequency", "monetary"],
        "unique_columns": ["customer_id"],
        "positive_columns": ["frequency", "monetary"],
        "date_columns": ["last_purchase_date"],
        "numeric_columns": ["frequency", "monetary"],
        "generator": "_generate_member_transactions",
        "overview_fields": ["customer_id", "frequency", "monetary"],
    },
    "inventory": { ... },
    "sales_forecast": { ... },
    "operations": { ... },
}
```

**设计优势**：
- **可扩展性**：新增数据类型只需在字典中添加配置项
- **可维护性**：所有规则集中管理，修改校验规则不影响代码逻辑
- **一致性**：所有数据类型使用同一套加载/校验/上传/重置流程

#### 5.3.2 数据加载方法

| 方法 | 数据类型 | 文件路径 | 模拟数据量 |
|------|---------|---------|-----------|
| `load_member_transactions()` | 会员交易数据 | data/raw/member_transactions.csv | 1000 条 |
| `load_inventory_data()` | 库存数据 | data/raw/inventory.csv | 100 个 SKU |
| `load_sales_forecast()` | 销售预测 | data/raw/sales_forecast.csv | 100 个 SKU |
| `load_daily_operations(date)` | 运营日报数据 | data/raw/operations_YYYY-MM-DD.csv | 1 天 |

**开发要点**：
- 采用"惰性生成"策略：文件存在则加载，不存在则生成并保存
- 模拟数据使用固定随机种子（`np.random.seed(42)`），保证可复现
- 数据字段设计参考真实零售业务场景
- 运营数据按日期存储，不同日期互不覆盖

#### 5.3.3 数据校验体系

**校验方法**：`validate_data(data_type: str, df: pd.DataFrame) -> Tuple[bool, List[str]]`

支持的校验规则（全部通过 DATA_TYPES 配置驱动）：

| 校验类型 | 配置键 | 说明 |
|---------|--------|------|
| 必填字段检查 | `required_columns` | 检查 DataFrame 是否包含所有必需列 |
| 唯一性检查 | `unique_columns` | 检查指定列是否存在重复值 |
| 正值检查 | `positive_columns` | 检查数值列是否全部 > 0 |
| 非负值检查 | `non_negative_columns` | 检查数值列是否全部 >= 0 |
| 数值类型检查 | `numeric_columns` | 检查列是否可转换为数值类型 |
| 日期格式检查 | `date_columns` | 检查日期列格式是否正确 |

**校验流程**：
1. 依次执行所有配置的校验规则
2. 累积所有错误信息（而非遇到第一个错误就停止）
3. 返回（是否通过，错误列表）元组
4. 前端一次性展示所有问题，减少用户反复试错

#### 5.3.4 数据上传与重置

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `upload_data(data_type, df)` | data_type, DataFrame | (success, message) | 上传数据，校验通过后保存 |
| `reset_data(data_type)` | data_type | (success, message) | 删除现有数据，重新生成模拟数据 |
| `get_data_overview(data_type, date)` | data_type, date(可选) | Dict | 获取数据概览统计信息 |

**上传安全机制**：
- 上传前先执行完整校验，校验失败不保存
- 校验通过后先展示预览，用户确认后才覆盖
- 重置功能提供一键回退到示例数据的能力

#### 5.3.5 数据概览

`get_data_overview()` 方法为每个数据类型生成关键统计指标，用于页面展示：

| 数据类型 | 概览指标 |
|---------|---------|
| 会员交易 | 会员数、平均频次、平均消费、平均最近购买天数 |
| 库存 | SKU 总数、低库存 SKU 数、平均库存、平均安全库存 |
| 销售预测 | SKU 总数、7 天总预测销量、平均预测销量 |
| 运营数据 | GMV、订单数、客单价、新增会员、复购率 |

**扩展开发**：
- 如需对接数据库，继承 `DataLoader` 类并重写加载方法
- 如需对接 API，在加载方法中添加 HTTP 请求逻辑
- 如需新增数据类型，在 `DATA_TYPES` 中添加配置并实现生成器函数

### 5.4 会员分析引擎 (member_analyzer.py)

**文件路径**：`core/member_analyzer.py`

**职责**：基于 RFM 模型分析会员消费行为，使用手动规则进行分群（不使用 KMeans 聚类）。

#### 5.4.1 RFM 模型说明

| 指标 | 全称 | 含义 | 计算方式 |
|------|------|------|---------|
| R | Recency | 最近购买距今天数 | `today - last_purchase_date` |
| F | Frequency | 购买频次 | `count(customer_id)` |
| M | Monetary | 消费总金额 | `sum(monetary)` |

**评分方法**：使用 `_safe_qcut()` 方法（封装 `pd.qcut()`）将每个指标分为 5 个等级（1-5 分）。`_safe_qcut` 处理重复值和异常情况，确保分箱稳健：

```python
def _safe_qcut(self, series: pd.Series, q: int = 5, labels: list = None, ascending: bool = True) -> pd.Series:
    n_unique = series.nunique()
    if n_unique <= 1:
        return pd.Series([3] * len(series), index=series.index).astype(int)

    actual_q = min(q, n_unique)
    if labels:
        labels = labels[:actual_q]

    try:
        result = pd.qcut(series, q=actual_q, labels=labels, duplicates="drop").astype(int)
    except ValueError:
        ranks = series.rank(method="dense", ascending=ascending)
        max_rank = ranks.max()
        bins = np.linspace(0, max_rank, actual_q + 1)
        result = pd.cut(ranks, bins=bins, labels=labels, include_lowest=True).astype(int)

    return result
```

#### 5.4.2 手动规则分群

不使用 KMeans 算法，而是基于 RFM 分数与平均值比较的规则进行分群。`segment_members()` 方法的分群规则：

| 群体 ID | 名称 | 分群规则 |
|---------|------|---------|
| 0 | 高价值会员 | `recency < 90` 且 `frequency > avg` 且 `monetary > avg` |
| 2 | 新会员 | `recency < 90` 且 `frequency <= avg` |
| 3 | 流失风险会员 | `recency >= 90` 且 `frequency > avg` 且 `monetary > avg` |
| 4 | 沉睡会员 | `recency >= 90` 且 `frequency <= avg` |
| 1 | 潜力会员 | 不满足以上条件的其他会员 |

**分群核心代码**：

```python
def segment_members(self, rfm_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    avg_frequency = rfm_df["frequency"].mean()
    avg_monetary = rfm_df["monetary"].mean()

    def assign_segment(row):
        recency = row["recency"]
        frequency = row["frequency"]
        monetary = row["monetary"]

        if recency < 90 and frequency > avg_frequency and monetary > avg_monetary:
            return "高价值会员"
        elif recency < 90 and frequency <= avg_frequency:
            return "新会员"
        elif recency >= 90 and frequency > avg_frequency and monetary > avg_monetary:
            return "流失风险会员"
        elif recency >= 90 and frequency <= avg_frequency:
            return "沉睡会员"
        else:
            return "潜力会员"

    rfm_df["segment_name"] = rfm_df.apply(assign_segment, axis=1)
    rfm_df["segment"] = rfm_df["segment_name"].map({
        "高价值会员": 0,
        "潜力会员": 1,
        "新会员": 2,
        "流失风险会员": 3,
        "沉睡会员": 4
    })

    return rfm_df
```

#### 5.4.3 核心方法

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `calculate_rfm()` | transactions_df | rfm_df | 计算 RFM 指标和评分（含 recency_score/frequency_score/monetary_score） |
| `_safe_qcut()` | series, q, labels, ascending | Series | 健壮的分箱方法，处理重复值和异常 |
| `segment_members()` | rfm_df | rfm_df (含 segment_name、segment) | 基于规则的手动分群 |
| `get_segment_stats()` | rfm_df | Dict | 各分群统计信息（数量、占比、均值等） |
| `get_churn_risk()` | rfm_df | DataFrame | 获取流失风险会员（recency >= risk_days） |
| `get_segment_distribution()` | rfm_df | Tuple[List[str], List[int]] | 分群分布（标签和数量） |
| `analyze()` | transactions_df | Dict | 完整分析流程主入口 |

**开发要点**：
- `analyze()` 是主入口方法，串联 calculate_rfm → segment_members → get_segment_stats → get_churn_risk → get_segment_distribution 完整流程
- 采用手动规则分群（基于平均值比较），可解释性强，不依赖 KMeans
- `_safe_qcut` 处理 `pd.qcut` 在重复值场景下的异常，保证分箱稳健
- `risk_days=90` 为流失判断阈值，可在 `get_churn_risk()` 中配置
- 分群标签同时提供 `segment_name`（中文名）和 `segment`（数字 ID）两种形式

### 5.5 RAG 检索引擎 (rag_engine.py)

**文件路径**：`core/rag_engine.py`

**职责**：构建向量知识库，支持基于 RAG 的智能问答。

#### 5.5.1 处理流程

```
文档加载（手动 os.walk 遍历） → 文本分割 → Embedding 编码 → 存入 ChromaDB
                                                                       ↓
用户提问 → Embedding 编码 → 向量检索 → 上下文拼接 → LLM 生成答案（invoke 调用）
```

#### 5.5.2 核心组件

| 组件 | 选型 | 配置 |
|------|------|------|
| 文档加载 | 手动 `os.walk` 遍历 + `TextLoader` | 兼容 Windows，支持 .txt、.md 格式 |
| 文本分割器 | RecursiveCharacterTextSplitter | chunk_size=500, chunk_overlap=50 |
| Embedding | HuggingFaceEmbeddings | BAAI/bge-large-zh-v1.5, device=cpu |
| HF 镜像加速 | `HF_ENDPOINT=https://hf-mirror.com` | 国内下载加速 |
| 向量存储 | Chroma | persist_directory=./vector_db |
| LLM | ChatOpenAI（兼容 DeepSeek） | base_url=https://api.deepseek.com/v1, model=deepseek-chat |
| 检索方式 | MMR (Max Marginal Relevance) | k=5, fetch_k=20 |
| 问答链 | RetrievalQA | chain_type="stuff", return_source_documents=True |

**模块头部环境配置**：

```python
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
```

#### 5.5.3 文档加载（手动遍历）

不使用 `DirectoryLoader`，改为手动 `os.walk` 遍历目录，逐文件调用 `TextLoader` 加载，兼容 Windows 路径：

```python
def load_documents(self, docs_dir: str) -> List:
    documents = []

    for root, dirs, files in os.walk(docs_dir):
        for file in files:
            if file.endswith('.txt') or file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    print(f'加载文件失败 {file_path}: {e}')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", " "]
    )
    chunks = splitter.split_documents(documents)
    return chunks
```

#### 5.5.4 核心方法

| 方法 | 说明 |
|------|------|
| `load_documents(docs_dir)` | 手动遍历目录加载并分割知识库文档 |
| `build_vectorstore(chunks)` | 构建向量数据库并持久化 |
| `load_existing_vectorstore()` | 加载已有向量库 |
| `has_vectorstore()` | 检查向量库是否有效（含 chroma.sqlite3 且文档数 > 0） |
| `get_document_count()` | 获取当前向量库文档数量 |
| `create_qa_chain(prompt_template)` | 创建问答链（基于 RetrievalQA） |
| `query(question)` | 执行问答查询（使用 invoke() 方法） |

#### 5.5.5 has_vectorstore() 有效性检查

`has_vectorstore()` 方法用于判断向量库是否已有效构建，避免空库查询：

```python
def has_vectorstore(self) -> bool:
    if self.vectorstore:
        return self.get_document_count() > 0
    if os.path.exists(os.path.join(self.db_dir, "chroma.sqlite3")):
        try:
            temp_vs = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
            count = temp_vs._collection.count()
            return count > 0
        except:
            return False
    return False
```

#### 5.5.6 query() 方法（使用 invoke）

`query()` 方法使用 LangChain 的 `invoke()` 接口（而非已废弃的 `__call__`）：

```python
def query(self, question: str) -> Dict:
    if not self.qa_chain:
        raise ValueError("请先调用 create_qa_chain()")

    result = self.qa_chain.invoke({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.metadata for doc in result["source_documents"]]
    }
```

#### 5.5.7 MMR 检索策略

使用 MMR（最大边际相关性）检索，兼顾相关性和多样性：

```python
search_type="mmr",
search_kwargs={"k": 5, "fetch_k": 20}
```

- `fetch_k=20`：先检索 20 个相关文档
- `k=5`：从中选出 5 个最相关且不重复的文档

### 5.6 库存预警 Agent (agent_inventory.py)

**文件路径**：`core/agent_inventory.py`

**职责**：基于 DeepSeek 原生 Tool Calling 的 AI Agent，通过工具调用完成库存监控和补货建议。**不使用 LangChain Agent**，而是直接通过 `DeepSeekClient` 调用 DeepSeek API，手动处理 `tool_calls` 响应。

#### 5.6.1 工作流程

```
用户提问 → 构造 tools 列表 → DeepSeek API 调用（tool_choice=auto）
                                         ↓
                              判断 response.tool_calls 是否存在
                                         ↓
                         是 → 执行各工具函数 → 拼接观察结果 → 二次调用 API 生成最终答案
                         否 → 直接返回 response.content
```

#### 5.6.2 工具定义

不使用 LangChain 的 `BaseTool` 类，而是以字典形式定义工具元信息，并实现对应的 Python 函数：

| 工具名称 | 功能 | 输入 | 输出 |
|---------|------|------|------|
| `inventory_query` | 查询指定 SKU 的库存信息 | sku_id | 当前库存、安全库存、最近 7 天销量 |
| `sales_forecast` | 预测指定 SKU 未来销量 | sku_id | 未来 7 天预测销量 |
| `list_low_stock_items` | 列出所有低于安全库存的商品 | 无 | 短缺商品列表及缺口数量 |
| `list_all_skus` | 获取所有 SKU 编号列表 | 无 | SKU 总数及编号 |

**工具定义代码**：

```python
self.tools = {
    "inventory_query": {
        "description": "查询指定 SKU 的当前库存数量、安全库存线和最近7天销量",
        "function": self.query_inventory
    },
    "sales_forecast": {
        "description": "预测指定 SKU 未来7天的销量",
        "function": self.query_forecast
    },
    "list_low_stock_items": {
        "description": "查询所有库存低于安全库存线的商品列表，返回需要补货的商品信息",
        "function": self.list_low_stock_items
    },
    "list_all_skus": {
        "description": "获取所有 SKU 编号列表",
        "function": self.list_all_skus
    }
}
```

#### 5.6.3 Tool Calling 执行逻辑

`run()` 方法构造 OpenAI 兼容的 `tools` 参数（含 JSON Schema），通过 `DeepSeekClient.chat()` 调用 API，并手动处理 `tool_calls` 响应：

```python
def run(self, query: str) -> Dict:
    # 1. 构造 tools 列表（OpenAI Function Calling 格式）
    tool_list = []
    for name, info in self.tools.items():
        params = {"type": "object", "properties": {}, "required": []}
        if name == "inventory_query" or name == "sales_forecast":
            params["properties"]["sku_id"] = {"type": "string", "description": "SKU 编号"}
            params["required"] = ["sku_id"]
        tool_list.append({
            "type": "function",
            "function": {"name": name, "description": info["description"], "parameters": params}
        })

    # 2. 第一次调用，让模型决定调用哪些工具
    messages = [
        {"role": "system", "content": "你是零售库存管理专家。请使用提供的工具帮助用户解决库存问题。"},
        {"role": "user", "content": query}
    ]
    response = self.client.chat(
        messages=messages, model="deepseek-chat",
        temperature=0.1, max_tokens=4096,
        tools=tool_list, tool_choice="auto"
    )
    result = response.choices[0].message

    # 3. 若模型决定调用工具，则执行工具并二次调用生成最终答案
    if result.tool_calls:
        observations = []
        for tool_call in result.tool_calls:
            func_name = tool_call.function.name
            args_str = tool_call.function.arguments
            # 解析参数并执行对应函数...
            observations.append(f"工具调用: {func_name}({sku_id}) -> {observation}")

        messages.append({"role": "assistant", "content": result.content})
        messages.append({
            "role": "user",
            "content": "以下是工具执行结果，请基于这些信息给出最终答案：\n" + "\n".join(observations)
        })
        final_response = self.client.chat(
            messages=messages, model="deepseek-chat",
            temperature=0.1, max_tokens=4096
        )
        return {"output": final_response.choices[0].message.content, "intermediate_steps": observations}
    else:
        return {"output": result.content, "intermediate_steps": []}
```

#### 5.6.4 Agent 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| model | deepseek-chat | 使用 V3 模型 |
| temperature | 0.1 | 低温度保证确定性 |
| max_tokens | 4096 | 最大生成 Token 数 |
| tool_choice | auto | 由模型自动决定是否调用工具 |
| 工具数量 | 4 | inventory_query、sales_forecast、list_low_stock_items、list_all_skus |

**开发要点**：
- 不依赖 LangChain Agent，直接使用 DeepSeek 原生 Tool Calling，更轻量、可控
- 工具参数 schema 通过 JSON 格式手动构造，支持无参工具（list_low_stock_items、list_all_skus）
- 工具执行结果通过 `eval(args_str)` 解析参数（生产环境建议改用 `json.loads`）
- 返回结果包含 `output`（最终答案）和 `intermediate_steps`（工具调用记录）

### 5.7 运营日报生成器 (report_generator.py)

**文件路径**：`core/report_generator.py`

**职责**：基于业务数据和 Prompt 模板，调用 LLM 生成结构化运营日报。使用 LangChain 的 `ChatOpenAI`（通过 `base_url` 兼容 DeepSeek API）和 `PromptTemplate`。

#### 5.7.1 生成流程

```
业务数据 → PromptTemplate 模板填充 → ChatOpenAI.invoke() → Markdown 日报
```

#### 5.7.2 核心组件

| 组件 | 选型 | 配置 |
|------|------|------|
| LLM | `langchain_openai.ChatOpenAI` | base_url=https://api.deepseek.com/v1, model=deepseek-chat |
| Prompt 模板 | `langchain_core.prompts.PromptTemplate` | `from_template()` 构造 |
| 温度 | 0.5 | 适中，保证创造性 |
| 调用方式 | `llm.invoke(prompt_text)` | 返回 `.content` |

**初始化代码**：

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class DailyReportGenerator:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.5
        )
        self.prompt = PromptTemplate.from_template("""...""")
```

#### 5.7.3 Prompt 模板设计

日报 Prompt 使用 `PromptTemplate.from_template()` 直接从字符串构造，包含以下变量：

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| report_date | 报告日期 | 2026-07-01 |
| gmv | 销售额 | 1256800 |
| gmv_mom | 销售额环比 | 5.2 |
| orders | 订单量 | 8920 |
| orders_mom | 订单量环比 | 3.8 |
| aov | 客单价 | 140.9 |
| aov_mom | 客单价环比 | 1.4 |
| new_members | 新增会员数 | 234 |
| new_members_mom | 新增会员环比 | -2.1 |
| repurchase_rate | 复购率 | 32.5 |
| repurchase_mom | 复购率环比 | 1.8 |
| top_categories | 品类 TOP3 | 食品、服装、数码 |
| abnormal_events | 异常事件 | 库存预警、系统延迟等 |

#### 5.7.4 generate() 方法

```python
def generate(self, data: Dict) -> str:
    prompt_text = self.prompt.format(**data)
    response = self.llm.invoke(prompt_text)
    return response.content
```

#### 5.7.5 日报结构

生成的日报包含 5 个章节：

1. **昨日概览** — 一句话总结
2. **核心指标解读** — 指标变化原因分析
3. **品类洞察** — TOP3 品类表现
4. **风险提示** — 异常事件预警
5. **今日策略建议** — 3-5 条可执行建议

---

## 六、前端页面开发

### 6.1 Streamlit 多页面机制

Streamlit 通过 `pages/` 目录实现多页面应用，文件名前缀决定导航顺序：

```
pages/
├── 00_data_management.py   → 导航栏第 1 项（数据管理）
├── 01_member_analysis.py   → 导航栏第 2 项
├── 02_knowledge_qa.py      → 导航栏第 3 项
├── 03_inventory_agent.py   → 导航栏第 4 项
└── 04_daily_report.py      → 导航栏第 5 项
```

### 6.2 页面开发规范

**页面结构模板**：

```python
import streamlit as st
from core.config import Config

# 1. 页面配置
st.set_page_config(page_title="页面标题", page_icon="图标", layout="wide")

# 2. API Key 检查（如需要）
api_key = Config.DEEPSEEK_API_KEY
if not api_key:
    st.error("请先配置 DeepSeek API Key！")
    st.stop()

# 3. 页面标题
st.title("图标 页面标题")
st.subheader("副标题")

# 4. 数据加载
# 5. 交互组件
# 6. 结果展示
```

### 6.3 页面功能详解

| 页面 | 文件 | 核心组件 | 交互方式 |
|------|------|---------|---------|
| 数据管理中心 | 00_data_management.py | 文件上传、数据预览、指标卡片、标签页 | 上传/校验/重置 |
| 会员分析 | 01_member_analysis.py | 饼图、散点图、数据表 | 按钮触发分析 |
| 知识问答 | 02_knowledge_qa.py | 文本输入、Markdown 渲染 | 输入问题实时回答 |
| 库存 Agent | 03_inventory_agent.py | 直方图、数据表、文本输入 | 输入问题 Agent 回答 |
| 运营日报 | 04_daily_report.py | 指标卡片、日期选择器 | 选择日期后生成 |

### 6.4 数据管理中心页面设计

**文件路径**：`pages/00_data_management.py`

**页面结构**：
- 四标签页布局：会员交易 / 库存 / 销售预测 / 运营数据
- 每个标签页包含：数据状态指标、数据概览、数据预览、上传区域、重置按钮

**核心交互流程**：

```
用户上传 CSV
    ↓
读取文件 → 调用 DataLoader.validate_data()
    ↓
校验通过？──否──→ 显示错误列表，用户修改后重传
    ↓ 是
显示数据预览 + 确认上传按钮
    ↓ 用户点击确认
调用 DataLoader.upload_data() 保存文件
    ↓
刷新页面，显示新数据状态
```

**设计要点**：
- 校验失败时显示所有错误，而非逐个提示
- 上传前必须用户确认，防止误操作覆盖
- 重置按钮提供一键回退能力
- 使用 st.tabs() 组织多类型数据管理

### 6.5 缓存使用

知识库问答页面使用 `@st.cache_resource` 缓存 RAG 引擎，避免每次交互重复加载模型：

```python
@st.cache_resource
def get_rag_engine():
    rag = SupplyChainRAG(api_key=api_key, db_dir=Config.VECTOR_DB_DIR)
    # ...
    return rag
```

---

## 七、Prompt 工程设计

### 7.1 Prompt 文件管理

所有 Prompt 模板存放在 `prompts/` 目录，与代码分离，便于迭代优化。

| 文件 | 用途 | 变量 |
|------|------|------|
| member_insight.txt | 会员洞察生成 | total_members, vip_ratio, ... |
| qa_system.txt | RAG 问答 | context, question |
| agent_react.txt | Agent 系统提示模板 | system 角色 content |
| daily_report.txt | 日报生成 | gmv, orders, aov, ... |

### 7.2 Prompt 设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **角色定义** | 明确 AI 的角色定位 | "你是一名资深零售运营分析师" |
| **任务明确** | 清晰描述任务目标 | "请生成 Markdown 格式的运营日报" |
| **上下文充足** | 提供充分的背景信息 | 提供完整的业务数据 |
| **格式约束** | 指定输出格式 | "使用 Markdown 标题和列表" |
| **质量要求** | 明确质量标准 | "策略建议需具体、可执行" |

### 7.3 Agent Prompt 设计

由于本项目采用 **DeepSeek 原生 Tool Calling**（不使用 LangChain ReAct Agent），Agent 不再需要 `{tools}`、`{tool_names}`、`{agent_scratchpad}` 等占位符。Agent 的 system 提示通过 messages 直接传入：

```python
messages = [
    {
        "role": "system",
        "content": "你是零售库存管理专家。请使用提供的工具帮助用户解决库存问题。"
    },
    {
        "role": "user",
        "content": query
    }
]
```

工具描述通过 `tools` 参数（JSON Schema 格式）传给 API，由模型自动决定调用哪个工具，无需在 Prompt 中显式列出工具清单。`prompts/agent_react.txt` 作为可扩展的提示模板保留，可在其中定义更详细的 Agent 角色与行为约束。

---

## 八、数据模型设计

### 8.1 会员交易数据 (member_transactions.csv)

| 字段名 | 数据类型 | 说明 | 示例 |
|--------|---------|------|------|
| customer_id | str | 会员唯一 ID | CUST0001 |
| last_purchase_date | datetime | 最近购买日期 | 2026-06-15 |
| frequency | int | 购买频次 | 12 |
| monetary | float | 消费金额 | 580.50 |
| preferred_category | str | 偏好品类 | 食品 |

### 8.2 库存数据 (inventory.csv)

| 字段名 | 数据类型 | 说明 | 示例 |
|--------|---------|------|------|
| sku_id | str | SKU 唯一 ID | SKU000001 |
| stock_qty | int | 当前库存 | 45 |
| safety_stock | int | 安全库存线 | 50 |
| weekly_sales | int | 最近 7 天销量 | 30 |
| category | str | 品类 | 食品 |

### 8.3 销售预测 (sales_forecast.csv)

| 字段名 | 数据类型 | 说明 | 示例 |
|--------|---------|------|------|
| sku_id | str | SKU 唯一 ID | SKU000001 |
| forecast_7d | int | 未来 7 天预测销量 | 55 |

### 8.4 运营日报数据

| 字段名 | 数据类型 | 说明 |
|--------|---------|------|
| report_date | str | 报告日期 |
| gmv | float | 销售额（元） |
| orders | int | 订单量 |
| aov | float | 客单价 |
| new_members | int | 新增会员数 |
| repurchase_rate | float | 复购率（%） |
| *_mom | float | 各指标环比变化（%） |
| top_categories | str | 品类 TOP3 |
| abnormal_events | str | 异常事件列表 |

---

## 九、API 接口说明

### 9.1 DeepSeek Chat API

**请求**：
```
POST https://api.deepseek.com/v1/chat/completions
Authorization: Bearer {API_KEY}
Content-Type: application/json

{
  "model": "deepseek-chat",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "temperature": 0.3,
  "max_tokens": 4096,
  "stream": false
}
```

**响应**：
```json
{
  "id": "chatcmpl-xxx",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "你好！有什么可以帮助您的吗？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 15,
    "total_tokens": 20
  }
}
```

### 9.2 模型参数推荐

| 场景 | 模型 | Temperature | Max Tokens | 说明 |
|------|------|-------------|------------|------|
| RAG 问答 | deepseek-chat | 0.3 | 4096 | 低温度保证准确 |
| Agent 推理 | deepseek-chat | 0.1 | 4096 | 最低温度保证确定 |
| 日报生成 | deepseek-chat | 0.5 | 4096 | 适中温度保证创意 |
| 复杂推理 | deepseek-reasoner | 0.1 | 8192 | R1 模型用于难题 |

### 9.3 费用估算

| DeepSeek 模型 | 输入价格 | 输出价格 |
|---------------|---------|---------|
| deepseek-chat (V3) | 0.5 元/百万 tokens | 1 元/百万 tokens |
| deepseek-reasoner (R1) | 4 元/百万 tokens | 16 元/百万 tokens |

> 示例：一次问答约消耗 1000 input + 500 output tokens，费用约 0.001 元。

---

## 十、开发时间表与里程碑

| 阶段 | 天数 | 任务 | 交付物 | 状态 |
|------|------|------|--------|------|
| Day 1-2 | 2 | 项目脚手架搭建、环境配置 | 可运行的空项目 + API 连通验证 | ✅ 已完成 |
| Day 3-5 | 3 | 会员数据分析模块开发 | 会员分析页面 + 测试数据 | ✅ 已完成 |
| Day 6-8 | 3 | 供应链 RAG 模块开发 | 知识库问答页面 + 测试文档 | ✅ 已完成 |
| Day 9-11 | 3 | 库存预警 Agent 开发 | 库存预警页面 + Agent 演示 | ✅ 已完成 |
| Day 12-13 | 2 | 运营日报生成模块 | 日报生成页面 + 示例报告 | ✅ 已完成 |
| Day 14-15 | 2 | 模型评估与优化 | 评估报告 | 🔲 待开发 |
| Day 16-17 | 2 | UI 美化与交互优化 | 完整可演示的 Web 应用 | 🔲 待开发 |
| Day 18-19 | 2 | 文档撰写 | 完整项目文档 | 🔲 待开发 |
| Day 20 | 1 | GitHub 仓库整理 | 规范的 GitHub 仓库 | 🔲 待开发 |

---

## 十一、测试策略

### 11.1 测试分层

| 测试类型 | 范围 | 工具 | 文件位置 | 用例数 |
|---------|------|------|---------|--------|
| 单元测试 | 核心模块函数 | pytest | `retailmind/tests/` | 179 |
| 集成测试 | 模块间协作 | pytest | `retailmind/tests/` | 含于单元测试 |
| 手动测试 | 前端交互 | 人工 | - | - |

### 11.2 测试文件结构

```
retailmind/tests/
├── conftest.py              # pytest 配置与 fixtures
├── test_config.py           # 配置模块测试
├── test_data_loader.py      # 数据加载模块测试（39个用例）
├── test_deepseek_client.py  # DeepSeek 客户端测试
├── test_member_analyzer.py  # 会员分析模块测试
├── test_rag_engine.py       # RAG 引擎测试
├── test_agent_inventory.py  # 库存 Agent 测试
├── test_report_generator.py # 日报生成器测试
├── test_logger.py           # 日志模块测试
├── test_rag_evaluator.py    # RAG 评估模块测试
├── test_agent_evaluator.py  # Agent 评估模块测试
└── test_eval_dashboard.py   # 评估可视化仪表盘测试
```

### 11.3 数据加载模块测试重点

数据加载模块是测试覆盖最全面的模块，包含以下测试类别：

| 测试类别 | 测试内容 |
|---------|---------|
| 初始化测试 | 默认目录、空目录校验 |
| 数据加载测试 | 四类数据的加载、自动生成、异常处理 |
| 数据校验测试 | 必填字段、唯一性、正负值、日期格式、数值类型 |
| 数据上传测试 | 正常上传、校验失败处理、非法数据拦截 |
| 数据重置测试 | 重置功能、非法类型处理 |
| 数据概览测试 | 四类数据概览查询、非法类型处理 |
| 确定性测试 | 同日期同数据、不同日期不同数据 |

### 11.4 单元测试示例

**测试数据校验功能**：

```python
# tests/test_data_loader.py - 数据校验测试示例
import pytest
import pandas as pd
from core.data_loader import DataLoader

def test_validate_member_transactions_missing_columns():
    """测试会员数据缺少必填字段时的校验"""
    loader = DataLoader()
    df = pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "frequency": [3, 5],
        # 缺少 last_purchase_date 和 monetary
    })
    valid, errors = loader.validate_data("member_transactions", df)
    assert not valid
    assert len(errors) >= 2
    assert any("last_purchase_date" in e for e in errors)
    assert any("monetary" in e for e in errors)

def test_validate_member_transactions_duplicate_ids():
    """测试会员ID重复时的校验"""
    loader = DataLoader()
    df = pd.DataFrame({
        "customer_id": ["C001", "C001", "C002"],
        "last_purchase_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "frequency": [3, 5, 2],
        "monetary": [100.0, 200.0, 50.0],
    })
    valid, errors = loader.validate_data("member_transactions", df)
    assert not valid
    assert any("customer_id" in e and "重复" in e for e in errors)

def test_validate_member_transactions_negative_values():
    """测试负数值时的校验"""
    loader = DataLoader()
    df = pd.DataFrame({
        "customer_id": ["C001", "C002"],
        "last_purchase_date": ["2024-01-01", "2024-01-02"],
        "frequency": [3, -1],  # 负值
        "monetary": [100.0, 200.0],
    })
    valid, errors = loader.validate_data("member_transactions", df)
    assert not valid
    assert any("frequency" in e for e in errors)
```

**测试会员分析模块**：

```python
# tests/test_member_analyzer.py
import pytest
from core.member_analyzer import MemberAnalyzer
from core.data_loader import DataLoader

def test_rfm_calculation():
    analyzer = MemberAnalyzer()
    loader = DataLoader()
    df = loader.load_member_transactions()
    rfm = analyzer.calculate_rfm(df)
    assert "recency" in rfm.columns
    assert "frequency" in rfm.columns
    assert "monetary" in rfm.columns
    assert "recency_score" in rfm.columns
    assert "frequency_score" in rfm.columns
    assert "monetary_score" in rfm.columns

def test_safe_qcut_handles_duplicates():
    analyzer = MemberAnalyzer()
    # 构造大量重复值场景
    series = pd.Series([1, 1, 1, 1, 2, 2, 3])
    result = analyzer._safe_qcut(series, q=5, labels=[1, 2, 3, 4, 5])
    assert len(result) == len(series)

def test_segmentation():
    analyzer = MemberAnalyzer()
    loader = DataLoader()
    df = loader.load_member_transactions()
    rfm = analyzer.calculate_rfm(df)
    rfm = analyzer.segment_members(rfm)
    assert "segment_name" in rfm.columns
    assert "segment" in rfm.columns
    # 分群后应包含至少一个已知群体标签
    valid_segments = {"高价值会员", "潜力会员", "新会员", "流失风险会员", "沉睡会员"}
    assert set(rfm["segment_name"].unique()).issubset(valid_segments)
```

### 11.5 运行测试

```powershell
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_data_loader.py -v

# 生成测试覆盖率报告
pytest tests/ --cov=core --cov-report=html
```

---

## 十二、模型评估体系

### 12.1 评估体系架构

```
┌─────────────────────────────────────────────────────────┐
│                    评估体系架构                          │
├───────────┬───────────┬─────────────┬─────────────────┤
│  RAG 评估  │ Agent 评估 │  报告生成   │   可视化仪表盘   │
│ (rag_eval)│(agent_eval)│(eval_rep..)│(eval_visualizer)│
└───────────┴───────────┴─────────────┴─────────────────┘
```

### 12.2 模块清单

| 模块 | 文件 | 职责 |
|------|------|------|
| **RAGEvaluator** | `core/rag_evaluator.py` | RAG 系统多维度评估 |
| **AgentEvaluator** | `core/agent_evaluator.py` | Agent 工具选择与执行评估 |
| **EvaluationReporter** | `core/eval_reporter.py` | 评估结果汇总与报告生成 |
| **EvalVisualizer** | `core/eval_visualizer.py` | 交互式 HTML 仪表盘生成 |
| **入口脚本** | `run_eval.py` | 一键评估命令行工具 |

### 12.3 RAG 评估（RAGEvaluator）

#### 评估指标

| 阶段 | 指标 | 定义 | 计算方法 |
|------|------|------|---------|
| **检索** | Hit Rate@k | 前 k 个结果中是否有相关文档 | 命中问题数 / 总问题数 |
| | MRR@k | 首个相关文档位置的倒数 | 平均倒数排名 |
| | 平均检索耗时 | 单次检索平均耗时 | 总时间 / 问题数 |
| **生成** | 关键词命中率 | 回答中关键词与 ground truth 重合度 | 关键词 Jaccard 系数 |
| | 可回答率 | 系统能给出有效回答的比例 | 可回答数 / 总问题数 |
| | 平均回答长度 | 回答的平均字符数 | 总字符数 / 问题数 |
| **整体** | 综合得分 | 加权汇总得分 | 检索 40% + 生成 60% |

#### 无模型评估方法

采用关键词匹配 + 规则判断实现轻量级离线评估，不依赖外部 LLM：

```python
# 关键词提取：数字+单位组合 + 中文名词短语
pattern = r'\d+\.?\d*(?:天|个|件|元|%|次|小时|日|周|月)' + '|' + \
          r'[\u4e00-\u9fa5]{2,}(?:规则|流程|标准|政策|线|量|品|类)'
```

### 12.4 Agent 评估（AgentEvaluator）

#### 评估指标

| 维度 | 指标 | 定义 |
|------|------|------|
| **工具选择** | 精确匹配率 | 预测工具集合与预期完全一致的比例 |
| | 精确率 | 预测正确的工具 / 所有预测工具 |
| | 召回率 | 预测正确的工具 / 所有预期工具 |
| | F1 分数 | 精确率和召回率的调和平均 |
| **工具函数** | 通过率 | 工具函数能正常返回结果的比例 |
| | 平均耗时 | 工具函数执行的平均时间 |
| **整体** | 综合得分 | 工具选择 60% + 函数执行 40% |

#### 离线工具预测

基于关键词规则预测应调用的工具，无需 LLM：

```python
keyword_map = {
    "inventory_query": ["库存", "存货", "SKU", "库存数量", "当前库存"],
    "sales_forecast": ["预测", "销量预测", "销售预测", "未来销量"],
    "list_low_stock_items": ["低库存", "缺货", "补货", "预警", "低于安全线"],
    "list_all_skus": ["所有SKU", "全部SKU", "商品列表", "SKU列表"],
}
```

### 12.5 报告生成（EvaluationReporter）

支持两种格式输出：

| 格式 | 用途 | 特点 |
|------|------|------|
| **Markdown** | 人类阅读 | 表格化展示，含指标说明 |
| **JSON** | 程序解析 | 结构化数据，便于历史对比 |

核心功能：
- `generate_report()`：生成 Markdown 或 JSON 格式报告
- `save_report()`：保存报告到文件
- `compare_reports()`：对比两份报告的指标变化

### 12.6 可视化仪表盘（EvalVisualizer）

使用 Plotly 生成交互式 HTML 仪表盘，包含以下图表：

| 图表 | 类型 | 说明 |
|------|------|------|
| 综合得分卡片 | 数值卡片 | RAG / Agent 总分一目了然 |
| 仪表盘（Gauge） | Indicator | 红黄绿分区显示 |
| 多维度雷达图 | Scatterpolar | 各维度均衡度可视化 |
| 检索/生成指标 | Bar | 百分率指标对比 |
| 工具选择指标 | Bar | 精确匹配率/F1/精确率/召回率 |
| 工具通过率 | Bar | 各工具通过/失败状态（绿/红） |
| 品类/任务类型对比 | Grouped Bar | 按分组维度对比 |

设计原则：
- 同单位同图（百分率不与毫秒/字数混在一起）
- RAG 与 Agent 各自独立雷达图，不混淆维度
- 响应式 flex 布局，自动适配窗口宽度
- 渐变色头部 + 卡片式布局，视觉层次清晰

### 12.7 评估数据集

| 数据集 | 文件 | 规模 | 覆盖范围 |
|--------|------|------|---------|
| RAG 评估集 | `data/eval/rag_eval_dataset.json` | 16 题 | 流程类、参数类、管理类、政策类 |
| Agent 评估集 | `data/eval/agent_eval_dataset.json` | 12 任务 | 单工具调用、多工具调用、无需工具 |

数据集结构：

```json
// RAG 评估集
{
  "dataset_name": "供应链知识库RAG评估集",
  "questions": [
    {
      "id": "rag_001",
      "question": "缺货商品的处理流程是什么？",
      "ground_truth": "缺货处理分为四步...",
      "relevant_docs": ["supply_chain_policy.txt"],
      "category": "流程类"
    }
  ]
}

// Agent 评估集
{
  "dataset_name": "库存Agent评估集",
  "tasks": [
    {
      "id": "agent_001",
      "query": "SKU000001 的库存是多少？",
      "expected_tools": ["inventory_query"],
      "expected_sku": "SKU000001",
      "task_type": "单工具调用"
    }
  ]
}
```

### 12.8 使用方式

```bash
# 离线评估（零成本，无需 API Key）
python run_eval.py --agent-only

# 在线评估（端到端验证）
python run_eval.py --online

# 仅评估 RAG
python run_eval.py --rag-only

# 运行单元测试
python -m pytest tests/test_*evaluator.py tests/test_eval_*.py -v
```

### 12.9 测试覆盖

| 模块 | 测试文件 | 测试用例数 |
|------|---------|-----------|
| RAGEvaluator | `tests/test_rag_evaluator.py` | 18 |
| AgentEvaluator | `tests/test_agent_evaluator.py` | 15 |
| EvaluationReporter | `tests/test_eval_reporter.py` | 7 |
| EvalVisualizer | `tests/test_eval_visualizer.py` | 10 |
| **合计** | | **50** |

---

## 十三、扩展开发指南

### 13.1 添加新的 RAG 知识库文档

1. 将文档（`.txt` 或 `.md`）放入 `data/knowledge/` 目录
2. 在知识库问答页面点击"构建知识库"
3. 系统自动加载新文档并重建向量库

### 13.2 添加新的 Agent 工具

本项目使用 DeepSeek 原生 Tool Calling，工具以字典形式注册，无需继承 `BaseTool`：

1. 在 `core/agent_inventory.py` 的 `InventoryAgent` 类中实现新的工具函数：

```python
def query_new_info(self, sku_id: str) -> str:
    """新工具的业务逻辑"""
    # 实现查询逻辑
    return f"SKU {sku_id} 的新信息：..."
```

2. 在 `__init__()` 的 `self.tools` 字典中注册新工具：

```python
self.tools = {
    "inventory_query": {
        "description": "查询指定 SKU 的当前库存数量、安全库存线和最近7天销量",
        "function": self.query_inventory
    },
    "sales_forecast": {
        "description": "预测指定 SKU 未来7天的销量",
        "function": self.query_forecast
    },
    "list_low_stock_items": {
        "description": "查询所有库存低于安全库存线的商品列表",
        "function": self.list_low_stock_items
    },
    "list_all_skus": {
        "description": "获取所有 SKU 编号列表",
        "function": self.list_all_skus
    },
    # 新增工具
    "query_new_info": {
        "description": "查询 SKU 的新维度信息",
        "function": self.query_new_info
    }
}
```

3. 若新工具需要参数（如 `sku_id`），需在 `run()` 方法的工具 schema 构造逻辑中加入：

```python
if name == "query_new_info":
    params["properties"]["sku_id"] = {"type": "string", "description": "SKU 编号"}
    params["required"] = ["sku_id"]
```

4. 若工具无需参数，沿用 `list_low_stock_items` / `list_all_skus` 的分支（不设置 properties）。

### 13.3 添加新的分析维度

在 `core/member_analyzer.py` 中添加新方法：

```python
def get_vip_analysis(self, rfm_df: pd.DataFrame) -> Dict:
    """VIP 会员深度分析"""
    vip = rfm_df[rfm_df["segment"] == 0]
    return {
        "count": len(vip),
        "avg_monetary": vip["monetary"].mean(),
        # ...
    }
```

### 13.4 替换为其他 LLM

如需替换为 OpenAI 或其他兼容 OpenAI SDK 的模型，修改 `core/deepseek_client.py` 的 `DeepSeekClient` 初始化参数：

```python
# OpenAI 版本
client = DeepSeekClient(
    api_key=api_key,
    base_url=None  # 使用 OpenAI 默认 base_url（去掉即默认 OpenAI 官方端点）
)
# 并在调用处将 model 改为 "gpt-4o-mini" 等
```

注意：`report_generator.py` 与 `rag_engine.py` 中直接使用了 `langchain_openai.ChatOpenAI`，替换模型时需同步修改其 `base_url` 和 `model` 参数。

### 13.5 添加新的页面

1. 在 `pages/` 目录创建新文件，如 `05_new_feature.py`
2. 遵循页面开发规范（见第六章）
3. Streamlit 自动在导航栏添加新页面

---

## 附录 A：依赖包说明

> 完整依赖以 `requirements.txt` 为准。以下为各包用途说明。

| 包名 | 用途 | 必需 |
|------|------|------|
| streamlit | Web 框架（前端页面） | ✅ |
| pandas | 数据处理（RFM、库存等 DataFrame 操作） | ✅ |
| numpy | 数值计算（模拟数据生成） | ✅ |
| plotly | 可视化图表（会员分析饼图、散点图等） | ✅ |
| langchain | LLM 框架核心 | ✅ |
| langchain-community | 社区组件（HuggingFaceEmbeddings、Chroma、TextLoader） | ✅ |
| langchain-core | 核心抽象（PromptTemplate） | ✅ |
| langchain-openai | ChatOpenAI 集成（兼容 DeepSeek API） | ✅ |
| langchain-text-splitters | 文本分割器（RecursiveCharacterTextSplitter） | ✅ |
| langchain-classic | 经典链（RetrievalQA） | ✅ |
| chromadb | 向量数据库 | ✅ |
| sentence-transformers | 本地 Embedding 模型加载 | ✅ |
| openai | OpenAI SDK（兼容 DeepSeek API，用于 Agent Tool Calling） | ✅ |
| requests | HTTP 请求 | ✅ |
| python-dotenv | 环境变量管理（.env 文件加载） | ✅ |
| scikit-learn | 数据分析工具库（项目预留依赖） | ✅ |
| pytest | 单元测试 | 可选 |

**完整 requirements.txt 内容**：

```
streamlit>=1.30.0
pandas>=2.1.0
numpy>=1.26.0
plotly>=5.18.0
langchain>=0.1.0
langchain-community>=0.4.0
langchain-core>=1.0.0
langchain-openai>=1.0.0
langchain-text-splitters>=1.0.0
langchain-classic>=1.0.0
chromadb>=0.4.22
sentence-transformers>=2.2.2
openai>=2.0.0
requests>=2.31.0
python-dotenv>=1.0.0
pytest>=7.4.3
scikit-learn>=1.3.0
```

---

## 附录 B：DeepSeek 与 OpenAI 差异说明

| 差异点 | 说明 | 应对方案 |
|--------|------|---------|
| Embedding API | DeepSeek 不提供 | 使用本地 bge-large-zh-v1.5 |
| API 格式 | 兼容 OpenAI SDK | 只需修改 base_url |
| 模型选择 | chat / reasoner | chat 用于对话，reasoner 用于推理 |
| 价格 | 显著低于 OpenAI | 可进行更多测试迭代 |
| 中文效果 | 优秀 | 适合中文零售场景 |

---

## 附录 C：关键代码引用

| 模块 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 配置管理 | core/config.py | Config |
| API 客户端 | core/deepseek_client.py | DeepSeekClient（chat / chat_stream） |
| 数据加载 | core/data_loader.py | DataLoader |
| 会员分析 | core/member_analyzer.py | MemberAnalyzer（calculate_rfm / _safe_qcut / segment_members / analyze） |
| RAG 引擎 | core/rag_engine.py | SupplyChainRAG（load_documents / build_vectorstore / has_vectorstore / query） |
| 库存 Agent | core/agent_inventory.py | InventoryAgent（run / query_inventory / query_forecast / list_low_stock_items / list_all_skus） |
| 日报生成 | core/report_generator.py | DailyReportGenerator（generate） |
| 主入口 | app.py | - |
| 会员页面 | pages/01_member_analysis.py | - |
| 问答页面 | pages/02_knowledge_qa.py | - |
| Agent 页面 | pages/03_inventory_agent.py | - |
| 日报页面 | pages/04_daily_report.py | - |

---

> 本开发说明书基于 RetailMind 项目当前版本编写，涵盖架构设计、模块开发、测试策略和扩展指南。如有疑问或需要更新，请联系开发者。
