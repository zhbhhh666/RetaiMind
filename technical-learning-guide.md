# RetailMind 项目技术学习指导

---

## 一、项目技术栈总览

### 1.1 技术栈架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端层 (Streamlit)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ 会员分析页  │  │ 知识库问答  │  │ 库存预警    │  │ 运营日报  │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
└─────────┼────────────────┼────────────────┼────────────────┼───────┘
          │                │                │                │
┌─────────▼────────────────▼────────────────▼────────────────▼───────┐
│                        核心逻辑层 (Python)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────┐   │
│  │ MemberAnalyzer  │  │ SupplyChainRAG  │  │ InventoryAgent    │   │
│  │ (RFM/手动分群)  │  │ (向量检索)       │  │ (DeepSeek Tool)   │   │
│  └─────────────────┘  └─────────────────┘  └───────────────────┘   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────┐   │
│  │ DataLoader      │  │ DeepSeekClient  │  │ ReportGenerator   │   │
│  │ (数据加载)       │  │ (API客户端)      │  │ (日报生成)        │   │
│  └─────────────────┘  └─────────────────┘  └───────────────────┘   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                        数据层                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ CSV 数据 │  │ ChromaDB │  │ 知识库文档│  │ DeepSeek LLM API │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心技术组件列表

| 技术组件 | 版本 | 用途 | 重要性 |
|---------|------|------|--------|
| **Python** | 3.11+ | 核心开发语言 | ⭐⭐⭐⭐⭐ |
| **Streamlit** | ≥1.30.0 | Web应用框架（多页应用） | ⭐⭐⭐⭐⭐ |
| **Pandas** | ≥2.1.0 | 数据处理与分析（RFM、聚合） | ⭐⭐⭐⭐⭐ |
| **NumPy** | ≥1.26.0 | 数值计算 | ⭐⭐⭐⭐ |
| **Plotly** | ≥5.18.0 | 数据可视化（含评估仪表盘） | ⭐⭐⭐⭐ |
| **LangChain** | ≥0.1.0 | LLM应用开发框架（RAG、日报生成） | ⭐⭐⭐⭐⭐ |
| **langchain-community** | ≥0.4.0 | 社区集成（向量库、Embedding） | ⭐⭐⭐⭐ |
| **langchain-core** | ≥1.0.0 | 核心抽象（Prompt、消息） | ⭐⭐⭐⭐ |
| **langchain-openai** | ≥1.0.0 | OpenAI/DeepSeek 兼容Chat模型 | ⭐⭐⭐⭐ |
| **langchain-text-splitters** | ≥1.0.0 | 文档切分 | ⭐⭐⭐ |
| **langchain-classic** | ≥1.0.0 | 经典链（RetrievalQA） | ⭐⭐⭐⭐ |
| **ChromaDB** | ≥0.4.22 | 向量数据库 | ⭐⭐⭐⭐ |
| **Sentence Transformers** | ≥2.2.2 | 本地Embedding（BAAI/bge-large-zh） | ⭐⭐⭐⭐ |
| **OpenAI SDK** | ≥2.0.0 | DeepSeek API调用（含Tool Calling） | ⭐⭐⭐⭐⭐ |
| **scikit-learn** | ≥1.3.0 | 数据处理辅助 | ⭐⭐⭐ |
| **python-dotenv** | ≥1.0.0 | 环境变量管理 | ⭐⭐⭐ |
| **requests** | ≥2.31.0 | HTTP请求 | ⭐⭐⭐ |
| **pytest** | ≥7.4.3 | 单元测试框架 | ⭐⭐⭐⭐ |
| **Plotly Figure/Subplots** | ≥5.18.0 | 评估可视化仪表盘 | ⭐⭐⭐ |
| **DeepSeek API** | - | LLM服务（兼容OpenAI接口） | ⭐⭐⭐⭐⭐ |
| **数据校验** | - | 上传数据字段/类型/唯一性校验 | ⭐⭐⭐⭐ |

---

## 二、基础技术技能

### 2.1 Python 编程基础

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| Python 语法基础（变量、数据类型、控制流） | 熟练 | [Python官方教程](https://docs.python.org/3/tutorial/) |
| 函数与类（OOP编程） | 熟练 | [Python面向对象编程](https://docs.python.org/3/tutorial/classes.html) |
| 文件操作（读写CSV/Excel） | 熟练 | [Python文件处理](https://docs.python.org/3/tutorial/inputoutput.html) |
| 异常处理（try-except） | 熟练 | [Python异常处理](https://docs.python.org/3/tutorial/errors.html) |
| 列表推导式与生成器 | 熟练 | [Python列表推导式](https://docs.python.org/3/tutorial/datastructures.html) |
| 类型注解（typing模块） | 熟悉 | [Python类型提示](https://docs.python.org/3/library/typing.html) |

#### 实践建议

```python
# 示例：类型注解与异常处理
from typing import Dict, Optional

def load_config(file_path: str) -> Optional[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            import json
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {file_path} 不存在")
        return None
    except json.JSONDecodeError:
        print(f"配置文件格式错误")
        return None
```

### 2.2 数据处理与分析（Pandas）

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| DataFrame 基本操作（创建、查询、修改） | 熟练 | [Pandas官方文档](https://pandas.pydata.org/docs/) |
| 数据清洗（缺失值处理、类型转换） | 熟练 | [Pandas数据清洗指南](https://pandas.pydata.org/docs/user_guide/missing_data.html) |
| 数据聚合（groupby、pivot_table） | 熟练 | [Pandas聚合操作](https://pandas.pydata.org/docs/user_guide/groupby.html) |
| 日期时间处理（to_datetime、resample） | 熟练 | [Pandas时间序列](https://pandas.pydata.org/docs/user_guide/timeseries.html) |
| 数据合并（merge、join、concat） | 熟练 | [Pandas数据合并](https://pandas.pydata.org/docs/user_guide/merging.html) |
| 数值计算（apply、map、lambda） | 熟练 | [Pandas函数应用](https://pandas.pydata.org/docs/user_guide/basics.html#function-application) |

#### 实践建议

```python
# 示例：会员RFM分析中的数据处理（手动规则分群，不使用KMeans聚类）
import pandas as pd

def _safe_qcut(series: pd.Series, q: int, labels, default=None):
    """安全分箱：处理重复值导致 qcut 报错的问题"""
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates='drop')
    except ValueError:
        # 重复值过多时退化为按中位数二分
        median = series.median()
        return series.apply(lambda x: labels[-1] if x >= median else labels[0])

def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    current_date = df['order_date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('member_id').agg(
        Recency=('order_date', lambda x: (current_date - x.max()).days),
        Frequency=('order_id', 'nunique'),
        Monetary=('amount', 'sum')
    )
    
    # RFM 打分（使用 _safe_qcut 处理重复值）
    rfm['R_score'] = _safe_qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
    rfm['F_score'] = _safe_qcut(rfm['Frequency'], q=5, labels=[1, 2, 3, 4, 5])
    rfm['M_score'] = _safe_qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
    
    return rfm

def segment_by_rules(rfm: pd.DataFrame) -> pd.DataFrame:
    """基于 RFM 分数与平均值比较的手动规则分群（非KMeans聚类）"""
    r_avg = rfm['R_score'].mean()
    f_avg = rfm['F_score'].mean()
    m_avg = rfm['M_score'].mean()
    
    def classify(row):
        r_high = row['R_score'] >= r_avg
        f_high = row['F_score'] >= f_avg
        m_high = row['M_score'] >= m_avg
        if r_high and f_high and m_high:
            return "重要价值会员"
        if r_high and f_high and not m_high:
            return "重要保持会员"
        if r_high and not f_high and m_high:
            return "重要发展会员"
        if not r_high and f_high and m_high:
            return "重要挽留会员"
        return "一般会员"
    
    rfm['segment'] = rfm.apply(classify, axis=1)
    return rfm
```

### 2.3 数据可视化（Plotly）

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| 基础图表类型（折线图、柱状图、散点图） | 熟练 | [Plotly官方文档](https://plotly.com/python/) |
| 高级图表（热力图、桑基图、漏斗图） | 熟悉 | [Plotly高级图表](https://plotly.com/python/charts/) |
| 图表布局与样式自定义 | 熟练 | [Plotly布局指南](https://plotly.com/python/layout/) |
| 交互式图表（悬停提示、下拉菜单） | 熟练 | [Plotly交互功能](https://plotly.com/python/interactive-features/) |

#### 实践建议

```python
# 示例：会员分群可视化
import plotly.express as px

def plot_segment_distribution(df: pd.DataFrame):
    fig = px.bar(
        df['segment'].value_counts().reset_index(),
        x='index',
        y='segment',
        color='index',
        title='会员分群分布',
        labels={'index': '分群', 'segment': '人数'}
    )
    fig.update_layout(showlegend=False)
    return fig
```

---

## 三、LLM 应用开发技能

### 3.1 LangChain 框架

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| LangChain 0.1.x 模块结构（langchain_community / langchain_core / langchain_openai / langchain_classic） | 熟练 | [LangChain官方文档](https://python.langchain.com/docs/get_started/introduction) |
| PromptTemplate 模板构建（langchain_core.prompts） | 熟练 | [LangChain Prompt指南](https://python.langchain.com/docs/modules/model_io/prompts/) |
| ChatOpenAI 客户端使用（兼容DeepSeek base_url） | 熟练 | [LangChain OpenAI集成](https://python.langchain.com/docs/integrations/chat/openai/) |
| RetrievalQA 检索增强问答（langchain_classic.chains） | 熟练 | [LangChain RAG指南](https://python.langchain.com/docs/use_cases/question_answering/) |
| 手动 os.walk 遍历加载文档（不使用 DirectoryLoader） | 熟练 | [os模块文档](https://docs.python.org/3/library/os.html#os.walk) |
| Text Splitter 文档切分（langchain_text_splitters） | 熟练 | [LangChain文本切分](https://python.langchain.com/docs/modules/data_connection/text_splitters/) |

> 注意：本项目 Agent 模块使用 **DeepSeek 原生 Tool Calling**（基于 OpenAI SDK），而非 LangChain Agent / ReAct 框架。

#### 实践建议

```python
# 示例：构建检索增强问答链（LangChain 0.1.x 新模块结构）
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

def build_qa_chain(vectorstore, api_key):
    llm = ChatOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        temperature=0.3
    )
    
    prompt = PromptTemplate(
        template="基于以下上下文回答问题：\n{context}\n\n问题：{question}\n\n回答：",
        input_variables=["context", "question"]
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(k=5),
        chain_type_kwargs={"prompt": prompt}
    )

# query 方法使用 invoke()（LangChain 0.1.x 推荐方式）
# result = qa_chain.invoke({"query": "会员积分规则是什么？"})
```

### 3.2 向量数据库（ChromaDB）

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| 向量数据库基本概念 | 熟练 | [ChromaDB官方文档](https://docs.trychroma.com/) |
| Embedding 模型原理与使用 | 熟练 | [Sentence Transformers文档](https://www.sbert.net/) |
| ChromaDB 数据存储与查询 | 熟练 | [ChromaDB快速开始](https://docs.trychroma.com/getting-started) |
| MMR 检索策略 | 熟悉 | [ChromaDB检索方式](https://docs.trychroma.com/usage-guide) |

#### 实践建议

```python
# 示例：向量数据库操作（含HF镜像加速与手动os.walk文档加载）
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 配置 HF 镜像加速（国内下载 BAAI/bge-large-zh-v1.5）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cpu"}
)

# 手动 os.walk 遍历加载文档（兼容Windows，不使用 DirectoryLoader）
def load_documents_by_walk(kb_dir: str) -> list:
    docs = []
    for root, dirs, files in os.walk(kb_dir):
        for fname in files:
            if fname.endswith((".md", ".txt")):
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    docs.append(Document(page_content=f.read(), metadata={"source": fpath}))
    return docs

# 切分并构建向量库
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(load_documents_by_walk("./knowledge_base"))

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vector_db"
)

# 检索相似文档
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

### 3.3 DeepSeek API 集成

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| DeepSeek API 认证方式（兼容OpenAI接口） | 熟练 | [DeepSeek开放平台](https://platform.deepseek.com/) |
| Chat Completion API 使用 | 熟练 | [DeepSeek API文档](https://platform.deepseek.com/docs/api) |
| DeepSeekClient 封装（chat / chat_stream 方法） | 熟练 | 项目 deepseek_client.py |
| 原生 Tool Calling API（tools / tool_choice 参数） | 熟练 | [DeepSeek Tool Calling](https://platform.deepseek.com/docs/tool_calling) |
| 手动处理 tool_calls 响应（非 LangChain Agent） | 熟练 | 项目 agent_inventory.py |
| 模型参数调优（temperature、max_tokens） | 熟悉 | [DeepSeek参数说明](https://platform.deepseek.com/docs/model) |

#### 实践建议

```python
# 示例1：DeepSeekClient 封装（基于 OpenAI SDK）
from openai import OpenAI

class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages, tools=None, tool_choice="auto", **kwargs):
        """同步对话，支持 tools 与 tool_choice 参数"""
        return self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )

    def chat_stream(self, messages, **kwargs):
        """流式对话"""
        return self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
            **kwargs
        )

# 示例2：DeepSeek 原生 Tool Calling（库存预警 Agent）
def run_inventory_agent(client: DeepSeekClient, user_query: str):
    tools = [
        {"type": "function", "function": {
            "name": "inventory_query",
            "description": "查询指定SKU的库存",
            "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}
        }},
        {"type": "function", "function": {
            "name": "list_low_stock_items",
            "description": "列出低库存商品",
            "parameters": {"type": "object", "properties": {}}
        }},
        {"type": "function", "function": {
            "name": "sales_forecast",
            "description": "销售预测",
            "parameters": {"type": "object", "properties": {"sku": {"type": "string"}}, "required": ["sku"]}
        }},
        {"type": "function", "function": {
            "name": "list_all_skus",
            "description": "列出所有SKU",
            "parameters": {"type": "object", "properties": {}}
        }},
    ]

    messages = [{"role": "user", "content": user_query}]
    response = client.chat(messages=messages, tools=tools, tool_choice="auto")

    # 手动处理 tool_calls 响应（不使用 LangChain Agent / ReAct）
    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            result = dispatch_tool(call.function.name, call.function.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
        # 再次调用让模型汇总结果
        final = client.chat(messages=messages, tools=tools, tool_choice="auto")
        return final.choices[0].message.content
    return msg.content
```

---

## 四、模型评估体系

### 4.1 RAG 评估（RAGEvaluator）

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| RAG 评估指标体系（Hit Rate、MRR、可回答率） | 熟练 | 本项目 rag_evaluator.py |
| 无模型评估方法（关键词匹配、规则判断） | 熟练 | 本项目 rag_evaluator.py |
| 关键词提取（正则表达式、名词短语匹配） | 熟悉 | Python re 模块文档 |
| 评估数据集构建（问答对、ground truth 设计） | 熟练 | 本项目 data/eval/ |
| 按维度分组评估（按品类/任务类型） | 熟悉 | pandas groupby 操作 |

#### 实践要点

```python
# RAG 评估核心流程
from core.rag_evaluator import RAGEvaluator

evaluator = RAGEvaluator(
    rag_engine=rag_engine,        # 在线模式：传入 RAG 引擎
    dataset_path="data/eval/rag_eval_dataset.json"
)

# 运行完整评估
result = evaluator.evaluate_all()

# 查看核心指标
print(f"综合得分: {result['overall']['overall_score']}")
print(f"检索命中率: {result['retrieval']['hit_rate']:.1%}")
print(f"MRR: {result['retrieval']['mrr']:.4f}")
```

### 4.2 Agent 评估（AgentEvaluator）

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| 工具选择评估（精确匹配、精确率、召回率、F1） | 熟练 | 信息检索基础概念 |
| 工具函数验证（输入输出正确性校验） | 熟练 | 单元测试思想 |
| 基于关键词的工具预测（离线评估） | 熟悉 | 本项目 agent_evaluator.py |
| 在线 vs 离线评估策略 | 理解 | 本项目设计文档 |
| 按任务类型分组统计 | 熟练 | pandas groupby |

#### 实践要点

```python
# Agent 评估核心流程
from core.agent_evaluator import AgentEvaluator

evaluator = AgentEvaluator(
    agent=agent,                          
    dataset_path="data/eval/agent_eval_dataset.json"
)

# 离线评估（无需 API Key）
result = evaluator.evaluate_all(online=False)

# 在线评估（调用真实 LLM）
result = evaluator.evaluate_all(online=True)
```

### 4.3 报告与可视化（EvalReporter + EvalVisualizer）

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| 评估报告生成（Markdown / JSON） | 熟练 | 本项目 eval_reporter.py |
| Plotly 图表组件（Bar、Scatterpolar、Indicator） | 熟练 | Plotly 官方文档 |
| 子图布局（make_subplots、specs 参数） | 熟悉 | Plotly subplots 文档 |
| HTML 模板 + CSS 自定义仪表盘 | 熟悉 | HTML/CSS 基础 |
| 历史报告对比（指标变化率计算） | 理解 | 本项目 compare_reports |

#### 实践要点

```python
# 生成交互式 HTML 仪表盘
from core.eval_visualizer import EvalVisualizer

viz = EvalVisualizer(output_dir="./data/eval/results")
html_path = viz.render_dashboard(
    rag_result=rag_result,
    agent_result=agent_result,
    filename="eval_dashboard"
)

# 生成 Markdown + JSON 报告
from core.eval_reporter import EvaluationReporter

reporter = EvaluationReporter(output_dir="./data/eval/results")
report_path = reporter.save_report(rag_result, agent_result)
```

### 4.4 单元测试（pytest）

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| pytest 基础（fixture、assert、parametrize） | 熟练 | pytest 官方文档 |
| 测试 fixture 共享（conftest.py） | 熟悉 | pytest 文档 |
| 异常测试（pytest.raises） | 熟练 | pytest 文档 |
| 测试分类与分层（单元/集成/端到端） | 理解 | 软件测试基础 |
| 边界情况测试（空值、None、非法输入） | 熟练 | 测试用例设计 |

---

## 五、Web 应用开发技能

### 5.1 Streamlit 框架

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| Streamlit 基础组件（text、dataframe、chart） | 熟练 | [Streamlit官方文档](https://docs.streamlit.io/) |
| 页面导航与多页应用 | 熟练 | [Streamlit多页应用](https://docs.streamlit.io/develop/tutorials/multipage) |
| 交互式组件（button、slider、selectbox） | 熟练 | [Streamlit组件库](https://docs.streamlit.io/library/api-reference) |
| Session State 状态管理 | 熟练 | [Streamlit状态管理](https://docs.streamlit.io/library/api-reference/session-state) |
| 文件上传与处理 | 熟练 | [Streamlit文件上传](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader) |

#### 实践建议

```python
# 示例：Streamlit多页应用结构
import streamlit as st
import pandas as pd

st.set_page_config(page_title="会员分析", layout="wide")

st.title("会员RFM分析")

uploaded_file = st.file_uploader("上传会员数据CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    if st.button("开始分析"):
        analyzer = MemberAnalyzer(df)
        rfm_df = analyzer.calculate_rfm()
        st.dataframe(rfm_df)
```

---

### 5.2 数据导入功能开发

> **核心定位**：数据导入是连接用户真实业务数据与系统分析能力的桥梁，是产品化的关键功能。本项目实现了配置驱动的数据校验框架，支持多类型数据的统一管理。

#### 必备知识点

| 知识点 | 掌握程度 | 学习资源 |
|--------|---------|---------|
| CSV 文件读写与编码处理 | 熟练 | Pandas read_csv/to_csv 文档 |
| 数据校验设计（字段/类型/唯一性/范围） | 熟练 | 本项目 `data_loader.py` 源码 |
| 配置驱动架构（DATA_TYPES 配置表） | 精通 | 阅读 DATA_TYPES 设计 |
| 文件上传与临时文件处理（Streamlit） | 熟练 | Streamlit file_uploader API |
| 错误信息友好化设计 | 熟练 | 用户体验设计原则 |

#### 核心设计模式：配置驱动的数据校验

```python
# 配置驱动：所有数据类型的校验规则集中在一个字典中
DATA_TYPES = {
    "member_transactions": {
        "filename": "member_transactions.csv",
        "required_columns": ["customer_id", "last_purchase_date", "frequency", "monetary"],
        "unique_columns": ["customer_id"],
        "positive_columns": ["frequency", "monetary"],
        "date_columns": ["last_purchase_date"],
        "numeric_columns": ["frequency", "monetary"],
    },
    # ... 其他数据类型
}

# 通用校验函数：读取配置，自动执行所有校验
def validate_data(self, data_type: str, df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """通用数据校验函数，根据 DATA_TYPES 配置自动执行"""
    errors = []
    config = DATA_TYPES[data_type]
    
    # 1. 必填字段检查
    for col in config.get("required_columns", []):
        if col not in df.columns:
            errors.append(f"缺少必填字段: {col}")
    
    # 2. 唯一性检查
    for col in config.get("unique_columns", []):
        if df[col].duplicated().any():
            errors.append(f"{col} 存在重复值")
    
    # 3. 正负值检查
    for col in config.get("positive_columns", []):
        if (df[col] <= 0).any():
            errors.append(f"{col} 存在非正值")
    
    # ... 更多校验规则
    return len(errors) == 0, errors
```

#### 关键实现要点

| 要点 | 说明 | 实践建议 |
|------|------|---------|
| **分层校验** | 先格式、后类型、再业务规则 | 提前失败，快速反馈 |
| **错误累积** | 一次校验返回所有错误，而非遇到第一个就停止 | 减少用户反复试错 |
| **操作幂等** | 同一操作多次执行结果一致（如重置） | 避免状态不一致 |
| **数据预览** | 上传后先展示预览和校验结果，用户确认后才保存 | 防止误操作覆盖数据 |
| **异常隔离** | 上传失败不影响原有数据 | 数据安全第一 |

#### 实践练习

**练习 1：扩展数据类型**
- 在 `DATA_TYPES` 中新增一种数据类型（如"商品信息"）
- 配置校验规则：必填字段、唯一性、数值范围
- 在数据管理页面新增对应标签页
- 编写测试用例验证

**练习 2：增强校验功能**
- 添加字段枚举值校验（如品类只能是指定的几个值）
- 添加日期范围校验（如不能早于 2020-01-01）
- 添加字符串长度校验
- 思考：如何让这些校验规则也通过配置实现？

**练习 3：数据导出功能**
- 在数据管理页面添加"导出当前数据"按钮
- 支持 CSV 和 Excel 两种格式
- 考虑大文件导出的性能优化

---

## 六、学习路径规划

### 6.1 入门阶段（1-2周）

**目标**：掌握基础技术，能运行项目并理解核心流程

| 学习内容 | 时间分配 | 产出 |
|---------|---------|------|
| Python 基础语法 | 3天 | 完成 Python 基础练习题 |
| Pandas 数据处理 | 3天 | 完成会员数据清洗实战 |
| Streamlit 基础 | 2天 | 创建简单的单页应用 |
| 项目环境搭建 | 2天 | 成功运行项目 |

### 6.2 进阶阶段（2-3周）

**目标**：深入理解核心模块，能独立开发新功能

| 学习内容 | 时间分配 | 产出 |
|---------|---------|------|
| LangChain 0.1.x 模块结构（langchain_core/openai/classic/community） | 4天 | 完成 RAG 问答模块开发 |
| 向量数据库与本地 Embedding（BAAI/bge-large-zh + HF镜像） | 3天 | 完成知识库构建与检索 |
| 手动 os.walk 文档加载与 Text Splitter 切分 | 1天 | 完成知识库文档预处理 |
| DeepSeek API（OpenAI SDK 兼容）与 DeepSeekClient 封装 | 2天 | 完成 LLM 调用集成 |
| Plotly 可视化 | 2天 | 完成数据分析图表 |

### 6.3 高级阶段（2周）

**目标**：掌握 DeepSeek 原生 Tool Calling，能优化系统性能

| 学习内容 | 时间分配 | 产出 |
|---------|---------|------|
| DeepSeek 原生 Tool Calling（手动处理 tool_calls，非 LangChain Agent） | 3天 | 完成库存预警 Agent（4个工具） |
| PromptTemplate 与 LangChain ChatOpenAI 生成运营日报 | 2天 | 完成运营日报生成器 |
| 手动规则分群（RFM 分数与平均值比较，非 KMeans） | 1天 | 完成会员分群优化 |
| 系统优化 | 3天 | 提升检索精度与响应速度 |
| 项目实战 | 4天 | 完整实现一个新功能模块 |

### 6.4 评估体系阶段（1周）

**目标**：掌握 RAG 和 Agent 评估方法，能建立质量监控体系

| 学习内容 | 时间分配 | 产出 |
|---------|---------|------|
| RAG 评估指标（Hit Rate、MRR、可回答率） | 1天 | 理解评估原理和指标计算 |
| Agent 评估指标（精确匹配、F1、工具通过率） | 1天 | 理解工具选择评估原理 |
| 无模型评估方法（关键词匹配、规则判断） | 1天 | 实现轻量级离线评估 |
| pytest 单元测试框架 | 1天 | 编写模块测试用例 |
| Plotly 可视化仪表盘 | 1天 | 生成交互式评估仪表盘 |
| 评估数据集构建 | 1天 | 设计评估样本和 ground truth |
| 综合实践 | 2天 | 完成完整评估体系搭建 |

---

## 七、推荐学习资源

### 7.1 在线课程

| 课程名称 | 平台 | 适用阶段 |
|---------|------|---------|
| Python for Data Science | Coursera | 入门 |
| Machine Learning Specialization | Coursera | 进阶 |
| LangChain for LLM Application Development | DeepLearning.AI | 进阶 |
| Building LLM-Powered Applications | Udemy | 进阶 |
| Streamlit Tutorials | YouTube | 入门 |

### 7.2 文档资源

| 资源名称 | 链接 |
|---------|------|
| Python 官方文档 | https://docs.python.org/3/ |
| Pandas 官方文档 | https://pandas.pydata.org/docs/ |
| Streamlit 官方文档 | https://docs.streamlit.io/ |
| LangChain 官方文档 | https://python.langchain.com/docs/get_started/introduction |
| DeepSeek API 文档 | https://platform.deepseek.com/docs/api |
| DeepSeek Tool Calling 文档 | https://platform.deepseek.com/docs/tool_calling |
| OpenAI SDK（兼容DeepSeek） | https://platform.openai.com/docs/api |
| Sentence Transformers（本地Embedding） | https://www.sbert.net/ |
| HuggingFace 镜像站 | https://hf-mirror.com |
| os.walk（手动文档遍历） | https://docs.python.org/3/library/os.html#os.walk |

### 7.3 实战项目

| 项目名称 | 学习点 |
|---------|--------|
| [LangChain Tutorials](https://github.com/langchain-ai/langchain) | LLM 应用开发 |
| [Streamlit Gallery](https://streamlit.io/gallery) | Web 应用设计 |
| [ChromaDB Examples](https://github.com/chroma-core/chroma) | 向量数据库 |

---

## 八、关键技能评估清单

### 8.1 基础技能

- [ ] 能熟练使用 Python 进行日常开发
- [ ] 能使用 Pandas 完成数据清洗和分析任务
- [ ] 能使用 Pandas 实现 RFM 打分与手动规则分群（基于平均值比较，非 KMeans 聚类）
- [ ] 能实现 _safe_qcut 方法处理重复值导致 qcut 报错的问题
- [ ] 能使用 Plotly 创建交互式图表
- [ ] 能使用 Streamlit 创建基本的 Web 应用（多页应用）

### 8.2 LLM 开发技能

- [ ] 理解 RAG（检索增强生成）原理
- [ ] 能使用 LangChain 0.1.x 新模块结构（langchain_core / langchain_openai / langchain_classic / langchain_community）
- [ ] 能使用 RetrievalQA（langchain_classic.chains）构建问答链，并使用 invoke() 调用
- [ ] 能使用 ChromaDB 构建向量知识库
- [ ] 能配置 HF 镜像加速并加载本地 Embedding（BAAI/bge-large-zh-v1.5）
- [ ] 能使用手动 os.walk 遍历加载文档（不使用 DirectoryLoader）
- [ ] 能封装 DeepSeekClient（chat / chat_stream，支持 tools 与 tool_choice）
- [ ] 能调用 DeepSeek API 完成 LLM 交互
- [ ] 能设计有效的 Prompt 模板（PromptTemplate）

### 8.3 Agent 开发技能

- [ ] 理解 DeepSeek 原生 Tool Calling 工作原理（非 LangChain Agent / ReAct）
- [ ] 能定义和实现自定义工具（inventory_query / sales_forecast / list_low_stock_items / list_all_skus）
- [ ] 能手动处理 tool_calls 响应并完成多轮工具调用流程
- [ ] 能管理 messages 历史并在工具返回后再次调用模型汇总结果

### 8.4 数据导入与校验技能

- [ ] 理解配置驱动的数据校验架构（DATA_TYPES 设计模式）
- [ ] 能实现完整的数据校验流程（字段检查/类型检查/唯一性/范围验证）
- [ ] 能使用 Streamlit file_uploader 实现文件上传功能
- [ ] 能设计友好的错误提示信息，帮助用户快速定位问题
- [ ] 能实现数据重置功能，保证操作的安全性和可回退性
- [ ] 能扩展新的数据类型，只需修改配置而无需改动核心逻辑

### 8.5 项目实战技能

- [ ] 能独立完成项目环境搭建
- [ ] 能理解项目代码结构和模块关系
- [ ] 能定位和修复常见错误
- [ ] 能开发新功能并集成到现有系统

---

### 8.6 评估体系技能

- [ ] 理解 RAG 评估指标（Hit Rate、MRR、关键词命中率、可回答率）
- [ ] 理解 Agent 评估指标（精确匹配率、F1、精确率、召回率、工具通过率）
- [ ] 能使用 RAGEvaluator 运行 RAG 评估
- [ ] 能使用 AgentEvaluator 运行 Agent 评估
- [ ] 能构建评估数据集（设计 ground truth 和评估样本）
- [ ] 能生成 Markdown 和 JSON 格式的评估报告
- [ ] 能使用 EvalVisualizer 生成交互式 HTML 仪表盘
- [ ] 能对比历史评估报告，分析优化效果
- [ ] 理解离线评估 vs 在线评估的适用场景
- [ ] 能编写 pytest 单元测试覆盖核心功能

## 九、学习建议

1. **理论与实践结合**：每个知识点学习后，务必在项目中实际应用
2. **源码阅读**：深入阅读项目核心代码，理解设计思路和实现细节
3. **调试技巧**：掌握 Streamlit 调试方法，学会使用日志和断点
4. **API 文档优先**：遇到问题时，优先查阅官方文档
5. **社区交流**：关注 LangChain、Streamlit 社区，学习最佳实践
6. **持续迭代**：定期回顾代码，优化实现方式

---

*文档版本：v3.0*  
*最后更新：2026年7月*  
*新增内容：模型评估体系（RAG评估、Agent评估、可视化仪表盘、pytest测试）*
