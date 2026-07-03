# RetailMind 操作指导书

> 面向零售业务的智能决策助手 — 从安装到运行的完整操作指南

---

## 目录

- [一、环境准备](#一环境准备)
  - [1.1 系统要求](#11-系统要求)
  - [1.2 Python 环境安装](#12-python-环境安装)
  - [1.3 DeepSeek API Key 获取](#13-deepseek-api-key-获取)
- [二、项目安装](#二项目安装)
  - [2.1 获取项目代码](#21-获取项目代码)
  - [2.2 项目结构说明](#22-项目结构说明)
  - [2.3 安装依赖](#23-安装依赖)
  - [2.4 配置环境变量](#24-配置环境变量)
- [三、启动应用](#三启动应用)
- [四、功能模块操作](#四功能模块操作)
  - [4.1 数据管理中心](#41-数据管理中心)
  - [4.2 会员数据分析](#42-会员数据分析)
  - [4.3 供应链知识库问答](#43-供应链知识库问答)
  - [4.4 库存预警 Agent](#44-库存预警-agent)
  - [4.5 运营日报生成](#45-运营日报生成)
  - [4.6 模型评估体系](#46-模型评估体系)
- [五、常见问题排查](#五常见问题排查)
- [六、数据替换指南](#六数据替换指南)
- [七、部署与分享](#七部署与分享)
- [八、快捷命令速查表](#八快捷命令速查表)

---

## 一、环境准备

### 1.1 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| 操作系统 | Windows 10 / macOS 11 / Ubuntu 20.04 | Windows 11 / macOS 13 |
| Python | 3.10 | 3.12 |
| 内存 | 4 GB | 8 GB 以上 |
| 磁盘空间 | 3 GB（含依赖） | 5 GB 以上 |
| 网络 | 需要访问 DeepSeek API 和 PyPI | 稳定的宽带连接 |

> **注意**：首次运行 RAG 模块时，系统会自动下载 `BAAI/bge-large-zh-v1.5` Embedding 模型（约 1.2 GB），请确保网络通畅。

### 1.2 Python 环境安装

#### Windows 系统

**步骤 1：下载 Python**

1. 访问 Python 官网：https://www.python.org/downloads/
2. 下载 Python 3.12 安装包
3. 双击运行安装程序

**步骤 2：安装时注意事项**

> ⚠️ **关键步骤**：安装时务必勾选 **"Add Python to PATH"** 选项，否则后续命令行无法识别 `python` 和 `pip` 命令。

```
勾选项：☑ Add Python 3.12 to PATH
点击：Install Now
```

**步骤 3：验证安装**

打开 PowerShell（按 `Win + X`，选择"终端"或"PowerShell"），输入：

```powershell
python --version
```

预期输出：
```
Python 3.12.x
```

继续验证 pip：
```powershell
pip --version
```

预期输出：
```
pip 23.2.x from ...
```

#### macOS 系统

```bash
# 使用 Homebrew 安装
brew install python@3.12

# 验证安装
python3 --version
```

### 1.3 DeepSeek API Key 获取

**步骤 1：注册 DeepSeek 账号**

1. 访问 DeepSeek 开放平台：https://platform.deepseek.com/
2. 点击"注册"，使用手机号或邮箱注册
3. 完成实名认证（国内用户必须）

**步骤 2：创建 API Key**

1. 登录后进入控制台
2. 左侧菜单选择「API Keys」
3. 点击「创建 API Key」
4. 输入名称（如 "RetailMind"）
5. 复制生成的 API Key

> ⚠️ **重要提示**：
> - API Key 只在创建时显示一次，请立即复制保存
> - 格式为：`sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
> - 请勿泄露给他人，否则可能导致费用损失

**步骤 3：账户充值**

1. 进入「费用管理」→「充值」
2. 支持支付宝/微信支付
3. 建议首次充值 10 元用于测试（DeepSeek 价格很低，10 元可进行大量测试）

**步骤 4：验证 API Key**

在 PowerShell 中执行以下命令验证 API Key 是否可用：

```powershell
$api_key = "你的API Key"
$headers = @{ "Authorization" = "Bearer $api_key"; "Content-Type" = "application/json" }
$body = '{"model":"deepseek-chat","messages":[{"role":"user","content":"你好"}]}'
$response = Invoke-RestMethod -Uri "https://api.deepseek.com/v1/chat/completions" -Method Post -Headers $headers -Body $body
$response.choices[0].message.content
```

预期输出：一段中文回复（如"你好！有什么可以帮助您的吗？"）

---

## 二、项目安装

### 2.1 获取项目代码

项目已位于 `f:\trae1\RetaiMind\retailmind` 目录。

如果需要复制到其他位置，执行：

```powershell
# 复制到其他目录（示例）
Copy-Item -Path "f:\trae1\RetaiMind\retailmind" -Destination "D:\projects\retailmind" -Recurse
```

### 2.2 项目结构说明

项目目录结构如下：

```
retailmind/
├── core/
│   ├── __init__.py
│   ├── agent_inventory.py    # 库存预警 Agent（DeepSeek 原生 Tool Calling）
│   ├── config.py             # 配置模块（读取 .env）
│   ├── data_loader.py        # 数据加载模块（含上传/校验/重置）
│   ├── deepseek_client.py    # DeepSeek API 客户端
│   ├── member_analyzer.py    # 会员分析引擎（手动规则分群）
│   ├── rag_engine.py         # RAG 检索引擎（HuggingFace Embeddings）
│   └── report_generator.py   # 运营日报生成器（LangChain ChatOpenAI）
├── data/
│   ├── knowledge/            # 知识库文档（.txt）
│   └── raw/                  # 业务数据（.csv）
├── pages/
│   ├── 00_data_management.py # 数据管理中心（上传/校验/重置）
│   ├── 01_member_analysis.py
│   ├── 02_knowledge_qa.py
│   ├── 03_inventory_agent.py
│   └── 04_daily_report.py
├── prompts/                  # Prompt 模板
├── vector_db/                # 向量数据库（ChromaDB）
├── .env                      # 环境变量
├── app.py                    # 主入口
└── requirements.txt
```

| 目录/文件 | 说明 |
|----------|------|
| `core/` | 业务核心逻辑模块 |
| `data/knowledge/` | 知识库原文档（`.txt`），RAG 数据源 |
| `data/raw/` | 业务数据（会员、库存、销量预测 CSV） |
| `pages/` | Streamlit 多页面应用入口 |
| `prompts/` | 各模块使用的 Prompt 模板 |
| `vector_db/` | ChromaDB 持久化目录，构建后自动生成 |
| `.env` | 存放 `DEEPSEEK_API_KEY` 等敏感配置 |
| `app.py` | Streamlit 主入口，启动后挂载四个功能页面 |

### 2.3 安装依赖

**步骤 1：进入项目目录**

```powershell
cd f:\trae1\RetaiMind\retailmind
```

**步骤 2：（推荐）创建虚拟环境**

```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows PowerShell）
.\venv\Scripts\Activate.ps1

# 如果遇到执行策略错误，先运行：
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

macOS/Linux 激活方式：
```bash
source venv/bin/activate
```

**步骤 3：安装依赖包**

```powershell
pip install -r requirements.txt
```

当前 `requirements.txt` 完整依赖列表：

| 包名 | 最低版本 | 用途 |
|------|---------|------|
| streamlit | >=1.30.0 | Web 应用框架 |
| pandas | >=2.1.0 | 数据处理 |
| numpy | >=1.26.0 | 数值计算 |
| plotly | >=5.18.0 | 图表可视化 |
| langchain | >=0.1.0 | LLM 应用框架 |
| langchain-community | >=0.4.0 | LangChain 社区扩展 |
| langchain-core | >=1.0.0 | LangChain 核心 |
| langchain-openai | >=1.0.0 | OpenAI/DeepSeek 兼容客户端 |
| langchain-text-splitters | >=1.0.0 | 文本切分 |
| langchain-classic | >=1.0.0 | LangChain 经典模块 |
| chromadb | >=0.4.22 | 向量数据库 |
| sentence-transformers | >=2.2.2 | 本地 Embedding 模型 |
| openai | >=2.0.0 | OpenAI SDK（兼容 DeepSeek） |
| requests | >=2.31.0 | HTTP 请求 |
| python-dotenv | >=1.0.0 | 加载 `.env` 配置 |
| pytest | >=7.4.3 | 单元测试 |
| scikit-learn | >=1.3.0 | 数据分析工具 |

> 安装过程可能需要 5-10 分钟（取决于网络速度），因为 `sentence-transformers` 会拉取 `torch` 等大型依赖包。

**步骤 4：验证安装**

```powershell
python -c "import streamlit; import langchain; import chromadb; import pandas; print('所有依赖安装成功！')"
```

预期输出：
```
所有依赖安装成功！
```

如果出现 `ModuleNotFoundError`，说明某些包未安装成功，请重新执行步骤 3。

### 2.4 配置环境变量

**步骤 1：创建 .env 文件**

在项目根目录（`f:\trae1\RetaiMind\retailmind\`）创建 `.env` 文件。

可以使用 PowerShell 创建：

```powershell
# 在项目目录下执行
@'
DEEPSEEK_API_KEY=sk-你的真实API Key
'@ | Out-File -FilePath ".env" -Encoding utf8
```

或手动创建文件并写入以下内容：

```
DEEPSEEK_API_KEY=sk-你的真实API Key
```

**步骤 2：验证配置**

```powershell
python -c "from core.config import Config; print('API Key 已配置' if Config.DEEPSEEK_API_KEY else 'API Key 未配置')"
```

预期输出：
```
API Key 已配置
```

---

## 三、启动应用

**步骤 1：确保在项目目录下**

```powershell
cd f:\trae1\RetaiMind\retailmind
```

**步骤 2：激活虚拟环境（如果使用了虚拟环境）**

```powershell
.\venv\Scripts\Activate.ps1
```

**步骤 3：启动 Streamlit**

```powershell
streamlit run app.py
```

**步骤 4：访问应用**

启动后浏览器会自动打开 `http://localhost:8501`。

如果没有自动打开，手动在浏览器地址栏输入：`http://localhost:8501`

**步骤 5：验证应用启动成功**

首页应该显示：
- RetailMind 标题和介绍
- ✅ "API Key 已配置"（绿色提示）
- 四个功能模块的导航链接（左侧边栏）

> **停止应用**：在终端按 `Ctrl + C`

---

## 四、功能模块操作

### 4.1 数据管理中心

**路径**：左侧导航栏 → "数据管理中心"

> **技术说明**：本模块提供集中式数据管理功能，支持四类业务数据（会员交易、库存、销售预测、运营数据）的上传、校验、预览和一键重置。所有上传数据均经过字段完整性、数据类型、唯一性、正负值等多重校验，确保数据质量。

#### 功能概览

数据管理中心包含四个标签页，分别对应四类业务数据：

| 标签页 | 数据类型 | 核心功能 |
|--------|---------|---------|
| 👥 会员交易数据 | member_transactions | 上传会员消费数据，校验RFM字段 |
| 📦 库存数据 | inventory | 上传库存数据，校验SKU和库存数量 |
| 📈 销售预测数据 | sales_forecast | 上传销量预测数据，校验预测值 |
| 📊 运营数据 | operations | 按日期上传运营日报数据 |

#### 操作步骤（以会员数据为例）

**步骤 1：查看当前数据状态**
- 进入数据管理中心页面
- 顶部显示三个核心指标：数据来源、数据行数、字段数量
- "数据概览"区域展示关键统计指标（如会员数、平均频次、平均消费等）
- 点击"查看数据预览（前10行）"可展开查看数据样本

**步骤 2：上传新数据**

1. 在"上传新数据"区域，点击文件上传框或拖拽 CSV 文件
2. 系统自动读取文件并执行校验：
   - 必填字段检查（如 customer_id、frequency 等）
   - 数据类型检查（数值字段是否为数字）
   - 唯一性检查（会员ID、SKU 不重复）
   - 正负值检查（消费金额、库存数量必须为正）
   - 日期格式检查（last_purchase_date 格式验证）

3. 校验通过：显示 ✓ 校验通过提示和数据预览
4. 校验失败：显示具体错误项，修正后重新上传

**步骤 3：确认上传**
- 校验通过后，点击"确认上传"按钮
- 数据保存到 `data/raw/` 目录，覆盖原有数据
- 页面自动刷新，显示新的数据状态

**步骤 4：重置为示例数据**
- 如需恢复初始状态，点击"🔄 重置为示例数据"按钮
- 系统删除当前数据并重新生成 1000 条模拟会员数据
- 所有其他业务模块（会员分析、库存预警等）会自动使用新数据

#### 各数据类型字段说明

**会员交易数据必填字段**：

| 字段名 | 类型 | 说明 | 校验规则 |
|--------|------|------|---------|
| customer_id | 字符串 | 会员唯一标识 | 必填，唯一 |
| last_purchase_date | 日期 | 最近购买日期 | 必填，日期格式 |
| frequency | 整数 | 购买次数 | 必填，必须 > 0 |
| monetary | 浮点数 | 消费总金额 | 必填，必须 > 0 |
| preferred_category | 字符串 | 偏好品类 | 可选 |

**库存数据必填字段**：

| 字段名 | 类型 | 说明 | 校验规则 |
|--------|------|------|---------|
| sku_id | 字符串 | SKU 唯一标识 | 必填，唯一 |
| stock_qty | 整数 | 当前库存数量 | 必填，必须 >= 0 |
| safety_stock | 整数 | 安全库存线 | 必填，必须 >= 0 |
| weekly_sales | 整数 | 最近 7 天销量 | 必填，必须 >= 0 |
| category | 字符串 | 品类 | 必填 |

**销售预测数据必填字段**：

| 字段名 | 类型 | 说明 | 校验规则 |
|--------|------|------|---------|
| sku_id | 字符串 | SKU 唯一标识 | 必填，唯一 |
| forecast_7d | 整数 | 未来 7 天预测销量 | 必填，必须 >= 0 |

**运营数据必填字段**：

| 字段名 | 类型 | 说明 | 校验规则 |
|--------|------|------|---------|
| gmv | 浮点数 | 销售额（元） | 必填，必须 >= 0 |
| orders | 整数 | 订单量 | 必填，必须 >= 0 |
| aov | 浮点数 | 客单价 | 必填，必须 >= 0 |
| new_members | 整数 | 新增会员数 | 必填，必须 >= 0 |
| repurchase_rate | 浮点数 | 复购率（%） | 必填，必须 >= 0 |

> 💡 **提示**：运营数据按日期存储，不同日期的数据互不覆盖。上传前请先选择正确的日期。

### 4.2 会员数据分析

**路径**：左侧导航栏 → "会员数据分析"

> **技术说明**：本模块采用**手动规则分群**（不使用 KMeans 聚类），基于 RFM 三个维度的分数与各自平均值进行比较，按业务规则将会员划分为五个群体。该方案对各业务数据规模都有良好稳定性，无需担心聚类中心波动。

#### 操作步骤

**步骤 1：加载数据**
- 进入页面后，系统自动加载 1000 条模拟会员交易数据
- 显示"加载成功！共 1000 条交易记录，1000 位会员"

**步骤 2：开始分析**
- 点击"开始分析"按钮
- 等待 3-5 秒，系统完成 RFM 分数计算与规则分群

**步骤 3：查看分析结果**

分析完成后，页面显示以下内容：

| 区域 | 内容 | 说明 |
|------|------|------|
| 左上 | 会员分群饼图 | 显示各分群占比（高价值/潜力/新会员/流失风险/沉睡） |
| 右上 | 核心指标卡片 | 总会员数、平均购买天数、平均频次、平均消费金额 |
| 中部 | 分群详细数据表 | 每个分群的数量、占比、平均指标 |
| 中下 | 流失风险会员表 | 超过 90 天未购买的会员列表（前 20 条） |
| 底部 | RFM 散点图 | X轴=最近购买天数，Y轴=消费金额，颜色=分群 |

**步骤 4：理解分群含义**

| 分群名称 | 含义 | 建议操作 |
|---------|------|---------|
| 高价值会员 | 近期购买、频次高、消费高 | 维护关系，提供 VIP 服务 |
| 潜力会员 | 有消费潜力但频次较低 | 促活营销，提升购买频次 |
| 新会员 | 刚注册不久 | 新人引导，首单优惠 |
| 流失风险会员 | 超过 90 天未购买 | 召回营销，个性化优惠 |
| 沉睡会员 | 长期未购买 | 低成本触达，判断是否值得召回 |

### 4.3 供应链知识库问答

**路径**：左侧导航栏 → "知识库问答"

> **技术说明**：
> - 本模块使用本地 **HuggingFaceEmbeddings** 加载 `BAAI/bge-large-zh-v1.5` 模型生成向量，无需调用云端 Embedding API。
> - 已配置 **HF 镜像加速**（`hf-mirror.com`），首次下载模型时显著提升速度。
> - 知识库文档加载采用**手动遍历目录**的方式实现（不使用 LangChain `DirectoryLoader`），以保证在 Windows 系统下的兼容性。
> - 向量检索调用链路使用 LangChain 新版 API：retriever 通过 `invoke()` 方法调用，而非旧版的 `__call__()`。

#### 首次使用：构建知识库

**步骤 1：构建向量库**
- 进入页面后，如果显示"向量库为空"，点击"构建知识库"按钮
- 等待 30-60 秒（首次需下载 Embedding 模型，约 1.2 GB，系统已配置 `hf-mirror.com` 镜像加速）
- 显示"知识库构建完成！"后页面自动刷新

**步骤 2：验证知识库**
- 页面显示"已加载向量库，共 X 个文档片段"
- 页面动态展示已加载的文档列表和数量
- 知识库包含 4 份文档：
  - supply_chain_policy.txt（供应链管理政策）
  - inventory_management.txt（库存管理操作指南）
  - order_processing.txt（订单处理流程规范）
  - member_service.txt（会员服务标准）

#### 日常使用：问答操作

**步骤 3：输入问题**

在输入框中输入问题，例如：

| 示例问题 | 预期回答内容 |
|---------|------------|
| 缺货商品如何处理？ | 缺货定义、处理步骤、审批流程 |
| 供应商如何分类？ | A/B/C 类供应商定义和考核指标 |
| 盘点周期是怎样的？ | 月度/季度/年度盘点安排 |
| 退货条件是什么？ | 退货条件、退货流程、退款时间 |
| 会员有哪些等级？ | 普通/银卡/金卡/钻石会员及权益 |
| 积分如何兑换？ | 积分获取规则和兑换方式 |

**步骤 4：查看回答**
- "回答"区域：AI 生成的答案
- "参考来源"区域：答案引用的文档文件名

### 4.4 库存预警 Agent

**路径**：左侧导航栏 → "库存预警 Agent"

> **技术说明**：本模块直接使用 **DeepSeek 原生 Tool Calling（Function Calling）**能力（不使用 LangChain Agent 框架），通过 `core/deepseek_client.py` 中封装的客户端向 DeepSeek 接口注册工具描述并解析 `tool_calls` 字段。相比 LangChain Agent，原生调用方式链路更短、调试更直观、对 DeepSeek 模型的兼容性更好。

#### 操作步骤

**步骤 1：查看库存概览**
- 页面自动加载 100 个 SKU 的库存数据
- 左上显示库存状态统计（正常/偏低/充足）
- 右上显示库存数量分布直方图

**步骤 2：查看预警列表**
- 中部表格显示所有库存低于安全线的 SKU
- 包含：SKU ID、品类、当前库存、安全库存、短缺数量、周销量

**步骤 3：使用 AI 顾问**

在输入框中输入问题，AI Agent 会通过 DeepSeek 原生 Tool Calling 自动选择并调用工具：

| 示例问题 | Agent 行为 |
|---------|-----------|
| SKU000015 是否需要补货？ | 调用 `inventory_query` 工具 → 分析库存状态 → 给出补货建议 |
| 哪些商品需要紧急补货？ | 调用 `list_low_stock_items` 工具 → 一次性返回所有低库存商品列表 |
| 请预测 SKU000020 未来销量 | 调用 `sales_forecast` 工具 → 返回 7 天预测数据 |
| 有哪些 SKU？ | 调用 `list_all_skus` 工具 → 返回所有 SKU 编号 |

**可用工具（共 4 个）**：

| 工具名称 | 功能 | 输入参数 |
|---------|------|---------|
| `inventory_query` | 查询单个 SKU 库存详情 | `sku_id`（必需） |
| `sales_forecast` | 查询单个 SKU 未来 7 天预测销量 | `sku_id`（必需） |
| `list_low_stock_items` | 查询所有库存低于安全线的商品 | 无参数 |
| `list_all_skus` | 获取所有 SKU 编号列表 | 无参数 |

**步骤 4：查看 AI 回答**
- "AI 顾问回答"区域：最终的补货建议
- "思考过程"区域：展示 Agent 的推理步骤和工具调用过程

### 4.5 运营日报生成

**路径**：左侧导航栏 → "运营日报生成"

> **技术说明**：本模块使用 **LangChain `ChatOpenAI`** 作为 LLM 客户端（兼容 DeepSeek API），通过设置 `base_url` 指向 DeepSeek 接口实现复用。结合 `prompts/` 中的模板完成结构化日报生成，便于未来切换至其他兼容 OpenAI 协议的模型。

#### 操作步骤

**步骤 1：选择日期**
- 使用日期选择器选择要生成日报的日期
- 默认为昨天

**步骤 2：查看运营数据**
- 页面显示该日期的核心指标（销售额、订单量、客单价等）
- 显示品类销售 TOP3 和异常事件

**步骤 3：生成日报**
- 点击"生成运营日报"按钮
- 等待 10-30 秒（AI 正在撰写报告）

**步骤 4：查看和下载日报**

生成的日报包含以下章节：

| 章节 | 内容 |
|------|------|
| 昨日概览 | 一句话总结经营状况 |
| 核心指标解读 | 分析销售额、订单量等变化原因 |
| 品类洞察 | TOP3 品类表现分析 |
| 风险提示 | 基于异常事件的风险预警 |
| 今日策略建议 | 3-5 条可执行建议 |

**步骤 5：下载日报**
- 点击"下载日报"按钮
- 文件保存为 Markdown 格式：`运营日报_YYYY-MM-DD.md`

### 4.6 模型评估体系

> **技术说明**：RetaiMind 内置完整的 RAG 和 Agent 效果评估体系，支持**离线评估**（无需调用 LLM，零成本）和**在线评估**（调用真实 LLM 端到端验证）。评估结果支持 Markdown / JSON / HTML 可视化仪表盘三种输出格式。

#### 评估体系构成

| 模块 | 评估对象 | 核心指标 |
|------|---------|---------|
| **RAG 评估** | 检索+生成 | Hit Rate、MRR、关键词命中率、可回答率、综合得分 |
| **Agent 评估** | 工具选择+执行 | 精确匹配率、F1分数、工具函数通过率、综合得分 |
| **报告生成** | 结果汇总 | Markdown 报告、JSON 原始数据 |
| **可视化** | 结果呈现 | 交互式 HTML 仪表盘（雷达图、柱状图、评分卡片） |

#### 4.5.1 运行评估

在项目根目录下执行：

```powershell
# 进入项目目录
cd f:\trae1\RetaiMind\retailmind

# 离线评估（推荐，零成本）
python run_eval.py --agent-only

# 在线评估（需要 DeepSeek API Key）
python run_eval.py --online

# 仅评估 RAG 或 Agent
python run_eval.py --rag-only
python run_eval.py --agent-only
```

#### 4.5.2 评估输出文件

评估完成后，结果保存在 `data/eval/results/` 目录下：

| 文件 | 格式 | 说明 |
|------|------|------|
| `eval_report_*.md` | Markdown | 人类可读的表格报告，含所有指标数值 |
| `eval_report_*.json` | JSON | 程序可解析的原始数据，用于历史对比 |
| `eval_dashboard_*.html` | HTML | **交互式可视化仪表盘**（浏览器打开） |

#### 4.5.3 查看可视化仪表盘

1. 在文件管理器中打开 `data/eval/results/` 目录
2. 双击最新的 `eval_dashboard_*.html` 文件
3. 浏览器中查看包含以下内容的仪表盘：

| 图表 | 类型 | 说明 |
|------|------|------|
| 综合得分卡片 | 卡片 | RAG / Agent 总分一目了然 |
| 仪表盘（Gauge） | 仪表盘 | 红黄绿分区显示综合得分 |
| 多维度雷达图 | 雷达 | 各维度得分均衡度可视化 |
| 检索指标 | 柱状图 | Hit Rate / MRR 对比 |
| 生成指标 | 柱状图 | 关键词命中率 / 可回答率 |
| 工具选择指标 | 柱状图 | 精确匹配率 / F1 / 精确率 / 召回率 |
| 工具通过率 | 柱状图 | 每个工具通过/失败状态（绿/红） |
| 品类/任务类型对比 | 分组柱状图 | 按品类或任务类型分组对比 |

> 💡 **提示**：HTML 仪表盘支持鼠标悬停查看数值、缩放、导出图片等交互操作。

#### 4.5.4 评估数据集

| 数据集文件 | 用途 | 规模 |
|-----------|------|------|
| `data/eval/rag_eval_dataset.json` | RAG 评估 | 16 道问答对，覆盖流程/参数/管理类 |
| `data/eval/agent_eval_dataset.json` | Agent 评估 | 12 个任务，覆盖单工具/多工具/无需工具 |

如需添加更多评估样本，直接在对应 JSON 文件中追加问题/任务即可。

---

## 五、常见问题排查

### 问题 1：启动时报错 "ModuleNotFoundError"

**现象**：
```
ModuleNotFoundError: No module named 'streamlit'
```

**原因**：依赖未安装或未激活虚拟环境

**解决**：
```powershell
# 确保在项目目录下
cd f:\trae1\RetaiMind\retailmind

# 如果使用了虚拟环境，先激活
.\venv\Scripts\Activate.ps1

# 重新安装依赖
pip install -r requirements.txt
```

### 问题 2：API Key 未配置

**现象**：页面显示 "⚠️ 未配置 API Key"

**解决**：
1. 检查项目根目录下是否有 `.env` 文件
2. 检查文件内容是否正确：`DEEPSEEK_API_KEY=sk-xxxx`
3. 确认 Key 前面没有空格或引号
4. 重启应用

### 问题 3：API 调用失败

**现象**：问答或日报生成时报错 "401 Unauthorized"

**可能原因**：
- API Key 错误或已失效
- 账户余额不足

**解决**：
1. 登录 DeepSeek 平台检查 API Key
2. 检查账户余额是否充足
3. 重新创建 API Key 并更新 `.env` 文件

### 问题 4：知识库构建失败

**现象**：点击"构建知识库"后报错

**可能原因**：
- 网络问题导致 Embedding 模型下载失败
- ChromaDB 目录权限问题
- 知识库目录下文档加载失败（Windows 路径问题）

**解决**：
```powershell
# 1. 清空向量库目录
Remove-Item -Path ".\vector_db\*" -Recurse -Force

# 2. 手动测试 Embedding 模型下载（已配置 hf-mirror.com 镜像加速）
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('BAAI/bge-large-zh-v1.5'); print('模型下载成功')"

# 3. 重新启动应用并构建知识库
streamlit run app.py
```

> **说明**：项目已通过手动遍历 `data/knowledge/` 目录的方式加载文档（不使用 LangChain `DirectoryLoader`），从而规避了 Windows 下目录加载的兼容性问题。如仍报错，请检查 `data/knowledge/` 目录是否存在 `.txt` 文件。

### 问题 5：会员分析报错 "qcut" 错误

**现象**：
```
ValueError: Bin edges must be unique
```

**原因**：数据中存在大量重复值导致分箱失败

**解决**：该问题已在代码中修复。系统采用 `_safe_qcut` 方法，优先使用 `pd.qcut` 并设置 `duplicates="drop"`，失败时自动降级使用 `rank+cut` 方式分箱；分群阶段使用基于 RFM 分数与平均值比较的**手动规则分群**（不依赖 KMeans 聚类），确保分群结果稳定且可解释。

### 问题 6：HuggingFace 模型下载缓慢或失败

**现象**：构建知识库时长时间卡住，或报 `ConnectionError`、`HFValidationError` 等下载相关错误。

**原因**：默认从 HuggingFace 官方源下载 `BAAI/bge-large-zh-v1.5` 模型，国内网络访问可能不稳定。

**解决**：项目已在代码中配置 **HF 镜像加速**（`hf-mirror.com`）。如仍出现下载失败，可手动设置环境变量后重新运行：

```powershell
# 临时设置（当前 PowerShell 会话有效）
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 重新启动应用
streamlit run app.py
```

或永久设置（写入用户环境变量）：
```powershell
[System.Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
```

### 问题 7：Agent 无响应

**现象**：库存预警 Agent 提问后长时间无响应

**解决**：
1. 检查网络连接
2. 检查 DeepSeek 账户余额
3. 查看终端是否有错误信息
4. 尝试简化问题，如 "SKU000001 的库存是多少？"

> **说明**：库存 Agent 采用 DeepSeek 原生 Tool Calling，不依赖 LangChain Agent。如终端报 `tool_calls` 解析错误，请确认 `openai` 包版本符合 `requirements.txt` 中 `>=2.0.0` 的要求。

### 问题 8：Streamlit 端口被占用

**现象**：启动时报错 `Port 8501 is already in use` 或浏览器无法打开页面。

**原因**：默认端口 8501 已被其他进程占用（可能是上次未正常退出的 Streamlit 实例）。

**解决**：指定其他端口启动：

```powershell
streamlit run app.py --server.port 8502
```

然后手动在浏览器访问 `http://localhost:8502`。

也可先结束占用进程：
```powershell
# 查找占用 8501 端口的进程
Get-NetTCPConnection -LocalPort 8501 | Select-Object OwningProcess
# 结束对应进程（替换 <PID> 为上一步输出的进程 ID）
Stop-Process -Id <PID> -Force
```

### 问题 9：PowerShell 执行策略错误

**现象**：
```
无法加载文件 ... 因为在此系统上禁止运行脚本
```

**解决**：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 六、数据替换指南

### 6.1 替换会员交易数据

将您的真实会员数据保存为 CSV 文件，包含以下字段：

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| customer_id | 字符串 | 会员唯一标识 | CUST0001 |
| last_purchase_date | 日期 | 最近购买日期 | 2026-06-15 |
| frequency | 整数 | 购买次数 | 12 |
| monetary | 浮点数 | 消费总金额 | 580.5 |
| preferred_category | 字符串 | 偏好品类 | 食品 |

将文件放置到 `data/raw/member_transactions.csv`，系统会自动加载。

### 6.2 替换库存数据

CSV 文件字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| sku_id | 字符串 | SKU 唯一标识 |
| stock_qty | 整数 | 当前库存数量 |
| safety_stock | 整数 | 安全库存线 |
| weekly_sales | 整数 | 最近 7 天销量 |
| category | 字符串 | 品类 |

文件路径：`data/raw/inventory.csv`

### 6.3 替换销售预测数据

CSV 文件字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| sku_id | 字符串 | SKU 唯一标识 |
| forecast_7d | 整数 | 未来 7 天预测销量 |

文件路径：`data/raw/sales_forecast.csv`

### 6.4 添加知识库文档

将您的文档（`.txt` 或 `.md` 格式）放入 `data/knowledge/` 目录，然后在知识库问答页面重新点击"构建知识库"。

> **建议**：每个文档聚焦一个主题，长度在 500-2000 字之间效果最佳。

---

## 七、部署与分享

### 7.1 局域网访问

启动时指定 server 地址：

```powershell
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

其他设备通过 `http://你的IP地址:8501` 访问。

### 7.2 免费云端部署（Streamlit Cloud）

**步骤 1**：将项目推送到 GitHub 仓库

**步骤 2**：访问 https://share.streamlit.io/

**步骤 3**：连接 GitHub 仓库，选择 `app.py` 作为入口

**步骤 4**：在 Secrets 中配置 `DEEPSEEK_API_KEY`

> **注意**：Streamlit Cloud 不支持持久化文件存储，向量库每次部署后需要重新构建。

### 7.3 Docker 部署

创建 `Dockerfile`：

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

构建并运行：

```bash
docker build -t retailmind .
docker run -p 8501:8501 -e DEEPSEEK_API_KEY=sk-xxx retailmind
```

---

## 附录：快捷命令速查表

| 操作 | 命令 |
|------|------|
| 进入项目目录 | `cd f:\trae1\RetaiMind\retailmind` |
| 激活虚拟环境 | `.\venv\Scripts\Activate.ps1` |
| 安装依赖 | `pip install -r requirements.txt` |
| 启动应用 | `streamlit run app.py` |
| 启动（指定端口） | `streamlit run app.py --server.port 8502` |
| 启动（局域网） | `streamlit run app.py --server.address 0.0.0.0` |
| 运行所有单元测试 | `python -m pytest tests/ -v` |
| 运行评估（离线 Agent） | `python run_eval.py --agent-only` |
| 运行评估（离线 RAG） | `python run_eval.py --rag-only` |
| 运行评估（在线全量） | `python run_eval.py --online` |
| 退出虚拟环境 | `deactivate` |
| 停止应用 | `Ctrl + C` |

---

> 本操作指导书基于 RetailMind 项目当前版本编写。如有问题，请参考开发指南或联系开发者。
