import sys
# 修复: Streamlit 模块监控器尝试检查 torch._classes.__path__ 时会触发 RuntimeError
# 配合 .streamlit/config.toml 的 fileWatcherType = "none" 彻底消除警告
if 'torch._classes' in sys.modules:
    sys.modules['torch._classes'].__path__ = []  # type: ignore

import streamlit as st
from core.config import Config

st.set_page_config(
    page_title="RetailMind - 零售智能决策助手",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛒 RetailMind - 零售智能决策助手")
st.subheader("面向零售业务的智能决策助手")

st.markdown("""
## 项目概述

RetailMind 是一个面向零售业务的智能决策助手，融合 **RAG、AI Agent、数据分析** 三大技术栈，同时覆盖会员运营和供应链两大核心场景。

### 核心功能模块

| 模块 | 功能描述 |
|------|---------|
| **会员数据洞察** | 自动分析会员消费行为，生成 RFM 分级与趋势洞察 |
| **供应链知识库问答** | 基于 RAG 的供应链文档智能问答 |
| **库存预警 Agent** | AI Agent 监控库存水位，执行预警并给出补货建议 |
| **运营日报生成** | AIGC 自动生成运营日报与策略建议 |

### 技术栈

- **Web 框架**: Streamlit
- **LLM 框架**: LangChain
- **向量数据库**: ChromaDB
- **大模型**: DeepSeek API (deepseek-chat / deepseek-reasoner)
- **可视化**: Plotly

---

### 开始使用

请从左侧导航栏选择您需要的功能模块。

**注意**: 使用前请先配置 DeepSeek API Key。

当前 API Key 状态: 
""")

if Config.DEEPSEEK_API_KEY:
    st.success("✅ API Key 已配置")
else:
    st.warning("⚠️ 未配置 API Key，请在 `.env` 文件中设置 `DEEPSEEK_API_KEY`")

st.markdown("""
### 配置方式

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
```

或通过环境变量设置：
```bash
$env:DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

### 获取 DeepSeek API Key

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册并完成实名认证
3. 进入「API Keys」页面，创建新 Key
""")