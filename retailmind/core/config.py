import os
from dotenv import load_dotenv

load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """从环境变量或 Streamlit Secrets 读取配置值。

    优先级：系统环境变量 > .env 文件 > Streamlit Cloud Secrets。
    本地开发用 .env，云端部署用 Streamlit Secrets。
    """
    # 1. 先从系统环境变量 / .env 读取
    val = os.getenv(key, "")
    if val:
        return val
    # 2. 再从 Streamlit Secrets 读取（云端部署）
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


_core_dir = os.path.dirname(os.path.abspath(__file__))
_base_dir = os.path.dirname(_core_dir)


class Config:
    DEEPSEEK_API_KEY = _get_secret("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    
    CHAT_MODEL = "deepseek-chat"
    REASONER_MODEL = "deepseek-reasoner"
    
    EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
    
    VECTOR_DB_DIR = os.path.join(_base_dir, "vector_db")
    
    TEMPERATURE_QA = 0.3
    TEMPERATURE_AGENT = 0.1
    TEMPERATURE_REPORT = 0.5
    MAX_TOKENS = 4096
    
    DATA_DIR = os.path.join(_base_dir, "data")
    RAW_DATA_DIR = os.path.join(_base_dir, "data", "raw")
    PROCESSED_DATA_DIR = os.path.join(_base_dir, "data", "processed")
    KNOWLEDGE_DIR = os.path.join(_base_dir, "data", "knowledge")