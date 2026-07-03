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