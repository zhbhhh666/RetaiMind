import streamlit as st
import os
import shutil
from core.rag_engine import SupplyChainRAG
from core.config import Config

st.set_page_config(page_title="知识库问答", page_icon="📚", layout="wide")

st.title("📚 供应链知识库问答")
st.subheader("基于 RAG 的智能文档问答系统")

with st.expander("🔧 环境信息（排查用）", expanded=False):
    st.write(f"**HF_ENDPOINT**: `{os.getenv('HF_ENDPOINT', '未设置')}`")
    st.write(f"**知识库目录**: `{Config.KNOWLEDGE_DIR}`")
    st.write(f"**知识库目录存在**: `{os.path.exists(Config.KNOWLEDGE_DIR)}`")
    st.write(f"**向量库目录**: `{Config.VECTOR_DB_DIR}`")
    st.write(f"**Embedding 模型**: `{Config.EMBEDDING_MODEL}`")
    st.write(f"**工作目录**: `{os.getcwd()}`")

api_key = Config.DEEPSEEK_API_KEY

if not api_key:
    st.error("请先配置 DeepSeek API Key！")
    st.stop()

rag_init_error = None

@st.cache_resource
def get_rag_engine():
    try:
        rag = SupplyChainRAG(api_key=api_key, db_dir=Config.VECTOR_DB_DIR)
        if rag.has_vectorstore():
            if not rag.vectorstore:
                rag.load_existing_vectorstore()
            rag.create_qa_chain()
            st.success(f"已加载向量库，共 {rag.get_document_count()} 个文档片段")
        return rag, None
    except ImportError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)

rag, rag_init_error = get_rag_engine()

if rag_init_error:
    st.error(f"RAG 引擎初始化失败：{rag_init_error}")
    st.info("请安装所需依赖后刷新页面：")
    st.code("pip install langchain_huggingface langchain_chroma langchain_classic", language="bash")
else:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.info(f"向量库状态：{'已加载 ' + str(rag.get_document_count()) + ' 个文档片段' if rag.get_document_count() > 0 else '空'}")

    with col2:
        if st.button("🔄 重建知识库", use_container_width=True):
            if os.path.exists(Config.VECTOR_DB_DIR):
                shutil.rmtree(Config.VECTOR_DB_DIR, ignore_errors=True)
            st.cache_resource.clear()
            st.rerun()

    if rag.get_document_count() == 0:
        st.warning("向量库为空，请先构建知识库")
        if st.button("📖 构建知识库", type="primary"):
            with st.spinner("正在加载文档并构建向量库..."):
                chunks = rag.load_documents(Config.KNOWLEDGE_DIR)
                st.info(f"加载了 {len(chunks)} 个文档片段")
                rag.build_vectorstore(chunks)
                rag.create_qa_chain()
            st.success("知识库构建完成！")
            st.rerun()

    st.divider()

    question = st.text_input("请输入您的问题：", placeholder="例如：缺货商品如何处理？")

    if question:
        if rag.get_document_count() == 0:
            st.error("请先构建知识库！")
        else:
            with st.spinner("正在检索并生成答案..."):
                try:
                    if not rag.qa_chain and rag.vectorstore:
                        rag.create_qa_chain()
                    result = rag.query(question)
                    
                    st.subheader("💡 回答")
                    st.markdown(result["answer"])
                    
                    st.subheader("📄 参考来源")
                    sources = set()
                    for source in result["sources"]:
                        filepath = source.get("source", "")
                        if filepath:
                            filename = os.path.basename(filepath)
                            sources.add(filename)
                    
                    if sources:
                        for i, source in enumerate(sources, 1):
                            st.write(f"{i}. {source}")
                    else:
                        st.write("未找到相关参考文档")
                
                except Exception as e:
                    st.error(f"查询失败：{str(e)}")

st.divider()

st.subheader("📚 知识库文档列表")
st.write("当前知识库包含以下文档：")

docs_dir = Config.KNOWLEDGE_DIR
if os.path.exists(docs_dir):
    for f in sorted(os.listdir(docs_dir)):
        fp = os.path.join(docs_dir, f)
        if os.path.isfile(fp):
            size = os.path.getsize(fp)
            st.write(f"- {f} ({size} 字节)")
else:
    st.write("文档目录不存在")
