import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List, Dict
from core.logger import get_logger

logger = get_logger(__name__)


class SupplyChainRAG:
    """供应链知识库 RAG 引擎。

    负责文档加载、文本切分、向量存储构建与问答检索。
    使用本地 HuggingFace Embedding 模型（BAAI/bge-large-zh-v1.5）生成向量，
    通过 ChromaDB 持久化存储，并基于 DeepSeek API 实现问答。
    """

    def __init__(self, api_key: str, db_dir: str = "./vector_db"):
        """初始化 RAG 引擎。

        Args:
            api_key: DeepSeek API Key。
            db_dir: 向量数据库持久化目录路径。

        Raises:
            ValueError: 当 api_key 为空或 db_dir 为空字符串时。
            ImportError: 当 langchain_huggingface 或 langchain_chroma 依赖缺失时。
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key 不能为空")
        if not db_dir or not isinstance(db_dir, str):
            raise ValueError("db_dir 不能为空")

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as e:
            raise ImportError(
                "缺少 langchain_huggingface 依赖，请运行: pip install langchain_huggingface"
            ) from e
        try:
            from langchain_chroma import Chroma
        except ImportError as e:
            raise ImportError(
                "缺少 langchain_chroma 依赖，请运行: pip install langchain_chroma"
            ) from e
        try:
            from langchain_classic.chains import RetrievalQA
        except ImportError as e:
            raise ImportError(
                "缺少 langchain_classic 依赖，请运行: pip install langchain_classic"
            ) from e

        self._HuggingFaceEmbeddings = HuggingFaceEmbeddings
        self._Chroma = Chroma
        self._RetrievalQA = RetrievalQA

        logger.info("初始化 RAG 引擎，向量库目录: %s", db_dir)
        self.embeddings = self._HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-zh-v1.5",
            model_kwargs={"device": "cpu"}
        )
        self.db_dir = db_dir
        self.vectorstore = None
        self.qa_chain = None
        self.api_key = api_key
        os.makedirs(db_dir, exist_ok=True)

    def load_documents(self, docs_dir: str) -> List:
        """从目录加载 .txt 和 .md 文档并切分为片段。

        使用 os.walk 遍历目录，兼容 Windows 路径。
        按 500 字符切分，重叠 50 字符。

        Args:
            docs_dir: 知识库文档目录路径。

        Returns:
            切分后的文档片段列表；目录不存在时返回空列表。

        Raises:
            ValueError: 当 docs_dir 为空字符串时。
        """
        if not docs_dir or not isinstance(docs_dir, str):
            raise ValueError("docs_dir 不能为空")

        if not os.path.exists(docs_dir):
            logger.warning("文档目录不存在: %s", docs_dir)
            return []

        documents = []
        file_count = 0
        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                if file.endswith('.txt') or file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                        if not text.strip():
                            logger.warning("文件内容为空，已跳过: %s", file_path)
                            continue
                        documents.append(Document(page_content=text, metadata={"source": file_path}))
                        file_count += 1
                    except Exception as e:
                        logger.error("加载文件失败 %s: %s", file_path, e, exc_info=True)

        logger.info("成功加载 %d 个文档文件", file_count)

        if not documents:
            logger.warning("未加载到任何文档，请检查目录 %s 是否包含 .txt/.md 文件", docs_dir)
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", " "]
        )
        chunks = splitter.split_documents(documents)
        logger.info("文档切分完成，共 %d 个片段", len(chunks))
        return chunks

    def build_vectorstore(self, chunks: List):
        """构建向量数据库并持久化到磁盘。

        Args:
            chunks: 文档片段列表（由 load_documents 生成）。

        Raises:
            ValueError: 当 chunks 为空列表时。
        """
        if not chunks:
            raise ValueError("chunks 不能为空，请先调用 load_documents 加载文档")

        logger.info("开始构建向量库，共 %d 个片段", len(chunks))
        self.vectorstore = self._Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        logger.info("向量库构建完成")

    def load_existing_vectorstore(self):
        """从磁盘加载已有的向量数据库。

        如果目录不存在或加载失败，vectorstore 将保持为 None。
        """
        if not os.path.exists(self.db_dir):
            logger.warning("向量库目录不存在: %s", self.db_dir)
            return
        try:
            self.vectorstore = self._Chroma(
                collection_name="default",
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
            logger.info("已加载现有向量库，文档数: %d", self.get_document_count())
        except Exception as e:
            logger.error("加载现有向量库失败: %s", e, exc_info=True)
            self.vectorstore = None

    def has_vectorstore(self) -> bool:
        """检查向量数据库是否已存在且包含文档。

        Returns:
            True 如果向量库存在且文档数 > 0，否则 False。
        """
        if self.vectorstore:
            return self.get_document_count() > 0
        if os.path.exists(os.path.join(self.db_dir, "chroma.sqlite3")):
            try:
                temp_vs = self._Chroma(
                    collection_name="default",
                    persist_directory=self.db_dir,
                    embedding_function=self.embeddings
                )
                count = temp_vs._collection.count()
                return count > 0
            except Exception as e:
                logger.debug("检查向量库状态时出错: %s", e)
                return False
        return False

    def create_qa_chain(self, prompt_template: str = None):
        """创建 RAG 问答链。

        Args:
            prompt_template: 自定义 Prompt 模板字符串，包含 {context} 和 {question} 占位符。
                            如果为 None，使用默认中文 Prompt。

        Raises:
            ValueError: 当未先调用 build_vectorstore 或 load_existing_vectorstore 时。
        """
        if not self.vectorstore:
            raise ValueError("vectorstore 未初始化，请先调用 build_vectorstore 或 load_existing_vectorstore")

        if prompt_template is None:
            prompt_template = """基于以下上下文回答问题：
{context}

问题：{question}

要求：
1. 如果上下文包含答案，请直接回答
2. 如果上下文不包含答案，请说明"根据现有资料无法回答"
3. 在回答末尾标注参考文档来源

回答："""

        llm = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.3
        )

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        self.qa_chain = self._RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("QA 链创建完成")

    def query(self, question: str) -> Dict:
        """执行 RAG 问答检索。

        Args:
            question: 用户问题字符串。

        Returns:
            包含 "answer"（回答文本）和 "sources"（来源文档元数据列表）的字典。

        Raises:
            ValueError: 当 question 为空或未先调用 create_qa_chain() 时。
        """
        if not question or not isinstance(question, str):
            raise ValueError("question 不能为空")
        if not self.qa_chain:
            raise ValueError("请先调用 create_qa_chain()")

        logger.info("执行查询: %s", question[:100])
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }

    def get_document_count(self) -> int:
        """获取向量数据库中的文档片段数量。

        Returns:
            文档片段数量，如果向量库未加载则返回 0。
        """
        if self.vectorstore:
            try:
                return self.vectorstore._collection.count()
            except Exception as e:
                logger.debug("获取文档数量失败: %s", e)
                return 0
        return 0
