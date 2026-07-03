"""SupplyChainRAG 模块单元测试。

覆盖初始化、文档加载、向量库状态检查等不依赖实际 LLM 调用的场景。
"""
import pytest
import os
import tempfile
import shutil
from core.rag_engine import SupplyChainRAG


@pytest.fixture(scope="module")
def rag_engine():
    """提供 RAG 引擎 fixture（使用临时目录，不影响生产数据）。"""
    tmp_dir = tempfile.mkdtemp(prefix="rag_test_")
    engine = SupplyChainRAG(api_key="sk-test-key", db_dir=tmp_dir)
    yield engine
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def knowledge_dir():
    """提供知识库目录 fixture，指向项目实际知识库。"""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge")


class TestSupplyChainRAGInit:
    """SupplyChainRAG 初始化测试。"""

    def test_init_with_valid_params(self, tmp_path):
        """有效参数应成功初始化。"""
        rag = SupplyChainRAG(
            api_key="sk-test",
            db_dir=str(tmp_path / "test_db")
        )
        assert rag.api_key == "sk-test"
        assert rag.vectorstore is None
        assert rag.qa_chain is None
        assert rag.embeddings is not None

    def test_init_with_empty_api_key_raises(self, tmp_path):
        """空 api_key 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="api_key"):
            SupplyChainRAG(api_key="", db_dir=str(tmp_path))

    def test_init_with_empty_db_dir_raises(self):
        """空 db_dir 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="db_dir"):
            SupplyChainRAG(api_key="sk-test", db_dir="")


class TestLoadDocuments:
    """load_documents 方法测试。"""

    def test_load_documents_from_knowledge_dir(self, rag_engine, knowledge_dir):
        """应从实际知识库目录加载文档。"""
        chunks = rag_engine.load_documents(knowledge_dir)
        assert len(chunks) > 0
        # 每个 chunk 应有 page_content 和 metadata
        for chunk in chunks:
            assert hasattr(chunk, "page_content")
            assert hasattr(chunk, "metadata")
            assert "source" in chunk.metadata

    def test_load_documents_nonexistent_dir_returns_empty(self, rag_engine, tmp_path):
        """不存在的目录应返回空列表（不抛出异常）。"""
        result = rag_engine.load_documents(str(tmp_path / "nonexistent"))
        assert result == []

    def test_load_documents_empty_dir_returns_empty(self, rag_engine, tmp_path):
        """空目录应返回空列表。"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = rag_engine.load_documents(str(empty_dir))
        assert result == []

    def test_load_documents_empty_dir_str_raises(self, rag_engine):
        """空字符串 docs_dir 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="docs_dir"):
            rag_engine.load_documents("")


class TestVectorstoreStatus:
    """向量库状态检查测试。"""

    def test_has_vectorstore_returns_bool(self, rag_engine):
        """has_vectorstore 应返回布尔值。"""
        result = rag_engine.has_vectorstore()
        assert isinstance(result, bool)

    def test_get_document_count_returns_int(self, rag_engine):
        """get_document_count 应返回整数。"""
        count = rag_engine.get_document_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_document_count_zero_when_no_vectorstore(self, tmp_path):
        """未构建向量库时文档数应为 0。"""
        rag = SupplyChainRAG(api_key="sk-test", db_dir=str(tmp_path / "fresh_db"))
        assert rag.get_document_count() == 0


class TestBuildVectorstoreValidation:
    """build_vectorstore 输入验证测试。"""

    def test_build_with_empty_chunks_raises(self, rag_engine):
        """空 chunks 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="chunks"):
            rag_engine.build_vectorstore([])


class TestQueryValidation:
    """query 方法输入验证测试。"""

    def test_query_without_qa_chain_raises(self, rag_engine):
        """未创建 qa_chain 时调用 query 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="create_qa_chain"):
            rag_engine.query("测试问题")

    def test_query_empty_question_raises(self, rag_engine):
        """空 question 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="question"):
            rag_engine.query("")
