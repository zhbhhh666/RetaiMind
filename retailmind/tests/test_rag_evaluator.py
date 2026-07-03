import os
import json
import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def rag_eval_dataset_path(tmp_path):
    """创建 RAG 评估数据集测试文件。"""
    data = {
        "dataset_name": "测试RAG评估集",
        "version": "1.0",
        "total_questions": 3,
        "questions": [
            {
                "id": "t1",
                "question": "安全库存线是多少天？",
                "ground_truth": "安全库存线一般为7-14天的销售量",
                "relevant_docs": ["supply_chain_policy.txt"],
                "category": "参数类",
                "difficulty": "简单"
            },
            {
                "id": "t2",
                "question": "会员有几个等级？",
                "ground_truth": "四个等级：普通会员、银卡会员、金卡会员、钻石会员",
                "relevant_docs": ["member_service.txt"],
                "category": "参数类",
                "difficulty": "简单"
            },
            {
                "id": "t3",
                "question": "补货策略有哪些？",
                "ground_truth": "三种：定量补货、定期补货、动态补货",
                "relevant_docs": ["inventory_management.txt"],
                "category": "策略类",
                "difficulty": "中等"
            }
        ]
    }
    path = tmp_path / "rag_eval.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return str(path)


class TestRAGEvaluatorInit:
    """测试 RAGEvaluator 初始化。"""

    def test_init_with_valid_dataset(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        assert len(evaluator.questions) == 3

    def test_init_with_empty_path_raises(self):
        from core.rag_evaluator import RAGEvaluator
        with pytest.raises(ValueError, match="dataset_path 不能为空"):
            RAGEvaluator(dataset_path="")

    def test_init_with_none_path_raises(self):
        from core.rag_evaluator import RAGEvaluator
        with pytest.raises(ValueError):
            RAGEvaluator(dataset_path=None)

    def test_init_with_nonexistent_path_raises(self, tmp_path):
        from core.rag_evaluator import RAGEvaluator
        with pytest.raises(ValueError, match="数据集文件不存在"):
            RAGEvaluator(dataset_path=str(tmp_path / "nonexistent.json"))

    def test_init_with_none_engine(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(rag_engine=None, dataset_path=rag_eval_dataset_path)
        assert evaluator.rag_engine is None


class TestKeywordExtraction:
    """测试关键词提取功能。"""

    def test_extract_keywords_from_text(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        keywords = evaluator._extract_keywords("安全库存线一般为7-14天的销售量")
        assert len(keywords) > 0
        assert any("7" in kw for kw in keywords) or any("14" in kw for kw in keywords)

    def test_extract_keywords_empty_text(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        keywords = evaluator._extract_keywords("")
        assert keywords == []

    def test_extract_keywords_deduplication(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        keywords = evaluator._extract_keywords("库存库存库存")
        assert len(keywords) <= 3


class TestKeywordScore:
    """测试关键词匹配得分。"""

    def test_perfect_match(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        score = evaluator._calc_keyword_score(
            "安全库存线一般为7-14天的销售量",
            "安全库存线一般为7-14天的销售量"
        )
        assert score > 0.5

    def test_partial_match(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        score = evaluator._calc_keyword_score(
            "安全库存线由品类经理设定",
            "安全库存线由品类经理根据历史销售数据设定，一般为7-14天的销售量"
        )
        assert 0 < score < 1

    def test_no_match(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        score = evaluator._calc_keyword_score(
            "今天天气真好",
            "安全库存线一般为7-14天的销售量"
        )
        assert score == 0.0

    def test_empty_answer(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        score = evaluator._calc_keyword_score("", "参考答案")
        assert score == 0.0

    def test_empty_ground_truth(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        score = evaluator._calc_keyword_score("回答内容", "")
        assert score == 0.0


class TestRetrievalEvaluation:
    """测试检索评估。"""

    def test_evaluate_retrieval_without_engine_raises(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        with pytest.raises(ValueError, match="rag_engine 未初始化"):
            evaluator.evaluate_retrieval()


class TestGenerationEvaluation:
    """测试生成评估。"""

    def test_evaluate_generation_without_engine_raises(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        with pytest.raises(ValueError, match="rag_engine 未初始化"):
            evaluator.evaluate_generation()


class TestEvaluateAll:
    """测试完整评估。"""

    def test_evaluate_all_without_engine_raises(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        with pytest.raises(ValueError):
            evaluator.evaluate_all()


class TestEvaluateByCategory:
    """测试按类别评估。"""

    def test_evaluate_by_category_without_engine_raises(self, rag_eval_dataset_path):
        from core.rag_evaluator import RAGEvaluator
        evaluator = RAGEvaluator(dataset_path=rag_eval_dataset_path)
        with pytest.raises(ValueError):
            evaluator.evaluate_by_category()
