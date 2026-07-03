import os
import json
import pytest
import tempfile
import pandas as pd
from pathlib import Path


@pytest.fixture
def agent_eval_dataset_path(tmp_path):
    """创建 Agent 评估数据集测试文件。"""
    data = {
        "dataset_name": "测试Agent评估集",
        "version": "1.0",
        "total_tasks": 5,
        "tasks": [
            {
                "id": "a1",
                "query": "SKU001 的库存是多少？",
                "expected_tools": ["inventory_query"],
                "expected_sku": "SKU001",
                "task_type": "单工具调用",
                "difficulty": "简单"
            },
            {
                "id": "a2",
                "query": "哪些商品库存不足？",
                "expected_tools": ["list_low_stock_items"],
                "expected_sku": None,
                "task_type": "单工具调用",
                "difficulty": "简单"
            },
            {
                "id": "a3",
                "query": "给我库存汇总",
                "expected_tools": ["get_stock_summary"],
                "expected_sku": None,
                "task_type": "单工具调用",
                "difficulty": "简单"
            },
            {
                "id": "a4",
                "query": "按品类分析库存",
                "expected_tools": ["get_category_analysis"],
                "expected_sku": None,
                "task_type": "单工具调用",
                "difficulty": "简单"
            },
            {
                "id": "a5",
                "query": "今天天气不错",
                "expected_tools": [],
                "expected_sku": None,
                "task_type": "无需工具",
                "difficulty": "简单"
            }
        ]
    }
    path = tmp_path / "agent_eval.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return str(path)


@pytest.fixture
def sample_inventory_for_eval():
    """提供测试用库存数据。"""
    return pd.DataFrame({
        "sku_id": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "stock_qty": [100, 30, 200, 10, 50],
        "safety_stock": [50, 50, 80, 40, 30],
        "weekly_sales": [20, 25, 30, 15, 10],
        "category": ["食品", "服装", "数码", "食品", "美妆"]
    })


@pytest.fixture
def sample_forecast_for_eval():
    """提供测试用预测数据。"""
    return pd.DataFrame({
        "sku_id": ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"],
        "forecast_7d": [25, 30, 35, 20, 15]
    })


class TestAgentEvaluatorInit:
    """测试 AgentEvaluator 初始化。"""

    def test_init_with_valid_dataset(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        assert len(evaluator.tasks) == 5

    def test_init_with_empty_path_raises(self):
        from core.agent_evaluator import AgentEvaluator
        with pytest.raises(ValueError, match="dataset_path 不能为空"):
            AgentEvaluator(dataset_path="")

    def test_init_with_none_path_raises(self):
        from core.agent_evaluator import AgentEvaluator
        with pytest.raises(ValueError):
            AgentEvaluator(dataset_path=None)

    def test_init_with_nonexistent_path_raises(self, tmp_path):
        from core.agent_evaluator import AgentEvaluator
        with pytest.raises(ValueError, match="数据集文件不存在"):
            AgentEvaluator(dataset_path=str(tmp_path / "nonexistent.json"))

    def test_init_with_none_agent(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(agent=None, dataset_path=agent_eval_dataset_path)
        assert evaluator.agent is None


class TestToolPrediction:
    """测试工具预测功能。"""

    def test_predict_inventory_query_with_sku(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        tools = evaluator._predict_tools("SKU001 的库存是多少？")
        assert "inventory_query" in tools

    def test_predict_low_stock(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        tools = evaluator._predict_tools("哪些商品库存不足？")
        assert "list_low_stock_items" in tools

    def test_predict_stock_summary(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        tools = evaluator._predict_tools("给我库存汇总数据")
        assert "get_stock_summary" in tools

    def test_predict_category_analysis(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        tools = evaluator._predict_tools("按品类分析一下库存")
        assert "get_category_analysis" in tools

    def test_predict_restock(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        tools = evaluator._predict_tools("需要补多少货？")
        assert "get_restock_recommendation" in tools

    def test_predict_forecast(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        tools = evaluator._predict_tools("SKU001 未来销量预测")
        assert "sales_forecast" in tools

    def test_predict_no_tool(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        tools = evaluator._predict_tools("今天天气真好")
        assert len(tools) == 0 or "inventory_query" not in tools


class TestToolSelectionEvaluation:
    """测试工具选择评估。"""

    def test_evaluate_tool_selection_returns_metrics(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        result = evaluator.evaluate_tool_selection()

        assert "total_tasks" in result
        assert "exact_match_rate" in result
        assert "avg_precision" in result
        assert "avg_recall" in result
        assert "avg_f1" in result
        assert "per_task" in result
        assert result["total_tasks"] == 5

    def test_exact_match_rate_in_range(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        result = evaluator.evaluate_tool_selection()
        assert 0 <= result["exact_match_rate"] <= 1

    def test_f1_in_range(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        result = evaluator.evaluate_tool_selection()
        assert 0 <= result["avg_f1"] <= 1


class TestToolFunctionsEvaluation:
    """测试工具函数评估。"""

    def test_evaluate_tool_functions_without_agent_raises(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        with pytest.raises(ValueError, match="agent 未初始化"):
            evaluator.evaluate_tool_functions()

    def test_evaluate_tool_functions_with_agent(self, agent_eval_dataset_path,
                                                sample_inventory_for_eval, sample_forecast_for_eval):
        from core.agent_evaluator import AgentEvaluator
        from core.agent_inventory import InventoryAgent

        agent = InventoryAgent(
            api_key="sk-test-key",
            inventory_df=sample_inventory_for_eval,
            forecast_df=sample_forecast_for_eval
        )
        evaluator = AgentEvaluator(agent=agent, dataset_path=agent_eval_dataset_path)
        result = evaluator.evaluate_tool_functions()

        assert "total_tools" in result
        assert "passed_tools" in result
        assert "pass_rate" in result
        assert "per_tool" in result
        assert result["total_tools"] == 7
        assert result["pass_rate"] == 1.0


class TestEvaluateAll:
    """测试完整评估。"""

    def test_evaluate_all_offline(self, agent_eval_dataset_path,
                                  sample_inventory_for_eval, sample_forecast_for_eval):
        from core.agent_evaluator import AgentEvaluator
        from core.agent_inventory import InventoryAgent

        agent = InventoryAgent(
            api_key="sk-test-key",
            inventory_df=sample_inventory_for_eval,
            forecast_df=sample_forecast_for_eval
        )
        evaluator = AgentEvaluator(agent=agent, dataset_path=agent_eval_dataset_path)
        result = evaluator.evaluate_all(online=False)

        assert "tool_selection" in result
        assert "tool_functions" in result
        assert "overall" in result
        assert "overall_score" in result["overall"]
        assert 0 <= result["overall"]["overall_score"] <= 100


class TestAnswerRelevance:
    """测试回答相关性判断。"""

    def test_relevant_answer(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        is_relevant = evaluator._check_answer_relevance(
            "SKU001 当前库存是100件，安全库存是50件",
            "SKU001 的库存是多少？"
        )
        assert is_relevant is True

    def test_irrelevant_answer(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        is_relevant = evaluator._check_answer_relevance(
            "今天天气晴朗",
            "SKU001 的库存是多少？"
        )
        assert is_relevant is False

    def test_empty_answer(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        is_relevant = evaluator._check_answer_relevance("", "问题")
        assert is_relevant is False


class TestEvaluateByTaskType:
    """测试按任务类型分组评估。"""

    def test_returns_grouped_results(self, agent_eval_dataset_path):
        from core.agent_evaluator import AgentEvaluator
        evaluator = AgentEvaluator(dataset_path=agent_eval_dataset_path)
        result = evaluator.evaluate_by_task_type()

        assert isinstance(result, dict)
        assert len(result) > 0
        for ttype, data in result.items():
            assert "count" in data
            assert "exact_match_rate" in data
            assert "avg_f1" in data
