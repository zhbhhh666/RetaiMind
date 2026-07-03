"""EvalVisualizer 可视化器单元测试。"""

import os
import json
import pytest


@pytest.fixture
def sample_rag_result():
    """提供示例 RAG 评估结果。"""
    return {
        "overall": {
            "total_questions": 16,
            "retrieval_hit_rate": 0.875,
            "retrieval_mrr": 0.75,
            "generation_keyword_hit_rate": 0.8125,
            "generation_answerability_rate": 0.9375,
            "overall_score": 82.5
        },
        "retrieval": {
            "total_questions": 16,
            "hit_rate": 0.875,
            "mrr": 0.75,
            "avg_retrieval_time_ms": 15.5,
            "top_k": 5,
            "per_question": [
                {"id": "rag_001", "hit": True, "mrr_rank": 1, "retrieval_time_ms": 12.0},
                {"id": "rag_002", "hit": False, "mrr_rank": 0, "retrieval_time_ms": 18.0},
            ]
        },
        "generation": {
            "total_questions": 16,
            "keyword_hit_rate": 0.8125,
            "answerability_rate": 0.9375,
            "avg_answer_length": 120.5,
            "avg_query_time_ms": 1500.0,
            "per_question": [
                {"id": "rag_001", "keyword_score": 0.8, "is_answerable": True, "answer_length": 110},
                {"id": "rag_002", "keyword_score": 0.0, "is_answerable": False, "answer_length": 50},
            ]
        }
    }


@pytest.fixture
def sample_agent_result():
    """提供示例 Agent 评估结果。"""
    return {
        "overall": {
            "tool_selection_exact_match_rate": 0.833,
            "tool_selection_f1": 0.9,
            "tool_function_pass_rate": 0.857,
            "overall_score": 91.0
        },
        "tool_selection": {
            "total_tasks": 12,
            "exact_match_rate": 0.833,
            "avg_precision": 0.95,
            "avg_recall": 0.87,
            "avg_f1": 0.9,
            "per_task": [
                {"id": "agent_001", "query": "SKU000001 库存", "expected_tools": ["inventory_query"], "predicted_tools": ["inventory_query"], "exact_match": True, "precision": 1.0, "recall": 1.0, "f1": 1.0},
                {"id": "agent_002", "query": "低库存商品", "expected_tools": ["list_low_stock_items"], "predicted_tools": ["list_low_stock_items"], "exact_match": True, "precision": 1.0, "recall": 1.0, "f1": 1.0},
            ]
        },
        "tool_functions": {
            "total_tools": 7,
            "passed_tools": 6,
            "pass_rate": 0.857,
            "per_tool": {
                "inventory_query": {"passed": True, "detail": "ok"},
                "sales_forecast": {"passed": True, "detail": "ok"},
                "list_low_stock_items": {"passed": True, "detail": "ok"},
                "list_all_skus": {"passed": True, "detail": "ok"},
                "get_stock_summary": {"passed": True, "detail": "ok"},
                "get_category_analysis": {"passed": True, "detail": "ok"},
                "get_restock_recommendation": {"passed": False, "detail": "fail"},
            }
        }
    }


class TestEvalVisualizerInit:
    """测试 EvalVisualizer 初始化。"""

    def test_init_with_valid_dir(self, tmp_path):
        from core.eval_visualizer import EvalVisualizer
        viz = EvalVisualizer(output_dir=str(tmp_path / "viz"))
        assert os.path.exists(str(tmp_path / "viz"))

    def test_init_with_empty_dir_raises(self):
        from core.eval_visualizer import EvalVisualizer
        with pytest.raises(ValueError, match="output_dir 不能为空"):
            EvalVisualizer(output_dir="")

    def test_init_with_none_dir_raises(self):
        from core.eval_visualizer import EvalVisualizer
        with pytest.raises(ValueError):
            EvalVisualizer(output_dir=None)

    def test_init_creates_dir(self, tmp_path):
        from core.eval_visualizer import EvalVisualizer
        target = str(tmp_path / "nested" / "results")
        viz = EvalVisualizer(output_dir=target)
        assert os.path.exists(target)


class TestRenderDashboard:
    """测试仪表盘生成。"""

    def test_render_both(self, sample_rag_result, sample_agent_result, tmp_path):
        from core.eval_visualizer import EvalVisualizer
        viz = EvalVisualizer(output_dir=str(tmp_path))
        html_path = viz.render_dashboard(
            rag_result=sample_rag_result,
            agent_result=sample_agent_result,
            filename="test_both",
        )
        assert os.path.exists(html_path)
        assert html_path.endswith(".html")
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "plotly" in content.lower()

    def test_render_rag_only(self, sample_rag_result, tmp_path):
        from core.eval_visualizer import EvalVisualizer
        viz = EvalVisualizer(output_dir=str(tmp_path))
        html_path = viz.render_dashboard(
            rag_result=sample_rag_result,
            filename="test_rag",
        )
        assert os.path.exists(html_path)

    def test_render_agent_only(self, sample_agent_result, tmp_path):
        from core.eval_visualizer import EvalVisualizer
        viz = EvalVisualizer(output_dir=str(tmp_path))
        html_path = viz.render_dashboard(
            agent_result=sample_agent_result,
            filename="test_agent",
        )
        assert os.path.exists(html_path)

    def test_render_both_none_raises(self, tmp_path):
        from core.eval_visualizer import EvalVisualizer
        viz = EvalVisualizer(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="不能同时为空"):
            viz.render_dashboard()

    def test_render_default_filename(self, sample_rag_result, tmp_path):
        from core.eval_visualizer import EvalVisualizer
        viz = EvalVisualizer(output_dir=str(tmp_path))
        html_path = viz.render_dashboard(rag_result=sample_rag_result)
        assert os.path.exists(html_path)
        assert "eval_dashboard_" in os.path.basename(html_path)

    def test_render_agent_with_failing_tool(self, sample_agent_result, tmp_path):
        """测试包含失败工具时的渲染。"""
        from core.eval_visualizer import EvalVisualizer
        viz = EvalVisualizer(output_dir=str(tmp_path))
        html_path = viz.render_dashboard(
            agent_result=sample_agent_result,
            filename="test_fail",
        )
        assert os.path.exists(html_path)
