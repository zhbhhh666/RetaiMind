import os
import json
import pytest
import tempfile
from pathlib import Path


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
            "per_question": []
        },
        "generation": {
            "total_questions": 16,
            "keyword_hit_rate": 0.8125,
            "answerability_rate": 0.9375,
            "avg_answer_length": 120.5,
            "avg_query_time_ms": 1500.0,
            "per_question": []
        }
    }


@pytest.fixture
def sample_agent_result():
    """提供示例 Agent 评估结果。"""
    return {
        "overall": {
            "tool_selection_exact_match_rate": 0.833,
            "tool_selection_f1": 0.9,
            "tool_function_pass_rate": 1.0,
            "overall_score": 91.0
        },
        "tool_selection": {
            "total_tasks": 12,
            "exact_match_rate": 0.833,
            "avg_precision": 0.95,
            "avg_recall": 0.87,
            "avg_f1": 0.9,
            "per_task": []
        },
        "tool_functions": {
            "total_tools": 7,
            "passed_tools": 7,
            "pass_rate": 1.0,
            "per_tool": {
                "inventory_query": {"passed": True, "detail": "ok"},
                "sales_forecast": {"passed": True, "detail": "ok"},
                "list_low_stock_items": {"passed": True, "detail": "ok"},
                "list_all_skus": {"passed": True, "detail": "ok"},
                "get_stock_summary": {"passed": True, "detail": "ok"},
                "get_category_analysis": {"passed": True, "detail": "ok"},
                "get_restock_recommendation": {"passed": True, "detail": "ok"}
            }
        }
    }


class TestEvaluationReporterInit:
    """测试 EvaluationReporter 初始化。"""

    def test_init_with_valid_dir(self, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path / "results"))
        assert os.path.exists(str(tmp_path / "results"))

    def test_init_with_empty_dir_raises(self):
        from core.eval_reporter import EvaluationReporter
        with pytest.raises(ValueError, match="output_dir 不能为空"):
            EvaluationReporter(output_dir="")

    def test_init_with_none_dir_raises(self):
        from core.eval_reporter import EvaluationReporter
        with pytest.raises(ValueError):
            EvaluationReporter(output_dir=None)


class TestGenerateReport:
    """测试报告生成。"""

    def test_generate_markdown_report(self, sample_rag_result, sample_agent_result, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))
        report = reporter.generate_report(
            rag_result=sample_rag_result,
            agent_result=sample_agent_result,
            report_format="markdown"
        )
        assert "# 模型评估报告" in report
        assert "RAG 系统评估" in report
        assert "库存 Agent 评估" in report
        assert "综合得分" in report
        assert "总览" in report

    def test_generate_json_report(self, sample_rag_result, sample_agent_result, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))
        report = reporter.generate_report(
            rag_result=sample_rag_result,
            agent_result=sample_agent_result,
            report_format="json"
        )
        data = json.loads(report)
        assert "timestamp" in data
        assert "rag" in data
        assert "agent" in data

    def test_generate_report_both_none_raises(self, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="不能同时为空"):
            reporter.generate_report(rag_result=None, agent_result=None)

    def test_generate_report_rag_only(self, sample_rag_result, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))
        report = reporter.generate_report(rag_result=sample_rag_result)
        assert "RAG 系统评估" in report
        assert "库存 Agent 评估" not in report

    def test_generate_report_agent_only(self, sample_agent_result, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))
        report = reporter.generate_report(agent_result=sample_agent_result)
        assert "库存 Agent 评估" in report
        assert "RAG 系统评估" not in report


class TestSaveReport:
    """测试报告保存。"""

    def test_save_report_creates_files(self, sample_rag_result, sample_agent_result, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))
        md_path = reporter.save_report(
            rag_result=sample_rag_result,
            agent_result=sample_agent_result,
            filename="test_report"
        )
        assert os.path.exists(md_path)
        assert os.path.exists(str(tmp_path / "test_report.json"))

    def test_save_report_default_filename(self, sample_rag_result, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))
        md_path = reporter.save_report(rag_result=sample_rag_result)
        assert os.path.exists(md_path)
        assert md_path.endswith(".md")


class TestCompareReports:
    """测试报告对比。"""

    def test_compare_reports_rag(self, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))

        report_a = {"rag": {"overall": {"overall_score": 80.0, "retrieval_hit_rate": 0.8}}}
        report_b = {"rag": {"overall": {"overall_score": 85.0, "retrieval_hit_rate": 0.85}}}

        path_a = str(tmp_path / "a.json")
        path_b = str(tmp_path / "b.json")
        with open(path_a, "w") as f:
            json.dump(report_a, f)
        with open(path_b, "w") as f:
            json.dump(report_b, f)

        comparison = reporter.compare_reports(path_a, path_b)
        assert "rag" in comparison
        assert "overall_score" in comparison["rag"]
        assert comparison["rag"]["overall_score"]["delta"] == 5.0

    def test_compare_reports_both_empty_rag_agent(self, tmp_path):
        from core.eval_reporter import EvaluationReporter
        reporter = EvaluationReporter(output_dir=str(tmp_path))

        report_a = {}
        report_b = {}
        path_a = str(tmp_path / "a.json")
        path_b = str(tmp_path / "b.json")
        with open(path_a, "w") as f:
            json.dump(report_a, f)
        with open(path_b, "w") as f:
            json.dump(report_b, f)

        comparison = reporter.compare_reports(path_a, path_b)
        assert "rag" not in comparison
        assert "agent" not in comparison
