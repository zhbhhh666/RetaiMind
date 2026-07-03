import json
import os
from datetime import datetime
from typing import Dict, Optional
from core.logger import get_logger

logger = get_logger(__name__)


class EvaluationReporter:
    """模型评估报告生成器。

    将 RAG 评估和 Agent 评估结果汇总为格式化报告，
    支持 Markdown 和 JSON 两种输出格式。
    """

    def __init__(self, output_dir: str = "./data/eval/results"):
        """初始化评估报告生成器。

        Args:
            output_dir: 评估结果输出目录。

        Raises:
            ValueError: 当 output_dir 为空时。
        """
        if not output_dir or not isinstance(output_dir, str):
            raise ValueError("output_dir 不能为空")

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("评估报告生成器初始化，输出目录: %s", output_dir)

    def generate_report(self, rag_result: Optional[Dict] = None,
                        agent_result: Optional[Dict] = None,
                        report_format: str = "markdown") -> str:
        """生成综合评估报告。

        Args:
            rag_result: RAG 评估结果字典。
            agent_result: Agent 评估结果字典。
            report_format: 报告格式，支持 "markdown" 或 "json"。

        Returns:
            格式化的评估报告字符串。

        Raises:
            ValueError: 当两种结果都为空时。
        """
        if not rag_result and not agent_result:
            raise ValueError("rag_result 和 agent_result 不能同时为空")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if report_format == "json":
            return self._generate_json_report(rag_result, agent_result, timestamp)
        else:
            return self._generate_markdown_report(rag_result, agent_result, timestamp)

    def save_report(self, rag_result: Optional[Dict] = None,
                    agent_result: Optional[Dict] = None,
                    filename: str = None) -> str:
        """生成并保存评估报告到文件。

        同时保存 Markdown 报告和 JSON 原始数据。

        Args:
            rag_result: RAG 评估结果。
            agent_result: Agent 评估结果。
            filename: 文件名前缀，默认使用时间戳。

        Returns:
            保存的报告文件路径。
        """
        if filename is None:
            filename = datetime.now().strftime("eval_report_%Y%m%d_%H%M%S")

        md_report = self.generate_report(rag_result, agent_result, "markdown")
        json_report = self.generate_report(rag_result, agent_result, "json")

        md_path = os.path.join(self.output_dir, f"{filename}.md")
        json_path = os.path.join(self.output_dir, f"{filename}.json")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_report)

        logger.info("评估报告已保存: %s, %s", md_path, json_path)
        return md_path

    def _generate_markdown_report(self, rag_result: Optional[Dict],
                                   agent_result: Optional[Dict],
                                   timestamp: str) -> str:
        """生成 Markdown 格式的评估报告。"""
        lines = []
        lines.append("# 模型评估报告")
        lines.append("")
        lines.append(f"**生成时间**: {timestamp}")
        lines.append("")

        overall_scores = []
        if rag_result:
            overall_scores.append(("RAG 系统", rag_result.get("overall", {}).get("overall_score", 0)))
        if agent_result:
            overall_scores.append(("库存 Agent", agent_result.get("overall", {}).get("overall_score", 0)))

        if overall_scores:
            lines.append("## 总览")
            lines.append("")
            lines.append("| 评估对象 | 综合得分 |")
            lines.append("|----------|----------|")
            for name, score in overall_scores:
                lines.append(f"| {name} | {score}/100 |")
            lines.append("")

        if rag_result:
            lines.extend(self._format_rag_markdown(rag_result))

        if agent_result:
            lines.extend(self._format_agent_markdown(agent_result))

        lines.append("---")
        lines.append("")
        lines.append("*本报告由 RetailMind 评估体系自动生成*")

        return "\n".join(lines)

    def _format_rag_markdown(self, result: Dict) -> list:
        """格式化 RAG 评估结果为 Markdown。"""
        lines = []
        overall = result.get("overall", {})
        retrieval = result.get("retrieval", {})
        generation = result.get("generation", {})

        lines.append("## RAG 系统评估")
        lines.append("")
        lines.append(f"**综合得分**: {overall.get('overall_score', 0)}/100")
        lines.append("")

        lines.append("### 检索阶段")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 问题总数 | {retrieval.get('total_questions', 0)} |")
        lines.append(f"| Hit Rate@{retrieval.get('top_k', 5)} | {retrieval.get('hit_rate', 0):.2%} |")
        lines.append(f"| MRR | {retrieval.get('mrr', 0):.4f} |")
        lines.append(f"| 平均检索耗时 | {retrieval.get('avg_retrieval_time_ms', 0)} ms |")
        lines.append("")

        lines.append("### 生成阶段")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 关键词命中率 | {generation.get('keyword_hit_rate', 0):.2%} |")
        lines.append(f"| 可回答率 | {generation.get('answerability_rate', 0):.2%} |")
        lines.append(f"| 平均回答长度 | {generation.get('avg_answer_length', 0)} 字 |")
        lines.append(f"| 平均查询耗时 | {generation.get('avg_query_time_ms', 0)} ms |")
        lines.append("")

        return lines

    def _format_agent_markdown(self, result: Dict) -> list:
        """格式化 Agent 评估结果为 Markdown。"""
        lines = []
        overall = result.get("overall", {})
        tool_selection = result.get("tool_selection", {})
        tool_functions = result.get("tool_functions", {})

        lines.append("## 库存 Agent 评估")
        lines.append("")
        lines.append(f"**综合得分**: {overall.get('overall_score', 0)}/100")
        lines.append("")

        lines.append("### 工具选择评估")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 任务总数 | {tool_selection.get('total_tasks', 0)} |")
        lines.append(f"| 精确匹配率 | {tool_selection.get('exact_match_rate', 0):.2%} |")
        lines.append(f"| 平均精确率 | {tool_selection.get('avg_precision', 0):.2%} |")
        lines.append(f"| 平均召回率 | {tool_selection.get('avg_recall', 0):.2%} |")
        lines.append(f"| 平均 F1 | {tool_selection.get('avg_f1', 0):.4f} |")
        lines.append("")

        lines.append("### 工具函数评估")
        lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|------|-----|")
        lines.append(f"| 工具总数 | {tool_functions.get('total_tools', 0)} |")
        lines.append(f"| 通过数 | {tool_functions.get('passed_tools', 0)} |")
        lines.append(f"| 通过率 | {tool_functions.get('pass_rate', 0):.2%} |")
        lines.append("")

        per_tool = tool_functions.get("per_tool", {})
        if per_tool:
            lines.append("#### 各工具详情")
            lines.append("")
            lines.append("| 工具 | 状态 |")
            lines.append("|------|------|")
            for tool_name, info in per_tool.items():
                status = "✅ 通过" if info.get("passed") else "❌ 失败"
                lines.append(f"| {tool_name} | {status} |")
            lines.append("")

        if "online" in result:
            online = result["online"]
            lines.append("### 在线评估（完整流程）")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 回答率 | {online.get('answer_rate', 0):.2%} |")
            lines.append(f"| 相关率 | {online.get('relevance_rate', 0):.2%} |")
            lines.append(f"| 平均耗时 | {online.get('avg_elapsed_ms', 0)} ms |")
            lines.append(f"| 平均步骤数 | {online.get('avg_step_count', 0)} |")
            lines.append("")

        return lines

    def _generate_json_report(self, rag_result: Optional[Dict],
                               agent_result: Optional[Dict],
                               timestamp: str) -> str:
        """生成 JSON 格式的评估报告。"""
        report = {
            "timestamp": timestamp,
            "rag": rag_result,
            "agent": agent_result,
        }
        return json.dumps(report, ensure_ascii=False, indent=2)

    def compare_reports(self, report_path_a: str, report_path_b: str) -> Dict:
        """对比两份评估报告，计算指标变化。

        Args:
            report_path_a: 基准报告 JSON 路径。
            report_path_b: 对比报告 JSON 路径。

        Returns:
            包含各指标变化的字典。
        """
        with open(report_path_a, "r", encoding="utf-8") as f:
            report_a = json.load(f)
        with open(report_path_b, "r", encoding="utf-8") as f:
            report_b = json.load(f)

        comparison = {}

        if report_a.get("rag") and report_b.get("rag"):
            comparison["rag"] = self._compare_metrics(
                report_a["rag"].get("overall", {}),
                report_b["rag"].get("overall", {}),
                ["retrieval_hit_rate", "retrieval_mrr", "generation_keyword_hit_rate",
                 "generation_answerability_rate", "overall_score"]
            )

        if report_a.get("agent") and report_b.get("agent"):
            comparison["agent"] = self._compare_metrics(
                report_a["agent"].get("overall", {}),
                report_b["agent"].get("overall", {}),
                ["tool_selection_exact_match_rate", "tool_selection_f1",
                 "tool_function_pass_rate", "overall_score"]
            )

        return comparison

    def _compare_metrics(self, a: Dict, b: Dict, metrics: list) -> Dict:
        """对比两组指标值。"""
        result = {}
        for metric in metrics:
            val_a = a.get(metric, 0)
            val_b = b.get(metric, 0)
            delta = val_b - val_a
            result[metric] = {
                "baseline": val_a,
                "current": val_b,
                "delta": round(delta, 4),
                "delta_pct": round(delta / val_a * 100, 2) if val_a != 0 else 0
            }
        return result
