import json
import os
import time
import re
from typing import List, Dict, Optional
from core.logger import get_logger

logger = get_logger(__name__)


class AgentEvaluator:
    """库存 Agent 效果评估器。

    提供 Agent 多维度评估指标：
    - 工具调用准确率：是否选择正确工具、参数是否正确
    - 任务完成率：用户问题是否得到有效回答
    - 效率指标：工具调用次数、总耗时
    - 回答质量：回答相关性、完整性

    支持两种评估模式：
    - 离线评估：直接评估工具函数输出（无需调用 LLM，快速稳定）
    - 在线评估：通过 Agent.run() 完整流程评估（需要 API Key）
    """

    def __init__(self, agent=None, dataset_path: str = None):
        """初始化 Agent 评估器。

        Args:
            agent: InventoryAgent 实例。
            dataset_path: 评估数据集 JSON 文件路径。

        Raises:
            ValueError: 当 dataset_path 为空或文件不存在时。
        """
        self.agent = agent

        if not dataset_path or not isinstance(dataset_path, str):
            raise ValueError("dataset_path 不能为空")
        if not os.path.exists(dataset_path):
            raise ValueError(f"数据集文件不存在: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.tasks = self.dataset.get("tasks", [])
        logger.info("Agent 评估器初始化完成，共 %d 条评估任务", len(self.tasks))

    def evaluate_tool_selection(self) -> Dict:
        """评估工具选择准确性（离线模式）。

        针对每个任务，通过关键词匹配判断应该调用哪些工具，
        与预期工具列表进行比对，计算工具选择准确率。

        Returns:
            包含工具选择准确率、精确率、召回率的评估结果。
        """
        logger.info("开始工具选择评估，共 %d 个任务", len(self.tasks))

        results = []
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        exact_match_count = 0

        for task in self.tasks:
            task_id = task["id"]
            query = task["query"]
            expected_tools = set(task.get("expected_tools", []))

            predicted_tools = self._predict_tools(query)
            predicted_set = set(predicted_tools)

            if predicted_set == expected_tools:
                exact_match_count += 1

            if len(predicted_set) > 0:
                precision = len(predicted_set & expected_tools) / len(predicted_set)
            else:
                precision = 1.0 if len(expected_tools) == 0 else 0.0

            if len(expected_tools) > 0:
                recall = len(predicted_set & expected_tools) / len(expected_tools)
            else:
                recall = 1.0 if len(predicted_set) == 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

            results.append({
                "id": task_id,
                "query": query,
                "expected_tools": list(expected_tools),
                "predicted_tools": predicted_tools,
                "exact_match": predicted_set == expected_tools,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4)
            })

        n = len(self.tasks)
        summary = {
            "total_tasks": n,
            "exact_match_rate": round(exact_match_count / n, 4) if n > 0 else 0,
            "avg_precision": round(total_precision / n, 4) if n > 0 else 0,
            "avg_recall": round(total_recall / n, 4) if n > 0 else 0,
            "avg_f1": round(total_f1 / n, 4) if n > 0 else 0,
            "per_task": results
        }

        logger.info("工具选择评估完成: 精确匹配率=%.2f%%, 平均F1=%.4f",
                    summary["exact_match_rate"] * 100, summary["avg_f1"])
        return summary

    def evaluate_tool_functions(self) -> Dict:
        """评估工具函数的输出正确性（离线模式）。

        直接调用每个工具函数，验证返回结果的格式和内容是否符合预期：
        - 返回值类型是否正确
        - 关键信息是否包含
        - 数据是否与输入 DataFrame 一致

        Returns:
            各工具函数的评估结果。
        """
        if not self.agent:
            raise ValueError("agent 未初始化，无法执行工具函数评估")

        logger.info("开始工具函数评估")
        results = {}
        total_tests = 0
        passed_tests = 0

        test_cases = {
            "inventory_query": self._test_inventory_query,
            "sales_forecast": self._test_sales_forecast,
            "list_low_stock_items": self._test_list_low_stock_items,
            "list_all_skus": self._test_list_all_skus,
            "get_stock_summary": self._test_get_stock_summary,
            "get_category_analysis": self._test_get_category_analysis,
            "get_restock_recommendation": self._test_get_restock_recommendation,
        }

        for tool_name, test_func in test_cases.items():
            total_tests += 1
            try:
                passed, detail = test_func()
                if passed:
                    passed_tests += 1
                results[tool_name] = {"passed": passed, "detail": detail}
            except Exception as e:
                logger.error("工具函数评估失败 [%s]: %s", tool_name, e, exc_info=True)
                results[tool_name] = {"passed": False, "detail": f"异常: {str(e)}"}

        summary = {
            "total_tools": total_tests,
            "passed_tools": passed_tests,
            "pass_rate": round(passed_tests / total_tests, 4) if total_tests > 0 else 0,
            "per_tool": results
        }

        logger.info("工具函数评估完成: 通过率=%.2f%% (%d/%d)",
                    summary["pass_rate"] * 100, passed_tests, total_tests)
        return summary

    def evaluate_agent_online(self) -> Dict:
        """评估 Agent 完整流程（在线模式，需要调用 LLM）。

        通过 Agent.run() 执行每个任务，评估：
        - 实际调用的工具与预期是否一致
        - 回答是否包含相关信息
        - 总耗时

        Returns:
            Agent 在线评估结果。

        Raises:
            ValueError: 当 agent 未初始化时。
        """
        if not self.agent:
            raise ValueError("agent 未初始化，无法执行在线评估")

        logger.info("开始 Agent 在线评估，共 %d 个任务", len(self.tasks))

        results = []
        total_time = 0.0
        answered_count = 0
        relevant_count = 0

        for task in self.tasks:
            task_id = task["id"]
            query = task["query"]
            expected_tools = set(task.get("expected_tools", []))

            start_time = time.time()
            try:
                result = self.agent.run(query)
                elapsed = time.time() - start_time
                total_time += elapsed

                output = result.get("output", "")
                intermediate = result.get("intermediate_steps", [])

                actual_tools = self._extract_called_tools(intermediate)
                tool_match = set(actual_tools) & expected_tools == expected_tools if expected_tools else True

                is_answered = len(output.strip()) > 10
                is_relevant = self._check_answer_relevance(output, query)

                if is_answered:
                    answered_count += 1
                if is_relevant:
                    relevant_count += 1

                results.append({
                    "id": task_id,
                    "query": query,
                    "output": output,
                    "expected_tools": list(expected_tools),
                    "actual_tools": actual_tools,
                    "tool_match": tool_match,
                    "is_answered": is_answered,
                    "is_relevant": is_relevant,
                    "step_count": len(intermediate),
                    "elapsed_ms": round(elapsed * 1000, 2)
                })
            except Exception as e:
                logger.error("Agent 在线评估失败 [%s]: %s", task_id, e, exc_info=True)
                results.append({
                    "id": task_id,
                    "query": query,
                    "error": str(e),
                    "tool_match": False,
                    "is_answered": False,
                    "is_relevant": False,
                    "step_count": 0,
                    "elapsed_ms": 0
                })

        n = len(self.tasks)
        summary = {
            "total_tasks": n,
            "answer_rate": round(answered_count / n, 4) if n > 0 else 0,
            "relevance_rate": round(relevant_count / n, 4) if n > 0 else 0,
            "avg_elapsed_ms": round(total_time / n * 1000, 2) if n > 0 else 0,
            "avg_step_count": round(
                sum(r.get("step_count", 0) for r in results) / n, 2
            ) if n > 0 else 0,
            "per_task": results
        }

        logger.info("Agent 在线评估完成: 回答率=%.2f%%, 相关率=%.2f%%",
                    summary["answer_rate"] * 100, summary["relevance_rate"] * 100)
        return summary

    def evaluate_all(self, online: bool = False) -> Dict:
        """执行完整的 Agent 评估。

        Args:
            online: 是否执行在线评估（需要 LLM API）。

        Returns:
            包含所有评估维度的汇总结果。
        """
        logger.info("启动完整 Agent 评估，在线模式=%s", online)

        tool_selection = self.evaluate_tool_selection()
        tool_functions = self.evaluate_tool_functions()

        overall = {
            "tool_selection_exact_match_rate": tool_selection["exact_match_rate"],
            "tool_selection_f1": tool_selection["avg_f1"],
            "tool_function_pass_rate": tool_functions["pass_rate"],
        }

        result = {
            "tool_selection": tool_selection,
            "tool_functions": tool_functions,
            "overall": overall
        }

        if online:
            online_result = self.evaluate_agent_online()
            result["online"] = online_result
            overall["answer_rate"] = online_result["answer_rate"]
            overall["relevance_rate"] = online_result["relevance_rate"]
            overall["avg_elapsed_ms"] = online_result["avg_elapsed_ms"]

        overall["overall_score"] = round(
            (tool_selection["avg_f1"] * 0.3 +
             tool_functions["pass_rate"] * 0.4 +
             (overall.get("relevance_rate", 0.8) if online else 0.8) * 0.3) * 100, 2
        )

        return result

    def _predict_tools(self, query: str) -> List[str]:
        """基于关键词匹配预测应该调用的工具。

        Args:
            query: 用户查询。

        Returns:
            预测的工具名称列表。
        """
        predictions = []
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["汇总", "整体", "统计", "总共有", "概况"]):
            predictions.append("get_stock_summary")

        if any(kw in query_lower for kw in ["品类", "分类", "各类", "按类"]):
            predictions.append("get_category_analysis")

        if any(kw in query_lower for kw in ["补货", "补多少", "建议", "需要补"]):
            predictions.append("get_restock_recommendation")

        if any(kw in query_lower for kw in ["低库存", "库存不足", "缺货", "低于安全", "预警"]):
            predictions.append("list_low_stock_items")

        if any(kw in query_lower for kw in ["所有sku", "全部sku", "sku列表", "有哪些sku", "列出"]):
            predictions.append("list_all_skus")

        if any(kw in query_lower for kw in ["预测", "未来", "预计", "预测销量"]):
            predictions.append("sales_forecast")

        if any(kw in query_lower for kw in ["库存是多少", "当前库存", "库存有多少", "库存查询"]):
            predictions.append("inventory_query")

        if re.search(r"sku\d+", query_lower):
            if "预测" in query_lower or "未来" in query_lower:
                if "sales_forecast" not in predictions:
                    predictions.append("sales_forecast")
            else:
                if "inventory_query" not in predictions:
                    predictions.append("inventory_query")

        return predictions

    def _extract_called_tools(self, intermediate_steps: List[str]) -> List[str]:
        """从中间步骤中提取调用的工具名称。

        Args:
            intermediate_steps: 中间步骤字符串列表。

        Returns:
            调用的工具名称列表。
        """
        tools = []
        for step in intermediate_steps:
            match = re.search(r"工具调用:\s*(\w+)", step)
            if match:
                tools.append(match.group(1))
        return tools

    def _check_answer_relevance(self, answer: str, query: str) -> bool:
        """检查回答是否与问题相关。

        通过关键词重叠度简单判断相关性。

        Args:
            answer: 回答文本。
            query: 问题文本。

        Returns:
            True 表示相关，False 表示不相关。
        """
        if not answer or not query:
            return False

        query_keywords = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", query))
        answer_keywords = set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]+", answer))

        if not query_keywords:
            return False

        overlap = query_keywords & answer_keywords
        return len(overlap) >= 1 and len(answer) > 20

    def _test_inventory_query(self) -> tuple:
        """测试 inventory_query 工具函数。"""
        sku = self.agent.inventory_df['sku_id'].iloc[0]
        expected_stock = self.agent.inventory_df[self.agent.inventory_df['sku_id'] == sku]['stock_qty'].values[0]
        result = self.agent.query_inventory(sku)
        passed = str(expected_stock) in result and sku in result
        detail = f"查询 {sku}，预期库存 {expected_stock}，结果包含: {passed}"
        return passed, detail

    def _test_sales_forecast(self) -> tuple:
        """测试 sales_forecast 工具函数。"""
        sku = self.agent.forecast_df['sku_id'].iloc[0]
        expected_forecast = self.agent.forecast_df[self.agent.forecast_df['sku_id'] == sku]['forecast_7d'].values[0]
        result = self.agent.query_forecast(sku)
        passed = str(expected_forecast) in result and sku in result
        detail = f"预测 {sku}，预期 {expected_forecast}，结果包含: {passed}"
        return passed, detail

    def _test_list_low_stock_items(self) -> tuple:
        """测试 list_low_stock_items 工具函数。"""
        result = self.agent.list_low_stock_items()
        low_stock_count = (self.agent.inventory_df['stock_qty'] < self.agent.inventory_df['safety_stock']).sum()
        if low_stock_count == 0:
            passed = "所有商品库存正常" in result
        else:
            passed = str(low_stock_count) in result or "个商品库存低于" in result
        detail = f"低库存预期 {low_stock_count} 个，结果验证: {passed}"
        return passed, detail

    def _test_list_all_skus(self) -> tuple:
        """测试 list_all_skus 工具函数。"""
        result = self.agent.list_all_skus()
        total = len(self.agent.inventory_df)
        passed = str(total) in result and "个 SKU" in result
        detail = f"总 SKU 数预期 {total}，结果验证: {passed}"
        return passed, detail

    def _test_get_stock_summary(self) -> tuple:
        """测试 get_stock_summary 工具函数。"""
        result = self.agent.get_stock_summary()
        total = len(self.agent.inventory_df)
        required = ["总 SKU 数", "库存正常", "库存偏低", "库存充足"]
        passed = str(total) in result and all(r in result for r in required)
        detail = f"汇总包含必要字段: {passed}"
        return passed, detail

    def _test_get_category_analysis(self) -> tuple:
        """测试 get_category_analysis 工具函数。"""
        result = self.agent.get_category_analysis()
        categories = self.agent.inventory_df['category'].unique()
        passed = all(cat in result for cat in categories[:3]) and "SKU数" in result
        detail = f"品类分析包含 {len(categories)} 个品类，验证: {passed}"
        return passed, detail

    def _test_get_restock_recommendation(self) -> tuple:
        """测试 get_restock_recommendation 工具函数。"""
        result = self.agent.get_restock_recommendation()
        low_stock = (self.agent.inventory_df['stock_qty'] < self.agent.inventory_df['safety_stock']).sum()
        if low_stock == 0:
            passed = "无需补货" in result
        else:
            passed = "补货建议" in result and "建议补货" in result
        detail = f"低库存 {low_stock} 个，补货建议验证: {passed}"
        return passed, detail

    def evaluate_by_task_type(self) -> Dict:
        """按任务类型分组评估。

        Returns:
            按 task_type 分组的评估结果。
        """
        task_types = {}
        for task in self.tasks:
            ttype = task.get("task_type", "未分类")
            if ttype not in task_types:
                task_types[ttype] = []
            task_types[ttype].append(task)

        results = {}
        original_tasks = self.tasks
        try:
            for ttype, type_tasks in task_types.items():
                self.tasks = type_tasks
                type_result = self.evaluate_tool_selection()
                results[ttype] = {
                    "count": len(type_tasks),
                    "exact_match_rate": type_result["exact_match_rate"],
                    "avg_precision": type_result["avg_precision"],
                    "avg_recall": type_result["avg_recall"],
                    "avg_f1": type_result["avg_f1"]
                }
        finally:
            self.tasks = original_tasks

        return results
