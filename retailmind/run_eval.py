"""RetaiMind 评估一键运行入口。

用法：
    python run_eval.py              # 离线评估（无需 API Key）
    python run_eval.py --online     # 在线评估（需要配置 DeepSeek API Key）
    python run_eval.py --rag-only   # 仅评估 RAG
    python run_eval.py --agent-only # 仅评估 Agent

输出：
    - Markdown 报告 (.md)
    - JSON 原始数据 (.json)
    - 交互式 HTML 仪表盘 (.html)
"""

import argparse
import os
import sys
import json

# 将项目根目录加入搜索路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from core.rag_evaluator import RAGEvaluator
from core.agent_evaluator import AgentEvaluator
from core.eval_reporter import EvaluationReporter
from core.eval_visualizer import EvalVisualizer
from core.logger import get_logger

logger = get_logger("run_eval")

# ---- 路径常量 ----
RAG_DATASET = os.path.join(BASE_DIR, "data", "eval", "rag_eval_dataset.json")
AGENT_DATASET = os.path.join(BASE_DIR, "data", "eval", "agent_eval_dataset.json")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "eval", "results")


def run_rag_evaluation(online: bool = False) -> dict:
    """运行 RAG 评估。

    Args:
        online: 是否在线评估（需要 RAG 引擎和 API Key）。

    Returns:
        RAG 评估结果字典，离线模式下返回 None。
    """
    if not online:
        logger.info("RAG 评估跳过（离线模式需要 RAG 引擎，使用 --online 启用）")
        return None

    try:
        from core.rag_engine import SupplyChainRAG
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        db_dir = os.path.join(BASE_DIR, "data", "vector_db")
        knowledge_dir = os.path.join(BASE_DIR, "data", "knowledge")

        rag_engine = SupplyChainRAG(
            api_key=api_key,
            db_dir=db_dir,
        )
        rag_engine.load_documents(knowledge_dir)
        rag_engine.build_vectorstore()
        rag_engine.create_qa_chain()

        evaluator = RAGEvaluator(rag_engine=rag_engine, dataset_path=RAG_DATASET)
        result = evaluator.evaluate_all()
        logger.info("RAG 评估完成，综合得分: %s/100", result["overall"]["overall_score"])
        return result
    except Exception as e:
        logger.error("RAG 评估失败: %s", e, exc_info=True)
        return None


def run_agent_evaluation(online: bool = False) -> dict:
    """运行 Agent 评估。

    Args:
        online: 是否在线评估（调用 LLM）。

    Returns:
        Agent 评估结果字典。
    """
    try:
        from core.data_loader import DataLoader
        from core.agent_inventory import InventoryAgent
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        data_loader = DataLoader()
        inventory_df = data_loader.load_inventory_data()
        forecast_df = data_loader.load_sales_forecast()

        agent = InventoryAgent(
            api_key=api_key,
            inventory_df=inventory_df,
            forecast_df=forecast_df,
        )

        evaluator = AgentEvaluator(agent=agent, dataset_path=AGENT_DATASET)
        result = evaluator.evaluate_all(online=online)
        logger.info("Agent 评估完成，综合得分: %s/100", result["overall"]["overall_score"])
        return result
    except Exception as e:
        logger.error("Agent 评估失败: %s", e, exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="RetaiMind 模型评估工具")
    parser.add_argument("--online", action="store_true", help="启用在线评估（调用 LLM）")
    parser.add_argument("--rag-only", action="store_true", help="仅评估 RAG")
    parser.add_argument("--agent-only", action="store_true", help="仅评估 Agent")
    args = parser.parse_args()

    print("=" * 60)
    print("  RetaiMind 模型评估体系")
    print("=" * 60)

    rag_result = None
    agent_result = None

    # ---- 运行评估 ----
    if not args.agent_only:
        print("\n[1/3] RAG 评估...")
        rag_result = run_rag_evaluation(online=args.online)

    if not args.rag_only:
        print("\n[2/3] Agent 评估...")
        agent_result = run_agent_evaluation(online=args.online)

    # ---- 生成报告 ----
    print("\n[3/3] 生成报告与可视化...")
    if not rag_result and not agent_result:
        print("\n❌ 评估未产出结果，请检查配置后重试。")
        print("   离线模式: python run_eval.py --agent-only")
        print("   在线模式: python run_eval.py --online")
        sys.exit(1)

    reporter = EvaluationReporter(output_dir=RESULTS_DIR)
    report_path = reporter.save_report(
        rag_result=rag_result,
        agent_result=agent_result,
    )

    visualizer = EvalVisualizer(output_dir=RESULTS_DIR)
    dashboard_path = visualizer.render_dashboard(
        rag_result=rag_result,
        agent_result=agent_result,
    )

    # ---- 输出摘要 ----
    print("\n" + "=" * 60)
    print("  评估完成！结果摘要")
    print("=" * 60)

    if rag_result:
        overall = rag_result.get("overall", {})
        print(f"\n  📊 RAG 系统")
        print(f"     综合得分:        {overall.get('overall_score', 0)}/100")
        print(f"     检索命中率:      {overall.get('retrieval_hit_rate', 0):.1%}")
        print(f"     MRR:            {overall.get('retrieval_mrr', 0):.4f}")
        print(f"     关键词命中率:    {overall.get('generation_keyword_hit_rate', 0):.1%}")
        print(f"     可回答率:        {overall.get('generation_answerability_rate', 0):.1%}")

    if agent_result:
        overall = agent_result.get("overall", {})
        print(f"\n  🤖 库存 Agent")
        print(f"     综合得分:        {overall.get('overall_score', 0)}/100")
        print(f"     工具精确匹配率:  {overall.get('tool_selection_exact_match_rate', 0):.1%}")
        print(f"     工具选择 F1:     {overall.get('tool_selection_f1', 0):.4f}")
        print(f"     工具函数通过率:  {overall.get('tool_function_pass_rate', 0):.1%}")

    print(f"\n  📄 报告文件:     {report_path}")
    print(f"  📈 可视化仪表盘: {dashboard_path}")
    print(f"\n  💡 在浏览器中打开 HTML 文件即可查看交互式图表。")
    print("=" * 60)


if __name__ == "__main__":
    main()
