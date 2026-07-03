import json
import os
import time
import re
from typing import List, Dict, Tuple, Optional
from core.logger import get_logger

logger = get_logger(__name__)


class RAGEvaluator:
    """RAG 系统效果评估器。

    提供检索阶段和生成阶段的多维度评估：
    - 检索指标：Hit Rate、MRR、检索耗时、来源准确率
    - 生成指标：关键词命中率、答案完整性、可回答率
    - 效率指标：平均响应时间

    使用无模型评估方法（关键词匹配、规则判断），不依赖外部 LLM，
    确保评估结果稳定可重复。
    """

    def __init__(self, rag_engine=None, dataset_path: str = None):
        """初始化 RAG 评估器。

        Args:
            rag_engine: SupplyChainRAG 实例，用于执行实际查询。
            dataset_path: 评估数据集 JSON 文件路径。

        Raises:
            ValueError: 当 dataset_path 为空或文件不存在时。
        """
        self.rag_engine = rag_engine

        if not dataset_path or not isinstance(dataset_path, str):
            raise ValueError("dataset_path 不能为空")
        if not os.path.exists(dataset_path):
            raise ValueError(f"数据集文件不存在: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        self.questions = self.dataset.get("questions", [])
        logger.info("RAG 评估器初始化完成，共 %d 条评估数据", len(self.questions))

    def evaluate_retrieval(self, top_k: int = 5) -> Dict:
        """评估检索阶段效果。

        通过遍历评估集中的每个问题，检查检索到的文档是否包含正确答案。
        计算 Hit Rate@k 和 MRR@k 指标。

        Args:
            top_k: 评估前 k 个检索结果，默认 5。

        Returns:
            包含 hit_rate、mrr、per_question 详情的字典。

        Raises:
            ValueError: 当 rag_engine 未初始化时。
        """
        if not self.rag_engine:
            raise ValueError("rag_engine 未初始化，无法执行检索评估")
        if not self.rag_engine.vectorstore:
            raise ValueError("vectorstore 未初始化，请先构建或加载向量库")

        logger.info("开始检索评估，top_k=%d，共 %d 个问题", top_k, len(self.questions))

        results = []
        total_hit = 0
        total_mrr = 0.0
        total_retrieval_time = 0.0

        for q in self.questions:
            q_id = q["id"]
            question = q["question"]
            relevant_docs = set(q.get("relevant_docs", []))

            start_time = time.time()
            try:
                retriever = self.rag_engine.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": top_k, "fetch_k": 20}
                )
                retrieved_docs = retriever.invoke(question)
                retrieval_time = time.time() - start_time
                total_retrieval_time += retrieval_time

                retrieved_sources = []
                for doc in retrieved_docs:
                    source = doc.metadata.get("source", "")
                    basename = os.path.basename(source)
                    retrieved_sources.append(basename)

                hit = False
                mrr_rank = 0
                for rank, source in enumerate(retrieved_sources[:top_k], 1):
                    if source in relevant_docs:
                        hit = True
                        mrr_rank = rank
                        break

                if hit:
                    total_hit += 1
                    total_mrr += 1.0 / mrr_rank

                results.append({
                    "id": q_id,
                    "question": question,
                    "expected_docs": list(relevant_docs),
                    "retrieved_docs": retrieved_sources[:top_k],
                    "hit": hit,
                    "mrr_rank": mrr_rank if hit else 0,
                    "retrieval_time_ms": round(retrieval_time * 1000, 2)
                })
            except Exception as e:
                logger.error("检索评估失败 [%s]: %s", q_id, e, exc_info=True)
                results.append({
                    "id": q_id,
                    "question": question,
                    "error": str(e),
                    "hit": False,
                    "mrr_rank": 0,
                    "retrieval_time_ms": 0
                })

        n = len(self.questions)
        summary = {
            "total_questions": n,
            "hit_rate": round(total_hit / n, 4) if n > 0 else 0,
            "mrr": round(total_mrr / n, 4) if n > 0 else 0,
            "avg_retrieval_time_ms": round(total_retrieval_time / n * 1000, 2) if n > 0 else 0,
            "top_k": top_k,
            "per_question": results
        }

        logger.info("检索评估完成: Hit Rate=%.2f%%, MRR=%.4f",
                    summary["hit_rate"] * 100, summary["mrr"])
        return summary

    def evaluate_generation(self, use_llm: bool = False) -> Dict:
        """评估生成阶段效果。

        通过关键词匹配和规则判断评估回答质量：
        - keyword_hit_rate: 参考答案关键词命中率
        - answerability_rate: 可回答率（未返回"无法回答"的比例）
        - avg_answer_length: 平均回答长度

        Args:
            use_llm: 是否使用 LLM 做深度评估（预留接口，当前仅支持无模型评估）。

        Returns:
            包含各生成指标的字典。

        Raises:
            ValueError: 当 rag_engine 未初始化时。
        """
        if not self.rag_engine:
            raise ValueError("rag_engine 未初始化，无法执行生成评估")
        if not self.rag_engine.qa_chain:
            raise ValueError("qa_chain 未初始化，请先调用 create_qa_chain()")

        logger.info("开始生成评估，共 %d 个问题", len(self.questions))

        results = []
        total_keyword_hit = 0
        total_answerable = 0
        total_answer_length = 0
        total_query_time = 0.0

        for q in self.questions:
            q_id = q["id"]
            question = q["question"]
            ground_truth = q.get("ground_truth", "")

            start_time = time.time()
            try:
                answer_data = self.rag_engine.query(question)
                query_time = time.time() - start_time
                total_query_time += query_time

                answer = answer_data.get("answer", "")
                sources = answer_data.get("sources", [])

                keyword_score = self._calc_keyword_score(answer, ground_truth)
                is_answerable = "无法回答" not in answer and "根据现有资料" not in answer

                if keyword_score > 0:
                    total_keyword_hit += 1
                if is_answerable:
                    total_answerable += 1
                total_answer_length += len(answer)

                results.append({
                    "id": q_id,
                    "question": question,
                    "answer": answer,
                    "ground_truth": ground_truth,
                    "keyword_score": keyword_score,
                    "is_answerable": is_answerable,
                    "answer_length": len(answer),
                    "source_count": len(sources),
                    "query_time_ms": round(query_time * 1000, 2)
                })
            except Exception as e:
                logger.error("生成评估失败 [%s]: %s", q_id, e, exc_info=True)
                results.append({
                    "id": q_id,
                    "question": question,
                    "error": str(e),
                    "keyword_score": 0,
                    "is_answerable": False,
                    "answer_length": 0,
                    "source_count": 0,
                    "query_time_ms": 0
                })

        n = len(self.questions)
        summary = {
            "total_questions": n,
            "keyword_hit_rate": round(total_keyword_hit / n, 4) if n > 0 else 0,
            "answerability_rate": round(total_answerable / n, 4) if n > 0 else 0,
            "avg_answer_length": round(total_answer_length / n, 1) if n > 0 else 0,
            "avg_query_time_ms": round(total_query_time / n * 1000, 2) if n > 0 else 0,
            "per_question": results
        }

        logger.info("生成评估完成: 关键词命中率=%.2f%%, 可回答率=%.2f%%",
                    summary["keyword_hit_rate"] * 100, summary["answerability_rate"] * 100)
        return summary

    def evaluate_all(self, top_k: int = 5) -> Dict:
        """执行完整的 RAG 评估（检索 + 生成）。

        Args:
            top_k: 检索评估的 top_k 参数。

        Returns:
            包含 retrieval、generation 和 overall 汇总的完整评估结果。
        """
        logger.info("启动完整 RAG 评估")
        retrieval_result = self.evaluate_retrieval(top_k=top_k)
        generation_result = self.evaluate_generation()

        overall = {
            "total_questions": retrieval_result["total_questions"],
            "retrieval_hit_rate": retrieval_result["hit_rate"],
            "retrieval_mrr": retrieval_result["mrr"],
            "generation_keyword_hit_rate": generation_result["keyword_hit_rate"],
            "generation_answerability_rate": generation_result["answerability_rate"],
            "avg_total_time_ms": round(
                retrieval_result["avg_retrieval_time_ms"] + generation_result["avg_query_time_ms"], 2
            ),
            "overall_score": round(
                (retrieval_result["hit_rate"] * 0.4 +
                 retrieval_result["mrr"] * 0.2 +
                 generation_result["keyword_hit_rate"] * 0.3 +
                 generation_result["answerability_rate"] * 0.1) * 100, 2
            )
        }

        return {
            "retrieval": retrieval_result,
            "generation": generation_result,
            "overall": overall
        }

    def _calc_keyword_score(self, answer: str, ground_truth: str) -> float:
        """计算答案与参考答案之间的关键词匹配得分。

        提取参考答案中的关键名词/数字短语，检查答案中是否包含。
        得分范围 0-1，越高表示关键词匹配越多。

        Args:
            answer: 系统生成的回答。
            ground_truth: 参考答案。

        Returns:
            关键词匹配得分（0-1 之间）。
        """
        if not answer or not ground_truth:
            return 0.0

        keywords = self._extract_keywords(ground_truth)
        if not keywords:
            return 0.0

        hit_count = 0
        answer_lower = answer.lower()
        for kw in keywords:
            if kw.lower() in answer_lower:
                hit_count += 1

        return round(hit_count / len(keywords), 4)

    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词用于匹配。

        提取策略：
        1. 数字+单位组合（如"7天"、"95%"、"1000元"）
        2. 明确的名词短语（3-8个中文字符）
        3. 去重

        Args:
            text: 输入文本。

        Returns:
            关键词列表。
        """
        keywords = []

        num_patterns = [
            r"\d+[-~]\d+\s*[天%元个件分钟小时日]",
            r"\d+\s*[天%元个件分钟小时日]",
            r"\d+\.\d+\s*[%]",
        ]
        for pattern in num_patterns:
            matches = re.findall(pattern, text)
            keywords.extend(matches)

        cn_phrases = re.findall(r"[\u4e00-\u9fa5]{3,8}", text)
        stopwords = {"的是", "有哪些", "是什么", "包括", "以及", "等等", "可以", "需要",
                     "分别", "如何", "什么", "哪些", "多少", "几个", "哪种", "怎样", "还是"}
        for phrase in cn_phrases:
            if phrase not in stopwords and len(phrase) >= 3:
                keywords.append(phrase)

        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:20]

    def evaluate_by_category(self, top_k: int = 5) -> Dict:
        """按问题类别分组评估。

        Returns:
            按 category 分组的评估结果字典。
        """
        if not self.rag_engine:
            raise ValueError("rag_engine 未初始化")

        categories = {}
        for q in self.questions:
            cat = q.get("category", "未分类")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(q)

        results = {}
        original_questions = self.questions
        try:
            for cat, cat_questions in categories.items():
                self.questions = cat_questions
                cat_result = self.evaluate_all(top_k=top_k)
                results[cat] = {
                    "count": len(cat_questions),
                    "hit_rate": cat_result["overall"]["retrieval_hit_rate"],
                    "mrr": cat_result["overall"]["retrieval_mrr"],
                    "keyword_hit_rate": cat_result["overall"]["generation_keyword_hit_rate"],
                    "answerability_rate": cat_result["overall"]["generation_answerability_rate"],
                    "overall_score": cat_result["overall"]["overall_score"]
                }
        finally:
            self.questions = original_questions

        return results
