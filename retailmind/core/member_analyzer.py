import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, List
from core.logger import get_logger

logger = get_logger(__name__)


class MemberAnalyzer:
    """会员 RFM 分析与分群引擎。

    基于 RFM（最近购买时间、购买频次、消费金额）模型对会员进行评分和分群，
    支持流失风险识别和分群统计分析。
    """

    def __init__(self):
        self.rfm_df = None
        self.segment_labels = {
            0: "高价值会员",
            1: "潜力会员",
            2: "新会员",
            3: "流失风险会员",
            4: "沉睡会员"
        }

    def calculate_rfm(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """计算会员的 RFM 指标和评分。

        支持两种输入格式：预聚合数据（含 frequency/monetary 列）和原始交易数据。
        使用 _safe_qcut 将连续值分为 5 档评分（1-5）。

        Args:
            transactions_df: 会员交易数据，需包含 customer_id 和 last_purchase_date 列。

        Returns:
            包含 recency/frequency/monetary 及其评分列的 DataFrame。

        Raises:
            ValueError: 当 transactions_df 为空、缺少 customer_id 或 last_purchase_date 列时。
        """
        if transactions_df is None or len(transactions_df) == 0:
            raise ValueError("transactions_df 不能为空")
        required_cols = {"customer_id", "last_purchase_date"}
        missing = required_cols - set(transactions_df.columns)
        if missing:
            raise ValueError(f"transactions_df 缺少必要列: {missing}")

        logger.info("计算 RFM 指标，共 %d 条记录", len(transactions_df))
        today = datetime.now()
        df = transactions_df.copy()

        df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

        if "frequency" in df.columns and "monetary" in df.columns and df["customer_id"].nunique() == len(df):
            rfm = df.set_index("customer_id").copy()
            rfm["recency"] = rfm["last_purchase_date"].apply(lambda x: (today - x).days)
        else:
            rfm = df.groupby("customer_id").agg(
                recency=("last_purchase_date", lambda x: (today - x.max()).days),
                frequency=("customer_id", "count"),
                monetary=("monetary", "sum")
            )

        rfm["recency_score"] = self._safe_qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1], ascending=False)
        rfm["frequency_score"] = self._safe_qcut(rfm["frequency"], q=5, labels=[1, 2, 3, 4, 5])
        rfm["monetary_score"] = self._safe_qcut(rfm["monetary"], q=5, labels=[1, 2, 3, 4, 5])

        rfm["rfm_total"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]
        return rfm

    def _safe_qcut(self, series: pd.Series, q: int = 5, labels: list = None, ascending: bool = True) -> pd.Series:
        """安全的 qcut 分箱方法，处理重复值和边界情况。

        当数据唯一值过少或 qcut 抛出异常时，使用 rank 方法作为备选方案。

        Args:
            series: 待分箱的数据列。
            q: 分箱数量（默认 5）。
            labels: 各箱的标签列表。
            ascending: rank 备选方案是否升序排列。

        Returns:
            分箱后的评分 Series（int 类型）。
        """
        n_unique = series.nunique()
        if n_unique <= 1:
            return pd.Series([3] * len(series), index=series.index).astype(int)

        actual_q = min(q, n_unique)
        if labels:
            labels = labels[:actual_q]

        try:
            result = pd.qcut(series, q=actual_q, labels=labels, duplicates="drop").astype(int)
        except ValueError:
            ranks = series.rank(method="dense", ascending=ascending)
            max_rank = ranks.max()
            bins = np.linspace(0, max_rank, actual_q + 1)
            result = pd.cut(ranks, bins=bins, labels=labels, include_lowest=True).astype(int)

        return result

    def segment_members(self, rfm_df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """基于 RFM 指标对手会员进行规则分群。

        分群规则（基于 90 天阈值和指标均值比较）：
        - 高价值会员: recency<90 且 frequency>avg 且 monetary>avg
        - 新会员: recency<90 且 frequency<=avg
        - 流失风险会员: recency>=90 且 frequency>avg 且 monetary>avg
        - 沉睡会员: recency>=90 且 frequency<=avg
        - 潜力会员: 其他情况

        Args:
            rfm_df: 含 RFM 指标的 DataFrame。
            n_clusters: 保留参数（实际使用规则分群，不使用聚类）。

        Returns:
            新增 segment_name 和 segment 列的 DataFrame。

        Raises:
            ValueError: 当 rfm_df 为空或缺少必要列时。
        """
        if rfm_df is None or len(rfm_df) == 0:
            raise ValueError("rfm_df 不能为空")
        required_cols = {"recency", "frequency", "monetary"}
        missing = required_cols - set(rfm_df.columns)
        if missing:
            raise ValueError(f"rfm_df 缺少必要列: {missing}")

        logger.info("执行会员分群，共 %d 个会员", len(rfm_df))
        avg_frequency = rfm_df["frequency"].mean()
        avg_monetary = rfm_df["monetary"].mean()

        def assign_segment(row):
            recency = row["recency"]
            frequency = row["frequency"]
            monetary = row["monetary"]

            if recency < 90 and frequency > avg_frequency and monetary > avg_monetary:
                return "高价值会员"
            elif recency < 90 and frequency <= avg_frequency:
                return "新会员"
            elif recency >= 90 and frequency > avg_frequency and monetary > avg_monetary:
                return "流失风险会员"
            elif recency >= 90 and frequency <= avg_frequency:
                return "沉睡会员"
            else:
                return "潜力会员"

        rfm_df["segment_name"] = rfm_df.apply(assign_segment, axis=1)
        rfm_df["segment"] = rfm_df["segment_name"].map({
            "高价值会员": 0,
            "潜力会员": 1,
            "新会员": 2,
            "流失风险会员": 3,
            "沉睡会员": 4
        })

        return rfm_df

    def get_segment_stats(self, rfm_df: pd.DataFrame) -> Dict:
        """计算各分群的统计数据。

        Args:
            rfm_df: 含 segment 列的 RFM DataFrame。

        Returns:
            各分群的数量、占比、均值统计，以及整体汇总数据。
        """
        stats = {}

        for segment_id, segment_name in self.segment_labels.items():
            segment_data = rfm_df[rfm_df["segment"] == segment_id]
            if not segment_data.empty:
                stats[segment_name] = {
                    "count": len(segment_data),
                    "percentage": round(len(segment_data) / len(rfm_df) * 100, 2),
                    "avg_recency": round(segment_data["recency"].mean(), 1),
                    "avg_frequency": round(segment_data["frequency"].mean(), 1),
                    "avg_monetary": round(segment_data["monetary"].mean(), 1),
                    "customers": segment_data.index.tolist()[:10]
                }

        stats["total"] = {
            "members": len(rfm_df),
            "avg_recency": round(rfm_df["recency"].mean(), 1),
            "avg_frequency": round(rfm_df["frequency"].mean(), 1),
            "avg_monetary": round(rfm_df["monetary"].mean(), 1),
            "avg_rfm_score": round(rfm_df["rfm_total"].mean(), 1)
        }

        return stats

    def get_churn_risk(self, rfm_df: pd.DataFrame, risk_days: int = 90) -> pd.DataFrame:
        """识别流失风险会员。

        Args:
            rfm_df: 含 recency 列的 RFM DataFrame。
            risk_days: 流失风险阈值天数（默认 90 天）。

        Returns:
            recency >= risk_days 的会员数据，按 recency 降序排列。
        """
        churn_risk = rfm_df[rfm_df["recency"] >= risk_days].copy()
        churn_risk = churn_risk.sort_values("recency", ascending=False)
        return churn_risk

    def get_segment_distribution(self, rfm_df: pd.DataFrame) -> Tuple[List[str], List[int]]:
        """获取各分群的分布数据。

        Args:
            rfm_df: 含 segment_name 列的 RFM DataFrame。

        Returns:
            (分群名称列表, 各分群会员数量列表)。
        """
        distribution = rfm_df["segment_name"].value_counts().sort_index()
        labels = distribution.index.tolist()
        values = distribution.values.tolist()
        return labels, values

    def analyze(self, transactions_df: pd.DataFrame) -> Dict:
        """执行完整的会员分析流程（RFM 计算 + 分群 + 统计）。

        Args:
            transactions_df: 会员交易数据。

        Returns:
            包含 rfm_data/segment_stats/churn_risk/distribution 的结果字典。
        """
        logger.info("启动完整会员分析流程")
        rfm_df = self.calculate_rfm(transactions_df)
        rfm_df = self.segment_members(rfm_df)
        self.rfm_df = rfm_df

        return {
            "rfm_data": rfm_df,
            "segment_stats": self.get_segment_stats(rfm_df),
            "churn_risk": self.get_churn_risk(rfm_df),
            "distribution": self.get_segment_distribution(rfm_df)
        }