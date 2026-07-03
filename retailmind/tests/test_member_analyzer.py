"""MemberAnalyzer 模块单元测试。

覆盖 RFM 计算、会员分群、流失风险识别、_safe_qcut 边界处理等场景。
"""
import pytest
import pandas as pd
import numpy as np
from core.member_analyzer import MemberAnalyzer


class TestMemberAnalyzerInit:
    """MemberAnalyzer 初始化测试。"""

    def test_init_creates_segment_labels(self):
        """初始化应创建 5 个分群标签。"""
        analyzer = MemberAnalyzer()
        assert len(analyzer.segment_labels) == 5
        assert 0 in analyzer.segment_labels
        assert 4 in analyzer.segment_labels

    def test_rfm_df_initial_none(self):
        """rfm_df 初始值应为 None。"""
        analyzer = MemberAnalyzer()
        assert analyzer.rfm_df is None


class TestCalculateRFM:
    """RFM 计算测试。"""

    def test_calculate_rfm_returns_required_columns(self, sample_transactions_df):
        """calculate_rfm 应返回包含所有 RFM 列的 DataFrame。"""
        analyzer = MemberAnalyzer()
        rfm = analyzer.calculate_rfm(sample_transactions_df)
        for col in ["recency", "frequency", "monetary",
                    "recency_score", "frequency_score", "monetary_score", "rfm_total"]:
            assert col in rfm.columns, f"缺少列: {col}"

    def test_scores_in_range_1_to_5(self, sample_transactions_df):
        """评分值应在 1-5 范围内。"""
        analyzer = MemberAnalyzer()
        rfm = analyzer.calculate_rfm(sample_transactions_df)
        for col in ["recency_score", "frequency_score", "monetary_score"]:
            assert rfm[col].min() >= 1
            assert rfm[col].max() <= 5

    def test_rfm_total_is_sum_of_scores(self, sample_transactions_df):
        """rfm_total 应等于三个评分之和。"""
        analyzer = MemberAnalyzer()
        rfm = analyzer.calculate_rfm(sample_transactions_df)
        expected = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]
        pd.testing.assert_series_equal(rfm["rfm_total"], expected, check_names=False)

    def test_empty_input_raises(self):
        """空 DataFrame 应抛出 ValueError。"""
        analyzer = MemberAnalyzer()
        with pytest.raises(ValueError, match="transactions_df"):
            analyzer.calculate_rfm(pd.DataFrame())

    def test_missing_columns_raises(self):
        """缺少必要列应抛出 ValueError。"""
        analyzer = MemberAnalyzer()
        with pytest.raises(ValueError, match="缺少必要列"):
            analyzer.calculate_rfm(pd.DataFrame({"foo": [1, 2]}))

    def test_none_input_raises(self):
        """None 输入应抛出 ValueError。"""
        analyzer = MemberAnalyzer()
        with pytest.raises(ValueError):
            analyzer.calculate_rfm(None)


class TestSafeQcut:
    """_safe_qcut 边界处理测试。"""

    def test_single_unique_value_returns_default_3(self):
        """唯一值数为 1 时应全部返回默认值 3。"""
        analyzer = MemberAnalyzer()
        s = pd.Series([5, 5, 5, 5])
        result = analyzer._safe_qcut(s, q=5, labels=[1, 2, 3, 4, 5])
        assert (result == 3).all()

    def test_duplicate_values_does_not_raise(self):
        """重复值数据不应抛出异常（原本 qcut 会报错）。"""
        analyzer = MemberAnalyzer()
        s = pd.Series([1, 1, 1, 1, 2, 2, 2, 2])
        result = analyzer._safe_qcut(s, q=5, labels=[1, 2, 3, 4, 5])
        assert len(result) == len(s)

    def test_returns_int_type(self, sample_transactions_df):
        """返回值应为 int 类型。"""
        analyzer = MemberAnalyzer()
        rfm = analyzer.calculate_rfm(sample_transactions_df)
        assert np.issubdtype(rfm["recency_score"].dtype, np.integer)


class TestSegmentMembers:
    """会员分群测试。"""

    def test_segment_adds_required_columns(self, sample_rfm_df):
        """分群后应新增 segment_name 和 segment 列。"""
        assert "segment_name" in sample_rfm_df.columns
        assert "segment" in sample_rfm_df.columns

    def test_all_segments_valid(self, sample_rfm_df):
        """所有 segment 值应在 0-4 范围内。"""
        assert set(sample_rfm_df["segment"].unique()).issubset({0, 1, 2, 3, 4})

    def test_segment_names_match_labels(self, sample_rfm_df):
        """segment_name 应与 segment_labels 对应。"""
        analyzer = MemberAnalyzer()
        for seg_id, name in analyzer.segment_labels.items():
            subset = sample_rfm_df[sample_rfm_df["segment"] == seg_id]
            if not subset.empty:
                assert (subset["segment_name"] == name).all()

    def test_empty_input_raises(self):
        """空 DataFrame 应抛出 ValueError。"""
        analyzer = MemberAnalyzer()
        with pytest.raises(ValueError, match="rfm_df"):
            analyzer.segment_members(pd.DataFrame())

    def test_missing_columns_raises(self):
        """缺少必要列应抛出 ValueError。"""
        analyzer = MemberAnalyzer()
        with pytest.raises(ValueError, match="缺少必要列"):
            analyzer.segment_members(pd.DataFrame({"foo": [1, 2]}))


class TestChurnRisk:
    """流失风险识别测试。"""

    def test_churn_risk_filters_by_days(self, sample_rfm_df):
        """get_churn_risk 应正确过滤 recency >= risk_days 的会员。"""
        analyzer = MemberAnalyzer()
        churn = analyzer.get_churn_risk(sample_rfm_df, risk_days=90)
        assert (churn["recency"] >= 90).all()

    def test_churn_risk_sorted_descending(self, sample_rfm_df):
        """流失风险应按 recency 降序排列。"""
        analyzer = MemberAnalyzer()
        churn = analyzer.get_churn_risk(sample_rfm_df, risk_days=90)
        if not churn.empty:
            assert churn["recency"].is_monotonic_decreasing

    def test_custom_risk_days(self, sample_rfm_df):
        """自定义风险阈值应生效。"""
        analyzer = MemberAnalyzer()
        churn = analyzer.get_churn_risk(sample_rfm_df, risk_days=30)
        assert (churn["recency"] >= 30).all()


class TestSegmentStats:
    """分群统计测试。"""

    def test_stats_contains_total(self, sample_rfm_df):
        """统计数据应包含 total 汇总。"""
        analyzer = MemberAnalyzer()
        stats = analyzer.get_segment_stats(sample_rfm_df)
        assert "total" in stats

    def test_total_members_count_matches(self, sample_rfm_df):
        """total.members 应等于总会员数。"""
        analyzer = MemberAnalyzer()
        stats = analyzer.get_segment_stats(sample_rfm_df)
        assert stats["total"]["members"] == len(sample_rfm_df)

    def test_segment_stats_keys(self, sample_rfm_df):
        """分群统计应包含 count/percentage/avg_* 字段。"""
        analyzer = MemberAnalyzer()
        stats = analyzer.get_segment_stats(sample_rfm_df)
        for seg_name, seg_stats in stats.items():
            if seg_name == "total":
                continue
            assert "count" in seg_stats
            assert "percentage" in seg_stats
            assert "avg_recency" in seg_stats


class TestSegmentDistribution:
    """分群分布测试。"""

    def test_distribution_returns_labels_and_values(self, sample_rfm_df):
        """应返回 (labels, values) 二元组。"""
        analyzer = MemberAnalyzer()
        labels, values = analyzer.get_segment_distribution(sample_rfm_df)
        assert isinstance(labels, list)
        assert isinstance(values, list)
        assert len(labels) == len(values)

    def test_distribution_sum_equals_total(self, sample_rfm_df):
        """分布数量之和应等于总会员数。"""
        analyzer = MemberAnalyzer()
        labels, values = analyzer.get_segment_distribution(sample_rfm_df)
        assert sum(values) == len(sample_rfm_df)


class TestAnalyzeFullFlow:
    """完整分析流程测试。"""

    def test_analyze_returns_required_keys(self, sample_transactions_df):
        """analyze 应返回 4 个关键 key。"""
        analyzer = MemberAnalyzer()
        result = analyzer.analyze(sample_transactions_df)
        for key in ["rfm_data", "segment_stats", "churn_risk", "distribution"]:
            assert key in result

    def test_analyze_sets_rfm_df_attribute(self, sample_transactions_df):
        """analyze 后 self.rfm_df 应被设置。"""
        analyzer = MemberAnalyzer()
        analyzer.analyze(sample_transactions_df)
        assert analyzer.rfm_df is not None
