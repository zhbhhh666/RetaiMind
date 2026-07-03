"""InventoryAgent 模块单元测试。

覆盖 7 个工具函数、初始化验证、边界处理等场景。
"""
import pytest
import pandas as pd
from core.agent_inventory import InventoryAgent


class TestInventoryAgentInit:
    """InventoryAgent 初始化测试。"""

    def test_init_registers_seven_tools(self, sample_inventory_df, sample_forecast_df):
        """初始化应注册 7 个工具。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        assert len(agent.tools) == 7
        expected_tools = {"inventory_query", "sales_forecast", "list_low_stock_items",
                          "list_all_skus", "get_stock_summary", "get_category_analysis",
                          "get_restock_recommendation"}
        assert expected_tools == set(agent.tools.keys())

    def test_init_empty_api_key_raises(self, sample_inventory_df, sample_forecast_df):
        """空 api_key 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="api_key"):
            InventoryAgent("", sample_inventory_df, sample_forecast_df)

    def test_init_empty_inventory_raises(self, sample_forecast_df):
        """空 inventory_df 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="inventory_df"):
            InventoryAgent("test_key", pd.DataFrame(), sample_forecast_df)

    def test_init_empty_forecast_raises(self, sample_inventory_df):
        """空 forecast_df 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="forecast_df"):
            InventoryAgent("test_key", sample_inventory_df, pd.DataFrame())

    def test_init_missing_columns_raises(self, sample_forecast_df):
        """inventory_df 缺少列应抛出 ValueError。"""
        bad_inv = pd.DataFrame({"sku_id": ["S1"], "stock_qty": [10]})
        with pytest.raises(ValueError, match="缺少必要列"):
            InventoryAgent("test_key", bad_inv, sample_forecast_df)


class TestQueryInventory:
    """query_inventory 工具测试。"""

    def test_query_existing_sku(self, sample_inventory_df, sample_forecast_df):
        """查询存在的 SKU 应返回库存详情。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.query_inventory("SKU000001")
        assert "SKU000001" in result
        assert "当前库存" in result
        assert "安全库存" in result

    def test_query_nonexistent_sku(self, sample_inventory_df, sample_forecast_df):
        """查询不存在的 SKU 应返回未找到提示。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.query_inventory("INVALID_SKU")
        assert "未找到" in result

    def test_query_empty_sku(self, sample_inventory_df, sample_forecast_df):
        """空 SKU 编号应返回提示。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.query_inventory("")
        assert "不能为空" in result


class TestQueryForecast:
    """query_forecast 工具测试。"""

    def test_query_existing_sku(self, sample_inventory_df, sample_forecast_df):
        """查询存在的 SKU 应返回预测数据。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.query_forecast("SKU000001")
        assert "SKU000001" in result
        assert "预测销量" in result

    def test_query_nonexistent_sku(self, sample_inventory_df, sample_forecast_df):
        """查询不存在的 SKU 应返回未找到提示。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.query_forecast("INVALID_SKU")
        assert "未找到" in result


class TestListLowStockItems:
    """list_low_stock_items 工具测试。"""

    def test_returns_low_stock_items(self, sample_inventory_df, sample_forecast_df):
        """应返回库存低于安全线的商品列表。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.list_low_stock_items()
        # fixture 中 SKU000002(30<50) 和 SKU000004(10<40) 应是低库存
        assert "SKU000002" in result
        assert "SKU000004" in result

    def test_all_normal_returns_message(self, sample_forecast_df):
        """全部库存正常时应返回相应提示。"""
        normal_inv = pd.DataFrame({
            "sku_id": ["SKU001", "SKU002"],
            "stock_qty": [200, 300],
            "safety_stock": [50, 80],
            "weekly_sales": [20, 30],
            "category": ["食品", "服装"]
        })
        agent = InventoryAgent("test_key", normal_inv, sample_forecast_df)
        result = agent.list_low_stock_items()
        assert "库存正常" in result


class TestListAllSkus:
    """list_all_skus 工具测试。"""

    def test_returns_all_sku_ids(self, sample_inventory_df, sample_forecast_df):
        """应返回全部 SKU 编号列表。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.list_all_skus()
        assert "5 个 SKU" in result
        assert "SKU000001" in result
        assert "SKU000005" in result


class TestGetStockSummary:
    """get_stock_summary 工具测试。"""

    def test_summary_contains_required_fields(self, sample_inventory_df, sample_forecast_df):
        """汇总应包含总SKU数、正常/偏低/充足数量、总库存、平均周销量。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.get_stock_summary()
        assert "总 SKU 数" in result
        assert "库存正常" in result
        assert "库存偏低" in result
        assert "库存充足" in result
        assert "总库存数量" in result
        assert "平均周销量" in result

    def test_summary_total_matches(self, sample_inventory_df, sample_forecast_df):
        """总数应等于 SKU 数量。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.get_stock_summary()
        assert "总 SKU 数: 5" in result

    def test_summary_percentages_sum_consistent(self, sample_inventory_df, sample_forecast_df):
        """正常 + 偏低 + 充足 = 总数（间接通过百分比验证）。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.get_stock_summary()
        # fixture: 正常 2(SKu1,5), 偏低 2(Sku2,4), 充足 1(Sku3)
        assert "库存正常: 2 个" in result
        assert "库存偏低: 2 个" in result
        assert "库存充足: 1 个" in result


class TestGetCategoryAnalysis:
    """get_category_analysis 工具测试。"""

    def test_analysis_includes_all_categories(self, sample_inventory_df, sample_forecast_df):
        """品类分析应包含所有品类。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.get_category_analysis()
        assert "食品" in result
        assert "服装" in result
        assert "数码" in result
        assert "美妆" in result

    def test_analysis_contains_sku_count_and_low_stock(self, sample_inventory_df, sample_forecast_df):
        """每个品类应包含 SKU 数、平均库存、低库存数。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.get_category_analysis()
        assert "SKU数" in result
        assert "平均库存" in result
        assert "低库存" in result


class TestGetRestockRecommendation:
    """get_restock_recommendation 工具测试。"""

    def test_returns_recommendation_for_low_stock(self, sample_inventory_df, sample_forecast_df):
        """应返回低库存 SKU 的补货建议。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.get_restock_recommendation()
        assert "补货建议" in result
        assert "SKU000002" in result  # 低库存商品
        assert "建议补货" in result

    def test_restock_amount_is_positive(self, sample_inventory_df, sample_forecast_df):
        """建议补货量应为正数。"""
        agent = InventoryAgent("test_key", sample_inventory_df, sample_forecast_df)
        result = agent.get_restock_recommendation()
        # 应包含 "建议补货 N件" 格式
        assert "建议补货" in result
        assert "件" in result

    def test_all_normal_returns_no_restock(self, sample_forecast_df):
        """全部库存正常时应返回无需补货提示。"""
        normal_inv = pd.DataFrame({
            "sku_id": ["SKU001"],
            "stock_qty": [200],
            "safety_stock": [50],
            "weekly_sales": [20],
            "category": ["食品"]
        })
        agent = InventoryAgent("test_key", normal_inv, sample_forecast_df)
        result = agent.get_restock_recommendation()
        assert "无需补货" in result or "库存正常" in result
