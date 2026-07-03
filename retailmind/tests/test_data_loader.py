"""DataLoader 模块单元测试。

覆盖数据加载、自动生成、日期确定性、输入验证等场景。
"""
import pytest
import pandas as pd
from core.data_loader import DataLoader


class TestDataLoaderInit:
    """DataLoader 初始化测试。"""

    def test_init_with_default_dirs(self, tmp_path):
        """默认目录参数应正确创建目录。"""
        raw = str(tmp_path / "raw")
        processed = str(tmp_path / "processed")
        dl = DataLoader(raw_dir=raw, processed_dir=processed)
        assert dl.raw_dir == raw
        assert dl.processed_dir == processed
        import os
        assert os.path.exists(raw)
        assert os.path.exists(processed)

    def test_init_with_empty_raw_dir_raises(self):
        """raw_dir 为空字符串应抛出 ValueError。"""
        with pytest.raises(ValueError, match="raw_dir"):
            DataLoader(raw_dir="", processed_dir="./p")

    def test_init_with_empty_processed_dir_raises(self):
        """processed_dir 为空字符串应抛出 ValueError。"""
        with pytest.raises(ValueError, match="processed_dir"):
            DataLoader(raw_dir="./r", processed_dir="")


class TestLoadMemberTransactions:
    """会员交易数据加载测试。"""

    def test_load_from_existing_csv(self, temp_data_dir):
        """已存在的 CSV 文件应被正确加载。"""
        # 先生成文件
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        dl.generate_sample_member_data()
        # 重新加载
        dl2 = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = dl2.load_member_transactions()
        assert len(df) == 1000
        assert "customer_id" in df.columns
        assert "frequency" in df.columns
        assert "monetary" in df.columns

    def test_load_auto_generate_when_missing(self, temp_data_dir):
        """文件不存在时应自动生成示例数据。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = dl.load_member_transactions()
        assert len(df) == 1000
        # 验证文件已被保存
        import os
        assert os.path.exists(os.path.join(temp_data_dir["raw"], "member_transactions.csv"))

    def test_empty_filename_raises(self, temp_data_dir):
        """filename 为空应抛出 ValueError。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        with pytest.raises(ValueError, match="filename"):
            dl.load_member_transactions(filename="")


class TestLoadInventoryData:
    """库存数据加载测试。"""

    def test_load_auto_generate(self, temp_data_dir):
        """文件不存在时自动生成 100 个 SKU。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = dl.load_inventory_data()
        assert len(df) == 100
        assert {"sku_id", "stock_qty", "safety_stock", "weekly_sales", "category"} <= set(df.columns)


class TestLoadSalesForecast:
    """销售预测数据加载测试。"""

    def test_load_auto_generate(self, temp_data_dir):
        """文件不存在时自动生成 100 条预测。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = dl.load_sales_forecast()
        assert len(df) == 100
        assert "forecast_7d" in df.columns


class TestLoadDailyOperations:
    """运营日报数据加载测试。"""

    def test_load_with_valid_date(self, temp_data_dir):
        """指定有效日期应返回运营数据。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        data = dl.load_daily_operations("2026-07-01")
        assert data["report_date"] == "2026-07-01"
        assert "gmv" in data
        assert "orders" in data
        assert "top_categories" in data
        assert "abnormal_events" in data

    def test_load_with_default_date(self, temp_data_dir):
        """不传日期时应使用昨天。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        data = dl.load_daily_operations()
        assert "report_date" in data
        # 应为昨天的日期
        from datetime import datetime, timedelta
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        assert data["report_date"] == yesterday

    def test_invalid_date_format_raises(self, temp_data_dir):
        """日期格式错误应抛出 ValueError。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            dl.load_daily_operations("2026/07/01")

    def test_invalid_date_str_raises(self, temp_data_dir):
        """完全无效的日期字符串应抛出 ValueError。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            dl.load_daily_operations("not-a-date")


class TestOperationsDataDeterminism:
    """运营数据生成确定性测试。"""

    def test_same_date_same_data(self, temp_data_dir):
        """相同日期应生成相同数据。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        d1 = dl.generate_sample_operations_data("2026-07-01")
        d2 = dl.generate_sample_operations_data("2026-07-01")
        assert d1["gmv"] == d2["gmv"]
        assert d1["orders"] == d2["orders"]

    def test_different_date_different_data(self, temp_data_dir):
        """不同日期应生成不同数据。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        d1 = dl.generate_sample_operations_data("2026-07-01")
        d2 = dl.generate_sample_operations_data("2026-07-02")
        assert d1["gmv"] != d2["gmv"]

    def test_data_types_and_ranges(self, temp_data_dir):
        """生成数据类型和值域应正确。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        data = dl.generate_sample_operations_data("2026-07-01")
        assert isinstance(data["gmv"], int)
        assert isinstance(data["orders"], int)
        assert isinstance(data["aov"], float)
        assert data["gmv"] > 0
        assert data["orders"] > 0
        assert data["new_members"] >= 0
        assert data["repurchase_rate"] >= 0

    def test_empty_date_raises(self, temp_data_dir):
        """date_str 为空应抛出 ValueError。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        with pytest.raises(ValueError, match="date_str"):
            dl.generate_sample_operations_data("")


class TestValidateData:
    """数据校验功能测试。"""

    def test_validate_member_valid(self, temp_data_dir):
        """合法会员数据应通过校验。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "customer_id": ["C001", "C002"],
            "last_purchase_date": ["2026-01-01", "2026-02-01"],
            "frequency": [5, 10],
            "monetary": [100.0, 200.0],
        })
        valid, errors = dl.validate_data("member_transactions", df)
        assert valid
        assert errors == []

    def test_validate_member_missing_column(self, temp_data_dir):
        """缺少必填字段应返回错误。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "customer_id": ["C001"],
            "frequency": [5],
            "monetary": [100.0],
        })
        valid, errors = dl.validate_data("member_transactions", df)
        assert not valid
        assert any("last_purchase_date" in e for e in errors)

    def test_validate_member_empty_df(self, temp_data_dir):
        """空 DataFrame 应返回错误。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame(columns=["customer_id", "last_purchase_date", "frequency", "monetary"])
        valid, errors = dl.validate_data("member_transactions", df)
        assert not valid
        assert any("为空" in e for e in errors)

    def test_validate_member_duplicate_id(self, temp_data_dir):
        """重复 customer_id 应返回错误。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "customer_id": ["C001", "C001"],
            "last_purchase_date": ["2026-01-01", "2026-02-01"],
            "frequency": [5, 10],
            "monetary": [100.0, 200.0],
        })
        valid, errors = dl.validate_data("member_transactions", df)
        assert not valid
        assert any("customer_id" in e for e in errors)

    def test_validate_member_negative_frequency(self, temp_data_dir):
        """非正值 frequency 应返回错误。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "customer_id": ["C001"],
            "last_purchase_date": ["2026-01-01"],
            "frequency": [0],
            "monetary": [100.0],
        })
        valid, errors = dl.validate_data("member_transactions", df)
        assert not valid
        assert any("frequency" in e for e in errors)

    def test_validate_inventory_valid(self, temp_data_dir):
        """合法库存数据应通过校验。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "sku_id": ["S001", "S002"],
            "stock_qty": [10, 20],
            "safety_stock": [5, 8],
            "weekly_sales": [3, 7],
            "category": ["食品", "服装"],
        })
        valid, errors = dl.validate_data("inventory", df)
        assert valid
        assert errors == []

    def test_validate_inventory_negative_stock(self, temp_data_dir):
        """负值库存应返回错误。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "sku_id": ["S001"],
            "stock_qty": [-5],
            "safety_stock": [5],
            "weekly_sales": [3],
            "category": ["食品"],
        })
        valid, errors = dl.validate_data("inventory", df)
        assert not valid
        assert any("stock_qty" in e for e in errors)

    def test_validate_inventory_nonnumeric(self, temp_data_dir):
        """非数值字段应返回错误。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "sku_id": ["S001"],
            "stock_qty": ["abc"],
            "safety_stock": [5],
            "weekly_sales": [3],
            "category": ["食品"],
        })
        valid, errors = dl.validate_data("inventory", df)
        assert not valid
        assert any("stock_qty" in e for e in errors)

    def test_validate_invalid_type_raises(self, temp_data_dir):
        """不支持的数据类型应抛出 ValueError。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="不支持"):
            dl.validate_data("invalid_type", df)

    def test_validate_forecast_valid(self, temp_data_dir):
        """合法销售预测数据应通过校验。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "sku_id": ["S001", "S002"],
            "forecast_7d": [15, 25],
        })
        valid, errors = dl.validate_data("sales_forecast", df)
        assert valid
        assert errors == []


class TestUploadData:
    """数据上传功能测试。"""

    def test_upload_member_data(self, temp_data_dir):
        """上传会员数据应保存到 CSV。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "customer_id": ["C001", "C002"],
            "last_purchase_date": ["2026-01-01", "2026-02-01"],
            "frequency": [5, 10],
            "monetary": [100.0, 200.0],
        })
        filepath = dl.upload_data("member_transactions", df)
        import os
        assert os.path.exists(filepath)
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 2
        assert loaded["customer_id"].tolist() == ["C001", "C002"]

    def test_upload_inventory_data(self, temp_data_dir):
        """上传库存数据应保存到 CSV。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "sku_id": ["S001"],
            "stock_qty": [50],
            "safety_stock": [20],
            "weekly_sales": [10],
            "category": ["食品"],
        })
        filepath = dl.upload_data("inventory", df)
        import os
        assert os.path.exists(filepath)
        loaded = pd.read_csv(filepath)
        assert len(loaded) == 1
        assert loaded["sku_id"][0] == "S001"

    def test_upload_invalid_data_raises(self, temp_data_dir):
        """校验失败的上传应抛出 ValueError。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "customer_id": ["C001"],
            "frequency": [-5],
            "monetary": [100.0],
        })
        with pytest.raises(ValueError, match="校验失败"):
            dl.upload_data("member_transactions", df)

    def test_upload_operations_data(self, temp_data_dir):
        """上传运营数据应保存到指定日期的文件。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame([{
            "gmv": 100000,
            "orders": 500,
            "aov": 200.0,
            "new_members": 50,
            "repurchase_rate": 35.0,
            "gmv_mom": 3.5,
            "orders_mom": 2.0,
            "aov_mom": 1.0,
            "new_members_mom": -2.0,
            "repurchase_mom": 0.5,
            "top_categories": "1. 食品",
            "abnormal_events": "今日无异常事件",
        }])
        filepath = dl.upload_data("operations", df, "2026-07-01")
        import os
        assert os.path.exists(filepath)
        assert "2026-07-01" in filepath


class TestResetData:
    """数据重置功能测试。"""

    def test_reset_member_data(self, temp_data_dir):
        """重置会员数据应生成新示例数据。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "customer_id": ["C001"],
            "last_purchase_date": ["2026-01-01"],
            "frequency": [5],
            "monetary": [100.0],
        })
        dl.upload_data("member_transactions", df)
        result = dl.reset_data("member_transactions")
        assert len(result) == 1000
        assert "customer_id" in result.columns

    def test_reset_inventory_data(self, temp_data_dir):
        """重置库存数据应生成新示例数据。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "sku_id": ["S001"],
            "stock_qty": [50],
            "safety_stock": [20],
            "weekly_sales": [10],
            "category": ["食品"],
        })
        dl.upload_data("inventory", df)
        result = dl.reset_data("inventory")
        assert len(result) == 100
        assert "sku_id" in result.columns

    def test_reset_forecast_data(self, temp_data_dir):
        """重置销售预测数据应生成新示例数据。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        df = pd.DataFrame({
            "sku_id": ["S001"],
            "forecast_7d": [15],
        })
        dl.upload_data("sales_forecast", df)
        result = dl.reset_data("sales_forecast")
        assert len(result) == 100
        assert "forecast_7d" in result.columns

    def test_reset_invalid_type_raises(self, temp_data_dir):
        """不支持的数据类型应抛出 ValueError。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        with pytest.raises(ValueError, match="不支持"):
            dl.reset_data("invalid_type")


class TestGetDataSummary:
    """数据概览功能测试。"""

    def test_summary_member_data(self, temp_data_dir):
        """会员数据概览应包含正确信息。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        dl.generate_sample_member_data()
        summary = dl.get_data_summary("member_transactions")
        assert summary["has_data"]
        assert summary["row_count"] == 1000
        assert "unique_customers" in summary["extra"]
        assert summary["extra"]["unique_customers"] == 1000

    def test_summary_inventory_data(self, temp_data_dir):
        """库存数据概览应包含正确信息。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        dl.generate_sample_inventory_data()
        summary = dl.get_data_summary("inventory")
        assert summary["has_data"]
        assert summary["row_count"] == 100
        assert "low_stock_count" in summary["extra"]
        assert "categories" in summary["extra"]

    def test_summary_forecast_data(self, temp_data_dir):
        """销售预测数据概览应包含正确信息。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        dl.generate_sample_forecast_data()
        summary = dl.get_data_summary("sales_forecast")
        assert summary["has_data"]
        assert summary["row_count"] == 100
        assert "avg_forecast_7d" in summary["extra"]

    def test_summary_operations_data(self, temp_data_dir):
        """运营数据概览应包含正确信息。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        summary = dl.get_data_summary("operations", "2026-07-01")
        assert summary["has_data"]
        assert summary["row_count"] == 1
        assert "gmv" in summary["extra"]
        assert summary["extra"]["date"] == "2026-07-01"

    def test_summary_invalid_type_raises(self, temp_data_dir):
        """不支持的数据类型应抛出 ValueError。"""
        dl = DataLoader(temp_data_dir["raw"], temp_data_dir["processed"])
        with pytest.raises(ValueError, match="不支持"):
            dl.get_data_summary("invalid_type")
