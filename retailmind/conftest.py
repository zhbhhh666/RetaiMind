"""pytest 全局配置。

提供共享 fixture 和 sys.path 配置，使测试用例可在项目根目录运行。
"""
import sys
import os
from pathlib import Path

# 将 retailmind 目录加入 sys.path，使测试可 import core 模块
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_inventory_df() -> pd.DataFrame:
    """提供示例库存数据 fixture。

    包含 5 个 SKU，涵盖库存正常、偏低、充足三种状态。
    """
    return pd.DataFrame({
        "sku_id": ["SKU000001", "SKU000002", "SKU000003", "SKU000004", "SKU000005"],
        "stock_qty": [100, 30, 200, 10, 50],  # 30<50(偏低), 10<40(偏低)
        "safety_stock": [50, 50, 80, 40, 30],
        "weekly_sales": [20, 25, 30, 15, 10],
        "category": ["食品", "服装", "数码", "食品", "美妆"]
    })


@pytest.fixture(scope="session")
def sample_forecast_df() -> pd.DataFrame:
    """提供示例销售预测数据 fixture。"""
    return pd.DataFrame({
        "sku_id": ["SKU000001", "SKU000002", "SKU000003", "SKU000004", "SKU000005"],
        "forecast_7d": [25, 30, 35, 20, 15]
    })


@pytest.fixture(scope="session")
def sample_transactions_df() -> pd.DataFrame:
    """提供示例会员交易数据 fixture。

    使用预聚合格式（含 frequency/monetary 列），便于测试可重复性。
    """
    from datetime import datetime, timedelta
    today = datetime.now()
    return pd.DataFrame({
        "customer_id": [f"CUST{str(i).zfill(4)}" for i in range(1, 11)],
        "last_purchase_date": [today - timedelta(days=d) for d in
                                [10, 20, 100, 120, 5, 200, 30, 150, 60, 90]],
        "frequency": [10, 5, 8, 3, 15, 2, 7, 4, 12, 6],
        "monetary": [1500, 800, 1200, 300, 2000, 100, 1000, 500, 1800, 900]
    })


@pytest.fixture(scope="session")
def sample_rfm_df(sample_transactions_df):
    """提供计算好的 RFM DataFrame fixture。"""
    from core.member_analyzer import MemberAnalyzer
    analyzer = MemberAnalyzer()
    rfm = analyzer.calculate_rfm(sample_transactions_df)
    return analyzer.segment_members(rfm)


@pytest.fixture
def temp_data_dir(tmp_path):
    """提供临时数据目录 fixture，测试后自动清理。"""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    return {"raw": str(raw_dir), "processed": str(processed_dir), "root": str(tmp_path)}
