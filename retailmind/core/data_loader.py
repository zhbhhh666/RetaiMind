import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from core.logger import get_logger

logger = get_logger(__name__)

DATA_TYPES = {
    "member_transactions": {
        "filename": "member_transactions.csv",
        "required_columns": ["customer_id", "last_purchase_date", "frequency", "monetary"],
        "optional_columns": ["preferred_category"],
        "unique_columns": ["customer_id"],
        "numeric_columns": ["frequency", "monetary"],
        "positive_columns": ["frequency", "monetary"],
        "display_name": "会员交易数据",
    },
    "inventory": {
        "filename": "inventory.csv",
        "required_columns": ["sku_id", "stock_qty", "safety_stock", "weekly_sales", "category"],
        "optional_columns": [],
        "unique_columns": ["sku_id"],
        "numeric_columns": ["stock_qty", "safety_stock", "weekly_sales"],
        "non_negative_columns": ["stock_qty", "safety_stock", "weekly_sales"],
        "display_name": "库存数据",
    },
    "sales_forecast": {
        "filename": "sales_forecast.csv",
        "required_columns": ["sku_id", "forecast_7d"],
        "optional_columns": [],
        "unique_columns": ["sku_id"],
        "numeric_columns": ["forecast_7d"],
        "non_negative_columns": ["forecast_7d"],
        "display_name": "销售预测数据",
    },
    "operations": {
        "filename": "operations_{date}.csv",
        "required_columns": ["gmv", "orders", "aov", "new_members", "repurchase_rate"],
        "optional_columns": ["gmv_mom", "orders_mom", "aov_mom", "new_members_mom", "repurchase_mom", "top_categories", "abnormal_events"],
        "unique_columns": [],
        "numeric_columns": ["gmv", "orders", "aov", "new_members", "repurchase_rate", "gmv_mom", "orders_mom", "aov_mom", "new_members_mom", "repurchase_mom"],
        "non_negative_columns": ["gmv", "orders", "aov", "new_members", "repurchase_rate"],
        "display_name": "运营数据",
    },
}


class DataLoader:
    """数据加载模块。

    负责加载 CSV 业务数据，当数据文件不存在时自动生成示例数据。
    支持会员交易、库存、销售预测和运营日报数据的加载与生成。
    """

    def __init__(self, raw_dir: str = "./data/raw", processed_dir: str = "./data/processed"):
        """初始化数据加载器。

        Args:
            raw_dir: 原始数据目录路径。
            processed_dir: 处理后数据目录路径。

        Raises:
            ValueError: 当 raw_dir 或 processed_dir 为空字符串时。
        """
        if not raw_dir or not isinstance(raw_dir, str):
            raise ValueError("raw_dir 不能为空")
        if not processed_dir or not isinstance(processed_dir, str):
            raise ValueError("processed_dir 不能为空")

        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        logger.info("DataLoader 初始化完成，raw_dir=%s", raw_dir)

    def load_member_transactions(self, filename: str = "member_transactions.csv") -> pd.DataFrame:
        """加载会员交易数据，文件不存在时自动生成示例数据。

        Args:
            filename: CSV 文件名。

        Returns:
            会员交易 DataFrame，含 customer_id/last_purchase_date/frequency/monetary 列。

        Raises:
            ValueError: 当 filename 为空字符串时。
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("filename 不能为空")

        filepath = os.path.join(self.raw_dir, filename)
        if os.path.exists(filepath):
            logger.info("从文件加载会员交易数据: %s", filepath)
            return pd.read_csv(filepath)
        logger.info("会员交易数据文件不存在，生成示例数据")
        return self.generate_sample_member_data()

    def load_inventory_data(self, filename: str = "inventory.csv") -> pd.DataFrame:
        """加载库存数据，文件不存在时自动生成示例数据。

        Args:
            filename: CSV 文件名。

        Returns:
            库存 DataFrame，含 sku_id/stock_qty/safety_stock/weekly_sales/category 列。

        Raises:
            ValueError: 当 filename 为空字符串时。
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("filename 不能为空")

        filepath = os.path.join(self.raw_dir, filename)
        if os.path.exists(filepath):
            logger.info("从文件加载库存数据: %s", filepath)
            return pd.read_csv(filepath)
        logger.info("库存数据文件不存在，生成示例数据")
        return self.generate_sample_inventory_data()

    def load_sales_forecast(self, filename: str = "sales_forecast.csv") -> pd.DataFrame:
        """加载销售预测数据，文件不存在时自动生成示例数据。

        Args:
            filename: CSV 文件名。

        Returns:
            销售预测 DataFrame，含 sku_id/forecast_7d 列。

        Raises:
            ValueError: 当 filename 为空字符串时。
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("filename 不能为空")

        filepath = os.path.join(self.raw_dir, filename)
        if os.path.exists(filepath):
            logger.info("从文件加载销售预测数据: %s", filepath)
            return pd.read_csv(filepath)
        logger.info("销售预测数据文件不存在，生成示例数据")
        return self.generate_sample_forecast_data()

    def load_daily_operations(self, date_str: Optional[str] = None) -> Dict:
        """加载指定日期的运营数据，如文件不存在则生成示例数据。

        Args:
            date_str: 日期字符串（YYYY-MM-DD 格式），默认为昨天。

        Returns:
            运营数据字典，包含 gmv/orders/aov/环比指标/品类TOP3/异常事件等字段。

        Raises:
            ValueError: 当 date_str 格式不是 YYYY-MM-DD 时。
        """
        if date_str is None:
            date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # 验证日期格式
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"date_str 格式错误，应为 YYYY-MM-DD，实际: {date_str}")

        filename = f"operations_{date_str}.csv"
        filepath = os.path.join(self.raw_dir, filename)

        if os.path.exists(filepath):
            logger.info("从文件加载运营数据: %s", filepath)
            df = pd.read_csv(filepath)
            return {
                "report_date": date_str,
                "gmv": float(df["gmv"].values[0]),
                "orders": int(df["orders"].values[0]),
                "aov": float(df["aov"].values[0]),
                "new_members": int(df["new_members"].values[0]),
                "repurchase_rate": float(df["repurchase_rate"].values[0]),
                "gmv_mom": float(df["gmv_mom"].values[0]),
                "orders_mom": float(df["orders_mom"].values[0]),
                "aov_mom": float(df["aov_mom"].values[0]),
                "new_members_mom": float(df["new_members_mom"].values[0]),
                "repurchase_mom": float(df["repurchase_mom"].values[0]),
                "top_categories": df["top_categories"].values[0],
                "abnormal_events": df["abnormal_events"].values[0]
            }
        logger.info("运营数据文件不存在，生成 %s 的示例数据", date_str)
        return self.generate_sample_operations_data(date_str)

    def generate_sample_member_data(self) -> pd.DataFrame:
        """生成示例会员交易数据并保存到 CSV。

        使用固定随机种子 42，生成 1000 个会员的消费记录。

        Returns:
            会员交易 DataFrame。
        """
        logger.info("生成示例会员交易数据（1000 条）")
        np.random.seed(42)
        n_customers = 1000
        customer_ids = [f"CUST{str(i).zfill(4)}" for i in range(1, n_customers + 1)]

        today = datetime.now()
        recency_days = np.random.randint(1, 365, n_customers)
        last_purchase_dates = [today - timedelta(days=int(d)) for d in recency_days]

        frequency = np.random.randint(1, 50, n_customers)
        monetary = np.random.normal(500, 200, n_customers).clip(50, 2000)

        categories = ["食品", "服装", "家居", "数码", "美妆"]
        preferred_category = np.random.choice(categories, n_customers)

        df = pd.DataFrame({
            "customer_id": customer_ids,
            "last_purchase_date": last_purchase_dates,
            "frequency": frequency,
            "monetary": monetary,
            "preferred_category": preferred_category
        })

        df.to_csv(os.path.join(self.raw_dir, "member_transactions.csv"), index=False)
        logger.info("示例会员交易数据已保存")
        return df

    def generate_sample_inventory_data(self) -> pd.DataFrame:
        """生成示例库存数据并保存到 CSV。

        使用固定随机种子 42，生成 100 个 SKU 的库存记录。

        Returns:
            库存 DataFrame。
        """
        logger.info("生成示例库存数据（100 个 SKU）")
        np.random.seed(42)
        sku_ids = [f"SKU{str(i).zfill(6)}" for i in range(1, 101)]
        stock_qty = np.random.randint(0, 200, 100)
        safety_stock = np.random.randint(20, 80, 100)
        weekly_sales = np.random.randint(10, 100, 100)

        categories = ["食品", "服装", "家居", "数码", "美妆"]
        sku_category = np.random.choice(categories, 100)

        df = pd.DataFrame({
            "sku_id": sku_ids,
            "stock_qty": stock_qty,
            "safety_stock": safety_stock,
            "weekly_sales": weekly_sales,
            "category": sku_category
        })

        df.to_csv(os.path.join(self.raw_dir, "inventory.csv"), index=False)
        logger.info("示例库存数据已保存")
        return df

    def generate_sample_forecast_data(self) -> pd.DataFrame:
        """生成示例销售预测数据并保存到 CSV。

        使用固定随机种子 42，生成 100 个 SKU 的未来 7 天销量预测。

        Returns:
            销售预测 DataFrame。
        """
        logger.info("生成示例销售预测数据（100 个 SKU）")
        np.random.seed(42)
        sku_ids = [f"SKU{str(i).zfill(6)}" for i in range(1, 101)]
        forecast_7d = np.random.randint(15, 120, 100)

        df = pd.DataFrame({
            "sku_id": sku_ids,
            "forecast_7d": forecast_7d
        })

        df.to_csv(os.path.join(self.raw_dir, "sales_forecast.csv"), index=False)
        logger.info("示例销售预测数据已保存")
        return df

    def generate_sample_operations_data(self, date_str: str) -> Dict:
        """基于日期生成确定性随机的示例运营数据。

        使用日期字符串作为随机种子，确保同一日期数据一致，不同日期数据不同。

        Args:
            date_str: 日期字符串（YYYY-MM-DD 格式）。

        Returns:
            包含 gmv/orders/aov/环比指标/品类TOP3/异常事件的运营数据字典。

        Raises:
            ValueError: 当 date_str 为空或格式错误时。
        """
        if not date_str or not isinstance(date_str, str):
            raise ValueError("date_str 不能为空")

        # 验证日期格式
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"date_str 格式错误，应为 YYYY-MM-DD，实际: {date_str}")

        seed_value = abs(hash(date_str)) % (10**8)
        np.random.seed(seed_value)

        base_gmv = 1256800
        base_orders = 8920
        base_aov = 140.9
        base_new_members = 234
        base_repurchase = 32.5

        gmv_variation = np.random.normal(0, 0.08)
        orders_variation = np.random.normal(0, 0.06)
        aov_variation = np.random.normal(0, 0.03)
        new_members_variation = np.random.normal(0, 0.15)
        repurchase_variation = np.random.normal(0, 0.05)

        gmv = int(base_gmv * (1 + gmv_variation))
        orders = int(base_orders * (1 + orders_variation))
        aov = round(base_aov * (1 + aov_variation), 1)
        new_members = int(base_new_members * (1 + new_members_variation))
        repurchase_rate = round(base_repurchase * (1 + repurchase_variation), 1)

        gmv_mom = round(np.random.normal(3, 4), 1)
        orders_mom = round(np.random.normal(2, 3), 1)
        aov_mom = round(np.random.normal(0.5, 2), 1)
        new_members_mom = round(np.random.normal(-1, 5), 1)
        repurchase_mom = round(np.random.normal(1, 2), 1)

        categories = ["食品", "服装", "数码", "家居", "美妆"]
        cat_percentages = np.random.dirichlet([1, 1, 1, 1, 1]) * 100
        cat_percentages = cat_percentages[:3]
        sorted_indices = np.argsort(cat_percentages)[::-1]
        cat_percentages = cat_percentages[sorted_indices]

        top_categories = ""
        for i in range(3):
            top_categories += f"{i+1}. {categories[sorted_indices[i]]}（{cat_percentages[i]:.1f}%）\n"

        abnormal_count = np.random.choice([0, 1, 2, 3], p=[0.1, 0.4, 0.3, 0.2])
        abnormal_events = ""
        if abnormal_count >= 1:
            abnormal_events += "1. 支付系统下午出现短暂延迟\n"
        if abnormal_count >= 2:
            sku_id = f"SKU{str(np.random.randint(1, 100)).zfill(6)}"
            stock = np.random.randint(5, 30)
            safety = np.random.randint(30, 60)
            abnormal_events += f"2. {sku_id} 库存预警（当前库存{stock}件，安全库存{safety}件）\n"
        if abnormal_count >= 3:
            abnormal_events += "3. 会员系统接口响应较慢\n"

        return {
            "report_date": date_str,
            "gmv": gmv,
            "orders": orders,
            "aov": aov,
            "new_members": max(0, new_members),
            "repurchase_rate": max(0, repurchase_rate),
            "gmv_mom": gmv_mom,
            "orders_mom": orders_mom,
            "aov_mom": aov_mom,
            "new_members_mom": new_members_mom,
            "repurchase_mom": repurchase_mom,
            "top_categories": top_categories.strip(),
            "abnormal_events": abnormal_events.strip() if abnormal_events else "今日无异常事件"
        }

    def validate_data(self, data_type: str, df: pd.DataFrame, date_str: Optional[str] = None) -> Tuple[bool, List[str]]:
        """校验上传数据的字段和数据质量。

        Args:
            data_type: 数据类型，必须是 DATA_TYPES 的键。
            df: 待校验的 DataFrame。
            date_str: 运营数据日期（仅 operations 类型需要）。

        Returns:
            (是否通过, 错误信息列表)

        Raises:
            ValueError: 当 data_type 不支持时。
        """
        if data_type not in DATA_TYPES:
            raise ValueError(f"不支持的数据类型: {data_type}，支持: {list(DATA_TYPES.keys())}")

        config = DATA_TYPES[data_type]
        errors = []

        if df.empty:
            errors.append("数据为空，至少需要 1 行数据")
            return False, errors

        actual_columns = set(df.columns.tolist())
        required_columns = set(config["required_columns"])

        missing = required_columns - actual_columns
        if missing:
            errors.append(f"缺少必填字段: {', '.join(sorted(missing))}")

        numeric_error_cols = set()
        for col in config.get("numeric_columns", []):
            if col in actual_columns:
                try:
                    pd.to_numeric(df[col], errors="raise")
                except (ValueError, TypeError):
                    errors.append(f"字段 {col} 包含非数值内容")
                    numeric_error_cols.add(col)

        for col in config.get("positive_columns", []):
            if col in actual_columns and col not in numeric_error_cols:
                try:
                    series = pd.to_numeric(df[col], errors="coerce")
                    if (series <= 0).any():
                        bad_count = (series <= 0).sum()
                        errors.append(f"字段 {col} 存在 {bad_count} 条非正值（必须 > 0）")
                except Exception:
                    pass

        for col in config.get("non_negative_columns", []):
            if col in actual_columns and col not in numeric_error_cols:
                try:
                    series = pd.to_numeric(df[col], errors="coerce")
                    if (series < 0).any():
                        bad_count = (series < 0).sum()
                        errors.append(f"字段 {col} 存在 {bad_count} 条负值（必须 >= 0）")
                except Exception:
                    pass

        for col in config.get("unique_columns", []):
            if col in actual_columns:
                if df[col].duplicated().any():
                    dup_count = df[col].duplicated().sum()
                    errors.append(f"字段 {col} 存在 {dup_count} 条重复值")

        if "last_purchase_date" in actual_columns:
            try:
                pd.to_datetime(df["last_purchase_date"], errors="raise")
            except (ValueError, TypeError):
                errors.append("字段 last_purchase_date 日期格式不正确")

        if not errors:
            return True, []
        return False, errors

    def upload_data(self, data_type: str, df: pd.DataFrame, date_str: Optional[str] = None) -> str:
        """保存上传的数据到 CSV 文件。

        Args:
            data_type: 数据类型。
            df: 待保存的 DataFrame。
            date_str: 运营数据日期（仅 operations 类型需要）。

        Returns:
            保存的文件路径。

        Raises:
            ValueError: 当 data_type 不支持或校验失败时。
        """
        if data_type not in DATA_TYPES:
            raise ValueError(f"不支持的数据类型: {data_type}")

        valid, errors = self.validate_data(data_type, df, date_str)
        if not valid:
            raise ValueError(f"数据校验失败: {'; '.join(errors)}")

        config = DATA_TYPES[data_type]
        if data_type == "operations":
            if not date_str:
                date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            filename = config["filename"].format(date=date_str)
        else:
            filename = config["filename"]

        filepath = os.path.join(self.raw_dir, filename)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info("已保存%s: %s（%d 行）", config["display_name"], filepath, len(df))
        return filepath

    def reset_data(self, data_type: str, date_str: Optional[str] = None) -> pd.DataFrame:
        """重置为示例数据。

        Args:
            data_type: 数据类型。
            date_str: 运营数据日期（仅 operations 类型需要）。

        Returns:
            生成的示例数据 DataFrame（运营数据返回空）。

        Raises:
            ValueError: 当 data_type 不支持时。
        """
        if data_type not in DATA_TYPES:
            raise ValueError(f"不支持的数据类型: {data_type}")

        config = DATA_TYPES[data_type]
        filepath = os.path.join(self.raw_dir, config["filename"])

        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info("已删除原数据文件: %s", filepath)

        if data_type == "member_transactions":
            return self.generate_sample_member_data()
        elif data_type == "inventory":
            return self.generate_sample_inventory_data()
        elif data_type == "sales_forecast":
            return self.generate_sample_forecast_data()
        elif data_type == "operations":
            if not date_str:
                date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            self.generate_sample_operations_data(date_str)
            return pd.DataFrame()

        return pd.DataFrame()

    def get_data_summary(self, data_type: str, date_str: Optional[str] = None) -> Dict[str, Any]:
        """获取数据概览信息。

        Args:
            data_type: 数据类型。
            date_str: 运营数据日期（仅 operations 类型需要）。

        Returns:
            包含 row_count/columns/has_data/source 等字段的概览字典。

        Raises:
            ValueError: 当 data_type 不支持时。
        """
        if data_type not in DATA_TYPES:
            raise ValueError(f"不支持的数据类型: {data_type}")

        config = DATA_TYPES[data_type]
        summary = {
            "data_type": data_type,
            "display_name": config["display_name"],
            "has_data": False,
            "row_count": 0,
            "columns": [],
            "source": "未上传",
            "extra": {},
        }

        try:
            if data_type == "member_transactions":
                df = self.load_member_transactions()
                summary["has_data"] = True
                summary["row_count"] = len(df)
                summary["columns"] = df.columns.tolist()
                summary["source"] = "上传数据" if os.path.exists(os.path.join(self.raw_dir, config["filename"])) else "示例数据"
                summary["extra"] = {
                    "unique_customers": int(df["customer_id"].nunique()),
                    "avg_frequency": float(df["frequency"].mean()),
                    "avg_monetary": float(df["monetary"].mean()),
                }
            elif data_type == "inventory":
                df = self.load_inventory_data()
                summary["has_data"] = True
                summary["row_count"] = len(df)
                summary["columns"] = df.columns.tolist()
                summary["source"] = "上传数据" if os.path.exists(os.path.join(self.raw_dir, config["filename"])) else "示例数据"
                low_stock = (df["stock_qty"] < df["safety_stock"]).sum()
                summary["extra"] = {
                    "unique_skus": int(df["sku_id"].nunique()),
                    "categories": int(df["category"].nunique()),
                    "low_stock_count": int(low_stock),
                }
            elif data_type == "sales_forecast":
                df = self.load_sales_forecast()
                summary["has_data"] = True
                summary["row_count"] = len(df)
                summary["columns"] = df.columns.tolist()
                summary["source"] = "上传数据" if os.path.exists(os.path.join(self.raw_dir, config["filename"])) else "示例数据"
                summary["extra"] = {
                    "unique_skus": int(df["sku_id"].nunique()),
                    "avg_forecast_7d": float(df["forecast_7d"].mean()),
                }
            elif data_type == "operations":
                if not date_str:
                    date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                data = self.load_daily_operations(date_str)
                summary["has_data"] = True
                summary["row_count"] = 1
                summary["columns"] = list(data.keys())
                summary["source"] = "上传数据" if os.path.exists(os.path.join(self.raw_dir, f"operations_{date_str}.csv")) else "示例数据"
                summary["extra"] = {
                    "date": date_str,
                    "gmv": data["gmv"],
                    "orders": data["orders"],
                    "aov": data["aov"],
                }
        except Exception as e:
            logger.warning("获取%s概览失败: %s", config["display_name"], e)
            summary["source"] = "加载失败"

        return summary
