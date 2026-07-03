from typing import Dict
import pandas as pd
import json
from core.deepseek_client import DeepSeekClient
from core.logger import get_logger

logger = get_logger(__name__)


class InventoryAgent:
    """库存预警 AI Agent。

    使用 DeepSeek 原生 Tool Calling 能力，通过 7 个工具函数
    （库存查询、销售预测、低库存列表、全部SKU列表、库存汇总、品类分析、补货建议）
    实现库存监控与补货建议。

    工作流程：用户提问 -> DeepSeek 决策调用工具 -> 执行工具获取数据 -> 生成最终回答。
    """

    def __init__(self, api_key: str, inventory_df: pd.DataFrame,
                 forecast_df: pd.DataFrame):
        """初始化库存 Agent。

        Args:
            api_key: DeepSeek API Key。
            inventory_df: 库存数据，需含 sku_id/stock_qty/safety_stock/weekly_sales/category 列。
            forecast_df: 销售预测数据，需含 sku_id/forecast_7d 列。

        Raises:
            ValueError: 当 api_key 为空、inventory_df 或 forecast_df 为空 DataFrame 时。
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key 不能为空")
        if inventory_df is None or len(inventory_df) == 0:
            raise ValueError("inventory_df 不能为空")
        if forecast_df is None or len(forecast_df) == 0:
            raise ValueError("forecast_df 不能为空")

        # 验证必要列
        required_inv_cols = {"sku_id", "stock_qty", "safety_stock", "weekly_sales", "category"}
        missing_cols = required_inv_cols - set(inventory_df.columns)
        if missing_cols:
            raise ValueError(f"inventory_df 缺少必要列: {missing_cols}")
        required_fc_cols = {"sku_id", "forecast_7d"}
        missing_cols = required_fc_cols - set(forecast_df.columns)
        if missing_cols:
            raise ValueError(f"forecast_df 缺少必要列: {missing_cols}")

        self.client = DeepSeekClient(api_key=api_key)
        self.inventory_df = inventory_df
        self.forecast_df = forecast_df

        self.tools = {
            "inventory_query": {
                "description": "查询指定 SKU 的当前库存数量、安全库存线和最近7天销量",
                "function": self.query_inventory
            },
            "sales_forecast": {
                "description": "预测指定 SKU 未来7天的销量",
                "function": self.query_forecast
            },
            "list_low_stock_items": {
                "description": "查询所有库存低于安全库存线的商品列表，返回需要补货的商品信息",
                "function": self.list_low_stock_items
            },
            "list_all_skus": {
                "description": "获取所有 SKU 编号列表",
                "function": self.list_all_skus
            },
            "get_stock_summary": {
                "description": "获取库存整体汇总统计：总SKU数、库存正常/偏低/充足数量、平均库存周转天数",
                "function": self.get_stock_summary
            },
            "get_category_analysis": {
                "description": "按品类分析库存状况，返回各品类的SKU数、平均库存、低库存数和占比",
                "function": self.get_category_analysis
            },
            "get_restock_recommendation": {
                "description": "基于销售预测生成补货建议，计算每个低库存SKU的建议补货量",
                "function": self.get_restock_recommendation
            }
        }
        logger.info("InventoryAgent 初始化完成，注册 %d 个工具", len(self.tools))

    def query_inventory(self, sku_id: str) -> str:
        """查询指定 SKU 的库存详情（工具函数）。

        Args:
            sku_id: SKU 编号。

        Returns:
            库存详情文本，未找到时返回提示信息。
        """
        if not sku_id:
            return "SKU 编号不能为空"
        logger.debug("查询 SKU 库存: %s", sku_id)
        row = self.inventory_df[self.inventory_df['sku_id'] == sku_id]
        if row.empty:
            return f"未找到 SKU {sku_id} 的库存信息"
        return (f"SKU {sku_id}: 当前库存 {row['stock_qty'].values[0]}件，"
                f"安全库存 {row['safety_stock'].values[0]}件，"
                f"最近7天销量 {row['weekly_sales'].values[0]}件")

    def query_forecast(self, sku_id: str) -> str:
        """查询指定 SKU 的未来 7 天销量预测（工具函数）。

        Args:
            sku_id: SKU 编号。

        Returns:
            销量预测文本，未找到时返回提示信息。
        """
        if not sku_id:
            return "SKU 编号不能为空"
        logger.debug("查询 SKU 预测: %s", sku_id)
        row = self.forecast_df[self.forecast_df['sku_id'] == sku_id]
        if row.empty:
            return f"未找到 SKU {sku_id} 的预测数据"
        return f"SKU {sku_id} 未来7天预测销量: {row['forecast_7d'].values[0]}件"

    def list_low_stock_items(self) -> str:
        """查询所有库存低于安全线的商品列表（工具函数）。

        Returns:
            低库存商品列表文本，按短缺量降序排列。
        """
        logger.info("查询低库存商品列表")
        low_stock_df = self.inventory_df[self.inventory_df['stock_qty'] < self.inventory_df['safety_stock']].copy()
        low_stock_df['shortage'] = low_stock_df['safety_stock'] - low_stock_df['stock_qty']
        low_stock_df = low_stock_df.sort_values('shortage', ascending=False)

        if low_stock_df.empty:
            return "所有商品库存正常，没有需要补货的商品"

        result = f"发现 {len(low_stock_df)} 个商品库存低于安全库存线：\n"
        for _, row in low_stock_df.iterrows():
            result += f"- {row['sku_id']} ({row['category']}): 当前库存 {row['stock_qty']}件，安全库存 {row['safety_stock']}件，短缺 {row['shortage']}件，周销量 {row['weekly_sales']}件\n"
        return result

    def list_all_skus(self) -> str:
        """获取所有 SKU 编号列表（工具函数）。

        Returns:
            SKU 总数及编号列表文本。
        """
        logger.info("查询所有 SKU 列表")
        skus = self.inventory_df['sku_id'].tolist()
        return f"共有 {len(skus)} 个 SKU: {', '.join(skus)}"

    def get_stock_summary(self) -> str:
        """获取库存整体汇总统计（工具函数）。

        统计总 SKU 数、库存正常/偏低/充足数量、平均周销量。

        Returns:
            库存汇总文本。
        """
        logger.info("生成库存汇总统计")
        total = len(self.inventory_df)
        low_stock = (self.inventory_df['stock_qty'] < self.inventory_df['safety_stock']).sum()
        high_stock = (self.inventory_df['stock_qty'] > self.inventory_df['safety_stock'] * 2).sum()
        normal_stock = total - low_stock - high_stock
        avg_weekly_sales = round(self.inventory_df['weekly_sales'].mean(), 1)
        total_stock_value = int((self.inventory_df['stock_qty']).sum())

        return (
            f"库存汇总统计：\n"
            f"- 总 SKU 数: {total}\n"
            f"- 库存正常: {normal_stock} 个 ({normal_stock/total*100:.1f}%)\n"
            f"- 库存偏低: {low_stock} 个 ({low_stock/total*100:.1f}%)\n"
            f"- 库存充足: {high_stock} 个 ({high_stock/total*100:.1f}%)\n"
            f"- 总库存数量: {total_stock_value} 件\n"
            f"- 平均周销量: {avg_weekly_sales} 件"
        )

    def get_category_analysis(self) -> str:
        """按品类分析库存状况（工具函数）。

        Returns:
            各品类的 SKU 数、平均库存、低库存数、低库存占比文本。
        """
        logger.info("执行品类库存分析")
        categories = self.inventory_df['category'].unique()
        result = "各品类库存分析：\n"

        for cat in sorted(categories):
            cat_df = self.inventory_df[self.inventory_df['category'] == cat]
            sku_count = len(cat_df)
            avg_stock = round(cat_df['stock_qty'].mean(), 1)
            low_count = (cat_df['stock_qty'] < cat_df['safety_stock']).sum()
            low_pct = low_count / sku_count * 100 if sku_count > 0 else 0
            total_sales = int(cat_df['weekly_sales'].sum())
            result += (
                f"- {cat}: SKU数 {sku_count}，平均库存 {avg_stock}件，"
                f"低库存 {low_count} 个 ({low_pct:.1f}%)，周销量合计 {total_sales}件\n"
            )
        return result.strip()

    def get_restock_recommendation(self) -> str:
        """基于销售预测生成补货建议（工具函数）。

        对每个低库存 SKU，结合未来 7 天销量预测，计算建议补货量
        （目标：补货后库存满足未来 7 天预测销量 + 安全库存）。

        Returns:
            补货建议文本，包含 SKU、当前库存、安全库存、预测销量、建议补货量。
        """
        logger.info("生成补货建议")
        low_stock_df = self.inventory_df[
            self.inventory_df['stock_qty'] < self.inventory_df['safety_stock']
        ].copy()

        if low_stock_df.empty:
            return "所有商品库存正常，无需补货"

        # 关联预测数据
        merged = low_stock_df.merge(self.forecast_df[['sku_id', 'forecast_7d']], on='sku_id', how='left')
        # 缺失预测时用周销量作为估算
        merged['forecast_7d'] = merged['forecast_7d'].fillna(merged['weekly_sales'])

        # 建议补货量 = 安全库存 + 预测销量 - 当前库存
        merged['suggested_restock'] = (
            merged['safety_stock'] + merged['forecast_7d'] - merged['stock_qty']
        ).clip(lower=0).astype(int)

        merged = merged.sort_values('suggested_restock', ascending=False)

        result = f"补货建议（共 {len(merged)} 个商品）：\n"
        for _, row in merged.iterrows():
            result += (
                f"- {row['sku_id']} ({row['category']}): 当前 {row['stock_qty']}件，"
                f"安全库存 {row['safety_stock']}件，未来7天预测 {int(row['forecast_7d'])}件，"
                f"建议补货 {row['suggested_restock']}件\n"
            )
        return result.strip()

    def run(self, query: str) -> Dict:
        """执行用户查询，自动决策工具调用并生成回答。

        流程：构建工具定义 -> DeepSeek 决策 -> 执行工具 -> 二次调用生成最终回答。

        Args:
            query: 用户的自然语言问题。

        Returns:
            包含 "output"（最终回答文本）和 "intermediate_steps"（工具执行过程列表）的字典。

        Raises:
            ValueError: 当 query 为空字符串时。
        """
        if not query or not isinstance(query, str):
            raise ValueError("query 不能为空")

        logger.info("执行 Agent 查询: %s", query[:100])
        tool_list = []
        # 无参数的工具集合
        no_param_tools = {"list_low_stock_items", "list_all_skus",
                          "get_stock_summary", "get_category_analysis",
                          "get_restock_recommendation"}

        for name, info in self.tools.items():
            params = {
                "type": "object",
                "properties": {},
                "required": []
            }
            if name not in no_param_tools:
                params["properties"]["sku_id"] = {
                    "type": "string",
                    "description": "SKU 编号"
                }
                params["required"] = ["sku_id"]

            tool_list.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": params
                }
            })

        messages = [
            {
                "role": "system",
                "content": "你是零售库存管理专家。请使用提供的工具帮助用户解决库存问题。"
            },
            {
                "role": "user",
                "content": query
            }
        ]

        response = self.client.chat(
            messages=messages,
            model="deepseek-chat",
            temperature=0.1,
            max_tokens=4096,
            tools=tool_list,
            tool_choice="auto"
        )

        result = response.choices[0].message

        if result.tool_calls:
            observations = []
            for tool_call in result.tool_calls:
                func_name = tool_call.function.name
                args_str = tool_call.function.arguments
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    logger.warning("工具参数 JSON 解析失败: %s", args_str)
                    args = {}
                sku_id = args.get("sku_id", "")

                if func_name in self.tools:
                    if func_name in no_param_tools:
                        observation = self.tools[func_name]["function"]()
                    else:
                        observation = self.tools[func_name]["function"](sku_id)
                    observations.append(f"工具调用: {func_name}({sku_id}) -> {observation}")
                    logger.info("工具调用 %s(%s) 完成", func_name, sku_id)

            messages.append({"role": "assistant", "content": result.content})
            messages.append({
                "role": "user",
                "content": "以下是工具执行结果，请基于这些信息给出最终答案：\n" + "\n".join(observations)
            })

            final_response = self.client.chat(
                messages=messages,
                model="deepseek-chat",
                temperature=0.1,
                max_tokens=4096
            )

            return {
                "output": final_response.choices[0].message.content,
                "intermediate_steps": observations
            }
        else:
            return {
                "output": result.content,
                "intermediate_steps": []
            }
