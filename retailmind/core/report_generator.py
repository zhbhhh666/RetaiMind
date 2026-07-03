from langchain_core.prompts import PromptTemplate
from typing import Dict
from langchain_openai import ChatOpenAI
from core.logger import get_logger

logger = get_logger(__name__)


class DailyReportGenerator:
    """运营日报 AI 生成器。

    使用 DeepSeek API（通过 LangChain ChatOpenAI 兼容），
    基于运营数据和 PromptTemplate 自动生成 Markdown 格式的运营日报。
    """

    def __init__(self, api_key: str):
        """初始化日报生成器。

        Args:
            api_key: DeepSeek API Key。

        Raises:
            ValueError: 当 api_key 为空时。
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key 不能为空")
        logger.info("初始化日报生成器")
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            temperature=0.5
        )
        
        self.prompt = PromptTemplate.from_template("""
你是一名资深零售运营分析师。请基于以下昨日运营数据生成运营日报。

日期：{report_date}

核心指标：
- 销售额：{gmv} 元（环比 {gmv_mom}%）
- 订单量：{orders} 单（环比 {orders_mom}%）
- 客单价：{aov} 元（环比 {aov_mom}%）
- 新增会员：{new_members} 人（环比 {new_members_mom}%）
- 会员复购率：{repurchase_rate}%（环比 {repurchase_mom}%）

品类销售 TOP3：
{top_categories}

异常事件：
{abnormal_events}

请生成 Markdown 格式的运营日报，包含以下章节：
1. 昨日概览（一句话总结经营状况）
2. 核心指标解读（分析关键指标变化原因）
3. 品类洞察（TOP3 品类表现分析）
4. 风险提示（基于异常事件的风险预警）
5. 今日策略建议（3-5 条可执行建议）

输出格式要求：
- 使用 Markdown 标题和列表
- 关键数据用粗体标注
- 策略建议需具体、可执行
""")
    
    def generate(self, data: Dict) -> str:
        """基于运营数据生成 Markdown 格式的运营日报。

        Args:
            data: 运营数据字典，需包含 report_date/gmv/orders/aov/new_members/
                  repurchase_rate/gmv_mom/orders_mom/aov_mom/new_members_mom/
                  repurchase_mom/top_categories/abnormal_events 字段。

        Returns:
            Markdown 格式的运营日报文本，包含概览、指标解读、品类洞察、风险提示和策略建议。

        Raises:
            ValueError: 当 data 为空或缺少必要字段时。
        """
        if not data:
            raise ValueError("data 不能为空")
        required_keys = {"report_date", "gmv", "orders", "aov", "new_members",
                         "repurchase_rate", "gmv_mom", "orders_mom", "aov_mom",
                         "new_members_mom", "repurchase_mom", "top_categories",
                         "abnormal_events"}
        missing = required_keys - set(data.keys())
        if missing:
            raise ValueError(f"data 缺少必要字段: {missing}")

        logger.info("生成 %s 的运营日报", data.get("report_date", "未知日期"))
        prompt_text = self.prompt.format(**data)
        response = self.llm.invoke(prompt_text)
        return response.content