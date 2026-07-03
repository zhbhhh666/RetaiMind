"""DailyReportGenerator 模块单元测试。

覆盖初始化验证和 generate 输入验证。
"""
import pytest
from core.report_generator import DailyReportGenerator


class TestDailyReportGeneratorInit:
    """DailyReportGenerator 初始化测试。"""

    def test_init_with_valid_api_key(self):
        """有效 api_key 应成功初始化并创建 llm 和 prompt。"""
        gen = DailyReportGenerator(api_key="sk-test-key")
        assert gen.llm is not None
        assert gen.prompt is not None

    def test_init_with_empty_api_key_raises(self):
        """空 api_key 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="api_key"):
            DailyReportGenerator(api_key="")

    def test_init_with_none_api_key_raises(self):
        """None api_key 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="api_key"):
            DailyReportGenerator(api_key=None)


class TestGenerateValidation:
    """generate 方法输入验证测试。"""

    def test_generate_with_empty_data_raises(self):
        """空 data 应抛出 ValueError。"""
        gen = DailyReportGenerator(api_key="sk-test")
        with pytest.raises(ValueError, match="data"):
            gen.generate({})

    def test_generate_with_none_data_raises(self):
        """None data 应抛出 ValueError。"""
        gen = DailyReportGenerator(api_key="sk-test")
        with pytest.raises(ValueError, match="data"):
            gen.generate(None)

    def test_generate_with_missing_fields_raises(self):
        """缺少必要字段的 data 应抛出 ValueError。"""
        gen = DailyReportGenerator(api_key="sk-test")
        incomplete_data = {"report_date": "2026-07-01", "gmv": 1000}  # 缺少大部分字段
        with pytest.raises(ValueError, match="缺少必要字段"):
            gen.generate(incomplete_data)
