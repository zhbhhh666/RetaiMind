"""DeepSeekClient 模块单元测试。

覆盖初始化验证，不实际调用 API。
"""
import pytest
from core.deepseek_client import DeepSeekClient


class TestDeepSeekClientInit:
    """DeepSeekClient 初始化测试。"""

    def test_init_with_valid_api_key(self):
        """有效 api_key 应成功初始化。"""
        client = DeepSeekClient(api_key="sk-test-key-123")
        assert client.client is not None

    def test_init_with_empty_api_key_raises(self):
        """空 api_key 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="api_key"):
            DeepSeekClient(api_key="")

    def test_init_with_none_api_key_raises(self):
        """None api_key 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="api_key"):
            DeepSeekClient(api_key=None)

    def test_init_with_custom_base_url(self):
        """自定义 base_url 应生效。"""
        client = DeepSeekClient(api_key="sk-test", base_url="https://custom.example.com/v1")
        # 内部 OpenAI client 不暴露 base_url，但初始化不应报错
        assert client.client is not None


class TestDeepSeekClientChatValidation:
    """DeepSeekClient.chat 输入验证测试。"""

    def test_chat_with_empty_messages_raises(self):
        """空 messages 应抛出 ValueError。"""
        client = DeepSeekClient(api_key="sk-test")
        with pytest.raises(ValueError, match="messages"):
            client.chat(messages=[])

    def test_chat_with_none_messages_raises(self):
        """None messages 应抛出 ValueError。"""
        client = DeepSeekClient(api_key="sk-test")
        with pytest.raises(ValueError, match="messages"):
            client.chat(messages=None)
