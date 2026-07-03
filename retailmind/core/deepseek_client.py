from openai import OpenAI
from typing import Iterator, Any
from core.logger import get_logger

logger = get_logger(__name__)


class DeepSeekClient:
    """DeepSeek API 客户端封装。

    基于 OpenAI SDK 兼容 DeepSeek API，支持普通对话、流式输出和 Tool Calling。
    """

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        """初始化 DeepSeek 客户端。

        Args:
            api_key: DeepSeek API Key。
            base_url: API 基础地址（默认官方地址）。

        Raises:
            ValueError: 当 api_key 为空时。
        """
        if not api_key or not isinstance(api_key, str):
            raise ValueError("api_key 不能为空")
        logger.info("初始化 DeepSeek 客户端，base_url=%s", base_url)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def chat(
        self,
        messages: list,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: list = None,
        tool_choice: str = None
    ) -> Any:
        """调用 DeepSeek 对话接口。

        Args:
            messages: 消息列表，格式为 [{"role": "system/user/assistant", "content": "..."}]。
            model: 模型名称（deepseek-chat 或 deepseek-reasoner）。
            temperature: 生成温度，控制随机性（0-2）。
            max_tokens: 最大生成 token 数。
            stream: 是否启用流式输出。
            tools: 工具定义列表，用于 Tool Calling。
            tool_choice: 工具选择策略（"auto" 或 "none" 或指定工具名）。

        Returns:
            API 响应对象。
        """
        if not messages:
            raise ValueError("messages 不能为空")

        logger.debug("调用 DeepSeek API: model=%s, msg_count=%d, stream=%s", model, len(messages), stream)
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        return self.client.chat.completions.create(**kwargs)

    def chat_stream(self, messages: list, **kwargs) -> Iterator[str]:
        """流式调用 DeepSeek 对话接口，逐字返回内容。

        Args:
            messages: 消息列表。
            **kwargs: 传递给 chat() 的其他参数。

        Yields:
            str: 每个流式响应块的内容文本。
        """
        response = self.chat(messages, stream=True, **kwargs)
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content